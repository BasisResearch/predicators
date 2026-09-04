"""Tests for scripts/continual_transcripts.py and the viewer's replay and
session pages, using transcripts written by the real log formatter."""
import asyncio
import json
import os
from typing import Any, Dict, List

from predicators import utils
from predicators.agent_sdk.log_formatter import format_conversation_markdown
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.continual_tools import PlayState, \
    build_continual_tools
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun, ProtocolSession
from scripts import continual_transcripts as tr
from scripts import continual_viewer as viewer


def _entries(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build parsed-message entries the formatter accepts."""
    collected: List[Dict[str, Any]] = []
    for c in calls:
        blocks: List[Dict[str, Any]] = []
        if c.get("thinking"):
            blocks.append({"type": "ThinkingBlock", "thinking": c["thinking"]})
        if c.get("text"):
            blocks.append({"type": "text", "text": c["text"]})
        blocks.append({
            "type": "tool_use",
            "name": "mcp__predicator_tools__" + c["tool"],
            "id": c["id"],
            "input": c["args"],
        })
        collected.append({"type": "assistant", "content": blocks})
        collected.append({
            "type":
            "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": c["id"],
                "content": [{
                    "type": "text",
                    "text": c["result"]
                }],
                "is_error": c.get("is_error", False),
            }]
        })
    collected.append({
        "type": "result",
        "num_turns": len(calls),
        "total_cost_usd": 0.42,
    })
    return collected


def test_parse_transcript_round_trip() -> None:
    """The parser recovers thinking, text, calls with multi-line args, and
    results from the formatter's markdown."""
    calls = [{
        "tool":
        "skills_execute_plan",
        "id":
        "t1",
        "thinking":
        "I should try the plan.\nTwo lines.",
        "text":
        "Executing the oracle plan.",
        "args": {
            "plan": "A(x:t)[1]\nB(y:t)[2]",
            "note": "first try"
        },
        "result":
        "[1/2] A(x)[1]: succeeded after 1 steps\n"
        "[2/2] B(y)[2]: succeeded after 1 steps\n  episode: WIN",
    }, {
        "tool": "session_end",
        "id": "t2",
        "text": "Done.",
        "args": {
            "handoff": "won"
        },
        "result": "Session ended.",
    }]
    md = format_conversation_markdown(_entries(calls),
                                      title="Local Sandbox Query",
                                      meta={
                                          "query_number": 1,
                                          "timestamp": "20260904_120000",
                                          "query": "the prompt text"
                                      })
    tx = tr.parse_transcript(md, "001_play_20260904_120000.md")
    assert tx.number == 1 and tx.kind == "play"
    assert tx.stamp is not None and tx.stamp.year == 2026
    assert tx.prompt == "the prompt text"
    assert len(tx.turns) == 2
    first = tx.turns[0]
    assert first.thinking == ["I should try the plan.\nTwo lines."]
    assert first.texts == ["Executing the oracle plan."]
    assert len(first.calls) == 1
    call = first.calls[0]
    assert call.short_name == "skills_execute_plan"
    assert call.args == {"plan": "A(x:t)[1]\nB(y:t)[2]", "note": "first try"}
    assert call.result.startswith("[1/2]") and not call.is_error
    assert tr.entries_consumed(call) == 2
    assert tx.turns[1].calls[0].args == {"handoff": "won"}
    assert "0.42" in tx.result_line


def _oracle_plan_lines(approach: Any, task: Any) -> List[str]:
    approach.solve(task, timeout=10)
    lines = []
    for opt in getattr(approach, "_last_plan"):
        objs = ", ".join(f"{o.name}:{o.type.name}" for o in opt.objects)
        params = ", ".join(f"{float(p):.6f}" for p in opt.params)
        lines.append(f"{opt.name}({objs})[{params}]")
    return lines


def test_replay_pairs_frames_with_reasoning(tmp_path: Any) -> None:
    """A real cover run driven through the tools, with a transcript of the
    same calls: the replay pairs each invocation with its call and text."""
    utils.reset_config({
        "env":
        "cover",
        "approach":
        "oracle",
        "seed":
        5,
        "num_train_tasks":
        1,
        "num_test_tasks":
        0,
        "horizon":
        30,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        100,
        "continual_render":
        True,
        "continual_scorecards_dir":
        os.path.join(str(tmp_path), "cards"),
        "continual_recordings_dir":
        os.path.join(str(tmp_path), "recs"),
        "experiment_id":
        "replay",
    })
    env = create_new_env("cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    ctx = ToolContext(types=set(env.types),
                      predicates=set(env.predicates),
                      options=set(options),
                      env=env)
    made: List[Dict[str, Any]] = []

    class _Driver:

        def play_level(self, session: ProtocolSession) -> None:
            """Reset once, then execute the oracle plan."""
            state = PlayState()
            tools = build_continual_tools(ctx,
                                          session,
                                          state,
                                          save_render=lambda tag: None)
            by_name = {t.name: t for t in tools}
            task = session.observe().level.task
            plan = "\n".join(_oracle_plan_lines(approach, task))
            r1 = asyncio.run(by_name["env_reset"].handler({"note": "warm"}))
            made.append({
                "tool": "env_reset",
                "id": "c1",
                "thinking": "Start clean.",
                "text": "Resetting first.",
                "args": {
                    "note": "warm"
                },
                "result": r1["content"][0]["text"],
            })
            r2 = asyncio.run(by_name["skills_execute_plan"].handler({
                "plan":
                plan,
                "note":
                "oracle"
            }))
            made.append({
                "tool": "skills_execute_plan",
                "id": "c2",
                "text": "Now the plan.",
                "args": {
                    "plan": plan,
                    "note": "oracle"
                },
                "result": r2["content"][0]["text"],
            })

    run = ContinualRun(env, approach, _Driver()).run()
    assert run.levels[0].won
    run_id = run.run_id
    adir = os.path.join(str(tmp_path), "recs", run_id, "agent")
    os.makedirs(os.path.join(adir, "sandbox"), exist_ok=True)
    md = format_conversation_markdown(_entries(made),
                                      title="Local Sandbox Query",
                                      meta={
                                          "query_number": 1,
                                          "timestamp": "20200101_000000",
                                          "query": "prompt"
                                      })
    with open(os.path.join(adir, "001_play_20200101_000000.md"),
              "w",
              encoding="utf-8") as f:
        f.write(md)

    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"))
    frames = viewer.build_replay(run_id, 0)
    events = [f["event"] for f in frames]
    assert events[0] == "level_start"
    assert "reset" in events and "win" in events
    n_invoke = events.count("invoke")
    assert n_invoke == run.levels[0].skill_invocations
    reset = next(f for f in frames if f["event"] == "reset")
    assert reset["call"]["name"] == "env_reset"
    assert reset["thinking"] == ["Start clean."]
    assert reset["texts"] == ["Resetting first."]
    invokes = [f for f in frames if f["event"] == "invoke"]
    assert all(f["call"]["name"] == "skills_execute_plan" for f in invokes)
    assert invokes[0]["texts"] == ["Now the plan."]
    assert invokes[0]["session"] == 1 and invokes[0]["turn"] == 2
    assert all(f["render"] for f in frames if f["event"] != "resume")
    assert any(f["marker"] for f in frames)

    html = viewer.replay_page(run_id, 0)
    assert html is not None
    assert "window.FRAMES" in html and "Agent reasoning" in html
    assert "Download JSON" in html and "filmstrip" in html
    data = viewer.replay_json(run_id, 0)
    assert data is not None
    assert json.loads(data.decode("utf-8"))[0]["event"] == "level_start"
    assert viewer.replay_page(run_id, 7) is None
    assert viewer.replay_json("nope", 0) is None

    session = viewer.session_page(run_id, "001_play_20200101_000000.md")
    assert session is not None
    assert "Start clean." in session and "class='think'" in session
    assert "skills_execute_plan" in session and "id='turn-2'" in session
    assert "Now the plan." in session
    level = viewer.level_page(run_id, 0)
    assert level is not None and "/L1/replay" in level
