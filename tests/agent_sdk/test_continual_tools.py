"""Tests for the continual-protocol play tools over a real session on the cover
env, driven directly (no SDK)."""
import asyncio
import glob
import json
import os
from typing import Any, Dict, List

from predicators import utils
from predicators.agent_sdk.play_prompts import build_play_query, \
    build_play_system_prompt, render_learning_status
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    PlayState, build_continual_tools, format_observation, parse_plan_lines
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun, ProtocolSession, RunEnded
from predicators.run.episode import EpisodeState
from predicators.settings import CFG


class _Driver:
    """A controller that hands the session to the test body."""

    def __init__(self) -> None:
        self.session: Any = None
        self.body: Any = None

    def play_level(self, session: ProtocolSession) -> None:
        """Run the test body against the session."""
        self.session = session
        self.body(session)


def _setup(tmp_path: Any, **overrides: Any) -> Any:
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
        1,
        "horizon":
        30,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        100,
        "continual_render":
        False,
        "continual_reset_cost":
        1,
        "continual_scorecards_dir":
        os.path.join(str(tmp_path), "cards"),
        "continual_recordings_dir":
        os.path.join(str(tmp_path), "recs"),
        "experiment_id":
        "tools",
        **overrides,
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
    return env, approach, ctx


def _call(tools: List[Any], name: str, **args: Any) -> str:
    tool = next(t for t in tools if t.name == name)
    result = asyncio.run(tool.handler(dict(args)))
    text = result["content"][0]["text"]
    if result.get("is_error"):
        return "ERROR: " + text
    return text


def _oracle_plan_text(approach: Any, task: Any) -> str:
    approach.solve(task, timeout=10)
    lines = []
    for opt in getattr(approach, "_last_plan"):
        objs = ", ".join(f"{o.name}:{o.type.name}" for o in opt.objects)
        params = ", ".join(f"{float(p):.6f}" for p in opt.params)
        lines.append(f"{opt.name}({objs})[{params}]")
    return "\n".join(lines)


def test_tools_play_a_level_to_a_win(tmp_path: Any) -> None:
    """observe, list, execute the oracle plan, and the level is won; the win is
    reported and later charged calls are refused."""
    env, approach, ctx = _setup(tmp_path)
    seen: Dict[str, Any] = {}
    driver = _Driver()

    def body(session: ProtocolSession) -> None:
        state = PlayState()
        tools = build_continual_tools(ctx,
                                      session,
                                      state,
                                      save_render=lambda tag: None)
        assert [t.name for t in tools] == CONTINUAL_TOOL_NAMES
        obs = _call(tools, "env_observe")
        assert "[episode] NOT_FINISHED" in obs
        assert "[atoms]" in obs and "[objects]" in obs and "[ledger]" in obs
        listing = _call(tools, "skills_list")
        assert "PickPlace" in listing and "One skill per line" in listing
        plan = _oracle_plan_text(approach, session.observe().level.task)
        out = _call(tools, "skills_execute_plan", plan=plan, note="oracle")
        assert "episode: WIN" in out
        assert "[ledger]" in out
        assert state.charged_calls == len(plan.splitlines())
        # Sandbox accounting reaches the card on disk at once.
        session.record_sandbox("sim_rollouts", 2)
        cards = glob.glob(os.path.join(CFG.continual_scorecards_dir, "*.json"))
        assert len(cards) == 1
        with open(cards[0], "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["levels"][0]["sandbox"]["sim_rollouts"] == 2
        # Charged calls after the win are refused with guidance.
        refused = _call(tools, "skills_invoke", skill=plan.splitlines()[0])
        assert refused.startswith("ERROR") and "already won" in refused
        refused = _call(tools, "env_reset", note="again")
        assert "already won" in refused
        assert "WIN" in _call(tools, "env_observe")
        ended = _call(tools, "session_end", handoff="won it")
        assert "Session ended" in ended
        assert state.session_ended and state.handoff == "won it"
        assert "has ended" in _call(tools, "env_observe")
        seen["ok"] = True

    driver.body = body
    card = ContinualRun(env, approach, driver).run()
    assert seen["ok"]
    assert card.levels[0].won and card.levels[1].won


def test_tools_divergence_reset_and_errors(tmp_path: Any) -> None:
    """Expected outcomes, parse errors, game over then reset, learn and end-run
    requests, and the run-ended path."""
    env, approach, ctx = _setup(tmp_path, continual_steps_per_level=6)
    # Plan with the normal horizon, then play under a two-step horizon
    # so the episode ends in GAME_OVER after two skills.
    first_task = env.get_train_tasks()[0].task
    plan = _oracle_plan_text(approach, first_task).splitlines()
    assert len(plan) >= 2
    utils.update_config({"horizon": 2})
    seen: Dict[str, Any] = {}
    driver = _Driver()

    def body(session: ProtocolSession) -> None:
        state = PlayState()
        tools = build_continual_tools(ctx,
                                      session,
                                      state,
                                      save_render=lambda tag: "./img.png")
        task = session.observe().level.task
        goal = ", ".join(str(a) for a in task.goal)
        # A wrong expectation is a divergence and stops the plan.
        annotated = plan[0] + " -> {" + goal + "}"
        out = _call(tools, "skills_execute_plan", plan=annotated)
        assert "DIVERGENCE" in out and "expected but absent" in out
        assert "[render] ./img.png" in out
        assert session.level_card().divergences == 1
        # A NOT expectation that holds is a divergence too.
        held = sorted(str(a) for a in session.observe().atoms)[0]
        out = _call(tools,
                    "skills_invoke",
                    skill=plan[1] + " -> {NOT " + held + "}")
        assert "DIVERGENCE" in out or "episode: GAME_OVER" in out
        # Parse errors are reported, not raised.
        bad = _call(tools, "skills_invoke", skill="Fly(robot:robot)[1]")
        assert bad.startswith("ERROR") and "parse" in bad
        bad = _call(tools, "skills_invoke", skill="PickPlace()[]")
        assert bad.startswith("ERROR") and "parameter" in bad
        # Horizon 2: the two steps above ended the episode.
        obs = _call(tools, "env_observe")
        assert "GAME_OVER" in obs
        refused = _call(tools, "env_step", action=[0.5])
        assert refused.startswith("ERROR") and "env_reset" in refused
        out = _call(tools, "env_reset", note="fresh start")
        assert "reset done" in out and "NOT_FINISHED" in out
        assert session.level_card().resets == 1
        # Wrong action shape is refused before anything is charged.
        refused = _call(tools, "env_step", action=[0.1, 0.2])
        assert "shape" in refused
        out = _call(tools, "env_step", action=[0.5])
        assert "step applied" in out
        # Learning and run end are queued, never executed by the tool.
        assert "queued" in _call(tools, "learn_run", note="please")
        assert state.pending_learn == "please"
        assert "Run end requested" in _call(tools, "env_end_run", note="stop")
        assert state.pending_end_run == "stop"
        # The step cap (6 per level, 2 levels = 12) is hit inside a tool:
        # the tool reports it and records the run end for the arm.
        for _ in range(40):
            out = _call(tools, "env_step", action=[0.5])
            if "RUN ENDED" in out:
                break
            if out.startswith("ERROR") and "env_reset" in out:
                out = _call(tools, "env_reset", note="again")
                if "RUN ENDED" in out:
                    break
        assert state.run_ended is not None
        assert state.run_ended[0] == "step_cap"
        assert "run has ended" in _call(tools, "env_observe")
        seen["ok"] = True
        raise RunEnded(*state.run_ended)

    driver.body = body
    card = ContinualRun(env, approach, driver).run()
    assert seen["ok"]
    assert card.end_reason == "step_cap"


def test_tools_on_a_level_without_resets(tmp_path: Any) -> None:
    """On a test level env_reset is refused without a charge, GAME_OVER ends
    the level as lost, and later charged calls point at session_end."""
    env, approach, ctx = _setup(tmp_path)
    seen: Dict[str, Any] = {}
    driver = _Driver()

    def body(session: ProtocolSession) -> None:
        state = PlayState()
        tools = build_continual_tools(ctx,
                                      session,
                                      state,
                                      save_render=lambda tag: None)
        if session.level_index == 0:
            plan = _oracle_plan_text(approach, session.observe().level.task)
            out = _call(tools, "skills_execute_plan", plan=plan, note="oracle")
            assert "episode: WIN" in out
            # The test level plays under a two-step horizon.
            utils.update_config({"horizon": 2})
            return
        obs = _call(tools, "env_observe")
        assert "(test task 0, no resets)" in obs
        assert "(none on this level)" in obs
        refused = _call(tools, "env_reset", note="early")
        assert refused.startswith("ERROR") and "no resets" in refused
        assert session.level_card().steps == 0
        assert "step applied" in _call(tools, "env_step", action=[0.5])
        out = _call(tools, "env_step", action=[0.5])
        assert "GAME_OVER" in out and "lost" in out and "session_end" in out
        assert session.level_card().lost
        refused = _call(tools, "env_step", action=[0.5])
        assert refused.startswith("ERROR") and "session_end" in refused
        assert "env_reset" not in refused
        refused = _call(tools, "env_reset", note="again")
        assert refused.startswith("ERROR") and "lost" in refused
        seen["ok"] = True

    driver.body = body
    card = ContinualRun(env, approach, driver).run()
    assert seen["ok"] and card.end_reason == "level_lost"
    assert card.levels[0].won and card.levels[1].lost
    assert card.levels[1].steps == 2 and card.levels[1].resets == 0


def test_parse_plan_lines_and_formatting(tmp_path: Any) -> None:
    """Plan parsing grounds skills with expected outcomes; the prompt builders
    render."""
    env, approach, ctx = _setup(tmp_path)
    task = env.get_train_tasks()[0].task
    plan = _oracle_plan_text(approach, task)
    goal = sorted(str(a) for a in task.goal)[0]
    lines = plan.splitlines()
    text = lines[0] + " -> {" + goal + ", NOT " + goal + "}"
    parsed = parse_plan_lines(text, ctx, task)
    assert len(parsed) == 1
    option, expected, absent = parsed[0]
    assert option.name == "PickPlace"
    assert {str(a) for a in expected} == {goal}
    assert {str(a) for a in absent} == {goal}
    try:
        parse_plan_lines("", ctx, task)
        assert False, "empty text must not parse"
    except ValueError as e:
        assert "no skill line" in str(e)

    system = build_play_system_prompt(["run_python"] + CONTINUAL_TOOL_NAMES,
                                      reset_cost=1000)
    for name in CONTINUAL_TOOL_NAMES:
        assert f"`{name}`" in system
    assert "charged\n  1000 steps" in system or "charged 1000 steps" in system
    assert "Skill grammar" in system and "./test_images/" in system
    none = render_learning_status(n_learn=0,
                                  sim_version=None,
                                  pred_version=None,
                                  fit_status="",
                                  n_episodes=0,
                                  n_steps=0,
                                  n_new_episodes=0)
    assert "No learning session has run yet" in none
    some = render_learning_status(n_learn=2,
                                  sim_version="v2",
                                  pred_version="p1",
                                  fit_status="ok",
                                  n_episodes=4,
                                  n_steps=100,
                                  n_new_episodes=1)
    assert "v2" in some and "p1" in some
    query = build_play_query(session_number=1,
                             resumed=False,
                             level_number=1,
                             levels_total=2,
                             goal_nl="",
                             goal_atoms=[goal],
                             ledger="[ledger] x",
                             observation="obs",
                             skills="skills",
                             predicates="preds",
                             types="types",
                             learning=none,
                             journal="",
                             attempts="",
                             handoff="")
    assert "first session of the run" in query
    assert "(empty: no journal yet)" in query and "(none)" in query
    assert "not expressible" not in query
    query2 = build_play_query(session_number=3,
                              resumed=True,
                              level_number=2,
                              levels_total=2,
                              goal_nl="do it",
                              goal_atoms=[],
                              ledger="l",
                              observation="o",
                              skills="s",
                              predicates="p",
                              types="t",
                              learning=some,
                              journal="j",
                              attempts="a",
                              handoff="h")
    assert "interrupted by a compute preemption" in query2
    assert "do it" in query2 and "\nj\n" in query2
    assert "not expressible in your predicates" in query2

    # format_observation on a live session.
    driver = _Driver()
    seen: Dict[str, Any] = {}

    def body(session: ProtocolSession) -> None:
        obs = session.observe()
        text = format_observation(obs, ctx, with_state=False, render_path=None)
        assert text.startswith("[episode] NOT_FINISHED")
        assert "[objects]" not in text
        assert obs.state is EpisodeState.NOT_FINISHED
        seen["ok"] = True
        session.end_run("done")

    driver.body = body
    ContinualRun(env, approach, driver).run()
    assert seen["ok"]
