"""Tests for the continual-protocol play tools over a real session on the cover
env, driven directly (no SDK)."""
import asyncio
import glob
import json
import os
import re
from typing import Any, Dict, List

import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.play_prompts import build_model_contract, \
    build_play_query, build_play_system_prompt, render_data_status
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    PlayState, build_continual_tools, context_status, format_observation, \
    parse_plan_lines
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
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
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
        assert "learn_run" not in CONTINUAL_TOOL_NAMES
        obs = _call(tools, "env_observe")
        assert "[episode] NOT_FINISHED" in obs
        assert "[atoms]" in obs and "[objects]" in obs and "[ledger]" in obs
        # The env's atoms are listed as the environment's, not the
        # arm's, even without a simulator env on the context.
        ctx.env = None
        obs = _call(tools, "env_observe")
        ctx.env = env
        assert "[your predicates]" not in obs
        assert "IsBlock(" in obs.split("[atoms]")[1].split("\n")[0]
        # The tool subset an arm declares is what it gets.
        subset = build_continual_tools(ctx,
                                       session,
                                       state,
                                       save_render=lambda tag: None,
                                       tool_names=["env_observe", "give_up"])
        assert [t.name for t in subset] == ["env_observe", "give_up"]
        listing = _call(tools, "skills_list")
        assert "PickPlace" in listing and "One skill per line" in listing
        plan = _oracle_plan_text(approach, session.observe().level.task)
        out = _call(tools, "skills_execute_plan", plan=plan, note="oracle")
        assert "episode: WIN" in out
        assert "[ledger]" in out
        assert state.charged_calls == len(plan.splitlines())
        # Sandbox accounting reaches the card on disk at once.
        session.record_sandbox("sim_rollouts", 2)
        cards = glob.glob(
            os.path.join(CFG.continual_runs_dir, "*", "*", "seed*", "run_*",
                         "scorecard.json"))
        assert len(cards) == 1
        with open(cards[0], "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["levels"][0]["sandbox"]["sim_rollouts"] == 2
        # Charged calls after the win are refused with guidance.
        refused = _call(tools, "skills_invoke", skill=plan.splitlines()[0])
        assert refused.startswith("ERROR") and "already won" in refused
        refused = _call(tools, "env_reset", note="again")
        assert "already won" in refused
        obs = _call(tools, "env_observe")
        assert "WIN" in obs and "Write your notes and stop" in obs
        assert "[context] size not reported yet" in obs
        seen["ok"] = True

    driver.body = body
    card = ContinualRun(env, approach, driver).run()
    assert seen["ok"]
    assert card.levels[0].won and card.levels[1].won


def test_tools_divergence_reset_and_errors(tmp_path: Any) -> None:
    """Expected outcomes, parse errors, game over then reset, learn and end-run
    requests, and the run-ended path."""
    env, approach, ctx = _setup(tmp_path, continual_steps_per_level=6)
    # Play under a two-step episode horizon so the episode ends in
    # GAME_OVER after two skills.
    first_task = env.get_train_tasks()[0].task
    plan = _oracle_plan_text(approach, first_task).splitlines()
    assert len(plan) >= 2
    utils.update_config({"continual_episode_horizon": 2})
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
        # The run end is queued, never executed by the tool.
        assert "Give-up recorded" in _call(tools, "give_up", note="stop")
        assert state.pending_give_up == "stop"
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
    the level as lost, and later charged calls say to stop."""
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
            # The test level plays under a two-step episode horizon.
            utils.update_config({"continual_episode_horizon": 2})
            return
        obs = _call(tools, "env_observe")
        assert "(test task 0, no resets)" in obs
        assert "(none on this level)" in obs
        refused = _call(tools, "env_reset", note="early")
        assert refused.startswith("ERROR") and "no resets" in refused
        assert session.level_card().steps == 0
        assert "step applied" in _call(tools, "env_step", action=[0.5])
        out = _call(tools, "env_step", action=[0.5])
        assert "GAME_OVER" in out and "lost" in out and "stop" in out
        assert session.level_card().lost
        refused = _call(tools, "env_step", action=[0.5])
        assert refused.startswith("ERROR") and "stop" in refused
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

    system = build_play_system_prompt(["run_python"] +
                                      list(CONTINUAL_TOOL_NAMES),
                                      base_sim_refs=["./reference/base.py"])
    for name in CONTINUAL_TOOL_NAMES:
        assert f"`{name}`" in system
    assert "counts one step" in system and "very expensive" in system
    assert "never a retry button" in system
    assert "## Your context" in system and "`[context]`" in system
    assert "`handoff`" not in system and "`session_end`" not in system
    # The [context] line the tools and queries carry, from the streamed
    # usage and compaction entries the sandbox session feeds the context.
    assert context_status(ctx) == ("[context] size not reported yet; 0 turns "
                                   "this run; compacted 0x")
    ctx.note_stream_entry({
        "type": "assistant",
        "usage": {
            "input_tokens": 1000,
            "cache_read_input_tokens": 86000
        }
    })
    ctx.note_stream_entry({"type": "system", "subtype": "compact_boundary"})
    ctx.context_window_tokens = 200000
    assert context_status(ctx) == ("[context] ~87k tokens of 200k; 1 turns "
                                   "this run; compacted 1x")
    assert "## Your model" in system and "`sim`" in system
    assert "`sim.fit()`" in system and "Model early and often" in system
    assert "./reference/base.py" in system
    # The model-free arm's prompt describes neither a model nor tools it
    # does not have.
    free = build_play_system_prompt(list(CONTINUAL_TOOL_NAMES))
    assert "`run_python`" not in free
    assert "`sim`" not in free and "## Your model" not in free
    assert "simulator.py" not in free
    assert "no learned model" in free
    assert "`give_up`" in free and "./data/trajectories.pkl" in free
    data = render_data_status(n_episodes=3, n_steps=40)
    assert "no belief model" in data and "3 (40 steps)" in data
    assert "Skill grammar" in system and "./test_images/" in system
    status = "Your model: simulator.py v2"
    query = build_play_query(kind="first",
                             round_number=1,
                             level_number=1,
                             levels_total=2,
                             goal_nl="",
                             goal_atoms=[goal],
                             ledger="[ledger] x",
                             context="[context] c",
                             observation="obs",
                             skills="skills",
                             predicates="preds",
                             types="types",
                             model=status,
                             journal="",
                             attempts="")
    assert "first round of the run" in query
    assert "(empty: no journal yet)" in query and "[context] c" in query
    assert "not expressible" not in query
    query2 = build_play_query(kind="resumed",
                              round_number=3,
                              level_number=2,
                              levels_total=2,
                              goal_nl="do it",
                              goal_atoms=[],
                              ledger="l",
                              context="c",
                              observation="o",
                              skills="s",
                              predicates="p",
                              types="t",
                              model=status,
                              journal="j",
                              attempts="a")
    assert "interrupted by a compute preemption" in query2
    assert "do it" in query2 and "\nj\n" in query2
    assert "not expressible in your predicates" in query2
    # A continuation is short: the conversation already holds the level.
    query3 = build_play_query(kind="continue",
                              round_number=2,
                              level_number=1,
                              levels_total=2,
                              goal_nl="topple the purple one",
                              goal_atoms=[goal],
                              ledger="l",
                              context="c",
                              observation="o",
                              skills="s",
                              predicates="p",
                              types="t",
                              model=status,
                              journal="j",
                              attempts="a")
    assert "you stopped" in query3 and "not settled" in query3
    assert "## Skills" not in query3 and "purple" not in query3

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


def test_probe_reset_from_the_current_observation(tmp_path: Any) -> None:
    """``sim.reset(current=True)`` starts from the last real observation the
    session recorded, with overrides on top; without one it says so."""
    env, _, ctx = _setup(tmp_path)
    task = env.get_train_tasks()[0].task
    ctx.current_task = task
    probe = BeliefProbe(ctx)
    with pytest.raises(ValueError, match="No real observation"):
        probe.reset(current=True)
    with pytest.raises(ValueError, match="task_idx does not apply"):
        probe.reset(task_idx=0, current=True)
    block = sorted(task.init, key=str)[0]
    moved = task.init.copy()
    moved.set(block, "pose", 0.123)
    ctx.current_observation = moved
    probe.reset(current=True)
    assert probe.state(block.name)["pose"] == pytest.approx(0.123)
    assert probe.reset().state(block.name)["pose"] == \
        pytest.approx(task.init.get(block, "pose"))
    probe.reset(current=True, mods={block.name: {"pose": 0.456}})
    assert probe.state(block.name)["pose"] == pytest.approx(0.456)
    assert ctx.current_observation.get(block, "pose") == pytest.approx(0.123)


_DOMAIN_WORDS = re.compile(
    r"\b(glue|glued|weld|welded|jug|burner|kettle|domino|fan|bridge|span|"
    r"leg|busyboard|boil|lamp|button|breaker|coffee|cup)s?\b", re.IGNORECASE)


def test_model_contract_is_domain_general_and_only_for_the_model_arm() -> None:
    """The model arm's prompt carries the contract of its model files, by
    observability, naming no environment; the model-free arm's has none."""
    fo = build_model_contract(partially_observable=False)
    po = build_model_contract(
        partially_observable=True,
        physical_params_section="## Base-sim system identification\n- `mu`")
    for text in (fo, po):
        assert "## The model files" in text and "## `simulator.py`" in text
        assert "RESIDUAL_RULES:" in text and "RESIDUAL_FEATURES:" in text
        assert "cmds.apply_force(ball, (fx, fy, 0.0))" in text
        assert "`cmds.attach(obj_a, obj_b)`" in text
        assert "## Writing conditions" in text
        assert "ParamSpec(name, init_value, lo=None, hi=None" in text
        assert "LEARNED_PREDICATES: List[Predicate]" in text
        assert "`sim.predicates()`" in text and "`Wait` terminates" in text
        assert "__" not in text.replace("__init__", "")
        assert not _DOMAIN_WORDS.search(text), _DOMAIN_WORDS.search(text)
    assert "def rule(state, updates, params):" in fo
    assert "def filling(state, updates, params):" in fo
    assert "def blowing(state, updates, params, cmds):" in fo
    assert "## Hidden state" not in fo and "latent" not in fo.lower()
    assert "system identification" not in fo
    assert "def rule(state, latent, history, updates, params):" in po
    assert "def blowing(state, latent, history, updates, params, cmds):" in po
    assert "## Hidden state" in po and "LATENT_INIT = {}" in po
    assert "cmds.attach(a, b)" in po and "history[-1][0]" in po
    assert "latent=None" in po and "- `mu`" in po
    # Placed after the model section of the model arm's prompt only.
    system = build_play_system_prompt(["run_python"] +
                                      list(CONTINUAL_TOOL_NAMES),
                                      model_contract=po)
    assert system.index("## Your model") < system.index("## The model files")
    assert system.index("## The model files") < system.index("## Journal")
    free = build_play_system_prompt(list(CONTINUAL_TOOL_NAMES),
                                    model_contract=po)
    assert "## The model files" not in free and "RESIDUAL_RULES" not in free


def test_an_invented_predicate_under_an_env_name_stays_the_arms(
        tmp_path: Any) -> None:
    """The vocabulary is matched by identity: a predicate the arm invents under
    the goal predicate's name neither reveals the goal atoms nor lists as the
    environment's, while the env's own object does both."""
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.agent_sdk.tools.continual_tools import _split_atoms, \
        visible_atoms, visible_goal
    from predicators.structs import Predicate
    env, _, ctx = _setup(tmp_path)
    task = env.get_train_tasks()[0].task
    state = task.init
    goal_pred = next(iter(task.goal)).predicate
    # With the env's own predicate object in the vocabulary the goal is
    # expressible and its atoms are the environment's.
    ctx.predicates = {goal_pred}
    assert visible_goal(ctx, task) == sorted(str(a) for a in task.goal)
    env_origin, invented = _split_atoms(ctx, visible_atoms(ctx, state),
                                        env.predicates)
    assert not invented and all(goal_pred.name in a for a in env_origin)
    # An invented look-alike: same name, same types, the arm's classifier.
    lookalike = Predicate(goal_pred.name, list(goal_pred.types),
                          lambda s, o: True)
    assert lookalike == goal_pred and lookalike is not goal_pred
    ctx.predicates = {lookalike}
    assert visible_goal(ctx, task) == []
    env_origin, invented = _split_atoms(ctx, visible_atoms(ctx, state),
                                        env.predicates)
    assert not env_origin and invented
    assert all(a.startswith(goal_pred.name + "(") for a in invented)
