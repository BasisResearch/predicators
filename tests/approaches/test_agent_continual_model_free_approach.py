"""Tests for AgentContinualModelFreeApproach: the model-free baseline of the
continual protocol, driven by a scripted agent over the real play tools on
pybullet_boil (the agent approaches need a PyBullet env to construct)."""
import asyncio
import os
from typing import Any, Dict, List

from predicators import utils
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach, AgentContinualModelFreeApproach, \
    ContinualPlayBase
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.controllers import create_controller
from predicators.structs import Dataset

MODEL_FREE_TOOLS = [n for n in CONTINUAL_TOOL_NAMES if n != "learn_run"]


def _config(tmp_path: Any, **overrides: Any) -> None:
    utils.reset_config({
        "env":
        "pybullet_boil",
        "approach":
        "agent_continual_model_free",
        "seed":
        0,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "boil_goal":
        "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "option_model_use_gui":
        False,
        "horizon":
        200,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        500,
        "continual_render":
        False,
        "continual_max_idle_sessions":
        3,
        "continual_scorecards_dir":
        os.path.join(str(tmp_path), "cards"),
        "continual_recordings_dir":
        os.path.join(str(tmp_path), "recs"),
        "approach_dir":
        os.path.join(str(tmp_path), "saved"),
        "agent_sdk_use_local_sandbox":
        True,
        # As the phased agent_model_free arm runs: no simulator at all.
        "agent_planner_use_simulator":
        False,
        "experiment_id":
        "modelfree",
        **overrides,
    })


def _make_approach() -> Any:
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("agent_continual_model_free", env.predicates,
                               options, env.types, env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualModelFreeApproach)
    return env, approach


def _call(approach: Any, name: str, **args: Any) -> str:
    tools = approach._tool_context.extra_mcp_tools  # pylint: disable=protected-access
    tool = next(t for t in tools if t.name == name)
    result = asyncio.run(tool.handler(dict(args)))
    text = result["content"][0]["text"]
    return ("ERROR: " + text) if result.get("is_error") else text


def _result(turns: int = 3, cost: float = 0.25) -> List[Dict[str, Any]]:
    return [{
        "type": "assistant",
        "content": [{
            "type": "text",
            "text": "done"
        }]
    }, {
        "type": "result",
        "subtype": "success",
        "num_turns": turns,
        "total_cost_usd": cost,
        "is_error": False,
        "result": "done",
        "session_id": "sess-1",
    }]


def test_model_free_arm_has_no_model_surface(tmp_path: Any) -> None:
    """The arm offers the env and skill tools only, builds no simulator, and
    its prompt describes neither a model nor tools it does not have."""
    _config(tmp_path)
    env, approach = _make_approach()
    del env
    assert approach.get_name() == "agent_continual_model_free"
    assert approach._get_solve_tool_names() == MODEL_FREE_TOOLS  # pylint: disable=protected-access
    assert approach._option_model is None  # pylint: disable=protected-access
    prompt = approach._get_agent_system_prompt()  # pylint: disable=protected-access
    assert "`learn_run`" not in prompt and "`run_python`" not in prompt
    assert "`sim`" not in prompt and "## Learning" not in prompt
    assert "no learned model" in prompt and "`session_end`" in prompt
    # The shared base is abstract, so the registry never offers it, and
    # the full agent still resolves to its own class.
    assert ContinualPlayBase.__abstractmethods__
    assert issubclass(AgentContinualApproach, ContinualPlayBase)
    assert AgentContinualApproach.get_name() == "agent_continual"


def test_play_loop_with_a_scripted_model_free_agent(tmp_path: Any) -> None:
    """Two sessions: act, end; then end the run.

    The loop records the session, syncs the data, checkpoints under the
    arm's own suffix, and the query carries the data status instead of a
    learning status.
    """
    _config(tmp_path)
    env, approach = _make_approach()
    queries: List[Dict[str, Any]] = []

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        queries.append({"message": message, "kind": kwargs.get("kind")})
        n = len(queries)
        zero = [0.0] * env.action_space.shape[0]
        names = [
            t.name for t in approach._tool_context.extra_mcp_tools  # pylint: disable=protected-access
        ]
        assert names == MODEL_FREE_TOOLS
        if n == 1:
            assert "first session of the run" in message
            assert "no belief model and no learning session" in message
            assert "Learning sessions so far" not in message
            obs = _call(approach, "env_observe")
            assert "[episode] NOT_FINISHED" in obs and "[atoms]" in obs
            assert "[your predicates]" not in obs
            assert "PickJug" in _call(approach, "skills_list")
            for _ in range(3):
                assert "step applied" in _call(approach,
                                               "env_step",
                                               action=zero)
            refused = _call(approach, "env_step", action=zero[:-1])
            assert refused.startswith("ERROR") and "shape" in refused
            assert "Session ended" in _call(approach,
                                            "session_end",
                                            handoff="stepped three times")
        else:
            assert "session 2 of the run" in message
            assert "stepped three times" in message
            assert "Run end requested" in _call(approach,
                                                "env_end_run",
                                                note="enough")
            _call(approach, "session_end", handoff="bye")
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    run = ContinualRun(env, approach, create_controller(env, approach))
    card = run.run()

    assert card.end_reason == "agent_ended" and card.end_note == "enough"
    assert [q["kind"] for q in queries] == ["play", "play"]
    lv = card.levels[0]
    assert lv.steps == 3 and lv.resets == 0 and not lv.won
    assert lv.sandbox["sessions"] == 2 and "learn_sessions" not in lv.sandbox
    assert lv.sandbox["sim_rollouts"] == 0
    trajs = approach._online_trajectories  # pylint: disable=protected-access
    assert len(trajs) == 1 and len(trajs[0].actions) == 3
    log_dir = approach._get_log_dir()  # pylint: disable=protected-access
    assert log_dir.startswith(os.path.join(str(tmp_path), "recs"))
    attempts = open(os.path.join(log_dir, "sandbox", "attempts.md"),
                    encoding="utf-8").read()
    assert "### Session 1" in attempts and "Learning session" not in attempts
    saved = [
        f for f in os.listdir(os.path.join(str(tmp_path), "saved"))
        if f.endswith(".AgentContinualModelFree")
    ]
    assert saved, "the approach checkpointed under its own suffix"
