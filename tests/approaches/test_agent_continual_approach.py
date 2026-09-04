"""Tests for AgentContinualApproach: the play loop over a scripted agent that
drives the real play tools, on pybullet_boil (the sim-learning family needs a
PyBullet env to construct)."""
import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.code_sim_learning.fit_space import FitResult
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.controllers import create_controller
from predicators.structs import Dataset


def _config(tmp_path: Any, **overrides: Any) -> None:
    utils.reset_config({
        "env":
        "pybullet_boil",
        "approach":
        "agent_continual",
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
        "agent_sim_learn_kept_predicates_names": ["Holding"],
        "experiment_id":
        "agenttest",
        **overrides,
    })


def _make_approach() -> Any:
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("agent_continual", env.predicates, options,
                               env.types, env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualApproach)
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


def test_fit_status_text_is_a_point_estimate_line() -> None:
    """The prompt's fit status names each fitted parameter's estimate and
    the sample count, not the result's repr."""
    result = FitResult(names=["lateral_friction", "chain_fwd_min"],
                       samples=np.array([[0.48989795, 0.04], [0.51, 0.04]]),
                       log_probs=np.array([0.0, -0.1]),
                       jacobian=np.zeros((3, 2)))
    render = AgentContinualApproach._fit_status_text  # pylint: disable=protected-access
    fitted: Any = SimpleNamespace(_last_fit_result=result)
    text = render(fitted)
    assert text.startswith("fitted 2 parameter(s) from 2 posterior")
    assert "lateral_friction=" in text and "chain_fwd_min=0.04" in text
    assert "jacobian" not in text and "array(" not in text
    empty: Any = SimpleNamespace(_last_fit_result=None)
    assert render(empty) == "no fit result"


@pytest.mark.slow
def test_play_loop_with_a_scripted_agent(tmp_path: Any) -> None:
    """Two sessions: act, queue learning, end; then end the run.

    The loop services the learn request, records the session, syncs the
    data, checkpoints, and ends the run as the agent asked.
    """
    _config(tmp_path)
    env, approach = _make_approach()
    queries: List[Dict[str, Any]] = []
    learned: List[int] = []

    def fake_learn(trajectories: Any) -> None:
        learned.append(len(trajectories))
        approach._current_simulator_version = "v1"  # pylint: disable=protected-access

    approach._learn_simulator = fake_learn  # type: ignore[method-assign]  # pylint: disable=protected-access

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        queries.append({"message": message, "kind": kwargs.get("kind")})
        n = len(queries)
        zero = [0.0] * env.action_space.shape[0]
        if n == 1:
            assert "first session of the run" in message
            obs = _call(approach, "env_observe")
            assert "[episode] NOT_FINISHED" in obs and "[render]" not in obs
            assert "PickJug" in _call(approach, "skills_list")
            for _ in range(3):
                out = _call(approach, "env_step", action=zero)
                assert "step applied" in out
            assert "queued" in _call(approach, "learn_run", note="first look")
            assert "Session ended" in _call(approach,
                                            "session_end",
                                            handoff="stepped three times")
        else:
            assert "session 2 of the run" in message
            assert "stepped three times" in message
            assert "Learning sessions so far: 1" in message
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
    assert lv.sandbox["sessions"] == 2
    assert lv.sandbox["learn_sessions"] == 1
    assert lv.sandbox["turns"] == 6
    assert lv.sandbox["llm_cost_usd"] == pytest.approx(0.5)
    assert learned == [1], "learning ran once over the recorded episode"
    trajs = approach._online_trajectories  # pylint: disable=protected-access
    assert len(trajs) == 1 and len(trajs[0].actions) == 3
    assert trajs[0].train_task_idx == 0
    # The stable agent directory lives under the recordings root.
    log_dir = approach._get_log_dir()  # pylint: disable=protected-access
    assert log_dir.startswith(os.path.join(str(tmp_path), "recs"))
    attempts = open(os.path.join(log_dir, "sandbox", "attempts.md"),
                    encoding="utf-8").read()
    assert "### Session 1" in attempts and "Learning session 1" in attempts
    assert "Handoff: stepped three times" in attempts
    assert "### Session 2" in attempts
    saved = [
        f for f in os.listdir(os.path.join(str(tmp_path), "saved"))
        if f.endswith(".AgentContinual")
    ]
    assert saved, "the approach checkpointed"


@pytest.mark.slow
def test_resume_reads_the_session_id_and_idle_guard(tmp_path: Any) -> None:
    """A checkpointed in-flight session resumes its transcript; sessions that
    never act trip the idle guard."""
    _config(tmp_path)
    env, approach = _make_approach()
    log_dir = approach._get_log_dir()  # pylint: disable=protected-access
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "session_info.json"),
              "w",
              encoding="utf-8") as f:
        json.dump({"session_id": "old-session"}, f)
    approach._session_in_flight = True  # pylint: disable=protected-access
    resumed: List[Any] = []

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        mgr = approach._agent_session  # pylint: disable=protected-access
        resumed.append((mgr.resume_session_id, "preemption" in message))
        _call(approach, "session_end", handoff="idle")
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_controller(env, approach)).run()
    assert card.end_reason == "agent_ended" and "stalled" in card.end_note
    # First session resumed the old transcript; later ones are fresh.
    assert resumed[0] == ("old-session", True)
    assert all(r == (None, False) for r in resumed[1:])
    assert len(resumed) == 3
