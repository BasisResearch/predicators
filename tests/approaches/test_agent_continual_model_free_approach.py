"""Tests for AgentContinualModelFreeApproach: the model-free baseline of the
continual protocol, driven by a scripted agent over the real play tools on
pybullet_boil (the agent approaches need a PyBullet env to construct)."""
import asyncio
import os
import pickle
from typing import Any, Dict, List

from predicators import utils
from predicators.agent_sdk.sandbox_setup import trajectories_path
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES
from predicators.approaches import BaseApproach, create_approach
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach, AgentContinualModelFreeApproach
from predicators.approaches.agent_sim_learning_approach import \
    resolve_kept_predicate_names
from predicators.approaches.continual_play_mixin import ContinualPlayMixin
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
        "continual_max_idle_rounds":
        3,
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
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


def _make_approach(name: str = "agent_continual_model_free") -> Any:
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    approach = create_approach(name, env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, ContinualPlayMixin)
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
    assert approach._get_all_predicates() == set()  # pylint: disable=protected-access
    prompt = approach._get_agent_system_prompt()  # pylint: disable=protected-access
    assert "`learn_run`" not in prompt and "`run_python`" not in prompt
    assert "`sim`" not in prompt and "## Learning" not in prompt
    assert "no learned model" in prompt and "`give_up`" in prompt
    assert "`handoff`" not in prompt and "## Your context" in prompt
    # The play loop is a mixin in front of each arm's phased base, not
    # an approach of its own, so the registry never sees it.
    assert not issubclass(ContinualPlayMixin, BaseApproach)
    assert issubclass(AgentContinualApproach, ContinualPlayMixin)
    assert issubclass(AgentContinualModelFreeApproach, ContinualPlayMixin)
    assert AgentContinualApproach.get_name() == "agent_continual"


def test_play_loop_with_a_scripted_model_free_agent(tmp_path: Any) -> None:
    """Two rounds of one conversation: act and stop; then give up.

    The loop records each round, checkpoints under the arm's own suffix,
    and the query carries the data status instead of a learning status.
    The arm's only data surface, the sandbox's data file, follows the
    recording inside the round.
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
            assert "first round of the run" in message
            assert "no belief model" in message
            assert "Your model" not in message
            assert "not expressible in your predicates" in message
            assert "Goal: Boil" in message
            obs = _call(approach, "env_observe")
            assert "[episode] NOT_FINISHED" in obs and "[atoms] (none)" in obs
            assert "[your predicates]" not in obs
            assert "PickJug" in _call(approach, "skills_list")
            data = trajectories_path(approach._tool_context.sandbox_dir)  # pylint: disable=protected-access
            for i in range(3):
                assert "step applied" in _call(approach,
                                               "env_step",
                                               action=zero)
                with open(data, "rb") as f:
                    on_disk = pickle.load(f)
                assert [len(t["actions"]) for t in on_disk] == [i + 1]
            refused = _call(approach, "env_step", action=zero[:-1])
            assert refused.startswith("ERROR") and "shape" in refused
            with open(data, "rb") as f:
                assert [len(t["actions"]) for t in pickle.load(f)] == [3]
        else:
            assert "you stopped" in message and "not settled" in message
            assert "Give-up recorded" in _call(approach,
                                               "give_up",
                                               note="enough")
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    run = ContinualRun(env, approach, create_controller(env, approach))
    card = run.run()

    assert card.end_reason == "agent_ended" and card.end_note == "enough"
    assert [q["kind"] for q in queries] == ["play", "play"]
    lv = card.levels[0]
    assert lv.steps == 3 and lv.resets == 0 and not lv.won
    assert lv.sandbox["rounds"] == 2 and "fits" not in lv.sandbox
    assert lv.sandbox["sim_rollouts"] == 0
    trajs = approach._online_trajectories  # pylint: disable=protected-access
    assert len(trajs) == 1 and len(trajs[0].actions) == 3
    log_dir = approach._get_log_dir()  # pylint: disable=protected-access
    assert log_dir.startswith(os.path.join(str(tmp_path), "runs"))
    assert log_dir.endswith("agent")
    attempts = open(os.path.join(log_dir, "sandbox", "attempts.md"),
                    encoding="utf-8").read()
    assert "### Round 1" in attempts and "Learning session" not in attempts
    saved = [
        f for f in os.listdir(os.path.join(str(tmp_path), "saved"))
        if f.endswith(".AgentContinualModelFree")
    ]
    assert saved, "the approach checkpointed under its own suffix"


def test_both_arms_start_with_no_predicates(tmp_path: Any) -> None:
    """Neither arm starts with an env predicate; the allowlist can hand either
    some, and ``["none"]`` spells the empty vocabulary."""
    _config(tmp_path)
    assert resolve_kept_predicate_names(None) is None
    assert resolve_kept_predicate_names(frozenset()) == frozenset()
    utils.update_config({"agent_sim_learn_kept_predicates_names": ["none"]})
    assert resolve_kept_predicate_names(None) == frozenset()
    utils.update_config(
        {"agent_sim_learn_kept_predicates_names": ["Holding", "none"]})
    assert resolve_kept_predicate_names(None) == frozenset({"Holding", "none"})

    _config(tmp_path, approach="agent_continual")
    utils.update_config({"agent_sim_learn_kept_predicates_names": []})
    env, learner = _make_approach("agent_continual")
    assert isinstance(learner, AgentContinualApproach)
    assert learner._get_all_predicates() == set()  # pylint: disable=protected-access
    names = learner._get_solve_tool_names()  # pylint: disable=protected-access
    assert names == ["run_python"] + list(CONTINUAL_TOOL_NAMES)
    prompt = learner._get_agent_system_prompt()  # pylint: disable=protected-access
    assert "You start with no predicates" in prompt
    assert "## Your model" in prompt and "`sim`" in prompt

    _config(tmp_path,
            agent_sim_learn_kept_predicates_names=["Holding"],
            agent_planner_use_simulator=True)
    _, free = _make_approach()
    pred_names = {p.name for p in free._get_all_predicates()}  # pylint: disable=protected-access
    assert pred_names == {"Holding"}
    # The model-free arm holds no simulator whatever the flag says.
    assert free._option_model is None  # pylint: disable=protected-access
    assert {p.name for p in env.predicates} > pred_names
