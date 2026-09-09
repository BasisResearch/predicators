"""Minimal-knowledge agents driven through real continual tools and physics."""
import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.primitive_policy import open_primitive_policy
from predicators.agent_sdk.sandbox_setup import write_pyguard
from predicators.agent_sdk.tools.continual_tools import PRIMITIVE_TOOL_NAMES
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_minimal_approach import \
    AgentContinualModelFreeMinimalApproach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.structs import Dataset
from scripts.cluster_utils import generate_run_configs

# pylint: disable=protected-access

ARMS = [
    "agent_continual_model_free_minimal", "agent_continual_model_based_minimal"
]


def _setup(tmp_path: Path,
           name: str,
           env_name: str = "pybullet_boil",
           **overrides: Any) -> Any:
    utils.reset_config({
        "env": env_name,
        "approach": name,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "boil_goal": "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "experiment_protocol": "continual",
        "continual_levels": "train_only",
        "continual_steps_per_level": 50,
        "continual_render": False,
        "continual_runs_dir": str(tmp_path / "runs"),
        "approach_dir": str(tmp_path / "saved"),
        "agent_sdk_use_local_sandbox": True,
        "partially_observable": True,
        # Conflicting flags must not hand these arms extra knowledge.
        "agent_planner_use_simulator": True,
        "agent_sim_learn_kept_predicates_names": ["Holding"],
        "use_gt_helpers": True,
        **overrides,
    })
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    options = get_gt_options(env_name)
    approach = create_approach(name, env.predicates, options, env.types,
                               env.action_space,
                               [task.task for task in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualModelFreeMinimalApproach)
    approach.prepare_for_continual(Dataset([]))
    return env, approach


def _call(approach: Any, name: str, **args: Any) -> str:
    tool = next(t for t in approach._tool_context.extra_mcp_tools
                if t.name == name)
    result = asyncio.run(tool.handler(args))
    text = result["content"][0]["text"]
    return ("ERROR: " if result.get("is_error") else "") + text


def _result() -> List[Dict[str, Any]]:
    return [{
        "type": "result",
        "subtype": "success",
        "num_turns": 1,
        "total_cost_usd": 0,
        "is_error": False,
        "result": "done"
    }]


@pytest.mark.parametrize("name", ARMS)
def test_minimal_agent_plays_raw_actions(tmp_path: Path, name: str) -> None:
    """Real PyBullet actions, guarded model import, data refresh and resume."""
    env, approach = _setup(tmp_path, name)
    assert approach._initial_options == set()
    assert approach._initial_predicates == set()
    assert approach._tool_context.options == set()
    assert approach._option_model is None
    assert approach._tool_context.env is None
    assert approach._get_all_predicates() == set()
    prompt = approach._get_agent_system_prompt()
    assert "skills_invoke" not in prompt and "Skill grammar" not in prompt
    assert "env_run_policy" in prompt
    assert ("from reference.base_sim.pybullet_env"
            in prompt) == (name == ARMS[1])

    def query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        assert "PickJug" not in message
        assert [t.name for t in approach._tool_context.extra_mcp_tools] == \
            PRIMITIVE_TOOL_NAMES
        sandbox = Path(approach._tool_context.sandbox_dir)
        refs = sorted(p.name for p in (sandbox / "reference").rglob("*.py"))
        assert refs == (["base_env.py", "pybullet_env.py"]
                        if name == ARMS[1] else [])
        obs_text = _call(approach, "env_observe")
        control = json.loads(obs_text.split("[control] ")[1].split("\n")[0])
        assert len(control["joint_positions"]) == env.action_space.shape[0]
        assert "joint_names" in control["action_space"]
        assert "simulator_state" not in control
        assert "privileged" not in control
        for obj in control["objects"].values():
            assert "heat" not in obj["features"]
        # Both direct and policy actions use the same validation.
        assert "shape" in _call(approach, "env_step", action=[])
        assert "finite" in _call(approach,
                                 "env_step",
                                 action=[float("nan")] *
                                 env.action_space.shape[0])
        assert "bounds" in _call(approach,
                                 "env_step",
                                 action=[1e10] * env.action_space.shape[0])
        hold = control["joint_positions"]
        assert "step applied" in _call(approach, "env_step", action=hold)
        source = ""
        if name == ARMS[1]:
            source = (
                "from reference.base_sim.pybullet_env import PyBulletEnv\n"
                "import pybullet as p\n"
                "assert PyBulletEnv.__name__ == 'PyBulletEnv'\n"
                "client = p.connect(p.DIRECT)\n"
                "p.stepSimulation(physicsClientId=client)\n"
                "p.disconnect(client)\n")
        # Validate that the sandbox module import path works, and that data
        # refreshes while the policy worker is alive, after EVERY step.
        (sandbox / "helpers.py").write_text("LIMIT = 2\n", encoding="utf-8")
        source += ("import pickle\nfrom helpers import LIMIT\n"
                   "def get_action(observation, memory):\n"
                   "    n = memory.get('n', 0)\n"
                   "    with open('data/trajectories.pkl', 'rb') as f:\n"
                   "        data = pickle.load(f)\n"
                   "    assert len(data[-1]['actions']) == n + 1\n"
                   "    memory['n'] = n + 1\n"
                   "    if n == LIMIT:\n"
                   "        return None\n"
                   "    return observation['joint_positions']\n")
        (sandbox / "policy.py").write_text(source, encoding="utf-8")
        result = _call(approach,
                       "env_run_policy",
                       path="policy.py",
                       max_steps=10)
        assert "policy applied 2 step(s)" in result and "ERROR" not in result
        with (sandbox / "data/trajectories.pkl").open("rb") as stream:
            trajectories = pickle.load(stream)
        assert len(trajectories[-1]["actions"]) == 3
        assert all(a["option"] is None for a in trajectories[-1]["actions"])
        assert "inside the sandbox" in _call(approach,
                                             "env_run_policy",
                                             path="../escape.py",
                                             max_steps=1)
        assert "positive integer" in _call(approach,
                                           "env_run_policy",
                                           path="policy.py",
                                           max_steps=True)
        # Domain imports remain refused even on the model-based arm.
        (sandbox / "bad.py").write_text(
            "import predicators.envs.pybullet_boil\n", encoding="utf-8")
        assert "sandbox guard" in _call(approach,
                                        "env_run_policy",
                                        path="bad.py",
                                        max_steps=1)
        (sandbox / "journal.md").write_text("Learned controller.\n",
                                            encoding="utf-8")
        _call(approach, "give_up", note="finished test")
        return _result()

    approach._query_agent_sync = query
    run = ContinualRun(env, approach, approach)
    assert not run.skills
    card = run.run()
    assert card.total_steps == 3
    assert card.levels[0].skill_invocations == 0
    assert card.end_note == "finished test"
    saved = list((tmp_path / "saved").glob("*" + approach._save_suffix))
    assert saved
    approach.load(0)
    assert approach._rounds_played == 1
    assert approach._get_all_options() == set()
    assert approach._option_model is None


@pytest.mark.parametrize("name", ARMS)
def test_policy_wins_and_stops_at_terminal(tmp_path: Path, name: str) -> None:
    """A closed-loop raw controller wins cover and cannot act after WIN."""
    env, approach = _setup(tmp_path,
                           name,
                           "cover",
                           cover_num_blocks=1,
                           cover_num_targets=1,
                           cover_block_widths=[0.1],
                           cover_target_widths=[0.05],
                           cover_initial_holding_prob=0.0)

    def query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del message, kwargs
        sandbox = Path(approach._tool_context.sandbox_dir)
        (sandbox / "policy.py").write_text(
            "def get_action(observation, memory):\n"
            "    objects = observation['objects'].values()\n"
            "    block = next(o['features'] for o in objects\n"
            "                 if o['type'] == 'block')\n"
            "    target = next(o['features'] for o in objects\n"
            "                  if o['type'] == 'target')\n"
            "    pose = (block['pose'] if block['grasp'] == -1\n"
            "            else target['pose'])\n"
            "    return [pose]\n",
            encoding="utf-8")
        result = _call(approach,
                       "env_run_policy",
                       path="policy.py",
                       max_steps=10)
        assert "[episode] WIN" in result
        assert "policy applied 2 step(s)" in result
        assert "already won" in _call(approach, "env_step", action=[0.5])
        return _result()

    approach._query_agent_sync = query
    card = ContinualRun(env, approach, approach).run()
    assert card.levels[0].won and card.total_steps == 2


def test_policy_worker_timeout(tmp_path: Path) -> None:
    """A non-returning policy is killed and its sandbox remains usable."""
    write_pyguard(str(tmp_path), str(Path(__file__).resolve().parents[2]))
    policy = tmp_path / "policy.py"
    policy.write_text("while True:\n    pass\n", encoding="utf-8")

    async def exercise() -> None:
        with pytest.raises(asyncio.TimeoutError):
            async with open_primitive_policy("policy.py", str(tmp_path), 0.5):
                pytest.fail("the infinite module must never load")
        policy.write_text(
            "def get_action(observation, memory):\n"
            "    return [0.5]\n",
            encoding="utf-8")
        async with open_primitive_policy("policy.py", str(tmp_path),
                                         10) as worker:
            assert np.allclose(await worker.action({}), [0.5])

    asyncio.run(exercise())


@pytest.mark.parametrize("ending", ["horizon", "cap", "invalid"])
def test_policy_preserves_partial_execution(tmp_path: Path,
                                            ending: str) -> None:
    """Every applied action remains charged after a cap or a policy error."""
    env, approach = _setup(
        tmp_path,
        ARMS[0],
        "cover",
        cover_initial_holding_prob=0.0,
        continual_steps_per_level=2 if ending == "cap" else 50,
        continual_episode_horizon=2 if ending == "horizon" else None)

    def query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del message, kwargs
        sandbox = Path(approach._tool_context.sandbox_dir)
        source = ("def get_action(observation, memory):\n"
                  "    memory['n'] = memory.get('n', 0) + 1\n"
                  "    return [0.0]\n")
        if ending == "invalid":
            source = source.replace(
                "return [0.0]", "return [2.0] if memory['n'] > 1 else [0.0]")
        (sandbox / "policy.py").write_text(source, encoding="utf-8")
        text = _call(approach,
                     "env_run_policy",
                     path="policy.py",
                     max_steps=10)
        if ending == "cap":
            assert "RUN ENDED" in text and "policy applied 2 step(s)" in text
        elif ending == "horizon":
            assert "GAME_OVER" in text and "policy applied 2 step(s)" in text
            assert "only env_reset" in _call(approach, "env_observe")
            assert "reset done" in _call(approach, "env_reset")
            _call(approach, "give_up", note="done")
        else:
            assert "bounds" in text and "policy applied 1 step(s)" in text
            assert "NOT_FINISHED" in _call(approach, "env_observe")
            _call(approach, "give_up", note="done")
        return _result()

    approach._query_agent_sync = query
    card = ContinualRun(env, approach, approach).run()
    assert card.total_steps == {"horizon": 3, "cap": 2, "invalid": 1}[ending]
    assert card.levels[0].skill_invocations == 0


def test_minimal_sweep() -> None:
    """The launcher sees four uniform arms times exactly five domains."""
    runs = list(
        generate_run_configs(
            "predicatorv3/protocol_continual_minimal_knowledge_sweep.yaml",
            batch_seeds=True))
    assert len(runs) == 20
    assert {r.approach
            for r in runs
            } == set(ARMS + ["agent_continual", "agent_continual_model_free"])
    assert {r.env
            for r in runs} == {
                "pybullet_" + domain
                for domain in ["balloons", "boil", "bridge", "domino", "fan"]
            }
    for run in runs:
        assert run.flags["partially_observable"] is True
        assert run.flags["experiment_protocol"] == "continual"
        assert run.flags["continual_steps_per_level"] == (
            10000 if run.env == "pybullet_bridge" else 5000)


def test_policy_serializes_environment_calls(tmp_path: Path) -> None:
    """A concurrent command cannot change a running policy's environment."""
    env, approach = _setup(tmp_path,
                           ARMS[0],
                           "cover",
                           cover_initial_holding_prob=0.0)

    def query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del message, kwargs
        sandbox = Path(approach._tool_context.sandbox_dir)
        (sandbox / "policy.py").write_text(
            "def get_action(observation, memory):\n"
            "    return [0.0]\n",
            encoding="utf-8")
        tools = {t.name: t for t in approach._tool_context.extra_mcp_tools}

        async def exercise() -> None:
            running = asyncio.create_task(tools["env_run_policy"].handler({
                "path":
                "policy.py",
                "max_steps":
                1
            }))
            # The handler marks the session busy before its first await.
            await asyncio.sleep(0)
            refused = await tools["env_step"].handler({"action": [0.0]})
            assert refused["is_error"]
            assert "A policy is running" in refused["content"][0]["text"]
            result = await running
            assert "policy applied 1 step(s)" in result["content"][0]["text"]

        asyncio.run(exercise())
        assert "step applied" in _call(approach, "env_step", action=[0.0])
        _call(approach, "give_up", note="done")
        return _result()

    approach._query_agent_sync = query
    card = ContinualRun(env, approach, approach).run()
    assert card.total_steps == 2
