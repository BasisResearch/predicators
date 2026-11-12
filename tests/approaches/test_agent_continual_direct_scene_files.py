"""The direct agent with the scene assets, and no engine wrapper."""
import json
from pathlib import Path
from typing import Any, Dict, List

from predicators.approaches import create_approach
from predicators.approaches.agent_continual_approach import \
    AgentContinualModelFreeApproach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from scripts.cluster_utils import generate_run_configs
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    _dispose_test_physics_clients

# pylint: disable=protected-access

CONFIG = "empiric/benchmark.yaml"
ARM = "mf_scene_package_opus"


def _arm_flags(env_name: str) -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False, approaches=[ARM])
               if c.env == env_name)
    # The launcher pins machine-specific output paths; tests keep their own.
    flags = {
        k: v
        for k, v in cfg.flags.items() if k not in ("log", "continual_runs_dir")
    }
    flags.update(approach=cfg.approach,
                 env=cfg.env,
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any, **overrides: Any) -> Any:
    _config(tmp_path, **{**_arm_flags("pybullet_boil"), **overrides})
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    approach = create_approach("agent_continual_model_free", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualModelFreeApproach)
    return env, approach


def test_config_is_the_direct_agent_with_the_scene_files() -> None:
    """Five settings, five seeds, the MF arm with the package flag and none of
    the model arm's flags."""
    runs = list(generate_run_configs(CONFIG, False, approaches=[ARM]))
    assert len(runs) == 25
    for run in runs:
        assert run.approach == "agent_continual_model_free"
        assert run.flags["continual_provide_scene_package"] is True
        assert run.flags["continual_provide_reward_model"] is True
        assert run.flags["agent_planner_use_simulator"] is False
        assert "continual_require_model_on_test" not in run.flags
        assert "agent_sim_provide_base_sim_source" not in run.flags


def test_boil_gets_the_files_and_only_a_task_prompt(tmp_path: Any,
                                                    monkeypatch: Any) -> None:
    """The files reach the sandbox and the prompt lists them, while the prompt
    asks for no simulator, model, fit or gate and the tools have no probe."""
    # The scripted session ends with give_up on its training level.
    env, approach = _make(tmp_path, continual_refuse_training_give_up=False)
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        sandbox = Path(approach._tool_context.sandbox_dir)
        refs = sorted(
            str(q.relative_to(sandbox / "reference"))
            for q in (sandbox / "reference").rglob("*") if q.is_file())
        assert "skills.md" in refs
        assert "scene/scene_manifest.json" in refs
        assert "assets/urdf/jug-pixel.urdf" in refs
        assert not any(r.startswith("base_sim/") for r in refs)
        assert not any("pybullet_boil" in r for r in refs)
        manifest = json.loads(
            (sandbox / "reference/scene/scene_manifest.json").read_text())
        assert manifest["objects"]["jug0"] == "jug"
        prompt = approach._play_system_prompt()
        assert "## Scene files" in prompt
        assert "scene_manifest.json (" in prompt
        assert "No simulator" not in prompt
        assert "pybullet_env.py" not in prompt
        assert "environment wrapper" not in prompt
        for absent in ("simulator.py", "RESIDUAL_FEATURES", "sim.fit",
                       "sim.validate", "SceneBase", "require a fitted model",
                       "run_python"):
            assert absent not in prompt, absent
        assert "run_python" not in approach._continual_tool_names()
        # The level's reward model, with no engine behind it: the tool and
        # its section, and a verdict of the three public keys.
        assert approach._continual_tool_names()[-1] == "evaluate_trajectory"
        assert "## Interpreting task verdicts" in prompt
        assert "The tool reads only the states you give it" in prompt
        assert "from a state is not checked" in prompt
        scored = _call(approach, "evaluate_trajectory", states=[{}])
        assert json.loads(scored.split("\n\n", 1)[0]) == {
            "reward": 0.0,
            "solved": False,
            "note": ""
        }
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1


def test_flag_off_keeps_the_direct_agent_unchanged(tmp_path: Any) -> None:
    """Without the flag the arm has only the skills reference and no scene
    files section."""
    _, approach = _make(tmp_path, continual_provide_scene_package=False)
    assert set(approach._get_sandbox_reference_files()) == {"skills.md"}
    assert "## Scene files" not in approach._play_system_prompt()


def test_reward_model_flag_off_keeps_the_arm_unchanged(tmp_path: Any) -> None:
    """Without the reward-model flag the arm has no ``evaluate_trajectory``
    tool and its prompt does not mention one."""
    _, approach = _make(tmp_path, continual_provide_reward_model=False)
    assert "evaluate_trajectory" not in approach._continual_tool_names()
    prompt = approach._play_system_prompt()
    assert "evaluate_trajectory" not in prompt
    assert "## Interpreting task verdicts" not in prompt
    assert "## Scene files" in prompt
