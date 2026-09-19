"""EMPIRIC with the agentic real-to-sim arm's references."""
import json
from pathlib import Path
from typing import Any, Dict, List

from predicators.approaches import create_approach
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
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

CONFIG = "predicatorv3/continual_empiric_scene_package_benchmark_r1.yaml"


def _arm_flags(env_name: str) -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False)
               if c.env == env_name)
    flags = {k: v for k, v in cfg.flags.items() if k != "log"}
    flags.update(approach=cfg.approach,
                 env=cfg.env,
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any, env_name: str) -> Any:
    _config(tmp_path, **_arm_flags(env_name))
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    approach = create_approach("agent_continual", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualApproach)
    return env, approach


def _refs(sandbox: Path) -> List[str]:
    return sorted(
        str(q.relative_to(sandbox / "reference"))
        for q in (sandbox / "reference").rglob("*") if q.is_file())


def test_config_is_empiric_with_the_scene_package() -> None:
    """Five settings, three seeds, the MB arm with both reference flags."""
    runs = list(generate_run_configs(CONFIG, False))
    assert len(runs) == 15
    for run in runs:
        assert run.approach == "agent_continual"
        assert run.flags["continual_provide_scene_package"] is True
        assert run.flags["agent_sim_provide_base_sim_source"] is True
        assert run.flags["continual_require_model_on_test"] is True
        assert "agent_sim_learn_declared_params_only" not in run.flags


def test_fan_lists_the_twin_core_and_the_package(tmp_path: Any) -> None:
    """A domain with a declared core module gets it beside the package."""
    _, approach = _make(tmp_path, "pybullet_fan")
    files = approach._get_sandbox_reference_files()
    assert "base_sim/pybullet_fan_base.py" in files
    assert "scene/scene_manifest.json" in files
    assert any(k.startswith("assets/") for k in files)
    assert not any("pybullet_fan.py" in k for k in files)
    paths = approach._base_sim_reference_paths()
    assert paths[0] == "./reference/base_sim/pybullet_fan_base.py"
    assert sum("pybullet_env.py" in p for p in paths) == 1
    assert any("scene_manifest.json" in p for p in paths)


def test_boil_plays_on_the_twin_with_the_package(tmp_path: Any,
                                                 monkeypatch: Any) -> None:
    """The references reach the sandbox and the prompt, and sim still runs on
    the domain twin before any model is written."""
    env, approach = _make(tmp_path, "pybullet_boil")
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        refs = _refs(sandbox)
        assert "base_sim/pybullet_env.py" in refs
        assert "base_sim/base_env.py" in refs
        assert "scene/scene_manifest.json" in refs
        assert "assets/urdf/jug-pixel.urdf" in refs
        assert "base_sim/scene_base.py" not in refs
        # The twin's observable core is listed; the hidden module never.
        assert "base_sim/pybullet_boil_base.py" in refs
        assert not any("pybullet_boil.py" in r for r in refs)
        manifest = json.loads(
            (sandbox / "reference/scene/scene_manifest.json").read_text())
        assert manifest["objects"]["jug0"] == "jug"
        prompt = approach._play_system_prompt()
        assert "already builds this scene" in prompt
        assert "scene_manifest.json" in prompt
        assert "SceneBase" not in prompt
        # The twin backs sim with no model written.
        run = _call(approach,
                    "run_python",
                    code="sim.reset(current=True)\n"
                    "print(sim.run('Wait(robot:robot)[]'))")
        assert not run.startswith("ERROR"), run
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1
