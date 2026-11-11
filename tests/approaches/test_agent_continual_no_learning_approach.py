"""The no-learning ablation, and the standalone model's scene assets."""
from pathlib import Path
from typing import Any, Dict, List

import pytest

from predicators.approaches import create_approach
from predicators.approaches.agent_continual_no_learning_approach import \
    mechanism_code
from predicators.code_sim_learning.scene_base import SceneBase
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
    SCENE_SOURCE, _dispose_test_physics_clients

# pylint: disable=protected-access

CONFIG = "empiric/benchmark.yaml"
ARM = "no_learning_opus"
# The real-to-sim test's scene with its mechanism, parameters and model
# memory taken out: what this arm's agent writes.
_SCENE_HEAD = SCENE_SOURCE[:SCENE_SOURCE.index("    AGENT_PARAM_SPECS = [")]
_SCENE_LOAD = SCENE_SOURCE[
    SCENE_SOURCE.index("\n    @classmethod\n    def initialize_pybullet"
                       ):SCENE_SOURCE.index("    def _domain_specific_step")]
LOADED_SCENE = (_SCENE_HEAD +
                '    RESIDUAL_FEATURES = {"jug": ["water_volume"]}\n' +
                _SCENE_LOAD + "\nRESIDUAL_ENV = BoilScene\n")


def _arm_flags(arm: str, env_name: str = "pybullet_boil") -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False, approaches=[arm])
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


def _make(tmp_path: Any, arm: str, **overrides: Any) -> Any:
    flags = {**_arm_flags(arm), **overrides}
    _config(tmp_path, **flags)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    approach = create_approach(flags["approach"], env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    return env, approach


@pytest.mark.parametrize("source, found", [
    ("class S(SceneBase):\n    RESIDUAL_FEATURES = {}\n", []),
    ("class S(SceneBase):\n    AGENT_PARAM_SPECS = []\n", []),
    ("class S(SceneBase):\n    def _domain_specific_step(self):\n"
     "        pass\n", []),
    ("class S(SceneBase):\n    def _domain_specific_step(self):\n"
     "        self.x = 1\n", ["_domain_specific_step"]),
    ("class S(SceneBase):\n    AGENT_PARAM_SPECS = [ParamSpec('a', 1)]\n",
     ["AGENT_PARAM_SPECS", "ParamSpec"]),
    ("from predicators.code_sim_learning.commands import Attach\n", ["Attach"
                                                                     ]),
    ("import pybullet as p\np.applyExternalForce(1, -1, [0], [0], 0)\n",
     ["applyExternalForce"]),
    ("class S(SceneBase):\n    MODEL_STATE_INIT = {'t': 0}\n"
     "    @classmethod\n    def update_model_state(cls, o, m, q, a):\n"
     "        m['t'] += 1\n", ["MODEL_STATE_INIT", "update_model_state"]),
    ("def broken(:\n", []),
])
def test_mechanism_code_finds_what_a_scene_does_not_need(
        source: str, found: List[str]) -> None:
    """Only a scene passes; empty stubs and declarations add nothing."""
    assert mechanism_code(source) == found


def test_config_is_from_assets_without_learning() -> None:
    """Five settings, five seeds, parked, and none of the switches that act on
    a fit or ask for a model revision."""
    assert all(run.approach != "agent_continual_no_learning"
               for run in generate_run_configs(CONFIG, False))
    runs = list(generate_run_configs(CONFIG, False, approaches=[ARM]))
    assert len(runs) == 25
    for run in runs:
        assert run.approach == "agent_continual_no_learning"
        assert run.flags["agent_sim_learn_declared_params_only"] is True
        assert run.flags["continual_rehearsal_gate"] is True
        for absent in ("continual_predictive_check", "continual_claim_check",
                       "belief_material_lines",
                       "code_sim_learning_prior_spans_bounds"):
            assert absent not in run.flags, absent


def test_agent_loads_a_scene_and_plans_without_learning(
        tmp_path: Any, monkeypatch: Any) -> None:
    """The prompt asks for a scene only, a mechanism file is refused with its
    reason, and a loaded scene rehearses and opens the test-level gate."""
    env, approach = _make(tmp_path,
                          ARM,
                          continual_refuse_training_give_up=False)
    assert isinstance(approach._base_env, SceneBase)
    assert approach._probe_surface().fit is False
    prompt = approach._get_agent_system_prompt()
    assert "## No-learning comparison" in prompt
    assert "SceneBase" in prompt and "scene_manifest.json" in prompt
    assert "sim.fit()" not in prompt
    assert "sealed" not in prompt.lower()
    assert "EMPIRIC from assets" not in prompt
    for absent in ("implements the mechanisms", "Write dynamics",
                   "Fitting and validating", "### Hidden model state",
                   "update_widgets"):
        assert absent not in prompt, absent
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        sandbox = Path(approach._tool_context.sandbox_dir)
        assert "add no mechanisms or parameters" in message
        refs = {
            str(q.relative_to(sandbox / "reference"))
            for q in (sandbox / "reference").rglob("*") if q.is_file()
        }
        assert {"base_sim/scene_base.py", "scene/scene_manifest.json"} <= refs
        (sandbox / "simulator.py").write_text(SCENE_SOURCE, encoding="utf-8")
        refused = _call(approach,
                        "run_python",
                        code="sim.reset(current=True)\n"
                        "print(sim.run('Wait(robot:robot)[]'))")
        assert "was refused" in refused and "ParamSpec" in refused, refused
        assert "_domain_specific_step" in refused, refused
        (sandbox / "simulator.py").write_text(LOADED_SCENE, encoding="utf-8")
        loaded = _call(approach,
                       "run_python",
                       code="sim.reset(current=True)\n"
                       "print(sorted(sim.state()))")
        assert not loaded.startswith("ERROR"), loaded
        assert "jug0:jug" in loaded
        run = _call(approach,
                    "run_python",
                    code="print(sim.run('Wait(robot:robot)[2]'))")
        assert not run.startswith("ERROR"), run
        assert "Wait" in run
        fit = _call(approach, "run_python", code="print(sim.fit())")
        assert "sim.fit is unavailable" in fit, fit
        assert type(approach._base_env).__name__ == "BoilScene"
        assert approach._model_readiness(str(sandbox / "simulator.py"),
                                         []) is None
        assert "scene only" in approach._fit_status_text()
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1


def test_standalone_model_gets_the_scene_assets(tmp_path: Any) -> None:
    """Standalone sim.

    receives the manifest and assets, is told so, and still has no base
    simulator.
    """
    _, approach = _make(tmp_path, "standalone_opus")
    files = approach._get_sandbox_reference_files()
    assert "scene/scene_manifest.json" in files
    assert "assets/urdf/jug-pixel.urdf" in files
    assert not any(path.startswith("base_sim/") for path in files)
    prompt = approach._get_agent_system_prompt()
    assert "## Scene files" in prompt
    assert "scene_manifest.json (" in prompt
    assert "no base simulator is supplied" in prompt
    assert "No prepared physical scene" not in prompt
    assert "pybullet_env.py" not in prompt


def test_standalone_run_reads_the_scene_assets(tmp_path: Any,
                                               monkeypatch: Any) -> None:
    """In a run the assets reach the sandbox before the first query, the prompt
    counts the manifest, and the level's recording keeps it."""
    env, approach = _make(tmp_path, "standalone_opus")
    seen: List[str] = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        sandbox = Path(approach._tool_context.sandbox_dir)
        reference = sandbox / "reference"
        assert (reference / "scene/scene_manifest.json").is_file()
        assert (reference / "assets/urdf/jug-pixel.urdf").is_file()
        assert not (reference / "base_sim").exists()
        log_dir = Path(approach._get_log_dir())
        system = next(log_dir.glob("full_system_prompt_*.md")).read_text()
        assert "bodies)" in system, system
        assert (log_dir.parent / "L01" / "scene_manifest.json").read_text() \
            == (reference / "scene/scene_manifest.json").read_text()
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1
