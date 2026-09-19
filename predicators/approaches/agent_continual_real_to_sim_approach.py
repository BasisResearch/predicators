"""The agentic real-to-sim baseline: the agent builds the simulator.

The other model arms start from the domain twin (``BaseSimulator``, the
real environment class with its hidden mechanisms disabled). This arm
starts from what a robot deployment has: the physics engine wrapper (the
generic ``PyBulletEnv``), a domain-agnostic scene base, a geometry
manifest of the scene and the asset files its bodies were loaded from,
plus the robot's skills. The agent writes the scene construction, the
sync of every feature a body pose does not carry, and the mechanisms,
and declares its own parameters. The harness fits nothing and runs no
uncertainty machinery; ``sim`` rehearses skills inside the agent's own
simulator once ``./simulator.py`` loads, and has no world before that.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from predicators.agent_sdk.tools.exploration import ProbeSurface
from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoFittingApproach
from predicators.code_sim_learning.scene_base import scene_base_class
from predicators.settings import CFG
from predicators.structs import State

# Uncertainty machinery this arm runs without (the point-estimate arm's
# list, plus the belief-frame switches the MF menu turns off).
_UNCERTAINTY_FLAGS = (
    "continual_uncertainty_decisions",
    "agent_sim_learn_param_uncertainty",
    "agent_plan_validation_rule_param_margin",
    "agent_plan_validation_physics_margin",
    "agent_explorer_info_seeking",
    "agent_explorer_info_seeking_adaptive",
    "agent_explorer_info_seeking_noise_aware",
    "code_sim_learning_interval_belief",
    "code_sim_learning_carry_posterior",
)


class AgentContinualRealToSimApproach(AgentContinualNoFittingApproach):
    """Build the scene twin from the engine, the manifest and the assets."""

    _save_suffix = "AgentContinualRealToSim"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wrong = [name for name in _UNCERTAINTY_FLAGS if getattr(CFG, name)]
        if wrong:
            raise ValueError("The real-to-sim arm runs without uncertainty "
                             "machinery; switch off " + ", ".join(wrong))
        super().__init__(*args, **kwargs)
        self._scene_base: Optional[type] = None
        # The domain twin never renders or predicts for this arm: the
        # probe has no world until the agent's simulator loads.
        self._tool_context.env = None

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_real_to_sim"

    # -- What the agent receives -------------------------------------------

    def _scene_base_class(self) -> type:
        if self._scene_base is None:
            asset_dir = os.path.join(self._resolve_synthesis_paths().base,
                                     "reference", "assets")
            self._scene_base = scene_base_class(CFG.env, self._types,
                                                asset_dir)
        return self._scene_base

    def _simulator_load_namespace(self) -> Dict[str, Any]:
        namespace = super()._simulator_load_namespace()
        del namespace["BaseSimulator"]
        namespace["SceneBase"] = self._scene_base_class()
        return namespace

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        # The scene package plus the scene base the simulator subclasses;
        # no domain twin, so none of the twin's core modules.
        files = self._scene_package_files()
        files["base_sim/scene_base.py"] = str(
            self._standalone_source(
                "scene_base.py",
                Path(self._get_log_dir()) / "reference_sources"))
        return files

    def _base_sim_reference_paths(self) -> List[str]:
        paths = self._scene_package_paths()
        paths.insert(2, "./reference/base_sim/scene_base.py")
        return paths

    # -- The prompt ---------------------------------------------------------

    def _play_prompt_options(self) -> Dict[str, Any]:
        return {"scene_built": True}

    def _play_model_contract(self, **options: Any) -> str:
        return super()._play_model_contract(scene_built=True, **options)

    def _physical_params_prompt_section(self) -> str:
        # No supplied base, no supplied parameter menu.
        return ""

    def _no_model_section(self) -> str:
        return "no_model_scene"

    def _probe_surface(self) -> ProbeSurface:
        return ProbeSurface(fit=False,
                            edit_model=True,
                            sealed=False,
                            alt_params=True,
                            uncertainty=False)

    # -- No domain twin behind the probe -------------------------------------

    def _base_physics_probe_model(self) -> Any:
        raise RuntimeError(
            "run_python probe: `sim` has no world yet. Write ./simulator.py "
            "(RESIDUAL_ENV, a SceneBase subclass whose initialize_pybullet "
            "loads the scene from reference/scene/scene_manifest.json and "
            "reference/assets/) first; every rollout, reset and render runs "
            "inside it.")

    def _base_predictions(
            self, obs_triples: List[Tuple[State, Any, State]]) -> List[Any]:
        # No engine predicts the recorded transitions for this arm: the
        # workbench's baseline is the recorded state itself, so the
        # residual hint names the features that changed at all.
        return list(obs_triples)

    def _install_residual_env_cls(self,
                                  residual_env_cls: Optional[type],
                                  content_key: Optional[str] = None) -> None:
        super()._install_residual_env_cls(residual_env_cls, content_key)
        # sim.render draws in the agent's own world, never the twin.
        self._tool_context.env = (self._base_env if getattr(
            self, "_residual_env_cls", None) is not None else None)

    def _sync_tool_context(self) -> None:
        super()._sync_tool_context()
        self._tool_context.env = (self._base_env if getattr(
            self, "_residual_env_cls", None) is not None else None)
