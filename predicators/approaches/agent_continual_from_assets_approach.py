"""EMPIRIC with agent-built scenes, fitting and uncertainty intact.

Only generic robot infrastructure, reconstructed geometry and assets are
supplied. The model owns scene construction, observation mappings,
mechanisms, parameters and inferred memory.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators.agent_sdk.sandbox_setup import ReferenceFiles
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.orchestrator import prior_parameter_belief
from predicators.code_sim_learning.parameter_belief import BeliefConfig, \
    ParameterBelief, join_beliefs, stable_seed
from predicators.code_sim_learning.scene_base import SceneBase, \
    scene_base_class
from predicators.code_sim_learning.utils import read_residual_env
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory, State, Type


class AgentContinualFromAssetsApproach(AgentContinualApproach):
    """Build the scene twin from the engine, the manifest and the assets."""

    _save_suffix = "AgentContinualFromAssets"
    # The model's physics starts from the agent's guesses: the fit's prior
    # spans each declared range (code_sim_learning_prior_spans_bounds), and
    # rehearsal samples the engine materials the model does not declare.
    # The real-to-sim comparison fits nothing and samples nothing.
    _belief_over_guesses = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self._belief_over_guesses and \
                not CFG.code_sim_learning_prior_spans_bounds:
            raise ValueError(
                "EMPIRIC from assets fits parameters that start from the "
                "agent's guesses; run it with "
                "code_sim_learning_prior_spans_bounds (as the "
                "from_assets_opus entry of scripts/configs/empiric/"
                "approaches.yaml does)")
        self._scene_base: Optional[type] = None
        super().__init__(*args, **kwargs)
        # The domain twin never renders or predicts for this arm: the
        # probe has no world until the agent's simulator loads.
        self._tool_context.env = None

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_from_assets"

    def _create_initial_base_env(self, types: Set[Type]) -> Any:
        """Construct only robot infrastructure, never a domain twin."""
        return scene_base_class(CFG.env, types,
                                "")(use_gui=CFG.option_model_use_gui)

    def _make_planning_base_env(self, use_gui: bool = False) -> Any:
        """Every predictive world must use the agent's scene class."""
        cls = getattr(self, "_residual_env_cls", None)
        if cls is None:
            # A disconnected model still needs a substrate for bookkeeping.
            # Probe rollouts are refused by _base_physics_probe_model.
            return self._scene_base_class()(use_gui=use_gui)
        return cls(use_gui=use_gui, skip_residual_dynamics=False)

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

    def _load_simulator_from_module_file(
        self,
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Any:
        """Reject artifacts that do not construct their own scene."""
        result = super()._load_simulator_from_module_file(path, trajectories)
        namespace = result[3]
        if namespace is not None:
            cls = read_residual_env(namespace)
            if cls is None or not issubclass(cls, SceneBase):
                logging.warning(
                    "Assets-only models must export RESIDUAL_ENV as a "
                    "SceneBase subclass; rule-only and domain twins "
                    "are not supported.")
                return None, None, None, None
        return result

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        # The scene package plus the scene base the simulator subclasses;
        # no domain twin, so none of the twin's core modules.
        files = self._scene_package_files()
        files["base_sim/scene_base.py"] = self._standalone_source(
            "scene_base.py")
        return files

    def _base_sim_reference_paths(self) -> List[str]:
        paths = self._scene_package_paths()
        paths.insert(2, "./reference/base_sim/scene_base.py")
        return paths

    # -- The prompt ---------------------------------------------------------

    def _play_prompt_options(self) -> Dict[str, Any]:
        return {"scene_built": True}

    def _play_model_contract(self, **options: Any) -> str:
        return super()._play_model_contract(
            scene_built=True,
            sampled_materials=self._belief_over_guesses,
            **options)

    def _physical_params_prompt_section(self) -> str:
        # No supplied base, no supplied parameter menu.
        return ""

    def _no_model_section(self) -> str:
        return "no_model_assets"

    # -- Materials the model does not declare -------------------------------

    def _sampled_material_specs(self) -> List[ParamSpec]:
        """The engine materials of the agent's scene it does not declare, as
        rehearsal samples them; none before the scene loads."""
        if not self._belief_over_guesses or \
                getattr(self, "_residual_env_cls", None) is None:
            return []
        sampler = getattr(self._base_env, "sampled_material_specs", None)
        return list(sampler()) if callable(sampler) else []

    def _physical_param_names(self) -> Set[str]:
        return super()._physical_param_names() | {
            spec.name
            for spec in self._sampled_material_specs()
        }

    def parameter_belief(self) -> Optional[ParameterBelief]:
        """The model's belief joined with a factor for each engine material it
        does not declare: never fitted, its prior over a plausible range.

        A plan whose success hangs on such a material then fails on some
        joint draws, and a physics sweep tests the ends of its range.
        """
        belief = super().parameter_belief()
        sampled = self._sampled_material_specs()
        if belief is None or not sampled:
            return belief
        ranges = tuple(
            (spec.name, spec.init_value, spec.lo, spec.hi) for spec in sampled)
        fitted = hashlib.sha256(
            repr(belief.names).encode() +
            np.ascontiguousarray(belief.draws).tobytes()).hexdigest()
        key = ("sampled materials", fitted, ranges)
        cache = self._belief_cache()
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            prior = prior_parameter_belief(sampled, {},
                                           BeliefConfig.from_cfg(),
                                           seed=stable_seed(
                                               CFG.seed, "materials", ranges))
            cache[key] = join_beliefs(belief, prior)
        return cache[key]

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
