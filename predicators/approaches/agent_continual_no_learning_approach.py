"""The no-learning ablation of EMPIRIC from assets.

The agent starts as EMPIRIC from assets does, from the domain-general
``SceneBase``, the scene manifest and the asset files. Its model is a
``SceneBase`` subclass that holds the scene alone: the agent loads the
bodies, or the base does under ``scene_manifest_loader``, and the file
may add no mechanisms or parameters. The harness fits nothing. The agent
still plans and rehearses in the scene, infers state from noisy
observations and adapts its plans from what it sees.
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Any, Dict, List, Optional

from predicators.agent_sdk.prompt_templates import render
from predicators.agent_sdk.tools.exploration import ProbeSurface
from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoFittingApproach
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory

logger = logging.getLogger(__name__)

# Methods through which a model changes the engine's dynamics. A scene
# may define them only as empty stubs.
_DYNAMICS_METHODS = frozenset({
    "_after_physics_substep",
    "_domain_specific_step",
    "_step_base",
    "_step_once",
    "_step_restored_state",
    "model_state",
    "simulate",
    "step",
    "update_model_state",
})
# Names that add a parameter, a force or model memory wherever they appear.
_MECHANISM_NAMES = frozenset({
    "AGENT_PARAM_SPECS",
    "Attach",
    "ApplyForce",
    "ApplyTorque",
    "CommandBuffer",
    "MODEL_STATE_INIT",
    "PARAM_DRAWS",
    "PARAM_SPECS",
    "ParamSpec",
    "RESIDUAL_RULES",
    "SetVelocity",
    "agent_param",
    "applyExternalForce",
    "applyExternalTorque",
    "queue_residual_commands",
})


def _is_stub(function: ast.FunctionDef) -> bool:
    """Whether a function holds only a docstring and ``pass``."""
    for statement in function.body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Constant):
            continue
        return False
    return True


def mechanism_code(source: str) -> List[str]:
    """What in a simulator file goes beyond loading a scene, sorted.

    Empty for a file that does not parse: the load reports that itself.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name in _DYNAMICS_METHODS and not _is_stub(node):
                found.add(node.name)
        elif isinstance(node, ast.Name) and node.id in _MECHANISM_NAMES:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _MECHANISM_NAMES:
            found.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            found.update(alias.name for alias in node.names
                         if alias.name in _MECHANISM_NAMES)
    # An empty declaration adds nothing.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(
                node.value, (ast.List, ast.Tuple)) and not node.value.elts:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.discard(target.id)
    return sorted(found)


class AgentContinualNoLearningApproach(AgentContinualFromAssetsApproach,
                                       AgentContinualNoFittingApproach):
    """Plan in a loaded scene: no mechanisms, no parameters, no fit."""

    _save_suffix = "AgentContinualNoLearning"
    _belief_over_guesses = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._mechanism_refusal: Optional[str] = None
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_no_learning"

    def _play_prompt_options(self) -> Dict[str, Any]:
        options = super()._play_prompt_options()
        options["frozen_section"] = render(
            "play_frozen", "no_learning_loaded"
            if CFG.scene_manifest_loader else "no_learning")
        options["frozen_scene"] = True
        return options

    def _play_model_contract(self, **options: Any) -> str:
        return super()._play_model_contract(frozen=True,
                                            scene_only=True,
                                            **options)

    def _no_model_section(self) -> str:
        return ("no_model_no_learning_loaded"
                if CFG.scene_manifest_loader else "no_model_no_learning")

    def _fit_status_text(self) -> str:
        return "scene only: no mechanisms, no parameters, no harness fit"

    def _probe_surface(self) -> ProbeSurface:
        return ProbeSurface(fit=False,
                            edit_model=True,
                            sealed=False,
                            alt_params=False,
                            uncertainty=CFG.continual_uncertainty_decisions)

    def _load_simulator_from_module_file(
        self,
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Any:
        """Refuse a file that does more than load the scene."""
        self._mechanism_refusal = None
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                found = mechanism_code(f.read())
            if found:
                self._mechanism_refusal = ", ".join(found)
                logger.warning("No-learning model refused: %s",
                               self._mechanism_refusal)
                return None, None, None, None
        return super()._load_simulator_from_module_file(path, trajectories)

    def _simulator_load_failure(self) -> str:
        if self._mechanism_refusal is None:
            return super()._simulator_load_failure()
        return ("run_python probe: ./simulator.py was refused: this arm's "
                "model only loads the scene, and the file uses "
                f"{self._mechanism_refusal}. Remove them and probe again.")
