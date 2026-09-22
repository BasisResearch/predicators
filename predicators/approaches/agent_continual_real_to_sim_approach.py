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

from typing import Any

from predicators.agent_sdk.tools.exploration import ProbeSurface
from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoFittingApproach
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach
from predicators.settings import CFG

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


class AgentContinualRealToSimApproach(AgentContinualFromAssetsApproach,
                                      AgentContinualNoFittingApproach):
    """Agent-built scenes without harness fitting or uncertainty."""

    _save_suffix = "AgentContinualRealToSim"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wrong = [name for name in _UNCERTAINTY_FLAGS if getattr(CFG, name)]
        if wrong:
            raise ValueError("The real-to-sim arm runs without uncertainty "
                             "machinery; switch off " + ", ".join(wrong))
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_real_to_sim"

    def _probe_surface(self) -> ProbeSurface:
        return ProbeSurface(fit=False,
                            edit_model=True,
                            sealed=False,
                            alt_params=True,
                            uncertainty=False)
