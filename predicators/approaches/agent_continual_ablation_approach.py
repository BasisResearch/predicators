"""Continual ablations with explicit capability contracts.

Configuration mistakes fail before a run starts rather than producing a
mislabeled baseline. These classes retain the shared continual play
loop.
"""
from typing import Any

from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.settings import CFG


class AgentContinualNoFittingApproach(AgentContinualApproach):
    """Revise declared values and ranges without the harness estimator.

    Only the harness-side fitting is removed (sim.fit, fitted residuals,
    parameter sweeps, the deployment-time fit). The agent may estimate
    values in its own sandbox code; they take effect through the
    declarations.
    """

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_no_fitting"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not CFG.agent_sim_learn_declared_params_only:
            raise ValueError("No-fitting arm requires declared_params_only")
        super().__init__(*args, **kwargs)

    def _no_model_section(self) -> str:
        return "no_model_declared"

    def _fit_status_text(self) -> str:
        return "agent-declared parameter values and ranges; no harness fit"

    def _fit_available(self) -> bool:
        return False


class AgentContinualNoUncertaintyApproach(AgentContinualApproach):
    """Fit raw observations, undeclared noise, no uncertainty handling."""

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_no_uncertainty"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        disabled = (
            "continual_uncertainty_decisions",
            "agent_sim_learn_param_uncertainty",
            "agent_plan_validation_rule_param_margin",
            "agent_plan_validation_physics_margin",
            "agent_explorer_info_seeking",
            "agent_explorer_info_seeking_adaptive",
            "agent_explorer_info_seeking_noise_aware",
            "code_sim_learning_interval_belief",
            "code_sim_learning_carry_posterior",
            "agent_sim_learn_declared_params_only",
            "code_sim_learning_rollout_noise_filter",
            "continual_belief_frame",
            # The arm is not told about the observation noise.
            "continual_obs_noise_declared",
        )
        wrong = [name for name in disabled if getattr(CFG, name)]
        if wrong:
            raise ValueError("Invalid point-estimate configuration: " +
                             ", ".join(wrong))
        super().__init__(*args, **kwargs)
