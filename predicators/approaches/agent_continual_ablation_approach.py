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
    """Revise declared values and ranges without numerical estimation."""

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_no_fitting"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not CFG.agent_sim_learn_declared_params_only:
            raise ValueError("No-fitting arm requires declared_params_only")
        super().__init__(*args, **kwargs)

    def _fit_status_text(self) -> str:
        return "agent-declared parameter values and ranges; no numerical fit"


class AgentContinualNoUncertaintyApproach(AgentContinualApproach):
    """Noise-aware estimation followed by point-estimate decisions."""

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
        )
        enabled = ("code_sim_learning_rollout_noise_filter",
                   "continual_belief_frame")
        wrong = [name for name in disabled if getattr(CFG, name)]
        wrong += [name for name in enabled if not getattr(CFG, name)]
        if wrong:
            raise ValueError("Invalid point-estimate configuration: " +
                             ", ".join(wrong))
        super().__init__(*args, **kwargs)
