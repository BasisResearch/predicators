"""The no-explicit-uncertainty ablation of EMPIRIC from assets.

The agent starts as EMPIRIC from assets does, from the domain-general
``SceneBase``, the scene manifest and the asset files, and writes the
masses, materials, readings and mechanisms of its model. The harness
still fits the declared parameters, to the raw recorded observations,
and every rollout runs one value for each. Nothing represents or uses
uncertainty: the observation noise is not declared, no state is
smoothed, no parameter belief is kept and no joint draw is made. With no
draws there is no harness-run rehearsal, so every gate, note and check
that reads one is off too. The agent plans by rolling plans out in its
fitted model with the ``sim`` probe, and no harness gate sits on that.
"""
from __future__ import annotations

from typing import Any, List

from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoUncertaintyApproach
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.settings import CFG

# The harness switches this arm refuses. All but the last read a belief,
# its joint draws or a harness-run rehearsal. Most would do nothing without
# the draws, and two would refuse every skill request:
# continual_practise_after_reset asks for a joint rehearsal that cannot
# run, and continual_claim_check asks for a record of a belief.
_REFUSED_SWITCHES = (
    # The rehearsal on the joint draws and the gates on it.
    "continual_rehearsal_gate",
    "continual_rehearsal_notes_gate",
    "continual_rehearsal_safe_plan",
    "continual_rehearsal_still_start",
    "continual_rehearsal_win_first",
    "continual_rehearsal_game_over_gate",
    "continual_practise_after_reset",
    "continual_claim_check",
    "continual_skill_preflight",
    "belief_undeclared_material_screen",
    # The notes a joint rehearsal's report carries.
    "continual_rehearsal_mean_verdict",
    "continual_rehearsal_game_over",
    "continual_rehearsal_support_note",
    "continual_rehearsal_speed_note",
    # The check of a real step against its rehearsal's draws.
    "continual_predictive_check",
    # The parameter belief of a fit and what its draws cover.
    "belief_material_lines",
    "belief_keep_undeclared_materials",
    "belief_segment_temperature",
    "continual_fit_bound_note",
    # The state belief over a level's opening frames.
    "continual_pooled_level_init",
    # The fit's reach check reads a gap in multiples of the declared
    # noise. This arm declares none, so each noisy extreme of a recording
    # would read as a gap.
    "continual_fit_reach_check",
)


class AgentContinualFromAssetsNoUncertaintyApproach(
        AgentContinualFromAssetsApproach, AgentContinualNoUncertaintyApproach):
    """Agent-built scene models, a harness fit, point estimates only."""

    _save_suffix = "AgentContinualFromAssetsNoUncertainty"
    # The fit still starts from the agent's guesses, so what EMPIRIC from
    # assets asks of a fit over guesses stays (``_belief_over_guesses``):
    # the fit's prior spans each declared range and its search covers the
    # whole box (code_sim_learning_prior_spans_bounds), a declared
    # material's range covers at least its plausible range, and a
    # test-level step waits for a fit of the current file. No belief
    # follows from the fit: belief_joint_draws is 0, so
    # ``parameter_belief`` is None.
    refused_switches = _REFUSED_SWITCHES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        wrong = [name for name in self.refused_switches if getattr(CFG, name)]
        if wrong:
            raise ValueError(
                "EMPIRIC from assets without explicit uncertainty keeps no "
                "belief, declares no noise and has no harness-run "
                "rehearsal; turn off " + ", ".join(wrong))
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_from_assets_no_uncertainty"

    # -- The prompt ---------------------------------------------------------

    def _play_model_contract(self, **options: Any) -> str:
        return super()._play_model_contract(sampled_materials=False,
                                            fitted_materials=True,
                                            **options)

    # -- Materials the model does not declare -------------------------------
    # No draw exists to sample them in, and no belief to trace a line for
    # them: the fit and every rollout keep the scene's own value of each.

    def _sampled_material_specs(self) -> List[ParamSpec]:
        return []

    def belief_free_specs(self) -> List[ParamSpec]:
        return []

    def _material_screen_specs(self) -> List[ParamSpec]:
        return []
