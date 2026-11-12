"""EMPIRIC from assets without explicit uncertainty.

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

The agent is never told what the arm leaves out
(``continual_point_values_only``). Its prompts, tool descriptions, probe
reports and refusals say what the model runs at: the latest observation
and one fitted value per parameter. Its ``sim`` has no call built on a
belief, a draw or a stress test, the reference files it reads name none,
and a traceback it is shown leaves out the harness's own frames.

The agent's working directory is its sandbox inside the run folder,
``<runs>/<approach name>/<experiment id>/seed<k>/run_<stamp>/``, and the
agent reads that path. So the approach's registered name and its menu
key name nothing withheld either (``agent_continual_from_assets_point``,
``from_assets_point_opus``: a model at one value per parameter), and the
class refuses an experiment id that does.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from predicators.agent_sdk.point_probe import reward_model
from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles
from predicators.agent_sdk.tools.exploration import ProbeSurface
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

# The comments and docstrings of the reference sources that name the
# observation noise, a belief, a draw or a rehearsal's sampling, each with
# the wording this arm's copy carries. A few name another kind of noise or
# use an everyday word of the same spelling; they are reworded too, so the
# copies need no exception for a comment. The code is the same in every
# arm's copy. An entry whose text is gone from its source fails the
# sandbox setup, so a changed source cannot ship its old wording unseen.
_REFERENCE_WORDING: Dict[str, List[Tuple[str, str]]] = {
    "base_sim/pybullet_env.py": [
        ("on it when a\n# state belief stages its velocity",
         "on it when a\n# state stages its velocity"),
        ("benign ~0.02 rad noise", "benign ~0.02 rad error"),
        ("recording every rollout would tax large\n        # parameter "
         "sweeps for data nobody reads.",
         "recording every rollout would tax large\n        # batches of "
         "rollouts for data nobody reads."),
        ("but is opt-in - a\n        parameter sweep should not pay for a "
         "log nobody reads.",
         "but is opt-in - a\n        batch of rollouts should not pay for a "
         "log nobody reads."),
        ("        # Hot-path filter: only pairs among these bodies",
         "        # Hot-path check: only pairs among these bodies"),
        ("and that noise would bury the", "and that clutter would bury the"),
        ("makes the carry follow the base smoothly.",
         "makes the carry follow the base without a jump."),
        ("        A rollout from a draw of the state belief starts here: "
         "the\n        draw's independent pose noise can put bodies inside "
         "one another,\n",
         "        A rollout can start here: a state whose poses are a "
         "little\n        off can put bodies inside one another,\n"),
        ("supports and attachments; the others keep their drawn poses (see",
         "supports and attachments; the others keep their poses (see"),
        ("virtual objects and the latent block keep the draw's. Under\n"
         "        ``continual_belief_velocities`` each body then moves at "
         "the\n        velocity the draw records (at rest when it records "
         "none), since\n        a rollout from a draw of a rolling body "
         "must roll on.",
         "virtual objects and the latent block keep the state's. Where a\n"
         "        state records velocities, each body then moves at the\n"
         "        velocity it records (at rest when it records none), "
         "since\n        a rollout from the state of a rolling body must "
         "roll on."),
        ("keep the\n        # draw's pose to the gripper.",
         "keep the\n        # state's pose to the gripper."),
        ("a body the draw records as\n            # moving",
         "a body the state records as\n            # moving"),
        ("fail on Euler->Quat float noise at the 1e-2",
         "fail on Euler->Quat float error at the 1e-2"),
        ("a large mismatch can't be benign IK noise, so",
         "a large mismatch can't be benign IK error, so"),
        ("mismatches too big to be IK noise.",
         "mismatches too big to be IK error."),
        ("so they are filtered out by the caller's intersection",
         "so they are left out by the caller's intersection"),
        ("        ``continual_belief_velocities`` a linear velocity recorded "
         "without\n        spin, the form the state belief estimates one "
         "in, gets the spin of",
         "        the setting that stages velocities, a linear velocity "
         "recorded without\n        spin gets the spin of"),
        ("radians; comparisons should", "radians; a check of them should"),
        ("fail an EE-pose comparison and trigger",
         "fail an EE-pose check and trigger"),
        ("and swept a carried deck through them.",
         "and moved a carried deck through them."),
    ],
    "base_sim/base_env.py": [
        ("    # Belief-side verification substrate: a sim-learning approach "
         "stamps\n    # this on its belief env so that",
         "    # Model-side verification substrate: a sim-learning approach "
         "stamps\n    # this on its model's env so that"),
    ],
    "base_sim/scene_base.py": [
        ("the declared value or else the scene's. Rehearsal samples an\n"
         "# undeclared material over this range, and a scene base bound "
         "with\n# ``plausible_floor`` widens a declared material's box to "
         "cover it.",
         "the declared value or else the scene's. A scene base bound with\n"
         "# ``plausible_floor`` widens a declared material's box to cover "
         "this\n# range."),
        ("# _SIZE_SIGMAS of the size's declared observation noise and by "
         "more than\n# _SIZE_ATOL: a noisy reading of the size it has is no "
         "new size.",
         "# the tolerance of the size's reading and by more than\n"
         "# _SIZE_ATOL: a reading that close to the size it has is no new "
         "size."),
        ("# _SIZE_SIGMAS of its declared observation noise and by more "
         "than\n# _SIZE_ATOL (scene_loader_static_poses).",
         "# the tolerance of its reading and by more than\n# _SIZE_ATOL "
         "(scene_loader_static_poses)."),
        ("override sets it: rehearsal samples it as one of\n    "
         ":meth:`sampled_material_specs`. Bound with",
         "override sets it (see\n    :meth:`sampled_material_specs`). "
         "Bound with"),
        ("        # Values set on undeclared materials (a rehearsal "
         "draw's).", "        # Values set on undeclared materials."),
        ("while each observation reads its pose with noise. Taken as "
         "read,\n        the noise would tilt neighbouring supports",
         "while each observation reads its pose again. Taken as read,\n"
         "        the readings would tilt neighbouring supports"),
        ("        A value for an undeclared material (a rehearsal draw's) "
         "is set\n", "        A value for an undeclared material is set\n"),
        ("as the\n        parameters rehearsal samples: each ranges over",
         "as\n        parameters: each ranges over"),
        ("        A fit that holds these at their starting values replays "
         "the\n        scene as it is, so the lines of its belief through "
         "them measure\n        how the recordings bear on each. A "
         "log-scale",
         "        A fit that holds these at their starting values replays "
         "the\n        scene as it is. A log-scale"),
    ],
}
# What names a withheld element in a path the agent can read. A path joins
# its words with underscores, so each counts wherever its letters stand.
_WITHHELD_IN_PATH = re.compile(
    r"uncertain|nois|belief|posterior|prior|draw|ensemble|sigma|p[-_]?hat|"
    r"interval|smooth|filter|sweep|disabled|ablat|comparison", re.IGNORECASE)


def reference_wording(name: str, source: Any) -> GeneratedReference:
    """The reference file ``name`` with every passage that names a withheld
    element reworded (``_REFERENCE_WORDING``); ``source`` is the file's path or
    its generated text."""
    if isinstance(source, GeneratedReference):
        text, origin = source.text, source.origin
    else:
        path = Path(source)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        text, origin = path.read_text(encoding="utf-8"), str(source)
    for old, new in _REFERENCE_WORDING[name]:
        if text.count(old) != 1:
            raise ValueError(f"reference/{name}: the passage {old!r} occurs "
                             f"{text.count(old)} times, not once; update "
                             "_REFERENCE_WORDING to the source's current text")
        text = text.replace(old, new)
    return GeneratedReference(text, f"{origin}, its comments reworded")


class AgentContinualFromAssetsNoUncertaintyApproach(
        AgentContinualFromAssetsApproach, AgentContinualNoUncertaintyApproach):
    """Agent-built scene models, a harness fit, point estimates only, and
    nothing withheld named to the agent."""

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
        if not CFG.continual_point_values_only:
            raise ValueError(
                "EMPIRIC from assets without explicit uncertainty never "
                "names what it leaves out to its agent; run it with "
                "continual_point_values_only (as the from_assets_point_opus "
                "entry of scripts/configs/empiric/approaches.yaml does)")
        for part in (self.get_name(), str(CFG.experiment_id)):
            if _WITHHELD_IN_PATH.search(part):
                raise ValueError(
                    f"The run folder's path holds {part!r}, and the agent "
                    "reads that path as its working directory; give this "
                    "arm a name, a menu key and a round that name nothing "
                    "the arm leaves out")
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_from_assets_point"

    # -- What the agent receives -------------------------------------------

    def _probe_surface(self) -> ProbeSurface:
        return ProbeSurface(fit=True, uncertainty=False, point_values=True)

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        files = super()._get_sandbox_reference_files()
        for name in _REFERENCE_WORDING:
            if name in files:
                files[name] = reference_wording(name, files[name])
        return files

    def _make_evaluate_trajectory_fn(self) -> Any:
        """The reward model in ``run_python``, with the arguments this arm has
        and documentation that names nothing else."""
        return reward_model(super()._make_evaluate_trajectory_fn())

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
