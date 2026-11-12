"""EMPIRIC from assets without harness fitting.

The agent writes its model as EMPIRIC from assets does: a ``SceneBase``
subclass with mechanisms and declared parameters (``AGENT_PARAM_SPECS``),
under the scene loader when ``scene_manifest_loader`` is on. The harness
estimates nothing. The deployed model runs each declaration's
``init_value``. A rehearsal draw pairs a state draw, and the memory it
implies, with a parameter draw from the declarations alone: a bell curve
centred on each ``init_value`` and cut to its ``[lo, hi]``, as in the
no-fitting arm. The agent may estimate values in its own sandbox code;
they take effect through the declarations.

The agent is never told that a harness fit exists
(``agent_sim_learn_fit_invisible``). Its prompts, tool descriptions,
probe reports and refusals say only what the model runs at, its ``sim``
has no fit entry point, the reference files it reads name no fit, and a
traceback it is shown leaves out the harness's own frames.

The agent's working directory is its sandbox inside the run folder,
``<runs>/<approach name>/<experiment id>/seed<k>/run_<stamp>/``, and the
agent reads that path. So the approach's registered name and its menu key
name no fit either (``agent_continual_from_assets_declared``,
``from_assets_declared_opus``), and the class refuses an experiment id
that does.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from predicators.agent_sdk.prompt_templates import render
from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles
from predicators.agent_sdk.tools.exploration import ProbeSurface
from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoFittingApproach
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach, _box
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.settings import CFG

# The comments and docstrings of the reference sources that name the
# harness fit, each with the wording this arm's copy carries. The code is
# the same in every arm's copy. An entry whose text is gone from its
# source fails the sandbox setup, so a changed source cannot ship its old
# wording unseen.
_REFERENCE_WORDING: Dict[str, List[Tuple[str, str]]] = {
    "base_sim/pybullet_env.py": [
        ("        # same rollout system-ID that fits an env's physical "
         "parameters\n        # fits these too.",
         "        # same path that sets an env's physical parameters sets "
         "these\n        # too."),
        ("        # the harness to fit (world_gap.CalibrationMenu).",
         "        # the harness to set (world_gap.CalibrationMenu)."),
        ("        the fit stack can identify it: the value comes from the "
         "last\n        ``apply_physical_param_overrides`` (the fitter's "
         "candidate) and\n",
         "        the harness can set it: the value comes from the last\n"
         "        ``apply_physical_param_overrides`` and\n"),
        ("        declared parameters here so the rollout system-ID fits "
         "them.",
         "        declared parameters here so the harness can set them."),
    ],
    "base_sim/base_env.py": [
        ("    # plans under the agent's FULL current world model - base sim "
         "plus\n    # fitted residual rules - instead of",
         "    # plans under the agent's FULL current world model - base sim "
         "plus\n    # residual rules - instead of"),
        ('        """Physical parameters this env exposes for system '
         'identification.',
         '        """Physical parameters this env exposes to be set from '
         'outside.'),
        ("``lo``/``hi`` (a reasonable fitting box)",
         "``lo``/``hi`` (a reasonable box)"),
        ("        that should be fitted in log-space",
         "        that vary in log-space"),
        ("        just the class default, because the sysID machinery "
         "restores a\n        param to it when a fit stops declaring that "
         "param.",
         "        just the class default, because the harness restores a\n"
         "        param to it when a model stops declaring that param."),
        ("        an identified value keeps applying during planning "
         "rollouts.",
         "        a value set here keeps applying during planning rollouts."),
    ],
    "base_sim/scene_base.py": [
        ("# The engine's material properties of a body, offered for fitting "
         "as\n# ``<group>_<property>``: property -> (fit box lo, hi, scale, "
         "meaning).",
         "# The engine's material properties of a body, offered as "
         "parameters\n# ``<group>_<property>``: property -> (box lo, hi, "
         "scale, meaning)."),
        ("plausibly has, narrower than the fit boxes\n",
         "plausibly has, narrower than the boxes\n"),
        ("        the fit forks for a replay then holds",
         "        forked for a replay then holds"),
        ("        another level's, and the fit replays every level's "
         "recordings.",
         "        another level's, and a replay runs every level's "
         "recordings."),
        ("like a declared one. Its default, which a\n        fit pins every "
         "parameter it does not estimate to, stands for the",
         "like a declared one. Its default\n        stands for the"),
        ("        A fit that holds these at their starting values replays "
         "the\n        scene as it is, so the lines of its belief through "
         "them measure\n        how the recordings bear on each. A "
         "log-scale",
         "        At their starting values a replay runs the scene as it "
         "is. A\n        log-scale"),
    ],
}
# The longest parameter listing a status line carries.
_STATUS_CHARS = 600
# What names a harness fit in a path the agent can read: "fit" and its
# forms wherever no letter touches them.
_FIT_IN_PATH = re.compile(
    r"(?<![A-Za-z])(?:re|un|mis)?fit(?:s|ted|ting)?(?![A-Za-z])",
    re.IGNORECASE)


def reference_wording(name: str, source: Any) -> GeneratedReference:
    """The reference file ``name`` with every passage that names the harness
    fit reworded (``_REFERENCE_WORDING``); ``source`` is the file's path or its
    generated text."""
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


class AgentContinualFromAssetsNoFittingApproach(
        AgentContinualFromAssetsApproach, AgentContinualNoFittingApproach):
    """Agent-built models at their declared values, with no harness fit and
    none named."""

    _save_suffix = "AgentContinualFromAssetsNoFitting"
    # The declarations stand as written: no plausible range widens a
    # declared material's, and no undeclared material is sampled.
    _belief_over_guesses = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not CFG.agent_sim_learn_fit_invisible:
            raise ValueError(
                "EMPIRIC from assets without harness fitting never names a "
                "harness fit to its agent; run it with "
                "agent_sim_learn_fit_invisible (as the "
                "from_assets_declared_opus entry of scripts/configs/"
                "empiric/approaches.yaml does)")
        for part in (self.get_name(), str(CFG.experiment_id)):
            if _FIT_IN_PATH.search(part):
                raise ValueError(
                    f"The run folder's path holds {part!r}, and the agent "
                    "reads that path as its working directory; give this "
                    "arm a name, a menu key and a round that name no fit")
        if CFG.code_sim_learning_prior_spans_bounds:
            raise ValueError(
                "EMPIRIC from assets without harness fitting draws each "
                "parameter with the no-fitting arm's width, which its "
                "prompt states; code_sim_learning_prior_spans_bounds sets "
                "the prior of a harness fit, so leave it off")
        if not CFG.continual_uncertainty_decisions or \
                int(CFG.belief_joint_draws) <= 0:
            raise ValueError(
                "EMPIRIC from assets without harness fitting rehearses on "
                "joint draws of the state belief and the declared ranges; "
                "run it with continual_uncertainty_decisions and "
                "belief_joint_draws > 0")
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_from_assets_declared"

    # -- What the agent receives -------------------------------------------

    def _probe_surface(self) -> ProbeSurface:
        return ProbeSurface(fit=False,
                            fit_invisible=True,
                            uncertainty=CFG.continual_uncertainty_decisions)

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        files = super()._get_sandbox_reference_files()
        for name in _REFERENCE_WORDING:
            if name in files:
                files[name] = reference_wording(name, files[name])
        return files

    # -- The prompt ---------------------------------------------------------

    def _no_model_section(self) -> str:
        return ("no_model_assets_loaded_declared"
                if CFG.scene_manifest_loader else "no_model_assets_declared")

    def _declared_specs(self) -> List[ParamSpec]:
        """The parameters the loaded model declares, as it runs them."""
        return list(self._physical_param_specs) + list(
            getattr(self, "_probe_rule_specs", None) or [])

    def _fit_status_text(self) -> str:
        """What the model runs at, as one line: each declared value with its
        range."""
        specs = self._declared_specs()
        if not specs:
            return "the model declares no parameters"
        head = "the model runs at its declared values: "
        listed: List[str] = []
        for spec in specs:
            entry = (f"{spec.name}={spec.init_value:.4g} in "
                     f"{_box(spec.lo, spec.hi)}")
            if len(head) + len(", ".join(listed + [entry])) > _STATUS_CHARS:
                listed.append(f"and {len(specs) - len(listed)} more")
                break
            listed.append(entry)
        return head + ", ".join(listed)

    def _loaded_model_status(self, n_eps: int, n_steps: int) -> str:
        return render("play_query",
                      "model_status_declared",
                      simulator_version=self._current_simulator_version
                      or "none",
                      predicates_version=self._current_predicates_version
                      or "none",
                      parameter_status=self._fit_status_text(),
                      n_episodes=str(n_eps),
                      n_steps=str(n_steps))

    def _round_record_extra(self, state: Any) -> str:
        del state
        if not self._last_round_modelled:
            return ""
        return (f"Model updated: simulator.py "
                f"{self._current_simulator_version}, predicates.py "
                f"{self._current_predicates_version or 'none'}; "
                f"{self._fit_status_text()}.")
