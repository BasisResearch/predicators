"""EMPIRIC from assets without explicit uncertainty.

The agent writes its model as EMPIRIC from assets does, the harness fits
the declared parameters to the raw recorded observations, and every
rollout runs one value per parameter from the latest observation. The
agent is never told what the arm leaves out: nothing it reads names the
noise, a belief, a draw or a stress test, and its ``sim`` has no call
built on one.
"""
# pylint: disable=protected-access
import ast
import inspect
import json
import pydoc
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.agent_sdk import play_prompts
from predicators.agent_sdk.belief_probe import BeliefProbe, \
    _motion_failure_note, build_probe_namespace, probe_class_name
from predicators.agent_sdk.local_sandbox import _LOCAL_SANDBOX_SYSTEM_PROMPT
from predicators.agent_sdk.point_probe import SimProbe
from predicators.agent_sdk.probe_results import ProbeGatherResult, \
    ProbeRefineResult, ProbeResult, ProbeTrialsResult
from predicators.agent_sdk.prompt_templates import load_sections
from predicators.agent_sdk.sandbox_setup import GeneratedReference
from predicators.agent_sdk.tools import ToolContext, point_fit_report
from predicators.agent_sdk.tools.assembly import create_mcp_tools
from predicators.agent_sdk.tools.exploration import ProbeSurface, \
    belief_probe_blurb
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths, \
    agent_traceback
from predicators.approaches import create_approach
# pylint: disable-next=line-too-long
from predicators.approaches.agent_continual_from_assets_no_uncertainty_approach import \
    _REFERENCE_WORDING, reference_wording
from predicators.approaches.base_approach import BaseApproach
from predicators.code_sim_learning.evidence import LaplaceEvidence
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.identifiability import Verdict
from predicators.code_sim_learning.orchestrator import run_rollout_sysid
from predicators.code_sim_learning.scene_base import SceneBase
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.settings import CFG
from predicators.structs import Action, Dataset, Object, State, Type
from scripts.cluster_utils import generate_run_configs
from tests.agent_sdk.test_prompt_goldens import _check_golden
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    _dispose_test_physics_clients

CONFIG = "empiric/benchmark.yaml"
# The menu key and the approach's name are in the run folder's path, which
# the agent reads as its working directory, so neither names what the arm
# leaves out.
ARM = "from_assets_point_opus"
NAME = "agent_continual_from_assets_point"
DOMAINS = ("pybullet_boil", "pybullet_domino", "pybullet_fan",
           "pybullet_bridge", "pybullet_balloons")

# What this arm's agent must not read: (pattern, why). Each counts in any
# letter case. A pattern is a word's stem where every form of the word is
# out (noise, noisy, denoise), and a whole word where a longer word only
# holds the letters (prior, draw).
WITHHELD: Tuple[Tuple[str, str], ...] = (
    (r"nois", "the observation noise is not declared: no statement of it "
     "or of its scale, and no denoising"),
    (r"sigma", "the scale of the noise, and a width of a belief"),
    (r"belief", "no state belief and no parameter belief is kept"),
    (r"posterior", "a fit gives one value per parameter, not a "
     "distribution"),
    (r"\bpriors?\b", "the fit's prior is not named to the agent"),
    (r"\bdraw(?:s|n|ing)?\b|\bdrew\b", "no draw of a state or a parameter "
     "is made"),
    (r"ensemble", "no set of states, parameters or models is carried"),
    (r"uncertain", "nothing represents uncertainty, and the word names the "
     "machinery"),
    (r"p[-_ ]?hat", "a rehearsal's success estimate over draws"),
    (r"interval", "a parameter has one value, not an interval"),
    (r"smooth", "no state is smoothed"),
    (r"filter", "no observation is filtered"),
    (r"sweep|swept", "the stress test over a parameter's range "
     "(physics_sweep)"),
    (r"rehearsal (?:samples|draw)|automatic rehearsal", "the harness runs "
     "no rehearsal of its own"),
    (r"disabled", "nothing is described as switched off"),
    (r"this arm", "the agent is not told that it is one arm of several"),
    (r"ablat", "the agent is not told that something was removed"),
    (r"comparison", "the agent is not told that it is compared"),
)
# The uses that stay, because they name nothing the arm leaves out.
# "prior evidence" (the decision workflow) is earlier evidence.
# "the renderer draws" (a note of `sim.run(fresh=True)`) is about an image.
_EVERYDAY = ("prior evidence", "the renderer draws the shared session env")
# `sim.residuals(sweep_params=...)` replays the recordings with one
# parameter at a time at other values. It is a search of how well the
# recordings fit, part of the harness fit that this arm has, and the word
# is its argument's name. These are the words of its description, its
# report and its errors.
_RESIDUALS_SEARCH = (
    "sweep_params",
    "sweep_num_points",
    "additionally sweeps each named",
    "your own targeted sweeps",
    "A sweep is slow",
    "Physical-parameter sweep",
    "param swept ALONE",
    "sweep failed",
    "SWEEP STOPPED EARLY",
    "still unswept",
    "a flat sweep means",
    "to sweep each named parameter alone",
    "Consult a sweep BEFORE",
    "a flat sweep is honest evidence",
    "nothing to sweep",
    "no sweepable entries",
    "sweepable",
)
# The agent's own loops over plans, seeds and start states.
_OWN_LOOPS = ("inside tight sweep loops", "seeds, mods sweeps",
              "Large sweeps are expensive")
_KEPT = _EVERYDAY + _RESIDUALS_SEARCH + _OWN_LOOPS
# The code of the reference sources is the same in every arm's copy, so
# these lines of it stay: an import, a threshold's name, two reads of a
# setting object, and a setting's name.
_REFERENCE_CODE = (
    "from predicators.observation_noise import ObservationNoise",
    "noise = ObservationNoise.from_cfg()",
    "noise.feature_sigma(",
    "_SIZE_SIGMAS",
    "CFG.continual_belief_velocities",
)
# The sections of play_point.md, each against the shared section it stands
# in for: (the shared lines it leaves out, the lines it has instead).
ARM_WORDING: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {
    ("play_system", "observations"): ([
        "A predicate inferred from model memory is a belief, not a measured "
        "fact."
    ], [
        "A predicate inferred from model memory is an inference, not a "
        "measured fact."
    ]),
    ("play_system", "model_gate_fit"): ([
        "Loading a model does not itself enable automatic rehearsal; use "
        "`sim` to check plans before execution."
    ], ["Use `sim` to check plans before execution."]),
    ("play_system", "model_repair_point_estimate"): ([
        "   A parameter sweep cannot detect an omitted mechanism, and no "
        "ensemble of candidates is carried: choose the candidate the "
        "recordings support best, then act.",
        "Record candidate comparisons and rejected hypotheses in the "
        "journal.",
    ], [
        "   Choose the candidate the recordings support best, then act.",
        "Record how the candidates compare, and the hypotheses you "
        "rejected, in the journal.",
    ]),
    ("play_system", "robustness_point_estimate"): ([
        "| Check reliability | Repeated rehearsals at the same state and "
        "dynamics (`trials>=2`) check controller reliability. Parameter "
        "sweeps, belief draws and `sim.belief()` are disabled in this run. |"
    ], [
        "| Check reliability | Repeated rehearsals at the same state and "
        "dynamics (`trials>=2`) check controller reliability. |"
    ]),
    ("play_system", "journal"): ([
        "Keep observed facts, hypotheses and uncertainty, candidate models "
        "and validation results, failed attempts, and the next action with "
        "its rationale."
    ], [
        "Keep observed facts, hypotheses and open questions, candidate "
        "models and validation results, failed attempts, and the next "
        "action with its rationale."
    ]),
    ("play_system", "point_estimate_decisions_raw"): ([
        "## No explicit uncertainty handling",
        "Use the latest observation as the current state and one fitted "
        "value per parameter for every decision.",
        "Parameter fitting over the recorded transitions, inferred "
        "mechanism memory, and model revision remain enabled.",
        "Take observed features as given: do not average, smooth, or "
        "filter observed state, or re-estimate trajectory initial states, "
        "and do not construct parameter intervals, state or parameter "
        "ensembles, uncertainty sweeps, or disagreement-based "
        "experiments, including in your own sandbox code.",
        "Mechanism memory may track action history and hidden processes, "
        "but must not re-estimate observed features.",
        "`sim.belief`, `belief_draws`, `physics_sweep`, and "
        "`sim.suggest_probes` are disabled.",
        "Repeated rehearsals at the same state and dynamics remain "
        "available to check controller reliability.",
    ], [
        "## State and parameter values",
        "Use the latest observation as the current state and the fitted "
        "value of each parameter for every decision.",
        "`sim.reset(current=True)` stages that observation, with the "
        "model memory that `update_model_state` infers from the "
        "observations so far.",
        "Model memory tracks action history and hidden processes; the "
        "observed features come from the observation.",
        "Repeated rehearsals at the same state and dynamics (`trials>=2`) "
        "check controller reliability.",
    ]),
    ("subclass_model", "memory"):
    (["Do not treat it as measured truth or as a particle filter."],
     ["Do not treat it as measured truth."]),
}
# The switches No uncertainty turns off (common.yaml turns most of them on).
NO_UNCERTAINTY_OFF = (
    "continual_uncertainty_decisions",
    "agent_sim_learn_param_uncertainty",
    "agent_explorer_info_seeking",
    "agent_explorer_info_seeking_adaptive",
    "agent_explorer_info_seeking_noise_aware",
    "code_sim_learning_interval_belief",
    "code_sim_learning_carry_posterior",
    "code_sim_learning_rollout_noise_filter",
    "continual_belief_frame",
    "continual_obs_noise_declared",
)
# The switches of EMPIRIC from assets the arm refuses: all but the last read
# a belief, its joint draws or a rehearsal on them, and the reach check
# reads a gap in multiples of the declared noise.
REFUSED_SWITCHES = (
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
    "continual_rehearsal_mean_verdict",
    "continual_rehearsal_game_over",
    "continual_rehearsal_support_note",
    "continual_rehearsal_speed_note",
    "continual_predictive_check",
    "belief_material_lines",
    "belief_keep_undeclared_materials",
    "belief_segment_temperature",
    "continual_fit_bound_note",
    "continual_pooled_level_init",
    "continual_fit_reach_check",
)
# What this arm keeps of EMPIRIC from assets: the switches that act on the
# fit, the scene and the model's own rollouts.
FROM_ASSETS_KEPT = (
    "continual_require_model_on_test",
    "code_sim_learning_prior_spans_bounds",
    "scene_observed_sizes",
    "scene_manifest_loader",
    "continual_restage_velocities",
    "continual_belief_velocities",
)
# A model for the Boil scene the harness loads: two parameters, a scored
# reading, a mechanism and model memory, and no body of its own.
MODEL = '''
class BoilModel(SceneBase):
    AGENT_PARAM_SPECS = [ParamSpec("heat_rate", 0.01, lo=0.0, hi=0.1),
                         ParamSpec("jug_spinning_friction", 0.005, lo=0.0,
                                   hi=0.05)]
    RESIDUAL_FEATURES = {"jug": ["water_volume"]}
    MODEL_STATE_INIT = {"ticks": 0}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["ticks"] += 1

    def _domain_specific_step(self):
        for obj in self._objects:
            if obj.type.name == "jug":
                key = (obj.name, "water_volume")
                self._feature_store[key] += self.agent_param("heat_rate")


RESIDUAL_ENV = BoilModel
'''
# The same model as a new version of the file.
REFITTED_MODEL = MODEL + "\n# The same model, fitted again.\n"

_SLIDER = Object("slider0", Type("slider", ["x"]))


class _GainWorld:
    """A rollout world for the fit: ``x`` advances by ``0.01 * gain`` a
    step."""

    def __init__(self) -> None:
        self._physics_client_id = p.connect(p.DIRECT)
        self._gain = 1.0
        self._x = 0.0

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """One parameter, as the model declares it."""
        return {
            "gain": {
                "default": 1.0,
                "lo": 0.1,
                "hi": 10.0,
                "scale": "log",
                "description": "",
            }
        }

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Set the gain a fit asks for."""
        self._gain = float(params.get("gain", self._gain))

    def _set_state(self, state: State) -> None:
        self._x = float(state.get(_SLIDER, "x"))

    def step(self, action: Action) -> State:
        """Advance ``x`` by one step at the current gain."""
        del action
        self._x += 0.01 * self._gain
        return State({_SLIDER: np.array([self._x], dtype=float)})


def _withheld(text: str, kept: Sequence[str] = _KEPT) -> List[str]:
    """Each passage of ``text`` that names what the arm leaves out, with its
    surroundings; the uses in ``kept`` aside."""
    for phrase in kept:
        text = text.replace(phrase, "")
    return [
        text[max(0,
                 m.start() - 60):m.end() + 60].replace("\n", " ")
        for pattern, _ in WITHHELD
        for m in re.finditer(pattern, text, re.IGNORECASE)
    ]


def _code(source: str) -> str:
    """The code of a module without its docstrings (a parse drops the
    comments)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and \
                ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def _arm_class() -> Any:
    """The class the arm's name is registered under."""
    return utils.get_registered_subclass(BaseApproach, NAME)


def _arm_run(env_name: str = "pybullet_boil", arm: str = ARM) -> Any:
    return next(c
                for c in generate_run_configs(CONFIG, False, approaches=[arm])
                if c.env == env_name)


def _arm_flags(env_name: str = "pybullet_boil",
               arm: str = ARM) -> Dict[str, Any]:
    cfg = _arm_run(env_name, arm)
    # The launcher pins machine-specific output paths; tests keep their own.
    flags = {
        k: v
        for k, v in cfg.flags.items() if k not in ("log", "continual_runs_dir")
    }
    flags.update(approach=cfg.approach,
                 env=cfg.env,
                 experiment_id=cfg.experiment_id,
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any,
          env_name: str = "pybullet_boil",
          arm: str = ARM,
          **overrides: Any) -> Any:
    flags = {**_arm_flags(env_name, arm), **overrides}
    _config(tmp_path, **flags)
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    approach = create_approach(flags["approach"], env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    return env, approach


def test_the_check_finds_what_the_arm_leaves_out() -> None:
    """The patterns count every form the harness's text uses, and leave the
    words that only hold the letters and the uses that stay."""
    for text in ("observation noise", "a noisy reading", "denoised",
                 "the declared sigma", "`sim.belief()`", "belief_draws=K",
                 "posterior_std / prior_std", "the fit's prior", "16 draws",
                 "a drawn pose", "drew wide", "ensemble disagreement",
                 "its uncertainty", "uncertain atoms", "P-hat = 0.50", "p_hat",
                 "95% interval", "smoothed features", "particle filter",
                 "physics_sweep=True", "parameter sweeps", "swept end to end",
                 "rehearsal samples it", "automatic rehearsal", "are disabled",
                 "this arm acts", "the ablation", "Point-estimate comparison"):
        assert _withheld(text), text
    for text in ("prior evidence", "a drawer", "priority", "sim.residuals("
                 "sweep_params='all')", "withdrawn"):
        assert not _withheld(text), text
    # A kept use excuses its own words alone.
    assert _withheld("prior evidence and a prior")
    assert _withheld("sweep_params and physics_sweep")


def test_config_is_from_assets_at_point_estimates() -> None:
    """Five settings, five seeds, parked, under its own class: No uncertainty's
    switches, the fit and scene switches of EMPIRIC from assets, none of its
    belief or rehearsal switches, and a name and a key that name none."""
    cls = _arm_class()
    assert cls.__name__ == "AgentContinualFromAssetsNoUncertaintyApproach"
    assert set(cls.refused_switches) == set(REFUSED_SWITCHES)
    assert all(run.approach != NAME
               for run in generate_run_configs(CONFIG, False))
    runs = list(generate_run_configs(CONFIG, False, approaches=[ARM]))
    assert len(runs) == 25
    assert {run.env for run in runs} == set(DOMAINS)
    for run in runs:
        assert run.approach == NAME
        assert not _withheld(f"{run.approach} {run.experiment_id}", ())
        assert run.flags["continual_point_values_only"] is True
        assert run.flags["belief_joint_draws"] == 0
        for name in NO_UNCERTAINTY_OFF:
            assert run.flags[name] is False, name
        for name in FROM_ASSETS_KEPT:
            assert run.flags[name] is True, name
        for name in REFUSED_SWITCHES + ("continual_belief_window", ):
            assert name not in run.flags, name


@pytest.mark.parametrize(
    "flag, value",
    [(name, True) for name in NO_UNCERTAINTY_OFF + REFUSED_SWITCHES] + [
        ("belief_joint_draws", 16),
        ("code_sim_learning_prior_spans_bounds", False),
        ("continual_point_values_only", False),
    ])
def test_arm_rejects_an_uncertainty_switch(flag: str, value: Any) -> None:
    """A config that turns on a switch of the belief, the rehearsal or the
    noise declaration, that drops the fit's range prior, or that lets the text
    name what the arm leaves out, fails before a run starts."""
    utils.reset_config({**_arm_flags(), flag: value})
    # Contract validation must fail before constructing an environment.
    with pytest.raises(ValueError, match=flag):
        _arm_class()()


@pytest.mark.parametrize("round_key", [
    "boil-from_assets_no_uncertainty_opus_r1", "boil-point_opus_noisefree",
    "boil-point_opus_no_belief_r2", "boil-point_ablation"
])
def test_arm_rejects_a_path_that_names_what_it_leaves_out(
        round_key: str) -> None:
    """The agent reads the run folder's path, so a round may name nothing the
    arm leaves out."""
    utils.reset_config({**_arm_flags(), "experiment_id": round_key})
    with pytest.raises(ValueError, match="run folder's path"):
        _arm_class()()


def test_fit_gives_point_estimates_and_no_belief() -> None:
    """Under the arm's flags a fit of raw noisy readings searches the declared
    range from the agent's guess, applies its point estimate and builds no
    belief; its report gives the residual error and the fitted value in plain
    words."""
    utils.reset_config(_arm_flags())
    rng = np.random.default_rng(0)
    # Raw noisy readings of a slider that moves at twice the agent's guess.
    states = [
        State({
            _SLIDER:
            np.array([0.02 * step + rng.normal(0.0, 0.002)], dtype=float)
        }) for step in range(21)
    ]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(20)]
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    outcome = run_rollout_sysid(_GainWorld(), [(states, actions)], [spec],
                                {"slider": ["x"]},
                                anchors={"gain": 1.0})
    assert outcome.num_survivors == 1
    assert outcome.fitted["gain"] == pytest.approx(2.0, rel=0.1)
    assert outcome.applied == outcome.fitted
    # No belief: no draws, no measured misfit and no basins to mix.
    assert outcome.belief is None
    assert outcome.fit_result.misfit is None
    assert len(outcome.fit_result.basins or []) <= 1
    report = point_fit_report.fit_report(
        version_tag="cycle_001_vers_001",
        exploratory=False,
        traj_idxs=None,
        scope="declared: slider: x",
        num_segments=1,
        num_used=outcome.num_survivors,
        sse_declared=outcome.pre_sse,
        sse_declared_used=outcome.pre_sse,
        sse_fitted=outcome.post_sse,
        declared={"gain": 1.0},
        fitted=outcome.inference.point_estimate,
        runs=outcome.inference.selected_parameters,
        constraints=outcome.inference.parameter_diagnostics,
        traj_rms=outcome.traj_rms,
        limit=1.0,
        unscored={},
        evidence=point_fit_report.evidence_lines(outcome.evidence, None),
        departures=[])
    assert not _withheld(report), _withheld(report)
    assert "[cycle_001_vers_001] FIT (canonical) of 1 declared " \
        "parameter(s) on 1 recorded motion segments" in report
    assert "at the declared values:" in report
    assert "at the fitted values:" in report
    assert re.search(
        r"gain +1\.0000 -> [12]\.\d{4} .* constrained by the "
        "recordings", report), report
    assert "the model runs it at" not in report


def test_fit_report_says_what_the_recordings_constrain() -> None:
    """The arm's fit report gives the residual error, each fitted value, the
    parameters the recordings do not constrain and the segments the fit left
    out, in plain words, whatever the fit's verdicts are."""
    utils.reset_config(_arm_flags())
    verdicts = list(Verdict)
    names = [f"p{i}" for i in range(len(verdicts))]
    declared = {name: 1.0 for name in names}
    fitted = {name: 1.5 for name in names}
    runs = {
        name: 1.5 if verdict.applies_fitted else 1.0
        for name, verdict in zip(names, verdicts)
    }
    constraints: Dict[str, Dict[str, Any]] = {
        name: {
            "verdict": verdict,
            "posterior_std": 0.1,
            "prior_std": 1.0,
            "contraction": 0.1,
            "note": "posterior ~= prior; MAP arbitrary",
            "flat_interval": (0.5, 2.0),
            "bound_edge": "hi",
        }
        for name, verdict in zip(names, verdicts)
    }
    first = LaplaceEvidence(-40.0, 120, 3, -30.0, -4.0, -6.0)
    second = LaplaceEvidence(-35.0, 120, len(names), -28.0, -3.0, -4.0)
    report = point_fit_report.fit_report(
        version_tag="cycle_002_vers_003",
        exploratory=False,
        traj_idxs=None,
        scope="declared: jug: water_volume",
        num_segments=4,
        num_used=1,
        sse_declared=3.0,
        sse_declared_used=2.0,
        sse_fitted=0.5,
        declared=declared,
        fitted=fitted,
        runs=runs,
        constraints=constraints,
        traj_rms=[0.01, 0.2, 0.3, 0.9],
        limit=0.05,
        unscored={"jug": ["x", "y"]},
        evidence=point_fit_report.evidence_lines(
            second, {"cycle_002_vers_002": first}),
        departures=point_fit_report.departure_lines([
            "Where your model leaves each dropped segment, replayed at "
            "the segment's own best values from that sweep: the feature."
        ]))
    assert not _withheld(report), _withheld(report)
    for line in (
            "Not scored by this fit, though they change in the recordings: "
            "jug: x, y.",
            "Segments left out: 3 of 4 recorded motion segments",
            "3 recorded segment(s) stayed above the limit 0.05",
            "best values from that search",
            "  at the declared values: 3.000000 (all 4 segments), 2.000000 "
            "on the 1 segments the fit used",
            "  at the fitted values:   0.500000 on the 1 segments the fit "
            "used  (75.0% lower than at the declared values, on the same "
            "segments)",
            "Model evidence (log scale, higher is better): -35.0 on 120 "
            "residuals",
            "Against cycle_002_vers_002: -40.0, change +5.0 (this version "
            "wins on evidence).",
            "NOT constrained by the recordings: other values explain them "
            "as well, so the fitted value is arbitrary; the model runs it "
            "at 1",
            "fitted to the hi end of its declared range",
            "(only 1 recorded segment(s) support it)",
            "the recordings are explained equally well from 0.5 to 2",
            "Applied: sim.run and sim.refine now roll the model out",
    ):
        assert line in report, (line, report)
    loose = next(line for line in report.splitlines()
                 if line.startswith("Not constrained by the recordings:"))
    assert [
        name for name, verdict in zip(names, verdicts)
        if verdict in (Verdict.NOT_IDENTIFIED, Verdict.INSENSITIVE,
                       Verdict.ANCHORED)
    ] == re.findall(r"p\d+", loose)
    # A subset fit applies nothing, and a fit that moved nothing says so.
    subset = point_fit_report.fit_report(
        version_tag="v",
        exploratory=True,
        traj_idxs=[2, 0],
        scope="declared: jug: water_volume",
        num_segments=2,
        num_used=2,
        sse_declared=1.0,
        sse_declared_used=1.0,
        sse_fitted=1.0,
        declared={"rate": 0.5},
        fitted={"rate": 0.5},
        runs={"rate": 0.5},
        constraints={},
        traj_rms=[0.01, 0.02],
        limit=0.05,
        unscored={},
        evidence=point_fit_report.evidence_lines(None, None),
        departures=[])
    assert not _withheld(subset), _withheld(subset)
    assert "FIT (EXPLORATORY, trajectories [0, 2] only)" in subset
    assert "NOTE: the fit moved no parameter." in subset
    assert "Model evidence: not available for this fit." in subset
    assert "EXPLORATORY subset fit: nothing was applied or recorded" in subset
    assert "Segments left out" not in subset
    # A fit that could explain no segment.
    refused = point_fit_report.no_fit_report(
        version_tag="v",
        scope="declared: jug: water_volume",
        traj_rms=[0.4, 0.9],
        limit=0.05,
        exploratory=False,
        departures=[])
    assert not _withheld(refused), _withheld(refused)
    assert "[v] NO FIT RAN: none of the 2 recorded motion segments" in refused
    assert "the model is UNVALIDATED at its declared values" in refused
    utils.reset_config({})


def test_arm_wording_differs_from_the_shared_sections_where_listed() -> None:
    """Each section of play_point.md is the shared section it stands in for,
    less the listed lines and plus the listed ones: a shared section that
    changes elsewhere shows up here."""
    own = load_sections("play_point")
    assert set(own) == {f"{t}__{s}" for t, s in ARM_WORDING}
    for (template, section), (removed, added) in ARM_WORDING.items():
        shared = load_sections(template)[section].splitlines()
        mine = own[f"{template}__{section}"].splitlines()
        assert [line for line in shared if line not in mine] == removed, \
            (template, section)
        assert [line for line in mine if line not in shared] == added, \
            (template, section)
        assert _withheld("\n".join(removed)), (template, section)
        assert not _withheld("\n".join(mine)), (template, section)


def test_other_arms_render_the_shared_sections() -> None:
    """The arm's wording file serves this arm alone: without its switch the
    builder renders the shared sections, and the probe and its description are
    the ones the other arms had."""
    utils.reset_config({})
    assert play_prompts.render("play_system", "journal") == \
        play_prompts._render_shared("play_system", "journal")
    assert "hypotheses and uncertainty" in play_prompts.render(
        "play_system", "journal")
    assert probe_class_name() == "BeliefProbe"
    namespace = build_probe_namespace(ToolContext())
    assert namespace["sim"].__class__ is BeliefProbe
    assert namespace["BeliefProbe"]().__class__ is BeliefProbe
    described = belief_probe_blurb(synthesis_probe=True,
                                   surface=ProbeSurface(uncertainty=False))
    assert "`sim` (a BeliefProbe over the CANDIDATE simulator" in described
    assert "`BeliefProbe()` (extra independent instances). BeliefProbe API" \
        in described
    assert "regions are DISABLED in this configuration" in described
    utils.reset_config({"continual_point_values_only": True})
    assert "hypotheses and open questions" in play_prompts.render(
        "play_system", "journal")
    utils.reset_config({})


def test_prompt_offers_the_fit_and_names_nothing_the_arm_leaves_out(
        tmp_path: Any) -> None:
    """The system prompt is the from-assets model contract with the harness fit
    and raw observations.

    It says what the agent has, and names no noise, belief, draw, stress
    test or harness rehearsal, as switched off or otherwise.
    """
    _, approach = _make(tmp_path)
    assert isinstance(approach, _arm_class())
    assert isinstance(approach._base_env, SceneBase)
    surface = approach._probe_surface()
    assert surface == ProbeSurface(fit=True,
                                   uncertainty=False,
                                   point_values=True)
    assert approach.parameter_belief() is None
    prompt = approach._get_agent_system_prompt()
    _check_golden("continual_system_from_assets_no_uncertainty", prompt)
    assert "__" not in prompt.replace("__init__", "")
    texts = {
        "system prompt":
        prompt + _LOCAL_SANDBOX_SYSTEM_PROMPT,
        "probe description":
        belief_probe_blurb(synthesis_probe=True, surface=surface),
        "solve probe description":
        belief_probe_blurb(synthesis_probe=False, surface=surface),
        "no-model status":
        approach._model_status(None),
        "probe class":
        pydoc.plain(pydoc.render_doc(SimProbe)),
        "reward model":
        pydoc.plain(pydoc.render_doc(approach._make_evaluate_trajectory_fn())),
    }
    for name, text in texts.items():
        assert not _withheld(text), (name, _withheld(text))
    # The from-assets contract, for a scene the harness loads.
    assert "## EMPIRIC from assets" in prompt
    assert "SceneBase" in prompt and "scene_manifest.json" in prompt
    assert "Do not build bodies and do not override `initialize_pybullet`" \
        in prompt
    assert "force, torque or constraint it applies" in prompt
    assert "Give each parameter to the part that owns it" in prompt
    assert "BaseSimulator" not in prompt
    assert "visible base physics" not in prompt
    assert "model or not" not in prompt
    # The harness fit is offered, and a test-level step waits for it.
    assert "sim.fit()" in prompt
    assert "the harness fits them" in prompt
    assert "### Test levels require a fitted model" in prompt
    assert "the fit and every rollout keep the value your scene gives it" \
        in prompt
    assert "A declared material's range covers at least its plausible " \
        "range" in prompt
    # What the agent works with, in positive words.
    assert "## State and parameter values" in prompt
    assert "latest observation as the current state" in prompt
    assert "the fitted value of each parameter" in prompt
    assert "starts every body at rest" in prompt
    assert "Use `sim` to check plans before execution." in prompt
    assert "(`trials>=2`) check controller reliability. |" in prompt
    # Nothing is described as withheld, switched off or forbidden.
    for absent in ("No explicit uncertainty", "do not average",
                   "must not re-estimate", "do not construct",
                   "remain enabled", "joint rehearsal",
                   "### Test-level steps wait",
                   "A rehearsed request is refused",
                   "Start a new plan only from rest",
                   "### A reset training level asks for a rehearsal",
                   "### Claim checks",
                   "### Each rehearsed step is checked against its rehearsal",
                   "### Every skill request is rehearsed first",
                   "### State estimates, timing, and execution discrepancies",
                   "Use margins", "margin across models"):
        assert absent not in prompt, absent
    # The probe's description names the class the agent holds.
    described = texts["probe description"]
    assert "`sim` (a SimProbe over the CANDIDATE simulator" in described
    assert "`SimProbe()` (extra independent instances). SimProbe API:" \
        in described
    assert "a `~ [w]` region after a step's params is ignored" in described
    for call in ("`sim.fit(traj_idxs=None, fixed=None)`", "`sim.validate(",
                 "`sim.residuals(", "`sim.run(plan_text, render=True, "
                 "trials=1, solved=False, contacts=False)`", "`sim.refine("):
        assert call in described, call
    for call in ("sim.belief", "suggest_probes", "physics_sweep",
                 "belief_draws", "draws="):
        assert call not in described, call
    # The reward model in run_python takes a state sequence, its labels and
    # a level, and nothing else.
    score = approach._make_evaluate_trajectory_fn()
    assert list(inspect.signature(score).parameters) == [
        "states", "actions", "task_idx"
    ]
    assert "your simulator at its current fit" in " ".join(
        texts["reward model"].split())
    with pytest.raises(TypeError,
                       match="unexpected keyword argument 'physics_sweep'"):
        score([], physics_sweep=True)  # pylint: disable=unexpected-keyword-arg


@pytest.mark.parametrize("env_name", DOMAINS)
def test_what_the_agent_is_sent_names_nothing_the_arm_leaves_out(
        tmp_path: Any, monkeypatch: Any, env_name: str) -> None:
    """On each of the five domains, two scripted rounds: the system prompt with
    the sandbox's suffix, CLAUDE.md, every tool's description and arguments,
    the round queries, the sandbox's path and the reference files name nothing
    the arm leaves out."""
    env, approach = _make(tmp_path, env_name)
    queries: List[str] = []
    read: Dict[str, str] = {}

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        queries.append(message)
        read[f"query {len(queries)}"] = message
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        if len(queries) > 1:
            _call(approach, "give_up", note="read")
            return _result()
        for tool in ctx.extra_mcp_tools:
            read[f"tool {tool.name}"] = (str(tool.description) + "\n" +
                                         json.dumps(tool.input_schema))
        assert "run_python" in {tool.name for tool in ctx.extra_mcp_tools}
        # Where the sandbox is: the agent reads its working directory.
        read["sandbox path"] = str(sandbox.relative_to(tmp_path))
        assert NAME in read["sandbox path"]
        assert _arm_run(env_name).experiment_id in read["sandbox path"]
        read["sent system prompt"] = next(
            Path(approach._get_log_dir()).glob(
                "full_system_prompt_*.md")).read_text()
        read["CLAUDE.md"] = (sandbox / "CLAUDE.md").read_text()
        for path in sorted((sandbox / "reference").rglob("*")):
            if path.suffix in (".py", ".json", ".urdf", ".md", ".txt"):
                leaks = _withheld(path.read_text(), _KEPT + _REFERENCE_CODE)
                assert not leaks, (path.name, leaks)
        hold = [0.0] * env.action_space.shape[0]
        read["env_step"] = _call(approach, "env_step", action=hold)
        read["env_observe"] = _call(approach, "env_observe")
        read["skills_list"] = _call(approach, "skills_list")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    ContinualRun(env, approach, create_level_player(env, approach)).run()
    assert len(queries) >= 2
    for tool in create_mcp_tools(approach._tool_context,
                                 tool_names=approach._continual_tool_names()):
        read[f"static tool {tool.name}"] = (str(tool.description) + "\n" +
                                            json.dumps(tool.input_schema))
    assert "## State and parameter values" in read["sent system prompt"]
    assert "call `sim.fit()` before you act on a test level" in read["query 1"]
    for name, text in read.items():
        assert not _withheld(text), (name, _withheld(text))


def test_probe_has_no_call_built_on_a_belief() -> None:
    """The arm's ``sim`` is a probe class of its own: the calls built on a
    belief are absent from the object, its class and its documentation, and
    their arguments are unknown keywords; the other arms' probe keeps them."""
    noisy = {
        "continual_obs_noise_position": 0.01,
        "continual_obs_noise_orientation": 0.02
    }
    utils.reset_config({**noisy, "continual_point_values_only": True})
    ctx = ToolContext()
    namespace = build_probe_namespace(ctx)
    sim: Any = namespace["sim"]
    assert sim.__class__ is SimProbe and SimProbe.__name__ == "SimProbe"
    assert probe_class_name() == "SimProbe"
    assert "BeliefProbe" not in namespace
    assert namespace["SimProbe"]().__class__ is SimProbe
    assert BeliefProbe not in SimProbe.__mro__
    for name in ("belief", "suggest_probes", "score", "run_points"):
        assert not hasattr(sim, name) and name not in dir(sim), name
        assert not hasattr(SimProbe, name) and name not in dir(SimProbe), name
        with pytest.raises(AttributeError,
                           match=f"'SimProbe' object has no attribute "
                           f"'{name}'"):
            getattr(sim, name)
    assert not _withheld(" ".join(dir(sim)))
    plan = "Wait(robot:robot)[2]"
    for name, value in (("draws", 4), ("physics_sweep", True), ("belief_draws",
                                                                4)):
        with pytest.raises(TypeError,
                           match=f"unexpected keyword argument '{name}'"):
            sim.run(plan, **{name: value})
        with pytest.raises(TypeError,
                           match=f"unexpected keyword argument '{name}'"):
            sim.run_async(plan, **{name: value})
    for kept in ("reset", "task", "fit", "validate", "residuals", "predicates",
                 "check_restore", "run", "run_async", "gather", "refine",
                 "state", "atoms", "render", "snapshot", "restore"):
        assert callable(getattr(sim, kept)) and kept in dir(sim), kept
    documented = pydoc.plain(pydoc.render_doc(SimProbe))
    assert not _withheld(documented), _withheld(documented)
    # The results a call returns: the agent can read their class, its
    # module's name and its documentation.
    for result in (ProbeResult, ProbeTrialsResult, ProbeRefineResult,
                   ProbeGatherResult, type(BeliefProbe(ctx).gather([]))):
        assert result.__module__ == "predicators.agent_sdk.probe_results"
        documented = pydoc.plain(pydoc.render_doc(result))
        assert not _withheld(documented), (result, _withheld(documented))
    # A skill that fails in a rollout: the note gives the real record alone.
    note = _motion_failure_note("A skill's controller failed.", ["PickJug"],
                                {"PickJug": (3, 1)})
    assert note == ("A skill's controller failed. This run's real calls of "
                    "it succeeded in: PickJug 3 of 4.")
    assert _motion_failure_note("A skill's controller failed.") is None
    # A traceback shown to this arm's agent keeps the agent's frames and
    # the error, and leaves out the harness's own.
    shown = full = ""
    try:
        sim.restore(99)
    except ValueError:
        shown = agent_traceback()
        full = traceback.format_exc()
    assert "belief_probe.py" in full and "in restore" in full
    assert "belief_probe.py" not in shown and "point_probe.py" not in shown
    assert "test_agent_continual_from_assets_no_uncertainty_approach.py" \
        in shown
    assert shown.rstrip().endswith(full.rstrip().splitlines()[-1])
    # The other arms' probe, notes and tracebacks are the ones they had.
    utils.reset_config(noisy)
    try:
        sim.restore(99)
    except ValueError:
        assert agent_traceback() == _scrub_host_paths(traceback.format_exc())
    other: Any = build_probe_namespace(ToolContext())["sim"]
    assert other.__class__ is BeliefProbe
    for name in ("belief", "suggest_probes", "score", "run_points"):
        assert hasattr(other, name) and name in dir(other), name
    assert "physics_sweep" in pydoc.plain(pydoc.render_doc(BeliefProbe))
    assert "observation noise (0.01 m, 0.02 rad)" in str(
        _motion_failure_note("A skill's controller failed.", ["PickJug"],
                             {"PickJug": (3, 1)}))
    utils.reset_config({})


def test_reference_copies_name_nothing_the_arm_leaves_out(
        tmp_path: Any) -> None:
    """The engine sources this arm's sandbox gets carry the same code as
    EMPIRIC from assets' and their comments name nothing the arm leaves out; a
    passage the source no longer has is refused."""
    _, approach = _make(tmp_path)
    files = approach._get_sandbox_reference_files()
    _, full_approach = _make(tmp_path / "full", arm="from_assets_opus")
    full = full_approach._get_sandbox_reference_files()
    assert set(files) == set(full)
    assert set(_REFERENCE_WORDING) == {
        "base_sim/pybullet_env.py", "base_sim/base_env.py",
        "base_sim/scene_base.py"
    }
    for name in _REFERENCE_WORDING:
        copy = files[name]
        assert isinstance(copy, GeneratedReference), name
        source = full[name]
        original = (source.text if isinstance(source, GeneratedReference) else
                    Path(source).read_text(encoding="utf-8"))
        assert _withheld(original, _KEPT + _REFERENCE_CODE), name
        assert not _withheld(copy.text, _KEPT + _REFERENCE_CODE), \
            (name, _withheld(copy.text, _KEPT + _REFERENCE_CODE))
        # Only comments and docstrings differ: the code is the same.
        assert copy.text != original
        assert _code(copy.text) == _code(original), name
    for name, source in files.items():
        if isinstance(source, GeneratedReference) and \
                name.endswith((".json", ".py", ".urdf")):
            assert not _withheld(source.text, _KEPT + _REFERENCE_CODE), name
    with pytest.raises(ValueError, match="occurs 0 times"):
        reference_wording("base_sim/base_env.py",
                          GeneratedReference("pass\n", "a changed source"))


@pytest.mark.slow
def test_agent_fits_and_acts_on_a_test_level_with_no_rehearsal_gate(
        tmp_path: Any, monkeypatch: Any) -> None:
    """On a test level a skill waits for a model, then for its fit, and for
    nothing else: the model loads in the scene the harness built, sim.fit runs,
    sim rolls a plan out at the point estimate, and the skill runs.

    The calls built on a belief are absent, and nothing the agent reads
    back names what the arm leaves out.
    """
    env, approach = _make(tmp_path, continual_levels="test_only")
    seen: List[str] = []
    read: Dict[str, str] = {}
    pick = "PickJug(robot:robot, jug0:jug)[0.05]"
    wait = "Wait(robot:robot)[2]"

    def python(code: str) -> str:
        out = _call(approach, "run_python", code=code)
        read[f"run_python {len(read)}: {code[:40]}"] = out
        return out

    def skill(plan: str) -> str:
        out = _call(approach, "skills_invoke", skill=plan)
        read[f"skills_invoke {len(read)}"] = out
        return out

    def first_round(message: str) -> None:
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        model_file = sandbox / "simulator.py"
        assert "sim` has no world until" in message
        assert "call `sim.fit()` before you act on a test level" in message
        # The model gate is installed; nothing that reads a rehearsal is.
        assert ctx.skill_gate is not None
        for name in ("rehearsal_gate", "practise_gate", "claim_check",
                     "skill_preflight", "material_screen",
                     "joint_draws_provider", "scored_features_provider",
                     "reach_note_provider", "bound_note_provider"):
            assert getattr(ctx, name) is None, name
        tools = {t.name: t for t in ctx.extra_mcp_tools}
        assert "claims_record" not in tools
        for tool in ctx.extra_mcp_tools:
            read[f"tool {tool.name}"] = (str(tool.description) + "\n" +
                                         json.dumps(tool.input_schema))
        assert "sim.fit" in tools["run_python"].description
        folder = str(sandbox.relative_to(tmp_path))
        assert NAME in folder and not _withheld(folder, ()), folder
        # The calls built on a belief are absent from `sim`.
        absent = python(
            "print([hasattr(sim, n) or n in dir(sim) for n in "
            "('belief', 'suggest_probes')], type(sim).__name__, "
            "type(SimProbe()).__name__, 'BeliefProbe' in globals())")
        assert "[False, False] SimProbe SimProbe False" in absent, absent
        for call, name in (("sim.belief()", "belief"),
                           (f"sim.suggest_probes('{wait}')",
                            "suggest_probes")):
            # The error repeats the name the caller wrote.
            refused = _call(approach, "run_python", code=call)
            assert "AttributeError" in refused, refused
            assert f"'SimProbe' object has no attribute '{name}'" in refused
        documented = python("help(sim)")
        assert "class SimProbe" in documented, documented
        assert "run(self, plan_text" in documented, documented
        python("print(sorted(n for n in dir(sim) if not n.startswith('__')))")
        # Raw control is not gated: one step that holds the arm where it
        # is gives the fit a recording.
        session = approach._play_session
        hold = [float(q) for q in session.observe().frame.joint_positions]
        assert len(hold) == env.action_space.shape[0]
        stepped = _call(approach, "env_step", action=hold)
        read["env_step"] = stepped
        assert "step applied" in stepped
        # An observation carries poses and readings, and no body speed.
        frame = ctx.current_observation_provider()
        assert "body_velocities" not in (frame.simulator_state or {})
        refused = skill(pick)
        assert refused.startswith("ERROR"), refused
        assert "Test level: a skill request waits for your model." in refused
        assert "Write `./simulator.py`" in refused
        assert "Nothing was charged" in refused
        # Before a model the probe has no world.
        python("sim.reset(current=True)\nsim.run('" + wait + "')")
        # The model adds parameters and a mechanism to the loaded scene.
        model_file.write_text(MODEL, encoding="utf-8")
        loaded = python("sim.reset(current=True)\nprint(sorted(sim.state()))")
        assert not loaded.startswith("ERROR"), loaded
        assert "jug0:jug" in loaded
        assert type(approach._base_env).__name__ == "BoilModel"
        assert approach._tool_context.env is approach._base_env
        # Unfitted, the model's values are the agent's guesses: the gate
        # asks for a fit, and for no rehearsal.
        refused = skill(pick)
        assert refused.startswith("ERROR"), refused
        assert "has not been fitted" in refused and "sim.fit()" in refused
        assert "Nothing was charged" in refused
        unfitted = python(f"print(sim.reset(current=True).run('{wait}', "
                          "render=False))")
        assert "PARAMS UNFITTED" in unfitted, unfitted
        # One held step shows no motion the model explains, so this fit
        # keeps the declared values. It is still the canonical fit of the
        # file, which is what the gate asks for.
        fitted = python("print(sim.fit())")
        assert not fitted.startswith("ERROR"), fitted
        assert "] NO FIT RAN: none of the 1 recorded motion segments" in \
            fitted, fitted
        assert approach._probe_fit_state().get("fit_result") is not None, \
            fitted
        assert approach._probe_fit_state().get("digest") == \
            approach._current_simulator_digest()
        # The fit leaves point estimates and no belief.
        assert approach._probe_fit_state().get("belief") is None
        assert approach.parameter_belief() is None
        assert not approach._sampled_material_specs()
        assert not approach.belief_free_specs()
        assert approach._model_readiness(str(model_file), []) is None
        python("print(sim.fit(traj_idxs=[0]))")
        # With a wider limit the fit keeps that recording, and its report
        # gives the residual error and each fitted value. The file is a
        # new version, so no earlier fit of it stands in.
        utils.update_config({
            "code_sim_learning_rollout_trim_rms_factor":
            100.0 * CFG.code_sim_learning_rollout_trim_rms_factor
        })
        model_file.write_text(REFITTED_MODEL, encoding="utf-8")
        refit = python("print(sim.fit())")
        for line in ("] FIT (canonical) of 2 declared parameter(s) on 1 "
                     "recorded motion segments. Scored features: declared: "
                     "jug: water_volume.", "Residual error (SSE:",
                     "  at the declared values: ", "  at the fitted values: ",
                     "Fitted values (declared -> fitted, and what the "
                     "recordings say):", "  heat_rate ",
                     "  jug_spinning_friction ",
                     "Applied: sim.run and sim.refine now roll the model out"):
            assert line in refit, (line, refit)
        subset = python("print(sim.fit(traj_idxs=[0]))")
        assert "] FIT (EXPLORATORY, trajectories [0] only) of 2 declared " \
            "parameter(s)" in subset, subset
        assert "EXPLORATORY subset fit: nothing was applied or recorded" in \
            subset, subset
        assert approach._probe_fit_state().get("belief") is None
        assert approach.parameter_belief() is None
        assert approach._model_readiness(str(model_file), []) is None
        # sim rolls a plan out once, at the fitted values, from the latest
        # observation; repeated trials check the controller.
        run = python(f"r = sim.reset(current=True).run('{wait}', "
                     "render=False)\n"
                     "print(type(r).__name__, len(r.steps), "
                     "r.steps[0]['failure'])\nprint(r)\n"
                     "print(type(r), type(sim), type(sim.state()))")
        assert "ProbeResult 1 None" in run, run
        trials = python(f"r = sim.reset(current=True).run('{wait}', "
                        "trials=2)\nprint(type(r).__name__)\nprint(r)")
        assert "ProbeTrialsResult" in trials, trials
        # A rollout takes the arguments of one run at one set of values.
        for argument in ("draws=4", "draws=0", "physics_sweep=True",
                         "belief_draws=4"):
            name = argument.split("=", maxsplit=1)[0]
            out = _call(approach,
                        "run_python",
                        code=f"sim.run('{wait}', {argument})")
            assert "TypeError" in out, out
            assert f"unexpected keyword argument '{name}'" in out, out
        assert not ctx.live_rehearsals
        # The other calls the description offers.
        for code in (
                "print(sim.task())",
                "print(sim.validate())",
                "print(sim.validate(traj_idxs=[0], "
                "params={'heat_rate': 0.02}))",
                "print(sim.residuals())",
                "print(sim.residuals(rollout=True))",
                "print(sim.residuals(rollout=True, "
                "sweep_params=['heat_rate'], sweep_num_points=2))",
                "print(sim.residuals(rollout=True, "
                "phys_params={'heat_rate': 0.02}))",
                "print(sim.predicates())",
                "print(sim.reset(current=True).check_restore())",
                f"print(sim.reset(current=True).run('{wait}', "
                "contacts=True, render=False))",
                f"print(sim.reset(current=True).run('{wait}', seed=3, "
                "fresh=True))",
                f"print(sim.reset(current=True).run('{wait}', trials=2, "
                "fresh=True))",
                f"print(sim.reset().run('{wait}', trials=2, solved=True))",
                f"print(sim.reset(current=True).refine('{wait} ~ [1]', "
                "timeout=20))",
                f"h = sim.reset(current=True).run_async('{wait}')\n"
                "print(sim.gather([h]))\nprint(h.result)",
                "sid = sim.snapshot()\nsim.restore(sid)\nsim.drop(sid)\n"
                "print(sim.atoms(), sim.state('jug0'))",
                "print(describe_trajectory(0))",
                "print(is_goal_state(trajectories[0].states[-1]))",
        ):
            python(code)
        # An error inside the harness shows the agent's line and the
        # error, with no frame of the harness.
        failed = python("sim.restore(99)")
        assert "Unknown snapshot id 99" in failed, failed
        assert "belief_probe.py" not in failed and "predicators/" not in \
            failed, failed
        # Loaded and fitted, the model opens the gate: the skill runs with
        # no rehearsal of it on record.
        before = session.observe().ledger.run_steps
        out = skill(pick)
        assert "Nothing was charged" not in out, out
        assert "rehears" not in out.lower(), out
        assert session.observe().ledger.run_steps > before
        # An edit closes the gate until the next fit; the fitted file
        # opens it again.
        with open(model_file, "a", encoding="utf-8") as f:
            f.write("\n# edited\n")
        refused = skill(pick)
        assert "has not been fitted" in refused, refused
        model_file.write_text(REFITTED_MODEL, encoding="utf-8")
        assert approach._model_readiness(str(model_file), []) is None
        read["env_observe"] = _call(approach, "env_observe")

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        read[f"query {len(seen)}"] = message
        sandbox = Path(approach._tool_context.sandbox_dir)
        if len(seen) == 1:
            first_round(message)
            return _result()
        # The round that follows reports the model the last one deployed.
        assert "Last fit: " in message, message
        read["attempts.md"] = (sandbox / "attempts.md").read_text()
        assert "Model updated: simulator.py" in read["attempts.md"]
        for path in sorted(sandbox.glob("session_logs/*.md")):
            read[f"session log {path.name}"] = path.read_text()
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 2
    assert card.levels[0].split == "test" and card.levels[0].steps > 1
    # The round's end deploys the fitted file at its point estimates.
    assert approach._last_round_modelled
    assert approach._last_fit_result is not None
    assert "UNFITTED" not in approach._fit_status_text()
    assert approach.parameter_belief() is None
    for name, text in read.items():
        assert not _withheld(text), (name, _withheld(text))
