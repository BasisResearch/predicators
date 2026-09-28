"""The model half of the EMPIRIC arms: the agent's simulator program, its
fitted parameters and the belief over them.

The agent writes a simulator (a subclass of the base env, or residual
rules over it); the harness loads it, fits or deploys its parameters,
keeps their uncertainty, and composes it with a base oracle (PyBullet
with process dynamics disabled) into the option model plans are
rehearsed in. The continual arms (``agent_continual_approach``) drive
all of it from their play rounds.
"""

import copy
import dataclasses
import hashlib
import logging
import math
import os
import subprocess
from contextlib import contextmanager
from typing import Any, Callable, Collection, ContextManager, Dict, \
    FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.fit_status import format_fit_status
from predicators.agent_sdk.play_prompts import render_physical_params_section
from predicators.agent_sdk.session_base import max_session_log_number
from predicators.agent_sdk.tools import SYNTHESIS_TOOL_NAMES, \
    _SnapshotTarget, evaluate_states_with, finalize_versioned_snapshot, \
    make_write_snapshot_hook
from predicators.agent_sdk.tools.digests import render_trajectory_digest
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.approaches.synthesis_validation import \
    build_candidate_option_model, carry_over_params
from predicators.code_sim_learning.active_experiment import laplace_ensemble, \
    mean_bernoulli_entropy, noisy_read_information, perturbation_ensemble, \
    subsample_ensemble
from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.evidence import LaplaceEvidence
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    declared_interval_fit_result, declared_interval_report
from predicators.code_sim_learning.fitting import FIT_NOISE_SIGMA, \
    compute_sse, compute_sse_recurrent, log_sse_breakdown
from predicators.code_sim_learning.identifiability import physics_sigma_points
from predicators.code_sim_learning.latent_tracker import LatentTracker, \
    make_latent_tracker, make_subclass_latent_tracker
from predicators.code_sim_learning.model_state import has_model_state
from predicators.code_sim_learning.orchestrator import prior_parameter_belief
from predicators.code_sim_learning.parameter_belief import BeliefConfig, \
    ParameterBelief, stable_seed
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    dispose_env, physical_param_anchors
from predicators.code_sim_learning.rollout_objective import compute_rollout_sse
from predicators.code_sim_learning.trajectory_prep import \
    split_at_rest_points, truncate_settled_tail
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, apply_rules_with_latent, has_latent_rules, \
    has_physics_rules, init_latent, iter_feature_residuals, merge_updates, \
    observation_view, read_latent_init, read_physical_param_specs, \
    read_residual_env, read_simulator_components, stamp_physical_spec_scales
from predicators.envs import create_new_env
from predicators.observation_noise import ObservationNoise
from predicators.option_model import _OptionModelBase, _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, GroundAtom, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type, \
    step_option_labels

logger = logging.getLogger(__name__)

# Rows in the stand-in fit result under agent_sim_learn_declared_params_only:
# uniform draws over the declared boxes that the ensemble subsamples.
_DECLARED_INTERVAL_SAMPLES = 64


def _fit_space_dist(a: float, b: float, scale: str) -> float:
    """Distance between two param values in their fit space.

    Log-scale params compare multiplicatively (the space their prior and
    posterior widths live in); linear params compare additively.
    """
    if scale == "log":
        return abs(float(np.log(max(a, 1e-300)) - np.log(max(b, 1e-300))))
    return abs(a - b)


@dataclasses.dataclass(frozen=True)
class _SynthesisPaths:
    """Host- and agent-visible paths for one synthesis session.

    ``simulator_file`` / ``versions_dir`` are host paths the harness
    reads and writes; ``simulator_file_for_agent`` /
    ``sandbox_dir_for_agent`` are how the sandboxed agent must refer to
    the same locations (see ``_resolve_synthesis_paths``).
    """
    base: str
    simulator_file: str
    versions_dir: str
    simulator_file_for_agent: str
    sandbox_dir_for_agent: Optional[str]


def _describe_git_revision() -> str:
    """Best-effort ``git describe`` of the running code, for checkpoint version
    stamping ("unknown" outside a repo or without git)."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True,
                             text=True,
                             timeout=10,
                             check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        # TimeoutExpired is a SubprocessError, not an OSError.
        return "unknown"


# ── Approach ─────────────────────────────────────────────────────

# The allowlist value that keeps no env predicate at all: an empty list
# means "not set" (keep the class default), so "none" is the explicit
# spelling of an empty vocabulary.
NO_KEPT_PREDICATES = "none"


def resolve_kept_predicate_names(
        default: Optional[FrozenSet[str]]) -> Optional[FrozenSet[str]]:
    """Names of the env predicates an agent starts with: the CFG allowlist
    ``agent_sim_learn_kept_predicates_names`` when non-empty (``["none"]``
    keeps none), else ``default`` (``None`` = every env predicate)."""
    override = getattr(CFG, "agent_sim_learn_kept_predicates_names", None)
    if override:
        names = frozenset(override)
        if names == {NO_KEPT_PREDICATES}:
            return frozenset()
        return names
    return default


def count_residual_hits(
    base_pred_triples: Sequence[Tuple[State, Action, State]],
    hits: Dict[Tuple[str, str], int],
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> None:
    """Add to ``hits``, per ``(type, feature)``, the number of triples on which
    the base-sim prediction diverges from the observation by more.

    than ``rel_tol * |obs| + abs_tol``.

    Split from :func:`residual_hint_from_hits` so a caller whose data
    grows (the continual play loop, after every environment call) scans
    only the new triples and keeps the counts.
    """
    pairs = [(s_base, s_obs) for s_base, _, s_obs in base_pred_triples]
    for _, _, tn, feat, pred, obs in iter_feature_residuals(pairs):
        if abs(pred - obs) > rel_tol * abs(obs) + abs_tol:
            hits[(tn, feat)] = hits.get((tn, feat), 0) + 1


def residual_hint_from_hits(hits: Dict[Tuple[str, str], int],
                            min_hits: int = 3) -> Dict[str, List[str]]:
    """The ``{type: [features]}`` hint of the features that diverged on at
    least ``min_hits`` triples; the floor keeps one-off PyBullet jitter from
    leaking base-handled features into the set."""
    out: Dict[str, List[str]] = {}
    for (t, f), n in hits.items():
        if n >= min_hits:
            out.setdefault(t, []).append(f)
    return {t: sorted(fs) for t, fs in out.items()}


class AgentSimLearningApproach(AgentModelFreeApproach):
    """The agent-written simulator, its parameters and their belief.

    Loads the simulator the agent writes, deploys its parameters (the
    agent's published ``sim.fit`` or its declared values), keeps the
    belief over them, and composes it with the base oracle into the
    ``_OracleOptionModel`` the probe rehearses plans in.
    """

    # Allowlist of env predicate names surfaced to the agent; None keeps
    # every env predicate. CFG.agent_sim_learn_kept_predicates_names
    # overrides this class default when non-empty, so an experiment can
    # strip predicates - even goal predicates - from the agent's
    # vocabulary (prompts, tools, subgoal annotations) without touching
    # env-side goal checking or the task evaluator. Tasks whose goal
    # atoms are stripped must carry ``goal_nl``: the natural-language
    # goal becomes the agent's only goal signal.
    KEPT_INITIAL_PREDICATE_NAMES: Optional[FrozenSet[str]] = None

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 *args: Any,
                 option_model: Optional[_OptionModelBase] = None,
                 **kwargs: Any) -> None:
        # Pass the option model in so the parent __init__ doesn't spin up
        # its own full-process env, which would fight this one for the
        # PyBullet GUI client.
        self._base_env = self._create_initial_base_env(types)
        if option_model is None:
            option_model = _OracleOptionModel(initial_options,
                                              self._base_env.simulate)
            option_model.sim_env = self._base_env
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         *args,
                         option_model=option_model,
                         **kwargs)
        # Probe trials and sweep rollouts each run on a freshly constructed
        # env (see ToolContext.validation_env_scope): repeats on the shared
        # ``_base_env`` are correlated across resets, so only fresh envs
        # sample the distribution the real episode will.
        self._tool_context.validation_env_scope = \
            self._fresh_validation_env_scope
        # The sim.run physics sweep's stress points (see _stress_points):
        # a callable so the probe always sees the current fit, not the
        # one deployed when the session opened. Under
        # agent_sim_learn_param_uncertainty False the legacy grid is never
        # built (see _physics_margin_points).
        self._tool_context.physics_margin_provider = self._stress_points
        # The joint belief's rehearsals split a draw's parameters into the
        # ones the env applies and the rule parameters read through the
        # draw scope.
        self._tool_context.physical_param_names_provider = \
            lambda: {s.name for s in self._physical_param_specs}
        self._tool_context.joint_draw_scope = self.joint_draw_scope
        # Env predicates surfaced to the agent (see
        # KEPT_INITIAL_PREDICATE_NAMES). Computed once here; everything
        # agent-facing flows through _get_all_predicates().
        self._kept_initial_predicates: Set[Predicate] = (
            self._compute_kept_initial_predicates())
        if self._resolve_kept_names() is not None:
            kept_names = sorted(p.name for p in self._kept_initial_predicates)
            stripped = sorted(p.name for p in self._initial_predicates
                              if p not in self._kept_initial_predicates)
            logger.info(
                "Predicate stripping: kept %s; stripped (hidden from the "
                "agent): %s", kept_names, stripped)
            missing_nl = [
                i for i, t in enumerate(self._train_tasks)
                if not t.goal_nl and any(
                    a.predicate not in self._kept_initial_predicates
                    for a in t.goal)
            ]
            assert not missing_nl, (
                f"Stripping hides goal predicates from the agent, so the "
                f"affected tasks must supply `goal_nl` as the goal signal. "
                f"Missing on train task indices: {missing_nl}")
        self._learned_simulator: Optional[LearnedSimulator] = None
        # The subclass model form: when the loaded simulator.py exports a
        # RESIDUAL_ENV, the planning base env is an instance of that
        # subclass (its own _domain_specific_step runs, so
        # skip_residual_dynamics is False) and there are no residual
        # rules; its AGENT_PARAM_SPECS ride in _physical_param_specs and
        # are fit by the rollout system-ID. None on the rule form and on
        # every stock arm, where the base env is the fixed base-sim class
        # with skip_residual_dynamics=True.
        self._residual_env_cls: Optional[type] = None
        # Content key (simulator.py SHA256) of the installed subclass, so
        # repeated execs of unchanged content reuse the planning base env
        # instead of rebuilding it every probe/fit call (each exec makes
        # a fresh class object, so identity comparison never matches).
        self._residual_env_key: Optional[str] = None
        # Loss-scope mask for parameter fitting (compute_sse).
        self._residual_features: Dict[str, List[str]] = {}
        self._residual_rules: Optional[List] = None
        # Always the same dict object: fits update it in place via
        # clear()+update() so _ParamsView (held by invented predicate
        # classifiers) picks up new values without holding a reference to
        # ``self``. Truthy iff a fit has populated it.
        self._fitted_params: Dict[str, float] = {}
        # ParamSpecs of the most recently fitted simulator (names + bounds);
        # kept so the active-experiment ensemble can perturb each param
        # within its declared box. Parallel to ``_fitted_params``.
        self._param_specs: List[ParamSpec] = []
        # Small ensemble of plausible parameter vectors, rebuilt after
        # every fit when active-experiment exploration is on. When a
        # posterior fit exists, member 0 is that fit's MAP; otherwise it
        # falls back to ``_fitted_params``. Empty when info-seeking is
        # disabled or no fit has run yet.
        self._param_ensemble: List[Dict[str, float]] = []
        # Full result used for ensemble calibration (the solver fit's LM
        # MAP + Laplace Jacobian). ``None`` after an oracle-param run.
        self._last_fit_result: Optional[FitResult] = None
        self._fit_sse: float = float("inf")
        self._learning_mode: bool = False
        # Snapshot tags of the most recent simulator / predicates files
        # committed by the synthesis agent, used to stamp newly collected
        # online trajectories with their source-version provenance
        # (consumed in the next learn-phase prompt).
        self._current_simulator_version: Optional[str] = None
        self._current_predicates_version: Optional[str] = None
        # Partial-observability latent block: loaded from a simulator's
        # LATENT_INIT export (None ⇒ no latent state). When the loaded
        # rules use the recurrent 5-arg signature, fitting, the combined
        # simulator, and the SSE diagnostics thread this latent across
        # steps; legacy 3-arg rules ignore it entirely (fully-observable
        # behavior is unchanged). Dispatch keys off the rule signatures
        # via ``has_latent_rules``, not this field.
        self._latent_init: Any = None
        # Cached per learn cycle so recurrent fitting can regroup the flat
        # base_pred_triples back into per-trajectory chunks (latent
        # threads within a trajectory, not across).
        self._fit_trajectories: List[LowLevelTrajectory] = []
        # System identification: PHYSICAL_PARAM_SPECS export (agent-declared
        # sparse subset of self._base_env.get_physical_param_info()),
        # identified values applied in place to the base env. The rollout
        # fit itself builds a fresh headless env per rollout (see
        # _get_rollout_fit_env), never touching the planning base env.
        self._physical_param_specs: List[ParamSpec] = []
        self._identified_physical_params: Dict[str, float] = {}
        # The carried posterior (code_sim_learning_carry_posterior): the
        # most likely value of every physical param the last applied fit
        # deployed, the prior centre of the next fit.
        self._carried_physical_prior: Dict[str, float] = {}
        # Per canonical simulator version, the fit's Laplace evidence
        # record (code_sim_learning_fit_evidence): the sim.fit report's
        # delta against the previous version reads from here.
        self._fit_evidence_history: Dict[str, Dict[str, float]] = {}
        # +-1-posterior-sigma perturbations of the applied params (the
        # legacy physics sweep's grid). Set only by the joint
        # rollout fit, which has the identifiability report; cleared by
        # every _apply_identified_physical_params call so points can
        # never outlive the fit they were derived from.
        self._identified_physical_sigma_points: List[Dict[str, float]] = []
        # Explainability (trimming) verdicts are memoized per learn phase
        # (cleared when _fit_trajectories is refreshed): repeated
        # sim.fit calls with the same declaration signature
        # reuse the sweep instead of re-rolling it, which both saves
        # rollouts and pins the verdict for identical inputs.
        self._explainability_cache: Dict[Tuple, Tuple[List[float],
                                                      List[Dict[str,
                                                                float]]]] = {}
        # Whole-fit memoization for the orchestrator (same lifecycle as
        # the explainability cache): repeated canonical sim.fit calls on
        # an unchanged artifact version + data reuse the entire fit core
        # instead of re-rolling it. Keyed by (artifact version tag,
        # declaration/data signature); values are
        # orchestrator._FitComputation bundles.
        self._sysid_fit_cache: Dict[Tuple, Any] = {}

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_learning"

    # ── Predicate set ───────────────────────────────────────────

    def _get_all_predicates(self) -> Set[Predicate]:
        return self._kept_initial_predicates

    def _resolve_kept_names(self) -> Optional[FrozenSet[str]]:
        """Names of env predicates kept for the agent (None = keep all).

        The CFG flag overrides the class default.
        """
        return resolve_kept_predicate_names(self.KEPT_INITIAL_PREDICATE_NAMES)

    def _compute_kept_initial_predicates(self) -> Set[Predicate]:
        """Apply the allowlist, then closure-strip derived predicates.

        A ``DerivedPredicate`` whose ``auxiliary_predicates`` reference
        any stripped predicate is itself stripped: keeping one with
        removed dependencies would expose a broken classifier to
        refinement.
        """
        kept_names = self._resolve_kept_names()
        if kept_names is None:
            return set(self._initial_predicates)
        kept = {p for p in self._initial_predicates if p.name in kept_names}
        kept_pred_set = set(kept)
        for pred in self._initial_predicates:
            if not isinstance(pred, DerivedPredicate):
                continue
            if pred in kept_pred_set:
                aux = pred.auxiliary_predicates or set()
                if any(a not in kept_pred_set for a in aux):
                    kept.discard(pred)
        return kept

    # ── Agent session hooks ──────────────────────────────────────

    def _get_sandbox_reference_files(self) -> Dict[str, str]:
        files = super()._get_sandbox_reference_files()
        # Base-sim source rides the standard reference channel so every
        # session (solve, explore, synthesis) gets the same copies.
        if CFG.agent_sim_provide_base_sim_source:
            for rel in self._base_env.get_base_sim_source_files():
                files[f"base_sim/{os.path.basename(rel)}"] = rel
        return files

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """Complete tool surface for the synthesis agent.

        The names of the dynamic synthesis callables (just
        ``run_python``); a continual round keeps only the toolkit tools
        named here. Fitting, residual reports, and plan validation are
        NOT tools: they live on the ``sim`` probe (``sim.fit`` /
        ``sim.residuals`` / ``sim.refine`` / ``sim.run``) inside
        ``run_python``.

        No inspect tools: trajectory access lives in ``run_python``
        (``trajectories`` + ``describe_trajectory``). The probe rides
        inside that same ``run_python`` namespace as ``sim`` (one exec
        namespace per session - a helper defined next to the data is
        visible to probe sweeps) and runs against the CANDIDATE
        simulator.py via ctx.probe_option_model_provider.
        """
        names: List[str] = list(SYNTHESIS_TOOL_NAMES)
        return names

    # ── Subclass hooks ──────────────────────────────────────────
    # Default implementations are no-ops so subclasses can add
    # predicate-invention (or other) extensions.

    def _learning_cycle_index(self) -> int:
        """Index used in versioned snapshot filenames; the continual arms
        number snapshots by level."""
        return 0

    def _compute_extra_synthesis_paths(self, base: str) -> Dict[str, str]:
        """Return extra path bindings for the synthesis sandbox."""
        del base
        return {}

    def _install_extra_synthesis_surfaces(
        self,
        exec_ns: Dict[str, Any],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        extra_paths: Dict[str, str],
    ) -> None:
        """Install per-arm probe surfaces for the synthesis session.

        Subclasses register loaders in
        ``self._tool_context.probe_artifact_loaders`` (the backends of
        ``sim.predicates()`` / ``sim.samplers()``); the base arm has
        none.
        """
        del exec_ns, base_pred_triples, inferred_hint, extra_paths

    def _post_synthesis_loading(
        self,
        extra_paths: Dict[str, str],
        specs: List[ParamSpec],
    ) -> None:
        """Hook run after the simulator file is loaded post-session.

        ``specs`` are the just-loaded ``PARAM_SPECS``; subclasses may
        seed ``self._fitted_params`` from their ``init_value``s before
        the proper fit runs (useful when loading other artifacts that
        close over ``params``).
        """
        del extra_paths, specs

    def _build_write_snapshot_targets(
        self,
        simulator_file: str,
        versions_dir: str,
        extra_paths: Dict[str, str],
    ) -> List[_SnapshotTarget]:
        """Files the PostToolUse snapshot hook should watch.

        Defaults to just the simulator. Subclasses (e.g. predicate
        invention) may append their own artifacts. ``extra_paths`` is
        the same dict returned by ``_compute_extra_synthesis_paths``.
        """
        del extra_paths
        return [
            _SnapshotTarget(
                live_file=simulator_file,
                versions_dir=versions_dir,
                artifact_name="simulator",
                cycle_index_provider=self._learning_cycle_index,
            ),
        ]

    @staticmethod
    def _build_synthesis_session_hooks(
        targets: List[_SnapshotTarget],
        sandbox_dir: str,
    ) -> Dict[str, list]:
        """Wrap snapshot targets in a Claude Agent SDK ``HookMatcher``.

        Returns the dict suitable for assignment to
        ``ToolContext.extra_session_hooks``. Falls back to an empty dict
        if the SDK ``HookMatcher`` isn't importable (so the approach
        still works against older SDK versions).
        """
        if not targets:
            return {}
        try:
            from claude_agent_sdk import \
                HookMatcher  # pylint: disable=import-outside-toplevel
        except ImportError:
            logger.warning("claude_agent_sdk.HookMatcher unavailable; "
                           "write-time snapshots disabled.")
            return {}
        hook = make_write_snapshot_hook(targets, sandbox_dir=sandbox_dir)
        return {
            "PostToolUse": [
                HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[hook]),
            ],
        }

    # ── Learning ────────────────────────────────────────────────

    # ── Checkpointing ────────────────────────────────────────────
    # The base checkpoint (AgentModelFreeApproach.save/load) persists
    # the datasets + cycle counter. This approach's real state is split
    # between plain fitted values (pickled below) and the sandbox
    # artifacts the agent wrote (simulator.py / predicates.py / ...),
    # which are embedded as file CONTENTS - run dirs are minted per run
    # and pruned, so a path reference to the old run's sandbox would be
    # fragile. Closures (_residual_rules, _learned_simulator, the option
    # model, learned predicates) are never pickled: they are
    # rebuilt from the restored files in _rehydrate_from_artifacts.

    _save_suffix: str = "AgentSimLearner"

    _CHECKPOINT_SANDBOX_FILES: Tuple[str,
                                     ...] = ("simulator.py", "predicates.py",
                                             "ground_samplers.py", "notes.md",
                                             "journal.md", "attempts.md",
                                             "strategy.md",
                                             "open_questions.md")
    _CHECKPOINT_SANDBOX_DIRS: Tuple[str, ...] = ("simulator_versions",
                                                 "predicates_versions")
    _CHECKPOINT_MAX_FILE_BYTES = 2 * 1024 * 1024

    def _checkpoint_sandbox_dir(self) -> str:
        """The run's sandbox root - the SAME base the synthesis paths use,
        so artifacts are collected from and restored to where
        ``_rehydrate_from_artifacts`` (via ``_resolve_synthesis_paths``)
        looks for them on every sandbox backend."""
        return self._resolve_synthesis_paths().base

    def _collect_sandbox_artifacts(self) -> Dict[str, bytes]:
        """Curated sandbox files as {relative path: content} for the
        checkpoint.

        Skips session logs, reference copies, images, and git state -
        bulky and reconstructable. Oversized files are skipped with a
        warning rather than failing the save.
        """
        base = self._checkpoint_sandbox_dir()
        rel_paths: List[str] = [
            f for f in self._CHECKPOINT_SANDBOX_FILES
            if os.path.isfile(os.path.join(base, f))
        ]
        for dirname in self._CHECKPOINT_SANDBOX_DIRS:
            dirpath = os.path.join(base, dirname)
            if not os.path.isdir(dirpath):
                continue
            for fname in sorted(os.listdir(dirpath)):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    rel_paths.append(os.path.join(dirname, fname))
        files: Dict[str, bytes] = {}
        for rel in rel_paths:
            fpath = os.path.join(base, rel)
            size = os.path.getsize(fpath)
            if size > self._CHECKPOINT_MAX_FILE_BYTES:
                logger.warning(
                    "Checkpoint skipping oversized sandbox file %s "
                    "(%d bytes).", rel, size)
                continue
            with open(fpath, "rb") as f:
                files[rel] = f.read()
        return files

    def _restore_sandbox_artifacts(self, files: Dict[str, bytes]) -> None:
        """Write embedded sandbox files into THIS run's sandbox.

        Safe against the lazy sandbox setup: ``setup_sandbox_directory``
        only writes reference/CLAUDE.md/hooks and seeds notes.md when
        missing, so restoring first never gets clobbered.
        """
        base = self._checkpoint_sandbox_dir()
        for rel, content in files.items():
            fpath = os.path.join(base, rel)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "wb") as f:
                f.write(content)
        if files:
            logger.info("Restored %d sandbox artifact(s) into %s.", len(files),
                        base)

    def _extra_save_state(self) -> Dict[str, Any]:
        return {
            "fitted_params":
            dict(self._fitted_params),
            "fit_sse":
            self._fit_sse,
            "param_specs":
            list(self._param_specs),
            "physical_param_specs":
            list(self._physical_param_specs),
            "last_fit_result":
            self._last_fit_result,
            "probe_fit_state":
            dict(self._probe_fit_state()),
            "param_ensemble":
            list(self._param_ensemble),
            "identified_physical_params":
            dict(self._identified_physical_params),
            "carried_physical_prior":
            dict(self._carried_physical_prior),
            "fit_evidence_history":
            dict(self._fit_evidence_history),
            "identified_physical_sigma_points":
            list(self._identified_physical_sigma_points),
            "residual_features":
            dict(self._residual_features),
            "current_simulator_version":
            self._current_simulator_version,
            "current_predicates_version":
            self._current_predicates_version,
            "sandbox_files":
            self._collect_sandbox_artifacts(),
            "git_describe":
            _describe_git_revision(),
            # Highest session-transcript id so far, so a resumed run
            # keeps numbering its transcripts after this run's.
            "agent_query_count":
            max_session_log_number(self._get_log_dir()),
        }

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        saved_rev = save_dict.get("git_describe")
        current_rev = _describe_git_revision()
        if saved_rev and saved_rev != current_rev:
            logger.warning(
                "Checkpoint was written at git revision %s but this run "
                "is at %s - resuming across code versions is untested.",
                saved_rev, current_rev)
        self._resume_query_count = int(save_dict.get("agent_query_count", 0))
        # In-place update: _ParamsView holders (invented predicate
        # closures) alias this exact dict object.
        self._fitted_params.clear()
        self._fitted_params.update(save_dict.get("fitted_params") or {})
        self._fit_sse = save_dict.get("fit_sse", float("inf"))
        self._param_specs = list(save_dict.get("param_specs") or [])
        self._physical_param_specs = list(
            save_dict.get("physical_param_specs") or [])
        self._last_fit_result = save_dict.get("last_fit_result")
        self._probe_fit_state().clear()
        self._probe_fit_state().update(save_dict.get("probe_fit_state") or {})
        self._probe_model_cache().clear()
        self._param_ensemble = list(save_dict.get("param_ensemble") or [])
        self._identified_physical_params = dict(
            save_dict.get("identified_physical_params") or {})
        self._carried_physical_prior = dict(
            save_dict.get("carried_physical_prior") or {})
        self._fit_evidence_history = dict(
            save_dict.get("fit_evidence_history") or {})
        self._residual_features = dict(
            save_dict.get("residual_features") or {})
        self._current_simulator_version = save_dict.get(
            "current_simulator_version")
        self._current_predicates_version = save_dict.get(
            "current_predicates_version")
        self._restore_sandbox_artifacts(save_dict.get("sandbox_files") or {})
        self._rehydrate_from_artifacts()
        # AFTER rehydration: _apply_identified_physical_params clears
        # the sigma points (they must never outlive the fit they came
        # from), so the checkpointed points are restored last.
        self._identified_physical_sigma_points = list(
            save_dict.get("identified_physical_sigma_points") or [])

    def _rehydrate_extra_artifacts(self, base: str) -> None:
        """Subclass hook: reload extra artifacts (e.g. predicates.py)."""

    def _rehydrate_from_artifacts(self) -> None:
        """Rebuild the learned simulator/option model from restored files.

        Order matters: simulator.py first (rules + latent init +
        physical specs), then the option model, then identified physics
        onto the base env, then subclass artifacts (predicates read the
        already- restored ``_fitted_params``), then the ensemble.
        """
        paths = self._resolve_synthesis_paths()
        if not os.path.isfile(paths.simulator_file):
            logger.info("Checkpoint carried no simulator.py; the initial "
                        "option model stands (resume before the first "
                        "successful synthesis).")
            self._rehydrate_extra_artifacts(paths.base)
            return
        trajectories = self._get_all_trajectories()
        self._fit_trajectories = list(trajectories)
        rules, specs, declared_features, sim_ns = (
            self._load_simulator_from_module_file(paths.simulator_file,
                                                  trajectories))
        residual_env_cls = read_residual_env(sim_ns) if isinstance(
            sim_ns, dict) else None
        # The subclass model form carries its dynamics on the class, so
        # its rules load empty; every other form must have non-empty
        # rules to be loadable.
        if (not rules and residual_env_cls is None) or specs is None:
            logger.warning(
                "Restored simulator.py failed to load; continuing with "
                "the initial option model (the next learn cycle will "
                "rebuild it).")
            self._rehydrate_extra_artifacts(paths.base)
            return
        # Past the guard rules is a list (empty only for the subclass form,
        # whose dynamics live on the class); coerce a None the guard let
        # through (subclass present) so the downstream step_fn sees a list.
        rules = rules or []
        self._install_residual_env_cls(residual_env_cls)
        self._residual_rules = rules
        if declared_features:
            self._residual_features = declared_features
        elif residual_env_cls is not None:
            self._residual_features = dict(
                getattr(residual_env_cls, "RESIDUAL_FEATURES", {}))
        self._latent_init = (read_latent_init(sim_ns) if isinstance(
            sim_ns, dict) else None)
        # Subclass form: the physical params to identify are the class's
        # AGENT_PARAM_SPECS (stamped against the now-installed subclass
        # instance); otherwise the optional PHYSICAL_PARAM_SPECS export.
        if residual_env_cls is not None:
            self._physical_param_specs = stamp_physical_spec_scales(
                list(residual_env_cls.AGENT_PARAM_SPECS), self._base_env)
        else:
            self._physical_param_specs = stamp_physical_spec_scales(
                list((read_physical_param_specs(sim_ns) if isinstance(
                    sim_ns, dict) else None) or []), self._base_env)
        # The agent may have edited simulator.py after the last fit:
        # pickled fitted params are only valid for matching spec names.
        spec_names = {s.name for s in specs}
        if set(self._fitted_params) != spec_names:
            logger.warning(
                "Checkpointed fitted params %s do not match the restored "
                "simulator's PARAM_SPECS %s; falling back to declared "
                "init values.", sorted(self._fitted_params),
                sorted(spec_names))
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})
        rules_ref, params_ref = self._residual_rules, self._fitted_params

        def _step_fn(s: State, c: Any) -> Any:
            return apply_rules(s, rules_ref, params_ref, cmds=c)

        self._learned_simulator = LearnedSimulator(step_fn=_step_fn,
                                                   name="agent_synthesized")
        combined_sim = self._build_combined_simulator(self._learned_simulator)
        self._option_model = self._build_option_model(combined_sim)
        if self._identified_physical_params:
            self._apply_identified_physical_params(
                self._identified_physical_params)
        self._rehydrate_extra_artifacts(paths.base)
        self._rebuild_param_ensemble()
        logger.info(
            "Rehydrated learned simulator from checkpoint artifacts "
            "(%d rules, %d fitted params, %d learned predicates).", len(rules),
            len(self._fitted_params),
            len(getattr(self, "_learned_predicates", set()) or set()))

    def _build_option_model(
        self,
        simulator_fn: Callable[[State, Action], State],
    ) -> _OracleOptionModel:
        """Wrap a simulator function in an OracleOptionModel.

        Uses ``self._get_all_options()`` rather than
        ``get_gt_options(CFG.env)`` to avoid spawning a second cached
        PyBullet env via ``get_or_create_env``.
        """
        model = _OracleOptionModel(self._get_all_options(), simulator_fn)
        # The learned simulator_fn rides on top of _base_env's physics,
        # so that env is the one physics-needing task-evaluator
        # certificates (the domino counterfactual push probe) must run
        # against. Without this the probe is silently unavailable in the
        # sandbox and plans are scored on the pure rules only.
        model.sim_env = self._base_env
        # Belief-side verdicts predict the real evaluator, so the
        # certificate's verification replay must run the agent's FULL
        # current model (base sim + these rules), not a rules-free base
        # sim: at miscalibrated base physics a rules-free replay
        # rejected every legitimate relay and taught the agent a
        # phantom task rule (run_20260727_210818 seed2).
        self._base_env.probe_process_model_factory = \
            self._make_probe_process_model_factory()
        if CFG.wait_option_terminate_on_atom_change:
            model._abstract_function = (  # pylint: disable=protected-access
                lambda s: utils.abstract(s, self._get_all_predicates()))
        return model

    def _probe_model_cache(self) -> Dict[str, Any]:
        """The candidate probe model cache (content digest -> model)."""
        cache = getattr(self, "_probe_model_cache_store", None)
        if cache is None:
            cache = {}
            setattr(self, "_probe_model_cache_store", cache)
        return cache

    def _probe_fit_state(self) -> Dict[str, Any]:
        """Which simulator.py content the last canonical fit deployed."""
        state = getattr(self, "_probe_fit_state_store", None)
        if state is None:
            state = {}
            setattr(self, "_probe_fit_state_store", state)
        return state

    def _publish_probe_fit(
        self,
        params: Dict[str, float],
        version_tag: str,
        simulator_file: str,
        fit_result: Optional[FitResult] = None,
        sse: float = float("nan"),
        applied_physical: Optional[Dict[str, float]] = None,
        sigma_points: Optional[List[Dict[str, float]]] = None,
        pinned: bool = False,
        coverage: Optional[Tuple[int, int]] = None,
        belief: Optional[ParameterBelief] = None,
    ) -> None:
        """Deploy a canonical ``sim.fit`` result to the candidate probe.

        Publishes the fitted values in place (invented predicates hold a
        live view over ``_fitted_params``), records the fitted file
        content, and drops the cached probe model so the next probe
        rebuilds at these values without fitting again. The full
        ``fit_result`` (point estimate plus the Laplace bundle the
        exploration ensemble is calibrated from), its ``sse``, and the
        physical values actually applied to the planning env are kept
        so the cycle's deployed model can be exactly this fit (see
        :meth:`_published_fit_for_file`).

        ``pinned`` marks a fit that never ran: rollout sysID found no
        trajectory explainable at any candidate parameters and left the
        values at the declared inits. Such a publish must not displace
        a real fit of the SAME file content: the earlier finite fit
        stays canonical and the pinned attempt is only logged, so the
        deployed model is never silently downgraded from a validated
        fit to unvalidated inits (bridge seed 3, 2026-09-03: an "SSE
        nan" model was deployed and every certification that cycle
        ran against it). ``sse`` must be finite; a pinned fit reports
        the SSE at the inits over all segments rather than nan.

        ``belief`` is the fit's parameter factor of the joint belief
        (``belief_joint_draws`` > 0), stored with the fit so that
        :meth:`parameter_belief` serves it while the file is unchanged
        and checkpoints carry it.
        """
        digest = None
        if os.path.isfile(simulator_file):
            with open(simulator_file, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        state = self._probe_fit_state()
        if pinned and state.get("fit_result") is not None and \
                state.get("digest") == digest and \
                not state.get("pinned", False) and \
                math.isfinite(float(state.get("sse", float("nan")))):
            logger.warning(
                "sim.fit (%s) ran no fit (nothing explainable at any "
                "params); keeping the earlier finite fit %s (SSE %.6f) of "
                "the same simulator.py as canonical.", version_tag,
                state.get("version"), float(state["sse"]))
            state["last_rejection"] = version_tag
            self._tool_context.probe_param_status = format_fit_status(state)
            self._probe_model_cache().clear()
            return
        self._fitted_params.clear()
        self._fitted_params.update(params)
        self._sync_subclass_parameters()
        state["digest"] = digest
        state["version"] = version_tag
        state["fit_result"] = fit_result
        state["sse"] = sse
        state["pinned"] = bool(pinned)
        state["coverage"] = coverage
        state.pop("last_rejection", None)
        state["applied_physical"] = dict(applied_physical or {})
        state["sigma_points"] = list(sigma_points or [])
        state["belief"] = belief.to_dict() if belief is not None else None
        self._probe_model_cache().clear()
        self._tool_context.probe_param_status = format_fit_status(state)
        logger.info("Synthesis probe: deployed %d params; %s.", len(params),
                    self._tool_context.probe_param_status)

    def _published_fit_for_file(
        self,
        simulator_file: str,
        expected_names: Collection[str],
    ) -> Optional[Tuple[FitResult, float, str]]:
        """The agent's last canonical ``sim.fit`` of exactly this file.

        Returns ``(fit_result, sse, version_tag)`` when the last
        published fit ran on the current content of ``simulator_file``
        and over exactly ``expected_names``; ``None`` when nothing was
        published, the file changed after the fit (an UNFITTED edit), or
        the parameter set differs (a spec added or dropped after the
        fit). A pinned publish (see :meth:`_publish_probe_fit`) is
        returned like any other; :meth:`_published_fit_is_pinned` says
        whether the values were ever fit.
        """
        state = self._probe_fit_state()
        fit = state.get("fit_result")
        if fit is None or not os.path.isfile(simulator_file):
            return None
        with open(simulator_file, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if state.get("digest") != digest:
            return None
        if set(fit.names) != set(expected_names):
            return None
        return fit, float(state.get("sse",
                                    float("nan"))), str(state.get("version"))

    def _published_fit_is_pinned(self) -> bool:
        """Whether the canonical published fit never actually ran."""
        return bool(self._probe_fit_state().get("pinned", False))

    def _stress_points(self) -> List[Dict[str, float]]:
        """The parameter settings a physics sweep stress-tests.

        ``sim.run(plan, physics_sweep=True)`` runs the plan at each.

        Under the joint belief: each physical parameter at the ends of
        its 95% posterior interval, the others at their most likely
        values; these locate failure boundaries and carry no
        probability. Otherwise the legacy +-1 sigma grid of the last
        applied fit.
        """
        if int(CFG.belief_joint_draws) <= 0:
            return list(self._identified_physical_sigma_points)
        belief = self.parameter_belief()
        if belief is None:
            return []
        physical = {s.name for s in self._physical_param_specs}
        base = {
            n: float(belief.map_estimate[n])
            for n in belief.names if n in physical
        }
        points: List[Dict[str, float]] = []
        for name in base:
            for value in belief.interval(name, coverage=0.95):
                if not np.isclose(value, base[name]):
                    points.append({**base, name: float(value)})
        return points

    def _current_simulator_digest(self) -> Optional[str]:
        """The hash of the current ``simulator.py``, or None without one."""
        try:
            path = self._resolve_synthesis_paths().simulator_file
        except Exception:  # pylint: disable=broad-except
            return None
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _belief_cache(self) -> Dict[Any, ParameterBelief]:
        """Parameter beliefs by (source, version or file, parameters)."""
        cache = getattr(self, "_belief_cache_store", None)
        if cache is None:
            cache = {}
            setattr(self, "_belief_cache_store", cache)
        return cache

    def parameter_belief(self) -> Optional[ParameterBelief]:
        """The parameter factor ``q(theta)`` of the joint belief, or None.

        None when the joint belief is off (``belief_joint_draws`` 0, the
        legacy uncertainty path). Otherwise the belief of the agent's
        canonical ``sim.fit`` when that fit ran on the current content
        of ``simulator.py`` over its current parameters, and the fit's
        own prior, restricted to the declared bounds, before any such
        fit, after an edit, or when harness fitting is disabled. A
        program without parameters, such as a supplied model, gets an
        empty belief whose draws vary nothing.
        """
        if int(CFG.belief_joint_draws) <= 0:
            return None
        specs = list(self._physical_param_specs) + list(
            getattr(self, "_probe_rule_specs", None) or [])
        names = tuple(s.name for s in specs)
        digest = self._current_simulator_digest()
        state = self._probe_fit_state()
        stored = state.get("belief")
        key: Tuple[Any, ...]
        if (stored is not None and digest is not None
                and state.get("digest") == digest
                and not state.get("pinned", False)
                and sorted(stored["names"]) == sorted(names)):
            key = ("fit", state.get("version"), digest)
        else:
            stored = None
            key = ("prior", digest, names)
        cache = self._belief_cache()
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            if stored is not None:
                cache[key] = ParameterBelief.from_dict(stored)
            else:
                cache[key] = prior_parameter_belief(
                    specs,
                    self.fit_prior_anchors(self._physical_param_specs),
                    BeliefConfig.from_cfg(),
                    seed=stable_seed(CFG.seed, "prior", digest, names))
        return cache[key]

    def _make_candidate_probe_model_provider(
        self,
        simulator_file: str,
        trajectories: List[LowLevelTrajectory],
    ) -> Callable[[], _OracleOptionModel]:
        """Lazy option-model builder behind the synthesis run_python.

        The returned callable is installed as
        ``ctx.probe_option_model_provider`` for the synthesis session:
        on first use, and again after every content change of
        ``simulator_file``, it loads the candidate simulator and builds
        the combined option model at the last fit's values (declared
        init values for new params) - it never fits. A canonical
        ``sim.fit`` publishes through :meth:`_publish_probe_fit`, which
        marks the fitted content and drops the cache so the next probe
        runs at the fitted values. ``ctx.probe_param_status`` tells the
        agent which of the two it is looking at. Content-hash caching
        keeps sweep loops cheap: an unchanged file is never rebuilt.

        Raises ``RuntimeError`` (surfaced in the tool output) when no
        loadable candidate exists yet - the probe must never fall back
        to the pre-synthesis option model, which on cycle 1 wraps the
        real env (a live-physics leak into learning).
        """
        self._tool_context.probe_validation_env_scope = \
            self._fresh_candidate_validation_scope
        cache = self._probe_model_cache()

        def _provider() -> _OracleOptionModel:
            if not os.path.isfile(simulator_file):
                raise RuntimeError(
                    "run_python probe: no candidate simulator yet - "
                    "write ./simulator.py (RESIDUAL_ENV / AGENT_PARAM_SPECS / "
                    "RESIDUAL_FEATURES) first; the probe runs against it.")
            with open(simulator_file, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if cache.get("digest") == digest:
                return cache["model"]
            fit_state = self._probe_fit_state()
            rules, specs, _, ns = self._load_simulator_from_module_file(
                simulator_file, trajectories)
            if rules is None or specs is None:
                raise RuntimeError(
                    "run_python probe: ./simulator.py failed to load "
                    "(exec error or missing simulator exports) - "
                    "fix the file and probe again.")
            # The candidate's rule parameters, which the joint belief
            # covers beside AGENT_PARAM_SPECS (see parameter_belief).
            setattr(self, "_probe_rule_specs", list(specs))
            latent_init = read_latent_init(ns) if isinstance(ns,
                                                             dict) else None
            # The subclass model form: install the candidate's RESIDUAL_ENV
            # as the planning base env (keyed on content so identical
            # re-execs reuse it) and take its AGENT_PARAM_SPECS as the
            # physical params to fit, so the option model built below runs
            # over an instance of the subclass and sim.run / sim.fit /
            # sim.residuals all exercise its dynamics. None on the rule
            # form, which clears any previously installed subclass.
            residual_env_cls = read_residual_env(ns) if isinstance(
                ns, dict) else None
            self._install_residual_env_cls(residual_env_cls, digest)
            if residual_env_cls is not None:
                self._physical_param_specs = stamp_physical_spec_scales(
                    list(residual_env_cls.AGENT_PARAM_SPECS), self._base_env)
            # Never fit here: fitting is the agent's explicit ``sim.fit``
            # (its own budget, its own report). The candidate runs at
            # the last published fit's values (declared init values for
            # new params) and every probe result says so until the
            # agent fits the current file - so a rollout never silently
            # runs a fit it may not afford (sketch seed1 learn 011: an
            # implicit refit inside a probe hit the call cap and came
            # back as an empty "param fitting failed:").
            model, params = build_candidate_option_model(
                self, rules, specs, latent_init=latent_init)
            if CFG.agent_sim_learn_declared_params_only:
                status = ("at the DECLARED values of the current "
                          "simulator.py (harness parameter estimation is "
                          "disabled in this run)")
            elif fit_state.get("digest") == digest:
                status = format_fit_status(fit_state)
            else:
                status = (
                    "UNFITTED for the current simulator.py - the candidate "
                    "runs at the last fit's values where a param still "
                    "exists (declared init values otherwise); run "
                    "sim.fit() to fit and deploy the current file before "
                    "trusting quantitative results or a GO verdict")
            self._tool_context.probe_param_status = status
            logger.info(
                "Synthesis probe: candidate model rebuilt from %s (%d "
                "params, %s).", simulator_file, len(params), status)
            cache["digest"] = digest
            cache["model"] = model
            return model

        return _provider

    # ── Active-experiment ensemble (info-seeking exploration) ────

    def _rebuild_param_ensemble(self) -> None:
        """Rebuild the learned model's rule-parameter ensemble.

        Its consumer is info-seeking exploration
        (:meth:`score_atom_disagreement`). Built when that is on, a fit
        has populated ``_fitted_params`` and parameter uncertainty is in
        use; cleared otherwise. The ensemble can use an exploration-only
        posterior even when solver params remain at the global-budget
        point estimate.

        Picks the most *calibrated* ensemble the fit affords, preferring
        spreads that reflect real posterior uncertainty over uniform
        jitter (see :meth:`_select_param_ensemble`).
        """
        if int(CFG.belief_joint_draws) > 0:
            # The joint belief replaces the legacy ensembles: its draws,
            # restricted to the rule parameters the predicates read.
            belief = self.parameter_belief()
            rule_names = set(self._fitted_params)
            members = [] if belief is None else [{
                n: v
                for n, v in draw.items() if n in rule_names
            } for draw in belief.draw_dicts()]
            self._param_ensemble = members if any(members) else []
            logger.info(
                "Rule-parameter ensemble: %d draws of the joint belief.",
                len(self._param_ensemble))
            return
        if (not CFG.agent_explorer_info_seeking or not self._fitted_params
                or not CFG.agent_sim_learn_param_uncertainty):
            self._param_ensemble = []
            return
        num_members = CFG.agent_explorer_info_ensemble_size
        self._param_ensemble, method = self._select_param_ensemble(num_members)
        logger.info(
            "Built active-experiment ensemble: %d members via %s over "
            "%d params.", len(self._param_ensemble), method,
            len(self._param_specs))

    def _select_param_ensemble(
            self, num_members: int) -> Tuple[List[Dict[str, float]], str]:
        """Choose and build the ensemble, returning (members, method-label).

        Dispatch, most- to least-calibrated:

        * ``subsample`` - when the fit carries an explicit multi-row
          sample set (the declared-params ablation's
          ``declared_interval_fit_result`` fills it with uniform draws
          over the declared boxes), subsample those rows and anchor at
          the fit's own combined point estimate (physical + rule params).
        * ``laplace`` - when the fit attached an LM Jacobian, draw from the
          Laplace covariance at the MAP (per-transition or recurrent).
        * ``uniform`` - otherwise (oracle params, LM skipped/failed, or
          calibration disabled), fall back to box-relative jitter.
        """
        fit = self._last_fit_result
        calibrated = CFG.agent_explorer_info_calibrated_ensemble
        if calibrated and fit is not None:
            samples = np.asarray(fit.samples, dtype=float)
            if samples.ndim == 2 and samples.shape[0] > 1:
                return subsample_ensemble(
                    fit.point_estimate,
                    fit.names,
                    samples,
                    num_members=num_members,
                    rng=self._rng,
                ), "subsample"
            if (fit.jacobian is not None and fit.noise_sigma is not None
                    and fit.prior_sigma is not None):
                return laplace_ensemble(
                    self._fitted_params,
                    fit.names,
                    self._param_specs,
                    fit.jacobian,
                    fit.noise_sigma,
                    fit.prior_sigma,
                    num_members=num_members,
                    rng=self._rng,
                ), "laplace"
        return perturbation_ensemble(
            self._fitted_params,
            self._param_specs,
            num_members=num_members,
            perturb_frac=CFG.agent_explorer_info_perturb_frac,
            rng=self._rng,
        ), "uniform-perturb"

    def score_atom_disagreement(self, state: State,
                                atoms: Collection[GroundAtom]) -> float:
        """Ensemble disagreement (mean Bernoulli entropy) over ``atoms``.

        Evaluates each atom's truth in ``state`` under every ensemble
        member by swapping ``_fitted_params`` (which the learned
        predicate classifiers read through ``_ParamsView``) to each
        member in turn, then restoring it. High disagreement marks a
        state that straddles a learned predicate's decision boundary,
        i.e. an informative experiment. Returns 0.0 when the ensemble is
        trivial (<=1 member) or no atoms are given.

        Wired into the probe's refinement as the info-scorer; a read-only
        query that leaves ``_fitted_params`` unchanged on return.

        Under ``agent_explorer_info_seeking_noise_aware`` with a declared
        observation-noise channel, each member reads the atoms from
        noisy views of ``state`` (:meth:`_noisy_read_views`) and the
        score is the mutual information between the member and the
        read truth (:func:`noisy_read_information`): members whose
        predictions differ by less than sigma all read a coin flip and
        contribute nothing, because one noisy observation cannot tell
        them apart.
        """
        atom_list = list(atoms)
        if len(self._param_ensemble) <= 1 or not atom_list:
            return 0.0
        views = self._noisy_read_views(state)
        saved = dict(self._fitted_params)
        try:
            rows: List[List[float]] = []
            for member in self._param_ensemble:
                self._fitted_params.clear()
                self._fitted_params.update(member)
                if views is None:
                    rows.append([float(a.holds(state)) for a in atom_list])
                else:
                    rows.append([
                        float(np.mean([bool(a.holds(v)) for v in views]))
                        for a in atom_list
                    ])
        finally:
            self._fitted_params.clear()
            self._fitted_params.update(saved)
        if views is None:
            return mean_bernoulli_entropy(np.asarray(rows, dtype=bool))
        return noisy_read_information(np.asarray(rows, dtype=float))

    # Noisy views per scored state: enough to resolve a coin flip from
    # a near-certain read, few enough that scoring stays cheap.
    _NOISY_READ_DRAWS = 8

    def _noisy_read_views(self, state: State) -> Optional[List[State]]:
        """The observations a real step ending at ``state`` could return under
        the declared noise channel, or None when the score is exact (flag off,
        channel off or undeclared).

        Common random numbers: the draws are seeded identically for
        every scored state, so candidate scores are comparable and a re-
        score is repeatable.
        """
        if not CFG.agent_explorer_info_seeking_noise_aware:
            return None
        noise = ObservationNoise.from_cfg()
        if not (noise.enabled and noise.declared):
            return None
        rng = np.random.default_rng(CFG.seed)
        return [
            noise.perturb(state, rng) for _ in range(self._NOISY_READ_DRAWS)
        ]

    @contextmanager
    def _rule_param_override_scope(
            self, override: Dict[str, float]) -> Iterator[None]:
        """Swap ``_fitted_params`` to ``override`` for the duration.

        The learned rules and frozen predicate classifiers read the
        fitted params through a live view (see ``_ParamsView``), so the
        swap changes their gates for the wrapped validation rollout and
        the restore returns the deployed fit untouched - the same
        pattern :meth:`score_atom_disagreement` uses for ensemble
        scoring.
        """
        saved = dict(self._fitted_params)
        self._fitted_params.clear()
        self._fitted_params.update(override)
        try:
            yield
        finally:
            self._fitted_params.clear()
            self._fitted_params.update(saved)

    # ── Agent-based synthesis ────────────────────────────────────

    def _resolve_synthesis_paths(self) -> _SynthesisPaths:
        """Host- and agent-visible paths for one synthesis session.

        The sandbox dir is resolved without depending on a live session
        manager: LocalSandboxSessionManager does set it on tool_context
        in __init__, but it isn't constructed until
        ``_ensure_agent_session()`` runs later in the session setup.

        The agent-visible paths are cwd-relative in the local sandbox
        (the validation hook resolves against cwd and rejects literal
        ``/sandbox/...`` paths) and absolute host paths otherwise.
        """
        if CFG.agent_sdk_use_local_sandbox:
            sandbox_dir: Optional[str] = os.path.abspath(
                os.path.join(self._get_log_dir(), "sandbox"))
        else:
            sandbox_dir = self._tool_context.sandbox_dir
        base = sandbox_dir or self._get_log_dir()
        simulator_file = os.path.join(base, "simulator.py")
        if CFG.agent_sdk_use_local_sandbox:
            simulator_file_for_agent = "./simulator.py"
            sandbox_dir_for_agent: Optional[str] = "."
        else:
            simulator_file_for_agent = simulator_file
            sandbox_dir_for_agent = None
        return _SynthesisPaths(
            base=base,
            simulator_file=simulator_file,
            versions_dir=os.path.join(base, "simulator_versions"),
            simulator_file_for_agent=simulator_file_for_agent,
            sandbox_dir_for_agent=sandbox_dir_for_agent)

    def _build_synthesis_exec_ns(
            self, trajectories: List[LowLevelTrajectory]) -> Dict[str, Any]:
        """Variables exposed to the synthesis agent's ``run_python``."""
        exec_ns: Dict[str, Any] = {
            "trajectories":
            trajectories,
            "train_tasks":
            self._train_tasks,
            "is_goal_state":
            lambda state, task_idx: self._train_tasks[task_idx].goal_holds(
                state),
            "np":
            np,
            "ParamSpec":
            ParamSpec,
        }
        # Curated per-trajectory digest, for a first look before
        # ad-hoc exploration over the raw ``trajectories`` objects.
        all_predicates = self._get_all_predicates()

        def describe_trajectory(traj_idx: int,
                                include_states: bool = True,
                                include_atoms: bool = False,
                                max_timesteps: int = 10) -> str:
            return render_trajectory_digest(trajectories,
                                            self._train_tasks,
                                            all_predicates,
                                            traj_idx,
                                            include_states=include_states,
                                            include_atoms=include_atoms,
                                            max_timesteps=max_timesteps)

        exec_ns["describe_trajectory"] = describe_trajectory
        # The task's reward model, next to is_goal_state (see
        # Task.evaluator). Verdict-only surface: dict of reward/solved/
        # note on a concrete state sequence - real trajectories or the
        # agent's own simulator rollouts (there the verdict is only as
        # good as the sim, and the note says what was simulated).
        if any(t.evaluator is not None for t in self._train_tasks):
            exec_ns["evaluate_trajectory"] = \
                self._make_evaluate_trajectory_fn()
        return exec_ns

    def _load_synthesis_artifacts(
        self,
        trajectories: List[LowLevelTrajectory],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        extra_paths: Dict[str, str],
    ) -> Optional[Tuple[List, List[ParamSpec], Dict[str, List[str]]]]:
        """Load the artifacts the finished session committed to disk.

        Returns ``(rules, specs, residual_features)`` or None when no
        loadable simulator exists. The optional LATENT_INIT /
        PHYSICAL_PARAM_SPECS side exports are recorded on ``self``
        before the loadability check, so they are picked up even from an
        artifact whose rules fail to load.
        """
        final_sim_tag = finalize_versioned_snapshot(
            paths.simulator_file,
            paths.versions_dir,
            cycle_idx=self._learning_cycle_index(),
            artifact_name="simulator",
        )
        if final_sim_tag is not None:
            self._current_simulator_version = final_sim_tag
            logger.info("Final simulator snapshot: %s", final_sim_tag)

        rules, specs, declared_features, sim_ns = (
            self._load_simulator_from_module_file(paths.simulator_file,
                                                  trajectories))
        # Pick up the optional LATENT_INIT export (partial
        # observability). None for fully-observable simulators, which
        # leaves every latent path dormant.
        self._latent_init = (read_latent_init(sim_ns) if isinstance(
            sim_ns, dict) else None)
        # The subclass model form (RESIDUAL_ENV): install the subclass as
        # the planning base env, and take its AGENT_PARAM_SPECS as the
        # physical params to system-ID (they are already fully-specified
        # ParamSpecs, so stamp them against an INSTANCE of the subclass -
        # which _install_residual_env_cls has just made self._base_env -
        # never the stock base env, which does not carry them). None on
        # the rule form and every stock arm, which keep today's behavior.
        residual_env_cls = read_residual_env(sim_ns) if isinstance(
            sim_ns, dict) else None
        self._install_residual_env_cls(residual_env_cls)
        if residual_env_cls is not None:
            self._physical_param_specs = stamp_physical_spec_scales(
                list(residual_env_cls.AGENT_PARAM_SPECS), self._base_env)
        else:
            # Optional PHYSICAL_PARAM_SPECS export: base-sim parameters to
            # identify jointly with the rule params (system ID). The fit
            # scale (log vs linear) is stamped from the env registry;
            # agents copy name/init/bounds but need not know about it.
            self._physical_param_specs = stamp_physical_spec_scales(
                list((read_physical_param_specs(sim_ns) if isinstance(
                    sim_ns, dict) else None) or []), self._base_env)
        if self._physical_param_specs:
            logger.info("Agent declared %d physical params for system ID: %s",
                        len(self._physical_param_specs),
                        [s.name for s in self._physical_param_specs])
        if rules is None or specs is None:
            return None
        assert declared_features is not None, (
            "Agent did not declare RESIDUAL_FEATURES; "
            "synthesis output is incomplete.")
        residual_features = declared_features
        self._log_feature_set_diff(inferred_hint, residual_features,
                                   "inferred", "declared")
        logger.info("Agent synthesized %d rules, %d params.", len(rules),
                    len(specs))
        self._post_synthesis_loading(extra_paths, specs)
        return rules, specs, residual_features

    def _fit_params_after_synthesis(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
    ) -> None:
        """Deploy the agent's parameters; the harness never fits here."""
        if getattr(self, "_residual_env_cls", None) is not None and \
                not specs and not self._physical_param_specs:
            self._fitted_params.clear()
            self._last_fit_result = None
            self._fit_sse = float("inf")
        elif CFG.agent_sim_learn_declared_params_only:
            self._deploy_declared_params(rules, specs, base_pred_triples,
                                         residual_features)
        else:
            # The deployed model is the agent's own canonical sim.fit of
            # the final simulator.py: the values its GO/NO-GO check
            # validated, with the Laplace bundle the exploration ensemble
            # is calibrated from. Without a matching fit, deploy the
            # same carried/declared values as the probe. An edit or an
            # interrupted session must not trigger optimization.
            expected = [s.name for s in self._physical_param_specs
                        ] + [s.name for s in specs]
            published = None
            if self._probe_fit_state().get("fit_result") is not None:
                published = self._published_fit_for_file(
                    self._resolve_synthesis_paths().simulator_file, expected)
            fit_result: Optional[FitResult] = None
            if published is not None:
                fit_result, self._fit_sse, version = published
                if self._published_fit_is_pinned() or \
                        not math.isfinite(self._fit_sse):
                    # Deployed all the same - the cycle has no better
                    # model, and refusing outright would strand the run
                    # on a structure the next learn session is meant to
                    # fix - but never as a validated fit: the SSE is the
                    # value AT THE DECLARED INITS, and the certification
                    # gates that trust this model are trusting an
                    # unvalidated guess.
                    logger.error(
                        "UNVALIDATED MODEL: the agent's canonical sim.fit "
                        "(%s) ran no fit - no recorded motion segment was "
                        "explainable at any candidate parameters - so the "
                        "deployed %d params are the DECLARED INITS "
                        "(SSE at inits %.6f). Plans certified against "
                        "this model are not evidence about the real "
                        "environment; the next learn session must change "
                        "the simulator's structure, not its numbers.", version,
                        len(expected), self._fit_sse)
                else:
                    logger.info(
                        "Deploying the agent's published sim.fit (%s) of "
                        "the final simulator.py: %d params, SSE %.6f.",
                        version, len(expected), self._fit_sse)
                applied = self._probe_fit_state().get("applied_physical")
                if self._physical_param_specs and applied:
                    # The physics-margin sigma points come from the
                    # published fit (applying resets the points, so set
                    # them after).
                    self._apply_identified_physical_params(dict(applied))
                    if CFG.agent_sim_learn_param_uncertainty and int(
                            CFG.belief_joint_draws) <= 0:
                        self._identified_physical_sigma_points = list(
                            self._probe_fit_state().get("sigma_points") or [])
            else:
                self._deploy_unfitted_params(specs)
            if fit_result is not None:
                self._last_fit_result = fit_result
                self._fitted_params.clear()
                self._fitted_params.update(fit_result.point_estimate)
                logger.info("Fitted %d solver params.", len(specs))

        # Remember the specs (names + bounds) and rebuild the active-
        # experiment ensemble. Cheap and only consumed when info-seeking
        # exploration is enabled. Physical specs lead so the ordering
        # matches the joint rollout fit's theta layout.
        self._sync_subclass_parameters()
        self._param_specs = list(self._physical_param_specs) + list(specs)
        self._rebuild_param_ensemble()

    def _deploy_unfitted_params(self, specs: List[ParamSpec]) -> None:
        """Carry compatible values without attributing an old fit to an
        edit."""
        params = carry_over_params(self._fitted_params, specs)
        self._fitted_params.clear()
        self._fitted_params.update(params)
        physical = carry_over_params(self._identified_physical_params,
                                     self._physical_param_specs)
        if physical or self._identified_physical_params:
            self._apply_identified_physical_params(physical)
        self._identified_physical_sigma_points = []
        # Retain the historical published fit for provenance and file
        # reversions, but do not use its SSE or posterior for this model.
        self._last_fit_result = None
        self._fit_sse = float("inf")
        logger.info("UNFITTED for the current simulator.py: deploying "
                    "carried or declared parameter values; call sim.fit() "
                    "to estimate them.")

    def _physics_margin_points(
        self,
        applied: Dict[str, float],
        report: Dict[str, Dict[str, Any]],
        physical_specs: List[ParamSpec],
    ) -> List[Dict[str, float]]:
        """The physics sweep's +-1-sigma grid for ``applied``.

        Empty under ``agent_sim_learn_param_uncertainty`` False (ablations
        A6+A7 combined: point estimates only, so there is no width to
        sweep) - the
        single place that decides, so the joint fit, the published-fit
        deploy and the declared-params deploy cannot disagree.
        """
        if not CFG.agent_sim_learn_param_uncertainty:
            return []
        return physics_sigma_points(
            applied,
            report,
            physical_specs,
            num_points=CFG.agent_plan_validation_physics_margin_points)

    def _deploy_declared_params(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
    ) -> None:
        """Deploy the agent's declaration as the estimate (ablation A4).

        No fit runs. Every rule param takes its ``init_value``; the
        declared physical inits are applied to the planning env; the
        physics-margin grid spans each physical param's declared
        ``[lo, hi]`` box (the plausible interval) and the standing fit
        result is a uniform draw over every declared box, so the
        rule-parameter ensemble - the rule-param margin and the
        info-seeking disagreement score - samples the agent's intervals
        exactly as it would sample a posterior. The SSE is logged for
        the record only.
        """
        physical_specs = list(self._physical_param_specs)
        self._fitted_params.clear()
        self._fitted_params.update({s.name: s.init_value for s in specs})
        if physical_specs:
            applied = {s.name: s.init_value for s in physical_specs}
            self._apply_identified_physical_params(applied)
            self._identified_physical_sigma_points = \
                self._physics_margin_points(
                    applied, declared_interval_report(physical_specs),
                    physical_specs)
        self._last_fit_result = declared_interval_fit_result(
            physical_specs + list(specs),
            num_samples=_DECLARED_INTERVAL_SAMPLES,
            rng=self._rng)
        if base_pred_triples:
            self._fit_sse = self._oracle_param_sse(rules, base_pred_triples,
                                                   residual_features,
                                                   FIT_NOISE_SIGMA)
        else:
            self._fit_sse = float("inf")
        logger.info(
            "Parameter estimation disabled: deployed %d rule and %d "
            "physical params at their declared inits (SSE %.6f); margins "
            "and ensemble span the declared intervals.", len(specs),
            len(physical_specs), self._fit_sse)

    # ── Parameter fitting ────────────────────────────────────────

    def _oracle_param_sse(
        self,
        rules: List,
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
        noise_sigma: float,
    ) -> float:
        """Compute and log the SSE for oracle params (no fitting).

        ``self._fitted_params`` is assumed already populated with the
        oracle values. Returns the SSE. Physics-command rules only act
        through engine stepping, so they are scored with the rollout
        objective; recurrent (5-arg) rules cannot run per-transition, so
        when the loaded rules carry a latent block this dispatches to
        :meth:`_oracle_param_sse_recurrent`; otherwise it rolls each
        transition independently through the legacy 3-arg
        ``apply_rules``.
        """
        if getattr(self, "_residual_env_cls",
                   None) is not None or has_physics_rules(rules):
            return self._oracle_param_sse_rollout(rules, residual_features)
        if has_latent_rules(rules):
            return self._oracle_param_sse_recurrent(rules, base_pred_triples,
                                                    residual_features,
                                                    noise_sigma)
        oracle_sim_fn = lambda s, a, p: apply_rules(  # noqa: E731
            s, rules, p)
        sse = compute_sse(oracle_sim_fn, base_pred_triples,
                          self._fitted_params, residual_features)
        fit_ll = -0.5 * sse / (noise_sigma**2)
        logger.info("Oracle params - SSE: %.6f  log-likelihood: %.2f", sse,
                    fit_ll)
        for name, val in sorted(self._fitted_params.items()):
            logger.info("  %-30s  %.4f", name, val)
        log_sse_breakdown(oracle_sim_fn,
                          base_pred_triples,
                          self._fitted_params,
                          residual_features,
                          label="oracle")
        return sse

    # ── System identification (PHYSICAL_PARAM_SPECS) support ──────────

    def _create_initial_base_env(self, types: Set[Type]) -> Any:
        """Initial model substrate; scene-built arms supply only the robot."""
        del types
        return create_new_env(CFG.env,
                              do_cache=False,
                              use_gui=CFG.option_model_use_gui,
                              skip_residual_dynamics=True)

    def _make_planning_base_env(self, use_gui: bool = False) -> Any:
        """A fresh planning base env.

        The subclass model form (``_residual_env_cls`` set) runs an
        instance of the agent's subclass with its own
        ``_domain_specific_step`` firing (``skip_residual_dynamics``
        False); every rule-form or stock arm gets the fixed base-sim
        class with ``skip_residual_dynamics`` True, i.e. exactly
        ``create_new_env(CFG.env, skip_residual_dynamics=True)`` as
        before.
        """
        residual_env_cls = getattr(self, "_residual_env_cls", None)
        if residual_env_cls is not None:
            return residual_env_cls(use_gui=use_gui,
                                    skip_residual_dynamics=False)
        return create_new_env(CFG.env,
                              do_cache=False,
                              use_gui=use_gui,
                              skip_residual_dynamics=True)

    def _get_rollout_fit_env(self) -> Any:
        """Factory for the headless envs the rollout fit rolls out in.

        Returns a zero-arg callable; ``rollout_states`` invokes it once
        per rollout and disconnects the fresh env's PyBullet client
        afterwards. A fresh DIRECT-mode world per rollout is required
        for the fit to be deterministic at all: on a reused env the same
        theta produced SSE alternating 0.15/78 (run_20260708_213258),
        corrupting the grid seed and flooring the identifiability
        probe's same-theta noise floor - state-level resets cannot flush
        PyBullet solver internals (see ``rollout_states``). Measured
        overhead ~0.15 s per rollout on the domino env. It also keeps
        the fit's dynamics mutations away from the planning
        ``self._base_env`` (whose GUI variant additionally corrupts
        visual-shape state after a few hundred steps).
        """

        def _make() -> Any:
            return self._make_planning_base_env(use_gui=False)

        return _make

    def _rollout_fit_trajectories(
        self,
        residual_features: Optional[Dict[str, List[str]]] = None,
        traj_idxs: Optional[Sequence[int]] = None,
    ) -> List[RolloutTrajectory]:
        """Raw observed (states, actions) sequences for rollout matching.

        Unlike ``base_pred_triples`` these keep each trajectory whole, so
        momentum can accrue across steps in the free-running rollout.
        When ``residual_features`` is given (the fit's scored features)
        and ``CFG.code_sim_learning_rollout_truncate_settled`` is on,
        each trajectory's static tail is cut (see
        :func:`trajectory_prep.truncate_settled_tail`) so the fit scores
        the active cascade, not hundreds of settled steps of accumulated
        rollout divergence.

        ``traj_idxs`` restricts the source to those trajectories (same
        indexing as the synthesis session's ``trajectories`` list) -
        subsetting happens *before* truncation/segmentation so the
        indices the agent reasons about are the ones that apply. Raises
        ``ValueError`` on an out-of-range index.
        """
        source = self._fit_trajectories
        if traj_idxs is not None:
            bad = sorted(i for i in traj_idxs if not 0 <= i < len(source))
            if bad:
                raise ValueError(
                    f"traj_idxs {bad} out of range (0-{len(source) - 1})")
            source = [source[i] for i in traj_idxs]
        rollouts: List[RolloutTrajectory] = []
        for traj in source:
            if traj.actions and len(traj.states) == len(traj.actions) + 1:
                rollouts.append((list(traj.states), list(traj.actions)))
        if has_model_state(getattr(self, "_residual_env_cls", None)):
            # A rest point does not reset a hidden process. Keep the full
            # prefix so every candidate parameter point reconstructs memory.
            return rollouts
        if (residual_features is not None
                and CFG.code_sim_learning_rollout_truncate_settled
                and rollouts):
            truncated = [
                truncate_settled_tail(r, residual_features) for r in rollouts
            ]
            logger.info(
                "Rollout sysID: settled-tail truncation %s (tol=%g, "
                "margin=%d).", ", ".join(f"{len(r[1])}->{len(t[1])}"
                                         for r, t in zip(rollouts, truncated)),
                CFG.code_sim_learning_rollout_settle_tol,
                CFG.code_sim_learning_rollout_settle_margin)
            rollouts = truncated
        if (residual_features is not None
                and CFG.code_sim_learning_rollout_segment_on_rest
                and rollouts):
            # Multiple shooting: re-anchor at observed rest points so
            # chaotic divergence cannot compound across manipulation
            # phases, and trimming can drop a chaotic phase without
            # discarding the clean cascade next to it.
            segments: List[RolloutTrajectory] = []
            for r in rollouts:
                segments.extend(split_at_rest_points(r, residual_features))
            logger.info(
                "Rollout sysID: rest-point segmentation %d trajectories -> "
                "%d segments (lengths %s).", len(rollouts), len(segments),
                [len(a) for _s, a in segments])
            if segments:
                rollouts = segments
            else:
                logger.warning(
                    "Rollout sysID: segmentation found no scored motion; "
                    "keeping the whole trajectories.")
        return rollouts

    def _apply_identified_physical_params(
            self, identified: Dict[str, float]) -> None:
        """Publish identified physical params into the planning base env.

        The applied set exactly mirrors ``identified``: params applied
        by an earlier fit but absent here (e.g. dropped from a later
        artifact's PHYSICAL_PARAM_SPECS) are reverted to the env's
        registry defaults, because the env-side override is sticky per
        param and a stale value from a superseded fit would otherwise
        silently keep steering the planner. The override survives resets
        but not env recreation; ``_recreate_base_env`` re-applies from
        ``self._identified_physical_params``.
        """
        stale = set(self._identified_physical_params) - set(identified)
        if stale:
            info = self._base_env.get_physical_param_info()
            reverts = {
                name: float(info[name]["default"])
                for name in sorted(stale) if name in info
            }
            if reverts:
                self._base_env.apply_physical_param_overrides(reverts)
                logger.info(
                    "Reverted physical params dropped from the current "
                    "declaration to env defaults: %s",
                    {k: f"{v:.4f}"
                     for k, v in reverts.items()})
        self._identified_physical_params = dict(identified)
        # Margin points derive from a specific fit's posterior; any
        # (re)application resets them, and the joint-rollout caller
        # rebuilds them from its fresh report right after this call.
        self._identified_physical_sigma_points = []
        self._base_env.apply_physical_param_overrides(identified)
        self._sync_subclass_parameters()
        logger.info("Applied identified physical params to base env: %s",
                    {k: f"{v:.4f}"
                     for k, v in identified.items()})

    def fit_prior_anchors(
            self, physical_specs: Sequence[ParamSpec]) -> Dict[str, float]:
        """The prior centres of a rollout fit: the env-registry anchors, or
        under ``code_sim_learning_carry_posterior`` the carried posterior's
        most likely values where one exists (see the flag in settings).

        Shared by the harness fit and the ``sim.fit`` tool, so both fits
        start from the same belief.
        """
        anchors = physical_param_anchors(self._base_env, physical_specs)
        if not CFG.code_sim_learning_carry_posterior:
            return anchors
        carried = {
            s.name: self._carried_physical_prior[s.name]
            for s in physical_specs if s.name in self._carried_physical_prior
        }
        if carried:
            logger.info(
                "Rollout sysID: prior centres carried from the last applied "
                "fit: %s (registry anchors for the rest).",
                {k: f"{v:.4f}"
                 for k, v in carried.items()})
        anchors.update(carried)
        return anchors

    def note_carried_posterior(self, applied: Dict[str, float],
                               report: Dict[str, Dict[str, Any]]) -> None:
        """Record the values a fit just deployed as the next fit's prior
        centres (``code_sim_learning_carry_posterior``): only params whose
        verdict applied the fitted value, so an anchor fallback is never
        carried as a belief."""
        if not CFG.code_sim_learning_carry_posterior:
            return
        for name, value in applied.items():
            verdict = report.get(name, {}).get("verdict")
            if verdict is not None and verdict.applies_fitted:
                self._carried_physical_prior[name] = float(value)

    def note_fit_evidence(self, version_tag: str,
                          evidence: LaplaceEvidence) -> None:
        """Record a canonical fit's Laplace evidence under its simulator
        version, the history the report's version delta reads from."""
        self._fit_evidence_history[version_tag] = evidence.as_dict()

    def previous_fit_evidence(
            self, version_tag: str) -> Optional[Dict[str, LaplaceEvidence]]:
        """The most recently recorded evidence of a version other than
        ``version_tag``, as a one-entry dict, or None."""
        for tag in reversed(list(self._fit_evidence_history)):
            if tag != version_tag:
                return {
                    tag:
                    LaplaceEvidence.from_dict(self._fit_evidence_history[tag])
                }
        return None

    def _sync_tool_context(self) -> None:
        super()._sync_tool_context()
        self._tool_context.latent_tracking_available = \
            self._latent_tracking_available()

    def _latent_tracking_available(self) -> bool:
        """Whether episodes will run with an execution-time latent tracker (the
        loaded simulator threads a latent block)."""
        model_cls = getattr(self, "_residual_env_cls", None)
        if model_cls is not None:
            return has_model_state(model_cls)
        rules = self._residual_rules
        if not rules:
            return False
        return has_latent_rules(rules)

    def model_state_revision(self) -> Optional[Any]:
        """The native model and parameter point used to infer current
        memory."""
        cls = getattr(self, "_residual_env_cls", None)
        if not has_model_state(cls):
            return None
        values = {
            **self._fitted_params,
            **getattr(self, "_identified_physical_params", {})
        }
        return (getattr(self, "_residual_env_key", None)
                or cls, tuple(sorted(values.items())))

    def make_latent_tracker(
            self,
            params: Optional[Dict[str,
                                  float]] = None) -> Optional[LatentTracker]:
        """A fresh tracker over the current rules, params, and latent init (see
        ``code_sim_learning.latent_tracker``), or None for a fully- observable
        simulator.

        Parameters are passed by reference, as the belief simulator's
        closure does, so a later in-place fit is seen. ``params`` (one
        draw of the parameter belief) instead binds the tracker to a
        fixed copy of the deployed values overridden by the draw, so
        each joint draw carries the memory its own parameters imply.
        """
        model_cls = getattr(self, "_residual_env_cls", None)
        if params is not None:
            fixed = {
                **self._fitted_params,
                **getattr(self, "_identified_physical_params", {}),
                **params
            }
            if model_cls is not None:
                return make_subclass_latent_tracker(model_cls,
                                                    lambda: dict(fixed))
            return make_latent_tracker(self._residual_rules, fixed,
                                       self._latent_init)
        if model_cls is not None:
            return make_subclass_latent_tracker(
                model_cls, lambda: {
                    **self._fitted_params,
                    **getattr(self, "_identified_physical_params", {})
                })
        return make_latent_tracker(self._residual_rules, self._fitted_params,
                                   self._latent_init)

    def joint_draw_scope(self, params: Dict[str,
                                            float]) -> ContextManager[None]:
        """Learned predicates and rules read ``params`` over the deployed
        values for the duration (one joint draw's parameters)."""
        return self._rule_param_override_scope({
            **self._fitted_params,
            **params
        })

    # ── Partial-observability (latent) support ───────────────────
    # Reached only when the loaded rules use the recurrent 5-arg
    # signature (``has_latent_rules``). Legacy 3-arg simulators never
    # enter these paths, so fully-observable behavior is unchanged.

    def _group_triples_by_trajectory(
        self,
        triples: List[Tuple[State, Action, State]],
    ) -> List[List[Tuple[State, Action, State]]]:
        """Slice the flat triples list back into per-trajectory groups."""
        if not self._fit_trajectories:
            return []
        lengths = [len(t.actions) for t in self._fit_trajectories]
        if sum(lengths) != len(triples):
            logger.warning(
                "Trajectory-length mismatch (sum=%d vs triples=%d); "
                "skipping grouping.", sum(lengths), len(triples))
            return []
        groups: List[List[Tuple[State, Action, State]]] = []
        idx = 0
        for n in lengths:
            groups.append(triples[idx:idx + n])
            idx += n
        return groups

    def _oracle_param_sse_recurrent(
        self,
        rules: List,
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
        noise_sigma: float,
    ) -> float:
        """Oracle-param SSE via the recurrent (latent-threaded) rollout.

        Latent counterpart to :meth:`_oracle_param_sse`'s per-transition
        body. The per-feature ``log_sse_breakdown`` is per-transition
        and so omitted; the recurrent rollout already reports its SSE.
        """
        groups = self._group_triples_by_trajectory(base_pred_triples)
        if not groups:
            logger.warning("No trajectory groups for recurrent oracle SSE; "
                           "falling back to single-trajectory rollout.")
            groups = [base_pred_triples]
        sse = compute_sse_recurrent(rules, groups, self._fitted_params,
                                    self._latent_init, residual_features)
        fit_ll = -0.5 * sse / (noise_sigma**2)
        logger.info(
            "Oracle params (recurrent) - SSE: %.6f  log-likelihood: %.2f", sse,
            fit_ll)
        for name, val in sorted(self._fitted_params.items()):
            logger.info("  %-30s  %.4f", name, val)
        return sse

    def _oracle_param_sse_rollout(
        self,
        rules: List,
        residual_features: Dict[str, List[str]],
    ) -> float:
        """Oracle-param SSE via the free-running rollout objective.

        Physics-command counterpart to :meth:`_oracle_param_sse`'s per-
        transition body: command effects only exist through engine
        stepping, so the score free-runs the base sim with the rules in-
        the-loop (the same objective the rollout fit minimizes), scored
        on the declared features. Raw (unscaled) residuals, like the
        other oracle SSE paths.
        """
        rollouts = self._rollout_fit_trajectories(residual_features)
        if not rollouts:
            logger.warning("No complete trajectories for the rollout oracle "
                           "SSE; reporting inf.")
            return float("inf")
        sse = compute_rollout_sse(
            self._get_rollout_fit_env(),
            rollouts,
            self._fitted_params,
            residual_features,
            physical_names=[s.name for s in self._physical_param_specs]
            if getattr(self, "_residual_env_cls", None) is not None else [],
            rules=rules,
            latent_init=self._latent_init)
        logger.info(
            "Oracle params (rollout, physics-command rules) - "
            "SSE: %.6f over %d trajectories", sse, len(rollouts))
        for name, val in sorted(self._fitted_params.items()):
            logger.info("  %-30s  %.4f", name, val)
        return sse

    def materialise_latent(
        self,
        traj: LowLevelTrajectory,
    ) -> List[Optional[Dict[str, Any]]]:
        """Roll a trajectory through the rules; return per-step latent.

        Used by ``sim.predicates()`` so latent-aware predicates can be
        scored against meaningful latent values. Returned list aligns
        with ``traj.states``; entry ``i`` is the latent *before*
        predicates are evaluated at state ``i``. If no rules are loaded,
        every entry is ``None`` so latent-aware classifiers fall back to
        their default branch.
        """
        if getattr(self, "_residual_env_cls", None) is not None:
            tracker = self.make_latent_tracker()
            if tracker is None:
                return [None] * len(traj.states)
            return [
                tracker.attach(state,
                               None if i == 0 else traj.actions[i - 1]).latent
                for i, state in enumerate(traj.states)
            ]
        if not self._residual_rules:
            return [None] * len(traj.states)
        rules = self._residual_rules
        params = self._fitted_params
        latent = init_latent(self._latent_init, params)
        out: List[Optional[Dict[str, Any]]] = [dict(latent)]
        history: List[Tuple[State, Optional[Action]]] = []
        for i in range(len(traj.actions)):
            obs = observation_view(traj.states[i])
            action = traj.actions[i]
            history.append((obs, action))
            try:
                apply_rules_with_latent(obs, latent, history, rules, params)
            except Exception:  # pylint: disable=broad-except
                # If a rule crashes, fall back to None for the remaining
                # steps so predicate evaluation continues.
                out.extend([None] * (len(traj.states) - len(out)))
                return out
            out.append(dict(latent))
        return out

    def _build_latent_combined_simulator(
            self) -> Callable[[State, Action], State]:
        """Compose base env + recurrent rules; carry latent on state.latent.

        The latent block rides on the opaque ``State.latent`` field, so
        backtracking restores it per search node. The simulator reads
        ``state.latent`` on entry, threads it through the rules, and
        attaches the updated latent to the returned state. If
        ``state.latent`` is None (e.g. the very first state), falls back to
        ``init_latent``. The latent-free ``learned_simulator`` used by
        :meth:`_build_combined_simulator` is bypassed.
        """
        assert self._residual_rules is not None, (
            "_build_latent_combined_simulator called before rules loaded")
        rules: List = self._residual_rules
        latent_init = self._latent_init
        # Reference the dict (not its values) so fitted-param updates are
        # picked up by the closure live.
        params = self._fitted_params
        # Physics-command hand-off across sequential calls; see the
        # matching block in _build_combined_simulator.
        pending: Dict[str, Any] = {"state": None, "commands": []}

        def combined_simulate(state: State, action: Action) -> State:
            if pending["commands"]:
                if pending["state"] is not None and \
                        state.allclose(pending["state"]):
                    self._base_env.queue_residual_commands(pending["commands"])
                pending["state"], pending["commands"] = None, []
            # `state` is one sample of the augmented state: observable
            # features in `.data` + inferred latent dims in `.latent`.
            # Deep-copy the incoming latent so this call can't mutate the
            # caller's state and sibling branches at the same parent stay
            # independent. The latent nests a per-jug dict, so a shallow
            # ``dict(...)`` would still alias (and clobber) it.
            latent = (copy.deepcopy(state.latent) if state.latent is not None
                      else init_latent(latent_init, params))
            try:
                base_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logging.warning(
                    "PyBullet error in recurrent combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                base_state = self._base_env.simulate(state, action)
            # Repair features the backtracking reset couldn't round-trip
            # (e.g. bubbling_level derived from a hidden heat_level): the
            # base env's value is meaningless there, so restore the carried
            # value before the rules read it.
            self._restore_unreconstructible_residual_features(
                base_state, state)
            # Single-step history window; rules needing longer context
            # must accumulate it in ``latent``.
            obs = observation_view(base_state)
            history: List[Tuple[State, Optional[Action]]] = [(obs, action)]
            cmds = CommandBuffer()
            updates = apply_rules_with_latent(obs,
                                              latent,
                                              history,
                                              rules,
                                              params,
                                              cmds=cmds)
            next_state = (merge_updates(base_state, updates)
                          if updates else base_state)
            next_state.latent = latent
            if cmds:
                pending["state"], pending["commands"] = (next_state,
                                                         cmds.commands)
            return next_state

        return combined_simulate

    # ── Residual-feature inference ────────────────────────────────

    @staticmethod
    def _compute_base_pred_triples(
        obs_triples: List[Tuple[State, Action, State]],
        base_env: Any,
    ) -> List[Tuple[State, Action, State]]:
        """Replace each ``s_t`` with the base sim's one-step prediction."""
        return [(base_env.simulate(s, a), a, s_next)
                for s, a, s_next in obs_triples]

    @staticmethod
    def _log_feature_set_diff(
        a: Dict[str, List[str]],
        b: Dict[str, List[str]],
        a_label: str,
        b_label: str,
    ) -> None:
        """Log set-difference between two {type: [feats]} maps."""
        a_pairs = {(t, f) for t, fs in a.items() for f in fs}
        b_pairs = {(t, f) for t, fs in b.items() for f in fs}
        only_a = sorted(a_pairs - b_pairs)
        only_b = sorted(b_pairs - a_pairs)
        common = a_pairs & b_pairs
        logger.info(
            "Feature-set diff: %s vs %s (%d common, %d only-%s, %d only-%s)",
            a_label, b_label, len(common), len(only_a), a_label, len(only_b),
            b_label)
        if only_a:
            logger.info("  only in %s: %s", a_label, only_a)
        if only_b:
            logger.info("  only in %s: %s", b_label, only_b)

    def _make_evaluate_trajectory_fn(self) -> Any:
        """Build the ``evaluate_trajectory`` helper exposed in the synthesis
        exec namespace (next to ``is_goal_state``).

        The returned function is the task's reward model: it scores a
        concrete state sequence with the task's env-defined
        ``TaskEvaluator`` and returns the public triple (dict of
        reward/solved/note) - never the evaluator itself, and never the
        certificate's internal legitimacy verdict or reason (the agent
        infers the scoring rules from the stated objective and the
        outcomes it observes; goal-atom termination it can check itself
        via ``is_goal_state``). A certificate that needs physics (the
        domino counterfactual push) gets the approach's planning base
        env, i.e. the agent's belief simulator at its current fit, so
        on a sequence the agent assembled the verdict is a prediction
        of that model; ``note`` says what it replayed and on what.
        ``actions`` may be ``Action`` objects (labeled via their
        producing options), pre-built ``(option_name, object_names[,
        params])`` labels with ``None`` for an unlabeled transition, or
        ``None`` (no labels: a replaying certificate then falls back to
        its canonical action).
        """
        tasks = self._train_tasks

        def evaluate_trajectory(states: Sequence[State],
                                actions: Optional[Sequence[Any]] = None,
                                task_idx: int = 0,
                                physics_sweep: bool = False) -> Dict[str, Any]:
            """Score ``states`` with the task's reward model.

            ``states``: the sequence, ``states[t]`` before action ``t``.
            ``actions``: the recorded ``Action`` objects, or one label
            per transition, ``(option_name, (object_name, ...),
            (param, ...))``, ``None`` for a transition you do not
            attribute to a skill; omit for no labels. A rule that
            replays an action replays the labeled one with its
            parameters, and its canonical action when the sequence
            carries none. Returns ``{"reward", "solved", "note"}``;
            ``note`` says what a replaying rule simulated and on which
            substrate ("" when nothing was replayed). On a rollout of
            your simulator, or a sequence you assembled, the substrate
            is your belief simulator at its current fit.

            ``physics_sweep=True`` also scores the sequence at every
            point of the identified physical parameters' belief
            interval (the same grid ``sim.run(physics_sweep=True)``
            uses), each on a fresh env at that physics,
            and adds ``sweep``: the per-point verdicts and the fraction
            scored solved. A verdict that replays physics can flip
            across the interval; a sequence is certified only when it
            is scored a solve at every point. ``sweep`` is None with a
            note when no identified parameter carries a width.
            """
            if physics_sweep and not CFG.continual_uncertainty_decisions:
                raise ValueError("Explicit uncertainty sweeps are disabled.")
            if not 0 <= task_idx < len(tasks):
                raise ValueError(f"task_idx {task_idx} out of range "
                                 f"(0-{len(tasks) - 1}).")
            evaluator = tasks[task_idx].evaluator
            if evaluator is None:
                raise ValueError(
                    f"Train task {task_idx} defines no task evaluator.")
            if not states:
                raise ValueError("`states` must be a non-empty sequence.")
            step_options: Optional[Sequence[Any]] = None
            if actions is not None:
                acts = list(actions)
                if acts and isinstance(acts[0], Action):
                    step_options = step_option_labels(acts)
                else:
                    step_options = acts
            verdict = evaluate_states_with(evaluator,
                                           list(states),
                                           step_options,
                                           sim_env=getattr(
                                               self._option_model, "sim_env",
                                               None))
            result = {
                "reward": verdict["reward"],
                "solved": verdict["solved"],
                "note": verdict.get("note", ""),
            }
            if physics_sweep:
                result["sweep"] = self._sweep_evaluation(
                    evaluator, list(states), step_options)
            return result

        return evaluate_trajectory

    def _sweep_evaluation(self, evaluator: Any, states: List[State],
                          step_options: Optional[Sequence[Any]]) -> Any:
        """``evaluate_trajectory``'s ``physics_sweep``: the verdict at every
        physics-margin point on a fresh env at that physics, and the fraction
        scored solved; None with a note when there is nothing to sweep."""
        points = list(self._identified_physical_sigma_points)
        if not points:
            return None
        per_point: List[Dict[str, Any]] = []
        for point in points:
            with self._fresh_validation_env_scope(physical_overrides=point):
                try:
                    verdict = evaluate_states_with(evaluator,
                                                   states,
                                                   step_options,
                                                   sim_env=getattr(
                                                       self._option_model,
                                                       "sim_env", None))
                    entry = {
                        "params": dict(point),
                        "solved": bool(verdict["solved"]),
                        "reward": float(verdict["reward"]),
                        "note": str(verdict.get("note") or ""),
                    }
                except Exception as e:  # pylint: disable=broad-except
                    entry = {
                        "params": dict(point),
                        "solved": None,
                        "reward": None,
                        "note": f"verdict failed: {e}",
                    }
            per_point.append(entry)
        solved = sum(1 for p in per_point if p["solved"])
        return {
            "points": per_point,
            "solved_fraction": solved / len(per_point),
            "certified": solved == len(per_point),
        }

    def _simulator_load_namespace(self) -> Dict[str, Any]:
        """The names pre-injected when ``simulator.py`` is exec'd.

        Every loader of the agent's model file (the deploy, the probe's
        candidate, the readiness gate, the synthesis tools) starts from
        this namespace, so the base class the file subclasses is decided
        in one place. The stock arms inject the env's ``BaseSimulator``,
        the visible physics of the domain twin; an arm that hands the
        agent a different base overrides this.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.code_sim_learning.base_simulator import \
            base_simulator_class
        return {
            "np": np,
            "ParamSpec": ParamSpec,
            "BaseSimulator": base_simulator_class(getattr(CFG, "env", "")),
        }

    def _load_simulator_from_module_file(
        self,
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[Optional[List], Optional[List[ParamSpec]], Optional[Dict[
            str, List[str]]], Optional[Dict[str, Any]]]:
        """Load RESIDUAL_RULES, PARAM_SPECS, RESIDUAL_FEATURES from one file.

        Execs ``path`` once in a fresh namespace and returns ``(rules,
        specs, features, ns)``, where ``ns`` is that exec namespace so
        callers/subclasses can read extra exports (e.g. ``LATENT_INIT``)
        without re-execing. ``ns`` is ``None`` only when no exec
        happened (missing file or exec failure). ``rules``/``specs`` are
        ``None`` when ``RESIDUAL_RULES``/``PARAM_SPECS`` is absent (the
        caller treats that as failure); ``features`` may be ``None``
        independently (``RESIDUAL_FEATURES`` is then asserted by the
        caller).
        """
        if not os.path.isfile(path):
            logger.warning("No simulator file at %s.", path)
            return None, None, None, None

        ns: Dict[str, Any] = dict(self._simulator_load_namespace())
        ns["trajectories"] = trajectories or []
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            exec(code, ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to exec %s.", path, exc_info=True)
            return None, None, None, None

        rules, specs, features = read_simulator_components(ns)
        residual_env_cls = read_residual_env(ns)
        # A physics-only artifact (PHYSICAL_PARAM_SPECS with no residual
        # rules) or a subclass artifact (RESIDUAL_ENV, whose overridden
        # _domain_specific_step is the dynamics) is valid without
        # RESIDUAL_RULES/PARAM_SPECS: the base sim / subclass carries all
        # the dynamics once its parameters are identified, so rules/specs
        # default to empty.
        physics_only = read_physical_param_specs(ns) is not None
        allow_no_rules = physics_only or residual_env_cls is not None
        if rules is None:
            if not allow_no_rules:
                logger.warning("Simulator file %s missing RESIDUAL_RULES.",
                               path)
                return None, None, None, ns
            rules = []
        if specs is None:
            if not allow_no_rules:
                logger.warning("Simulator file %s missing PARAM_SPECS.", path)
                return None, None, None, ns
            specs = []
        # The subclass declares the features it owns on the class; use it
        # as the fallback when the file omits the module-level export.
        if residual_env_cls is not None and features is None:
            features = getattr(residual_env_cls, "RESIDUAL_FEATURES", None)

        kind = (" (subclass artifact)" if residual_env_cls is not None else
                " (physics-only artifact)" if physics_only else "")
        logger.info("Loaded %d rules, %d param specs from %s%s.", len(rules),
                    len(specs), path, kind)
        return rules, specs, features, ns

    # ── Static helpers ───────────────────────────────────────────

    def _base_sim_reference_paths(self) -> List[str]:
        """Agent-visible paths of the provisioned base-sim sources.

        The channel behind ``CFG.agent_sim_provide_base_sim_source``:
        the env declares its observable sim-core modules via
        ``get_base_sim_source_files()``, and sandbox setup copies them
        verbatim into ``reference/base_sim/`` at every session creation
        (see :meth:`_get_sandbox_reference_files`) - the visibility
        split is structural, so there is nothing to redact. Returns an
        empty list when the flag is off, the env declares no files, or
        the session has no sandbox (no file surface to read them from).
        """
        if not CFG.agent_sim_provide_base_sim_source:
            return []
        src_files = self._base_env.get_base_sim_source_files()
        if not src_files:
            logger.warning(
                "agent_sim_provide_base_sim_source is on, but env %s "
                "declares no base-sim source files; providing none.",
                type(self._base_env).__name__)
            return []
        names = [os.path.basename(rel) for rel in src_files]
        # Reference copies exist only in the local sandbox.
        if CFG.agent_sdk_use_local_sandbox:
            return [f"./reference/base_sim/{n}" for n in names]
        return []

    def _recreate_base_env(self) -> None:
        """Reconnect after a PyBullet physics-server crash."""
        self._rebuild_base_env("PyBullet physics client crashed; recreating "
                               "base env")

    def _rebuild_base_env(self, reason: str) -> None:
        """Dispose the current base env and build a fresh planning base env.

        Shared by the PyBullet-crash recovery
        (:meth:`_recreate_base_env`) and the subclass-form install
        (:meth:`_install_residual_env_cls`): both need a fresh env from
        :meth:`_make_planning_base_env` (the stock base sim, or an
        instance of the agent's ``RESIDUAL_ENV`` subclass when one is
        installed) with the identified physical params re-applied (the
        in-place override does not survive env recreation) and the
        option model's certificate env and probe substrate re-pointed at
        the new instance.
        """
        try:
            # dispose_env releases the secondary probe world too; the
            # domino override disposes it BEFORE the (possibly dead)
            # main client so a raise here cannot strand it.
            dispose_env(self._base_env)
        except Exception:  # pylint: disable=broad-except  # client may already be dead
            pass
        logging.warning("%s (use_gui=%s).", reason, CFG.option_model_use_gui)
        self._base_env = self._make_planning_base_env(
            use_gui=CFG.option_model_use_gui)
        # A fresh env comes up with built-in physics; re-assert any
        # identified physical params (the in-place override does not
        # survive env recreation).
        if self._identified_physical_params:
            info = self._base_env.get_physical_param_info()
            supported = {
                n: v
                for n, v in self._identified_physical_params.items()
                if n in info
            }
            cls = getattr(self, "_residual_env_cls", None)
            self._identified_physical_params = (carry_over_params(
                supported, list(cls.AGENT_PARAM_SPECS)) if cls is not None else
                                                supported)
            self._base_env.apply_physical_param_overrides(
                self._identified_physical_params)
        # The option model's transient certificate env rides on
        # _base_env; re-point it so probes don't run against the dead
        # client's stale physics overrides.
        if self._option_model is not None and \
                getattr(self._option_model, "sim_env", None) is not None:
            self._option_model.sim_env = self._base_env
        # The probe's combined substrate rides on the env instance too.
        self._base_env.probe_process_model_factory = \
            self._make_probe_process_model_factory()

    def _install_residual_env_cls(self,
                                  residual_env_cls: Optional[type],
                                  content_key: Optional[str] = None) -> None:
        """Install (or clear) the subclass model form's base-env class.

        The subclass model form (``simulator.py`` exports ``RESIDUAL_ENV``)
        supplies its dynamics by overriding ``_domain_specific_step``, so
        the planning base env must be an INSTANCE of the subclass
        (``skip_residual_dynamics`` False) rather than the stock base sim:
        then the deployed option model, the combined simulator and the
        rollout system-ID all step the agent's dynamics, and the fit
        identifies its ``AGENT_PARAM_SPECS``. Clearing it (``None``, the
        rule form and every stock arm) restores the stock base sim.

        ``content_key`` (the simulator.py SHA256) makes the install
        idempotent across the many re-execs of one file within a session:
        each exec defines a fresh class object, so an identity check would
        rebuild the base env every probe/fit call. When the key is
        unchanged the existing base env (whose class carries the identical
        code) is kept; a genuine edit changes the key and rebuilds. A
        ``None`` key forces a rebuild whenever the class presence changes
        (the once-per-cycle finalize/rehydrate paths).
        """
        cur_cls = getattr(self, "_residual_env_cls", None)
        cur_key = getattr(self, "_residual_env_key", None)
        unchanged = ((residual_env_cls is None and cur_cls is None) or
                     (residual_env_cls is not None and cur_cls is not None
                      and content_key is not None and content_key == cur_key))
        if unchanged:
            return
        self._residual_env_cls = residual_env_cls
        self._residual_env_key = content_key
        self._identified_physical_sigma_points = []
        self._probe_model_cache().clear()
        self._rebuild_base_env(
            "Installing subclass model base env" if residual_env_cls
            is not None else "Restoring stock base env (no subclass model)")
        self._sync_subclass_parameters()

    def _sync_subclass_parameters(self) -> None:
        """Keep existing predicate and sampler views on the native model point.

        These views deliberately hold only this dict, so serializing a
        predicate cannot accidentally serialize the complete approach.
        """
        if getattr(self, "_residual_env_cls", None) is None:
            return
        self._fitted_params.clear()
        self._fitted_params.update(
            getattr(self._base_env, "_agent_param_values"))

    @contextmanager
    def _fresh_candidate_validation_scope(
        self,
        physical_overrides: Optional[Dict[str,
                                          float]] = None) -> Iterator[None]:
        """Isolate the deployed candidate, without refitting or resampling."""
        provider = self._tool_context.probe_option_model_provider
        if provider is None:
            raise RuntimeError("No candidate model provider")
        # Loading may replace the base class and publish parameter values.
        # Do this before constructing the world, not inside the fresh scope.
        model = provider()
        with self._fresh_model_env_scope(model, physical_overrides):
            yield

    @contextmanager
    def _fresh_validation_env_scope(
        self,
        physical_overrides: Optional[Dict[str,
                                          float]] = None) -> Iterator[None]:
        """Isolate the solve-time option model's physics."""
        with self._fresh_model_env_scope(self._option_model,
                                         physical_overrides):
            yield

    @contextmanager
    def _fresh_model_env_scope(
        self,
        model: Any,
        physical_overrides: Optional[Dict[str,
                                          float]] = None) -> Iterator[None]:
        """Run the option model on a freshly constructed base env.

        ``physical_overrides`` (a physics-sweep point) is applied to the
        fresh env ON TOP of the identified params, so the rollout runs at
        a perturbed physics; the shared session env is never touched.

        Installed as ``ToolContext.validation_env_scope`` so the probe's
        trials and sweep rollouts each sample a fresh physics world. The
        shared ``_base_env``'s reset cannot reconstruct state exactly
        (solver warm-start state, velocity residuals, near-matching bodies
        skipped by the reconstruction diff - the same mechanism measured
        in :func:`rollout_states`), so
        repeats on it are correlated with each other and systematically
        offset from the fresh env the real episode runs in
        (run_20260717_182321: a placement swept 20/20 on the shared env
        validated 3/3, then missed the target on the real rollout).

        Swaps ``_base_env`` (the learned combined simulator reads it
        dynamically), the option model's ``sim_env`` (backs certificate
        probes), and - for the pre-learning model, whose simulator is the
        bound method ``_base_env.simulate`` - the model's ``_simulator``.
        Everything is restored and the fresh env disposed on exit,
        including the replacement env a mid-rollout PyBullet-crash
        recovery (``_recreate_base_env``) may have installed.
        """
        fresh = self._make_planning_base_env(use_gui=False)
        if self._identified_physical_params:
            fresh.apply_physical_param_overrides(
                self._identified_physical_params)
        # Candidate-declared values can differ from the last fitted values.
        # Preserve the actual deployed subclass parameters, not their init.
        deployed = getattr(self._base_env, "_agent_param_values", {})
        if deployed:
            fresh.apply_physical_param_overrides(dict(deployed))
        if physical_overrides:
            fresh.apply_physical_param_overrides(dict(physical_overrides))
        prev_env = self._base_env
        # Typed Any: sim_env and _simulator are dynamic attributes not on
        # _OptionModelBase.
        prev_sim = getattr(model, "_simulator", None)
        rebind_sim = getattr(prev_sim, "__self__", None) is prev_env
        prev_sim_env = getattr(model, "sim_env", None)
        # Certificate probes on the fresh env must judge on the same
        # combined substrate as the shared env's probes.
        fresh.probe_process_model_factory = getattr(
            prev_env, "probe_process_model_factory", None)
        self._base_env = fresh
        if rebind_sim:
            model._simulator = fresh.simulate  # pylint: disable=protected-access
        if prev_sim_env is not None:
            model.sim_env = fresh
        try:
            yield
        finally:
            current = self._base_env
            self._base_env = prev_env
            if rebind_sim:
                model._simulator = prev_sim  # pylint: disable=protected-access
            if prev_sim_env is not None:
                model.sim_env = prev_sim_env
            if current is not prev_env:
                try:
                    dispose_env(current)
                except Exception:  # pylint: disable=broad-except
                    pass  # client already dead (crashed mid-rollout)

    def _restore_unreconstructible_residual_features(
            self, base_state: State, prev_state: State) -> None:
        """Restore residual features the base env's reset couldn't round-trip.

        When the option model backtracks (jumps to a non-current node), the
        base PyBullet env reconstructs the State from observables only, so a
        feature derived from a hidden sim-feature (e.g. ``bubbling_level``,
        projected from a hidden ``heat_level``) comes back at its default
        (0) instead of its carried value. The learned model *owns* those
        features, so the base value is meaningless; overwrite ``base_state``
        with the value carried in ``prev_state`` before the rules read it.

        Scoping is the key to not breaking co-owned features: restore only
        the intersection of (a) the env's reported unreconstructible set for
        this step and (b) the declared ``RESIDUAL_FEATURES``. A kinematic,
        base-reconstructible feature that a robot legitimately moves (e.g. a
        wind-blown ball's ``x, y`` in the fans env) round-trips through the
        reset, so it never enters the env's set and is left to the base sim.
        On sequential rollouts the env's set is empty, so this is a no-op.
        """
        lossy = getattr(self._base_env, "_last_unreconstructible_features",
                        None)
        if not lossy or not self._residual_features:
            return
        for obj, feat in lossy:
            if feat in self._residual_features.get(obj.type.name, []) \
                    and obj in prev_state.data:
                base_state.set(obj, feat, prev_state.get(obj, feat))

    def _build_combined_simulator(
        self,
        learned_simulator: LearnedSimulator,
    ) -> Callable[[State, Action], State]:
        """Compose base env with learned step-level dynamics.

        Captures ``self`` so the closure can recreate ``_base_env`` and
        retry once on a PyBullet crash (common on macOS Metal + GUI).
        When the loaded rules carry a latent block (partial
        observability), delegates to
        :meth:`_build_latent_combined_simulator`, which threads
        ``state.latent`` through the recurrent rules instead of the
        latent-free ``learned_simulator``.
        """
        if getattr(self, "_residual_env_cls", None) is not None:
            self._sync_subclass_parameters()

            def native_simulate(state: State, action: Action) -> State:
                try:
                    return self._base_env.simulate(state, action)
                except pybullet.error:
                    self._recreate_base_env()
                    return self._base_env.simulate(state, action)

            return native_simulate
        if has_latent_rules(self._residual_rules or []):
            return self._build_latent_combined_simulator()

        # Physics commands emitted by the rules at step t act during the
        # substeps of step t+1 (the same cadence a hidden
        # _domain_specific_step's applyExternalForce has). They are held
        # here keyed to the exact state they were computed for and only
        # queued on the env when the next call continues from that state
        # - a planner backtrack to a different state silently drops
        # them, exactly like a reset drops an env-applied force.
        pending: Dict[str, Any] = {"state": None, "commands": []}

        def combined_simulate(state: State, action: Action) -> State:
            if pending["commands"]:
                if pending["state"] is not None and \
                        state.allclose(pending["state"]):
                    self._base_env.queue_residual_commands(pending["commands"])
                pending["state"], pending["commands"] = None, []
            try:
                base_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logging.warning(
                    "PyBullet error in combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                base_state = self._base_env.simulate(state, action)
            self._restore_unreconstructible_residual_features(
                base_state, state)
            cmds = CommandBuffer()
            updates = learned_simulator.predict_step(base_state, cmds)
            next_state = (merge_updates(base_state, updates)
                          if updates else base_state)
            if cmds:
                # Keyed to the state the planner will hand back on the
                # next sequential call (the merged one, not base_state).
                pending["state"], pending["commands"] = (next_state,
                                                         cmds.commands)
            return next_state

        return combined_simulate

    def _make_probe_process_model_factory(
            self) -> Optional[Callable[[], Callable[[State, Action], State]]]:
        """Per-replay process-model steppers for certificate probes.

        Stamped on the belief env as
        ``BaseEnv.probe_process_model_factory`` so physics-replaying
        task-evaluator certificates judge plans on the same combined
        substrate the option model plans on (see
        :meth:`_build_combined_simulator`): each probe attempt gets a
        fresh stepper that applies the current rules to every post-step
        state (threading a fresh latent for recurrent rules, like
        :meth:`_build_latent_combined_simulator` does per plan step).
        Reads the live ``self._fitted_params`` dict so in-session
        ``sim.fit`` updates reach the probe, matching the combined
        simulator's closure. Returns None (probe stays base-only) until
        rules exist. Limitation: residual features the env cannot
        round-trip through ``_set_state`` (hidden-derived, e.g. a
        ``bubbling_level``) are not restored inside the probe replay -
        no env with a physics-replaying certificate declares any today.
        Physics commands are likewise not replayed here (the steppers
        run the rules with a throwaway buffer): the only consumer is
        the domino cascade probe, whose GT dynamics are command-free.
        """
        if getattr(self, "_residual_env_cls", None) is not None:
            return None
        rules = getattr(self, "_residual_rules", None)
        if not rules:
            return None
        params = self._fitted_params
        if has_latent_rules(rules):
            latent_init = self._latent_init

            def make_latent_stepper() -> Callable[[State, Action], State]:
                latent = init_latent(latent_init, params)

                def step(state: State, action: Action) -> State:
                    obs = observation_view(state)
                    history: List[Tuple[State,
                                        Optional[Action]]] = [(obs, action)]
                    updates = apply_rules_with_latent(obs, latent, history,
                                                      rules, params)
                    return merge_updates(state, updates) if updates else state

                return step

            return make_latent_stepper

        def make_stepper() -> Callable[[State, Action], State]:

            def step(state: State, action: Action) -> State:
                del action  # 3-arg rules read only the state
                updates = apply_rules(state, rules, params)
                return merge_updates(state, updates) if updates else state

            return step

        return make_stepper

    def _physical_params_prompt_section(self) -> str:
        """Markdown for the optional PHYSICAL_PARAM_SPECS (system-ID) block.

        Built from the base env's revealed parameter menu
        (``get_physical_param_info``); empty when the env reveals none,
        so non-parameterized envs never see the feature mentioned.
        """
        info: Dict[str, Dict[str, Any]] = {}
        # getattr chain (not plain attribute access) so the prompt still
        # renders on instances without a base env (see the bare-instance
        # rendering tests in test_agent_sim_prompt_formatting.py).
        base_env = getattr(self, "_base_env", None)
        getter = getattr(base_env, "get_physical_param_info", None)
        if callable(getter):
            info = getter() or {}
        return render_physical_params_section(info)
