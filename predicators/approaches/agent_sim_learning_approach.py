"""Agent sim-learning approach: learns a simulator program online.

Extends AgentModelBasedApproach to learn residual dynamics via an
agent-synthesized step-level simulator with parameterized process
rules. Parameters are fitted via emcee ensemble MCMC (training.py).

The approach creates a base oracle (PyBullet with process
dynamics disabled) and composes it with the learned step-level
dynamics into a single simulator function, plugged into a standard
_OracleOptionModel for true per-step interleaving.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_learning --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --num_online_learning_cycles 5 --explorer agent_model_free
"""

import copy
import dataclasses
import hashlib
import inspect
import logging
import os
import subprocess
from contextlib import contextmanager
from typing import Any, Callable, Collection, Dict, FrozenSet, Iterator, \
    List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
import pybullet
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    max_session_log_number, query_fatal_error
from predicators.agent_sdk.tools import SYNTHESIS_TOOL_NAMES, \
    _SnapshotTarget, create_synthesis_tools, evaluate_states_with, \
    finalize_versioned_snapshot, make_write_snapshot_hook
from predicators.agent_sdk.tools.digests import render_options_digest, \
    render_trajectory_digest, render_types_digest
from predicators.approaches.agent_model_based_approach import \
    AgentModelBasedApproach
from predicators.approaches.sampler_learning_mixin import SamplerLearningMixin
from predicators.approaches.synthesis_validation import \
    build_candidate_option_model
from predicators.code_sim_learning.active_experiment import laplace_ensemble, \
    mean_bernoulli_entropy, perturbation_ensemble, \
    posterior_subsample_ensemble
from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    declared_interval_fit_result, declared_interval_report
from predicators.code_sim_learning.fitting import FIT_NOISE_SIGMA, \
    compute_sse, compute_sse_recurrent, fit_rule_parameters, \
    fit_rule_parameters_latent, log_param_changes, log_sse_breakdown
from predicators.code_sim_learning.identifiability import Verdict, \
    format_identifiability, physics_sigma_points
from predicators.code_sim_learning.latent_tracker import LatentTracker, \
    make_latent_tracker
from predicators.code_sim_learning.orchestrator import run_rollout_sysid
from predicators.code_sim_learning.physical_sysid import fit_params_rollout
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    dispose_env, physical_param_anchors
from predicators.code_sim_learning.rollout_objective import compute_rollout_sse
from predicators.code_sim_learning.trajectory_prep import \
    split_at_rest_points, truncate_settled_tail
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, apply_rules_with_latent, has_latent_rules, \
    has_physics_rules, init_latent, iter_feature_residuals, merge_updates, \
    observation_view, read_latent_init, read_physical_param_specs, \
    read_simulator_components, stamp_physical_spec_scales
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_simulator
from predicators.option_model import _OptionModelBase, _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import Action, Dataset, DerivedPredicate, \
    GroundAtom, InteractionResult, LowLevelTrajectory, ParameterizedOption, \
    Predicate, State, Task, Type, step_option_labels

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


class AgentSimLearningApproach(SamplerLearningMixin, AgentModelBasedApproach):
    """Bilevel planning with a learned step-level simulator.

    During online learning:
    1. Collect trajectories (inherited from AgentModelBasedApproach)
    2. Segment into option-level transitions
    3. Synthesize parameterized residual rules via Claude agent
    4. Fit rule parameters via emcee ensemble MCMC
    5. Compose with base oracle into a combined simulator
    6. Build _OracleOptionModel with the combined simulator

    During solving:
    - Uses the learned model for plan validation in backtracking
      refinement.

    Per-skill sampler learning (mode resolution, synthesis session
    plumbing, loading) lives in :class:`SamplerLearningMixin`.
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
        self._base_env = create_new_env(CFG.env,
                                        do_cache=False,
                                        use_gui=CFG.option_model_use_gui,
                                        skip_residual_dynamics=True)
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
        # Capture-validation rollouts each run on a freshly constructed env
        # (see ToolContext.validation_env_scope): repeats on the shared
        # ``_base_env`` are correlated across resets, so only fresh envs
        # sample the distribution the real episode will.
        self._tool_context.validation_env_scope = \
            self._fresh_validation_env_scope
        # Physics-margin points for the capture gate (+-1 posterior sigma
        # of the latest applied fit): a callable so the tool always sees
        # the current fit, not the one deployed when the session opened.
        # Under agent_sim_learn_param_uncertainty False the points are
        # never built (see _physics_margin_points), so this returns [].
        self._tool_context.physics_margin_provider = \
            lambda: list(self._identified_physical_sigma_points)
        # Rule-parameter margin points for the capture gate: the
        # calibrated ensemble the info-seeking explorer scores with
        # (posterior subsample / Laplace / jitter, see
        # _select_param_ensemble) doubles as the uncertainty sweep over
        # LEARNED rule constants - a submission must survive every
        # member, not just the fitted point estimate. Callables so the
        # gate always sees the latest fit's ensemble.
        self._tool_context.rule_param_margin_provider = \
            lambda: [dict(m) for m in self._param_ensemble]
        self._tool_context.rule_param_override_scope = \
            self._rule_param_override_scope
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
        # Full result used for ensemble calibration. Usually this is the
        # solver fit; when info-seeking runs extra MCMC, it is the
        # exploration-only posterior. ``None`` after an oracle-param run.
        self._last_fit_result: Optional[FitResult] = None
        self._fit_sse: float = float("inf")
        self._learning_mode: bool = False
        # Snapshot tags of the most recent simulator / predicates files
        # committed by the synthesis agent, used to stamp newly collected
        # online trajectories with their source-version provenance
        # (consumed in the next learn-phase prompt).
        self._current_simulator_version: Optional[str] = None
        self._current_predicates_version: Optional[str] = None
        self._init_sampler_learning_state()
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
        # System identification: PHYSICAL_PARAMS export (agent-declared
        # sparse subset of self._base_env.get_physical_param_info()),
        # identified values applied in place to the base env. The rollout
        # fit itself builds a fresh headless env per rollout (see
        # _get_rollout_fit_env), never touching the planning base env.
        self._physical_param_specs: List[ParamSpec] = []
        self._identified_physical_params: Dict[str, float] = {}
        # +-1-posterior-sigma perturbations of the applied params (the
        # capture gate's physics-margin points). Set only by the joint
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
        # Final per-cycle fit history for the cross-cycle consistency
        # check: name -> (map_value, posterior_std_fit_space, scale).
        # Mutually-incompatible confident fits across cycles are the
        # signature of an overconfident probe; flagged, and the verdict
        # downgraded, rather than silently trusted.
        self._sysid_fit_history: Dict[str, Tuple[float, float, str]] = {}
        # A rejected (INCONSISTENT) fit awaiting confirmation:
        # name -> (map_value, posterior_std_fit_space). If the NEXT
        # cycle's independent fit lands within the consistency band of
        # the pending value, the jump is accepted as real (two
        # independent fits agree); until then the trusted history value
        # holds. Without this, a genuinely-updated fit would read
        # INCONSISTENT against stale history forever.
        self._sysid_pending_fit: Dict[str, Tuple[float, float]] = {}
        # The applied physical params as of the last CYCLE-LEVEL fit -
        # the reference the INCONSISTENT hold policy reverts to.
        # Deliberately not _identified_physical_params: the agent's
        # in-session sim.fit calls mutate that dict, so "hold the
        # currently-applied value" was a no-op that held the very fit
        # it refused to trust (run_20260724_232411 seed2 cycle 2:
        # "holding the currently-applied 0.6267" - 0.6267 WAS the
        # distrusted new fit, applied minutes earlier in-session).
        self._cycle_applied_physical: Dict[str, float] = {}
        # Agent-facing digest of the latest rollout fit (unexplainable
        # segments, unidentified/insensitive params, cross-cycle
        # conflicts); surfaced to the explorer as experiment objectives.
        self._last_sysid_diagnostics: str = ""

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
        cfg_override = getattr(CFG, "agent_sim_learn_kept_predicates_names",
                               None)
        if cfg_override:
            return frozenset(cfg_override)
        return self.KEPT_INITIAL_PREDICATE_NAMES

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

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:
            return self._build_synthesis_system_prompt()
        prompt = super()._get_agent_system_prompt()
        base_sim_refs = self._base_sim_reference_paths()
        if base_sim_refs:
            ref_listing = "\n".join(f"  - {r}" for r in base_sim_refs)
            prompt += (
                "\n\n## Base Simulator Source\n"
                "The environment simulator's own source code is "
                "available (read-only):\n"
                f"{ref_listing}\n"
                "It covers the observable sim core: scene geometry and "
                "constants, body construction, physics stepping, and "
                "state read/write. It deliberately omits the hidden "
                "domain-specific dynamics, task generation, and goal "
                "semantics. Read it to ground your spatial and physical "
                "reasoning (dimensions, contact geometry, actuation) "
                "instead of guessing from images or trial and error.\n")
        return prompt

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
        ``run_python``) attached to ``ctx.extra_mcp_tools`` inside
        :meth:`_synthesize_with_agent`. The mixin asserts the attached
        instances and this list agree. Fitting, residual reports, and
        plan validation are NOT tools: they live on the ``sim`` probe
        (``sim.fit`` / ``sim.residuals`` / ``sim.refine`` /
        ``sim.run``) inside ``run_python``.

        No inspect tools: the type/option digests are injected into the
        learn message (see :meth:`_build_synthesis_learn_message`) and
        trajectory access lives in ``run_python`` (``trajectories`` +
        ``describe_trajectory``). The probe rides inside that same
        ``run_python`` namespace as ``sim`` (one exec namespace per
        session - a helper defined next to the data is visible to probe
        sweeps; the solve-phase instance of the tool is not built when
        this one is attached). In the
        agent-synthesis session the probe runs against the CANDIDATE
        simulator.py via ctx.probe_option_model_provider (installed in
        _synthesize_with_agent); in the oracle-sim-program sampler
        session no provider is installed and the probe falls back to
        ctx.option_model, which there IS the deployed belief model.
        """
        names: List[str] = list(SYNTHESIS_TOOL_NAMES)
        return names

    # ── Subclass hooks ──────────────────────────────────────────
    # Default implementations are no-ops so subclasses can add
    # predicate-invention (or other) extensions without copying
    # _synthesize_with_agent.

    def _learning_cycle_index(self) -> int:
        """0-based cycle index used in versioned snapshot filenames.

        Matches main.py's "ONLINE LEARNING CYCLE i" numbering exactly:
        ``_online_learning_cycle`` is incremented before this class's
        online simulator learn runs, so subtracting 1 recovers the
        cycle the session belongs to. The offline (pre-cycle-0) learn
        yields -1, which the snapshot/journal formatters render as
        "offline" - keeping it distinct from cycle 0's online pass.
        """
        return self._online_learning_cycle - 1

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

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        """Return text to append to the agent's first synthesis message.

        Under ``CFG.partially_observable`` this is the short partial-
        observability note; subclasses that override MUST chain via
        ``super()`` so the note survives.
        """
        del extra_paths
        if CFG.partially_observable:
            return learn_prompts.render_partial_observability_message()
        return ""

    def _extra_synthesis_system_prompt_sections(self) -> List[str]:
        """Sections a subclass adds to the synthesis system prompt.

        Inserted after the validation guidance and before the recurrent
        rules tutorial (partial observability) and the plan format.
        Subclasses that override MUST chain via ``super()``.
        """
        return []

    def _extra_synthesis_latent_sections(self) -> List[str]:
        """Sections a subclass adds after the recurrent-rules tutorial.

        Only rendered under ``CFG.partially_observable``.
        """
        return []

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

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        super().learn_from_offline_dataset(dataset)
        self._learn_simulator(self._get_all_trajectories())
        # The single post-offline checkpoint, AFTER the simulator learn
        # (the base hook is a no-op for this class, see below).
        self.save(None)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # Capture the index BEFORE super() increments it: the checkpoint
        # below must be the one this cycle's filename denotes.
        cycle = self._online_learning_cycle
        super().learn_from_interaction_results(results)
        self._learn_simulator(self._get_all_trajectories())
        # The single per-cycle checkpoint, AFTER this cycle's simulator
        # learning, so a resume never re-pays a completed learn and never
        # mistakes a pre-learn file for a completed cycle. (The base hook
        # that would have saved pre-learn is a no-op for this class.)
        self.save(cycle)

    def _checkpoint_after_offline_learning(self) -> None:
        """No-op: this class checkpoints after its own simulator learn."""

    def _checkpoint_after_interaction_results(self, cycle: int) -> None:
        """No-op: this class checkpoints after its own simulator learn."""
        del cycle

    # ── Checkpointing ────────────────────────────────────────────
    # The base checkpoint (AgentModelFreeApproach.save/load) persists
    # the datasets + cycle counter. This approach's real state is split
    # between plain fitted values (pickled below) and the sandbox
    # artifacts the agent wrote (simulator.py / predicates.py / ...),
    # which are embedded as file CONTENTS - run dirs are minted per run
    # and pruned, so a path reference to the old run's sandbox would be
    # fragile. Closures (_residual_rules, _learned_simulator, the option
    # model, learned predicates/samplers) are never pickled: they are
    # rebuilt from the restored files in _rehydrate_from_artifacts.

    _save_suffix: str = "AgentSimLearner"

    _CHECKPOINT_SANDBOX_FILES: Tuple[str,
                                     ...] = ("simulator.py", "predicates.py",
                                             "samplers.py",
                                             "ground_samplers.py", "notes.md",
                                             "journal.md", "attempts.md",
                                             "strategy.md",
                                             "open_questions.md")
    _CHECKPOINT_SANDBOX_DIRS: Tuple[str, ...] = ("simulator_versions",
                                                 "predicates_versions",
                                                 "samplers_versions")
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
            "param_ensemble":
            list(self._param_ensemble),
            "identified_physical_params":
            dict(self._identified_physical_params),
            "identified_physical_sigma_points":
            list(self._identified_physical_sigma_points),
            "sysid_fit_history":
            dict(self._sysid_fit_history),
            "residual_features":
            dict(self._residual_features),
            "current_simulator_version":
            self._current_simulator_version,
            "current_predicates_version":
            self._current_predicates_version,
            "current_samplers_version":
            self._current_samplers_version,
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
        # In-place update: _ParamsView holders (invented predicate and
        # sampler closures) alias this exact dict object.
        self._fitted_params.clear()
        self._fitted_params.update(save_dict.get("fitted_params") or {})
        self._fit_sse = save_dict.get("fit_sse", float("inf"))
        self._param_specs = list(save_dict.get("param_specs") or [])
        self._physical_param_specs = list(
            save_dict.get("physical_param_specs") or [])
        self._last_fit_result = save_dict.get("last_fit_result")
        self._param_ensemble = list(save_dict.get("param_ensemble") or [])
        self._identified_physical_params = dict(
            save_dict.get("identified_physical_params") or {})
        self._sysid_fit_history = dict(
            save_dict.get("sysid_fit_history") or {})
        self._residual_features = dict(
            save_dict.get("residual_features") or {})
        self._current_simulator_version = save_dict.get(
            "current_simulator_version")
        self._current_predicates_version = save_dict.get(
            "current_predicates_version")
        # pylint: disable-next=attribute-defined-outside-init
        # (initialized by SamplerLearningMixin's init hook)
        self._current_samplers_version = save_dict.get(
            "current_samplers_version")
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
        already- restored ``_fitted_params``), then samplers and the
        ensemble.
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
        if not rules or specs is None:
            logger.warning(
                "Restored simulator.py failed to load; continuing with "
                "the initial option model (the next learn cycle will "
                "rebuild it).")
            self._rehydrate_extra_artifacts(paths.base)
            return
        self._residual_rules = rules
        if declared_features:
            self._residual_features = declared_features
        self._latent_init = (read_latent_init(sim_ns) if isinstance(
            sim_ns, dict) else None)
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
        if self._samplers_enabled():
            sampler_paths = self._sampler_paths(paths.base)
            self._synthesized_samplers = self._load_samplers_from_module_file(
                sampler_paths["samplers_file"])
        self._rebuild_param_ensemble()
        logger.info(
            "Rehydrated learned simulator from checkpoint artifacts "
            "(%d rules, %d fitted params, %d learned predicates, "
            "%d samplers).", len(rules), len(self._fitted_params),
            len(getattr(self, "_learned_predicates", set()) or set()),
            len(self._synthesized_samplers))

    def _learn_simulator(self, trajectories: List[LowLevelTrajectory]) -> None:
        """Synthesize rules, fit parameters, and build the option model."""
        # Cache for recurrent fitting: lets _group_triples_by_trajectory
        # slice the flat base_pred_triples back into per-trajectory chunks
        # (latent threads within a trajectory, not across). Harmless for
        # fully-observable (legacy) simulators, which never regroup.
        self._fit_trajectories = list(trajectories)
        # Dumped HERE, where the data arrives, rather than only inside the
        # sysID fit: a cycle where the agent declines to fit is exactly the
        # one worth post-morteming, and that is the branch that never ran.
        # run_20260817_171402 declined on a sweep that returned one identical
        # SSE for every value of five parameters, and left nothing on disk to
        # explain it -- the episode had to be written off.
        self._persist_fit_trajectories("recorded")
        # New data invalidates the memoized explainability verdicts and
        # the memoized whole fits.
        self._explainability_cache.clear()
        self._sysid_fit_cache.clear()
        # Decide how samplers are obtained this cycle: ground-truth (if
        # requested and available for the env) else agent synthesis. GT
        # samplers are static, so install them up front, independent of
        # whether simulator learning runs below (it is skipped when there
        # are no step transitions and no oracle sim program to fall
        # back on, e.g. when every demo failed).
        self._maybe_install_oracle_samplers()
        # Two parallel triple lists drive the rest of this method:
        # * obs_triples       - raw (s_t, a, s_{t+1}) from the data.
        # * base_pred_triples - same triples but s_t replaced by the
        #   base sim's one-step prediction. The rules run on top of that
        #   prediction; SSE compares against s_{t+1}.
        obs_triples = self._extract_obs_triples(trajectories)
        if (not obs_triples and not CFG.agent_sim_learn_oracle_sim_program
                and not CFG.agent_sim_learn_zero_shot):
            logger.warning("No step transitions; skipping simulator learning.")
            return
        if obs_triples:
            # Headless env for the pre-compute: reusing the GUI base_env
            # corrupts its visual-shape state after a few hundred steps.
            fit_env = create_new_env(CFG.env,
                                     do_cache=False,
                                     use_gui=False,
                                     skip_residual_dynamics=True)
            logger.info("Pre-computing base states for %d transitions.",
                        len(obs_triples))
            try:
                base_pred_triples = self._compute_base_pred_triples(
                    obs_triples, fit_env)
            finally:
                # This env is rebuilt every learning cycle; dispose it
                # (main client AND any secondary probe world) or each
                # cycle leaks a full physics world (~145MB for the
                # domino env).
                dispose_env(fit_env)
            inferred_hint = self._infer_residual_features_from_scan(
                obs_triples, base_pred_triples)
            logger.info("Residual features (data-driven hint): %s",
                        inferred_hint)
        elif CFG.agent_sim_learn_oracle_sim_program:
            # The oracle sim program is data-free (rules and parameter
            # inits come from get_gt_simulator), so a run whose every
            # demo failed still gets a working option model; the fit
            # below degrades to the declared inits.
            logger.warning("No step transitions; loading oracle sim "
                           "program without data.")
            base_pred_triples = []
            inferred_hint = {}
        else:
            # Zero-shot synthesis (ablation A2): the session runs with
            # nothing recorded, so the artifacts come from the task
            # description, the scene and the agent's own knowledge; the
            # params deploy at their declared inits.
            logger.info("Zero-shot synthesis: no step transitions; the "
                        "agent writes its artifacts without data.")
            base_pred_triples = []
            inferred_hint = {}

        self._synthesize_with_agent(trajectories, obs_triples,
                                    base_pred_triples, inferred_hint)

        if self._residual_rules is not None and self._fitted_params:
            rules, params = self._residual_rules, self._fitted_params
            self._learned_simulator = LearnedSimulator(
                step_fn=lambda s, c, _r=rules, _p=params:  # type: ignore[misc]
                apply_rules(s, _r, _p, cmds=c),
                name="agent_synthesized")
        elif self._learned_simulator is None:
            logger.warning("Synthesis produced no simulator, skipping.")
            return

        combined_sim = self._build_combined_simulator(self._learned_simulator)
        self._option_model = self._build_option_model(combined_sim)
        logger.info("Built learned option model (SSE: %.6f).", self._fit_sse)

        # When the simulator came from the oracle short-circuit no agent
        # session ran above, so per-skill samplers (if enabled) get their
        # own session here, after the option model is built so the
        # session's probe (sim.refine) has a working simulator. When
        # the agent *did* synthesize the simulator, samplers already rode
        # along in that session and this is skipped.
        if self._do_synthesize_samplers and \
                CFG.agent_sim_learn_oracle_sim_program:
            if base_pred_triples:
                self._synthesize_samplers_standalone(trajectories,
                                                     base_pred_triples,
                                                     inferred_hint)
            else:
                logger.warning("No step transitions; skipping standalone "
                               "sampler synthesis.")

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
        # sandbox and captures are accepted on the pure rules only.
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
        """
        self._fitted_params.clear()
        self._fitted_params.update(params)
        digest = None
        if os.path.isfile(simulator_file):
            with open(simulator_file, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        state = self._probe_fit_state()
        state["digest"] = digest
        state["version"] = version_tag
        state["fit_result"] = fit_result
        state["sse"] = sse
        state["applied_physical"] = dict(applied_physical or {})
        state["sigma_points"] = list(sigma_points or [])
        self._probe_model_cache().clear()
        self._tool_context.probe_param_status = f"fitted ({version_tag})"
        logger.info("Synthesis probe: sim.fit deployed %d params (%s).",
                    len(params), version_tag)

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
        fit).
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

    def _make_candidate_probe_model_provider(
        self,
        simulator_file: str,
        trajectories: List[LowLevelTrajectory],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
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
        cache = self._probe_model_cache()

        def _provider() -> _OracleOptionModel:
            if not os.path.isfile(simulator_file):
                raise RuntimeError(
                    "run_python probe: no candidate simulator yet - "
                    "write ./simulator.py (RESIDUAL_RULES / PARAM_SPECS / "
                    "RESIDUAL_FEATURES) first; the probe runs against it.")
            with open(simulator_file, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if cache.get("digest") == digest:
                return cache["model"]
            fit_state = self._probe_fit_state()
            rules, specs, features, ns = \
                self._load_simulator_from_module_file(
                    simulator_file, trajectories)
            if rules is None or specs is None:
                raise RuntimeError(
                    "run_python probe: ./simulator.py failed to load "
                    "(exec error, or RESIDUAL_RULES / PARAM_SPECS missing) - "
                    "fix the file and probe again.")
            residual_features = (features
                                 if features is not None else inferred_hint)
            latent_init = read_latent_init(ns) if isinstance(ns,
                                                             dict) else None
            # Never fit here: fitting is the agent's explicit ``sim.fit``
            # (its own budget, its own report). The candidate runs at
            # the last published fit's values (declared init values for
            # new params) and every probe result says so until the
            # agent fits the current file - so a rollout never silently
            # runs a fit it may not afford (sketch seed1 learn 011: an
            # implicit refit inside a probe hit the call cap and came
            # back as an empty "param fitting failed:").
            model, params, _ = build_candidate_option_model(
                self,
                rules,
                specs,
                residual_features,
                base_pred_triples,
                latent_init=latent_init,
                fit=False)
            if CFG.agent_sim_learn_declared_params_only:
                status = ("at the DECLARED values of the current "
                          "simulator.py (parameter estimation is disabled "
                          "in this run)")
            elif fit_state.get("digest") == digest:
                status = f"fitted ({fit_state.get('version')})"
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

    @staticmethod
    def _exploration_fit_num_steps() -> Optional[int]:
        """MCMC budget for the active-experiment posterior fit.

        The synthesis fit surfaces (``sim.fit``, ``sim.residuals``)
        share the fit statics and run repeatedly inside the agent loop,
        so they always use the global
        ``CFG.code_sim_learning_num_mcmc_steps`` (typically 0 - LM +
        Laplace only). The solver/test-time fit also uses that global
        setting. The exploration posterior fit is different: it runs
        once per learning cycle, only when it needs more MCMC than the
        solver fit already ran, and its posterior feeds only the
        info-seeking ensemble. With real posterior samples,
        ``_select_param_ensemble`` upgrades from the Laplace draw to a
        posterior subsample, calibrating ensemble spread for
        gate/threshold params whose flat likelihood has a near-zero
        Jacobian column at the MAP (invisible to Laplace).

        Returns ``None`` (no override; ``fit_params`` falls back to the
        global setting) when info-seeking is off, else the max of the
        global and exploration budgets so the override never *reduces*
        an explicitly configured global MCMC run.
        """
        if not CFG.agent_explorer_info_seeking:
            return None
        return max(CFG.code_sim_learning_num_mcmc_steps,
                   CFG.agent_explorer_info_mcmc_steps)

    @staticmethod
    def _separate_exploration_fit_num_steps() -> Optional[int]:
        """Return an exploration-only MCMC budget, if one is needed."""
        fit_num_steps = AgentSimLearningApproach._exploration_fit_num_steps()
        if fit_num_steps is None:
            return None
        if fit_num_steps <= CFG.code_sim_learning_num_mcmc_steps:
            return None
        return fit_num_steps

    def _rebuild_param_ensemble(self) -> None:
        """Rebuild the learned model's rule-parameter ensemble.

        Two consumers share it: info-seeking exploration (off under
        ablation A6) and the capture gate's rule-param margin (off under
        ablation A7). Built when either is on, a fit has populated
        ``_fitted_params`` and parameter uncertainty is in use; cleared
        otherwise. The ensemble can use an exploration-only posterior
        even when solver params remain at the global-budget point
        estimate.

        Picks the most *calibrated* ensemble the fit affords, preferring
        spreads that reflect real posterior uncertainty over uniform
        jitter (see :meth:`_select_param_ensemble`).
        """
        wanted = (CFG.agent_explorer_info_seeking
                  or CFG.agent_plan_validation_rule_param_margin)
        if (not wanted or not self._fitted_params
                or not CFG.agent_sim_learn_param_uncertainty):
            self._param_ensemble = []
            return
        if CFG.agent_sim_learn_oracle_sim_params:
            # Oracle params carry no uncertainty: no fit ran, so the only
            # ensemble on offer would be box jitter around the truth,
            # which manufactures wrong models (a zero rate, a rewired
            # lamp) that the capture gate would then demand every plan
            # survive. Nothing to hedge against, so no ensemble.
            self._param_ensemble = []
            logger.info("Oracle sim params: no rule-parameter ensemble "
                        "(nothing uncertain to sweep).")
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

        * ``posterior`` - when MCMC ran (``num_mcmc_steps > 0``), subsample
          the real posterior ``samples`` (works for both per-transition and
          recurrent fits).
        * ``laplace`` - else, when the fit attached an LM Jacobian
          (``num_mcmc_steps == 0``, per-transition or recurrent), draw
          from the Laplace covariance at the MAP.
        * ``uniform`` - otherwise (oracle params, LM skipped/failed, or
          calibration disabled), fall back to box-relative jitter.
        """
        fit = self._last_fit_result
        calibrated = CFG.agent_explorer_info_calibrated_ensemble
        if calibrated and fit is not None:
            samples = np.asarray(fit.samples, dtype=float)
            if samples.ndim == 2 and samples.shape[0] > 1:
                return posterior_subsample_ensemble(
                    fit.point_estimate,
                    fit.names,
                    samples,
                    num_members=num_members,
                    rng=self._rng,
                ), "posterior-subsample"
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

        Wired into refinement as the info-scorer for the agent_model_based
        explorer; a read-only query that leaves ``_fitted_params``
        unchanged on return.
        """
        atom_list = list(atoms)
        if len(self._param_ensemble) <= 1 or not atom_list:
            return 0.0
        saved = dict(self._fitted_params)
        try:
            rows: List[List[bool]] = []
            for member in self._param_ensemble:
                self._fitted_params.clear()
                self._fitted_params.update(member)
                rows.append([bool(a.holds(state)) for a in atom_list])
        finally:
            self._fitted_params.clear()
            self._fitted_params.update(saved)
        return mean_bernoulli_entropy(np.asarray(rows, dtype=bool))

    @contextmanager
    def _rule_param_override_scope(
            self, override: Dict[str, float]) -> Iterator[None]:
        """Swap ``_fitted_params`` to ``override`` for the duration.

        The learned rules and frozen predicate classifiers read the
        fitted params through a live view (see ``_ParamsView``), so the
        swap changes their gates for the wrapped validation rollout and
        the restore returns the deployed fit untouched - the same
        pattern :meth:`score_atom_disagreement` uses for ensemble
        scoring. Installed on the tool context as
        ``rule_param_override_scope`` for the capture gate's
        rule-parameter margin sweep.
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

    def _synthesize_with_agent(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
    ) -> None:
        """Obtain RESIDUAL_RULES / PARAM_SPECS / RESIDUAL_FEATURES, then fit.

        ``inferred_hint`` is passed to the agent as a starting point and
        used as the eval/test scope until it declares its own
        ``RESIDUAL_FEATURES``. CFG flag
        ``agent_sim_learn_oracle_sim_program`` short-circuits the agent
        session by loading the GT simulator instead (and
        ``agent_sim_learn_oracle_sim_params`` additionally skips the
        MCMC fit; see :meth:`_fit_params_after_synthesis`).
        """
        if CFG.agent_sim_learn_oracle_sim_program:
            rules, specs, residual_features = \
                self._load_oracle_sim_program(inferred_hint)
        else:
            loaded = self._run_agent_synthesis_session(trajectories,
                                                       obs_triples,
                                                       base_pred_triples,
                                                       inferred_hint)
            if loaded is None:
                return
            rules, specs, residual_features = loaded
        self._residual_rules = rules
        self._residual_features = residual_features
        self._fit_params_after_synthesis(rules, specs, base_pred_triples,
                                         residual_features)

    def _load_oracle_sim_program(
        self, inferred_hint: Dict[str, List[str]]
    ) -> Tuple[List, List[ParamSpec], Dict[str, List[str]]]:
        """Load the ground-truth simulator instead of running an agent.

        ``get_gt_simulator`` dispatches by observability: in
        partially-observable mode it returns the PO GT simulator
        (gt_simulator_po.py - latent heat threaded across steps,
        surfaced as the observable bubbling_level), which predicts only
        observable features; otherwise it returns the fully-observable
        gt_simulator.py (which reads/writes heat_level as a State
        feature). The two factories gate on CFG.partially_observable so
        the env-name dispatch resolves to exactly one module per run.

        Unless ``agent_sim_learn_oracle_sim_params`` also holds, the
        declared parameter inits are perturbed so the subsequent fit
        starts from a miscalibrated - not oracle - belief.
        """
        rules, specs, residual_features = get_gt_simulator(CFG.env)
        self._log_feature_set_diff(inferred_hint, residual_features,
                                   "inferred", "oracle")
        if not CFG.agent_sim_learn_oracle_sim_params:
            specs = self._perturb_spec_inits(specs)
        logger.info("Loaded oracle sim program (%d rules, %d params).",
                    len(rules), len(specs))
        return rules, specs, residual_features

    @staticmethod
    def _perturb_spec_inits(specs: List[ParamSpec]) -> List[ParamSpec]:
        """Perturb each spec's init with multiplicative Gaussian noise.

        Used when the oracle sim PROGRAM is loaded but its param VALUES
        must still be learned: the fit then starts from a plausible but
        wrong belief instead of the answer. Each perturbed init is
        clipped to its spec's box.
        """
        rng = np.random.default_rng(CFG.seed)
        noise_scale = CFG.agent_sim_learn_oracle_sim_param_noise_scale
        if noise_scale < 0.0:
            raise ValueError("agent_sim_learn_oracle_sim_param_noise_scale "
                             "must be non-negative.")
        perturbed = []
        for s in specs:
            val = float(
                np.clip(s.init_value * (1.0 + rng.normal(0, noise_scale)),
                        s.lo, s.hi))
            perturbed.append(
                ParamSpec(s.name,
                          val,
                          lo=s.lo,
                          hi=s.hi,
                          scale=getattr(s, "scale", "linear")))
        return perturbed

    def _run_agent_synthesis_session(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
    ) -> Optional[Tuple[List, List[ParamSpec], Dict[str, List[str]]]]:
        """Run one agent synthesis session and load what it committed.

        Returns ``(rules, specs, residual_features)``, or None when the
        session left no loadable simulator artifact. Per-skill samplers
        (when enabled) ride along in the same session.
        """
        paths = self._resolve_synthesis_paths()
        extra_paths = self._compute_extra_synthesis_paths(paths.base)
        sampler_paths = (self._sampler_paths(paths.base)
                         if self._do_synthesize_samplers else {})
        exec_ns = self._build_synthesis_exec_ns(trajectories)
        self._attach_synthesis_session_state(exec_ns, trajectories,
                                             base_pred_triples, inferred_hint,
                                             paths, extra_paths, sampler_paths)
        # Fresh session so the synthesis prompt + tools take effect.
        self._close_agent_session()
        self._ensure_agent_session()
        structs_ref = self._write_structs_reference()
        base_sim_refs = self._base_sim_reference_paths()
        message = self._build_synthesis_learn_message(
            trajectories, obs_triples, inferred_hint, paths, structs_ref,
            extra_paths, sampler_paths, base_sim_refs)
        try:
            responses = self._query_agent_sync(message, kind="learn")
            dead = query_fatal_error(responses)
            if dead is not None:
                # The synthesis session never ran (usage limit, auth,
                # transport): nothing was learned, so this cycle must
                # not be checkpointed as learned. The cycle's explore
                # episodes are stashed (main._save_inflight_interactions),
                # so a relaunch resumes at exactly this learn. Silently
                # continuing once wrote a byte-identical checkpoint and
                # burned a whole cycle (2026-08-27 run_20260827_121111).
                raise AgentSessionFatalError(
                    "The learn session died without the agent doing any "
                    f"work ({dead}); refusing to checkpoint this cycle as "
                    "learned.")
        finally:
            self._tool_context.extra_session_hooks = {}
            self._tool_context.extra_mcp_tools = []
            self._tool_context.probe_artifact_loaders.clear()
            self._tool_context.probe_option_model_provider = None
            self._tool_context.probe_fit_provider = None
            self._tool_context.probe_param_status = None
            self._tool_context.probe_residuals_provider = None
            self._tool_context.learn_cycle_index = None
            self._learning_mode = False
            self._close_agent_session()
        return self._load_synthesis_artifacts(trajectories, inferred_hint,
                                              paths, extra_paths,
                                              sampler_paths)

    def _resolve_synthesis_paths(self) -> _SynthesisPaths:
        """Host- and agent-visible paths for one synthesis session.

        The sandbox dir is resolved without depending on a live session
        manager: LocalSandboxSessionManager does set it on tool_context
        in __init__, but it isn't constructed until
        ``_ensure_agent_session()`` runs later in the session setup.

        The agent-visible paths differ by sandbox backend: cwd-relative
        for local-sandbox (the validation hook resolves against cwd and
        rejects literal ``/sandbox/...`` paths), the docker mount point
        for docker, the absolute host path otherwise.
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
        elif sandbox_dir:
            simulator_file_for_agent = "/sandbox/simulator.py"
            sandbox_dir_for_agent = "/sandbox"
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
        # Env ground-truth scoring, next to is_goal_state (see
        # Task.evaluator). Verdict-only surface: dict of reward/solved
        # on a concrete state sequence - real trajectories or the
        # agent's own simulator rollouts (there the verdict is only as
        # good as the sim).
        if any(t.evaluator is not None for t in self._train_tasks):
            exec_ns["evaluate_trajectory"] = \
                self._make_evaluate_trajectory_fn()
        return exec_ns

    def _attach_synthesis_session_state(
        self,
        exec_ns: Dict[str, Any],
        trajectories: List[LowLevelTrajectory],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        extra_paths: Dict[str, str],
        sampler_paths: Dict[str, str],
    ) -> None:
        """Install this synthesis session's state on the tool context.

        Everything installed here is cleared by the caller's ``finally``
        once the session query returns.
        """
        # Label tool output (e.g. attempt-log headers) with the
        # learning cycle for the duration of this session.
        self._tool_context.learn_cycle_index = self._learning_cycle_index()
        # Build dynamic synthesis tools and attach them to the tool
        # context *before* opening the session. The attached set is
        # filtered against ``_get_synthesis_tool_names`` so that method
        # is the single source of truth for what the agent sees:
        # anything a builder constructs but the names list omits is
        # dropped here.
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.belief_probe import _check_time_budget
        toolkit = create_synthesis_tools(
            exec_ns,
            base_pred_triples,
            inferred_hint,
            simulator_file=paths.simulator_file,
            versions_dir=paths.versions_dir,
            approach=self,
            sandbox_dir=paths.base,
            sandbox_dir_for_agent=paths.sandbox_dir_for_agent,
            cycle_index_provider=self._learning_cycle_index,
            budget_check=lambda: _check_time_budget(self._tool_context),
        )
        tools = list(toolkit.tools)
        self._install_extra_synthesis_surfaces(exec_ns, base_pred_triples,
                                               inferred_hint, extra_paths)
        if self._do_synthesize_samplers:
            self._install_sampler_surface(sampler_paths)
        declared = set(self._get_synthesis_tool_names() or ())
        self._tool_context.extra_mcp_tools = [
            t for t in tools if getattr(t, "name", "") in declared
        ]
        # Point the probe at the CANDIDATE simulator for this session
        # (never the stale pre-synthesis option model; on cycle 1 that
        # wraps the real env), then merge the probe facade into
        # run_python's namespace: synthesis sessions offer ONE exec
        # namespace, so helpers defined next to the data are visible to
        # probe sweeps (create_mcp_tools skips the solve-phase instance
        # when this one is attached). Unconditional: with fit / refine /
        # forward-validation all living on ``sim``, the probe IS the
        # validation surface, so a synthesis session without it would
        # have no way to test what it writes. Only ``sim``/``BeliefProbe``
        # are taken from the probe namespace: ``trajectories`` already
        # binds the fit list and solve-only extras do not apply.
        self._tool_context.probe_option_model_provider = \
            self._make_candidate_probe_model_provider(
                paths.simulator_file, trajectories, base_pred_triples,
                inferred_hint)
        self._tool_context.probe_fit_provider = toolkit.fit_runner
        self._tool_context.probe_residuals_provider = \
            toolkit.residuals_runner
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.belief_probe import build_probe_namespace
        probe_ns = build_probe_namespace(self._tool_context)
        exec_ns["sim"] = probe_ns["sim"]
        exec_ns["BeliefProbe"] = probe_ns["BeliefProbe"]
        self._learning_mode = True
        # PostToolUse hook: snapshot simulator.py / predicates.py on
        # every successful Write/Edit/MultiEdit, so the version history
        # covers everything the agent committed to file (not just
        # states that happened to coincide with an eval call). Only
        # active for this synthesis session.
        snapshot_targets = self._build_write_snapshot_targets(
            paths.simulator_file, paths.versions_dir, extra_paths)
        if self._do_synthesize_samplers:
            snapshot_targets.append(
                self._sampler_snapshot_target(sampler_paths))
        self._tool_context.extra_session_hooks = (
            self._build_synthesis_session_hooks(snapshot_targets, paths.base))

    def _build_synthesis_learn_message(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        structs_ref: str,
        extra_paths: Dict[str, str],
        sampler_paths: Dict[str, str],
        base_sim_refs: Optional[List[str]] = None,
    ) -> str:
        """Compose the synthesis session's first user message.

        Gathers this cycle's data roster, digests, and reports and
        renders ``learn_message.md``. Reads the just-opened session's
        tool names, so the session must be open before this is called.
        """
        n_trajs = len(trajectories)
        n_demos = sum(1 for t in trajectories if t.is_demo)
        # Start-of-session divergence report: when a prior model exists,
        # score it (params refit to ALL data, so what remains is the
        # structural gap) before the agent's first turn - the session
        # then starts from "here is where the model breaks" instead of
        # spending turns rediscovering it. The same report stays callable
        # as `sim.residuals()` against every subsequent edit. With no
        # prior model the "prior" is the bare base simulator and every
        # mismatch is an unmodeled mechanism.
        prior_state_block = self._format_prior_state_block(paths.base)
        divergence_block = ""
        if (self._tool_context.probe_residuals_provider is not None
                and obs_triples):
            try:
                report = self._tool_context.probe_residuals_provider(
                    max_transitions=100000,
                    fit_params=not CFG.agent_sim_learn_declared_params_only)
                divergence_block = learn_prompts.render_divergence_block(
                    report, has_prior_model=bool(prior_state_block))
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Skipping start-of-session residual report: %s",
                               e)
        session_tool_names = (self._agent_session.tool_names
                              if self._agent_session is not None else [])
        extra_messages = []
        if not trajectories and CFG.agent_sim_learn_zero_shot:
            extra_messages.append(learn_prompts.render_zero_shot_message())
        extra_message = self._extra_synthesis_message(extra_paths)
        if extra_message:
            extra_messages.append(extra_message)
        if self._do_synthesize_samplers:
            extra_messages.append(
                self._sampler_synthesis_message(sampler_paths))
        return learn_prompts.build_learn_message(
            n_trajs=n_trajs,
            n_transitions=len(obs_triples),
            n_demos=n_demos,
            n_interaction=n_trajs - n_demos,
            trajectory_listing=self._format_trajectory_listing(trajectories),
            structs_ref=structs_ref,
            inferred_hint=str(inferred_hint),
            predicate_listing=self._format_predicate_signatures(
                self._get_all_predicates()),
            types_digest=render_types_digest(self._tool_context.types),
            options_digest=render_options_digest(
                self._tool_context.options,
                gt_options_ref_path=self._tool_context.gt_options_ref_path),
            simulator_file=paths.simulator_file_for_agent,
            objective_block=self._format_objective_block(),
            prior_state_block=prior_state_block,
            divergence_block=divergence_block,
            base_sim_block=learn_prompts.render_base_sim_block(base_sim_refs
                                                               or []),
            tools_block=learn_prompts.render_tools_block(session_tool_names),
            extra_messages=extra_messages,
        )

    def _load_synthesis_artifacts(
        self,
        trajectories: List[LowLevelTrajectory],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        extra_paths: Dict[str, str],
        sampler_paths: Dict[str, str],
    ) -> Optional[Tuple[List, List[ParamSpec], Dict[str, List[str]]]]:
        """Load the artifacts the finished session committed to disk.

        Returns ``(rules, specs, residual_features)`` or None when no
        loadable simulator exists. The optional LATENT_INIT /
        PHYSICAL_PARAMS side exports are recorded on ``self`` before the
        loadability check, so they are picked up even from an artifact
        whose rules fail to load.
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
        # Optional PHYSICAL_PARAMS export: base-sim parameters to
        # identify jointly with the rule params (system ID). The fit
        # scale (log vs linear) is stamped from the env registry; agents
        # copy name/init/bounds but need not know about it.
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
        if self._do_synthesize_samplers:
            self._finalize_and_load_samplers(sampler_paths)
        return rules, specs, residual_features

    def _fit_params_after_synthesis(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
    ) -> None:
        """Fit/store solver params and, separately, explorer posterior."""
        if CFG.agent_sim_learn_oracle_sim_params:
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})
            if self._physical_param_specs:
                # Oracle mode: trust the agent-declared physical inits.
                self._apply_identified_physical_params(
                    {s.name: s.init_value
                     for s in self._physical_param_specs})
            # No fit ran; the ensemble falls back to uniform perturbation.
            self._last_fit_result = None
            if base_pred_triples:
                self._fit_sse = self._oracle_param_sse(rules,
                                                       base_pred_triples,
                                                       residual_features,
                                                       FIT_NOISE_SIGMA)
            else:
                logger.info("No transitions; skipping oracle-param SSE.")
                self._fit_sse = float("inf")
        elif CFG.agent_sim_learn_declared_params_only:
            self._deploy_declared_params(rules, specs, base_pred_triples,
                                         residual_features)
        elif not base_pred_triples:
            # No data to fit against (e.g. every demo failed, or a
            # zero-shot learn): seed from the declared inits so the
            # simulator still builds; later cycles refit once
            # transitions arrive. The declared physical inits go to the
            # planning env too - the agent's stated belief, not the
            # registry default, is what its plans were written against.
            logger.warning("No transitions to fit; seeding params from "
                           "declared inits.")
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})
            if self._physical_param_specs:
                self._apply_identified_physical_params(
                    {s.name: s.init_value
                     for s in self._physical_param_specs})
            self._last_fit_result = None
            self._fit_sse = float("inf")
        else:
            # The deployed model is the agent's own canonical sim.fit of
            # the final simulator.py: the values its GO/NO-GO check
            # validated, with the Laplace bundle the exploration ensemble
            # is calibrated from. The harness fits only when no such fit
            # exists (session ended UNFITTED, ran out of turns, or an
            # oracle sim program with no session at all), and says so.
            expected = [s.name for s in self._physical_param_specs
                        ] + [s.name for s in specs]
            published = None
            if self._probe_fit_state().get("fit_result") is not None:
                published = self._published_fit_for_file(
                    self._resolve_synthesis_paths().simulator_file, expected)
            if published is not None:
                fit_result, self._fit_sse, version = published
                logger.info(
                    "Deploying the agent's published sim.fit (%s) of the "
                    "final simulator.py: %d params, SSE %.6f.", version,
                    len(expected), self._fit_sse)
                applied = self._probe_fit_state().get("applied_physical")
                if self._physical_param_specs and applied:
                    # Mirror _fit_parameters_joint_rollout's deploy: the
                    # cycle-level applied snapshot and the physics-margin
                    # sigma points come from the published fit (applying
                    # resets the points, so set them after).
                    self._apply_identified_physical_params(dict(applied))
                    self._cycle_applied_physical = dict(applied)
                    if CFG.agent_sim_learn_param_uncertainty:
                        self._identified_physical_sigma_points = list(
                            self._probe_fit_state().get("sigma_points") or [])
            else:
                if CFG.agent_sim_learn_oracle_sim_program:
                    logger.info("Oracle sim program: fitting its "
                                "parameters on the harness side.")
                else:
                    logger.warning(
                        "FIT FALLBACK: the learn session ended without a "
                        "canonical sim.fit() of the final simulator.py "
                        "(last published fit: %s). Fitting on the "
                        "harness side - the deployed parameters were "
                        "never validated by the agent's GO check.",
                        self._probe_fit_state().get("version") or "none")
                if self._physical_param_specs or has_physics_rules(rules):
                    fit_result, self._fit_sse = (
                        self._fit_parameters_joint_rollout(
                            rules, specs, residual_features))
                elif has_latent_rules(rules):
                    fit_result, self._fit_sse = \
                        self._fit_parameters_recurrent(
                            rules, specs, base_pred_triples,
                            residual_features)
                else:
                    fit_result, self._fit_sse = fit_rule_parameters(
                        rules, specs, base_pred_triples, residual_features)
            self._last_fit_result = fit_result
            self._fitted_params.clear()
            self._fitted_params.update(fit_result.point_estimate)
            if CFG.code_sim_learning_num_mcmc_steps == 0:
                logger.info("Skipped solver MCMC; using %d fitted params.",
                            len(specs))
            else:
                logger.info("Fitted %d solver params.", len(specs))

            self._maybe_refit_exploration_posterior(rules, specs,
                                                    base_pred_triples,
                                                    residual_features)

        # Remember the specs (names + bounds) and rebuild the active-
        # experiment ensemble. Cheap and only consumed when info-seeking
        # exploration is enabled. Physical specs lead so the ordering
        # matches the joint rollout fit's theta layout.
        self._param_specs = list(self._physical_param_specs) + list(specs)
        self._rebuild_param_ensemble()

    def _physics_margin_points(
        self,
        applied: Dict[str, float],
        report: Dict[str, Dict[str, Any]],
        physical_specs: List[ParamSpec],
    ) -> List[Dict[str, float]]:
        """The capture gate's physics-margin grid for ``applied``.

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
            self._cycle_applied_physical = dict(applied)
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

    def _maybe_refit_exploration_posterior(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
    ) -> None:
        """Run the exploration-only posterior fit, when one is needed.

        A no-op unless info-seeking exploration asks for more MCMC than
        the solver fit already ran (see
        :meth:`_separate_exploration_fit_num_steps`). The resulting
        posterior replaces ``_last_fit_result`` for ensemble calibration
        only; ``_fitted_params`` (the solver's point estimate) is left
        untouched.
        """
        num_steps = self._separate_exploration_fit_num_steps()
        if num_steps is None:
            return
        if self._physical_param_specs or has_physics_rules(rules):
            logger.info("Skipping separate active-experiment fit: the joint "
                        "rollout sysID posterior is reused for exploration.")
            return
        # Reuse the solver fit's LM MAP + jacobian instead of re-running
        # the (expensive, full-data) LM fit for the identical objective.
        # Only safe when the solver fit was LM-only: with real solver
        # MCMC its point_estimate is the MCMC MAP, not the LM MAP the
        # jacobian was computed at.
        lm_seed: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
        prev = self._last_fit_result
        if (prev is not None and CFG.code_sim_learning_num_mcmc_steps == 0
                and prev.samples.shape[0] == 1
                and list(prev.names) == [s.name for s in specs]):
            theta = np.array([prev.point_estimate[n] for n in prev.names])
            lm_seed = (theta, prev.jacobian)
        if has_latent_rules(rules):
            fit_result, sse = self._fit_parameters_recurrent(
                rules,
                specs,
                base_pred_triples,
                residual_features,
                num_steps=num_steps,
                lm_seed=lm_seed)
        else:
            fit_result, sse = fit_rule_parameters(rules,
                                                  specs,
                                                  base_pred_triples,
                                                  residual_features,
                                                  num_steps=num_steps,
                                                  lm_seed=lm_seed)
        self._last_fit_result = fit_result
        logger.info(
            "Fitted active-experiment posterior with %d MCMC steps "
            "for exploration planning only (SSE: %.6f).", num_steps, sse)

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
        if has_physics_rules(rules):
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

    # ── System identification (PHYSICAL_PARAMS) support ──────────

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
            return create_new_env(CFG.env,
                                  do_cache=False,
                                  use_gui=False,
                                  skip_residual_dynamics=True)

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

    def _persist_fit_trajectories(self, label: str = "fitted") -> None:
        """Dump the raw fit trajectories for offline post-mortems.

        ``label`` distinguishes the two moments this is called from:
        ``recorded`` when a cycle's data arrives (always), ``fitted``
        when a sysID fit has just run and the payload's identified
        params mean something. The first is what makes a cycle that
        declined to fit replayable at all.

        The rollout sysID fit data otherwise exists only in memory:
        when run_20260724_232411 shipped friction fits 2-7 sigma from
        the truth, the failing fits could not be replayed offline - the
        episodes had to be approximately re-executed from logged plans,
        which cannot reproduce mid-episode replans or the warm-env
        recording context (exactly the suspected corruption channel).
        One pickle per cycle-level fit under ``<log_dir>/fit_data/``;
        never raises - persistence must not take down a run.
        """
        if not CFG.code_sim_learning_persist_fit_data:
            return
        try:
            out_dir = os.path.join(self._get_log_dir(), "fit_data")
            os.makedirs(out_dir, exist_ok=True)
            idx = len([f for f in os.listdir(out_dir) if f.endswith(".pkl")])
            path = os.path.join(out_dir,
                                f"fit_trajectories_{idx:03d}_{label}.pkl")
            payload = {
                "trajectories":
                list(self._fit_trajectories),
                "physical_param_specs":
                list(self._physical_param_specs),
                "identified_physical_params":
                dict(self._identified_physical_params),
            }
            with open(path, "wb") as f:
                pkl.dump(payload, f)
            logger.info("Persisted %d fit trajectories to %s",
                        len(self._fit_trajectories), path)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Could not persist fit trajectories: %s", e)

    def _apply_identified_physical_params(
            self, identified: Dict[str, float]) -> None:
        """Publish identified physical params into the planning base env.

        The applied set exactly mirrors ``identified``: params applied
        by an earlier fit but absent here (e.g. dropped from a later
        artifact's PHYSICAL_PARAMS) are reverted to the env's registry
        defaults, because the env-side override is sticky per param and
        a stale value from a superseded fit would otherwise silently
        keep steering the planner. The override survives resets but not
        env recreation; ``_recreate_base_env`` re-applies from
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
        logger.info("Applied identified physical params to base env: %s",
                    {k: f"{v:.4f}"
                     for k, v in identified.items()})

    def _fit_parameters_joint_rollout(
        self,
        rules: List,
        rule_specs: List[ParamSpec],
        residual_features: Dict[str, List[str]],
    ) -> Tuple[FitResult, float]:
        """Joint physical+rule fit against free-running base-sim rollouts.

        Reached when the artifact declares ``PHYSICAL_PARAMS``. Consumes
        the RAW observed trajectories rather than ``base_pred_triples``:
        physical parameters only manifest when momentum free-runs, which
        the teacher-forced triples destroy (``State`` has no
        velocities). One theta = physical + rule params, one joint fit,
        so rules cannot silently absorb physics error; with no rules
        this degenerates to pure identification. The identified physical
        values are applied in place to the planning base env, and the
        per-parameter identifiability report (posterior contraction) is
        logged so null parameters are visible rather than silently
        trusted.
        """
        physical_specs = self._physical_param_specs
        physical_names = [s.name for s in physical_specs]
        # Factory, not an instance: every rollout runs in a fresh env.
        fit_env = self._get_rollout_fit_env()
        self._persist_fit_trajectories()
        rollouts = self._rollout_fit_trajectories(residual_features)
        init_params = {
            s.name: s.init_value
            for s in physical_specs + rule_specs
        }
        anchors = physical_param_anchors(self._base_env, physical_specs)
        if not rollouts:
            logger.warning(
                "No complete trajectories for rollout sysID; keeping the "
                "declared physical-param inits unfitted.")
            result = fit_params_rollout(fit_env, [],
                                        physical_specs,
                                        residual_features,
                                        rules=rules,
                                        rule_specs=rule_specs,
                                        latent_init=self._latent_init,
                                        anchors=anchors)
            self._apply_identified_physical_params(
                {n: init_params[n]
                 for n in physical_names})
            return result, float("nan")

        # The adjuster's third argument is the fit's own SSE-at-theta
        # probe (survivor set + shared scaling - the objective the fit
        # minimized), which the cross-cycle consistency check uses to
        # arbitrate flagged jumps on evidence. Deliberately NOT a
        # full-set SSE: trimmed (unexplainable) segments would add the
        # same large error to both candidates and dilute the ratio.
        outcome = run_rollout_sysid(
            fit_env,
            rollouts,
            physical_specs,
            residual_features,
            rules=rules,
            rule_specs=rule_specs,
            latent_init=self._latent_init,
            anchors=anchors,
            rms_cache=self._explainability_cache,
            report_adjuster=lambda result, report, sse_fn:
            (self._check_cross_cycle_consistency(
                result, report, physical_names, pooled_sse=sse_fn)),
            held={
                **self._identified_physical_params,
                **self._cycle_applied_physical
            })
        if outcome.num_survivors == 0:
            # NO fit ran (the result is pinned at the declared inits).
            # Apply nothing: the planner keeps its standing belief -
            # the previous cycle's applied values - rather than being
            # reverted to baselines (or moved to this call's declared
            # inits) by data that supports neither.
            logger.warning(
                "Rollout sysID: no explainable segments this cycle; "
                "leaving the planner's physical params untouched.")
            self._record_sysid_diagnostics({}, physical_names, 0,
                                           len(rollouts), outcome.traj_rms)
            return outcome.fit_result, float("nan")
        logger.info("Identifiability (posterior/prior contraction):\n%s",
                    format_identifiability(outcome.report))
        log_param_changes(init_params, outcome.fitted)
        self._apply_identified_physical_params(outcome.applied)
        # Snapshot the cycle-level decision: this (not whatever the
        # agent's in-session sim.fit last applied) is what a future
        # INCONSISTENT verdict holds on to.
        self._cycle_applied_physical = dict(outcome.applied)
        # Physics-margin points for the capture gate: the fit's posterior
        # widths (floored, see identifiability_report) turned into a grid
        # of perturbations spanning +-1 sigma of the applied values.
        self._identified_physical_sigma_points = self._physics_margin_points(
            outcome.applied, outcome.report, physical_specs)
        if self._identified_physical_sigma_points:
            logger.info("Physics-margin points for capture validation: %s",
                        [{k: f"{v:.4f}"
                          for k, v in pt.items()}
                         for pt in self._identified_physical_sigma_points])
        self._record_sysid_diagnostics(outcome.report,
                                       physical_names, outcome.num_survivors,
                                       len(rollouts), outcome.traj_rms)
        return outcome.fit_result, outcome.post_sse

    def _check_cross_cycle_consistency(
        self,
        result: FitResult,
        report: Dict[str, Dict[str, Any]],
        physical_names: Sequence[str],
        pooled_sse: Optional[Callable[[Dict[str, float]],
                                      float]] = None) -> None:
        """Flag params whose confident MAP jumped since the previous cycle.

        The curvature probe measures local *precision*: a biased
        objective yields precisely-wrong values that the probe still
        stamps "identified" (observed: per-cycle friction fits 0.0585,
        0.0614, 0.0919, 0.0794, each with posterior_std ~0.003 -
        mutually incompatible by many sigmas). Comparing successive
        final fits in FIT space (log for log-scale params) catches
        exactly this: a jump above
        ``CFG.code_sim_learning_rollout_cross_cycle_sigma`` combined
        sigmas sets ``Verdict.INCONSISTENT``, and the trust selection
        then HOLDS the currently-applied value instead of hopping to
        the new fit - neither of two mutually-incompatible confident
        fits can be preferred on this evidence, and hopping churned the
        belief env for whole runs (run_20260721_205821 seed1:
        restitution 0.71 -> 0.52 -> 0.02 -> 0.32 -> 0.02). History
        records only these final per-cycle fits, not the agent's
        in-session tool fits, whose param sets churn.

        Sigma distance alone cannot tell a real correction from probe
        churn, and successive cycle fits are NOT independent equals:
        the new fit minimized the objective over a superset of the old
        fit's data. So before holding, a flagged jump is arbitrated on
        evidence via ``pooled_sse`` (the fit's own SSE-at-theta probe
        over its surviving segments - the objective it minimized):
        when the held value explains that data decisively worse than
        the new fit
        (``CFG.code_sim_learning_rollout_consistency_sse_ratio``), the
        jump is accepted. Without a decisive gap the hold stands
        (run_20260724_232411-style subset disagreement stays held +
        hull-swept). Motivated by run_20260727_210827 seed1: a sharp
        but biased 2-trajectory cycle-0 fit (0.9313, true 0.5) was held
        over the 4-trajectory refit (0.4748, pooled SSE 0.14 vs ~4.4)
        for the rest of the run.
        """
        k = CFG.code_sim_learning_rollout_cross_cycle_sigma
        fitted = result.point_estimate
        scales = result.scales or ["linear"] * len(result.names)
        for i, name in enumerate(result.names):
            if name not in physical_names:
                continue
            if report.get(name, {}).get("verdict") is Verdict.ANCHORED:
                # An ablation-reverted param's point estimate IS the
                # baseline, not a fit. Recording it would make the next
                # cycle's genuine fit read as a many-sigma jump (and
                # spuriously downgrade it); keep the previous history
                # entry, which holds the last real fit.
                continue
            post = float(
                report.get(name, {}).get("posterior_std", float("nan")))
            scale = scales[i]
            value = fitted[name]
            prev = self._sysid_fit_history.get(name)
            flagged = False
            if (k > 0 and prev is not None and np.isfinite(post)
                    and np.isfinite(prev[1])):
                prev_val, prev_std, prev_scale = prev
                if prev_scale == scale:
                    dist = _fit_space_dist(value, prev_val, scale)
                    combined = float(np.sqrt(post**2 + prev_std**2))
                    if combined > 0 and dist / combined > k:
                        n_sigma = dist / combined
                        pending = self._sysid_pending_fit.get(name)
                        if pending is not None and _fit_space_dist(
                                value, pending[0], scale) / max(
                                    float(np.sqrt(post**2 + pending[1]**2)),
                                    1e-12) <= k:
                            # Two INDEPENDENT cycles agree on the new
                            # value: the jump was real, not probe
                            # overconfidence - accept it.
                            logger.info(
                                "Rollout sysID cross-cycle consistency: "
                                "%s jump to ~%.4f confirmed by an "
                                "independent refit (pending %.4f); "
                                "accepting the new value.", name, value,
                                pending[0])
                            self._sysid_pending_fit.pop(name, None)
                        elif self._arbitrate_cross_cycle_jump(
                                name, fitted, prev_val, pooled_sse):
                            # Pooled evidence decisively prefers the
                            # new fit over the held value (the helper
                            # logs the SSE gap); accept it now instead
                            # of waiting a cycle for confirmation.
                            self._sysid_pending_fit.pop(name, None)
                        else:
                            flagged = True
                            logger.warning(
                                "Rollout sysID cross-cycle consistency: %s "
                                "moved %.4f -> %.4f (%.1f combined sigmas > "
                                "%g) since the previous cycle; the posterior "
                                "is overconfident.", name, prev_val, value,
                                n_sigma, k)
                            entry = report.get(name)
                            if (entry is not None and
                                    entry["verdict"] is Verdict.IDENTIFIED):
                                entry["verdict"] = Verdict.INCONSISTENT
                                entry["note"] = (
                                    f"{prev_val:.4f} -> {value:.4f} is "
                                    f"{n_sigma:.1f} combined sigmas; holding "
                                    "the last trusted value, margin sweep "
                                    "spans both")
                                # Both incompatible fits become hull
                                # candidates so the margin sweep covers
                                # the whole disagreement - the interval
                                # [0.3236, 0.6267] contained the true
                                # 0.5 in run_20260724_232411 seed2.
                                cands = set(entry.get("candidate_values", ()))
                                cands.update((float(prev_val), float(value)))
                                if pending is not None:
                                    cands.add(float(pending[0]))
                                entry["candidate_values"] = sorted(cands)
                            self._sysid_pending_fit[name] = (value, post)
            if flagged:
                # Keep the trusted value as the comparison reference;
                # the rejected fit waits in _sysid_pending_fit for an
                # independent confirmation. Recording the rejected fit
                # here would make it the NEXT cycle's reference, i.e.
                # accept the hop one cycle late without any new
                # evidence.
                continue
            self._sysid_pending_fit.pop(name, None)
            self._sysid_fit_history[name] = (value, post, scale)

    @staticmethod
    def _arbitrate_cross_cycle_jump(
            name: str, fitted: Dict[str, float], prev_val: float,
            pooled_sse: Optional[Callable[[Dict[str, float]], float]]) -> bool:
        """Settle a flagged cross-cycle jump by pooled-data evidence.

        Evaluates ``pooled_sse`` (the fit's own objective over its
        surviving segments) under the new joint fit and under the same
        fit with ``name`` swapped back to the held value. Returns True
        (accept the jump) only when the held
        value's explanation is decisively worse - at least
        ``CFG.code_sim_learning_rollout_consistency_sse_ratio`` times
        the new fit's SSE. Anything short of decisive (including an SSE
        evaluation failure) returns False and leaves the hold-and-
        hull-sweep behavior in charge.
        """
        ratio = CFG.code_sim_learning_rollout_consistency_sse_ratio
        if pooled_sse is None or ratio <= 0:
            return False
        try:
            sse_new = pooled_sse(dict(fitted))
            held_theta = dict(fitted)
            held_theta[name] = prev_val
            sse_held = pooled_sse(held_theta)
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Rollout sysID cross-cycle arbitration: pooled SSE "
                "evaluation failed for %s; holding the trusted value.",
                name,
                exc_info=True)
            return False
        if not (np.isfinite(sse_new) and np.isfinite(sse_held)):
            return False
        decisive = sse_held > ratio * sse_new
        logger.info(
            "Rollout sysID cross-cycle arbitration: %s pooled SSE %.4g at "
            "the new fit %.4g vs %.4g at the held value %.4g - %s.", name,
            sse_new, fitted[name], sse_held, prev_val,
            ("decisively better, accepting the jump"
             if decisive else "not decisive, holding"))
        return decisive

    def _record_sysid_diagnostics(self, report: Dict[str, Dict[str, Any]],
                                  physical_names: Sequence[str],
                                  num_survivors: int, num_segments: int,
                                  rms: List[float]) -> None:
        """Digest the fit's weak spots for the next explore phase.

        Generic (domain-free) statements of what the data could not
        support - unexplainable segments, parameters the rollouts do not
        constrain, cross-cycle conflicts - phrased as experiment
        objectives. The explorer appends this to its guidance so the
        agent designs interactions that fill the gaps, instead of
        relying on whatever manipulation data the tasks happen to
        produce.
        """
        lines: List[str] = []
        dropped = num_segments - num_survivors
        if dropped > 0:
            lines.append(
                f"- {dropped} of {num_segments} recorded motion segments "
                "were unexplainable at ANY physical parameters (best RMS "
                f"{[f'{r:.3g}' for r in rms]}): their dynamics are not "
                "repeatable under replay. Prefer experiments whose outcome "
                "is dominated by object dynamics rather than prolonged "
                "robot-object contact: actuate cleanly, then let the scene "
                "evolve and settle on its own.")
        for name in physical_names:
            entry = report.get(name, {})
            verdict = entry.get("verdict", Verdict.UNKNOWN)
            note = entry.get("note", "")
            label = verdict.value + (f" ({note})" if note else "")
            if verdict is Verdict.IDENTIFIED:
                cands = entry.get("candidate_values", ())
                if len(cands) > 1:
                    lines.append(
                        f"- physical param '{name}': identified, but the "
                        "recorded segments preferred mutually-incompatible "
                        f"explanations spanning [{min(cands):.4g}, "
                        f"{max(cands):.4g}] (the physics-margin sweep "
                        "covers that whole hull). A clean, repeatable "
                        "interaction that excites this parameter and "
                        "little else would collapse the hull.")
                continue
            interval = entry.get("flat_interval")
            if (verdict in (Verdict.WEAKLY_IDENTIFIED, Verdict.NOT_IDENTIFIED)
                    and interval is not None and interval[0] != interval[1]):
                lines.append(
                    f"- physical param '{name}': the data cannot "
                    f"distinguish values in [{interval[0]:.4g}, "
                    f"{interval[1]:.4g}]. An experiment whose observable "
                    "outcome DIFFERS across this interval would pin it "
                    "down.")
                continue
            if verdict is Verdict.ANCHORED:
                # Anchor ablation handled this param correctly (the move
                # was compensatory; the baseline is applied) - it is NOT
                # a failed identification, so don't advise dropping it.
                lines.append(
                    f"- physical param '{name}': the fitted move was "
                    "compensatory (a refit with it at its baseline explains "
                    "the data equally well), so the baseline was kept. An "
                    "experiment that excites this parameter SPECIFICALLY "
                    "(not jointly with the others) would distinguish the "
                    "two explanations.")
                continue
            if verdict is Verdict.INCONSISTENT:
                lines.append(
                    f"- physical param '{name}': successive cycles produced "
                    f"confident but mutually-incompatible fits ({note}). "
                    "The objective is biased somewhere: collect a clean, "
                    "repeatable interaction that excites this parameter and "
                    "little else, so one of the two values can be refuted.")
                continue
            lines.append(
                f"- physical param '{name}': {label}. An experiment whose "
                "observable outcome CHANGES when this parameter changes "
                "would identify it; if none exists, drop it from "
                "PHYSICAL_PARAMS.")
        self._last_sysid_diagnostics = ("\n".join(lines) if lines else "")

    def _sync_tool_context(self) -> None:
        super()._sync_tool_context()
        self._tool_context.sysid_diagnostics = (self._last_sysid_diagnostics
                                                or None)
        self._tool_context.latent_tracking_available = \
            self._latent_tracking_available()

    def _latent_tracking_available(self) -> bool:
        """Whether episodes will run with an execution-time latent tracker (the
        loaded simulator threads a latent block)."""
        rules = self._residual_rules
        if not rules:
            return False
        return has_latent_rules(rules)

    def make_latent_tracker(self) -> Optional[LatentTracker]:
        """A fresh tracker over the current rules, params, and latent init (see
        ``code_sim_learning.latent_tracker``), or None for a fully- observable
        simulator.

        Parameters are passed by reference, as the belief simulator's
        closure does, so a later in-place fit is seen.
        """
        return make_latent_tracker(self._residual_rules, self._fitted_params,
                                   self._latent_init)

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

    def _fit_parameters_recurrent(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
        num_steps: Optional[int] = None,
        lm_seed: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None,
    ) -> Tuple[FitResult, float]:
        """MCMC over the recurrent (per-trajectory) SSE.

        Counterpart to :func:`fitting.fit_rule_parameters` for rules
        that carry a latent block. Re-groups the flat
        ``base_pred_triples`` into per-trajectory chunks (latent threads
        within a trajectory, not across) via the lengths cached in
        ``self._fit_trajectories``; falls back to a single trajectory if
        no grouping info exists. Delegates the actual fit/log to
        :func:`fitting.fit_rule_parameters_latent` so the agent's
        ``sim.fit`` surface scores latent rules through the exact
        same path.
        """
        groups = self._group_triples_by_trajectory(base_pred_triples)
        if not groups:
            logger.warning("No trajectory groups for recurrent fitting; "
                           "falling back to single-trajectory rollout.")
            groups = [base_pred_triples]
        return fit_rule_parameters_latent(rules,
                                          specs,
                                          groups,
                                          self._latent_init,
                                          residual_features,
                                          num_steps=num_steps,
                                          lm_seed=lm_seed)

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
        sse = compute_rollout_sse(self._get_rollout_fit_env(),
                                  rollouts,
                                  self._fitted_params,
                                  residual_features,
                                  physical_names=[],
                                  rules=rules,
                                  latent_init=self._latent_init)
        logger.info(
            "Oracle params (rollout, physics-command rules) - "
            "SSE: %.6f over %d trajectories", sse, len(rollouts))
        for name, val in sorted(self._fitted_params.items()):
            logger.info("  %-30s  %.4f", name, val)
        return sse

    def _attach_initial_latent(self, task: Task) -> Task:
        """Seed ``task.init.latent`` with the initial latent block.

        Refinement starts at ``task.init`` (the planner's ``traj[0]``), so
        the combined simulator must find a well-formed latent there. If no
        ``LATENT_INIT`` was loaded (or the resulting block is empty), leave
        the task alone so downstream code keeps the legacy
        ``state.latent is None`` behaviour. Overrides the no-op default in
        :class:`AgentModelBasedApproach`.
        """
        if self._latent_init is None:
            return task
        initial_latent = init_latent(self._latent_init, self._fitted_params
                                     or {})
        if not initial_latent:
            return task
        init_state = task.init.copy()
        init_state.latent = initial_latent
        return Task(init=init_state,
                    goal=task.goal,
                    alt_goal=task.alt_goal,
                    goal_nl=task.goal_nl)

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
        # Reference the dict (not its values) so MCMC param updates are
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
    def _infer_residual_features_from_scan(
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
        min_hits: int = 3,
    ) -> Dict[str, List[str]]:
        """Features whose base-sim prediction diverges from observation.

        Flags ``(type, feat)`` if ``|pred - obs| > rel_tol*|obs| + abs_tol``
        on at least ``min_hits`` triples. The ``min_hits`` floor keeps
        one-off PyBullet jitter from leaking base-handled features into the set.
        """
        del obs_triples  # objects are identical across both triple lists
        pairs = [(s_base, s_obs) for s_base, _, s_obs in base_pred_triples]
        hits: Dict[Tuple[str, str], int] = {}
        for _, _, tn, feat, pred, obs in iter_feature_residuals(pairs):
            if abs(pred - obs) > rel_tol * abs(obs) + abs_tol:
                hits[(tn, feat)] = hits.get((tn, feat), 0) + 1
        out: Dict[str, List[str]] = {}
        for (t, f), n in hits.items():
            if n >= min_hits:
                out.setdefault(t, []).append(f)
        return {t: sorted(fs) for t, fs in out.items()}

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

    @staticmethod
    def _format_predicate_signatures(predicates: Set[Predicate]) -> str:
        """Pretty-print predicates as ``Name(type1, type2)`` lines.

        Mirrors the ``## Available Predicates`` block in
        ``bilevel_sketch.build_solve_prompt``.
        """
        lines = []
        for pred in sorted(predicates, key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in pred.types)
            line = f"  {pred.name}({type_sig})"
            if pred.natural_language_assertion is not None:
                names = [t.name for t in pred.types]
                line += f" - {pred.natural_language_assertion(names)}"
            lines.append(line)
        return "\n".join(lines)

    def _make_evaluate_trajectory_fn(self) -> Any:
        """Build the ``evaluate_trajectory`` helper exposed in the synthesis
        exec namespace (next to ``is_goal_state``).

        The returned function scores a concrete state sequence with the
        task's env-defined ``TaskEvaluator`` and returns only the public
        pair (dict of reward/solved) - never the evaluator itself, and
        never the certificate's internal legitimacy verdict or reason
        (the agent infers the scoring rules from the stated objective
        and the outcomes it observes; goal-atom termination it can
        check itself via ``is_goal_state``). ``actions`` may be
        ``Action`` objects (labeled via their producing options),
        pre-built ``(option_name, object_names[, params])`` labels, or
        ``None`` (kinematics-only scoring).
        """
        tasks = self._train_tasks

        def evaluate_trajectory(states: Sequence[State],
                                actions: Optional[Sequence[Any]] = None,
                                task_idx: int = 0) -> Dict[str, Any]:
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
            return {
                "reward": verdict["reward"],
                "solved": verdict["solved"],
            }

        return evaluate_trajectory

    @staticmethod
    def _format_trajectory_listing(
            trajectories: List[LowLevelTrajectory]) -> str:
        """Render a per-trajectory listing with provenance tags.

        Each interaction trajectory shows the simulator / predicates
        snapshot used to generate the plan that collected it (if
        tracked). Demo trajectories list as ``demo``. Listed in the same
        order the agent sees them via the ``trajectories`` var.
        """
        if not trajectories:
            return ""
        lines = ["Trajectory roster (matches the `trajectories` list):"]
        for idx, traj in enumerate(trajectories):
            kind = "demo" if traj.is_demo else "interaction"
            try:
                task_str = f"task {traj.train_task_idx}"
            except AssertionError:
                task_str = "task ?"
            provenance: List[str] = []
            sim_v = traj.source_simulator_version
            preds_v = traj.source_predicates_version
            if sim_v:
                provenance.append(f"sim {sim_v}")
            if preds_v:
                provenance.append(f"predicates {preds_v}")
            tail = (f" - generated using {', '.join(provenance)}"
                    if provenance else "")
            if traj.env_reward is not None:
                solved = int(
                    bool(traj.env_terminated) and not traj.env_rejected)
                tail += (f" - env reward={traj.env_reward:.2f} "
                         f"(solved={solved})")
            lines.append(f"  [{idx}] {kind}, {task_str}{tail}")
        return "\n".join(lines) + "\n"

    def _format_objective_block(self) -> str:
        """The env's public task objective (reward form), or empty.

        Emitted when a train task's evaluator states an objective. The
        statement is public by design: it contains the reward FORM
        (success condition + costs), never oracle quantities like the
        true minimum block count.
        """
        description = next(
            (t.evaluator.objective_description()
             for t in self._train_tasks if t.evaluator is not None
             and t.evaluator.objective_description()), "")
        return learn_prompts.render_objective_block(description)

    def _format_prior_state_block(self, base: str) -> str:
        """Tell the agent about any simulator/predicates left over from a
        previous learning cycle.

        Returns a paragraph the agent can act on (read the files first
        and treat this cycle as incremental refinement) or an empty
        string if no prior state exists. The base sandbox dir is scanned
        for ``simulator.py`` / ``predicates.py``.
        """
        prior: List[str] = []
        if os.path.isfile(os.path.join(base, "simulator.py")):
            prior.append("`./simulator.py`")
        if os.path.isfile(os.path.join(base, "predicates.py")):
            prior.append("`./predicates.py`")
        return learn_prompts.render_prior_state_block(prior)

    @staticmethod
    def _load_simulator_from_module_file(
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

        ns: Dict[str, Any] = {
            "np": np,
            "ParamSpec": ParamSpec,
            "trajectories": trajectories or [],
        }
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            exec(code, ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to exec %s.", path, exc_info=True)
            return None, None, None, None

        rules, specs, features = read_simulator_components(ns)
        # A physics-only artifact (PHYSICAL_PARAMS with no residual rules)
        # is valid: the base sim carries all the dynamics once its
        # parameters are identified, so rules/specs default to empty.
        physics_only = read_physical_param_specs(ns) is not None
        if rules is None:
            if not physics_only:
                logger.warning("Simulator file %s missing RESIDUAL_RULES.",
                               path)
                return None, None, None, ns
            rules = []
        if specs is None:
            if not physics_only:
                logger.warning("Simulator file %s missing PARAM_SPECS.", path)
                return None, None, None, ns
            specs = []

        logger.info("Loaded %d rules, %d param specs from %s%s.", len(rules),
                    len(specs), path,
                    " (physics-only artifact)" if physics_only else "")
        return rules, specs, features, ns

    # ── Static helpers ───────────────────────────────────────────

    def _write_structs_reference(self) -> str:
        """Write key struct sources to the sandbox; return the agent-visible
        path."""
        # pylint: disable=import-outside-toplevel,reimported
        from predicators.structs import Action as _Action
        from predicators.structs import LowLevelTrajectory as _LLT
        from predicators.structs import Object as _Object
        from predicators.structs import State as _State
        from predicators.structs import Type as _Type

        source = "\n\n".join(
            inspect.getsource(cls)
            for cls in [_Type, _Object, _State, _Action, _LLT])

        base = self._tool_context.sandbox_dir or self._get_log_dir()
        ref_dir = os.path.join(base, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, "structs.py")
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Same backend-dependent agent-visible path mapping as
        # _resolve_synthesis_paths.
        if CFG.agent_sdk_use_local_sandbox:
            return "./reference/structs.py"
        if self._tool_context.sandbox_dir:
            return "/sandbox/reference/structs.py"
        return ref_path

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
        # Same backend-dependent path mapping as _write_structs_reference.
        if CFG.agent_sdk_use_local_sandbox:
            return [f"./reference/base_sim/{n}" for n in names]
        if CFG.agent_sdk_use_docker_sandbox:
            return [f"/sandbox/reference/base_sim/{n}" for n in names]
        return []

    @staticmethod
    def _extract_obs_triples(
        trajectories: List[LowLevelTrajectory],
    ) -> List[Tuple[State, Action, State]]:
        """Extract observed (s_t, action_t, s_{t+1}) triples."""
        triples: List[Tuple[State, Action, State]] = []
        for traj in trajectories:
            for i in range(len(traj.actions)):
                triples.append(
                    (traj.states[i], traj.actions[i], traj.states[i + 1]))
        return triples

    def _recreate_base_env(self) -> None:
        """Reconnect after a PyBullet physics-server crash."""
        try:
            # dispose_env releases the secondary probe world too; the
            # domino override disposes it BEFORE the (possibly dead)
            # main client so a raise here cannot strand it.
            dispose_env(self._base_env)
        except Exception:  # pylint: disable=broad-except  # client may already be dead
            pass
        logging.warning(
            "PyBullet physics client crashed; recreating base env "
            "(use_gui=%s).", CFG.option_model_use_gui)
        self._base_env = create_new_env(CFG.env,
                                        do_cache=False,
                                        use_gui=CFG.option_model_use_gui,
                                        skip_residual_dynamics=True)
        # A fresh env comes up with built-in physics; re-assert any
        # identified physical params (the in-place override does not
        # survive env recreation).
        if self._identified_physical_params:
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

    @contextmanager
    def _fresh_validation_env_scope(
        self,
        physical_overrides: Optional[Dict[str,
                                          float]] = None) -> Iterator[None]:
        """Run the option model on a freshly constructed base env.

        ``physical_overrides`` (the capture gate's physics-margin
        rollouts) is applied to the fresh env ON TOP of the identified
        params, so the rollout runs at a perturbed physics; the shared
        session env is never touched.

        Installed as ``ToolContext.validation_env_scope`` so
        ``submit_plan``'s capture-validation rollouts each sample
        a fresh physics world. The shared ``_base_env``'s reset cannot
        reconstruct state exactly (solver warm-start state, velocity
        residuals, near-matching bodies skipped by the reconstruction diff
        - the same mechanism measured in :func:`rollout_states`), so
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
        fresh = create_new_env(CFG.env,
                               do_cache=False,
                               use_gui=False,
                               skip_residual_dynamics=True)
        if self._identified_physical_params:
            fresh.apply_physical_param_overrides(
                self._identified_physical_params)
        if physical_overrides:
            fresh.apply_physical_param_overrides(dict(physical_overrides))
        prev_env = self._base_env
        # Typed Any: sim_env and _simulator are dynamic attributes not on
        # _OptionModelBase.
        model: Any = self._option_model
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

    def _build_synthesis_system_prompt(self) -> str:
        """Compose the synthesis system prompt from the templates.

        Per-instance choices: the rule signature (flag-gated on
        ``CFG.partially_observable``, which also swaps the env's
        observation and the GT simulator module, so prompt and world
        never disagree; under the flag only the recurrent 5-arg form is
        shown), the optional PHYSICAL_PARAMS section (env parameter
        menu), the scene-visualization hint, and the subclass extras.
        """
        return learn_prompts.build_learn_system_prompt(
            partially_observable=CFG.partially_observable,
            residual_rule_signature=self._residual_rule_signature(),
            scene_viz_hint=self._scene_viz_hint(),
            physical_params_section=self._physical_params_prompt_section(),
            extra_sections=self._extra_synthesis_system_prompt_sections(),
            latent_extra_sections=self._extra_synthesis_latent_sections(),
            workflow_extra=self._synthesis_workflow_extra(),
            declared_params_only=CFG.agent_sim_learn_declared_params_only,
        )

    def _synthesis_workflow_extra(self) -> str:
        """Extra text appended to the Workflow list.

        Subclasses with additional deliverables (e.g. invented
        predicates) extend the workflow here so the numbered list stays
        the single authoritative loop description.
        """
        return ""

    @staticmethod
    def _scene_viz_hint() -> str:
        """The find-the-anchor-offset sentence.

        The probe is unconditional in synthesis sessions, so the hint
        always names its staging + overlay surface.
        """
        return ("use the `sim` probe in `run_python`: "
                "`sim.reset(task_idx=..., "
                "mods={...})` to stage a representative state from each "
                "bucket and `sim.render(label, annotations=[...])` to "
                "overlay, on one render, the recorded origin and the "
                "positions where the effect did vs. did not fire")

    def _physical_params_prompt_section(self) -> str:
        """Markdown for the optional PHYSICAL_PARAMS (system-ID) block.

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
        return learn_prompts.render_physical_params_section(info)

    def _residual_rule_signature(self) -> str:
        """The ``def`` line used in the geometric-gate example.

        Matches the canonical rule signature the prompt advertises
        (``CFG.partially_observable`` selects it) so the worked example
        doesn't contradict it.
        """
        if CFG.partially_observable:
            return ("def residual_rule(observation, latent, history, "
                    "updates, params):")
        return "def residual_rule(state, updates, params):"
