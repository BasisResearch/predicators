"""Shared mutable state between the approach and the MCP tools."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Set, Tuple

from predicators import utils
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


@dataclass
class ToolContext:
    """Shared mutable state between the approach and MCP tools."""
    types: Set[Type] = field(default_factory=set)
    predicates: Set[Predicate] = field(default_factory=set)
    processes: Set[CausalProcess] = field(default_factory=set)
    options: Set[ParameterizedOption] = field(default_factory=set)
    train_tasks: List[Task] = field(default_factory=list)
    offline_trajectories: List[LowLevelTrajectory] = field(
        default_factory=list)
    online_trajectories: List[LowLevelTrajectory] = field(default_factory=list)
    example_state: Optional[State] = None
    option_model: Optional[_OptionModelBase] = None
    # Synthesis-session override for the run_python probe: a lazy
    # builder over the CANDIDATE simulator.py (fresh LM fit, cached
    # until the file changes). When set, BeliefProbe executes against it
    # instead of ``option_model`` - which during synthesis is the stale
    # pre-synthesis model (real physics on cycle 1: a live-env leak).
    # Installed by the sim-learning approach around its synthesis
    # session only; None everywhere else (solve sessions, and
    # oracle-sim-program sampler sessions where ``option_model`` IS the
    # deployed belief model).
    probe_option_model_provider: Optional[Callable[[],
                                                   _OptionModelBase]] = None
    # The ``sim.fit`` backend for synthesis sessions: fits the candidate
    # simulator's PARAM_SPECS against the recorded data and returns the
    # report text (see ``SynthesisToolkit.fit_runner``). None in solve
    # sessions - the deployed belief model is fixed there, so the probe
    # rejects ``fit`` calls.
    probe_fit_provider: Optional[Callable[..., str]] = None
    # Standalone program models cannot request engine replay diagnostics.
    probe_engine_available: bool = True
    # ``sim`` calls this session refuses: method names (``refine``,
    # ``predicates``, ``render``, ...) plus the ``run`` modes ``trials``
    # (more than one trial) and ``render`` (per-step scene images, which
    # need the real engine). A comparison arm lists what its surface
    # withholds; the probe raises on a listed call instead of serving it.
    probe_disabled: FrozenSet[str] = frozenset()
    # Synthesis-session loader behind ``sim.predicates()``: it reloads
    # the agent-authored predicates.py fresh, installs the result into
    # the approach so refinement sees the draft, and returns the report
    # text. Empty in sessions that do not offer the artifact.
    probe_artifact_loaders: Dict[str,
                                 Callable[...,
                                          str]] = field(default_factory=dict)
    # Synthesis sessions: which parameter values the candidate probe
    # model is running with - "fitted (<version>)" after a canonical
    # ``sim.fit`` of the current simulator.py, or an UNFITTED notice
    # (carried-over / declared init values) that every probe result
    # surfaces until the agent fits the file. None outside synthesis.
    probe_param_status: Optional[str] = None
    # Installed by the model arm under continual_require_model_on_test:
    # returns the reason the skill tools must refuse right now (no
    # deployable, fitted simulator.py on a test level), or None when
    # a skill may run. Nothing is charged for a refusal.
    skill_gate: Optional[Callable[[], Optional[str]]] = None
    # Installed by the model arm under continual_skill_preflight: given
    # the plan text of a skills_invoke / skills_execute_plan request,
    # rehearses it in the arm's sim from the last real observation and
    # returns the refusal text (the controller's diagnostic) when a
    # skill fails there, or None when the request may run. Nothing is
    # charged for a refusal; force=true on the request skips it.
    skill_preflight: Optional[Callable[[str], Optional[str]]] = None
    # Optional nonblocking observer of requested and executed actions.
    # Kept separate from the legacy refusal callback, including raw routes.
    execution_audit: Optional[Any] = None
    # Synthesis-session ``sim.residuals`` backend: computes the
    # per-feature residual report for the current simulator.py rules
    # (see ``SynthesisToolkit.residuals_runner``). None in solve
    # sessions - residuals are a learning diagnostic.
    probe_residuals_provider: Optional[Callable[..., str]] = None
    # Replay the current model on recorded actions without fitting or trimming.
    probe_validation_provider: Optional[Callable[..., str]] = None
    # The ``sim.score`` backend of a program-world-model synthesis
    # session (particle-filter pseudo-likelihood of the candidate
    # world_model.py on the recorded data); None everywhere else.
    probe_score_provider: Optional[Callable[..., str]] = None
    # Active-experiment info-gain scorer, synced from the learning
    # approach when info-seeking exploration is on:
    # ``(state, atoms) -> disagreement``. The probe passes it into
    # refinement so continuous-parameter search prefers candidates that
    # straddle the learned model's decision boundaries. None ⇒ plain
    # feasibility search (default).
    atom_disagreement_fn: Optional[Callable[[State, Any], float]] = None
    current_task: Optional[Task] = None
    # The last real observation of the level in progress (continual
    # play: every env tool result and the session query refresh it), so
    # ``sim.reset(current=True)`` can start a rollout from it.
    current_observation: Optional[State] = None
    # Refresh inferred memory after a model edit/refit before a current-state
    # probe. Continual MB sessions install this; other sessions keep None.
    current_observation_provider: Optional[Callable[[], State]] = None
    execution_step_budget_provider: Optional[Callable[[], int]] = None
    # Arm-specific invariant checked before any charged continual request.
    before_real_action: Optional[Callable[[], None]] = None
    # The execution-time belief over it (observation_belief.BeliefFrame)
    # when the run carries one, so sim.run(belief_draws=K) draws from
    # the belief the agent was shown.
    current_belief: Optional[Any] = None
    # The joint belief (belief_joint_draws > 0): K joint draws
    # (theta_i, x_t_i) at the current decision point, each state carrying
    # the memory its parameters imply, and the fraction of those draws on
    # which each atom holds. Continual MB sessions install both.
    joint_draws_provider: Optional[Callable[..., List[Tuple[Dict[str, float],
                                                            State]]]] = None
    joint_fractions_provider: Optional[Callable[[], Dict[Any, float]]] = None
    # Which parameter names the env applies (the rest are rule parameters,
    # read through joint_draw_scope), the scope itself, and the current
    # episode's recorded frames and step labels for scoring a rehearsal
    # on the episode so far.
    physical_param_names_provider: Optional[Callable[[], Set[str]]] = None
    joint_draw_scope: Optional[Callable[[Dict[str, float]], Any]] = None
    episode_prefix_provider: Optional[Callable[[], Tuple[List[State],
                                                         List[Any]]]] = None
    log_dir: Optional[str] = None
    env: Optional[Any] = None  # simulator env reference (for rendering)
    image_save_dir: Optional[str] = None  # sandbox path for rendered images
    sandbox_dir: Optional[str] = None  # sandbox root directory
    gt_options_ref_path: Optional[str] = None  # sandbox-relative ref file
    show_option_source: bool = True  # set False when using GT options
    iteration_id: int = 0  # current learning iteration (outer loop)
    turn_id: int = 0  # current query/turn within the session
    # Index of the test task currently being solved (0-based), mirroring
    # main.py's ``test_task_idx``. None outside the test phase. Threaded into
    # the saved session-log filename so test queries are attributable to a task.
    test_task_idx: Optional[int] = None
    test_call_id: int = 0  # incremented per probe rollout call
    # 0-based learning cycle (matching main.py's "ONLINE LEARNING CYCLE i";
    # -1 = the offline pass) while a synthesis (learn) session is active,
    # None otherwise. Set/cleared around the synthesis query so tools that
    # label output by phase (e.g. attempt-log headers) can attribute
    # entries to the learning cycle instead of "pre-test phase".
    learn_cycle_index: Optional[int] = None
    # Managed by AgentSessionMixin: populated from
    # `_build_synthesis_mcp_tools` at session-open, reset to [] for
    # solve sessions. Approaches should not write to this directly —
    # override the builder hook instead.
    extra_mcp_tools: list = field(default_factory=list)
    # Extra Claude Agent SDK ``HookMatcher`` instances applied to the
    # next session that's started. Read once at session start, then
    # frozen for the session's lifetime. Subclasses set this before
    # opening a fresh session and clear it on close.
    extra_session_hooks: Dict[str, list] = field(default_factory=dict)
    # Agent-session phase the tools are serving: "solve" or
    # "synthesis" (the model-writing rounds). Set by the session mixin
    # when it builds the session; None before any session exists.
    phase: Optional[str] = None
    # True when the approach will track the simulator's latent block at
    # execution (code_sim_learning.latent_tracker), so latent-reading
    # atoms are evaluable on real observations and refinement must keep
    # them as Wait targets; False (bare observations) strips them.
    latent_tracking_available: bool = False
    # Fresh-physics scope for validation rollouts: a callable returning a
    # context manager. While entered, ``ctx.option_model`` simulates on a
    # freshly constructed env instance instead of the shared session env,
    # whose reset cannot reconstruct state exactly (solver warm-start
    # state, velocity residuals), making repeated rollouts correlated
    # with each other and systematically offset from the fresh real env.
    # Accepts an optional ``physical_overrides`` keyword (a param-name ->
    # value dict applied to the fresh env on top of the identified
    # params) for the physics-sweep rollouts. Installed by
    # AgentSimLearningApproach (see ``_fresh_validation_env_scope``);
    # None ⇒ probe rollouts share the session env. Gated by
    # agent_plan_validation_fresh_env.
    validation_env_scope: Optional[Callable[..., Any]] = None
    # Candidate-aware counterpart: loads the deployed candidate before
    # cloning its physics and rebinds its option model for the whole rollout.
    # Never substitute the deployed model for a synthesis candidate.
    probe_validation_env_scope: Optional[Callable[..., Any]] = None
    # Physics-margin points: a zero-arg callable returning the current grid
    # of perturbations spanning +-1 posterior sigma of the identified
    # physical params (full override dicts, ascending; empty when no fit
    # with nonzero posterior width is deployed). A callable rather than a
    # stored list so the points always track the LATEST applied fit.
    # Installed by AgentSimLearningApproach; consumed by the sim.run
    # physics sweep.
    physics_margin_provider: Optional[Callable[[], List[Dict[str,
                                                             float]]]] = None
    # Round bookkeeping (a continual play round is one attempt):
    # ``attempt_start`` is a time.monotonic() value, surfaced in tool
    # results as the budget footer's elapsed time. None ⇒ no round in
    # flight.
    attempt_start: Optional[float] = None
    # Count of full-plan belief-sim rollouts this round (probe runs,
    # trials). Reset per round; shown in the budget footer so sweeps
    # carry a visible price.
    attempt_rollout_count: int = 0
    # The run's conversation as the play tools show it in the [context]
    # line (continual protocol): the prompt size of the latest assistant
    # turn (input plus cached tokens), the assistant turns so far, the
    # compactions the SDK performed, and the window size once the CLI
    # has reported one. Fed by note_stream_entry from the sandbox
    # session's receive loop; never reset within a run.
    context_tokens: Optional[int] = None
    context_turns: int = 0
    context_compactions: int = 0
    context_window_tokens: Optional[int] = None
    # Per-call deadline for the run_python call currently executing
    # (agent_sdk_python_call_timeout); enforced at the probe's
    # checkpoints. None ⇒ no call in flight.
    python_call_deadline: Optional[float] = None
    # Adaptive info-seeking trigger (agent_explorer_info_seeking_adaptive):
    # set True the first time the probe's physics sweep finds a plan's
    # success straddling the belief interval. While True the proactive
    # info-seeking apparatus (suggest_probes ranking,
    # disagreement guidance) is active; while False, and
    # under the adaptive flag, it stays dormant so easy levels pay no
    # info-seeking step tax. Ignored unless the adaptive flag is on.
    param_sensitive_refusal_pending: bool = False

    def execution_step_budget(self) -> int:
        """The live protocol allowance, or the phased protocol's horizon."""
        if self.execution_step_budget_provider is not None:
            return self.execution_step_budget_provider()
        return utils.real_episode_step_budget(self.phase)

    def info_seeking_active(self) -> bool:
        """Whether the proactive info-seeking apparatus should run now.

        Off when info-seeking exploration is disabled outright. On
        whenever it is enabled and the adaptive flag is off (the
        original always-on behaviour). Under the adaptive flag it turns
        on only once the probe's physics sweep has found a plan whose
        success straddles the belief interval this run
        (``param_sensitive_refusal_pending``), so the agent spends real
        steps reducing uncertainty only after a fragile plan has
        actually been found.
        """
        if not CFG.agent_explorer_info_seeking:
            return False
        if not CFG.agent_explorer_info_seeking_adaptive:
            return True
        return self.param_sensitive_refusal_pending

    def note_stream_entry(self, entry: Dict[str, Any]) -> None:
        """Fold one streamed SDK entry into the context counters."""
        kind = entry.get("type")
        if kind == "assistant":
            self.context_turns += 1
            usage = entry.get("usage") or {}
            total = sum(
                int(usage.get(key) or 0)
                for key in ("input_tokens", "cache_creation_input_tokens",
                            "cache_read_input_tokens"))
            if total > 0:
                self.context_tokens = total
        elif kind == "system" and entry.get("subtype") == "compact_boundary":
            self.context_compactions += 1

    def begin_attempt(self) -> None:
        """Start a round's bookkeeping: its clock and rollout count."""
        self.attempt_rollout_count = 0
        self.attempt_start = time.monotonic()

    def pause_attempt_clock(self, seconds: float) -> None:
        """Push every armed wall-clock mark ``seconds`` into the future.

        Called by the session manager after it slept out a usage limit,
        so the wait is charged to neither the round nor the run_python
        call in flight, and the budget footer's elapsed time stays
        honest.
        """
        if seconds <= 0:
            return
        if self.attempt_start is not None:
            self.attempt_start += seconds
        if self.python_call_deadline is not None:
            self.python_call_deadline += seconds


@contextmanager
def decorrelated_rollout_seed(rollout_idx: int) -> Iterator[None]:
    """Give one repeat rollout its own motion-planner seed.

    A fresh env per rollout is NOT sufficient for independent samples:
    every stochastic step of a rollout (BiRRT sampling, IK restarts)
    reads the constant ``CFG.seed`` at call time, so N fresh-env repeats
    of the same plan are bit-identical replays and "N/N reliable" is one
    effective sample. run_20260722_204632 (domino_high_friction_turn
    seed 2) captured a plan validated 13/13 that was a coin flip on the
    real episode; every trials-mode result in that run's sessions was
    0/N or N/N, never mixed. Offsetting the seed per rollout varies the
    planned motion paths within the tolerance the planner already
    accepts - the same execution variability that separates the
    validation context from the real episode - so repeats become a real
    sample of execution noise.

    ``rollout_idx`` 0 keeps the base seed: the canonical first rollout
    stays reproducible and single-run (``trials=1``) behavior is
    unchanged. Enter this scope AFTER any fresh env is created so env
    construction (and its task-cache key) still sees the base seed.
    """
    if rollout_idx == 0:
        yield
        return
    base_seed = CFG.seed
    CFG.seed = base_seed + rollout_idx
    try:
        yield
    finally:
        CFG.seed = base_seed


@contextmanager
def absolute_rollout_seed(seed: Optional[int]) -> Iterator[None]:
    """Run a scope at an explicit motion-planner seed (None = no-op).

    The agent-facing counterpart of ``decorrelated_rollout_seed``:
    validation repeats and probe trials REPORT the planner seed each
    rollout ran at, and this scope lets a follow-up call re-run a plan
    at exactly that seed - the only way to reproduce a seed-dependent
    failure (e.g. one FLAKY validation rollout out of five) instead of
    re-sampling and hoping to draw it again. Composes with
    ``decorrelated_rollout_seed``: enter this first, and trial ``i``
    runs at ``seed + i``. Enter AFTER any fresh env is created so env
    construction (and its task-cache key) still sees the base seed.
    """
    if seed is None:
        yield
        return
    base_seed = CFG.seed
    CFG.seed = seed
    try:
        yield
    finally:
        CFG.seed = base_seed
