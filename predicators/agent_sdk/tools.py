"""Custom MCP tool definitions for the agent SDK approach."""
import contextlib
import copy
import functools
import hashlib
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, \
    Set, Tuple, Union

import numpy as np

from predicators.agent_sdk.proposal_exec import ProposalBundle, \
    build_exec_context, exec_code_safely, load_ground_samplers, \
    load_learned_samplers, validate_predicate
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, LowLevelTrajectory, Object, \
    ParameterizedOption, ParameterizedSampler, Predicate, State, Task, Type

MCP_SERVER_NAME = "predicator_tools"

# Built-in Claude tools available to the sandboxed agent.
BUILTIN_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Task",
    "TaskOutput",
    "TaskStop",
    "TaskCreate",
    "TaskGet",
    "TaskUpdate",
    "TaskList",
]

INSPECTION_TOOL_NAMES = [
    "inspect_types",
    "inspect_predicates",
    "inspect_processes",
    "inspect_options",
    "inspect_trajectories",
    "inspect_train_tasks",
    "inspect_planning_results",
    "inspect_past_proposals",
]
PROPOSAL_TOOL_NAMES = [
    "propose_types",
    "propose_predicates",
    "propose_task_augmentor",
    "propose_processes",
    "propose_options",
]
RETRACTION_TOOL_NAMES = [
    "retract_abstractions",
]
TESTING_TOOL_NAMES = [
    "evaluate_predicate_on_trajectory",
    "evaluate_option_plan",
]
PLANNING_TOOL_NAMES = [
    "generate_bilevel_plan",
    "generate_abstract_plan",
    "refine_plan_sketch",
]
SCENE_TOOL_NAMES = [
    "annotate_scene",
    "visualize_state",
]
# Solve-phase exploration: ``explore_python`` over the ProbeSim facade
# (predicators/agent_sdk/probe_api.py). Named distinctly from the
# synthesis-phase ``run_python`` (same execution core, different
# namespace) so sessions, transcripts, and log greps never conflate the
# two capabilities. Built only when CFG.agent_planner_use_explore_python
# is on - the legacy ``tool_names=None`` surface ("all static MCP
# tools") must not hand baseline arms an ungated code-execution tool.
EXPLORATION_TOOL_NAMES = [
    "explore_python",
]
# Solve-journal writing (CFG.agent_solve_use_journal): agent-authored
# lessons for future fresh-context attempts. Read side is prompt
# injection, so this is the only journal tool.
JOURNAL_TOOL_NAMES = [
    "record_journal",
]


def explore_python_replaces_tools() -> bool:
    """Whether explore_python replaces the tools it subsumes this session.

    The single definition of the tool-roster policy (visualize_state ->
    ``sim.reset(mods)`` + ``sim.render``; annotate_scene ->
    ``sim.render(annotations=...)``; refine_plan_sketch ->
    ``sim.refine``): the approaches' tool lists and every prompt surface
    read this one predicate, so the offered tools and the guidance that
    names them cannot drift apart.
    """
    return (CFG.agent_planner_use_explore_python
            and not CFG.agent_planner_explore_python_keep_replaced_tools)


ALL_TOOL_NAMES = (INSPECTION_TOOL_NAMES + PROPOSAL_TOOL_NAMES +
                  RETRACTION_TOOL_NAMES + TESTING_TOOL_NAMES +
                  PLANNING_TOOL_NAMES + SCENE_TOOL_NAMES +
                  EXPLORATION_TOOL_NAMES + JOURNAL_TOOL_NAMES)

# Names of tools returned by ``create_synthesis_tools`` (sim-learning)
# and ``create_predicate_synthesis_tools`` (predicate invention). These
# tools are produced by ``AgentSessionMixin._build_synthesis_mcp_tools``
# and joined to the static MCP set at session-open time; the constants
# exist so callers / tests can refer to them without typing the strings
# twice. ``tests/agent_sdk/test_tool_registry.py`` asserts that the
# factory outputs match these tuples.
SYNTHESIS_TOOL_NAMES = (
    "run_python",
    "report_residuals",
    "evaluate_step_fit",
    "evaluate_plan_refinement",
)
PREDICATE_SYNTHESIS_TOOL_NAMES = ("evaluate_predicate_quality", )
SAMPLER_SYNTHESIS_TOOL_NAMES = ("evaluate_sampler", )


def get_allowed_tool_list(tool_names: Optional[List[str]] = None) -> List[str]:
    """Compute the allowed_tools list for the agent SDK.

    ``tool_names`` is the caller's declared tool surface; it may mix
    static MCP names (in ``ALL_TOOL_NAMES``) with names of dynamic
    ``SdkMcpTool`` instances supplied via ``ctx.extra_mcp_tools``. We do
    not silently filter — typos surface as "unknown tool" errors from
    the SDK rather than as missing-allowlist mysteries. Passing ``None``
    keeps the legacy "all static MCP tools" default.
    """
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    names = list(ALL_TOOL_NAMES) if tool_names is None else list(tool_names)
    return [f"{prefix}{n}" for n in names]


def list_session_tool_names(
    *,
    mcp_filter: Optional[Sequence[str]] = None,
    extra_mcp_tools: Sequence[Any] = (),
    include_builtin: bool = True,
) -> Dict[str, List[str]]:
    """Return the tool names active in a session, grouped by source.

    A convenience view of "what does this agent session see?" — useful
    for logs and prompt-construction debugging. Names are bare (no
    ``mcp__predicator_tools__`` prefix); use ``get_allowed_tool_list``
    for the prefixed form Claude Agent SDK expects.

    Args:
        mcp_filter: Subset of ``ALL_TOOL_NAMES`` to keep. ``None`` (the
            default) lists every MCP tool.
        extra_mcp_tools: Synthesis tools supplied for the session
            (e.g. by ``_build_synthesis_mcp_tools``). Their names are
            read off each tool's ``name`` attribute.
        include_builtin: Whether to include the Claude built-in tools
            (``Bash``, ``Read``, ``Write``, …).

    Returns ``{"builtin": [...], "mcp": [...], "extra": [...]}``.
    """
    valid = set(ALL_TOOL_NAMES)
    if mcp_filter is None:
        mcp_names = list(ALL_TOOL_NAMES)
    else:
        mcp_names = [n for n in mcp_filter if n in valid]
    extra_names = [
        getattr(t, "name", "") for t in extra_mcp_tools
        if getattr(t, "name", "")
    ]
    out: Dict[str, List[str]] = {"mcp": mcp_names, "extra": extra_names}
    if include_builtin:
        out["builtin"] = list(BUILTIN_TOOLS)
    return out


@dataclass(frozen=True)
class PlanCapture:
    """A captured plan popped off a :class:`ToolContext` in one piece.

    Returned by :meth:`ToolContext.take_plan_capture` so consumers see
    the four ``solved_plan*`` fields as the single value they are:
    ``plan`` is falsy when nothing was captured.
    """
    plan: Optional[Any]
    sketch: Optional[Any]
    reached_goal: Optional[bool]
    eval_reward: Optional[float]


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
    # Synthesis-session override for the explore_python probe: a lazy
    # builder over the CANDIDATE simulator.py (fresh MCMC fit, cached
    # until the file changes). When set, ProbeSim executes against it
    # instead of ``option_model`` - which during synthesis is the stale
    # pre-synthesis model (real physics on cycle 1: a live-env leak).
    # Installed by the sim-learning approach around its synthesis
    # session only; None everywhere else (solve sessions, and
    # oracle-sim-program sampler sessions where ``option_model`` IS the
    # deployed belief model).
    probe_option_model_provider: Optional[Callable[[],
                                                   _OptionModelBase]] = None
    # Active-experiment info-gain scorer, synced from the learning
    # approach when info-seeking exploration is on:
    # ``(state, atoms) -> disagreement``. The agent_bilevel explorer
    # passes it into refinement so continuous-parameter search prefers
    # candidates that straddle the learned model's decision boundaries.
    # None ⇒ plain feasibility search (default).
    atom_disagreement_fn: Optional[Callable[[State, Any], float]] = None
    # Synthesized per-skill samplers (option name -> sampler), synced from
    # the learning approach when agent_sim_learn_parameterized_samplers is on.
    # The agent_bilevel explorer and synthesis tools pass these into
    # refinement so continuous-parameter search aims at each step's subgoal
    # instead of drawing uniformly. Empty ⇒ uniform sampling (default).
    parameterized_samplers: Dict[str, ParameterizedSampler] = field(
        default_factory=dict)
    current_task: Optional[Task] = None
    iteration_proposals: ProposalBundle = field(default_factory=ProposalBundle)
    planning_results: Dict[str, Any] = field(default_factory=dict)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    skill_factory_context: Dict[str, Any] = field(default_factory=dict)
    proposals_disabled: bool = False  # set True during test-time solving
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
    test_call_id: int = 0  # incremented per evaluate_option_plan call
    visualized_state: Optional[State] = None  # last state from visualize_state
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
    # Populated by AgentBilevelExplorer so learning approaches can diff
    # mental-model subgoals against real trajectories.
    # TODO(sim-learning): consume these in learn_from_interaction_results.
    last_sketch_subgoals: Optional[Any] = None
    last_sketch_options: Optional[Any] = None
    # Set by AgentBilevelExplorer per request: did the mental model reach
    # the task goal during refinement? Read by get_interaction_requests to
    # stamp InteractionRequest.mental_model_solved (None ⇒ no verdict).
    last_mental_model_solved: Optional[bool] = None
    # Sketch-line descriptions of the exploration plans already generated
    # this online-learning cycle (a cycle's requests are all generated
    # before any executes). Cleared by get_interaction_requests per cycle,
    # appended by AgentBilevelExplorer per request, and shown in the next
    # explore prompt so the agent proposes a complementary plan instead of
    # repeating the identical one for every request.
    cycle_scheduled_plans: List[str] = field(default_factory=list)
    # Digest of the latest rollout system-ID fit's weak spots
    # (unexplainable segments, unidentified/insensitive params,
    # cross-cycle conflicts), synced from the sim-learning approach.
    # The agent_bilevel explorer appends it to its experiment guidance
    # so the next exploration targets the gaps. None ⇒ no fit ran yet
    # (or it had no weak spots).
    sysid_diagnostics: Optional[str] = None
    # Set by refine_plan_sketch / evaluate_option_plan when a plan is verified
    # to reach the goal on the CURRENT solve task: the simulator-verified plan
    # (grounded options with found params) and the parallel subgoal sketch.
    # The bilevel approach returns this directly instead of re-refining, so
    # the agent's tool-validated answer is exactly what gets executed. None ⇒
    # nothing captured this query.
    solved_plan: Optional[Any] = None
    solved_sketch: Optional[Any] = None
    # Whether the captured solved_plan counts as a validated solve in its
    # belief-sim rollout(s): goal reached, evaluator-certified, and every
    # validation rollout passed. False ⇒ it was a best-effort capture (see
    # below). Cleared together with solved_plan.
    solved_plan_reached_goal: Optional[bool] = None
    # Gate for the above: only approaches that consume captured plans
    # (AgentModelBasedApproach) set this True. Keeps the open-loop
    # planner, which
    # also uses evaluate_option_plan, from recording spurious captures.
    capture_goal_reaching_plans: bool = False
    # Set (with capture_goal_reaching_plans) only for the final-submission
    # nudge after an attempt exhausted its turn budget: evaluate_option_plan
    # then captures the agent's submitted plan on the current task even if it
    # does not reach the goal, is scored a non-solve by the task evaluator,
    # or is flaky, so the approach executes the best-effort plan (for its
    # honest reward) instead of paying for another full-budget attempt. A
    # best-effort capture never displaces a validated-solve capture.
    capture_best_effort_plan: bool = False
    # Fresh-physics scope for capture-validation rollouts: a zero-arg
    # callable returning a context manager. While entered,
    # ``ctx.option_model`` simulates on a freshly constructed env instance
    # instead of the shared session env, whose reset cannot reconstruct
    # state exactly (solver warm-start state, velocity residuals), making
    # repeated rollouts correlated with each other and systematically
    # offset from the fresh real env. Installed by AgentSimLearningApproach
    # (see ``_fresh_validation_env_scope``); None ⇒ validation rollouts
    # share the session env. Gated by CFG.agent_plan_validation_fresh_env.
    validation_env_scope: Optional[Callable[[], Any]] = None
    # Capture-task keys (see ``_capture_task_key``) that have produced a
    # FLAKY rejection in evaluate_option_plan. A flaky submission is direct
    # evidence the agent is tuning in a marginal region where a lucky
    # streak can pass the base rollout gate (run_20260717_182321: a
    # 20/20-swept placement validated 3/3, then failed the real episode),
    # so subsequent captures on these tasks must clear the escalated
    # CFG.agent_plan_validation_rollouts_after_flaky gate instead.
    flaky_capture_task_keys: Set[Any] = field(default_factory=set)
    # Task-evaluator reward of the rollout that produced the current
    # solved_plan capture (None when no evaluator verdict was computed).
    # The restart loop ranks best-effort captures across attempts by it.
    # Cleared together with solved_plan.
    solved_plan_eval_reward: Optional[float] = None
    # Restart-loop attempt bookkeeping, set by AgentModelBasedApproach._solve
    # around each attempt. ``attempt_start``/``attempt_deadline`` are
    # time.monotonic() values; the deadline is enforced cooperatively by
    # the probe (every sim call) and explore_python, and surfaced in tool
    # results as a budget footer. None ⇒ no attempt in flight / no wall
    # clock. The deadline is cleared before the final-submission nudge so
    # nothing blocks the submission itself.
    attempt_index: int = 0
    attempt_start: Optional[float] = None
    attempt_deadline: Optional[float] = None
    # Count of full-plan belief-sim rollouts this attempt (probe runs,
    # trials, capture-validation repeats). Reset per attempt; shown in
    # the budget footer so sweeps carry a visible price.
    attempt_rollout_count: int = 0
    # Best submission on the current task this attempt that
    # evaluate_option_plan evaluated but refused to capture (evaluator
    # scored it a non-solve, or it was flaky), ranked by evaluator
    # reward. Reset per attempt; the journal auto-entry records it so a
    # later attempt (or the final best-effort nudge) can resubmit it
    # instead of the attempt's work vanishing with its context.
    best_uncaptured_plan_lines: Optional[List[str]] = None
    best_uncaptured_reward: Optional[float] = None
    # Per-call deadline for the explore_python call currently executing
    # (CFG.agent_sdk_explore_python_call_timeout); enforced at the same
    # probe checkpoints as attempt_deadline. None ⇒ no call in flight.
    explore_call_deadline: Optional[float] = None

    def begin_attempt(self, index: int, wall_clock: float) -> None:
        """Start restart-loop bookkeeping for solve attempt ``index``.

        Resets everything scoped to a single attempt (rollout count,
        best refused submission) and arms the wall-clock deadline
        (``wall_clock <= 0`` ⇒ no deadline). The matching teardown stays
        in ``AgentModelBasedApproach._solve``'s finally block,
        interleaved with its journal write.
        """
        self.attempt_index = index
        self.attempt_rollout_count = 0
        self.best_uncaptured_plan_lines = None
        self.best_uncaptured_reward = None
        self.attempt_start = time.monotonic()
        self.attempt_deadline = (self.attempt_start +
                                 wall_clock if wall_clock > 0 else None)

    def clear_plan_capture(self) -> None:
        """Clear the four ``solved_plan*`` fields together.

        They form one value (see :class:`PlanCapture`); clearing any of
        them individually would leave a stale mix.
        """
        self.solved_plan = None
        self.solved_sketch = None
        self.solved_plan_reached_goal = None
        self.solved_plan_eval_reward = None

    def take_plan_capture(self) -> PlanCapture:
        """Pop the captured plan, clearing it so it cannot be reused.

        The returned capture's ``plan`` is falsy when nothing was
        captured since the last clear.
        """
        capture = PlanCapture(plan=self.solved_plan,
                              sketch=self.solved_sketch,
                              reached_goal=self.solved_plan_reached_goal,
                              eval_reward=self.solved_plan_eval_reward)
        self.clear_plan_capture()
        return capture


def session_log_filename(query_count: int,
                         kind: str,
                         timestamp: str,
                         test_task_idx: Optional[int] = None,
                         ext: str = "md") -> str:
    """Build the session-log filename shared by the sandbox backends.

    Layout: ``NNN_<kind>[_task<idx>]_<timestamp>.<ext>``. The counter comes
    first so alphabetical sort matches chronological order; for test queries
    the ``_task<idx>`` segment ties the file to ``main.py``'s test task index.
    """
    suffix = ""
    if kind == "test" and test_task_idx is not None:
        suffix = f"_task{test_task_idx}"
    return f"{query_count:03d}_{kind}{suffix}_{timestamp}.{ext}"


def _text_result(text: str) -> Dict[str, Any]:
    """Helper to format a successful text result."""
    return {"content": [{"type": "text", "text": text}]}


def _error_result(text: str) -> Dict[str, Any]:
    """Helper to format an error result."""
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _capture_task_key(ctx: "ToolContext") -> Any:
    """Stable identity of the task behind a ``task_idx="current"`` capture.

    Keys ``ctx.flaky_capture_task_keys`` so a FLAKY rejection escalates
    the validation gate for later submissions on the SAME task only.
    Test-time solves are keyed by the test task index (stable across the
    sessions and replans of one task); exploration/synthesis captures
    fall back to the learning iteration, which at worst escalates
    conservatively across that cycle's tasks.
    """
    if ctx.test_task_idx is not None:
        return ("test", ctx.test_task_idx)
    return ("iter", ctx.iteration_id)


def _region_syntax_blurb() -> str:
    """The `~ [w]` region mention for tool descriptions, flag-aware.

    Advertising the syntax while ``agent_bilevel_ground_samplers`` is
    off sent every audited run through 1-3 turns of syntax guessing
    against a feature that could not work.
    """
    if CFG.agent_bilevel_ground_samplers:
        return ", incl. `~ [w]` half-width regions after a step's params"
    return (" - note `~ [w]` regions are DISABLED in this configuration "
            "and are ignored if given")


def _make_coercing_tool(tool: Callable) -> Callable:
    """Wrap the SDK ``tool`` decorator with numeric-string coercion.

    Harness-side JSON-schema validation rejects ``"0"`` for an
    ``integer`` property before the handler ever runs (agents lost
    whole tools to it - ``inspect_trajectories`` went 0-for-6 in
    run_20260717_154753 seed2), and which tools accept strings was
    inconsistent. This wrapper loosens every top-level ``integer`` /
    ``number`` property to also accept a string, then coerces the value
    back to the numeric type before the handler sees it, so handlers
    keep their exact-type assumptions.
    """

    def _coercing_tool(name: str, description: str, schema: Any) -> Callable:
        numeric_props: Dict[str, type] = {}
        loosened = schema
        if isinstance(schema, dict) and isinstance(schema.get("properties"),
                                                   dict):
            loosened = copy.deepcopy(schema)
            for prop, spec in loosened["properties"].items():
                if not isinstance(spec, dict):
                    continue
                if spec.get("type") == "integer":
                    numeric_props[prop] = int
                    spec["type"] = ["integer", "string"]
                elif spec.get("type") == "number":
                    numeric_props[prop] = float
                    spec["type"] = ["number", "string"]

        def _decorate(fn: Callable) -> Any:
            if not numeric_props:
                return tool(name, description, schema)(fn)

            @functools.wraps(fn)
            async def _wrapped(args: Dict[str, Any]) -> Dict[str, Any]:
                for prop, target in numeric_props.items():
                    val = args.get(prop)
                    if isinstance(val, str):
                        try:
                            args[prop] = target(val)
                        except ValueError:
                            return _error_result(
                                f"`{prop}` must be a "
                                f"{target.__name__}; got {val!r}.")
                return await fn(args)

            return tool(name, description, loosened)(_wrapped)

        return _decorate

    return _coercing_tool


def _make_spilling_text_result(
    sandbox_dir: Optional[str],
    *,
    subdir: str = "tool_outputs",
    agent_prefix: Optional[str] = None,
    char_limit: int = 30000,
    head_lines: int = 30,
    tail_lines: int = 30,
) -> Callable[[str], Dict[str, Any]]:
    """Build a ``_text_result``-style helper that spills oversize output.

    A tool result returned inline that exceeds the agent SDK's MCP
    tool-result token cap is truncated by the SDK and dumped to
    ``~/.claude/projects/.../tool-results/`` — *outside* the sandbox.
    The agent is then instructed to read that host path, which both
    defeats the sandbox boundary and is the only legitimate reason the
    agent ever needs to touch a path outside its sandbox.

    To remove that need, when ``sandbox_dir`` is set and ``text`` exceeds
    ``char_limit`` (kept well under the SDK cap), this writes the full
    text to ``<sandbox_dir>/<subdir>/result_NNNN.txt`` and returns a
    head/tail preview plus the in-sandbox path for the agent to
    ``Read``/``Grep``. Small results, or the no-sandbox case, are
    returned inline unchanged.

    ``agent_prefix`` is the path prefix the agent sees (``"."`` for the
    local sandbox, ``"/sandbox"`` for docker); when ``None`` a relative
    ``./<subdir>`` path is used, which resolves correctly because the
    agent's cwd is always the sandbox root.
    """
    counter = [0]
    host_dir = os.path.join(sandbox_dir, subdir) if sandbox_dir else None
    prefix = agent_prefix.rstrip("/") if agent_prefix else "."
    agent_dir = f"{prefix}/{subdir.replace(os.sep, '/')}"

    def _text(text: str) -> Dict[str, Any]:
        if host_dir is None or len(text) <= char_limit:
            return _text_result(text)
        counter[0] += 1
        os.makedirs(host_dir, exist_ok=True)
        filename = f"result_{counter[0]:04d}.txt"
        with open(os.path.join(host_dir, filename), "w",
                  encoding="utf-8") as f:
            f.write(text)
        lines = text.splitlines()
        total = len(lines)
        head = lines[:head_lines]
        tail = (lines[-tail_lines:] if total > head_lines + tail_lines else [])
        parts = [
            f"[output too large to inline: {len(text):,} chars across "
            f"{total:,} lines; full output saved to "
            f"{agent_dir}/{filename}. Use Read/Grep to inspect it.]",
            "",
            f"--- head ({len(head)} lines) ---",
            *head,
        ]
        if tail:
            omitted = total - len(head) - len(tail)
            parts.extend([
                "",
                f"... [{omitted:,} lines omitted] ...",
                "",
                f"--- tail ({len(tail)} lines) ---",
                *tail,
            ])
        return _text_result("\n".join(parts))

    return _text


# Filesystem roots that, when they prefix an absolute path outside the
# sandbox, mark it as a real escape (vs. data like "/done" printed by code).
SANDBOX_SYSTEM_ROOTS = (
    "/Users",
    "/home",
    "/root",
    "/etc",
    "/usr",
    "/opt",
    "/var",
    "/private",
    "/tmp",
    "/bin",
    "/sbin",
    "/lib",
    "/sys",
    "/proc",
    "/dev",
    "/mnt",
    "/srv",
)
# Predicators-source introspection that reaches outside the sandbox. The
# bare ``getsource`` substring also covers ``getsourcefile`` /
# ``getsourcelines``; ``inspect.getfile`` is matched explicitly so the
# generic ``getfilesystemencoding`` etc. don't false-positive.
SANDBOX_INTROSPECTION = ("getsource", "inspect.getfile", "site-packages")
# Hidden-implementation predicators imports inside agent-executed
# Python: the exec runs in-process, so ``from predicators.envs... import
# ...`` would hand the agent the real env classes and ground-truth
# constants the sandbox deliberately hides (observed in
# run_20260717_182040 seed1 turn 22). Public authoring surfaces
# (``predicators.structs``, ``predicators.utils``) stay importable -
# agent-written simulator.py code depends on them.
# The module alternation is shared with the sandbox hook script (see
# sandbox_setup.VALIDATE_SANDBOX_SCRIPT), which anchors it differently
# for shell-command strings; sharing the alternation keeps the two
# guards covering the same modules by construction.
SANDBOX_HIDDEN_MODULES_PATTERN = r"predicators\.(?:envs|ground_truth_models)\b"
_SANDBOX_PREDICATORS_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+" + SANDBOX_HIDDEN_MODULES_PATTERN, re.MULTILINE)
# Host filesystem prefixes scrubbed from tracebacks surfaced to agents:
# absolute source paths both leak the layout and invite out-of-sandbox
# probing (an audited run responded to one with ``find /``).
_HOST_REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


def _scrub_host_paths(text: str) -> str:
    """Strip host-absolute path prefixes from traceback text."""
    text = text.replace(_HOST_REPO_ROOT + os.sep, "")
    home = os.path.expanduser("~")
    if home and home != "/":
        text = text.replace(home + os.sep, "~" + os.sep)
    return text


def _budget_footer(ctx: "ToolContext", rollouts_before: int = 0) -> str:
    """``[budget]`` line appended to tool results during a solve attempt.

    Shows attempt wall-clock (elapsed, and the budget when one is set)
    and the attempt's cumulative sim-rollout count (plus this call's
    delta). Agents pace well when they can see a clock and terribly when
    they can't: the 47k-rollout single-call sweep of run_20260717_230436
    ran 7 h with zero cost feedback. Empty when no attempt is in flight.
    """
    start = ctx.attempt_start
    if start is None:
        return ""
    parts = []
    elapsed_min = (time.monotonic() - start) / 60.0
    deadline = ctx.attempt_deadline
    if deadline is not None:
        total_min = (deadline - start) / 60.0
        parts.append(f"attempt time {elapsed_min:.1f}/{total_min:.0f} min")
    else:
        parts.append(f"attempt time {elapsed_min:.1f} min")
    total_rollouts = ctx.attempt_rollout_count
    delta = total_rollouts - rollouts_before
    rollout_part = f"sim rollouts this attempt: {total_rollouts}"
    if delta > 0:
        rollout_part += f" (+{delta} this call)"
    parts.append(rollout_part)
    return "\n\n[budget] " + "; ".join(parts)


def _arm_budget_watchdog(seconds: float) -> Callable[[], None]:
    """Schedule a ProbeBudgetExceeded in the CALLING thread after ``seconds``;
    returns an idempotent disarm callable.

    ``explore_python``'s exec() runs on the event-loop thread, so
    pure-Python code that never reaches a probe checkpoint blocks every
    cooperative deadline check AND the sandbox's message-stream
    interrupt backstop - an async exception from a watchdog timer is the
    only preemption that reaches it. Delivery happens at the next
    bytecode boundary, so a long blocking C call (a physics step) defers
    it; those paths are exactly the ones the cooperative probe checks
    already cover.
    """
    # pylint: disable=import-outside-toplevel
    import ctypes
    import threading

    from predicators.agent_sdk.probe_api import ProbeBudgetExceeded

    # pylint: enable=import-outside-toplevel
    target_id = threading.get_ident()
    lock = threading.Lock()
    armed = [True]

    def _fire() -> None:
        # The lock makes fire/disarm mutually exclusive so the async
        # exception cannot be injected after the call has already
        # returned and disarmed.
        with lock:
            if not armed[0]:
                return
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(target_id),
                ctypes.py_object(ProbeBudgetExceeded))

    timer = threading.Timer(seconds, _fire)
    timer.daemon = True
    timer.start()

    def _disarm() -> None:
        with lock:
            armed[0] = False
        timer.cancel()

    return _disarm


# Path-like tokens: absolute (``/foo``) or parent-traversal (``..``/``../foo``),
# anchored at a boundary (start, whitespace, quote, ``(`` or ``=``) so we skip
# ``/`` inside URLs (preceded by ``:``), division, and ``./relative`` paths
# (which stay inside the sandbox).
_SANDBOX_PATH_RE = re.compile(
    r"""(?:^|(?<=[\s'"`(=]))((?:/|\.\.)[^\s'"`)<>|;:,]*)""")


def _screen_text_for_sandbox_escape(text: str,
                                    sandbox_dir: str) -> Optional[str]:
    """Best-effort screen of a Bash command / ``run_python`` code string.

    Returns a short deny reason if ``text`` looks like it reads outside
    ``sandbox_dir`` — an absolute or ``..`` path resolving out of the
    sandbox, or predicators-source introspection — else ``None``.

    This is a heuristic: a determined script can still escape (env vars,
    ``subprocess``, computed paths), so OS-level isolation (the docker
    sandbox) remains the only hard boundary. The equivalent self-contained
    logic for Bash lives in ``sandbox_setup.VALIDATE_SANDBOX_SCRIPT``;
    keep the two in sync.
    """
    for needle in SANDBOX_INTROSPECTION:
        if needle in text:
            return (f"'{needle}' may read predicators source outside the "
                    "sandbox; use the MCP tools and ./reference/ files")
    if _SANDBOX_PREDICATORS_IMPORT_RE.search(text):
        return ("importing predicators env/ground-truth modules would "
                "expose implementation the sandbox hides; use the "
                "provided namespace and the MCP tools / ./reference/ "
                "files")
    sandbox = os.path.realpath(sandbox_dir)
    for match in _SANDBOX_PATH_RE.finditer(text):
        token = match.group(1)
        resolved = os.path.realpath(
            token if os.path.isabs(token) else os.path.join(sandbox, token))
        if resolved == sandbox or resolved.startswith(sandbox + os.sep):
            continue
        if token.startswith("/") and not any(
                token == root or token.startswith(root + "/")
                for root in SANDBOX_SYSTEM_ROOTS):
            # Absolute but not a real filesystem path (e.g. printed data
            # like "/done") — don't flag.
            continue
        return f"path '{token}' resolves outside the sandbox directory"
    return None


def _render_scene_image(ctx: ToolContext,
                        step_label: str) -> Optional[Dict[str, Any]]:
    """Render a scene image from the pybullet env and return as content block.

    Returns an image content block dict, or None if rendering is not
    available.  Also saves the image to ``ctx.image_save_dir`` if set.
    """
    return _render_pybullet_image(ctx, step_label)


@contextlib.contextmanager
def agent_render_resolution() -> Iterator[None]:
    """Scoped camera-resolution cap for agent-facing scene renders.

    While inside the block, pybullet_camera_width/height are scaled so
    the longest side is CFG.agent_sdk_image_max_px (0 disables; never
    upscales). Every image the agent views stays in its conversation for
    the rest of the session, so pixel count directly drives per-turn
    cost - and rendering at the capped size is cheaper than rendering
    full-res and resampling. Videos render outside this scope and keep
    the full camera resolution.
    """
    max_px = CFG.agent_sdk_image_max_px
    old_w = CFG.pybullet_camera_width
    old_h = CFG.pybullet_camera_height
    if not max_px or max(old_w, old_h) <= max_px:
        yield
        return
    scale = max_px / max(old_w, old_h)
    CFG.pybullet_camera_width = max(1, round(old_w * scale))
    CFG.pybullet_camera_height = max(1, round(old_h * scale))
    try:
        yield
    finally:
        CFG.pybullet_camera_width = old_w
        CFG.pybullet_camera_height = old_h


def _render_pybullet_image(
    ctx: ToolContext,
    step_label: str,
    state: Optional[State] = None,
) -> Optional[Dict[str, Any]]:
    """Render a pybullet scene image and return as content block.

    If *state* is provided, the env is reset to that state before
    rendering. Returns an image content block dict, or None if rendering
    is not available. Also saves the image to ``ctx.image_save_dir`` if
    set.
    """
    if ctx.env is None:
        return None
    try:
        # pylint: disable=import-outside-toplevel
        from predicators.envs.pybullet_env import PyBulletEnv
        if not isinstance(ctx.env, PyBulletEnv):
            return None
    except ImportError:
        return None

    try:
        # pylint: disable=import-outside-toplevel
        import base64
        import io

        from PIL import Image as PILImage

        if state is not None:
            ctx.env._set_state(state)  # pylint: disable=protected-access

        with agent_render_resolution():
            video = ctx.env.render()
        if not video:
            return None
        rgb_array = np.asarray(video[0], dtype=np.uint8)
        img = PILImage.fromarray(rgb_array)  # type: ignore[no-untyped-call]

        # Save to sandbox if possible
        saved_path: Optional[str] = None
        if ctx.image_save_dir:
            os.makedirs(ctx.image_save_dir, exist_ok=True)
            safe_label = step_label.replace(" ", "_").replace("/", "_")
            task_tag = (f"_task{ctx.test_task_idx:03d}"
                        if ctx.test_task_idx is not None else "")
            filename = (f"iter{ctx.iteration_id:03d}"
                        f"{task_tag}"
                        f"_test{ctx.test_call_id:03d}"
                        f"_{safe_label}.png")
            saved_path = os.path.join(ctx.image_save_dir, filename)
            img.save(saved_path)
            logging.info("Saved scene image to %s", saved_path)

        # Encode as base64 for inline return
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        block: Dict[str, Any] = {
            "type": "image",
            "data": b64,
            "mimeType": "image/png"
        }
        if saved_path:
            block["saved_path"] = saved_path
        return block
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to render scene image: %s", e)
        return None


def _draw_pybullet_annotation(annotation: Dict[str, Any],
                              physics_client_id: int) -> List[int]:
    """Draw a single annotation as a temporary visual body in PyBullet.

    Uses createVisualShape + createMultiBody so annotations render in
    getCameraImage (unlike addUserDebugLine which only shows in GUI).
    Returns a list of body IDs for cleanup via removeBody.
    """
    import pybullet as p  # pylint: disable=import-outside-toplevel

    body_ids: List[int] = []
    ann_type = annotation["type"]
    color = annotation.get("color", [1, 0, 0])
    rgba = list(color) + [1.0] if len(color) == 3 else list(color)

    if ann_type == "marker":
        pos = annotation["position"]
        size = annotation.get("size", 0.015)
        vis = p.createVisualShape(p.GEOM_SPHERE,
                                  radius=size,
                                  rgbaColor=rgba,
                                  physicsClientId=physics_client_id)
        body = p.createMultiBody(baseVisualShapeIndex=vis,
                                 basePosition=pos,
                                 physicsClientId=physics_client_id)
        body_ids.append(body)

    elif ann_type == "line":
        from_pt = np.array(annotation["from"], dtype=float)
        to_pt = np.array(annotation["to"], dtype=float)
        diff = to_pt - from_pt
        length = float(np.linalg.norm(diff))
        if length < 1e-6:
            return body_ids
        radius = annotation.get("size", 0.005)
        mid = ((from_pt + to_pt) / 2).tolist()
        # Align cylinder z-axis with line direction
        direction = diff / length
        # Quaternion from [0,0,1] to direction
        up = np.array([0.0, 0.0, 1.0])
        cross = np.cross(up, direction)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.dot(up, direction))
        if cross_norm < 1e-6:
            quat = [0, 0, 0, 1] if dot > 0 else [1, 0, 0, 0]
        else:
            cross /= cross_norm
            angle = np.arctan2(cross_norm, dot)
            half = angle / 2
            s = np.sin(half)
            quat = [cross[0] * s, cross[1] * s, cross[2] * s, np.cos(half)]
        vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=radius,
                                  length=length,
                                  rgbaColor=rgba,
                                  physicsClientId=physics_client_id)
        body = p.createMultiBody(baseVisualShapeIndex=vis,
                                 basePosition=mid,
                                 baseOrientation=quat,
                                 physicsClientId=physics_client_id)
        body_ids.append(body)

    elif ann_type == "rectangle":
        min_c = annotation["min_corner"]
        max_c = annotation["max_corner"]
        z = min_c[2]
        radius = annotation.get("size", 0.005)
        corners = [
            [min_c[0], min_c[1], z],
            [max_c[0], min_c[1], z],
            [max_c[0], max_c[1], z],
            [min_c[0], max_c[1], z],
        ]
        for i in range(4):
            edge = {
                "type": "line",
                "from": corners[i],
                "to": corners[(i + 1) % 4],
                "color": color,
                "size": radius,
            }
            body_ids.extend(_draw_pybullet_annotation(edge, physics_client_id))

    return body_ids


def _format_object_poses(state: State) -> str:
    """Format object positions from state for diagnostic output."""
    pose_lines = []
    for obj in sorted(state, key=str):
        feats = obj.type.feature_names
        parts = [f"{obj.name}:{obj.type.name}"]
        for f in ("x", "y", "z"):
            if f in feats:
                parts.append(f"{f}={state.get(obj, f):.3f}")
        for f in ("rot", "yaw"):
            if f in feats:
                parts.append(f"{f}={state.get(obj, f):.3f}")
        if "is_held" in feats:
            parts.append(f"held={int(state.get(obj, 'is_held'))}")
        if len(parts) > 1:  # has at least one spatial feature
            pose_lines.append("  " + " ".join(parts))
    return "\n".join(pose_lines)


def _apply_state_modifications(
        state: State,
        modifications: List[Dict[str, Any]]) -> Tuple[State, List[str], str]:
    """Apply ``[{object, features}]`` overrides to a copy of ``state``.

    Shared by ``visualize_state`` and ``ProbeSim.reset``
    (explore_python) so both accept the same modification format.
    Returns ``(modified_state, summaries, error)``; ``error`` is ``""``
    on success.
    """
    modified_state = state.copy()
    obj_lookup = {o.name: o for o in modified_state}
    summaries: List[str] = []
    for mod in modifications:
        obj_name = mod.get("object", "")
        features = mod.get("features", {})
        if obj_name not in obj_lookup:
            available = sorted(obj_lookup.keys())
            return modified_state, summaries, (f"Unknown object '{obj_name}'. "
                                               f"Available: {available}")
        obj = obj_lookup[obj_name]
        for feat_name, value in features.items():
            try:
                modified_state.set(obj, feat_name, value)
                summaries.append(f"  {obj_name}.{feat_name} = {value}")
            except Exception as e:  # pylint: disable=broad-except
                return modified_state, summaries, (
                    f"Failed to set {obj_name}.{feat_name}: {e}")
    return modified_state, summaries, ""


def evaluate_states_with(evaluator: Any,
                         states: Sequence[State],
                         step_options: Optional[Sequence[Any]],
                         sim_env: Optional[Any] = None) -> Dict[str, Any]:
    """Score a state/option-label sequence with a task's ``TaskEvaluator``.

    The single verdict surface: only booleans/scalars/reasons leave this
    function, never the evaluator object. Verdicts on belief-sim
    rollouts are exactly as trustworthy as the belief sim itself -
    including ``sim_env``, the belief env backing the rollout, passed
    through so physics-needing certificates (the domino counterfactual
    push probe) can probe with the same belief physics.
    ``legitimate``/``reason`` are HARNESS-INTERNAL (capture gating,
    logs), and ``terminated`` is agent-computable from the public goal
    atoms: agent-facing surfaces expose only the public (solved,
    reward) pair - the standard RL end-of-episode observables - so the
    agent must infer the scoring rules from the stated objective and
    the outcomes its rollouts earn.
    """
    ok, reason = evaluator._certify(states, step_options, sim_env=sim_env)  # pylint: disable=protected-access
    return {
        "terminated": evaluator.terminated(states[-1]),
        "reward": evaluator.reward(states, step_options, sim_env=sim_env),
        "solved": evaluator.solved(states, step_options, sim_env=sim_env),
        "legitimate": ok,
        "reason": reason,
    }


def make_solved_check(
    evaluator: Any,
    sim_env: Optional[Any],
    on_reject: Optional[Callable[[float], None]] = None
) -> Callable[[List[State], List[Any], bool], Tuple[bool, str]]:
    """Build the evaluator gate used inside refinement searches.

    One policy for every surface (the MCP ``refine_plan_sketch`` and
    ``ProbeSim.refine``), so identical parameters can never get
    contradictory verdicts across tools:
    - a coarse rollout (option-boundary states only) never blocks, the
      same rule the capture path applies (a coarse certificate can
      falsely reject a legitimate cascade);
    - evaluator exceptions never block (fail-open, logged) - a flaky
      certificate must not abort a search mid-budget;
    - a non-terminated verdict never blocks (the goal-atom check is the
      caller's job; the gate only vetoes certified-non-solves).
    ``on_reject`` is called with the rejected verdict's reward.
    """

    def solved_check(states: List[State], labels: List[Any],
                     coarse: bool) -> Tuple[bool, str]:
        if coarse:
            return True, ""
        try:
            v = evaluate_states_with(evaluator,
                                     states,
                                     labels,
                                     sim_env=sim_env)
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("In-search solved gate failed: %s", e)
            return True, ""
        if v["solved"] or not v["terminated"]:
            return True, ""
        if on_reject is not None:
            on_reject(v["reward"])
        return False, f"solved=False, reward={v['reward']:.2f}"

    return solved_check


def _format_evaluator_verdict(verdict: Dict[str, Any],
                              *,
                              coarse: bool = False) -> str:
    """One report line for an evaluator verdict on a belief-sim rollout.

    Emits only the public (solved, reward) pair; the certificate's
    legitimacy bool and reason stay harness-internal, and goal-atom
    termination is already reported (and agent-computable) separately.
    """
    line = (f"Task evaluator (belief-sim rollout - trustworthy only insofar "
            f"as your simulator is): solved={verdict['solved']}, "
            f"reward={verdict['reward']:.2f}")
    if coarse:
        line += ("\n  NOTE: per-step states were unavailable for part of the "
                 "rollout, so the verdict is coarse (computed on "
                 "option-boundary states only).")
    return line


def _resolve_task_evaluator(ctx: ToolContext, task_idx: Union[int, str,
                                                              None]) -> Any:
    """The evaluator of a tool's referenced task (``Task.evaluator``), or None.

    ``task_idx`` follows the tools' convention: an int indexes the train
    tasks; ``"current"``/None means the current solve/explore task.
    """
    if isinstance(task_idx, int):
        if 0 <= task_idx < len(ctx.train_tasks):
            return ctx.train_tasks[task_idx].evaluator
        return None
    if ctx.current_task is not None:
        return ctx.current_task.evaluator
    return None


def _ground_samplers_path(ctx: ToolContext) -> Optional[str]:
    """Host path of the agent-editable ``ground_samplers.py``.

    Resolves the sandbox base the same way the sampler-learning mixin
    does: the local sandbox lives under ``<log_dir>/sandbox``, the
    docker sandbox at ``ctx.sandbox_dir``, else the log dir itself.
    """
    if CFG.agent_sdk_use_local_sandbox and ctx.log_dir:
        base: Optional[str] = os.path.abspath(
            os.path.join(ctx.log_dir, "sandbox"))
    elif ctx.sandbox_dir:
        base = ctx.sandbox_dir
    else:
        base = ctx.log_dir
    if not base:
        return None
    return os.path.join(base, "ground_samplers.py")


def _load_ground_sampler_fns(
        ctx: ToolContext) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load named ground samplers for ``~ my_sampler`` sketch references.

    Reads ``ground_samplers.py`` fresh (the agent edits it between
    calls) and validates its ``GROUND_SAMPLERS`` dict. Returns ``(fns,
    error)``: a missing file, or the feature being disabled, is simply
    ``({}, None)``; a broken file returns an error message for the agent
    so it can fix the code instead of silently sampling uniformly.
    """
    if not CFG.agent_bilevel_ground_samplers:
        return {}, None
    path = _ground_samplers_path(ctx)
    if path is None or not os.path.isfile(path):
        return {}, None
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    exec_ctx = build_exec_context(
        types=ctx.types,
        predicates=ctx.predicates
        | ctx.iteration_proposals.proposed_predicates,
        options=ctx.options | ctx.iteration_proposals.proposed_options)
    fns, warnings, err = load_ground_samplers(code, exec_ctx)
    if err is not None:
        return {}, f"Error loading {path}:\n{err}"
    for warning in warnings:
        logging.warning("ground_samplers.py: %s", warning)
    return fns, None


def _belief_rollout_verdict(
        ctx: ToolContext, task: Task, task_idx: Union[int, str, None],
        grounded_plan: List[Any],
        predicates: Set[Predicate]) -> Optional[Tuple[Dict[str, Any], bool]]:
    """Execute ``grounded_plan`` in the belief sim and score it with the task's
    evaluator, returning ``(verdict, coarse)`` or None.

    Used by ``refine_plan_sketch``, whose internal refinement rollouts
    don't expose per-step states; costs one extra plan rollout. Fully
    failure-tolerant: any problem returns None.
    """
    evaluator = _resolve_task_evaluator(ctx, task_idx)
    if evaluator is None or not grounded_plan or ctx.option_model is None:
        return None
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk import bilevel_sketch
    states: List[State] = [task.init]
    labels: List[Any] = []
    coarse = False

    def _collect(_i: int, outcome: Any) -> None:
        nonlocal coarse
        if outcome.post_state is None:
            return
        opt = outcome.option
        label = (opt.name, tuple(o.name for o in opt.objects),
                 tuple(float(p) for p in opt.params))
        step_traj = getattr(ctx.option_model, "last_trajectory", None)
        if step_traj is not None and len(step_traj.states) >= 2:
            states.extend(step_traj.states[1:])
            labels.extend([label] * len(step_traj.actions))
        else:
            states.append(outcome.post_state)
            labels.append(label)
            coarse = True

    try:
        bilevel_sketch.execute_plan_forward(task,
                                            grounded_plan,
                                            ctx.option_model,
                                            predicates=predicates,
                                            on_step=_collect,
                                            stop_on_failure=True)
        if len(states) < 2:
            return None
        verdict = evaluate_states_with(evaluator,
                                       states,
                                       labels,
                                       sim_env=getattr(ctx.option_model,
                                                       "sim_env", None))
        return verdict, coarse
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("Belief-rollout evaluator verdict failed: %s", e)
        return None


def _belief_rollout_verdict_line(ctx: ToolContext, task: Task,
                                 task_idx: Union[int, str, None],
                                 grounded_plan: List[Any],
                                 predicates: Set[Predicate]) -> Optional[str]:
    """Format the belief-rollout evaluator verdict as a report line."""
    scored = _belief_rollout_verdict(ctx, task, task_idx, grounded_plan,
                                     predicates)
    if scored is None:
        return None
    verdict, coarse = scored
    return _format_evaluator_verdict(verdict, coarse=coarse)


def _save_option_to_sandbox(ctx: ToolContext, option_name: str,
                            code: str) -> Optional[str]:
    """Save option source code to sandbox/proposed_code/<name>.py.

    Returns the relative path (e.g. ``./proposed_code/Pick.py``) or None
    if the sandbox directory is not set.
    """
    if ctx.sandbox_dir is None:
        return None
    proposed_dir = os.path.join(ctx.sandbox_dir, "proposed_code")
    os.makedirs(proposed_dir, exist_ok=True)
    filename = f"{option_name}.py"
    filepath = os.path.join(proposed_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    return f"./proposed_code/{filename}"


def _build_inspection_tools(ctx: ToolContext, _text_result: Callable,
                            tool: Callable) -> Dict[str, Any]:
    """Read-only inspection tools (views over ToolContext state)."""

    @tool("inspect_types", "List all object types and their features", {})
    async def inspect_types(_args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for t in sorted(ctx.types, key=lambda t: t.name):
            features = ", ".join(
                t.feature_names) if t.feature_names else "(no features)"
            parent_str = f" (parent: {t.parent.name})" if t.parent else ""
            lines.append(f"- {t.name}{parent_str}: [{features}]")
        if not lines:
            return _text_result("No types defined.")
        return _text_result("Current types:\n" + "\n".join(lines))

    @tool("inspect_predicates",
          "List all predicates and their type signatures", {})
    async def inspect_predicates(_args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for p in sorted(ctx.predicates, key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in p.types)
            lines.append(f"- {p.name}({type_sig})")
        if not lines:
            return _text_result("No predicates defined.")
        return _text_result("Current predicates:\n" + "\n".join(lines))

    @tool("inspect_processes",
          "List all processes with conditions, effects, and delays", {})
    async def inspect_processes(_args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for proc in sorted(ctx.processes, key=lambda p: p.name):
            conds = ", ".join(str(a) for a in sorted(proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(proc.add_effects))
            dels = ", ".join(str(a) for a in sorted(proc.delete_effects))
            lines.append(f"- {proc.name}\n"
                         f"    Conditions: {{{conds}}}\n"
                         f"    Add effects: {{{adds}}}\n"
                         f"    Delete effects: {{{dels}}}\n"
                         f"    Delay: {proc.delay_distribution}")
        if not lines:
            return _text_result("No processes defined.")
        return _text_result("Current processes:\n" + "\n".join(lines))

    @tool(
        "inspect_options",
        "List all options, or inspect a specific option in detail. "
        "When given an option_name, saves source code to "
        "./proposed_code/<name>.py in the sandbox for you to Read.",
        {
            "type": "object",
            "properties": {
                "option_name": {
                    "type":
                    "string",
                    "description":
                    "Name of a specific option to inspect. Saves its "
                    "source code to ./proposed_code/<name>.py. "
                    "Omit to list all options.",
                },
            },
        },
    )
    async def inspect_options(args: Dict[str, Any]) -> Dict[str, Any]:
        option_name = args.get("option_name")

        if option_name is None:
            # List all options (existing behavior)
            lines = []
            for opt in sorted(ctx.options, key=lambda o: o.name):
                type_sig = ", ".join(t.name for t in opt.types)
                dim = (opt.params_space.shape[0]
                       if opt.params_space.shape else 0)
                lines.append(f"- {opt.name}({type_sig}), params_dim={dim}")
            if not lines:
                return _text_result("No options defined.")
            return _text_result("Current options:\n" + "\n".join(lines))

        # Detailed inspection of a specific option
        opt_map = {o.name: o for o in ctx.options}
        if option_name not in opt_map:
            return _error_result(f"Unknown option '{option_name}'. "
                                 f"Available: {sorted(opt_map.keys())}")

        opt = opt_map[option_name]
        type_sig = ", ".join(t.name for t in opt.types)
        dim = opt.params_space.shape[0] if opt.params_space.shape else 0

        lines = [f"## {opt.name}({type_sig})", ""]

        # Params space
        if dim > 0:
            lines.append(f"params_dim: {dim}")
            lines.append(f"params_low:  {opt.params_space.low.tolist()}")
            lines.append(f"params_high: {opt.params_space.high.tolist()}")
            if opt.params_description:
                lines.append(f"params_desc: {list(opt.params_description)}")
        else:
            lines.append("params_dim: 0 (no continuous parameters)")

        # Source code — point to existing file or reference
        if ctx.show_option_source:
            lines.append("")
            code_path = os.path.join(ctx.sandbox_dir, "proposed_code",
                                     f"{option_name}.py") \
                if ctx.sandbox_dir else None

            if code_path and os.path.exists(code_path):
                # Already saved (e.g. from propose_options)
                lines.append(
                    f"Source code: `./proposed_code/{option_name}.py`")
            elif ctx.gt_options_ref_path:
                # GT options — point to reference file instead of extracting
                lines.append(f"Definition code: `{ctx.gt_options_ref_path}` "
                             f"(search for \"{option_name}\")")
            else:
                # Extract source and save it
                # pylint: disable-next=import-outside-toplevel
                import inspect as _inspect
                code_parts = []
                for attr_name in ("policy", "initiable", "terminal"):
                    fn = getattr(opt, attr_name, None)
                    if fn is None:
                        continue
                    try:
                        src = _inspect.getsource(fn)
                        code_parts.append(f"# {attr_name}\n{src.rstrip()}")
                    except (TypeError, OSError):
                        code_parts.append(
                            f"# {attr_name}: (source not available)")

                if code_parts:
                    full_code = "\n\n".join(code_parts)
                    rel_path = _save_option_to_sandbox(ctx, option_name,
                                                       full_code)
                    if rel_path:
                        lines.append(f"Source code: `{rel_path}`")
                    else:
                        # No sandbox — inline as fallback
                        lines.append("### Source Code")
                        lines.append("```python")
                        lines.append(full_code)
                        lines.append("```")

        return _text_result("\n".join(lines))

    @tool(
        "inspect_trajectories",
        "Inspect trajectory data. Returns state features and/or atoms.",
        {
            "type": "object",
            "properties": {
                "traj_idx": {
                    "type": "integer",
                    "description": "Trajectory index (0-based)"
                },
                "include_states": {
                    "type": "boolean",
                    "description": "Include state feature dicts",
                    "default": True
                },
                "include_atoms": {
                    "type": "boolean",
                    "description": "Include abstract atoms",
                    "default": False
                },
                "max_timesteps": {
                    "type": "integer",
                    "description": "Max timesteps to show",
                    "default": 10
                },
            },
            "required": ["traj_idx"],
        },
    )
    async def inspect_trajectories(args: Dict[str, Any]) -> Dict[str, Any]:
        from predicators import \
            utils  # pylint: disable=import-outside-toplevel

        traj_idx = args["traj_idx"]
        include_states = args.get("include_states", True)
        include_atoms = args.get("include_atoms", False)
        max_timesteps = args.get("max_timesteps", 10)

        all_trajs = ctx.offline_trajectories + ctx.online_trajectories
        if not all_trajs:
            return _error_result("No trajectories available yet.")
        if traj_idx < 0 or traj_idx >= len(all_trajs):
            return _error_result(f"Invalid traj_idx {traj_idx}. "
                                 f"Available: 0-{len(all_trajs)-1}")

        traj = all_trajs[traj_idx]
        provenance = "demo" if traj.is_demo else "interaction"
        task_idx = traj._train_task_idx  # pylint: disable=protected-access
        header = (f"Trajectory {traj_idx}: {len(traj.states)} states, "
                  f"{len(traj.actions)} actions  "
                  f"[provenance={provenance}, task={task_idx}")
        if task_idx is not None and 0 <= task_idx < len(ctx.train_tasks):
            task = ctx.train_tasks[task_idx]
            reached = task.goal_holds(traj.states[-1])
            goal_str = ", ".join(str(g) for g in sorted(task.goal))
            header += f", reached_goal={reached}]"
            lines = [header, f"Goal: {{{goal_str}}}"]
        else:
            header += "]"
            lines = [header]

        for t_step, state in enumerate(traj.states[:max_timesteps]):
            lines.append(f"\n--- Timestep {t_step} ---")
            if include_states:
                lines.append("State:")
                lines.append(state.dict_str(indent=2, num_decimal_points=4))
            if include_atoms:
                atoms = utils.abstract(state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Atoms: {{{atoms_str}}}")
            if t_step < len(traj.actions):
                act = traj.actions[t_step]
                opt = act.get_option()
                lines.append(f"Action: {opt.name}({opt.objects})")

        if len(traj.states) > max_timesteps:
            lines.append(
                f"\n... ({len(traj.states) - max_timesteps} more timesteps)")

        return _text_result("\n".join(lines))

    @tool(
        "inspect_train_tasks",
        "Inspect training tasks (goals, initial atoms, objects, state "
        "details, and optionally an image of the initial scene)",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Task index (0-based). Omit to see summary of all.",
                },
                "include_image": {
                    "type":
                    "boolean",
                    "description":
                    "If true and task_idx is given, render and return an "
                    "image of the initial state (PyBullet envs only).",
                },
            },
        },
    )
    async def inspect_train_tasks(args: Dict[str, Any]) -> Dict[str, Any]:
        from predicators import \
            utils  # pylint: disable=import-outside-toplevel

        task_idx = args.get("task_idx")
        include_image = args.get("include_image", False)

        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            if task.goal_nl:
                goal_line = f"  Goal (natural language): {task.goal_nl}"
            else:
                goal_str = ", ".join(str(g) for g in sorted(task.goal))
                goal_line = f"  Goal: {{{goal_str}}}"
            init_atoms = utils.abstract(task.init, ctx.predicates)
            atoms_str = ", ".join(str(a) for a in sorted(init_atoms))
            objects = sorted(task.init, key=str)
            obj_str = ", ".join(f"{o.name}:{o.type.name}" for o in objects)
            state_str = task.init.pretty_str()
            text = (f"Task {task_idx}:\n"
                    f"{goal_line}\n"
                    f"  Goal achievement: query "
                    f"`is_goal_state(state, {task_idx})` or "
                    f"`train_tasks[{task_idx}].goal_holds(state)`.\n"
                    f"  Initial atoms: {{{atoms_str}}}\n"
                    f"  Objects: [{obj_str}]\n\n"
                    f"Initial state details:\n{state_str}")

            content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

            if include_image:
                img_block = _render_pybullet_image(ctx,
                                                   f"task_{task_idx}_init",
                                                   state=task.init)
                if img_block is not None:
                    content.append(img_block)

            return {"content": content}

        lines = [f"Total tasks: {len(ctx.train_tasks)}"]
        for i, task in enumerate(ctx.train_tasks[:10]):
            if task.goal_nl:
                lines.append(f"  Task {i}: {task.goal_nl}")
            else:
                goal_str = ", ".join(str(g) for g in sorted(task.goal))
                lines.append(f"  Task {i}: goal={{{goal_str}}}")
        if len(ctx.train_tasks) > 10:
            lines.append(f"  ... ({len(ctx.train_tasks) - 10} more tasks)")
        return _text_result("\n".join(lines))

    @tool("inspect_planning_results",
          "Get latest planning performance metrics", {})
    async def inspect_planning_results(
            _args: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.planning_results:
            return _text_result("No planning results available yet.")
        return _text_result(
            json.dumps(ctx.planning_results, indent=2, default=str))

    @tool("inspect_past_proposals",
          "Get summaries of proposals and retractions from all past "
          "iterations", {})
    async def inspect_past_proposals(_args: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.iteration_history:
            return _text_result("No past proposals available yet.")
        lines = []
        for entry in ctx.iteration_history:
            lines.append(json.dumps(entry, indent=2, default=str))
        return _text_result("\n---\n".join(lines))

    return {
        "inspect_types": inspect_types,
        "inspect_predicates": inspect_predicates,
        "inspect_processes": inspect_processes,
        "inspect_options": inspect_options,
        "inspect_trajectories": inspect_trajectories,
        "inspect_train_tasks": inspect_train_tasks,
        "inspect_planning_results": inspect_planning_results,
        "inspect_past_proposals": inspect_past_proposals,
    }


def _build_proposal_tools(ctx: ToolContext, _text_result: Callable,
                          tool: Callable) -> Dict[str, Any]:
    """Proposal tools (agent authors new types/predicates/options/etc.)."""
    _propose_count = [0]  # mutable counter in closure

    def _save_proposal_code(tool_name: str, code: str, names: List[str],
                            description: str) -> None:
        if not ctx.sandbox_dir:
            return
        _propose_count[0] += 1
        subdir = os.path.join(ctx.sandbox_dir, "proposed_code")
        os.makedirs(subdir, exist_ok=True)
        names_slug = "_".join(names)[:80]
        filename = f"{_propose_count[0]:03d}_{tool_name}_{names_slug}.py"
        filepath = os.path.join(subdir, filename)
        header = f'"""{tool_name}: {description}"""\n\n'
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + code)
        logging.info(f"Saved proposal code to {filepath}")

    @tool(
        "propose_types",
        "Propose new types. Code must define `proposed_types` (a list of "
        "Type objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_types"
                },
                "description": {
                    "type": "string",
                    "description": "Why these types are needed"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_types(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_types:
            return _error_result("Type proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "proposed_types")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_types must be a list/set, got {type(result)}")
        for t in result:
            if not isinstance(t, Type):
                return _error_result(
                    f"Each item must be a Type, got {type(t)}: {t}")
        proposed = set(result)
        ctx.iteration_proposals.proposed_types |= proposed
        names = [t.name for t in proposed]
        logging.info(f"Agent proposed types: {names}")
        _save_proposal_code("propose_types", code, names,
                            args.get("description", ""))
        return _text_result(
            f"Successfully proposed {len(proposed)} types: {names}")

    @tool(
        "propose_predicates",
        "Propose new predicates. Code must define `proposed_predicates` "
        "(a list of Predicate objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_predicates"
                },
                "description": {
                    "type": "string",
                    "description": "What these predicates capture"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_predicates(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_predicates:
            return _error_result("Predicate proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "proposed_predicates")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_predicates must be a list/set, got {type(result)}")

        validated = []
        errors = []
        for pred in result:
            if not isinstance(pred, Predicate):
                errors.append(f"Not a Predicate: {type(pred)}: {pred}")
                continue
            if ctx.example_state is not None:
                err = validate_predicate(pred, ctx.types, ctx.example_state)
                if err:
                    errors.append(f"{pred.name}: {err}")
                    continue
            validated.append(pred)

        proposed = set(validated)
        ctx.iteration_proposals.proposed_predicates |= proposed
        names = [p.name for p in proposed]
        logging.info(f"Agent proposed predicates: {names}")
        _save_proposal_code("propose_predicates", code, names,
                            args.get("description", ""))

        msg = f"Successfully proposed {len(proposed)} predicates: {names}"
        if errors:
            msg += f"\n\nValidation errors ({len(errors)}):\n" + \
                   "\n".join(errors)
        return _text_result(msg)

    @tool(
        "propose_task_augmentor",
        "Propose a task augmentation function. Code must define "
        "`augment_task(task) -> Task`.",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type":
                    "string",
                    "description":
                    "Python code defining augment_task(task) -> Task"
                },
                "description": {
                    "type": "string",
                    "description": "What the augmentor does"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_task_augmentor(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_objects:
            return _error_result("Object augmentor proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "augment_task")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not callable(result):
            return _error_result(
                f"augment_task must be callable, got {type(result)}")

        # Test on first train task if available
        if ctx.train_tasks:
            try:
                test_task = ctx.train_tasks[0]
                augmented = result(test_task)
                orig_objs = set(test_task.init)
                new_objs = set(augmented.init) - orig_objs
                obj_names = [str(o) for o in sorted(new_objs, key=str)]
            except Exception:  # pylint: disable=broad-except
                return _error_result(
                    f"augment_task failed on test task:\n"
                    f"{_scrub_host_paths(traceback.format_exc())}")
        else:
            obj_names = ["(no tasks to test on)"]

        ctx.iteration_proposals.augment_task_fn = result
        ctx.iteration_proposals.augment_task_code = code
        logging.info(f"Agent proposed augmentor adding objects: {obj_names}")
        _save_proposal_code("propose_task_augmentor", code, obj_names,
                            args.get("description", ""))
        return _text_result(
            f"Successfully proposed augmentor. Test added objects: {obj_names}"
        )

    @tool(
        "propose_processes",
        "Propose new causal processes. Code must define "
        "`proposed_processes` (a list of CausalProcess objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_processes"
                },
                "description": {
                    "type": "string",
                    "description": "What these processes model"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_processes(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_processes:
            return _error_result("Process proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "proposed_processes")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_processes must be a list/set, got {type(result)}")
        for proc in result:
            if not isinstance(proc, CausalProcess):
                return _error_result(
                    f"Each item must be a CausalProcess, got {type(proc)}")
        proposed = set(result)
        ctx.iteration_proposals.proposed_processes |= proposed
        names = [p.name for p in proposed]
        logging.info(f"Agent proposed processes: {names}")
        _save_proposal_code("propose_processes", code, names,
                            args.get("description", ""))
        return _text_result(
            f"Successfully proposed {len(proposed)} processes: {names}")

    @tool(
        "propose_options",
        "Propose new parameterized options. Code must define "
        "`proposed_options` (a list of ParameterizedOption objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_options"
                },
                "description": {
                    "type": "string",
                    "description": "What these options do"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_options(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_options:
            return _error_result("Option proposals are disabled.")
        if ctx.proposals_disabled:
            return _error_result(
                "Proposals are disabled during test-time solving. "
                "Options can only be proposed during learning.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types,
                                      ctx.predicates,
                                      ctx.options,
                                      extra_context=ctx.skill_factory_context)
        result, error = exec_code_safely(code, exec_ctx, "proposed_options")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_options must be a list/set, got {type(result)}")
        for opt in result:
            if not isinstance(opt, ParameterizedOption):
                return _error_result(
                    f"Each item must be a ParameterizedOption, "
                    f"got {type(opt)}")
        proposed = set(result)
        ctx.iteration_proposals.proposed_options |= proposed
        ctx.options |= proposed
        names = [o.name for o in proposed]
        # Save proposal code to sandbox for each option
        for opt in proposed:
            _save_option_to_sandbox(ctx, opt.name, code)
        logging.info(f"Agent proposed options: {names}")
        _save_proposal_code("propose_options", code, names,
                            args.get("description", ""))
        return _text_result(
            f"Successfully proposed {len(proposed)} options: {names}")

    return {
        "propose_types": propose_types,
        "propose_predicates": propose_predicates,
        "propose_task_augmentor": propose_task_augmentor,
        "propose_processes": propose_processes,
        "propose_options": propose_options,
    }


def _build_retraction_tools(ctx: ToolContext, _text_result: Callable,
                            tool: Callable) -> Dict[str, Any]:
    """Retraction tools (remove agent-proposed abstractions)."""

    @tool(
        "retract_abstractions",
        "Remove previously proposed abstractions that are no longer needed. "
        "Specify names of predicates, processes, options, or helper types to "
        "remove, and/or set clear_task_augmentor to remove the augmentor.",
        {
            "type": "object",
            "properties": {
                "predicate_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of predicates to remove",
                },
                "process_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of processes to remove",
                },
                "option_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of options to remove",
                },
                "type_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of helper types to remove",
                },
                "clear_task_augmentor": {
                    "type": "boolean",
                    "description":
                    "Set to true to remove the object augmentor",
                },
                "reason": {
                    "type": "string",
                    "description": "Why these abstractions are being removed",
                },
            },
            "required": ["reason"],
        },
    )
    async def retract_abstractions(args: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.proposals_disabled:
            return _error_result(
                "Retractions are disabled during test-time solving. "
                "Abstractions can only be retracted during learning.")
        pred_names = set(args.get("predicate_names") or [])
        proc_names = set(args.get("process_names") or [])
        opt_names = set(args.get("option_names") or [])
        type_names = set(args.get("type_names") or [])
        clear_augmentor = bool(args.get("clear_task_augmentor", False))

        if not any(
            [pred_names, proc_names, opt_names, type_names, clear_augmentor]):
            return _text_result("Nothing to retract.")

        lines = [f"Retracting abstractions. Reason: {args['reason']}"]

        if pred_names:
            existing = {p.name for p in ctx.predicates}
            unknown = pred_names - existing
            valid = pred_names & existing
            ctx.iteration_proposals.retract_predicate_names |= valid
            lines.append(f"Predicates to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if proc_names:
            existing = {p.name for p in ctx.processes}
            unknown = proc_names - existing
            valid = proc_names & existing
            ctx.iteration_proposals.retract_process_names |= valid
            lines.append(f"Processes to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if opt_names:
            existing = {o.name for o in ctx.options}
            unknown = opt_names - existing
            valid = opt_names & existing
            ctx.iteration_proposals.retract_option_names |= valid
            ctx.options = {o for o in ctx.options if o.name not in valid}
            lines.append(f"Options to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if type_names:
            existing = {t.name for t in ctx.types}
            unknown = type_names - existing
            valid = type_names & existing
            ctx.iteration_proposals.retract_type_names |= valid
            lines.append(f"Helper types to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if clear_augmentor:
            ctx.iteration_proposals.retract_task_augmentor = True
            lines.append("Object augmentor will be cleared.")

        logging.info(f"Agent retraction request: {args}")
        return _text_result("\n".join(lines))

    return {
        "retract_abstractions": retract_abstractions,
    }


def _build_testing_tools(ctx: ToolContext, _text_result: Callable,
                         tool: Callable) -> Dict[str, Any]:
    """Evaluation tools (run predicates / option plans against tasks)."""

    @tool(
        "evaluate_predicate_on_trajectory",
        "Evaluate a predicate's truth value across timesteps in a trajectory",
        {
            "type": "object",
            "properties": {
                "predicate_name": {
                    "type": "string",
                    "description": "Name of the predicate to test"
                },
                "traj_idx": {
                    "type": "integer",
                    "description": "Trajectory index"
                },
                "object_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Object names to ground the predicate on"
                },
            },
            "required": ["predicate_name", "traj_idx", "object_names"],
        },
    )
    async def evaluate_predicate_on_trajectory(
            args: Dict[str, Any]) -> Dict[str, Any]:
        pred_name = args["predicate_name"]
        traj_idx = args["traj_idx"]
        object_names = args["object_names"]

        # Find the predicate
        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        pred = None
        for p in all_preds:
            if p.name == pred_name:
                pred = p
                break
        if pred is None:
            return _error_result(f"Predicate '{pred_name}' not found.")

        all_trajs = ctx.offline_trajectories + ctx.online_trajectories
        if not all_trajs:
            return _error_result("No trajectories available yet.")
        if traj_idx < 0 or traj_idx >= len(all_trajs):
            return _error_result(f"Invalid traj_idx {traj_idx}. "
                                 f"Available: 0-{len(all_trajs)-1}")

        traj = all_trajs[traj_idx]

        # Find objects by name
        objects = []
        for name in object_names:
            found = None
            for obj in traj.states[0]:
                if obj.name == name:
                    found = obj
                    break
            if found is None:
                avail = [o.name for o in sorted(traj.states[0], key=str)]
                return _error_result(f"Object '{name}' not found in "
                                     f"trajectory {traj_idx}. "
                                     f"Available: {avail}")
            objects.append(found)

        results = []
        for t_step, state in enumerate(traj.states):
            try:
                val = pred.holds(state, objects)
                results.append(f"t={t_step}: {val}")
            except Exception as e:  # pylint: disable=broad-except
                results.append(f"t={t_step}: ERROR ({e})")

        return _text_result(
            f"Predicate {pred_name}({', '.join(object_names)}) "
            f"over trajectory {traj_idx}:\n" + "\n".join(results))

    _gs_eval_doc = (
        "Runs your exact params with NO sampling (a `~` ground-sampler "
        "annotation - `~ [w1, w2]` region or `~ my_sampler` - is accepted "
        "but IGNORED here; only refine_plan_sketch uses it). "
        if CFG.agent_bilevel_ground_samplers else
        "Runs your exact params with NO sampling. ")

    # When the session carries explore_python, the two surfaces divide
    # cleanly: exploration (modified states, partial plans, sweeps)
    # belongs in the probe, and this tool is the SUBMISSION path.
    _probe_split_doc = (
        " This tool always runs from the task's TRUE initial state and is "
        "the ONLY path that captures an answer: do your exploration "
        "(modified states, partial plans, parameter sweeps) in "
        "explore_python, then validate and SUBMIT the final plan here."
        if CFG.agent_planner_use_explore_python else "")

    @tool(
        "evaluate_option_plan",
        "Execute a fully-specified plan on a task via the option model and "
        "report the result at each step. `plan` is text — one option per "
        "line, same grammar as refine_plan_sketch: "
        "`Option(obj1:type1, obj2:type2)[param1, param2] -> {Atom(obj:type), "
        "...}` (typed object refs; EXACT continuous params in `[]`, `[]` for "
        "none; optional `-> {atoms}` subgoals, prefix NOT to require false). "
        + _gs_eval_doc + "Use include_states/"
        "include_atoms to control output. If the plan reaches the goal on the "
        "CURRENT task (omit task_idx), it is captured as your answer, and the "
        "per-step subgoals make it execute closed-loop (monitored, with "
        "replan-on-divergence). Capture is gated: a goal-reaching plan is "
        "re-run several times (simulation varies across runs) and a FLAKY "
        "plan is reported instead of captured - add margin and resubmit. "
        "When the task has an evaluator, a goal-reaching plan the evaluator "
        "still scores as a non-solve (no success credit in its reward) is "
        "NOT captured (the real env applies the same scoring, so it could "
        "never count as a solve)." + _probe_split_doc,
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Plan text, one option per line: "
                    "`Option(obj1:type1, obj2:type2)[p1, p2] -> "
                    "{Atom(obj:type), ...}` (exact params in `[]`; `[]` for "
                    "none; optional `-> {atoms}` subgoals, NOT-prefix to "
                    "require false).",
                },
                "include_states": {
                    "type":
                    "boolean",
                    "description":
                    "Include the full low-level state feature dict after each "
                    "step",
                    "default":
                    True
                },
                "include_atoms": {
                    "type": "boolean",
                    "description":
                    "Include atoms added/deleted after each step",
                    "default": True
                },
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index to test on. Omit to use "
                    "the current solve-time task."
                },
            },
            "required": ["plan"],
        },
    )
    async def evaluate_option_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np  # pylint: disable=reimported,redefined-outer-name,import-outside-toplevel

        from predicators import \
            utils  # pylint: disable=import-outside-toplevel

        ctx.test_call_id += 1
        # Snapshot for the [budget] footer's per-call delta; this handler
        # increments the counter itself (initial rollout + validation
        # repeats), and without the snapshot the footer reports the
        # attempt's cumulative total as "+N this call".
        rollouts_before = ctx.attempt_rollout_count

        if ctx.option_model is None:
            return _error_result("No option model available in ToolContext.")

        # Sync the option model's option map with all current options
        # (GT + proposed) so it stays in sync after propose/retract.
        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        opt_map = {o.name: o for o in all_options}
        model = ctx.option_model
        model._name_to_parameterized_option = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
            opt_map)

        task_idx = args.get("task_idx")
        plan_text = (args.get("plan") or "").strip()
        include_states = args.get("include_states", False)
        include_atoms = args.get("include_atoms", True)

        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_idx = "current"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        lines = [f"Testing option plan on task {task_idx}:"]
        saved_image_paths: List[str] = []

        from predicators.agent_sdk import \
            bilevel_sketch  # pylint: disable=import-outside-toplevel
        all_predicates = (ctx.predicates
                          | ctx.iteration_proposals.proposed_predicates)

        if not plan_text:
            return _error_result("`plan` is required (option plan text).")
        # Parse the text plan into a sketch (options + objects + exact params +
        # subgoals) using the SAME grammar/parser as refine_plan_sketch.
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in task.init)
        try:
            # strict: the `plan` argument is pure plan text, so a line that
            # fails to parse is an error the agent must see - silently
            # dropping it (the freeform default) executes a different plan
            # than the agent asked for.
            gs_fns, gs_err = _load_ground_sampler_fns(ctx)
            if gs_err is not None:
                return _error_result(gs_err)
            parse_notices: List[str] = []
            sketch_steps = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=True,
                strict=True,
                parse_ground_samplers=CFG.agent_bilevel_ground_samplers,
                ground_sampler_fns=gs_fns or None,
                notices=parse_notices)
        except Exception as e:  # pylint: disable=broad-except
            return _error_result(f"Could not parse plan: {e}")
        lines.extend(f"NOTE: {n}" for n in parse_notices)
        if not sketch_steps:
            return _error_result(
                "Parsed empty plan. Each line must be "
                "`Option(obj:type, ...)[params] -> {subgoals}` with a known "
                "option, typed object refs, and exact params in `[]`.")
        # Ground each step with its parsed exact params.
        grounded_plan: List[Any] = []
        for step_idx, st in enumerate(sketch_steps):
            params = (st.initial_params if st.initial_params is not None else
                      np.array([], dtype=np.float32))
            try:
                grounded_plan.append(
                    st.option.ground(list(st.objects),
                                     np.asarray(params, dtype=np.float32)))
            except Exception as e:  # pylint: disable=broad-except
                return _error_result(f"Failed to ground step {step_idx} "
                                     f"({st.option.name}): {e}")

        # Per-low-level-step states + option labels for the task-evaluator
        # verdict below. The cascade certificate needs per-step states
        # (topple-onset analysis); option-boundary states give garbage
        # verdicts, so prefer the option model's last_trajectory and flag
        # the verdict as coarse when it is unavailable.
        eval_states: List[State] = [task.init]
        eval_labels: List[Any] = []
        eval_coarse = False

        def _collect_eval_steps(outcome: Any) -> None:
            nonlocal eval_coarse
            if outcome.post_state is None:
                return
            opt = outcome.option
            label = (opt.name, tuple(o.name for o in opt.objects),
                     tuple(float(p) for p in opt.params))
            step_traj = getattr(ctx.option_model, "last_trajectory", None)
            if step_traj is not None and len(step_traj.states) >= 2:
                eval_states.extend(step_traj.states[1:])
                eval_labels.extend([label] * len(step_traj.actions))
            else:
                eval_states.append(outcome.post_state)
                eval_labels.append(label)
                eval_coarse = True

        # Per-step report callback, driven by the shared forward executor.
        def _report_step(i: int, outcome: Any) -> None:
            _collect_eval_steps(outcome)
            opt = outcome.option
            sig = f"{opt.name}({[o.name for o in opt.objects]})"
            if not outcome.initiable:
                atoms = utils.abstract(outcome.pre_state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Step {i}: {sig} - NOT INITIABLE\n"
                             f"  Current atoms: {{{atoms_str}}}\n"
                             f"  Object poses at failure:\n"
                             f"{_format_object_poses(outcome.pre_state)}")
                return
            step_line = f"Step {i}: {sig} ({outcome.num_actions} actions)"
            if outcome.failure_reason is not None:
                step_line += (f"\n  FAILURE REASON: {outcome.failure_reason}"
                              "\n  Object poses at failure:\n"
                              f"{_format_object_poses(outcome.pre_state)}")
            post = outcome.post_state
            if post is not None and include_atoms:
                before = utils.abstract(outcome.pre_state, ctx.predicates)
                after = utils.abstract(post, ctx.predicates)
                added_s = ", ".join(str(a) for a in sorted(after - before))
                del_s = ", ".join(str(a) for a in sorted(before - after))
                step_line += (f"\n  Added:   {{{added_s}}}"
                              f"\n  Deleted: {{{del_s}}}")
            if post is not None and include_states:
                step_line += ("\n  State:\n" +
                              post.dict_str(indent=4, num_decimal_points=4))
            lines.append(step_line)
            img_block = _render_scene_image(ctx, f"step_{i}_{opt.name}")
            if img_block and img_block.get("saved_path"):
                saved_image_paths.append(img_block["saved_path"])

        # Execute exactly like the real closed-loop executor: abort at the
        # first failing option (0-action collision / not-initiable / env
        # failure) instead of pressing on. Otherwise forward simulation can
        # continue past a collision and report a goal that the real rollout —
        # which ends the episode at that failed option — never reaches.
        ctx.attempt_rollout_count += 1
        result = bilevel_sketch.execute_plan_forward(task,
                                                     grounded_plan,
                                                     ctx.option_model,
                                                     predicates=all_predicates,
                                                     sketch=sketch_steps,
                                                     on_step=_report_step,
                                                     stop_on_failure=True)

        final_atoms = utils.abstract(result.final_state, ctx.predicates)
        # Use the env's goal-check (its own classifiers); robust to invented
        # predicates that don't reuse env names.
        goal_reached = result.goal_reached
        # One more real-executor constraint the option model doesn't enforce:
        # the episode is capped at `CFG.horizon` low-level steps. A plan whose
        # goal is reached only after more steps than the horizon allows will
        # time out in real rollout, so don't count it as achieved/captured.
        horizon = CFG.horizon
        within_horizon = (result.actions_to_goal is not None
                          and result.actions_to_goal <= horizon)
        goal_achieved = (goal_reached and result.clean_to_goal
                         and within_horizon)
        # Task-evaluator verdict on this belief-sim rollout, computed BEFORE
        # capture: the real evaluator applies the same certificate, so a
        # goal-reaching but illegitimate plan can never count as a solve and
        # must not be captured as the answer (run_20260712_173955 tasks 1-2:
        # flagged-illegitimate captures stood all session and were executed
        # only to be rejected). Failure-tolerant: verdict stays None when the
        # task has no evaluator or nothing executed. A coarse verdict
        # (option-boundary states only) can falsely reject a legitimate
        # cascade, so it never blocks capture.
        evaluator = _resolve_task_evaluator(ctx, task_idx)
        verdict: Optional[Dict[str, Any]] = None
        if evaluator is not None and len(eval_states) > 1:
            try:
                verdict = evaluate_states_with(evaluator,
                                               eval_states,
                                               eval_labels,
                                               sim_env=getattr(
                                                   ctx.option_model, "sim_env",
                                                   None))
            except Exception as e:  # pylint: disable=broad-except
                logging.debug("Task-evaluator verdict failed: %s", e)
        evaluator_rejected = (verdict is not None and not verdict["legitimate"]
                              and not eval_coarse)
        # An evaluator rejection only disqualifies a capture when the goal
        # atoms actually hold via an illegitimate route - a reward hack (e.g.
        # the agent knocked the target over directly). An honest shortfall,
        # where the rollout simply fails to reach the goal, is ALSO
        # legitimate=False (there is no genuine cascade to certify), but that
        # is exactly what a best-effort submission is meant to capture, so it
        # must not be conflated with a reward hack.
        reward_hack = (evaluator_rejected and verdict is not None
                       and verdict["terminated"])

        # Multi-rollout validation of a capture candidate. The shared sim
        # env is nondeterministic across repeats (motion-planner sampling,
        # physics-solver state), which is the same variability the real
        # rollout will sample - a plan that only sometimes succeeds here is
        # a margin-free plan that will likely fail on the real env
        # (run_20260712_192457 task 1: a sim-validated 2-hop relay died on a
        # ~9mm placement drift). So a goal-reaching plan is captured only
        # after every one of CFG.agent_plan_validation_rollouts total
        # rollouts succeeds; a flaky repeat is reported to the agent, who
        # still has the session to add margin and resubmit.
        def _validation_rollout() -> Tuple[bool, str]:
            """One extra rollout of the exact plan; (ok, failure detail)."""
            v_states: List[State] = [task.init]
            v_labels: List[Any] = []
            v_coarse = False

            def _collect(_i: int, outcome: Any) -> None:
                nonlocal v_coarse
                if outcome.post_state is None:
                    return
                opt = outcome.option
                label = (opt.name, tuple(o.name for o in opt.objects),
                         tuple(float(p) for p in opt.params))
                step_traj = getattr(ctx.option_model, "last_trajectory", None)
                if step_traj is not None and len(step_traj.states) >= 2:
                    v_states.extend(step_traj.states[1:])
                    v_labels.extend([label] * len(step_traj.actions))
                else:
                    v_states.append(outcome.post_state)
                    v_labels.append(label)
                    v_coarse = True

            r = bilevel_sketch.execute_plan_forward(task,
                                                    grounded_plan,
                                                    model,
                                                    predicates=all_predicates,
                                                    sketch=sketch_steps,
                                                    on_step=_collect,
                                                    stop_on_failure=True)
            if r.first_failure_idx is not None:
                fr = r.steps[r.first_failure_idx].failure_reason
                opt = r.steps[r.first_failure_idx].option
                return False, (f"step {r.first_failure_idx} "
                               f"({opt.name}) failed: {fr}")
            if not r.goal_reached:
                missing = task.goal - utils.abstract(r.final_state,
                                                     ctx.predicates)
                missing_str = ", ".join(str(a) for a in sorted(missing))
                detail = f" (missing: {{{missing_str}}})" if missing else ""
                return False, f"goal not reached{detail}"
            if not (r.actions_to_goal is not None
                    and r.actions_to_goal <= horizon):
                return False, (f"goal reached only after "
                               f"{r.actions_to_goal} low-level steps, past "
                               f"the episode horizon ({horizon})")
            # Same legitimacy rule as the first rollout: a non-coarse
            # illegitimate verdict fails the validation.
            if evaluator is not None and len(v_states) > 1 and not v_coarse:
                try:
                    v = evaluate_states_with(evaluator,
                                             v_states,
                                             v_labels,
                                             sim_env=getattr(
                                                 ctx.option_model, "sim_env",
                                                 None))
                    if not v["legitimate"]:
                        return False, (
                            "this rollout reached the goal atoms but the "
                            "task evaluator scored it as a non-solve "
                            f"(solved=False, reward={v['reward']:.2f})")
                except Exception as e:  # pylint: disable=broad-except
                    logging.debug("Validation-rollout verdict failed: %s", e)
            return True, ""

        flaky_detail: Optional[str] = None
        validation_note = ""
        n_rollouts = max(1, CFG.agent_plan_validation_rollouts)
        # Escalated gate once this task has produced a FLAKY rejection: the
        # agent is provably tuning in a marginal region, where a lucky
        # streak passes the base gate and dies on the single real episode
        # (run_20260717_182321: a 20/20-swept relay placement validated 3/3,
        # then missed the target for real).
        capture_task_key = _capture_task_key(ctx)
        if capture_task_key in ctx.flaky_capture_task_keys:
            n_rollouts = max(n_rollouts,
                             CFG.agent_plan_validation_rollouts_after_flaky)
        # Fresh env per validation rollout when the approach provides one:
        # repeats on the shared env are correlated (its reset cannot
        # reconstruct state exactly), so only fresh envs sample the same
        # distribution the real episode will.
        fresh_scope = (ctx.validation_env_scope
                       if CFG.agent_plan_validation_fresh_env else None)
        rollout_outcomes: List[str] = []
        if (ctx.capture_goal_reaching_plans and task_idx == "current"
                and goal_achieved and not evaluator_rejected and grounded_plan
                and n_rollouts > 1):
            # Run ALL validation rollouts even after a failure: the
            # per-rollout outcome list distinguishes failure modes (a
            # physics-tail fizzle vs. an IK stall vs. a certificate
            # rejection) and yields a reliability estimate - a bare
            # "rollout k FAILED" left agents guessing which
            # (run_20260717_182040 seed0 turn 214).
            for repeat_idx in range(2, n_rollouts + 1):
                ctx.attempt_rollout_count += 1
                with (fresh_scope() if fresh_scope is not None else
                      contextlib.nullcontext()):
                    ok, why = _validation_rollout()
                if ok:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx}: goal reached")
                else:
                    rollout_outcomes.append(
                        f"rollout {repeat_idx}: FAILED - {why}")
                    if flaky_detail is None:
                        flaky_detail = (f"rollout {repeat_idx}/{n_rollouts} "
                                        f"FAILED: {why}")
            if flaky_detail is None:
                fresh_note = (", each on a freshly constructed simulator "
                              "instance" if fresh_scope is not None else "")
                validation_note = (
                    f" Validated {n_rollouts}/{n_rollouts} rollouts (the "
                    "simulator's motion planning and physics stepping vary "
                    "across runs; repeats sample that execution "
                    f"variability{fresh_note}).")

        # Capture a goal-reaching plan on the current task with a sketch that
        # keeps only the subgoals that actually held (so the closed-loop
        # monitor won't flag a spurious divergence on a wrong annotation).
        # With capture_best_effort_plan (final-submission nudge after turn-cap
        # exhaustion) capture the submission unconditionally: honest
        # shortfall, evaluator-rejected rollout, and flaky repeat alike. The
        # budget is spent, so executing the agent's best plan for its honest
        # reward beats forfeiting the task (run_20260714_145053 task 4: a
        # goal-reaching but certificate-rejected final submission was refused
        # and the task forfeited, scoring n/a instead of its honest reward).
        # Only a validated solve is marked as one; everything else is a
        # best-effort capture that executes but cannot count as a solve - the
        # certificate still protects the score. A best-effort capture never
        # displaces a validated-solve capture.
        def _stash_uncaptured_submission() -> None:
            """Remember the best refused submission of this attempt.

            The journal auto-entry records it at attempt end, so the
            plan (and its honest evaluator reward) survives the fresh-
            context restart and the final best-effort nudge can resubmit
            it instead of the attempt's work vanishing with its context.
            """
            reward = float(verdict["reward"]) if verdict is not None else None
            prev = ctx.best_uncaptured_reward
            if ctx.best_uncaptured_plan_lines is not None and (
                    reward is None or (prev is not None and reward <= prev)):
                return
            ctx.best_uncaptured_reward = reward
            ctx.best_uncaptured_plan_lines = list(
                bilevel_sketch.format_plan_lines(grounded_plan))

        best_effort_capture = (ctx.capture_best_effort_plan
                               and not ctx.solved_plan_reached_goal)
        validated_solve = (goal_achieved and not reward_hack
                           and flaky_detail is None)
        captured = False
        if (ctx.capture_goal_reaching_plans and task_idx == "current"
                and (validated_solve or best_effort_capture)
                and grounded_plan):
            captured = True
            captured_sketch = []
            for i, st in enumerate(sketch_steps):
                post = (result.steps[i].post_state
                        if i < len(result.steps) else None)
                if post is not None:
                    after = utils.abstract(post, all_predicates)
                    pos_held = {
                        a
                        for a in (st.subgoal_atoms or set()) if a in after
                    }
                    neg_held = {
                        a
                        for a in (st.subgoal_neg_atoms or set())
                        if a not in after
                    }
                else:
                    pos_held, neg_held = set(), set()
                captured_sketch.append(
                    bilevel_sketch.SketchStep(option=st.option,
                                              objects=st.objects,
                                              subgoal_atoms=pos_held or None,
                                              subgoal_neg_atoms=neg_held
                                              or None))
            ctx.solved_plan = grounded_plan
            ctx.solved_sketch = captured_sketch
            ctx.solved_plan_reached_goal = validated_solve
            ctx.solved_plan_eval_reward = (float(verdict["reward"])
                                           if verdict is not None else None)
            n_annot = sum(1 for s in captured_sketch
                          if s.subgoal_atoms or s.subgoal_neg_atoms)
            if validated_solve:
                best_effort_note = ""
            elif not goal_achieved:
                best_effort_note = (" (best-effort: goal NOT reached, "
                                    "accepted because the attempt budget is "
                                    "exhausted; it executes for its honest "
                                    "reward but will not count as a solve)")
            elif reward_hack:
                best_effort_note = (" (best-effort: the rollout reaches the "
                                    "goal atoms but the task evaluator "
                                    "scores it as a non-solve, and the real "
                                    "env applies the same scoring; accepted "
                                    "because the attempt budget is exhausted "
                                    "- it executes for its honest reward but "
                                    "will not count as a solve)")
            else:
                best_effort_note = (f" (best-effort: {flaky_detail}; "
                                    "accepted because the attempt budget is "
                                    "exhausted - it executes for its honest "
                                    "reward but may not reproduce its "
                                    "solve)")
            lines.append(f"Captured as the current answer{best_effort_note}: "
                         f"{len(grounded_plan)} steps, "
                         f"{n_annot} with subgoal annotations for closed-loop "
                         f"monitoring.{validation_note}")
        elif (ctx.capture_goal_reaching_plans and task_idx == "current"
              and goal_achieved and not evaluator_rejected
              and flaky_detail is not None):
            # Loudly refuse a flaky capture: the agent still has this
            # session to add margin and resubmit, which beats discovering
            # the flakiness as a failed real episode. Record the task so
            # later submissions face the escalated gate - flakiness here is
            # evidence the whole parameter region is marginal, not just
            # this point.
            ctx.flaky_capture_task_keys.add(capture_task_key)
            _stash_uncaptured_submission()
            escalated_n = max(max(1, CFG.agent_plan_validation_rollouts),
                              CFG.agent_plan_validation_rollouts_after_flaky)
            n_ok = 1 + sum(1 for o in rollout_outcomes if "FAILED" not in o)
            per_rollout = "\n".join(f"  {o}" for o in rollout_outcomes)
            lines.append(
                f"FLAKY (plan NOT captured): the plan reached the goal on "
                f"rollout 1 but {flaky_detail}. Per-rollout outcomes "
                f"(estimated reliability {n_ok}/{n_rollouts}):\n"
                f"  rollout 1: goal reached\n{per_rollout}\n"
                "The simulator's motion "
                "planning and physics stepping vary across runs, and the "
                "real environment samples the same variability - a plan "
                "that only sometimes succeeds in simulation will likely "
                "fail for real. Add margin (e.g. tighter spacing, aim "
                "impacts closer to the middle of the fall path) and "
                "resubmit. Because this task has now produced a flaky "
                f"submission, captures require {escalated_n}/{escalated_n} "
                "successful rollouts: fix the margin rather than "
                "resubmitting near-identical parameters.")
        elif (ctx.capture_goal_reaching_plans and task_idx == "current"
              and reward_hack):
            # Loudly refuse a reward hack: the rollout reaches the goal atoms
            # but the evaluator's certificate rejects the route (e.g. the
            # target was knocked over directly), and the real evaluator
            # applies the same certificate, so it can never count as a solve.
            # (Under a best-effort final submission the same plan is instead
            # captured above, flagged as a non-solve, to execute for its
            # honest reward.)
            assert verdict is not None
            _stash_uncaptured_submission()
            lines.append(
                "NOT CAPTURED: the rollout reaches the goal atoms but the "
                "task evaluator scores it as a non-solve (solved=False, "
                f"reward={verdict['reward']:.2f}). The real env applies the "
                "same scoring, so executing this plan cannot count as a "
                "solve. Find a plan whose rollout the evaluator scores "
                "solved=True.")
        elif (ctx.capture_goal_reaching_plans and task_idx != "current"
              and goal_achieved):
            # Loudly flag a success that cannot count: agents have burned
            # whole sessions validating on a train task, believing they
            # were done (run_20260707_112310 test task 0, session 3).
            lines.append(
                f"NOTE: this ran on train task {task_idx}, NOT the current "
                "task, so it is NOT captured as your answer. To submit, "
                "re-run the plan on the current task (omit task_idx).")
        if result.first_failure_idx is not None:
            fr = result.steps[result.first_failure_idx].failure_reason
            lines.append(
                f"\nPlan FAILED at step {result.first_failure_idx}: {fr}")
        final_atoms_str = ", ".join(str(a) for a in sorted(final_atoms))
        lines.append(f"\nFinal atoms: {{{final_atoms_str}}}")
        if task.goal_nl:
            lines.append(f"Goal (natural language): {task.goal_nl}")
        else:
            goal_str = ", ".join(str(g) for g in sorted(task.goal))
            lines.append(f"Goal: {{{goal_str}}}")
        lines.append(f"Goal achieved: {goal_achieved}")
        # Task-evaluator verdict line (verdict computed above, before the
        # capture decision it gates). On a FLAKY rejection this verdict is
        # rollout 1's only - printing it unlabeled next to a failing
        # rollout's non-solve read as two contradictory verdicts in one
        # message (run_20260717_182040 seed1 turn 96).
        if verdict is not None:
            vline = _format_evaluator_verdict(verdict, coarse=eval_coarse)
            if flaky_detail is not None and not captured:
                vline += (" [rollout 1 only - NOT the operative outcome; "
                          "this submission was rejected as FLAKY above]")
            lines.append(vline)
        # Goal atoms hold but the plan needs more low-level steps than the
        # episode horizon allows: say so and that it was NOT captured, so the
        # agent shortens the plan instead of stopping on a false positive.
        # (A best-effort capture still happens above; then only warn.)
        if goal_reached and not within_horizon and not captured:
            lines.append(
                f"NOT EXECUTABLE (plan was NOT captured): reaching the goal "
                f"takes {result.actions_to_goal} low-level steps but the "
                f"episode horizon is {horizon}. The real executor will run "
                f"out of steps — shorten the plan (fewer or quicker steps) "
                f"before resubmitting.")
        elif goal_reached and not within_horizon:
            lines.append(
                f"WARNING: reaching the goal takes {result.actions_to_goal} "
                f"low-level steps but the episode horizon is {horizon}, so "
                f"the real executor will run out of steps before the goal.")
        # Print the missing goal atoms even when the goal is stated in
        # natural language: "Goal achieved: False" with no per-atom
        # diagnosis left agents unable to tell a near-miss from a
        # non-starter, and validation-rollout failures already name the
        # missing atoms - this just makes rollout 1 report the same way.
        if not goal_reached:
            missing = task.goal - final_atoms
            missing_str = ", ".join(str(a) for a in sorted(missing))
            lines.append(f"Missing goal atoms: {{{missing_str}}}")

        # Append image save paths to text output
        if saved_image_paths:
            lines.append("\nSaved images:")
            for p in saved_image_paths:
                lines.append(f"  {p}")

        # Build result with text only (images are saved to disk)
        return _text_result("\n".join(lines) +
                            _budget_footer(ctx, rollouts_before))

    return {
        "evaluate_predicate_on_trajectory": evaluate_predicate_on_trajectory,
        "evaluate_option_plan": evaluate_option_plan,
    }


def _build_planning_tools(ctx: ToolContext, _text_result: Callable,
                          tool: Callable) -> Dict[str, Any]:
    """Planning tools (generate bilevel / abstract plans)."""

    @tool(
        "generate_bilevel_plan",
        "Generate a concrete option plan using the bilevel planner. Returns "
        "grounded options with sampled continuous parameters, simulated "
        "step-by-step via the option model.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available)."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
        },
    )
    async def generate_bilevel_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=reimported,redefined-outer-name,import-outside-toplevel

        from predicators import utils
        from predicators.planning_with_processes import \
            run_task_plan_with_processes_once

        task_idx = args.get("task_idx")
        timeout = args.get("timeout", 30)

        # Resolve task
        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            task_label = f"train task {task_idx}"
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_label = "current task"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        all_procs = ctx.processes | ctx.iteration_proposals.proposed_processes
        all_types = ctx.types | ctx.iteration_proposals.proposed_types

        # Get abstract plan
        try:
            plan, _atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                all_procs,
                all_preds,
                all_types,
                timeout,
                seed=CFG.seed,
                _task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
        except Exception as e:  # pylint: disable=broad-except
            return _text_result(f"Planning failed for {task_label}.\n"
                                f"Reason: {type(e).__name__}: {e}")

        if not plan:
            return _text_result(
                f"Planner returned empty plan for {task_label}.")

        # Sample options and simulate
        rng = np.random.default_rng(CFG.seed)
        state = task.init
        lines = [
            f"Bilevel plan for {task_label} "
            f"({len(plan)} steps, "
            f"{metrics.get('num_nodes_expanded', '?')} nodes expanded):"
        ]

        option_plan_lines = []
        for step_idx, ground_proc in enumerate(plan):
            try:
                option = ground_proc.sample_option(state, task.goal, rng)
            except Exception as e:  # pylint: disable=broad-except
                lines.append(
                    f"Step {step_idx}: {ground_proc.name}"
                    f"({', '.join(str(o) for o in ground_proc.objects)}) "
                    f"- SAMPLE FAILED: {e}")
                break

            # Format option
            obj_strs = ", ".join(f"{o.name}:{o.type.name}"
                                 for o in option.objects)
            params_str = ", ".join(f"{p:.4f}" for p in option.params)
            option_line = f"{option.name}({obj_strs})[{params_str}]"
            option_plan_lines.append(option_line)

            # Simulate
            if ctx.option_model is not None:
                try:
                    next_state, num_actions = \
                        ctx.option_model.get_next_state_and_num_actions(
                            state, option)
                    atoms_before = utils.abstract(state, all_preds)
                    atoms_after = utils.abstract(next_state, all_preds)
                    added = atoms_after - atoms_before
                    deleted = atoms_before - atoms_after
                    lines.append(
                        f"Step {step_idx}: {option_line} "
                        f"({num_actions} actions)"
                        f"\n  Added:   "
                        f"{{{', '.join(str(a) for a in sorted(added))}}}"
                        f"\n  Deleted: "
                        f"{{{', '.join(str(a) for a in sorted(deleted))}}}")
                    state = next_state
                except Exception as e:  # pylint: disable=broad-except
                    lines.append(f"Step {step_idx}: {option_line} "
                                 f"- SIMULATION ERROR: {e}")
                    break
            else:
                lines.append(f"Step {step_idx}: {option_line}")

        # Check goal via env-side classifiers so the result is robust
        # to invented predicates that don't reuse env names.
        if ctx.option_model is not None:
            goal_achieved = task.goal_holds(state)
            lines.append(f"\nGoal achieved: {goal_achieved}")

        lines.append("\n## Option Plan (copy-paste format):")
        lines.extend(option_plan_lines)

        return _text_result("\n".join(lines))

    @tool(
        "generate_abstract_plan",
        "Generate an abstract plan skeleton without continuous parameters. "
        "Returns option names and objects with parameter space info so you "
        "can fill in continuous parameters yourself.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available)."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
        },
    )
    async def generate_abstract_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable=import-outside-toplevel
        from predicators.planning_with_processes import \
            run_task_plan_with_processes_once

        task_idx = args.get("task_idx")
        timeout = args.get("timeout", 30)

        # Resolve task
        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            task_label = f"train task {task_idx}"
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_label = "current task"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        all_procs = ctx.processes | ctx.iteration_proposals.proposed_processes
        all_types = ctx.types | ctx.iteration_proposals.proposed_types

        try:
            plan, _atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                all_procs,
                all_preds,
                all_types,
                timeout,
                seed=CFG.seed,
                _task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
        except Exception as e:  # pylint: disable=broad-except
            return _text_result(f"Planning failed for {task_label}.\n"
                                f"Reason: {type(e).__name__}: {e}")

        if not plan:
            return _text_result(
                f"Planner returned empty plan for {task_label}.")

        lines = [
            f"Abstract plan for {task_label} "
            f"({len(plan)} steps, "
            f"{metrics.get('num_nodes_expanded', '?')} nodes expanded):",
            "",
        ]

        for step_idx, ground_proc in enumerate(plan):
            obj_strs = ", ".join(f"{o.name}:{o.type.name}"
                                 for o in ground_proc.option_objs)
            option = ground_proc.option
            params_dim = option.params_space.shape[0]
            if params_dim > 0:
                low = option.params_space.low.tolist()
                high = option.params_space.high.tolist()
                param_info = (f"  params_dim={params_dim}, "
                              f"low={low}, high={high}")
            else:
                param_info = "  (no continuous params)"
            lines.append(
                f"Step {step_idx}: {option.name}({obj_strs})\n{param_info}")

        # Include conditions for context
        lines.append("\n## Process conditions:")
        for step_idx, ground_proc in enumerate(plan):
            conds = ", ".join(
                str(a) for a in sorted(ground_proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(ground_proc.add_effects))
            dels = ", ".join(
                str(a) for a in sorted(ground_proc.delete_effects))
            lines.append(f"Step {step_idx} ({ground_proc.name}):"
                         f"\n  Conditions: {{{conds}}}"
                         f"\n  Add effects: {{{adds}}}"
                         f"\n  Delete effects: {{{dels}}}")

        return _text_result("\n".join(lines))

    _gs_refine_doc = (
        "Confine a step's sampling with a GROUND SAMPLER after its "
        "`[params]`: either a region `~ [w1, w2]` (per-parameter "
        "half-widths; the exact center is tried first, then ALL further "
        "samples for the step are drawn uniformly from "
        "`[center - w, center + w]` clipped to the option's range - a zero "
        "width pins every draw to the center), or `~ my_sampler` naming an "
        "entry of `GROUND_SAMPLERS` in the sandbox file "
        "`ground_samplers.py`, which you Write/Edit and which is reloaded "
        "fresh on every call (each entry is "
        "`fn(state, subgoal_atoms, rng, objects) -> params`, so it can "
        "shape any state-dependent distribution). A ground sampler "
        "overrides any learned per-skill sampler for that step. "
        if CFG.agent_bilevel_ground_samplers else "")
    _gs_refine_plan_doc = (
        ", optionally followed by a ground sampler: `~ [w1, w2]` "
        "half-widths around those params, or `~ my_sampler` naming a "
        "GROUND_SAMPLERS entry in ground_samplers.py"
        if CFG.agent_bilevel_ground_samplers else "")

    @tool(
        "refine_plan_sketch",
        "FIND continuous parameters for a plan SKETCH: run a backtracking "
        "search over the option model, then — on success — forward-validate "
        "the refined plan. Unlike evaluate_option_plan (which runs your EXACT "
        "params with no search), this takes a sketch and lets the search find "
        "params. You may seed it by appending `[p1, p2]` per step (use `[]` "
        "for none); the search tries them first, then samples. " +
        _gs_refine_doc + "`plan` is one "
        "option call per line with typed object references (`obj:type`) and "
        "every argument supplied; add `-> {Atom(obj:type, ...)}` subgoal "
        "annotations (effectively required after open-ended skills like Place, "
        "and for Wait to say when it should end — prefix an atom with NOT to "
        "require it become false). When the task has an evaluator, success is "
        "also gated on its scoring: a parameterization that reaches the goal "
        "atoms but scores as a non-solve (no success credit in its reward) is "
        "discarded and the search resamples. On SUCCESS it reports the exact "
        "PARAMETERS it found per step — submit those via evaluate_option_plan, "
        "which is the delivery path; refine_plan_sketch itself does NOT "
        "submit. Also reports the verdict (SUCCESS / TIMEOUT / "
        "SAMPLE_EXHAUSTED with the stuck step / FORWARD_VALIDATION_FAILED / "
        "SCORED_NON_SOLVE) and time used. Requires a simulator (option "
        "model). Slower than evaluate_option_plan — use it to find params "
        "for hard steps, not to submit.",
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Option-skeleton plan text, one option call per "
                    "line, typed `obj:type` references, every argument "
                    "supplied; optional `-> {Atom(...)}` subgoal per step, "
                    "and `[p1, p2]` proposed continuous params per step "
                    "(`[]` for none) when param-proposing is enabled" +
                    _gs_refine_plan_doc + ".",
                },
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available).",
                },
                "timeout": {
                    "type":
                    "number",
                    "description":
                    "Refinement timeout in seconds. Omit for an auto "
                    "value that scales with sketch length; the value "
                    "used is reported back.",
                },
            },
            "required": ["plan"],
        },
    )
    async def refine_plan_sketch(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable=import-outside-toplevel,reimported,redefined-outer-name
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch

        if ctx.option_model is None:
            return _error_result(
                "refine_plan_sketch requires a simulator (no option model "
                "in ToolContext).")

        # Resolve the task (mirrors evaluate_option_plan).
        task_idx = args.get("task_idx")
        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_idx = "current"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        all_predicates = (ctx.predicates
                          | ctx.iteration_proposals.proposed_predicates)
        # Keep the option model's name map in sync with proposed options so
        # refinement can ground them (matches evaluate_option_plan).
        model = ctx.option_model
        model._name_to_parameterized_option = (  # type: ignore[attr-defined]  # pylint: disable=protected-access
            {o.name: o
             for o in all_options})
        # Union declared types with those reachable from options/predicates/
        # objects so typed `obj:type` references in the sketch resolve.
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in task.init)

        plan_text = (args.get("plan") or "").strip()
        if not plan_text:
            return _error_result("`plan` is required (option-skeleton text).")
        try:
            # strict: the `plan` argument is pure sketch text (see
            # evaluate_option_plan) - unparseable lines must error, not be
            # silently dropped.
            # Named `~ my_sampler` references resolve against the agent's
            # ground_samplers.py, reloaded fresh so edits between calls
            # take effect; a broken file is surfaced instead of silently
            # falling back to uniform draws.
            gs_fns, gs_err = _load_ground_sampler_fns(ctx)
            if gs_err is not None:
                return _error_result(gs_err)
            parse_notices: List[str] = []
            sketch = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=CFG.
                agent_bilevel_use_llm_initial_params,
                strict=True,
                parse_ground_samplers=CFG.agent_bilevel_ground_samplers,
                ground_sampler_fns=gs_fns or None,
                notices=parse_notices,
            )
        except Exception as e:  # pylint: disable=broad-except
            return _error_result(f"Could not parse plan sketch: {e}")
        if not sketch:
            return _error_result(
                "Parsed empty plan sketch. Check that every line names a "
                "known option with typed `obj:type` arguments matching what "
                "the inspect tools report.")

        timeout, timeout_source = bilevel_sketch.resolve_refine_timeout(
            args.get("timeout"),
            len(sketch),
            per_step=CFG.agent_bilevel_refinement_timeout_per_step,
            minimum=CFG.agent_bilevel_refinement_timeout_min)

        # Refinement accepts a parameterization only if the task evaluator
        # also scores its rollout as a solve: a candidate that reaches the
        # goal atoms yet earns no success credit (e.g. the poker, not the
        # cascade, toppled the target) is discarded and refinement is
        # resampled with a fresh rng, all attempts sharing the one timeout
        # budget. Without this gate the search happily converges onto
        # parameterizations the env would score as non-solves and reports
        # SUCCESS on them (run_20260713_172854 seed0 task1 test034). The
        # gate reads ONLY the public (terminated, reward, solved) triple -
        # the standard RL end-of-episode observables - so it grants the
        # search nothing the agent could not compute itself, and it never
        # depends on a reward sign convention.
        attempts = max(1, CFG.agent_bilevel_refine_evaluator_attempts)
        discarded_rewards: List[float] = []
        verdict_line: Optional[str] = None
        non_solve = False
        start = time.perf_counter()
        success, report = False, ""
        plan: List[Any] = []

        # In-search version of the same gate: reject a goal-atom-reaching
        # candidate DURING backtracking when the evaluator scores it as a
        # non-solve, so the search keeps moving from the same node (its
        # upstream samples intact) instead of converging onto uncertifiable
        # parameters and needing a cold restart below. The restart loop
        # stays as the safety net for verdict flakiness: the post-hoc
        # check re-rolls the accepted plan, and a re-roll that scores
        # differently (the sim is nondeterministic across runs) still
        # triggers a resample.
        _gate_evaluator = _resolve_task_evaluator(ctx, task_idx)
        solved_check = None
        if _gate_evaluator is not None:
            solved_check = make_solved_check(
                _gate_evaluator,
                getattr(ctx.option_model, "sim_env", None),
                on_reject=discarded_rewards.append)

        for attempt in range(attempts):
            remaining = timeout - (time.perf_counter() - start)
            if attempt and remaining < 5.0:
                break
            try:
                success, report, plan = \
                    bilevel_sketch.refine_and_validate_report(
                        task,
                        sketch,
                        ctx.option_model,
                        predicates=all_predicates,
                        timeout=remaining if attempt else timeout,
                        rng=np.random.default_rng(CFG.seed + attempt),
                        max_samples_per_step=CFG.
                        agent_bilevel_max_samples_per_step,
                        check_subgoals=CFG.agent_bilevel_check_subgoals,
                        log_state=CFG.agent_bilevel_log_state,
                        parameterized_samplers=ctx.parameterized_samplers
                        or None,
                        run_id="planner_refine",
                        timeout_source=timeout_source,
                        solved_check=solved_check,
                    )
            except Exception:  # pylint: disable=broad-except
                tb = _scrub_host_paths(traceback.format_exc())
                return _error_result(f"Refinement raised:\n{tb}")
            non_solve = False
            if not (success and plan):
                break
            scored = _belief_rollout_verdict(ctx, task, task_idx, plan,
                                             all_predicates)
            if scored is None:
                break
            verdict, coarse = scored
            if coarse or not verdict["terminated"] or verdict["solved"]:
                verdict_line = _format_evaluator_verdict(verdict,
                                                         coarse=coarse)
                break
            discarded_rewards.append(verdict["reward"])
            success = False
            non_solve = True

        # A failed search whose DEEPEST blocker was the in-search gate is
        # the verdict the restart loop expresses as SCORED_NON_SOLVE: the
        # sketch reaches the goal atoms but never certifiably, so say that
        # (with the change-the-sketch advice). A search that rejected a
        # candidate along the way but then failed on something else (an
        # upstream IK wall, a timeout mid-descent) keeps its own headline
        # - the near-miss line and the discard NOTE still surface the
        # rejections.
        if (not success and discarded_rewards
                and "scored non-solve" in report):
            non_solve = True
        rewards_str = ", ".join(f"{r:.2f}" for r in discarded_rewards)
        if non_solve:
            report = (
                "FAILURE: SCORED_NON_SOLVE\n"
                f"  Refinement found goal-atom-reaching parameters "
                f"{len(discarded_rewards)} time(s), but the task evaluator "
                f"scored every such rollout as a non-solve (rewards: "
                f"{rewards_str}; no success credit). The real env applies "
                "the same scoring, so these parameters can never count as "
                "a solve - change the sketch (e.g. different placements or "
                "orientations), not just the parameters.\n"
                "Last attempt detail:\n" + report)
        elif discarded_rewards:
            passed_tail = (
                "; the result above is from parameters that passed the "
                "evaluator's scoring." if success else ".")
            report += (
                f"\n  NOTE: {len(discarded_rewards)} earlier "
                f"parameterization(s) reached the goal atoms but scored as "
                f"non-solves (rewards: {rewards_str}) and were discarded "
                f"during the search{passed_tail}")

        # refine_plan_sketch is a parameter FINDER, not a submission path: on
        # success, append the parameters the search found per step so the
        # agent can submit these exact values via evaluate_option_plan (the
        # only delivery path). It deliberately does NOT capture a solved plan.
        if success and plan:
            param_lines = []
            for i, gopt in enumerate(plan):
                objs = ", ".join(o.name for o in gopt.objects)
                par = ", ".join(f"{p:.4f}" for p in gopt.params)
                param_lines.append(f"  {i}: {gopt.name}({objs})[{par}]")
            report += ("\n\nParameters found (submit these exact values via "
                       "evaluate_option_plan):\n" + "\n".join(param_lines))
            if verdict_line is not None:
                report += "\n" + verdict_line

        if parse_notices:
            report = "\n".join(f"NOTE: {n}" for n in parse_notices) + \
                "\n" + report

        return _text_result(f"Task {task_idx}:\n{report}")

    # ------------------------------------------------------------------ #
    # Scene annotation
    # ------------------------------------------------------------------ #

    return {
        "generate_bilevel_plan": generate_bilevel_plan,
        "generate_abstract_plan": generate_abstract_plan,
        "refine_plan_sketch": refine_plan_sketch,
    }


def _build_scene_tools(ctx: ToolContext, _text_result: Callable,
                       tool: Callable) -> Dict[str, Any]:
    """Scene tools (render / annotate / mutate env states)."""

    @tool(
        "annotate_scene",
        "Draw annotations (markers, lines, rectangles) at world "
        "coordinates in the 3D scene, render an image, and save it. "
        "Use this to visualize candidate placement positions or spatial "
        "relationships before committing to evaluate_option_plan. Annotations "
        "are temporary and cleaned up after rendering.",
        {
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "description": "List of annotations to draw",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["marker", "line", "rectangle"],
                                "description": "Annotation type"
                            },
                            "position": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "description": "[x, y, z] for marker"
                            },
                            "from": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "description": "[x, y, z] start point for line"
                            },
                            "to": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "description": "[x, y, z] end point for line"
                            },
                            "min_corner": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "description":
                                "[x_min, y_min, z] for rectangle"
                            },
                            "max_corner": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "description":
                                "[x_max, y_max, z] for rectangle"
                            },
                            "color": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                },
                                "description": "[r, g, b] 0-1. Default: red"
                            },
                            "size": {
                                "type": "number",
                                "description": "Marker radius or line width"
                            },
                        },
                        "required": ["type"],
                    },
                },
            },
            "required": ["annotations"],
        },
    )
    async def annotate_scene(args: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.env is None:
            return _error_result("No environment available for rendering.")
        try:
            # pylint: disable-next=import-outside-toplevel
            from predicators.envs.pybullet_env import PyBulletEnv
            if not isinstance(ctx.env, PyBulletEnv):
                return _error_result(
                    "annotate_scene requires a PyBullet environment.")
        except ImportError:
            return _error_result("PyBullet not available.")

        import pybullet as pb  # pylint: disable=import-outside-toplevel

        # Reset to visualized state (if set) or current task state
        render_state = ctx.visualized_state or (ctx.current_task.init
                                                if ctx.current_task else None)
        if render_state is not None:
            ctx.env._set_state(render_state)  # pylint: disable=protected-access

        physics_id = ctx.env._physics_client_id  # pylint: disable=protected-access
        annotations = args.get("annotations", [])
        step_label = "annotated"

        # Draw annotations, collecting IDs for cleanup
        all_debug_ids: List[int] = []
        summaries: List[str] = []
        for ann in annotations:
            try:
                ids = _draw_pybullet_annotation(ann, physics_id)
                all_debug_ids.extend(ids)
                pos = ann.get("position") or ann.get("min_corner", "?")
                summaries.append(f"  {ann['type']} at {pos}")
            except Exception as e:  # pylint: disable=broad-except
                summaries.append(f"  {ann['type']} FAILED: {e}")

        # Render with unique ID
        ctx.test_call_id += 1
        img_block = _render_pybullet_image(ctx, f"annotated_{step_label}")

        # Cleanup annotation bodies (not removeAll — preserve env's own)
        for body_id in all_debug_ids:
            try:
                pb.removeBody(body_id, physicsClientId=physics_id)
            except Exception:  # pylint: disable=broad-except
                pass

        # Build response — file path only, no inline image
        text = (f"Rendered scene with "
                f"{len(annotations)} annotation(s):\n")
        text += "\n".join(summaries)
        if img_block and img_block.get("saved_path"):
            text += (f"\n\nSaved image: "
                     f"{img_block['saved_path']}")
        return _text_result(text)

    @tool(
        "visualize_state",
        "Render the scene with objects moved to hypothetical positions. "
        "Copies a task's initial state, applies the given "
        "modifications, stores the result so subsequent annotate_scene "
        "calls render against it, and returns a rendered image.",
        {
            "type": "object",
            "properties": {
                "modifications": {
                    "type": "array",
                    "description": "List of object modifications to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "object": {
                                "type": "string",
                                "description": "Object name (e.g. 'widget0')"
                            },
                            "features": {
                                "type":
                                "object",
                                "description":
                                "Feature name → new value "
                                "(e.g. {'x': 1.05, 'y': 1.27})"
                            },
                        },
                        "required": ["object", "features"],
                    },
                },
                "step_label": {
                    "type": "string",
                    "description": "Optional label for the saved image",
                },
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index to visualize. Omit to use "
                    "the current solve-time task.",
                },
            },
            "required": ["modifications"],
        },
    )
    async def visualize_state(args: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.env is None:
            return _error_result("No environment available for rendering.")
        try:
            # pylint: disable-next=import-outside-toplevel
            from predicators.envs.pybullet_env import PyBulletEnv
            if not isinstance(ctx.env, PyBulletEnv):
                return _error_result(
                    "visualize_state requires a PyBullet environment.")
        except ImportError:
            return _error_result("PyBullet not available.")

        task_idx = args.get("task_idx")
        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_idx = "current"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        modifications = args.get("modifications", [])
        if not modifications:
            return _error_result("No modifications provided. Pass a list of "
                                 "{object, features} dicts.")

        # Copy the initial state and apply the overrides.
        modified_state, summaries, mod_err = _apply_state_modifications(
            task.init, modifications)
        if mod_err:
            return _error_result(mod_err)

        # Store for subsequent annotate_scene calls
        ctx.visualized_state = modified_state

        # Render
        step_label = args.get("step_label", "visualize")
        ctx.test_call_id += 1
        img_block = _render_pybullet_image(ctx,
                                           f"visualize_{step_label}",
                                           state=modified_state)

        text = (f"Modified state (task {task_idx}) with "
                f"{len(summaries)} feature change(s):\n")
        text += "\n".join(summaries)
        if img_block and img_block.get("saved_path"):
            text += (f"\n\nSaved image: "
                     f"{img_block['saved_path']}")
        text += ("\n\nSubsequent annotate_scene calls will render "
                 "against this modified state.")
        return _text_result(text)

    return {
        "annotate_scene": annotate_scene,
        "visualize_state": visualize_state,
    }


def _build_exploration_tools(ctx: ToolContext, _text_result: Callable,
                             tool: Callable) -> Dict[str, Any]:
    """Solve-phase ``explore_python`` over the ProbeSim exploration facade.

    The namespace is deliberately tiny (probe facade + numpy): the probe
    reuses the exact machinery behind ``evaluate_option_plan`` (same
    plan grammar, same option-model executor, same renderer) but carries
    no scoring surface - nothing run here can be captured as the answer,
    so it is safe to hand the agent as a freely composable physics
    probe. Built only when the session's config opts in: the
    ``tool_names=None`` legacy surface would otherwise grant every
    default-configured session an in-process exec tool.
    """
    if not CFG.agent_planner_use_explore_python:
        return {}
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.probe_api import build_probe_namespace

    # Synthesis sessions install a candidate-simulator provider before
    # opening the session (see ToolContext.probe_option_model_provider);
    # its presence at build time is exactly what flips the probe's
    # semantics, so the description follows it.
    synthesis_probe = ctx.probe_option_model_provider is not None
    if synthesis_probe:
        sim_desc = (
            "`sim` (a ProbeSim over the CANDIDATE simulator: your current "
            "simulator.py with freshly MCMC-fitted params, rebuilt "
            "automatically when the file changes - so probes always "
            "exercise what you just wrote; errors until a loadable "
            "simulator.py exists)")
        reset_desc = (
            "`sim.reset(task_idx, mods=None)` sets the current state "
            "to a train task's init (task_idx is required in this "
            "session), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        submit_desc = (
            "EXPLORATORY "
            "ONLY: nothing run here is captured and no task-evaluator "
            "verdict is computed - validate the simulator itself via "
            "evaluate_plan_refinement.")
    else:
        sim_desc = "`sim` (a ProbeSim over the belief simulator)"
        reset_desc = (
            "`sim.reset(task_idx=None, mods=None)` sets the current state "
            "to a task's init (current task by default), optionally with "
            "feature overrides (`mods={'obj': {'x': 1.05}}`); ")
        submit_desc = (
            "EXPLORATORY "
            "ONLY: nothing run here is captured as your answer and no "
            "task-evaluator verdict is computed - validate and submit the "
            "final plan via evaluate_option_plan from the true initial "
            "state.")
    explore_python = _make_python_exec_tool(
        tool,
        name="explore_python",
        description=
        ("Execute Python code for cheap physics/geometry exploration in "
         "a persistent namespace (variables survive across calls - "
         "define helpers and sweep loops once, reuse them). Available: "
         f"{sim_desc}, `ProbeSim()` "
         "(extra independent instances), `np`. ProbeSim API: "
         f"{reset_desc}"
         "`sim.run(plan_text, render=True, trials=1)` executes an option "
         "plan FROM THE CURRENT "
         "STATE (same grammar as evaluate_option_plan; print the result "
         "for per-step outcomes incl. saved per-step scene-image paths - "
         "view them with the Read tool; pass render=False inside tight "
         "sweep loops) and advances the state; trials=N repeats the plan "
         "N times (fresh physics per trial when available) and returns "
         "the per-trial outcomes + success count WITHOUT advancing the "
         "state - use it for reliability estimates instead of "
         "hand-rolled repeat loops (restore/rerun repeats share solver "
         "state and read optimistic); "
         "`sim.state()` / "
         "`sim.state('obj')` full-precision features; `sim.atoms()`; "
         "`sim.render(label, annotations=None)` saves an image "
         "(returns its path; Read it to view), "
         "optionally overlaying marker/line/rectangle dicts "
         "(`{'type': 'marker', 'position': [x, y, z], 'color': "
         "[r, g, b], 'size': s}`; lines use `from`/`to`, rectangles "
         "`min_corner`/`max_corner`) to check offsets and reference "
         "points visually; `sim.snapshot()` / "
         "`sim.restore(id)` bank and rewind states (use to re-try "
         "different actions from one setup, or resume after a fixed "
         "plan prefix without re-running it); "
         "`sim.refine(sketch_text, timeout=60, require_goal=False, "
         "require_solved=False)` runs "
         "backtracking parameter search FROM THE CURRENT STATE (same "
         "grammar/search as refine_plan_sketch"
         f"{_region_syntax_blurb()}; "
         "success = each step establishes its `-> {subgoals}` "
         "annotation, and the result's Verdict line states what it "
         "certifies) - refine a plan SUFFIX from a snapshot so the "
         "budget goes to the step that matters; the result reports "
         "best-found params even on timeout, per-step sample counts, and "
         "the deepest near-miss. require_solved=True (only from an "
         "unmodified reset() state) additionally requires the task "
         "evaluator to score the final rollout solved=True, rejecting "
         "goal-reaching-but-unscored candidates during the search. "
         "print() output is "
         "returned; oversize output is spilled to "
         "`tool_outputs/explore_python/` (Read/Grep it back). " +
         (f"Each call has a {CFG.agent_sdk_explore_python_call_timeout:.0f}s "
          "wall-clock limit (checked between sim calls, plus a hard stop "
          "for sim-free code; printed output up to the stop is "
          "returned): budget sweeps accordingly - "
          "prefer coarse-to-fine over exhaustive grids, and print "
          "intermediate bests so partial results survive a stop. "
          if CFG.agent_sdk_explore_python_call_timeout > 0
          and not synthesis_probe else "") + f"{submit_desc}"),
        exec_ns=build_probe_namespace(ctx),
        sandbox_dir=ctx.sandbox_dir,
        text_result=_text_result,
        budget_ctx=ctx,
    )
    return {"explore_python": explore_python}


def _build_journal_tools(ctx: ToolContext, _text_result: Callable,
                         tool: Callable) -> Dict[str, Any]:
    """``record_journal`` - agent-authored entries in the run's solve
    journal.

    Built only when the journal is enabled. The journal is the curated
    cross-attempt/cross-task memory channel for fresh-context solve
    sessions, so the tool guidance insists on facts and measurements:
    recorded verdicts ("X is impossible") from a failed attempt would
    re-import exactly the anchoring a restart is meant to shed
    (run_20260717_230436 seed1 concluded a "hard collision boundary"
    its sibling run placed through minutes later).
    """
    if not CFG.agent_solve_use_journal:
        return {}
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk import journal as journal_mod

    @tool(
        "record_journal",
        ("Append a short entry to the run's persistent solve journal "
         "(journal.md), which future solve attempts - starting with FRESH "
         "context - read in their prompt. Record durable, transferable "
         "facts: what you tried with exact parameters, what you measured, "
         "what worked (and its load-bearing values), and what a fresh "
         "attempt should try differently. Facts and measurements ONLY - do "
         "NOT record conclusions like 'X is impossible' or 'the task "
         "requires Y' (a wrong verdict anchors every later attempt; the "
         "evidence lets them re-judge). Keep it skimmable: a few bullets, "
         f"under {journal_mod.MAX_ENTRY_CHARS} chars."),
        {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "The journal entry (markdown bullets).",
                }
            },
            "required": ["entry"],
        },
    )
    async def record_journal(args: Dict[str, Any]) -> Dict[str, Any]:
        entry = (args.get("entry") or "").strip()
        if not entry:
            return _error_result("`entry` is required.")
        if not ctx.sandbox_dir:
            return _error_result("No sandbox directory in this session.")
        if ctx.test_task_idx is not None:
            where = f"test task {ctx.test_task_idx}"
        else:
            where = "pre-test phase"
        attempt = f", attempt {ctx.attempt_index}" if ctx.attempt_index else ""
        note = journal_mod.append_entry(ctx.sandbox_dir,
                                        f"Agent notes ({where}{attempt})",
                                        entry)
        msg = "Recorded to the solve journal."
        if note is not None:
            msg += f" NOTE: {note}."
        return _text_result(msg)

    return {"record_journal": record_journal}


def create_mcp_tools(ctx: ToolContext,
                     tool_names: Optional[List[str]] = None) -> list:
    """Create MCP tools with the given ToolContext via closures.

    Args:
        ctx: Shared mutable state between the approach and MCP tools.
        tool_names: If provided, only return tools with these names.
            If None, return all tools.

    Returns a list of SdkMcpTool objects to pass to create_sdk_mcp_server.
    """
    from claude_agent_sdk import \
        tool as _sdk_tool  # pylint: disable=import-outside-toplevel
    tool = _make_coercing_tool(_sdk_tool)

    # Spill oversize tool output into the sandbox (``./tool_outputs/``)
    # instead of returning it inline. Each builder names its parameter
    # ``_text_result`` so every nested tool's ``_text_result(...)`` call
    # routes through the spiller with no call-site edits.
    _text_result = _make_spilling_text_result(ctx.sandbox_dir)

    _all = {
        **_build_inspection_tools(ctx, _text_result, tool),
        **_build_proposal_tools(ctx, _text_result, tool),
        **_build_retraction_tools(ctx, _text_result, tool),
        **_build_testing_tools(ctx, _text_result, tool),
        **_build_planning_tools(ctx, _text_result, tool),
        **_build_scene_tools(ctx, _text_result, tool),
        **_build_exploration_tools(ctx, _text_result, tool),
        **_build_journal_tools(ctx, _text_result, tool),
    }
    if tool_names is None:
        tools = list(_all.values())
    else:
        tools = [_all[n] for n in tool_names if n in _all]
    tools.extend(ctx.extra_mcp_tools)
    return tools


# ── Sim-learning tools ───────────────────────────────────────────


class _SnapshotTarget:  # pylint: disable=too-few-public-methods
    """One file to watch for write-time snapshots."""

    def __init__(
        self,
        live_file: str,
        versions_dir: str,
        artifact_name: str,
        cycle_index_provider: Callable[[], int],
    ) -> None:
        self.live_file = os.path.realpath(live_file)
        self.versions_dir = versions_dir
        self.artifact_name = artifact_name
        self.cycle_index_provider = cycle_index_provider


def make_write_snapshot_hook(
    targets: List[_SnapshotTarget],
    sandbox_dir: str,
) -> Callable[..., Any]:
    """Build a PostToolUse hook that snapshots target files on Write/Edit.

    The returned async callable matches the Claude Agent SDK's hook
    signature ``(hook_input, tool_use_id, hook_context) -> dict``. It
    fires after a successful Write / Edit / MultiEdit / NotebookEdit
    and, if the tool's ``file_path`` (resolved against ``sandbox_dir``)
    matches any target's ``live_file``, writes a new versioned snapshot
    (via :func:`finalize_versioned_snapshot`).

    Dedup-by-hash means a no-op Edit that produces identical content
    leaves no new file. Failures are swallowed — a snapshot hook
    failing should never break the agent's edit loop.
    """
    abs_sandbox = os.path.abspath(sandbox_dir)

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return os.path.realpath(path)
        return os.path.realpath(os.path.join(abs_sandbox, path))

    target_by_path: Dict[str,
                         _SnapshotTarget] = {t.live_file: t
                                             for t in targets}

    async def _hook(hook_input: Any, _tool_use_id: Any,
                    _context: Any) -> Dict[str, Any]:
        try:
            tool_name = getattr(hook_input, "tool_name", None)
            if tool_name not in {"Write", "Edit", "MultiEdit"}:
                return {}
            tool_input = getattr(hook_input, "tool_input", None) or {}
            raw_path = tool_input.get("file_path")
            if not raw_path:
                return {}
            resolved = _resolve(raw_path)
            target = target_by_path.get(resolved)
            if target is None:
                return {}
            finalize_versioned_snapshot(
                target.live_file,
                target.versions_dir,
                cycle_idx=int(target.cycle_index_provider()),
                artifact_name=target.artifact_name,
            )
        except Exception:  # pylint: disable=broad-except
            # Never let a snapshot failure break the agent's edit loop.
            pass
        return {}

    return _hook


def finalize_versioned_snapshot(
    live_file: str,
    versions_dir: str,
    cycle_idx: int,
    artifact_name: str,
) -> Optional[str]:
    """Take a final ``cycle_XXX_vers_(YYY+1)`` snapshot if needed.

    Called from the approach after the agent session ends so that any
    post-evaluation edits to ``live_file`` (which would otherwise be
    lost — the synthesis tools only snapshot on eval calls) are
    captured. If the live file's hash matches the highest existing
    ``cycle_XXX_vers_YYY_<artifact_name>.py`` in ``versions_dir`` (this
    cycle), the existing tag is returned and no new file is written.

    Args:
        live_file: Host path to the file (e.g. simulator.py).
        versions_dir: Directory containing the per-call snapshots.
        cycle_idx: Current cycle (1-indexed) — used to find the highest
            existing ``vers_YYY`` for this cycle and to name the new
            snapshot.
        artifact_name: Stem used in the filename, e.g. ``"simulator"``
            or ``"predicates"``.

    Returns the final version tag (``cycle_XXX_vers_YYY``) or ``None``
    if ``live_file`` does not exist.
    """
    if not os.path.isfile(live_file):
        return None
    with open(live_file, "rb") as f:
        live_raw = f.read()
    live_digest = hashlib.sha256(live_raw).hexdigest()

    prefix = f"cycle_{cycle_idx:03d}_vers_"
    suffix = f"_{artifact_name}.py"
    highest_vers = 0
    highest_path: Optional[str] = None
    if os.path.isdir(versions_dir):
        for name in os.listdir(versions_dir):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            vers_str = name[len(prefix):-len(suffix)]
            try:
                vers = int(vers_str)
            except ValueError:
                continue
            if vers > highest_vers:
                highest_vers = vers
                highest_path = os.path.join(versions_dir, name)

    if highest_path is not None:
        with open(highest_path, "rb") as f:
            existing_digest = hashlib.sha256(f.read()).hexdigest()
        if existing_digest == live_digest:
            return f"cycle_{cycle_idx:03d}_vers_{highest_vers:03d}"

    os.makedirs(versions_dir, exist_ok=True)
    new_vers = highest_vers + 1
    snap_path = os.path.join(
        versions_dir,
        f"cycle_{cycle_idx:03d}_vers_{new_vers:03d}_{artifact_name}.py")
    with open(snap_path, "wb") as f:
        f.write(live_raw)
    return f"cycle_{cycle_idx:03d}_vers_{new_vers:03d}"


class _ArtifactSnapshotter:
    """Per-call versioned snapshotting for one artifact file.

    Used by the synthesis-tools factories to dedup snapshots by SHA256
    and tag each load with ``cycle_XXX_vers_YYY``. ``YYY`` is per
    instance and starts at 0 — it resets each time a new snapshotter is
    created (typically once per factory call). ``XXX`` is read from
    ``cycle_index_provider`` at each call so live cycle bumps are
    reflected in subsequent tags.
    """

    def __init__(
        self,
        live_file: str,
        versions_dir: str,
        artifact_name: str,
        cycle_index_provider: Optional[Callable[[], int]],
        missing_file_hint: str = "",
    ) -> None:
        self._live_file = live_file
        self._versions_dir = versions_dir
        self._artifact_name = artifact_name
        self._cycle_index_provider = cycle_index_provider
        self._missing_file_hint = missing_file_hint
        self._version_count = 0
        self._last_digest: Optional[str] = None

    def current_cycle(self) -> int:
        """Return the active learning-cycle index, or 0 if unknown."""
        if self._cycle_index_provider is None:
            return 0
        try:
            return int(self._cycle_index_provider())
        except Exception:  # pylint: disable=broad-except
            return 0

    def snapshot(
        self,
        path: Optional[str] = None,
    ) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Read the live file and write a versioned snapshot on change.

        Returns ``(raw_bytes, version_tag, error_msg)``. On a missing
        file, ``raw_bytes`` and ``version_tag`` are ``None`` and
        ``error_msg`` carries a user-facing message (suffixed with
        ``missing_file_hint`` when configured).

        ``path`` may override the configured ``live_file`` per call —
        the snapshotter still writes into the configured
        ``versions_dir`` under ``artifact_name``, sharing the version
        counter and digest cache so dedup spans both files.
        """
        target = path or self._live_file
        if not os.path.isfile(target):
            msg = (f"{self._artifact_name.capitalize()} file not found: "
                   f"{target}.")
            if self._missing_file_hint:
                msg = f"{msg} {self._missing_file_hint}"
            return None, None, msg
        with open(target, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        cycle_idx = self.current_cycle()
        if digest != self._last_digest:
            self._version_count += 1
            os.makedirs(self._versions_dir, exist_ok=True)
            snap_path = os.path.join(
                self._versions_dir, f"cycle_{cycle_idx:03d}_vers_"
                f"{self._version_count:03d}_{self._artifact_name}.py")
            with open(snap_path, "wb") as f:
                f.write(raw)
            self._last_digest = digest
        return raw, (f"cycle_{cycle_idx:03d}_vers_"
                     f"{self._version_count:03d}"), None


def _make_python_exec_tool(
    tool: Callable,
    *,
    name: str,
    description: str,
    exec_ns: Dict[str, Any],
    sandbox_dir: Optional[str],
    sandbox_dir_for_agent: Optional[str] = None,
    text_result: Callable[[str], Dict[str, Any]],
    budget_ctx: Optional[ToolContext] = None,
) -> Any:
    """Build a code-execution MCP tool over a persistent namespace.

    Shared core behind the synthesis-phase ``run_python`` (namespace =
    trajectory data) and the solve-phase ``explore_python`` (namespace =
    the ``ProbeSim`` exploration facade): sandbox-escape screening, in-
    process ``exec`` with stdout capture, and oversize-output spill to
    ``<sandbox_dir>/tool_outputs/<name>/``. The namespace persists
    across calls, so agents can define helpers once and reuse them.

    ``budget_ctx`` (the solve session's ToolContext) opts the tool into
    wall-clock budgeting: each call arms the per-call deadline
    (``CFG.agent_sdk_explore_python_call_timeout``) that probe sim calls
    enforce cooperatively, a call arriving after the attempt deadline is
    refused with a submit-now message, and every result carries a
    ``[budget]`` footer (attempt time + rollout counts) so sweeps have a
    visible price.
    """
    # pylint: disable=import-outside-toplevel
    import io
    import sys
    import traceback  # pylint: disable=redefined-outer-name,reimported

    # pylint: enable=import-outside-toplevel

    inline_char_limit = 30000
    preview_head_lines = 30
    preview_tail_lines = 30
    outputs_subdir = os.path.join("tool_outputs", name)
    outputs_dir_host: Optional[str] = (os.path.join(
        sandbox_dir, outputs_subdir) if sandbox_dir else None)
    if sandbox_dir_for_agent:
        outputs_dir_agent: Optional[str] = (
            f"{sandbox_dir_for_agent.rstrip('/')}/"
            f"{outputs_subdir.replace(os.sep, '/')}")
    else:
        outputs_dir_agent = outputs_dir_host
    # Continue numbering after any spill files already in the directory,
    # so re-created instances sharing a sandbox never overwrite earlier
    # outputs.
    count = [0]
    if outputs_dir_host and os.path.isdir(outputs_dir_host):
        count[0] = len(os.listdir(outputs_dir_host))

    @tool(
        name,
        description,
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                }
            },
            "required": ["code"],
        },
    )
    async def python_exec(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.probe_api import ProbeBudgetExceeded
        code = args["code"]
        # The code execs in-process with full filesystem access, and the
        # sandbox's PreToolUse file-path hook does not cover MCP tools, so
        # screen the code here for out-of-sandbox reads / source
        # introspection before executing (best-effort; see
        # _screen_text_for_sandbox_escape).
        if sandbox_dir is not None:
            reason = _screen_text_for_sandbox_escape(code, sandbox_dir)
            if reason is not None:
                return text_result(
                    f"Error: sandbox guard blocked this code - {reason}. "
                    "Read files with Read/Grep and use the MCP tools and "
                    "./reference/ files instead.")
        rollouts_before = 0
        if budget_ctx is not None:
            rollouts_before = budget_ctx.attempt_rollout_count
            attempt_dl = budget_ctx.attempt_deadline
            if (attempt_dl is not None and time.monotonic() > attempt_dl
                    and not budget_ctx.capture_best_effort_plan):
                return text_result(
                    "The attempt's wall-clock exploration budget is "
                    "exhausted - this call was not run. Submit your single "
                    "best plan NOW via evaluate_option_plan on the current "
                    "task (omit task_idx)." +
                    _budget_footer(budget_ctx, rollouts_before))
            call_timeout = CFG.agent_sdk_explore_python_call_timeout
            if budget_ctx.probe_option_model_provider is not None:
                # Synthesis sessions probe the CANDIDATE simulator, whose
                # rollouts are far slower than belief-sim ones and whose
                # reset can trigger a fresh fit - a legitimate single call
                # can exceed the solve-tuned cap, so synthesis is exempt
                # from the per-call limit.
                call_timeout = 0.0
            budget_ctx.explore_call_deadline = (time.monotonic() + call_timeout
                                                if call_timeout > 0 else None)

        def _footer() -> str:
            if budget_ctx is None:
                return ""
            return _budget_footer(budget_ctx, rollouts_before)

        # Hard watchdog for pure-Python code that never reaches a probe
        # checkpoint (see _arm_budget_watchdog): armed to the nearest of
        # the per-call and attempt deadlines.
        watchdog_disarm: Optional[Callable[[], None]] = None
        if budget_ctx is not None:
            wd_deadlines = []
            if budget_ctx.explore_call_deadline is not None:
                wd_deadlines.append(budget_ctx.explore_call_deadline)
            if (budget_ctx.attempt_deadline is not None
                    and not budget_ctx.capture_best_effort_plan):
                wd_deadlines.append(budget_ctx.attempt_deadline)
            if wd_deadlines:
                remaining = min(wd_deadlines) - time.monotonic()
                if remaining > 0:
                    watchdog_disarm = _arm_budget_watchdog(remaining)

        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        try:
            exec(code, exec_ns)  # pylint: disable=exec-used
        except ProbeBudgetExceeded as e:
            partial = captured.getvalue()
            prefix = f"{partial}\n" if partial else ""
            # The watchdog's async exception carries no message; give it
            # the same actionable framing as the cooperative checks.
            msg = str(e) or (
                "this call exceeded its wall-clock budget and was stopped "
                "mid-execution; output printed so far is returned above. "
                "Split the work into smaller calls, and print intermediate "
                "results so partial progress survives a stop.")
            return text_result(f"{prefix}TIME BUDGET: {msg}{_footer()}")
        except Exception:  # pylint: disable=broad-except
            tb = _scrub_host_paths(traceback.format_exc())
            partial = captured.getvalue()
            prefix = f"{partial}\n" if partial else ""
            return text_result(f"{prefix}Error:\n{tb}{_footer()}")
        finally:
            if watchdog_disarm is not None:
                watchdog_disarm()
            sys.stdout = old_stdout
            if budget_ctx is not None:
                budget_ctx.explore_call_deadline = None

        output = captured.getvalue()
        if not output:
            return text_result(f"(no output){_footer()}")

        output += _footer()
        if len(output) <= inline_char_limit or outputs_dir_host is None:
            return text_result(output)

        count[0] += 1
        os.makedirs(outputs_dir_host, exist_ok=True)
        filename = f"call_{count[0]:04d}.txt"
        host_path = os.path.join(outputs_dir_host, filename)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(output)

        out_lines = output.splitlines()
        total_lines = len(out_lines)
        head = out_lines[:preview_head_lines]
        tail = (out_lines[-preview_tail_lines:] if total_lines >
                (preview_head_lines + preview_tail_lines) else [])
        agent_path = (f"{outputs_dir_agent}/{filename}"
                      if outputs_dir_agent else host_path)
        preview_parts = [
            f"[{name} output too large to inline: "
            f"{len(output):,} chars across {total_lines:,} lines; "
            f"full output saved to {agent_path}. Use Read/Grep to "
            f"inspect, or rerun with narrower print() to keep results "
            f"inline.]",
            "",
            f"--- head ({len(head)} lines) ---",
            *head,
        ]
        if tail:
            omitted = total_lines - len(head) - len(tail)
            preview_parts.extend([
                "",
                f"... [{omitted:,} lines omitted] ...",
                "",
                f"--- tail ({len(tail)} lines) ---",
                *tail,
            ])
        return text_result("\n".join(preview_parts))

    return python_exec


def create_synthesis_tools(
    exec_ns: Dict[str, Any],
    base_pred_triples: list,
    inferred_process_features: Dict[str, List[str]],
    simulator_file: str,
    versions_dir: str,
    approach: Optional[Any] = None,
    sandbox_dir: Optional[str] = None,
    sandbox_dir_for_agent: Optional[str] = None,
    cycle_index_provider: Optional[Callable[[], int]] = None,
) -> list:
    """Create MCP tools for the sim-learning synthesis agent.

    Returns ``[run_python, evaluate_step_fit, report_residuals,
    evaluate_plan_refinement]``.

    The agent's source-of-truth for the simulator is the file at
    ``simulator_file`` (which it edits with ``Write`` / ``Edit``). The
    three synthesis tools each ``exec`` that file fresh into an
    isolated namespace per call and read ``PROCESS_RULES``,
    ``PARAM_SPECS``, ``PROCESS_FEATURES`` from it — no namespace state
    leaks across iterations. Before loading, every call also snapshots
    the current contents into ``versions_dir`` as
    ``cycle_XXX_vers_YYY_simulator.py`` (``XXX`` from
    ``cycle_index_provider()``, ``YYY`` resetting per
    ``create_synthesis_tools`` call) so the full history of evaluated
    versions is preserved across cycles; identical-content calls reuse
    the prior snapshot. Each tool's output is prefixed with the version
    tag (``[cycle_XXX_vers_YYY]``).

    * ``run_python`` — executes arbitrary Python in a persistent
      namespace pre-loaded with trajectory data. Use this for ad-hoc
      exploration of ``trajectories`` etc.; it does **not** define
      rules — write ``simulator.py`` for that.
    * ``evaluate_step_fit`` — SSE of the current ``PROCESS_RULES`` at
      init_value params, plus post-fit SSE, percent improvement, and
      fitted parameter values from a parameter fit.
    * ``report_residuals`` — per-feature breakdown of where the
      current rules disagree with observations: mismatch counts,
      mean/max abs error, comparison to the no-rule baseline, and
      worst-N example transitions per feature.
    * ``evaluate_plan_refinement`` — builds the combined simulator
      from current rules+params and runs backtracking refinement on a
      training task, reporting where (if anywhere) the planner gets
      stuck. Requires ``approach`` to be passed.

    Args:
        exec_ns: Persistent namespace for ``run_python``. Should
            contain ``trajectories``, ``np``, ``ParamSpec``.
        base_pred_triples: ``(s_base, action, s_next_obs)`` triples
            with the base step already advanced — eval/test consume
            ``s_base`` directly so no live env is needed.
        inferred_process_features: Data-driven default scope used
            when the agent hasn't declared ``PROCESS_FEATURES`` in
            ``simulator.py`` yet.
        simulator_file: Host path to the canonical simulator file
            the agent edits. Synthesis tools ``exec`` this file
            fresh on every call.
        versions_dir: Directory to write per-call snapshots into
            (created on first use).
        approach: ``AgentSimLearningApproach`` instance, used by
            ``evaluate_plan_refinement`` to access training tasks,
            build the combined simulator/option model, and run
            refinement. If ``None``, that tool returns an error.
        sandbox_dir: Host path to the agent's sandbox root.  When set,
            ``run_python`` spills oversize output to
            ``<sandbox_dir>/tool_outputs/run_python/`` instead of
            letting the agent SDK truncate and dump it to
            ``~/.claude/projects/.../tool-results/``.  When ``None``,
            output is always returned inline.
        sandbox_dir_for_agent: Path prefix the agent sees for
            ``sandbox_dir`` (e.g. ``"."`` for local sandbox or
            ``"/sandbox"`` for docker).  Used only when building the
            human-readable path included in the spilled-output message.
        cycle_index_provider: Callable returning the current online
            learning cycle (1-indexed). Read at snapshot time so the
            same tools instance reflects later cycle bumps. If ``None``,
            cycle defaults to 0 (still valid; produces
            ``cycle_000_vers_YYY``).
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported
    from collections import defaultdict

    from claude_agent_sdk import tool as _sdk_tool
    tool = _make_coercing_tool(_sdk_tool)

    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    from predicators.code_sim_learning.fit_space import ParamSpec
    from predicators.code_sim_learning.fitting import compute_sse, \
        compute_sse_recurrent
    from predicators.code_sim_learning.physical_sysid import \
        compute_residual_scaling, compute_rollout_sse, \
        fit_params_rollout_trimmed, format_identifiability, \
        identifiability_report, physical_param_anchors, \
        select_trustworthy_params
    from predicators.code_sim_learning.synthesis_validation import \
        run_refinement_for_synthesis
    from predicators.code_sim_learning.utils import apply_rules, \
        has_latent_rules, iter_feature_residuals, read_latent_init, \
        read_physical_param_specs, read_simulator_components, \
        rollout_predictions, stamp_physical_spec_scales

    # pylint: enable=import-outside-toplevel

    _snapshotter = _ArtifactSnapshotter(
        live_file=simulator_file,
        versions_dir=versions_dir,
        artifact_name="simulator",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with PROCESS_RULES, "
                           "PARAM_SPECS, PROCESS_FEATURES."),
    )
    # Spill oversize output from the synthesis tools into the sandbox,
    # so nothing is dumped to ``~/.claude/projects/.../tool-results/``.
    # (``run_python``'s own spill lives in ``_make_python_exec_tool``.)
    _text = _make_spilling_text_result(sandbox_dir,
                                       agent_prefix=sandbox_dir_for_agent)

    def _snapshot_and_load(
            path: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(rules, specs, features, latent_init, physical_specs,
        version_tag, error_msg)``; ``error_msg`` is ``None`` on success.
        ``latent_init`` is the optional ``LATENT_INIT`` export (``None``
        for fully- observable simulators) — the synthesis tools need it
        to score recurrent (5-arg) rules through the latent-threaded
        path. ``physical_specs`` is the optional ``PHYSICAL_PARAMS``
        export (system identification); when present, a physics-only
        artifact is valid and missing rules/specs default to empty
        lists. Snapshots are deduped by SHA256, so repeated calls on
        unchanged content reuse the prior ``cycle_XXX_vers_YYY`` tag.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return None, None, None, None, None, None, err
        assert raw is not None and version_tag is not None
        ns: Dict[str, Any] = {"np": np, "ParamSpec": ParamSpec}
        try:
            exec(raw.decode("utf-8"), ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            return None, None, None, None, None, version_tag, (
                f"[{version_tag}] Error executing {path}:\n"
                f"{_scrub_host_paths(traceback.format_exc())}")
        rules, specs, features = read_simulator_components(ns)
        latent_init = read_latent_init(ns)
        physical_specs = read_physical_param_specs(ns)
        if rules is None:
            if not physical_specs:
                return None, None, None, None, None, version_tag, (
                    f"[{version_tag}] PROCESS_RULES missing or empty in "
                    f"{path}.")
            rules = []
        if specs is None:
            if not physical_specs:
                return None, None, None, None, None, version_tag, (
                    f"[{version_tag}] PARAM_SPECS missing or empty in "
                    f"{path}.")
            specs = []
        return (rules, specs, features, latent_init, physical_specs,
                version_tag, None)

    def _groups_for(triples: list) -> List[List[Tuple[Any, Any, Any]]]:
        """Slice flat base-pred triples into per-trajectory groups.

        Recurrent rules thread their latent block within a trajectory,
        so scoring/residuals must regroup the flat triples the same way
        the engine does. Reuses the bound approach's grouping (keyed off
        the same ``_fit_trajectories`` cache the engine uses); falls
        back to a single group when no approach is bound or the lengths
        don't line up — correct for the common single-demo case.
        """
        if approach is not None and hasattr(approach,
                                            "_group_triples_by_trajectory"):
            grouped = approach._group_triples_by_trajectory(  # pylint: disable=protected-access
                triples)
            if grouped:
                return grouped
        return [triples]

    def _evaluate_rollout_fit(rules: list, rule_specs: list,
                              physical_specs: list, latent_init: Any,
                              process_features: Dict[str, List[str]],
                              scope_note: str,
                              version_tag: str) -> Dict[str, Any]:
        """Joint physical+rule system-ID fit on free-running rollouts.

        Reached from ``evaluate_step_fit`` when the artifact declares
        ``PHYSICAL_PARAMS``. Needs the bound approach for the raw
        (states, actions) trajectories and the dedicated headless fit
        env. On success the identified physical values are applied in
        place to the approach's planning base env, so a subsequent
        ``evaluate_plan_refinement`` call plans against the calibrated
        sim.
        """
        if approach is None:
            return _text(
                f"[{version_tag}] Error: PHYSICAL_PARAMS requires a bound "
                "approach (raw trajectories + base env) — unavailable in "
                "this session.")
        # Stamp the fit scale (log vs linear) from the env registry —
        # the agent's declaration carries name/init/bounds only.
        physical_specs = stamp_physical_spec_scales(physical_specs,
                                                    approach._base_env)  # pylint: disable=protected-access
        rollouts = approach._rollout_fit_trajectories(  # pylint: disable=protected-access
            process_features)
        if not rollouts:
            return _text(
                f"[{version_tag}] Error: no complete (states, actions) "
                "trajectories are available, so the rollout system-ID fit "
                "cannot run. PHYSICAL_PARAMS needs full trajectories, not "
                "isolated transitions.")
        # Factory, not an instance: every rollout runs in a fresh env.
        fit_env = approach._get_rollout_fit_env()  # pylint: disable=protected-access
        physical_names = [s.name for s in physical_specs]
        init_params = {
            s.name: s.init_value
            for s in list(physical_specs) + list(rule_specs)
        }
        anchors = physical_param_anchors(
            approach._base_env,  # pylint: disable=protected-access
            physical_specs)
        try:
            # One scaling object per fit: every SSE/RMS below must share
            # it or their values are incomparable.
            scaling = compute_residual_scaling(rollouts, process_features)
            pre_sse = compute_rollout_sse(fit_env, rollouts, init_params,
                                          process_features, physical_names,
                                          rules, latent_init, scaling)
            fit_result, survivors, traj_rms = fit_params_rollout_trimmed(
                fit_env,
                rollouts,
                physical_specs,
                process_features,
                rules=rules,
                rule_specs=rule_specs,
                latent_init=latent_init,
                scaling=scaling,
                anchors=anchors,
                rms_cache=getattr(approach, "_explainability_cache", None))
            if not survivors:
                # Honest empty-data output: no fit ran, nothing was
                # applied - do NOT print fitted values or identifiability
                # verdicts computed on zero surviving data (chaos makes
                # the probe report "identified" for everything).
                rms_str = ", ".join(f"{r:.4g}" for r in traj_rms)
                return _text("\n".join([
                    f"[{version_tag}] NO FIT RAN: all {len(rollouts)} "
                    "recorded motion segments were unexplainable at ANY "
                    "candidate physical parameters (per-segment "
                    f"best-achievable RMS [{rms_str}] all above the "
                    "trimming threshold).",
                    "",
                    "Parameters were left at their baselines; nothing was "
                    "applied to the planning base env.",
                    "",
                    "This usually means the recorded interactions are not "
                    "repeatable under replay (prolonged scraping/jamming "
                    "robot-object contact is chaotic). Collect experiments "
                    "whose outcome is dominated by object dynamics: actuate "
                    "one or two objects cleanly, then let the scene evolve "
                    "and settle on its own.",
                ]))
            probe_rollouts = survivors
            fitted = fit_result.point_estimate
            post_sse = compute_rollout_sse(fit_env, probe_rollouts, fitted,
                                           process_features, physical_names,
                                           rules, latent_init, scaling)
        except Exception as e:  # pylint: disable=broad-except
            return _text(
                f"[{version_tag}] Error: rollout system-ID fit failed:\n{e}")

        def rollout_sse_fn(params: Dict[str, float]) -> float:
            return compute_rollout_sse(fit_env, probe_rollouts, params,
                                       process_features, physical_names, rules,
                                       latent_init, scaling)

        ident_report = identifiability_report(fit_result,
                                              rollout_sse_fn,
                                              list(physical_specs) +
                                              list(rule_specs),
                                              num_explainable=len(survivors))
        applied = select_trustworthy_params(fitted, init_params,
                                            physical_names, ident_report,
                                            anchors)
        approach._apply_identified_physical_params(applied)  # pylint: disable=protected-access
        if hasattr(approach, "_record_sysid_diagnostics"):
            approach._record_sysid_diagnostics(  # pylint: disable=protected-access
                ident_report, physical_names, len(survivors), len(rollouts),
                traj_rms)
        kept_at_init = sorted(n for n in physical_names
                              if applied[n] != fitted[n])
        if pre_sse > 0:
            pct_str = f"({(pre_sse - post_sse) / pre_sse * 100:+.1f}% vs init)"
        else:
            pct_str = "(init SSE was 0)"
        lines = [
            f"[{version_tag}] JOINT ROLLOUT SYSTEM-ID FIT (PHYSICAL_PARAMS "
            f"declared) on {len(rollouts)} motion segments (scope: "
            f"{scope_note}; {len(physical_names)} physical + "
            f"{len(list(rule_specs))} rule params). Residuals are "
            "per-feature normalized (angles wrapped), so SSE/RMS are "
            "dimensionless fractions of typical motion.",
            "",
            f"At init params:   rollout SSE = {pre_sse:.6f}",
            f"After joint fit:  rollout SSE = {post_sse:.6f}  {pct_str}",
            "",
            "Fitted parameters:",
        ]
        if len(survivors) < len(rollouts):
            rms_str = ", ".join(f"{r:.4g}" for r in traj_rms)
            lines.insert(
                1,
                f"Goodness-of-fit trimming: {len(rollouts) - len(survivors)}"
                f" of {len(rollouts)} motion segments were unexplainable at "
                "ANY candidate params (per-segment best-achievable RMS: "
                f"[{rms_str}]) and were dropped before fitting; the fit "
                "below used only the explainable ones. Unexplainable "
                "segments are not repeatable under replay - prefer "
                "experiments whose outcome is dominated by object dynamics "
                "rather than prolonged robot-object contact.")
        for name in sorted(fitted):
            init_val = init_params[name]
            fit_val = fitted[name]
            delta = fit_val - init_val
            ppct = (delta / init_val * 100) if init_val != 0 else float("nan")
            kind = "physical" if name in physical_names else "rule"
            lines.append(f"  {name:<28} [{kind:<8}] {init_val:.4f} -> "
                         f"{fit_val:.4f}  (delta={delta:+.4f}, {ppct:+.1f}%)")

        lines.extend([
            "",
            "Identifiability (posterior_std / prior_std; ~1 means the data "
            "did NOT constrain the parameter — its fitted value is "
            "arbitrary, so remove it from PHYSICAL_PARAMS or collect data "
            "that exercises it):",
            format_identifiability(ident_report),
            "",
        ])
        if kept_at_init:
            lines.append(
                "Applied to the planning base env: fitted values for the "
                "identified params only; "
                f"{', '.join(kept_at_init)} did not contract (or failed "
                "the sensitivity screen), so their baseline values were "
                "kept (the fitted values above for them are arbitrary). "
                "evaluate_plan_refinement now plans against the partially "
                "calibrated sim.")
        else:
            lines.append(
                "The identified physical params were applied to the "
                "planning base env; evaluate_plan_refinement now plans "
                "against the calibrated sim.")
        return _text("\n".join(lines))

    # ── run_python ──────────────────────────────────────────

    run_python = _make_python_exec_tool(
        tool,
        name="run_python",
        description=(
            "Execute Python code for ad-hoc data exploration. Available "
            "variables: trajectories (List[LowLevelTrajectory]; each has "
            "`is_demo`, `train_task_idx`, `states`, `actions`), train_tasks "
            "(List[Task]; each has `init`, `goal`, `goal_holds(state)`), "
            "is_goal_state (callable: state, task_idx -> bool - do the goal "
            "atoms hold in this one STATE; reaching the goal atoms does not "
            "by itself mean solved), np, ParamSpec, and (when the "
            "env defines task evaluators) evaluate_trajectory(states, "
            "actions=None, task_idx=0) -> {reward, solved} - the env's "
            "ground-truth episode scoring over a full TRAJECTORY. "
            "print() output "
            "is returned. The namespace persists across calls. If output "
            "exceeds ~30k chars it is saved to "
            "`tool_outputs/run_python/call_NNNN.txt` in the sandbox and only "
            "a head/tail preview plus that path is returned - use Read/Grep "
            "to inspect the full file. This does NOT define rules - write "
            "`simulator.py` for that; the synthesis tools "
            "(evaluate_step_fit, report_residuals, evaluate_plan_refinement) "
            "load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from that "
            "file."),
        exec_ns=exec_ns,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_text,
    )

    # ── evaluate_step_fit ────────────────────────────────────────

    @tool(
        "evaluate_step_fit",
        "Score the current PROCESS_RULES (loaded fresh from "
        "`simulator.py`) by SSE on the step transitions. Reports SSE "
        "at init_value params from PARAM_SPECS, then fits parameters "
        "and reports the post-fit SSE plus percent improvement and the "
        "fitted parameter values with their delta from init. If the "
        "file declares PHYSICAL_PARAMS (base-sim system "
        "identification), the fit instead matches free-running "
        "base-sim rollouts of full trajectories, fits physical + rule "
        "params jointly, reports per-parameter identifiability, and "
        "applies the identified physical values to the planning base "
        "env. Each call snapshots the simulator file into "
        "simulator_versions/; output is tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type":
                    "string",
                    "description":
                    "Override simulator file path "
                    "(defaults to the canonical simulator.py).",
                },
            },
        },
    )
    async def evaluate_step_fit(args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path") or simulator_file
        rules, specs, declared, latent_init, physical_specs, version_tag, \
            err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_note = ("declared" if isinstance(declared, dict) else
                      "inferred (PROCESS_FEATURES not declared)")

        # PHYSICAL_PARAMS declared -> joint system-identification fit on
        # free-running rollouts (the per-transition/teacher-forced paths
        # below cannot see physical params: State carries no velocities).
        if physical_specs:
            return _evaluate_rollout_fit(rules, specs, physical_specs,
                                         latent_init, process_features,
                                         scope_note, version_tag)

        # Dispatch on the rule signature exactly as the fitting engine
        # does: recurrent (5-arg, latent-declaring) rules are scored with
        # the latent block threaded per trajectory, never through the
        # legacy per-transition path (which would call them with 3 args).
        latent_mode = has_latent_rules(rules)
        init_params = {s.name: s.init_value for s in specs}
        try:
            if latent_mode:
                groups = _groups_for(base_pred_triples)
                pre_sse = compute_sse_recurrent(rules, groups, init_params,
                                                latent_init, process_features)
            else:
                sim_fn = lambda s, _a, p: apply_rules(  # noqa: E731
                    s, rules, p)
                pre_sse = compute_sse(sim_fn, base_pred_triples, init_params,
                                      process_features)
        except Exception as e:  # pylint: disable=broad-except
            return _text(
                f"[{version_tag}] Error: SSE computation failed:\n{e}")

        sig_note = ("recurrent (latent threaded per trajectory)"
                    if latent_mode else "per-transition")
        lines = [
            f"[{version_tag}] Fit evaluation on {len(base_pred_triples)} "
            f"step transitions (scope: {scope_note}; rules: {sig_note}).",
            "",
            f"At init_value params:  SSE = {pre_sse:.6f}",
        ]

        try:
            if latent_mode:
                fit_result, post_sse = (
                    AgentSimLearningApproach._fit_parameters_latent(  # pylint: disable=protected-access
                        rules, specs, groups, latent_init, process_features))
            else:
                fit_result, post_sse = (
                    AgentSimLearningApproach._fit_parameters(  # pylint: disable=protected-access
                        rules, specs, base_pred_triples, process_features))
            fitted_params = fit_result.point_estimate
        except Exception as e:  # pylint: disable=broad-except
            return _text(f"[{version_tag}] Error: fit_params failed:\n{e}")
        if pre_sse > 0:
            pct = (pre_sse - post_sse) / pre_sse * 100
            pct_str = f"({pct:+.1f}% vs init)"
        else:
            pct_str = "(init SSE was 0)"
        lines.append(f"After fit:             SSE = {post_sse:.6f}  "
                     f"{pct_str}")
        lines.append("")
        lines.append("Fitted parameters:")
        for name in sorted(fitted_params):
            init_val = init_params[name]
            fit_val = fitted_params[name]
            delta = fit_val - init_val
            ppct = ((delta / init_val *
                     100) if init_val != 0 else float("nan"))
            lines.append(f"  {name:<30} {init_val:.4f} -> "
                         f"{fit_val:.4f}  (delta={delta:+.4f}, "
                         f"{ppct:+.1f}%)")

        return _text("\n".join(lines))

    # ── report_residuals ────────────────────────────────────

    @tool(
        "report_residuals",
        "Per-feature breakdown of where the current PROCESS_RULES "
        "(loaded fresh from `simulator.py`) disagree with "
        "observations on step transitions. For each feature in "
        "PROCESS_FEATURES (or the inferred fallback) reports mismatch "
        "count, mean abs error, max abs error, and the relative "
        "improvement over the no-rule baseline (negative means rules "
        "are worse than not running them at all). Also lists the "
        "worst-N example transitions per feature so you can see what "
        "edge cases break. Uses init_value from PARAM_SPECS by "
        "default; pass fit_params=true to MCMC-fit first. Tolerance: "
        "|pred - obs| > rel_tol * |obs| + abs_tol. Each call "
        "snapshots the simulator file into simulator_versions/; "
        "output is tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {
                "max_transitions": {
                    "type": "integer",
                    "description": "Max transitions to inspect "
                    "(default 100).",
                },
                "abs_tol": {
                    "type": "number",
                    "description": "Absolute tolerance (default 1e-4).",
                },
                "rel_tol": {
                    "type": "number",
                    "description": "Relative tolerance (default 1e-3).",
                },
                "num_worst_examples": {
                    "type":
                    "integer",
                    "description":
                    "Worst-N mismatched transitions to "
                    "list per feature (default 3, 0 to suppress).",
                },
                "fit_params": {
                    "type":
                    "boolean",
                    "description":
                    "If true, run MCMC fit before "
                    "computing residuals; otherwise use init_value "
                    "(default false).",
                },
                "path": {
                    "type":
                    "string",
                    "description":
                    "Override simulator file path "
                    "(defaults to the canonical simulator.py).",
                },
            },
        },
    )
    async def report_residuals(args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path") or simulator_file
        rules, specs, declared, latent_init, _physical_specs, version_tag, \
            err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_label = ("declared"
                       if isinstance(declared, dict) else "inferred")

        max_n = int(args.get("max_transitions", 100))
        abs_tol = float(args.get("abs_tol", 1e-4))
        rel_tol = float(args.get("rel_tol", 1e-3))
        n_examples = int(args.get("num_worst_examples", 3))
        do_fit = bool(args.get("fit_params", False))

        # Same engine-matching dispatch as evaluate_step_fit: recurrent
        # rules are fit and rolled out with the latent threaded per
        # trajectory, never called per-transition with 3 args.
        latent_mode = has_latent_rules(rules)
        groups = _groups_for(base_pred_triples)
        if do_fit:
            try:
                if latent_mode:
                    fit_result, _ = (
                        AgentSimLearningApproach._fit_parameters_latent(  # pylint: disable=protected-access
                            rules, specs, groups, latent_init,
                            process_features))
                else:
                    fit_result, _ = (
                        AgentSimLearningApproach._fit_parameters(  # pylint: disable=protected-access
                            rules, specs, base_pred_triples, process_features))
                t_params = fit_result.point_estimate
                param_label = "fitted"
            except Exception as e:  # pylint: disable=broad-except
                return _text(
                    f"[{version_tag}] Error: param fitting failed:\n{e}")
        else:
            t_params = {s.name: s.init_value for s in specs}
            param_label = "init_value"

        # Predicted next states, latent threaded per trajectory for
        # recurrent rules (legacy rules roll each transition independently).
        # Roll out all groups in flat order, then truncate to max_n so the
        # reported step indices line up with the flat triples slice below.
        try:
            all_preds = rollout_predictions(rules, t_params, groups,
                                            latent_init)
        except Exception as e:  # pylint: disable=broad-except
            return _text(f"[{version_tag}] Error: rule rollout failed:\n{e}")
        triples_rules: List = all_preds[:max_n]
        triples_base: List = [(bs, sn)
                              for bs, _a, sn in base_pred_triples[:max_n]]

        # Per-feature accumulators keyed by (type_name, feat_name).
        rule_n_total: Dict = defaultdict(int)
        rule_n_mismatch: Dict = defaultdict(int)
        rule_sum_err: Dict = defaultdict(float)
        rule_max_err: Dict = defaultdict(float)
        base_n_total: Dict = defaultdict(int)
        base_sum_err: Dict = defaultdict(float)
        worst: Dict = defaultdict(list)
        mismatched_steps: set = set()

        for i, obj, tn, feat, pred, obs in iter_feature_residuals(
                triples_rules, process_features):
            key = (tn, feat)
            err = abs(pred - obs)
            thr = rel_tol * abs(obs) + abs_tol
            rule_n_total[key] += 1
            rule_sum_err[key] += err
            if err > rule_max_err[key]:
                rule_max_err[key] = err
            if err > thr:
                rule_n_mismatch[key] += 1
                mismatched_steps.add(i)
                worst[key].append((i, obj.name, pred, obs, err))

        for _, _, tn, feat, pred, obs in iter_feature_residuals(
                triples_base, process_features):
            key = (tn, feat)
            base_n_total[key] += 1
            base_sum_err[key] += abs(pred - obs)

        if not rule_n_total:
            return _text(f"[{version_tag}] PROCESS_FEATURES is empty; "
                         "nothing to report.")

        n_steps = len(triples_rules)
        perfect_steps = n_steps - len(mismatched_steps)
        lines = [
            f"[{version_tag}] Residual report — {n_steps} step transitions, "
            f"scope: {scope_label} PROCESS_FEATURES, "
            f"params: {param_label}, "
            f"tol: {rel_tol:g}*|obs| + {abs_tol:g}.",
            f"Steps with all in-scope features within tol: "
            f"{perfect_steps}/{n_steps}.",
            "",
            f"{'feature':<35} {'misses/total':<14} {'mean_err':<10} "
            f"{'max_err':<10} {'vs base':<14}",
        ]
        for key in sorted(rule_n_total):
            tn, feat = key
            n_tot = rule_n_total[key]
            n_mm = rule_n_mismatch[key]
            mean = rule_sum_err[key] / max(1, n_tot)
            mx = rule_max_err[key]
            bn = max(1, base_n_total[key])
            base_mean = base_sum_err[key] / bn
            if base_mean > 0:
                improvement = (base_mean - mean) / base_mean * 100
                vs_base = f"{improvement:+.0f}%"
                if improvement < 0:
                    vs_base += " (worse)"
            elif mean == 0:
                vs_base = "exact"
            else:
                vs_base = "rules add err"
            lines.append(f"{tn + '.' + feat:<35} {f'{n_mm}/{n_tot}':<14} "
                         f"{mean:<10.4f} {mx:<10.4f} {vs_base:<14}")

        if n_examples > 0 and worst:
            lines.append("")
            lines.append(f"Worst {n_examples} mismatches per feature "
                         f"(step N = trajectory transition state[N] -> "
                         f"state[N+1]):")
            for key in sorted(worst):
                tn, feat = key
                entries = sorted(worst[key], key=lambda x: x[4], reverse=True)
                for step, oname, pred, obs, err in entries[:n_examples]:
                    lines.append(f"  step {step:>4}  {oname}.{feat}: "
                                 f"pred={pred:.6f} obs={obs:.6f} "
                                 f"err={err:.6f}")

        return _text("\n".join(lines))

    # ── evaluate_plan_refinement ────────────────────────────

    @tool(
        "evaluate_plan_refinement",
        "MCMC-fit PARAM_SPECS (loaded fresh from `simulator.py`), "
        "build the combined simulator from current PROCESS_RULES + "
        "the fitted params, then run **both** backtracking refinement "
        "and continuous forward validation on a training task against "
        "a plan you propose. Always fits first because refinement "
        "needs to test the simulator at its deployed (fitted) params, "
        "not at init_value. `plan` is required — pass the "
        "option-skeleton you believe should solve the task, one "
        "option call per line, with every option argument supplied "
        "and typed object references (`obj:type`) matching what the "
        "inspect tools report. The parser is strict and will not "
        "auto-fill omitted arguments. Example shape (substitute the "
        "options/types/predicates your task actually exposes): "
        "`PickWidget(robot:robot, widget0:widget)\\nPlace(robot:robot) "
        "-> {WidgetAtFixture(widget0:widget, fixture0:fixture)}\\n...`. "
        "Subgoal annotations (`-> {Atom(obj:type, ...)}`) are "
        "optional in general but effectively required after "
        "open-ended skills like `Place`: without a subgoal the "
        "search has no preference for *where* to put the object, so "
        "a downstream `Wait` may get stuck and look like a rule bug. "
        "For `Wait`, the annotation also specifies when the wait "
        "should terminate; prefix an atom with `NOT` to require it "
        "become false. The `timeout` argument auto-scales with "
        "sketch length when omitted (see the `timeout` field "
        "below). Reports the verdict for refinement (success, "
        "TIMEOUT, SAMPLE_EXHAUSTED with stuck step) and — when "
        "refinement passes — also the verdict for forward validation "
        "(SUCCESS, or FORWARD_VALIDATION_FAILED with the first "
        "subgoal/goal divergence). Refinement may pass while forward "
        "validation fails: refinement resets state between options "
        "and resamples up to 50× per step, while forward validation "
        "runs the same plan once continuously. A refinement-pass "
        "+ forward-validation-fail almost always means a learned "
        "threshold/rule is more permissive than the env's effective "
        "behavior, so refinement believes a subgoal holds when the "
        "env-driven post-state actually doesn't. The agent must "
        "treat forward-validation failure the same as refinement "
        "failure — keep iterating, do not declare done. Each call "
        "snapshots the simulator file into simulator_versions/; "
        "output is tagged [cycle_XXX_vers_YYY]. Slow — use sparingly.",
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Option-skeleton plan text, one "
                    "option call per line. Use typed object "
                    "references (`obj:type`) and supply every "
                    "option argument. Optional `-> {Atom(...)}` "
                    "subgoal after each step; effectively required "
                    "after open-ended skills like `Place`.",
                },
                "task_idx": {
                    "type": "integer",
                    "description": "Index into training tasks "
                    "(default 0).",
                },
                "timeout": {
                    "type":
                    "number",
                    "description":
                    "Refinement timeout in seconds. Omit "
                    "for an auto value that scales with the "
                    "number of steps in the sketch; the actual "
                    "value used is reported back. Override only "
                    "if the previous report said TIMEOUT. MCMC "
                    "fitting runs before refinement and is not "
                    "subject to this timeout.",
                },
                "path": {
                    "type":
                    "string",
                    "description":
                    "Override simulator file path "
                    "(defaults to the canonical simulator.py).",
                },
            },
        },
    )
    async def evaluate_plan_refinement(args: Dict[str, Any]) -> Dict[str, Any]:
        if approach is None:
            return _text("Error: evaluate_plan_refinement is unavailable "
                         "(no approach instance bound to the tool).")

        path = args.get("path") or simulator_file
        rules, specs, declared, latent_init, _physical_specs, version_tag, \
            err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)

        task_idx = int(args.get("task_idx", 0))
        # Treat missing/None timeout as "auto-scale by sketch length"
        # (computed inside run_refinement_for_synthesis from
        # CFG.agent_bilevel_refinement_timeout_per_step / _min).
        timeout_arg = args.get("timeout", None)
        timeout = float(timeout_arg) if timeout_arg is not None else None
        plan_text = args.get("plan", "") or ""

        try:
            report = run_refinement_for_synthesis(
                approach,
                rules=rules,
                specs=specs,
                process_features=process_features,
                base_pred_triples=base_pred_triples,
                task_idx=task_idx,
                timeout=timeout,
                plan_text=plan_text,
                latent_init=latent_init,
            )
        except Exception:  # pylint: disable=broad-except
            tb = _scrub_host_paths(traceback.format_exc())
            return _text(f"[{version_tag}] Error: validation failed:\n{tb}")

        return _text(f"[{version_tag}] {report}")

    return [
        report_residuals,
        run_python,
        evaluate_step_fit,
        evaluate_plan_refinement,
    ]


# ── Predicate-invention tools ─────────────────────────────────────


class _ParamsView:
    """Read-through view onto a fitted-parameters dict.

    Holds the dict directly (not the approach) so predicate classifiers
    that close over this view do not transitively reference the
    approach. The approach must mutate the same dict object in place on
    each re-fit (clear + update) so the view picks up new values
    automatically; replacing the dict would break the live link.
    """

    def __init__(self, params: Dict[str, float]) -> None:
        self._params = params

    def __getitem__(self, key: str) -> float:
        if key not in self._params:
            raise KeyError(
                f"params[{key!r}] accessed before any parameter fit; "
                "call evaluate_step_fit or evaluate_plan_refinement to "
                "populate self._fitted_params first.")
        return self._params[key]

    def __contains__(self, key: object) -> bool:
        return key in self._params

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style fallback lookup; mirrors ``dict.get``."""
        return self._params.get(key, default)

    def __repr__(self) -> str:
        return f"_ParamsView({self._params!r})"


def create_predicate_synthesis_tools(
    predicates_file: str,
    predicates_versions_dir: str,
    approach: Any,
    trajectories: List[LowLevelTrajectory],
    cycle_index_provider: Optional[Callable[[], int]] = None,
) -> list:
    """Create the predicate-invention synthesis tool.

    Returns ``[evaluate_predicate_quality]``. The tool loads
    ``predicates.py`` fresh on each call (snapshotting into
    ``predicates_versions_dir`` as
    ``cycle_XXX_vers_YYY_predicates.py``), validates each
    ``Predicate``, mutates ``approach._learned_predicates`` so
    subsequent refinement calls see the agent's draft, and reports
    milestone behaviour over the demo trajectories.

    Args:
        predicates_file: Host path to the canonical ``predicates.py``
            file the agent edits.
        predicates_versions_dir: Directory for per-call snapshots
            (created on first use).
        approach: The ``AgentSimPredicateInventionApproach`` instance.
            Must expose ``_types``, ``_kept_initial_predicates``,
            ``_get_all_options()``, and ``_learned_predicates``.
        trajectories: Demo trajectories used for milestone reporting.
        cycle_index_provider: Callable returning the current cycle
            (1-indexed) at snapshot time. Defaults to a constant 0.
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported

    from claude_agent_sdk import tool as _sdk_tool
    tool = _make_coercing_tool(_sdk_tool)

    from predicators.code_sim_learning.fit_space import ParamSpec

    # pylint: enable=import-outside-toplevel
    # ``predicates_file`` lives at ``<sandbox>/predicates.py``, so its
    # parent is the sandbox root — spill oversize output there rather than
    # letting the agent SDK dump it outside the sandbox.
    _text = _make_spilling_text_result(os.path.dirname(predicates_file))
    _snapshotter = _ArtifactSnapshotter(
        live_file=predicates_file,
        versions_dir=predicates_versions_dir,
        artifact_name="predicates",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with "
                           "LEARNED_PREDICATES = [...]."),
    )

    params_view = _ParamsView(approach._fitted_params)  # pylint: disable=protected-access

    def _snapshot_and_load_predicates(
        path: str,
    ) -> Tuple[List[Predicate], Optional[str], Optional[str], List[str]]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(predicates, version_tag, error_msg, warnings)``.
        ``error_msg`` is ``None`` on success. Predicates that failed
        validation are excluded; ``warnings`` describes them.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return [], None, err, []
        assert raw is not None and version_tag is not None

        ctx = build_exec_context(
            types=approach._types,  # pylint: disable=protected-access
            predicates=approach._kept_initial_predicates,  # pylint: disable=protected-access
            options=approach._get_all_options(),  # pylint: disable=protected-access
            extra_context={
                "params": params_view,
                "ParamSpec": ParamSpec,
            })
        result, err = exec_code_safely(raw.decode("utf-8"), ctx,
                                       "LEARNED_PREDICATES")
        if err is not None:
            return [], version_tag, (f"[{version_tag}] Error executing "
                                     f"{path}:\n{err}"), []
        if not isinstance(result, list):
            return [], version_tag, (
                f"[{version_tag}] LEARNED_PREDICATES must be a list, "
                f"got {type(result).__name__}."), []

        kept_names = {
            p.name
            for p in approach._kept_initial_predicates  # pylint: disable=protected-access
        }
        example_state = (
            approach._train_tasks[0].init  # pylint: disable=protected-access
            if approach._train_tasks else None)  # pylint: disable=protected-access

        valid: List[Predicate] = []
        warnings: List[str] = []
        seen_names = set()
        for entry in result:
            if not isinstance(entry, Predicate):
                warnings.append(f"Skipped non-Predicate entry: {entry!r}")
                continue
            if entry.name in kept_names:
                warnings.append(f"Skipped '{entry.name}' (collides "
                                "with a kept env predicate).")
                continue
            if entry.name in seen_names:
                warnings.append(f"Skipped duplicate '{entry.name}'.")
                continue
            if example_state is not None:
                verr = validate_predicate(
                    entry,
                    approach._types,  # pylint: disable=protected-access
                    example_state)
                if verr is not None:
                    warnings.append(
                        f"Predicate '{entry.name}' failed validation: "
                        f"{verr}")
                    continue
            valid.append(entry)
            seen_names.add(entry.name)

        # Mutate approach state so evaluate_plan_refinement sees draft.
        approach._learned_predicates = set(valid)  # pylint: disable=protected-access
        return valid, version_tag, None, warnings

    def _enumerate_groundings(
        state: State,
        pred_types: Sequence[Type],
        max_groundings: int,
    ) -> List[Tuple[Any, ...]]:
        """Distinct-object groundings of ``pred_types`` from ``state``.

        Capped at ``max_groundings``; sufficient for milestone
        reporting.
        """
        objs_by_type: Dict[str, List[Any]] = {}
        for obj in state:
            objs_by_type.setdefault(obj.type.name, []).append(obj)

        out: List[Tuple[Any, ...]] = []

        def rec(idx: int, picked: List[Any], used: set) -> None:
            if len(out) >= max_groundings:
                return
            if idx == len(pred_types):
                out.append(tuple(picked))
                return
            for c in objs_by_type.get(pred_types[idx].name, []):
                if id(c) in used:
                    continue
                used.add(id(c))
                picked.append(c)
                rec(idx + 1, picked, used)
                picked.pop()
                used.remove(id(c))
                if len(out) >= max_groundings:
                    return

        rec(0, [], set())
        return out

    @tool(
        "evaluate_predicate_quality",
        "Load LEARNED_PREDICATES (fresh from `predicates.py`) and "
        "report milestone behaviour over demo trajectories. For each "
        "predicate × each grounding, evaluates pred.holds(state) at "
        "every step and reports: coverage (ever-true / ever-false), "
        "transition counts, first-flip step, and monotonicity (ideal "
        "milestone flips False->True exactly once and stays true). "
        "After loading, the predicate set used by "
        "evaluate_plan_refinement is updated — so call this tool any "
        "time you edit predicates.py before re-running refinement. "
        "Snapshots the predicates file into predicates_versions/; "
        "output tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {
                "max_trajectories": {
                    "type": "integer",
                    "description": "Max trajectories to scan "
                    "(default 10).",
                },
                "max_groundings_per_predicate": {
                    "type":
                    "integer",
                    "description":
                    "Max object groundings to evaluate "
                    "per predicate (default 4).",
                },
            },
        },
    )
    async def evaluate_predicate_quality(
            args: Dict[str, Any]) -> Dict[str, Any]:
        max_trajs = int(args.get("max_trajectories", 10))
        max_groundings = int(args.get("max_groundings_per_predicate", 4))

        try:
            preds, version_tag, err, warnings = (
                _snapshot_and_load_predicates(predicates_file))
        except Exception:  # pylint: disable=broad-except
            return _text(f"Error loading predicates.py:\n"
                         f"{_scrub_host_paths(traceback.format_exc())}")

        if err is not None:
            return _text(err)

        prefix = f"[{version_tag}]"
        scanned = trajectories[:max_trajs]
        lines = [
            f"{prefix} Predicate quality report — "
            f"{len(preds)} predicate(s), {len(scanned)} trajector(ies), "
            f"up to {max_groundings} grounding(s)/predicate.",
        ]
        if warnings:
            lines.append("")
            lines.append("Warnings (entries skipped during load):")
            for w in warnings:
                lines.append(f"  - {w}")

        if not preds:
            lines.append("")
            lines.append("LEARNED_PREDICATES is empty — add "
                         "Predicate(...) entries to predicates.py.")
            return _text("\n".join(lines))

        # Pre-materialise per-step `latent` per trajectory. For
        # recurrent approaches this rolls the trajectory through the
        # agent's simulator and produces `[lat_0, lat_1, ...]` so
        # latent-aware predicates can evaluate against a meaningful
        # latent; for non-recurrent approaches it returns a list of
        # ``None``s and latent-aware classifiers see `latent=None`.
        materialise_latent_fn = getattr(approach, "materialise_latent", None)
        latent_per_traj: Dict[int, List[Optional[Dict[str, Any]]]] = {}
        for ti, traj in enumerate(scanned):
            if materialise_latent_fn is not None and traj.states:
                try:
                    latent_per_traj[ti] = materialise_latent_fn(traj)
                except Exception:  # pylint: disable=broad-except
                    # Approach-side materialisation crashed — fall back
                    # to None so observation-only predicates still work.
                    latent_per_traj[ti] = [None] * len(traj.states)
            else:
                latent_per_traj[ti] = [None] * len(traj.states)

        for pred in preds:
            sig = ", ".join(t.name for t in pred.types)
            lines.append("")
            lines.append(f"{pred.name}({sig})")
            ever_true = ever_false = False
            flip_records: List[Tuple[int, Tuple[Any, ...], int, int,
                                     bool]] = []
            no_grounding_trajs = 0
            error_lines: List[str] = []
            for ti, traj in enumerate(scanned):
                if not traj.states:
                    continue
                groundings = _enumerate_groundings(traj.states[0], pred.types,
                                                   max_groundings)
                if not groundings:
                    no_grounding_trajs += 1
                    continue
                lats = latent_per_traj[ti]
                for gr in groundings:
                    try:
                        truth = [
                            pred.holds(s, gr, latent=lats[si])
                            for si, s in enumerate(traj.states)
                        ]
                    except Exception:  # pylint: disable=broad-except
                        last_line = traceback.format_exc().strip().splitlines(
                        )[-1]
                        error_lines.append(
                            f"  traj {ti} ({', '.join(o.name for o in gr)})"
                            f": classifier raised — {last_line}")
                        continue
                    if any(truth):
                        ever_true = True
                    if not all(truth):
                        ever_false = True
                    flips_up = sum(1 for i in range(1, len(truth))
                                   if truth[i] and not truth[i - 1])
                    flips_dn = sum(1 for i in range(1, len(truth))
                                   if truth[i - 1] and not truth[i])
                    flip_records.append(
                        (ti, gr, flips_up, flips_dn, truth[-1]))

            coverage = ("ever-T + ever-F" if ever_true and ever_false else (
                "always-T (likely useless)" if ever_true else
                ("always-F (likely useless)" if ever_false else "no-data")))
            n_records = len(flip_records)
            n_monotone = sum(1 for _, _, up, dn, _ in flip_records
                             if up == 1 and dn == 0)
            n_never_flipped = sum(1 for _, _, up, dn, _ in flip_records
                                  if up == 0 and dn == 0)
            lines.append(f"  coverage: {coverage}")
            lines.append(f"  groundings scored: {n_records}, "
                         f"monotone (1↑ 0↓): {n_monotone}, "
                         f"never-flipped: {n_never_flipped}, "
                         f"no-grounding trajs: {no_grounding_trajs}")
            for ti, gr, up, dn, final in flip_records[:max_trajs]:
                names = ", ".join(o.name for o in gr)
                lines.append(f"  traj {ti} ({names}): ↑={up}, ↓={dn}, "
                             f"final={'T' if final else 'F'}")
            for el in error_lines[:max_trajs]:
                lines.append(el)

        return _text("\n".join(lines))

    return [evaluate_predicate_quality]


def create_sampler_synthesis_tools(
    samplers_file: str,
    samplers_versions_dir: str,
    approach: Any,
    cycle_index_provider: Optional[Callable[[], int]] = None,
) -> list:
    """Create the per-skill sampler-synthesis tool.

    Returns ``[evaluate_sampler]``. On each call the tool loads
    ``samplers.py`` fresh (snapshotting into ``samplers_versions_dir``),
    validates the ``LEARNED_SAMPLERS`` dict (option name -> callable),
    installs it into ``approach._synthesized_samplers`` so refinement
    uses it, and reports a per-option shape/in-box sanity check.

    Args:
        samplers_file: Host path to the agent-edited ``samplers.py``.
        samplers_versions_dir: Directory for per-call snapshots.
        approach: The ``AgentSimLearningApproach`` instance.
        cycle_index_provider: Returns the current 1-indexed cycle.
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported

    from claude_agent_sdk import tool as _sdk_tool
    tool = _make_coercing_tool(_sdk_tool)

    from predicators.code_sim_learning.fit_space import ParamSpec

    # pylint: enable=import-outside-toplevel
    _text = _make_spilling_text_result(os.path.dirname(samplers_file))
    _snapshotter = _ArtifactSnapshotter(
        live_file=samplers_file,
        versions_dir=samplers_versions_dir,
        artifact_name="samplers",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with "
                           "LEARNED_SAMPLERS = {\"OptionName\": fn, ...}."),
    )
    params_view = _ParamsView(approach._fitted_params)  # pylint: disable=protected-access

    def _snapshot_and_load_samplers(
        path: str,
    ) -> Tuple[Dict[str, Any], Optional[str], Optional[str], List[str]]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(samplers, version_tag, error_msg, warnings)``.
        Entries keyed by an unknown option name, or whose value is not
        callable, are skipped and described in ``warnings``. On success,
        mutates ``approach._synthesized_samplers`` to the validated
        dict.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return {}, None, err, []
        assert raw is not None and version_tag is not None

        ctx = build_exec_context(
            types=approach._types,  # pylint: disable=protected-access
            predicates=approach._get_all_predicates(),  # pylint: disable=protected-access
            options=approach._get_all_options(),  # pylint: disable=protected-access
            extra_context={
                "params": params_view,
                "ParamSpec": ParamSpec,
            })
        option_names = {o.name for o in approach._get_all_options()}  # pylint: disable=protected-access
        valid, warnings, err = load_learned_samplers(raw.decode("utf-8"), ctx,
                                                     option_names)
        if err is not None:
            return {}, version_tag, (f"[{version_tag}] Error executing "
                                     f"{path}:\n{err}"), []

        # Mutate approach state so evaluate_plan_refinement / test-time
        # refinement draw from the agent's draft samplers.
        approach._synthesized_samplers = valid  # pylint: disable=protected-access
        return valid, version_tag, None, warnings

    def _sanity_check(name: str, fn: Any) -> str:
        """Draw a few params from a representative state; report shape/box."""
        # pylint: disable=protected-access,import-outside-toplevel
        import numpy as np  # pylint: disable=redefined-outer-name,reimported

        from predicators.settings import \
            CFG  # pylint: disable=redefined-outer-name,reimported
        options_by_name = {o.name: o for o in approach._get_all_options()}
        opt = options_by_name[name]
        train_tasks = approach._train_tasks
        if not train_tasks:
            return f"  {name}: no train task to sanity-check against."
        state = train_tasks[0].init
        # Pick the first object of each option-arg type present in the state.
        objs: List[Object] = []
        for t in opt.types:
            match = next((o for o in state if o.type.name == t.name), None)
            if match is None:
                return (f"  {name}: no object of type '{t.name}' in the "
                        "train-task state to sanity-check against.")
            objs.append(match)
        box = opt.params_space
        expected = box.shape[0]
        rng = np.random.default_rng(CFG.seed)
        in_box = 0
        n_draws = 3
        for _ in range(n_draws):
            try:
                raw = fn(state, set(), rng, objs)
                arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            except Exception:  # pylint: disable=broad-except
                last = traceback.format_exc().strip().splitlines()[-1]
                return (f"  {name}: ERROR — sampler raised: {last} "
                        "(note: this check, and refinement at steps with "
                        "no subgoal annotation, call the sampler with "
                        "subgoal_atoms=set(); it must not crash on an "
                        "empty set — fall back to a default or uniform "
                        "draw).")
            if arr.shape != (expected, ):
                return (f"  {name}: ERROR — returned shape {arr.shape}, "
                        f"expected ({expected},).")
            if bool(np.all(arr >= box.low - 1e-6)) and \
                    bool(np.all(arr <= box.high + 1e-6)):
                in_box += 1
        return (f"  {name}: OK — {n_draws} draws, {in_box}/{n_draws} "
                f"within the params box.")

    @tool(
        "evaluate_sampler",
        "Load LEARNED_SAMPLERS (fresh from `samplers.py`) and install "
        "them as the per-skill samplers used by refinement. Each entry "
        "maps an option name to a function "
        "(state, subgoal_atoms, rng, objects) -> params array (the same "
        "signature as the env's NSRT samplers); refinement calls it "
        "instead of drawing uniformly so the sampler can aim continuous "
        "params at the step's subgoal, then clips the result to the box. "
        "At steps with no subgoal annotation, subgoal_atoms is the empty "
        "set - the sampler must handle that without crashing. A sketch "
        "step carrying a `~ [widths]` region annotation bypasses the "
        "sampler (precedence: per-step region > per-skill sampler > "
        "uniform); samplers are the reusable cross-task prior, regions a "
        "per-call override. "
        "Reports a per-option sanity check (return shape + within-box) "
        "over a representative train-task state. After loading, the "
        "samplers used by evaluate_plan_refinement are updated — so call "
        "this any time you edit samplers.py before re-running "
        "refinement. Snapshots samplers.py into samplers_versions/; "
        "output tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {},
        },
    )
    async def evaluate_sampler(args: Dict[str, Any]) -> Dict[str, Any]:
        del args
        try:
            samplers, version_tag, err, warnings = (
                _snapshot_and_load_samplers(samplers_file))
        except Exception:  # pylint: disable=broad-except
            return _text(f"Error loading samplers.py:\n"
                         f"{_scrub_host_paths(traceback.format_exc())}")

        if err is not None:
            return _text(err)

        prefix = f"[{version_tag}]"
        lines = [
            f"{prefix} Sampler report — {len(samplers)} per-skill "
            f"sampler(s) installed.",
        ]
        if warnings:
            lines.append("")
            lines.append("Warnings (entries skipped during load):")
            for w in warnings:
                lines.append(f"  - {w}")

        if not samplers:
            lines.append("")
            lines.append("LEARNED_SAMPLERS is empty — add "
                         "{\"OptionName\": fn} entries to samplers.py.")
            return _text("\n".join(lines))

        lines.append("")
        lines.append("Sanity check (representative train-task state):")
        for name in sorted(samplers):
            lines.append(_sanity_check(name, samplers[name]))
        lines.append("")
        lines.append("Now call evaluate_plan_refinement with a sketch that "
                     "uses these options to measure the samples-to-refine "
                     "improvement.")
        return _text("\n".join(lines))

    return [evaluate_sampler]
