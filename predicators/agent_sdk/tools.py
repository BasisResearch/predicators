"""Custom MCP tool definitions for the agent SDK approach."""
import hashlib
import json
import logging
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.agent_sdk.proposal_parser import ProposalBundle, \
    build_exec_context, exec_code_safely, validate_predicate
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, LowLevelTrajectory, Object, \
    OptionSampler, ParameterizedOption, Predicate, State, Task, Type

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
ALL_TOOL_NAMES = (INSPECTION_TOOL_NAMES + PROPOSAL_TOOL_NAMES +
                  RETRACTION_TOOL_NAMES + TESTING_TOOL_NAMES +
                  PLANNING_TOOL_NAMES + SCENE_TOOL_NAMES)

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
    # Active-experiment info-gain scorer, synced from the learning
    # approach when info-seeking exploration is on:
    # ``(state, atoms) -> disagreement``. The agent_bilevel explorer
    # passes it into refinement so continuous-parameter search prefers
    # candidates that straddle the learned model's decision boundaries.
    # None ⇒ plain feasibility search (default).
    atom_disagreement_fn: Optional[Callable[[State, Any], float]] = None
    # Synthesized per-skill samplers (option name -> sampler), synced from
    # the learning approach when agent_sim_learn_synthesize_samplers is on.
    # The agent_bilevel explorer and synthesis tools pass these into
    # refinement so continuous-parameter search aims at each step's subgoal
    # instead of drawing uniformly. Empty ⇒ uniform sampling (default).
    option_samplers: Dict[str, OptionSampler] = field(default_factory=dict)
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
    # Set by refine_plan_sketch / evaluate_option_plan when a plan is verified
    # to reach the goal on the CURRENT solve task: the simulator-verified plan
    # (grounded options with found params) and the parallel subgoal sketch.
    # The bilevel approach returns this directly instead of re-refining, so
    # the agent's tool-validated answer is exactly what gets executed. None ⇒
    # nothing captured this query.
    solved_plan: Optional[Any] = None
    solved_sketch: Optional[Any] = None
    # Gate for the above: only approaches that consume captured plans
    # (AgentBilevelApproach) set this True. Keeps the open-loop planner, which
    # also uses evaluate_option_plan, from recording spurious captures.
    capture_goal_reaching_plans: bool = False


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
_SANDBOX_SYSTEM_ROOTS = (
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
_SANDBOX_INTROSPECTION = ("getsource", "inspect.getfile", "site-packages")
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
    logic for Bash lives in ``sandbox_prompts.VALIDATE_SANDBOX_SCRIPT``;
    keep the two in sync.
    """
    for needle in _SANDBOX_INTROSPECTION:
        if needle in text:
            return (f"'{needle}' may read predicators source outside the "
                    "sandbox; use the MCP tools and ./reference/ files")
    sandbox = os.path.realpath(sandbox_dir)
    for match in _SANDBOX_PATH_RE.finditer(text):
        token = match.group(1)
        resolved = os.path.realpath(
            token if os.path.isabs(token) else os.path.join(sandbox, token))
        if resolved == sandbox or resolved.startswith(sandbox + os.sep):
            continue
        if token.startswith("/") and not any(
                token == root or token.startswith(root + "/")
                for root in _SANDBOX_SYSTEM_ROOTS):
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
            filename = (f"iter{ctx.iteration_id:03d}"
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
                state_dict = {}
                for obj in sorted(state, key=str):
                    obj_feats = {}
                    for feat in obj.type.feature_names:
                        val = state.get(obj, feat)
                        obj_feats[feat] = round(float(val), 4) \
                            if isinstance(val, (float, int)) else str(val)
                    state_dict[str(obj)] = obj_feats
                lines.append(f"State: {json.dumps(state_dict, indent=2)}")
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
                return _error_result(f"augment_task failed on test task:\n"
                                     f"{traceback.format_exc()}")
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

    @tool(
        "evaluate_option_plan",
        "Execute a fully-specified plan on a task via the option model and "
        "report the result at each step. `plan` is text — one option per "
        "line, same grammar as refine_plan_sketch: "
        "`Option(obj1:type1, obj2:type2)[param1, param2] -> {Atom(obj:type), "
        "...}` (typed object refs; EXACT continuous params in `[]`, `[]` for "
        "none; optional `-> {atoms}` subgoals, prefix NOT to require false). "
        "Runs your exact params with NO sampling. Use include_states/"
        "include_atoms to control output. If the plan reaches the goal on the "
        "CURRENT task (omit task_idx), it is captured as your answer, and the "
        "per-step subgoals make it execute closed-loop (monitored, with "
        "replan-on-divergence).",
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
        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        opt_map = {o.name: o for o in all_options}

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
            sketch_steps = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=True)
        except Exception as e:  # pylint: disable=broad-except
            return _error_result(f"Could not parse plan: {e}")
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

        # Per-step report callback, driven by the shared forward executor.
        def _report_step(i: int, outcome: Any) -> None:
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
                state_dict = {}
                for obj in sorted(post, key=str):
                    obj_feats = {}
                    for feat in obj.type.feature_names:
                        val = post.get(obj, feat)
                        obj_feats[feat] = round(float(val), 4) \
                            if isinstance(val, (float, int)) else str(val)
                    state_dict[str(obj)] = obj_feats
                step_line += f"\n  State: {json.dumps(state_dict, indent=4)}"
            lines.append(step_line)
            img_block = _render_scene_image(ctx, f"step_{i}_{opt.name}")
            if img_block and img_block.get("saved_path"):
                saved_image_paths.append(img_block["saved_path"])

        # Execute exactly like the real closed-loop executor: abort at the
        # first failing option (0-action collision / not-initiable / env
        # failure) instead of pressing on. Otherwise forward simulation can
        # continue past a collision and report a goal that the real rollout —
        # which ends the episode at that failed option — never reaches.
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
        # Capture a goal-reaching plan on the current task with a sketch that
        # keeps only the subgoals that actually held (so the closed-loop
        # monitor won't flag a spurious divergence on a wrong annotation).
        if (ctx.capture_goal_reaching_plans and task_idx == "current"
                and goal_achieved and grounded_plan):
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
            n_annot = sum(1 for s in captured_sketch
                          if s.subgoal_atoms or s.subgoal_neg_atoms)
            lines.append(
                f"Captured as the current answer: {len(grounded_plan)} steps, "
                f"{n_annot} with subgoal annotations for closed-loop "
                "monitoring.")
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
        # Goal atoms hold but the plan needs more low-level steps than the
        # episode horizon allows: say so and that it was NOT captured, so the
        # agent shortens the plan instead of stopping on a false positive.
        if goal_reached and not within_horizon:
            lines.append(
                f"NOT EXECUTABLE (plan was NOT captured): reaching the goal "
                f"takes {result.actions_to_goal} low-level steps but the "
                f"episode horizon is {horizon}. The real executor will run "
                f"out of steps — shorten the plan (fewer or quicker steps) "
                f"before resubmitting.")
        if not goal_reached and not task.goal_nl:
            missing = task.goal - final_atoms
            missing_str = ", ".join(str(a) for a in sorted(missing))
            lines.append(f"Missing goal atoms: {{{missing_str}}}")

        # Append image save paths to text output
        if saved_image_paths:
            lines.append("\nSaved images:")
            for p in saved_image_paths:
                lines.append(f"  {p}")

        # Build result with text only (images are saved to disk)
        return _text_result("\n".join(lines))

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
        from predicators.approaches import ApproachFailure, ApproachTimeout
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
        except (ApproachFailure, ApproachTimeout, Exception) as e:  # pylint: disable=broad-except
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
        from predicators.approaches import ApproachFailure, ApproachTimeout
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
        except (ApproachFailure, ApproachTimeout, Exception) as e:  # pylint: disable=broad-except
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

    @tool(
        "refine_plan_sketch",
        "FIND continuous parameters for a plan SKETCH: run a backtracking "
        "search over the option model, then — on success — forward-validate "
        "the refined plan. Unlike evaluate_option_plan (which runs your EXACT "
        "params with no search), this takes a sketch and lets the search find "
        "params. You may seed it by appending `[p1, p2]` per step (use `[]` "
        "for none); the search tries them first, then samples. `plan` is one "
        "option call per line with typed object references (`obj:type`) and "
        "every argument supplied; add `-> {Atom(obj:type, ...)}` subgoal "
        "annotations (effectively required after open-ended skills like Place, "
        "and for Wait to say when it should end — prefix an atom with NOT to "
        "require it become false). On SUCCESS it reports the exact PARAMETERS "
        "it found per step — submit those via evaluate_option_plan, which is "
        "the delivery path; refine_plan_sketch itself does NOT submit. Also "
        "reports the verdict (SUCCESS / TIMEOUT / SAMPLE_EXHAUSTED with the "
        "stuck step / FORWARD_VALIDATION_FAILED) and time used. Requires a "
        "simulator (option model). Slower than evaluate_option_plan — use it "
        "to find params for hard steps, not to submit.",
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
                    "(`[]` for none) when param-proposing is enabled.",
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
            sketch = bilevel_sketch.parse_sketch_from_text(
                plan_text,
                task,
                predicates=all_predicates,
                options=all_options,
                types=types,
                parse_continuous_params=CFG.
                agent_bilevel_use_llm_initial_params,
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

        try:
            success, report, plan = bilevel_sketch.refine_and_validate_report(
                task,
                sketch,
                ctx.option_model,
                predicates=all_predicates,
                timeout=timeout,
                rng=np.random.default_rng(CFG.seed),
                max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
                check_subgoals=CFG.agent_bilevel_check_subgoals,
                log_state=CFG.agent_bilevel_log_state,
                option_samplers=ctx.option_samplers or None,
                run_id="planner_refine",
                timeout_source=timeout_source,
            )
        except Exception:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            return _error_result(f"Refinement raised:\n{tb}")

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
                                "description": "[x, y, z] for marker or text"
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
                            "text": {
                                "type": "string",
                                "description": "Label text (for type=text)"
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
                            "label": {
                                "type":
                                "string",
                                "description":
                                "Optional text label near the "
                                "annotation"
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

        # Copy the initial state
        modified_state = task.init.copy()
        obj_lookup = {o.name: o for o in modified_state}

        summaries: List[str] = []
        for mod in modifications:
            obj_name = mod.get("object", "")
            features = mod.get("features", {})
            if obj_name not in obj_lookup:
                available = sorted(obj_lookup.keys())
                return _error_result(f"Unknown object '{obj_name}'. "
                                     f"Available: {available}")
            obj = obj_lookup[obj_name]
            for feat_name, value in features.items():
                try:
                    modified_state.set(obj, feat_name, value)
                    summaries.append(f"  {obj_name}.{feat_name} = {value}")
                except Exception as e:  # pylint: disable=broad-except
                    return _error_result(
                        f"Failed to set {obj_name}.{feat_name}: {e}")

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
        tool  # pylint: disable=import-outside-toplevel

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
    import io
    import sys
    import traceback  # pylint: disable=redefined-outer-name,reimported
    from collections import defaultdict

    from claude_agent_sdk import tool

    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    from predicators.code_sim_learning.physical_sysid import \
        compute_rollout_sse, fit_params_rollout_trimmed, \
        format_identifiability, identifiability_report, \
        select_trustworthy_params
    from predicators.code_sim_learning.synthesis_validation import \
        run_refinement_for_synthesis
    from predicators.code_sim_learning.training import ParamSpec, \
        compute_sse, compute_sse_recurrent
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
    _run_python_count = [0]

    # Threshold above which run_python output is spilled to a file in the
    # sandbox rather than returned inline. Kept well under the agent SDK's
    # MCP tool-result token cap so the harness never has to truncate and
    # dump to ``~/.claude/projects/.../tool-results/``.
    _run_python_inline_char_limit = 30000
    _run_python_preview_head_lines = 30
    _run_python_preview_tail_lines = 30

    # Where oversize ``run_python`` outputs are written. The agent reads
    # these back via ``Read``/``Grep`` using ``sandbox_dir_for_agent`` as
    # the path prefix (e.g. ``./tool_outputs/run_python/...`` for local
    # sandbox, ``/sandbox/tool_outputs/run_python/...`` for docker, or an
    # absolute host path otherwise).
    _run_python_outputs_subdir = os.path.join("tool_outputs", "run_python")
    _run_python_outputs_dir_host: Optional[str] = (os.path.join(
        sandbox_dir, _run_python_outputs_subdir) if sandbox_dir else None)
    if sandbox_dir_for_agent:
        _run_python_outputs_dir_agent: Optional[str] = (
            f"{sandbox_dir_for_agent.rstrip('/')}/"
            f"{_run_python_outputs_subdir.replace(os.sep, '/')}")
    elif _run_python_outputs_dir_host:
        _run_python_outputs_dir_agent = _run_python_outputs_dir_host
    else:
        _run_python_outputs_dir_agent = None

    # Spill oversize output from the synthesis tools into the sandbox too,
    # so nothing is dumped to ``~/.claude/projects/.../tool-results/``.
    # ``run_python`` keeps its own bespoke spill below (with a tailored
    # "narrow your print()" hint); this covers the remaining tools.
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
                f"{traceback.format_exc()}")
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
        try:
            pre_sse = compute_rollout_sse(fit_env, rollouts, init_params,
                                          process_features, physical_names,
                                          rules, latent_init)
            fit_result, survivors, traj_rms = fit_params_rollout_trimmed(
                fit_env,
                rollouts,
                physical_specs,
                process_features,
                rules=rules,
                rule_specs=rule_specs,
                latent_init=latent_init)
            # Post-SSE and the identifiability probe run on the SURVIVING
            # trajectories only — a trimmed (unexplainable) recording
            # would re-poison the verdicts with its noise. With no
            # survivors, keep the full set so the probe reports NOT
            # identified and the trustworthiness guard keeps the inits.
            probe_rollouts = survivors if survivors else rollouts
            fitted = fit_result.point_estimate
            post_sse = compute_rollout_sse(fit_env, probe_rollouts, fitted,
                                           process_features, physical_names,
                                           rules, latent_init)
        except Exception as e:  # pylint: disable=broad-except
            return _text(
                f"[{version_tag}] Error: rollout system-ID fit failed:\n{e}")

        def rollout_sse_fn(params: Dict[str, float]) -> float:
            return compute_rollout_sse(fit_env, probe_rollouts, params,
                                       process_features, physical_names, rules,
                                       latent_init)

        ident_report = identifiability_report(
            fit_result, rollout_sse_fn,
            list(physical_specs) + list(rule_specs))
        applied = select_trustworthy_params(fitted, init_params,
                                            physical_names, ident_report)
        approach._apply_identified_physical_params(applied)  # pylint: disable=protected-access
        kept_at_init = sorted(n for n in physical_names
                              if applied[n] != fitted[n])
        if pre_sse > 0:
            pct_str = f"({(pre_sse - post_sse) / pre_sse * 100:+.1f}% vs init)"
        else:
            pct_str = "(init SSE was 0)"
        lines = [
            f"[{version_tag}] JOINT ROLLOUT SYSTEM-ID FIT (PHYSICAL_PARAMS "
            f"declared) on {len(rollouts)} full trajectories (scope: "
            f"{scope_note}; {len(physical_names)} physical + "
            f"{len(list(rule_specs))} rule params).",
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
                f" of {len(rollouts)} trajectories were unexplainable at ANY "
                "candidate params (per-trajectory best-achievable RMS: "
                f"[{rms_str}]) and were dropped before fitting; the fit "
                "below used only the explainable ones. If NONE survived, no "
                "fit ran and the declared inits were kept — collect cleaner "
                "interaction data (e.g. pushes near the top of the domino, "
                "which topple cleanly, rather than center-height pushes "
                "that shove chaotically).")
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
                f"{', '.join(kept_at_init)} did not contract, so the "
                "declared init(s) were kept (the fitted values above for "
                "them are arbitrary). evaluate_plan_refinement now plans "
                "against the partially calibrated sim.")
        else:
            lines.append(
                "The identified physical params were applied to the "
                "planning base env; evaluate_plan_refinement now plans "
                "against the calibrated sim.")
        return _text("\n".join(lines))

    # ── run_python ──────────────────────────────────────────

    @tool(
        "run_python",
        "Execute Python code for ad-hoc data exploration. Available "
        "variables: trajectories (List[LowLevelTrajectory]; each has "
        "`is_demo`, `train_task_idx`, `states`, `actions`), train_tasks "
        "(List[Task]; each has `init`, `goal`, `goal_holds(state)`), "
        "is_goal_state (callable: state, task_idx -> bool — a "
        "ground-truth black-box reward), np, ParamSpec. print() output "
        "is returned. The namespace persists across calls. If output "
        "exceeds ~30k chars it is saved to "
        "`tool_outputs/run_python/call_NNNN.txt` in the sandbox and only "
        "a head/tail preview plus that path is returned — use Read/Grep "
        "to inspect the full file. This does NOT define rules — write "
        "`simulator.py` for that; the synthesis tools "
        "(evaluate_step_fit, report_residuals, evaluate_plan_refinement) "
        "load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from that "
        "file.",
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
    async def run_python(args: Dict[str, Any]) -> Dict[str, Any]:
        code = args["code"]
        # run_python execs in-process with full filesystem access, and the
        # sandbox's PreToolUse file-path hook does not cover MCP tools, so
        # screen the code here for out-of-sandbox reads / source
        # introspection before executing (best-effort; see
        # _screen_text_for_sandbox_escape).
        if sandbox_dir is not None:
            reason = _screen_text_for_sandbox_escape(code, sandbox_dir)
            if reason is not None:
                return _text(
                    f"Error: sandbox guard blocked this code — {reason}. "
                    "Read files with Read/Grep and use the MCP tools and "
                    "./reference/ files instead.")
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        try:
            exec(code, exec_ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            return _text(f"Error:\n{tb}")
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        if not output:
            return _text("(no output)")

        if (len(output) <= _run_python_inline_char_limit
                or _run_python_outputs_dir_host is None):
            return _text(output)

        _run_python_count[0] += 1
        os.makedirs(_run_python_outputs_dir_host, exist_ok=True)
        filename = f"call_{_run_python_count[0]:04d}.txt"
        host_path = os.path.join(_run_python_outputs_dir_host, filename)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(output)

        lines = output.splitlines()
        total_lines = len(lines)
        head = lines[:_run_python_preview_head_lines]
        tail = (lines[-_run_python_preview_tail_lines:] if total_lines >
                (_run_python_preview_head_lines +
                 _run_python_preview_tail_lines) else [])
        agent_path = (f"{_run_python_outputs_dir_agent}/{filename}"
                      if _run_python_outputs_dir_agent else host_path)
        preview_parts = [
            f"[run_python output too large to inline: "
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
        return _text("\n".join(preview_parts))

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
            tb = traceback.format_exc()
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

    from claude_agent_sdk import tool

    from predicators.code_sim_learning.training import ParamSpec

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
            return _text(
                f"Error loading predicates.py:\n{traceback.format_exc()}")

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

    from claude_agent_sdk import tool

    from predicators.code_sim_learning.training import ParamSpec

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
        result, err = exec_code_safely(raw.decode("utf-8"), ctx,
                                       "LEARNED_SAMPLERS")
        if err is not None:
            return {}, version_tag, (f"[{version_tag}] Error executing "
                                     f"{path}:\n{err}"), []
        if not isinstance(result, dict):
            return {}, version_tag, (
                f"[{version_tag}] LEARNED_SAMPLERS must be a dict "
                f"{{option_name: sampler_fn}}, got "
                f"{type(result).__name__}."), []

        option_names = {o.name for o in approach._get_all_options()}  # pylint: disable=protected-access
        valid: Dict[str, Any] = {}
        warnings: List[str] = []
        for name, fn in result.items():
            if name not in option_names:
                warnings.append(
                    f"Skipped '{name}' (not a known option name; known: "
                    f"{', '.join(sorted(option_names))}).")
                continue
            if not callable(fn):
                warnings.append(
                    f"Skipped '{name}' (value is not callable, got "
                    f"{type(fn).__name__}).")
                continue
            valid[name] = fn

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
                return f"  {name}: ERROR — sampler raised: {last}"
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
            return _text(
                f"Error loading samplers.py:\n{traceback.format_exc()}")

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
