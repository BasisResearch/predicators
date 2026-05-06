"""Custom MCP tool definitions for the agent SDK approach."""
import hashlib
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators.agent_sdk.proposal_parser import ProposalBundle, \
    build_exec_context, exec_code_safely, validate_predicate
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type

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
    "propose_object_augmentor",
    "propose_processes",
    "propose_options",
]
RETRACTION_TOOL_NAMES = [
    "retract_abstractions",
]
TESTING_TOOL_NAMES = [
    "test_predicate_on_states",
    "test_planning",
    "test_option_plan",
]
PLANNING_TOOL_NAMES = [
    "generate_bilevel_plan",
    "generate_abstract_plan",
]
SCENE_TOOL_NAMES = [
    "annotate_scene",
    "visualize_state",
]
ALL_TOOL_NAMES = (INSPECTION_TOOL_NAMES + PROPOSAL_TOOL_NAMES +
                  RETRACTION_TOOL_NAMES + TESTING_TOOL_NAMES +
                  PLANNING_TOOL_NAMES + SCENE_TOOL_NAMES)


def get_allowed_tool_list(
    tool_names: Optional[List[str]] = None,
    extra_names: Optional[List[str]] = None,
) -> List[str]:
    """Compute the allowed_tools list for the agent SDK.

    Args:
        tool_names: If provided, only include these tool names.
            If None, include all tools.
    """
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    names = ALL_TOOL_NAMES if tool_names is None else \
        [n for n in tool_names if n in set(ALL_TOOL_NAMES)]
    if extra_names:
        names = list(names) + list(extra_names)
    return [f"{prefix}{n}" for n in names]


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
    test_call_id: int = 0  # incremented per test_option_plan call
    visualized_state: Optional[State] = None  # last state from visualize_state
    extra_mcp_tools: list = field(default_factory=list)  # injected by subclass
    # Populated by AgentBilevelExplorer so learning approaches can diff
    # mental-model subgoals against real trajectories.
    # TODO(sim-learning): consume these in learn_from_interaction_results.
    last_sketch_subgoals: Optional[Any] = None
    last_sketch_options: Optional[Any] = None


def _text_result(text: str) -> Dict[str, Any]:
    """Helper to format a successful text result."""
    return {"content": [{"type": "text", "text": text}]}


def _error_result(text: str) -> Dict[str, Any]:
    """Helper to format an error result."""
    return {"content": [{"type": "text", "text": text}], "is_error": True}


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

    # ===== INSPECTION TOOLS =====

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
        lines = [
            f"Trajectory {traj_idx}: {len(traj.states)} states, "
            f"{len(traj.actions)} actions"
        ]

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
            goal_str = ", ".join(str(g) for g in sorted(task.goal))
            init_atoms = utils.abstract(task.init, ctx.predicates)
            atoms_str = ", ".join(str(a) for a in sorted(init_atoms))
            objects = sorted(task.init, key=str)
            obj_str = ", ".join(f"{o.name}:{o.type.name}" for o in objects)
            state_str = task.init.pretty_str()
            text = (f"Task {task_idx}:\n"
                    f"  Goal: {{{goal_str}}}\n"
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

    # ===== PROPOSAL TOOLS =====

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
        "propose_object_augmentor",
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
    async def propose_object_augmentor(args: Dict[str, Any]) -> Dict[str, Any]:
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
        _save_proposal_code("propose_object_augmentor", code, obj_names,
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

    # ===== RETRACTION TOOLS =====

    @tool(
        "retract_abstractions",
        "Remove previously proposed abstractions that are no longer needed. "
        "Specify names of predicates, processes, options, or helper types to "
        "remove, and/or set clear_object_augmentor to remove the augmentor.",
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
                "clear_object_augmentor": {
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
        clear_augmentor = bool(args.get("clear_object_augmentor", False))

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
            ctx.iteration_proposals.retract_object_augmentor = True
            lines.append("Object augmentor will be cleared.")

        logging.info(f"Agent retraction request: {args}")
        return _text_result("\n".join(lines))

    # ===== TESTING TOOLS =====

    @tool(
        "test_predicate_on_states",
        "Test a predicate's truth value across timesteps in a trajectory",
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
    async def test_predicate_on_states(args: Dict[str, Any]) -> Dict[str, Any]:
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
        "test_planning",
        "Run the task planner on a specific task and report results",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type": "integer",
                    "description": "Task index to plan for"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
            "required": ["task_idx"],
        },
    )
    async def test_planning(args: Dict[str, Any]) -> Dict[str, Any]:
        # pylint: disable=import-outside-toplevel
        from predicators.approaches import ApproachFailure, ApproachTimeout
        from predicators.planning_with_processes import \
            run_task_plan_with_processes_once

        task_idx = args["task_idx"]
        timeout = args.get("timeout", 30)

        if task_idx < 0 or task_idx >= len(ctx.train_tasks):
            return _error_result(f"Invalid task_idx {task_idx}. "
                                 f"Available: 0-{len(ctx.train_tasks)-1}")

        task = ctx.train_tasks[task_idx]
        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates

        try:
            plan, _atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                ctx.processes | ctx.iteration_proposals.proposed_processes,
                all_preds,
                ctx.types | ctx.iteration_proposals.proposed_types,
                timeout,
                seed=CFG.seed,
                _task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
            plan_desc = " -> ".join(p.name for p in plan)
            return _text_result(
                f"Planning succeeded for task {task_idx}!\n"
                f"Plan length: {len(plan)}\n"
                f"Nodes expanded: {metrics.get('num_nodes_expanded', '?')}\n"
                f"Plan: {plan_desc}")
        except (ApproachFailure, ApproachTimeout, Exception) as e:  # pylint: disable=broad-except
            return _text_result(f"Planning failed for task {task_idx}.\n"
                                f"Reason: {type(e).__name__}: {e}")

    @tool(
        "test_option_plan",
        "Execute a sequence of grounded options on a task via the option model "
        "and report the result at each step. Use include_states and/or "
        "include_atoms to control what is shown at each step.",
        {
            "type": "object",
            "properties": {
                "option_plan": {
                    "type": "array",
                    "description": "Ordered list of options to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option_name": {
                                "type": "string",
                                "description":
                                "Name of the ParameterizedOption"
                            },
                            "object_names": {
                                "type":
                                "array",
                                "items": {
                                    "type": "string"
                                },
                                "description":
                                "Object names to ground the option on"
                            },
                            "params": {
                                "type":
                                "array",
                                "items": {
                                    "type": "number"
                                },
                                "description":
                                "Continuous parameters (empty list if none)"
                            },
                        },
                        "required": ["option_name", "object_names", "params"],
                    },
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
                "save_low_level_action_images": {
                    "type":
                    "boolean",
                    "description":
                    "Save per low-level action images (one per "
                    "simulator step) to a separate directory. "
                    "Useful for debugging fine-grained behavior.",
                    "default":
                    False
                },
            },
            "required": ["option_plan"],
        },
    )
    async def test_option_plan(args: Dict[str, Any]) -> Dict[str, Any]:
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
        option_plan_spec = args["option_plan"]
        include_states = args.get("include_states", False)
        include_atoms = args.get("include_atoms", True)
        save_low_level_action_images = args.get("save_low_level_action_images",
                                                False)

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

        state = task.init
        lines = [f"Testing option plan on task {task_idx}:"]
        saved_image_paths: List[str] = []

        for step_idx, opt_spec in enumerate(option_plan_spec):
            opt_name = opt_spec["option_name"]
            obj_names = opt_spec["object_names"]
            params = opt_spec["params"]

            if opt_name not in opt_map:
                return _error_result(f"Unknown option '{opt_name}'. "
                                     f"Available: {sorted(opt_map.keys())}")

            param_opt = opt_map[opt_name]

            obj_name_to_obj = {o.name: o for o in state}
            objects = []
            for name in obj_names:
                if name not in obj_name_to_obj:
                    return _error_result(
                        f"Object '{name}' not found in state at step "
                        f"{step_idx}. Available: "
                        f"{sorted(obj_name_to_obj.keys())}")
                objects.append(obj_name_to_obj[name])

            try:
                params_arr = np.array(params, dtype=np.float32)
                option = param_opt.ground(objects, params_arr)
            except Exception as e:  # pylint: disable=broad-except
                return _error_result(
                    f"Failed to ground option '{opt_name}' at step "
                    f"{step_idx}: {e}")

            if not option.initiable(state):
                atoms = utils.abstract(state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Step {step_idx}: {opt_name}({obj_names}) - "
                             f"NOT INITIABLE\n"
                             f"  Current atoms: {{{atoms_str}}}\n"
                             f"  Object poses at failure:\n"
                             f"{_format_object_poses(state)}")
                return _text_result("\n".join(lines) +
                                    "\n\nPlan FAILED: option not initiable.")

            try:
                next_state, num_actions = \
                    ctx.option_model.get_next_state_and_num_actions(
                        state, option)
            except Exception as e:  # pylint: disable=broad-except
                tb = traceback.format_exc()
                lines.append(f"Step {step_idx}: {opt_name}({obj_names}) - "
                             f"EXECUTION ERROR: {type(e).__name__}: {e}\n"
                             f"  Traceback:\n{tb}")
                return _text_result("\n".join(lines) +
                                    "\n\nPlan FAILED: execution error.")

            step_line = (f"Step {step_idx}: {opt_name}({obj_names}) "
                         f"({num_actions} actions)")
            if num_actions == 0:
                failure = getattr(ctx.option_model, 'last_execution_failure',
                                  None)
                if failure:
                    step_line += f"\n  FAILURE REASON: {failure}"
                else:
                    step_line += ("\n  FAILURE REASON: Option terminated "
                                  "immediately (terminal condition was True "
                                  "before any action was taken)")
                step_line += ("\n  Object poses at failure:\n"
                              f"{_format_object_poses(state)}")
            if include_atoms:
                atoms_before = utils.abstract(state, ctx.predicates)
                atoms_after = utils.abstract(next_state, ctx.predicates)
                added = atoms_after - atoms_before
                deleted = atoms_before - atoms_after
                added_s = ", ".join(str(a) for a in sorted(added))
                del_s = ", ".join(str(a) for a in sorted(deleted))
                step_line += (f"\n  Added:   {{{added_s}}}"
                              f"\n  Deleted: {{{del_s}}}")
            if include_states:
                state_dict = {}
                for obj in sorted(next_state, key=str):
                    obj_feats = {}
                    for feat in obj.type.feature_names:
                        val = next_state.get(obj, feat)
                        obj_feats[feat] = round(float(val), 4) \
                            if isinstance(val, (float, int)) else str(val)
                    state_dict[str(obj)] = obj_feats
                step_line += f"\n  State: {json.dumps(state_dict, indent=4)}"
            lines.append(step_line)

            # Save per low-level action images in a separate directory
            last_traj = getattr(ctx.option_model, 'last_trajectory', None)
            if save_low_level_action_images and last_traj is not None \
                    and ctx.image_save_dir:
                action_img_dir = ctx.image_save_dir + "_low_level"
                orig_img_dir = ctx.image_save_dir
                ctx.image_save_dir = action_img_dir
                for act_idx, act_state in enumerate(last_traj.states):
                    _render_pybullet_image(
                        ctx,
                        f"step_{step_idx}_{opt_name}_act_{act_idx}",
                        state=act_state)
                ctx.image_save_dir = orig_img_dir

            # Always render and save scene image after this step
            img_block = _render_scene_image(ctx, f"step_{step_idx}_{opt_name}")
            if img_block and img_block.get("saved_path"):
                saved_image_paths.append(img_block["saved_path"])

            state = next_state

        final_atoms = utils.abstract(state, ctx.predicates)
        goal_achieved = task.goal.issubset(final_atoms)
        goal_str = ", ".join(str(g) for g in sorted(task.goal))
        final_atoms_str = ", ".join(str(a) for a in sorted(final_atoms))
        lines.append(f"\nFinal atoms: {{{final_atoms_str}}}")
        lines.append(f"Goal: {{{goal_str}}}")
        lines.append(f"Goal achieved: {goal_achieved}")
        if not goal_achieved:
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

    # ===== PLANNING TOOLS =====

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

        # Check goal
        if ctx.option_model is not None:
            final_atoms = utils.abstract(state, all_preds)
            goal_achieved = task.goal.issubset(final_atoms)
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

    # ------------------------------------------------------------------ #
    # Scene annotation
    # ------------------------------------------------------------------ #

    @tool(
        "annotate_scene",
        "Draw annotations (markers, lines, rectangles) at world "
        "coordinates in the 3D scene, render an image, and save it. "
        "Use this to visualize candidate placement positions or spatial "
        "relationships before committing to test_option_plan. Annotations "
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
                                "description": "Object name (e.g. 'jug0')"
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

    _all = {
        "inspect_types": inspect_types,
        "inspect_predicates": inspect_predicates,
        "inspect_processes": inspect_processes,
        "inspect_options": inspect_options,
        "inspect_trajectories": inspect_trajectories,
        "inspect_train_tasks": inspect_train_tasks,
        "inspect_planning_results": inspect_planning_results,
        "inspect_past_proposals": inspect_past_proposals,
        "propose_types": propose_types,
        "propose_predicates": propose_predicates,
        "propose_object_augmentor": propose_object_augmentor,
        "propose_processes": propose_processes,
        "propose_options": propose_options,
        "retract_abstractions": retract_abstractions,
        "test_predicate_on_states": test_predicate_on_states,
        "test_planning": test_planning,
        "test_option_plan": test_option_plan,
        "generate_bilevel_plan": generate_bilevel_plan,
        "generate_abstract_plan": generate_abstract_plan,
        "annotate_scene": annotate_scene,
        "visualize_state": visualize_state,
    }
    if tool_names is None:
        tools = list(_all.values())
    else:
        tools = [_all[n] for n in tool_names if n in _all]
    tools.extend(ctx.extra_mcp_tools)
    return tools


# ── Sim-learning tools ───────────────────────────────────────────


def create_synthesis_tools(
    exec_ns: Dict[str, Any],
    base_pred_triples: list,
    inferred_process_features: Dict[str, List[str]],
    simulator_file: str,
    versions_dir: str,
    approach: Optional[Any] = None,
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
    the current contents into ``versions_dir`` (``001_simulator.py``,
    ``002_simulator.py`` …) so the full history of evaluated versions
    is preserved; identical-content calls reuse the prior snapshot.
    Each tool's output is prefixed with the version tag (``[vNNN]``).

    * ``run_python`` — executes arbitrary Python in a persistent
      namespace pre-loaded with trajectory data. Use this for ad-hoc
      exploration of ``trajectories`` etc.; it does **not** define
      rules — write ``simulator.py`` for that.
    * ``evaluate_step_fit`` — SSE of the current ``PROCESS_RULES`` at
      init_value params; optional MCMC fit reports post-fit SSE,
      percent improvement, and fitted parameter values.
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
    """
    # pylint: disable=import-outside-toplevel
    import io
    import sys
    import traceback  # pylint: disable=redefined-outer-name,reimported
    from collections import defaultdict

    from claude_agent_sdk import tool

    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    from predicators.code_sim_learning.synthesis_validation import \
        run_refinement_for_synthesis
    from predicators.code_sim_learning.training import ParamSpec, compute_sse
    from predicators.code_sim_learning.utils import apply_rules, \
        iter_feature_residuals, merge_updates, read_simulator_components

    # pylint: enable=import-outside-toplevel

    _version_count = [0]
    _last_snapshot_hash: List[Optional[str]] = [None]

    def _text(msg: str) -> Dict[str, Any]:
        # MCP @tool callables must return a CallToolResult-shape dict
        # (``{"content": [<block>, ...]}``), not a bare content block.
        # Returning the bare block triggers a Pydantic validation error
        # on every tool call (the runtime falls through to the default
        # CallToolResult fields and tries to validate ``meta`` / empty
        # ``content`` as TextContent items).
        return {"content": [{"type": "text", "text": msg}]}

    def _snapshot_and_load(path: str) -> Tuple[Any, Any, Any, Any, Any]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(rules, specs, features, version_tag, error_msg)``;
        ``error_msg`` is ``None`` on success. Snapshots are deduped by
        SHA256, so repeated calls on unchanged content reuse the prior
        ``vNNN`` tag.
        """
        if not os.path.isfile(path):
            return None, None, None, None, (
                f"Simulator file not found: {path}. Use Write to create it "
                "with PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES.")
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != _last_snapshot_hash[0]:
            _version_count[0] += 1
            os.makedirs(versions_dir, exist_ok=True)
            snap_path = os.path.join(versions_dir,
                                     f"{_version_count[0]:03d}_simulator.py")
            with open(snap_path, "wb") as f:
                f.write(raw)
            _last_snapshot_hash[0] = digest
        version_tag = f"v{_version_count[0]:03d}"

        ns: Dict[str, Any] = {"np": np, "ParamSpec": ParamSpec}
        try:
            exec(raw.decode("utf-8"), ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            return None, None, None, version_tag, (
                f"[{version_tag}] Error executing {path}:\n"
                f"{traceback.format_exc()}")
        rules, specs, features = read_simulator_components(ns)
        if rules is None:
            return None, None, None, version_tag, (
                f"[{version_tag}] PROCESS_RULES missing or empty in {path}.")
        if specs is None:
            return None, None, None, version_tag, (
                f"[{version_tag}] PARAM_SPECS missing or empty in {path}.")
        return rules, specs, features, version_tag, None

    # ── run_python ──────────────────────────────────────────

    @tool(
        "run_python",
        "Execute Python code for ad-hoc data exploration. Available "
        "variables: trajectories (List[LowLevelTrajectory]), np, "
        "ParamSpec. print() output is returned. The namespace persists "
        "across calls. This does NOT define rules — write `simulator.py` "
        "for that; the synthesis tools (evaluate_step_fit, report_residuals, "
        "evaluate_plan_refinement) load PROCESS_RULES, PARAM_SPECS, "
        "PROCESS_FEATURES from that file.",
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
        return _text(output or "(no output)")

    # ── evaluate_step_fit ────────────────────────────────────────

    @tool(
        "evaluate_step_fit",
        "Score the current PROCESS_RULES (loaded fresh from "
        "`simulator.py`) by SSE on the step transitions. By default "
        "evaluates at init_value params from PARAM_SPECS — fast, "
        "repeatable, ideal for comparing proposals. Pass fit=true to "
        "additionally run MCMC, report the post-fit SSE and percent "
        "improvement, and show fitted parameter values with their "
        "delta from init. Each call snapshots the simulator file into "
        "simulator_versions/; output is tagged [vNNN].",
        {
            "type": "object",
            "properties": {
                "fit": {
                    "type":
                    "boolean",
                    "description":
                    "If true, run MCMC fit and also "
                    "report post-fit SSE plus fitted parameters "
                    "(slow). Default false.",
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
    async def evaluate_step_fit(args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path") or simulator_file
        rules, specs, declared, version_tag, err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_note = ("declared" if isinstance(declared, dict) else
                      "inferred (PROCESS_FEATURES not declared)")

        do_fit = bool(args.get("fit", False))

        init_params = {s.name: s.init_value for s in specs}
        sim_fn = lambda s, _a, p: apply_rules(s, rules, p)  # noqa: E731
        try:
            pre_sse = compute_sse(sim_fn, base_pred_triples, init_params,
                                  process_features)
        except Exception as e:  # pylint: disable=broad-except
            return _text(
                f"[{version_tag}] Error: SSE computation failed:\n{e}")

        lines = [
            f"[{version_tag}] Fit evaluation on {len(base_pred_triples)} "
            f"step transitions (scope: {scope_note}).",
            "",
            f"At init_value params:  SSE = {pre_sse:.6f}",
        ]

        if do_fit:
            try:
                fitted_params, post_sse = (
                    AgentSimLearningApproach._fit_parameters(  # pylint: disable=protected-access
                        rules, specs, base_pred_triples, process_features))
            except Exception as e:  # pylint: disable=broad-except
                return _text(f"[{version_tag}] Error: fit_params failed:\n{e}")
            if pre_sse > 0:
                pct = (pre_sse - post_sse) / pre_sse * 100
                pct_str = f"({pct:+.1f}% vs init)"
            else:
                pct_str = "(init SSE was 0)"
            lines.append(f"After MCMC fit:        SSE = {post_sse:.6f}  "
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
        "output is tagged [vNNN].",
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
        rules, specs, declared, version_tag, err = _snapshot_and_load(path)
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

        pairs = base_pred_triples[:max_n]
        if do_fit:
            try:
                t_params, _ = (
                    AgentSimLearningApproach._fit_parameters(  # pylint: disable=protected-access
                        rules, specs, base_pred_triples, process_features))
                param_label = "fitted"
            except Exception as e:  # pylint: disable=broad-except
                return _text(
                    f"[{version_tag}] Error: param fitting failed:\n{e}")
        else:
            t_params = {s.name: s.init_value for s in specs}
            param_label = "init_value"

        triples_rules: List = []
        triples_base: List = []
        for base_state, _action, s_next_obs in pairs:
            updates = apply_rules(base_state, rules, t_params)
            s_pred_rules = (merge_updates(base_state, updates)
                            if updates else base_state)
            triples_rules.append((s_pred_rules, s_next_obs))
            triples_base.append((base_state, s_next_obs))

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

        n_steps = len(pairs)
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
            lines.append(f"Worst {n_examples} mismatches per feature:")
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
        "the fitted params, then run backtracking refinement on a "
        "training task against a plan you propose. Always fits first "
        "because refinement needs to test the simulator at its "
        "deployed (fitted) params, not at init_value. Pass `plan` as "
        "the option-skeleton you believe should solve the task, one "
        "option call per line, e.g. `PickJug(jug0)\\nSwitchFaucetOn"
        "(faucet0)\\n...`. Subgoal annotations are supported (see the "
        "bilevel sketch parser). Falls back to "
        "CFG.agent_bilevel_plan_sketch_file or oracle task planning "
        "when `plan` is empty. Reports success, refined-plan length, "
        "sketch source, post-fit SSE, and (on failure) which step "
        "refinement got stuck on. Each call snapshots the simulator "
        "file into simulator_versions/; output is tagged [vNNN]. "
        "Slow — use sparingly.",
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Option-skeleton plan text, one "
                    "option call per line. This is the primary "
                    "interface — supply it whenever you can.",
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
                    "Refinement timeout in seconds "
                    "(default 30). Note: MCMC fitting runs before "
                    "refinement and is not subject to this timeout.",
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
        rules, specs, declared, version_tag, err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)

        task_idx = int(args.get("task_idx", 0))
        timeout = float(args.get("timeout", 30.0))
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
