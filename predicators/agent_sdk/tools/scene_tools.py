"""Scene MCP tools: annotate_scene and visualize_state."""
from typing import Any, Callable, Dict, List

from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.results import _error_result
from predicators.agent_sdk.tools.scene import apply_state_modifications, \
    draw_pybullet_annotation, render_pybullet_image
from predicators.agent_sdk.tools.tasks import _resolve_task


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
                ids = draw_pybullet_annotation(ann, physics_id)
                all_debug_ids.extend(ids)
                pos = ann.get("position") or ann.get("min_corner", "?")
                summaries.append(f"  {ann['type']} at {pos}")
            except Exception as e:  # pylint: disable=broad-except
                summaries.append(f"  {ann['type']} FAILED: {e}")

        # Render with unique ID
        ctx.test_call_id += 1
        img_block = render_pybullet_image(ctx, f"annotated_{step_label}")

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

        resolved, task_err = _resolve_task(ctx, args.get("task_idx"))
        if task_err is not None:
            return task_err
        assert resolved is not None
        task = resolved.task
        task_idx = resolved.label

        modifications = args.get("modifications", [])
        if not modifications:
            return _error_result("No modifications provided. Pass a list of "
                                 "{object, features} dicts.")

        # Copy the initial state and apply the overrides.
        modified_state, summaries, mod_err = apply_state_modifications(
            task.init, modifications)
        if mod_err:
            return _error_result(mod_err)

        # Store for subsequent annotate_scene calls
        ctx.visualized_state = modified_state

        # Render
        step_label = args.get("step_label", "visualize")
        ctx.test_call_id += 1
        img_block = render_pybullet_image(ctx,
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
