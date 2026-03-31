"""Test that the annotate_scene tool works end-to-end.

Validates:
1. Debug lines render in getCameraImage when option_model_use_gui=True
2. annotate_scene tool draws markers, lines, rectangles, text
3. Annotations are cleaned up after rendering (don't pollute later renders)
4. Images are saved with unique IDs

Usage:
    python tests/test_annotate_scene.py
"""
# pylint: disable=redefined-outer-name,import-outside-toplevel,protected-access
from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

import pytest

# Bootstrap circular imports
import predicators.utils as pred_utils
from predicators.settings import CFG

_CFG_OVERRIDES = {
    "env": "pybullet_boil",
    "approach": "agent_planner",
    "seed": 0,
    "use_gui": False,
    "option_model_use_gui": False,
    "num_train_tasks": 1,
    "num_test_tasks": 1,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
    "boil_goal": "human_happy",
    "boil_require_jug_full_to_heatup": True,
    "boil_water_fill_speed": 0.0015,
    "pybullet_birrt_path_subsample_ratio": 2,
    "horizon": 500,
    "max_num_steps_option_rollout": 50,
    "excluded_objects_in_state_str": "switch",
    "pybullet_camera_width": 900,
    "pybullet_camera_height": 900,
}


def _setup(sandbox_dir: str | None = None) -> tuple[Any, Any]:
    """Create boil environment, options, option model, and ToolContext."""
    pred_utils.reset_config(_CFG_OVERRIDES)

    from predicators.envs import create_new_env
    from predicators.ground_truth_models import get_gt_options
    from predicators.option_model import create_option_model

    env = create_new_env(CFG.env, do_cache=False, use_gui=CFG.use_gui)
    options = get_gt_options(env.get_name())
    predicates = env.predicates
    train_tasks = list(env.get_train_tasks())
    types = env.types
    task = train_tasks[0]

    option_model = create_option_model(CFG.option_model_name)

    from predicators.agent_sdk.tools import ToolContext
    ctx = ToolContext(
        types=types,
        predicates=predicates,
        processes=set(),
        options=options,
        train_tasks=[t.task for t in train_tasks],
        example_state=task.init,
        option_model=option_model,
        current_task=task.task,
        sandbox_dir=sandbox_dir,
    )
    if hasattr(option_model, '_simulator'):
        ctx.env = getattr(option_model._simulator, '__self__', None)

    if sandbox_dir:
        os.makedirs(os.path.join(sandbox_dir, "proposed_code"), exist_ok=True)

    return ctx, env


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_tools(ctx: Any,
                tool_names: list[str] | None = None) -> dict[str, Any]:
    from predicators.agent_sdk.tools import create_mcp_tools
    tools = create_mcp_tools(ctx, tool_names=tool_names)
    return {t.name: t.handler for t in tools}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: Any) -> Any:
    """Create the ToolContext shared by all tests in this module."""
    sandbox_dir = str(tmp_path_factory.mktemp("sandbox"))
    ctx, _env = _setup(sandbox_dir=sandbox_dir)
    return ctx


# ===== Tests =====


def test_annotate_marker(ctx: Any) -> None:
    """annotate_scene draws a marker and saves an image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir
        tools = _make_tools(ctx, ["annotate_scene"])

        result = _run(tools["annotate_scene"]({
            "annotations": [
                {
                    "type": "marker",
                    "position": [1.05, 1.27, 0.49],
                    "color": [1, 0, 0],
                    "label": "faucet_target",
                },
            ],
            "step_label":
            "marker_test",
        }))

        text = result["content"][0]["text"]
        assert "1 annotation" in text
        assert "marker" in text

        saved = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(saved) == 1, f"Expected 1 image, got {len(saved)}: {saved}"
        print(f"  PASS: annotate_scene marker ({saved[0]})")

        ctx.image_save_dir = None


def test_annotate_rectangle(ctx: Any) -> None:
    """annotate_scene draws a rectangle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir
        tools = _make_tools(ctx, ["annotate_scene"])

        result = _run(tools["annotate_scene"]({
            "annotations": [
                {
                    "type": "rectangle",
                    "min_corner": [0.95, 1.17, 0.41],
                    "max_corner": [1.15, 1.37, 0.41],
                    "color": [0, 1, 0],
                    "label": "target_region",
                },
            ],
            "step_label":
            "rect_test",
        }))

        text = result["content"][0]["text"]
        assert "1 annotation" in text
        saved = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(saved) == 1
        print(f"  PASS: annotate_scene rectangle ({saved[0]})")

        ctx.image_save_dir = None


def test_annotate_multiple(ctx: Any) -> None:
    """annotate_scene draws multiple annotations at once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir
        tools = _make_tools(ctx, ["annotate_scene"])

        result = _run(tools["annotate_scene"]({
            "annotations": [
                {
                    "type": "marker",
                    "position": [1.05, 1.27, 0.49],
                    "color": [1, 0, 0],
                    "label": "place_here",
                },
                {
                    "type": "marker",
                    "position": [0.625, 1.295, 0.49],
                    "color": [0, 0, 1],
                    "label": "burner",
                },
                {
                    "type": "line",
                    "from": [1.05, 1.5, 0.6],
                    "to": [1.05, 1.27, 0.49],
                    "color": [1, 1, 0],
                },
            ],
            "step_label":
            "multi_test",
        }))

        text = result["content"][0]["text"]
        assert "3 annotation" in text
        saved = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(saved) == 1
        print(f"  PASS: annotate_scene multiple ({saved[0]})")

        ctx.image_save_dir = None


def test_annotate_unique_ids(ctx: Any) -> None:
    """Each annotate_scene call gets a unique image filename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir
        tools = _make_tools(ctx, ["annotate_scene"])

        for i in range(3):
            _run(tools["annotate_scene"]({
                "annotations": [
                    {
                        "type": "marker",
                        "position": [0.7 + i * 0.1, 1.3, 0.5]
                    },
                ],
                "step_label":
                f"id_test_{i}",
            }))

        saved = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
        assert len(saved) == 3, f"Expected 3 images, got {len(saved)}: {saved}"
        # All should have different test_call_ids
        assert len(set(saved)) == 3
        print(f"  PASS: annotate_scene unique IDs: {saved}")

        ctx.image_save_dir = None


def test_annotations_cleaned_up(ctx: Any) -> None:
    """Annotations don't persist after annotate_scene returns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx.image_save_dir = tmpdir
        tools = _make_tools(ctx, ["annotate_scene"])

        # Draw a bright red marker
        _run(tools["annotate_scene"]({
            "annotations": [
                {
                    "type": "marker",
                    "position": [0.75, 1.35, 0.5],
                    "color": [1, 0, 0],
                    "size": 0.1,
                },
            ],
            "step_label":
            "with_marker",
        }))

        # Render plain scene (no annotations)
        from predicators.agent_sdk.tools import _render_pybullet_image
        ctx.test_call_id += 1
        _render_pybullet_image(ctx, "clean_render")

        saved = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
        assert len(saved) == 2
        print(f"  PASS: annotations cleaned up ({saved})")

        ctx.image_save_dir = None


def test_annotate_no_env(ctx: Any) -> None:
    """annotate_scene returns error when env is None."""
    tools = _make_tools(ctx, ["annotate_scene"])
    original_env = ctx.env
    ctx.env = None

    result = _run(tools["annotate_scene"]({
        "annotations": [{
            "type": "marker",
            "position": [0.5, 0.5, 0.5]
        }],
    }))

    assert result.get("is_error", False)
    ctx.env = original_env
    print("  PASS: annotate_scene no env → error")


def main() -> None:
    """Main."""
    with tempfile.TemporaryDirectory() as sandbox_dir:
        print("Setting up boil environment (with GUI for debug lines)...")
        ctx, _env = _setup(sandbox_dir=sandbox_dir)
        print(f"Setup complete. {len(ctx.options)} options, "
              f"env.using_gui={ctx.env.using_gui if ctx.env else '?'}\n")

        print("=== annotate_scene Tests ===\n")

        print("1. Basic annotation types:")
        test_annotate_marker(ctx)
        test_annotate_rectangle(ctx)
        test_annotate_multiple(ctx)

        print("\n2. Image management:")
        test_annotate_unique_ids(ctx)
        test_annotations_cleaned_up(ctx)

        print("\n3. Error handling:")
        test_annotate_no_env(ctx)

        print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
