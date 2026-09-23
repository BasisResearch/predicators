"""Render fan task initial states for inspection.

For every requested seed and task index this writes one figure with the
PyBullet render on the left and a top-down schematic on the right: the
grid, the wall cells, the ball, the target, the four fan rows with the
direction each blows, and the route with the fewest straight runs (one
run per fan activation) that the generator validated. A contact sheet
of all schematics is written next to them.

Usage:

    python scripts/render_fan_tasks.py --out /tmp/fan_tasks \\
        --seeds 0 1 2 --num-tasks 2 --split test

Any settings flag can be overridden with ``--flag name=value`` (the
value is parsed as YAML), for example
``--flag fan_test_num_walls_per_task=[24]``.
"""
import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import yaml
from matplotlib import patches
from PIL import Image

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object, State

matplotlib.use("Agg")

Cell = Tuple[int, int]

# side_idx -> (label, unit push direction on the ball)
_FAN_SIDES = {
    0: ("fan_0 (left)", (1.0, 0.0)),
    1: ("fan_1 (right)", (-1.0, 0.0)),
    2: ("fan_2 (down)", (0.0, 1.0)),
    3: ("fan_3 (up)", (0.0, -1.0)),
}


def _parse_flag(text: str) -> Tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    name, value = text.split("=", 1)
    return name.strip(), yaml.safe_load(value)


def _cell_of(x: float, y: float, xs: Sequence[float],
             ys: Sequence[float]) -> Cell:
    return (int(np.argmin([abs(c - x) for c in xs])),
            int(np.argmin([abs(c - y) for c in ys])))


def _first(state: State, type_name: str) -> Optional[Object]:
    return next((o for o in state if o.type.name == type_name), None)


def _draw_schematic(ax: Any, env: PyBulletFanEnv, task: EnvironmentTask,
                    split: str, seed: int, index: int) -> Dict[str, Any]:
    """Draw the top-down layout and return the layout summary."""
    state = task.init
    if split == "train":
        num_x, num_y = CFG.fan_train_num_pos_x, CFG.fan_train_num_pos_y
    else:
        num_x, num_y = CFG.fan_test_num_pos_x, CFG.fan_test_num_pos_y
    xs, ys = env._generate_grid_coordinates(num_x, num_y)  # pylint: disable=protected-access
    gap = env.pos_gap
    ball = _first(state, "ball")
    target = _first(state, "target")
    assert ball is not None and target is not None
    walls: Set[Cell] = {
        _cell_of(state.get(o, "x"), state.get(o, "y"), xs, ys)
        for o in state if o.type.name == "wall"
    }
    ball_cell = _cell_of(state.get(ball, "x"), state.get(ball, "y"), xs, ys)
    target_cell = _cell_of(state.get(target, "x"), state.get(target, "y"), xs,
                           ys)
    path = env._min_segment_path(  # pylint: disable=protected-access
        ball_cell, target_cell, walls, num_x, num_y)
    segments = env._count_segments(path) if path else 0  # pylint: disable=protected-access

    x_min, x_max = min(xs) - gap / 2, max(xs) + gap / 2
    y_min, y_max = min(ys) - gap / 2, max(ys) + gap / 2
    ax.set_aspect("equal")
    ax.set_xlim(env.left_fan_x - 0.05, env.right_fan_x + 0.05)
    ax.set_ylim(env.down_fan_y - 0.05, env.up_fan_y + 0.05)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # Cell centres.
    for cx in xs:
        for cy in ys:
            ax.plot(cx, cy, ".", color="0.85", markersize=3, zorder=1)
    # Boundary slabs.
    ax.add_patch(
        patches.Rectangle((x_min, y_min),
                          x_max - x_min,
                          y_max - y_min,
                          fill=False,
                          linewidth=2,
                          edgecolor="0.4",
                          zorder=2))
    # Walls at their physical size.
    for j, i in walls:
        ax.add_patch(
            patches.Rectangle(
                (xs[j] - env.wall_x_len / 2, ys[i] - env.wall_y_len / 2),
                env.wall_x_len,
                env.wall_y_len,
                facecolor="0.25",
                edgecolor="black",
                zorder=3))
    # Fan rows: a bar along each side plus the push direction.
    for side, (label, (dx, dy)) in _FAN_SIDES.items():
        if side == 0:
            bx, by, w, h = (env.left_fan_x - env.fan_x_len / 2, env.fan_y_lb,
                            env.fan_x_len, env.fan_y_ub - env.fan_y_lb)
            ax_pos = (env.left_fan_x, (env.fan_y_lb + env.fan_y_ub) / 2)
        elif side == 1:
            bx, by, w, h = (env.right_fan_x - env.fan_x_len / 2, env.fan_y_lb,
                            env.fan_x_len, env.fan_y_ub - env.fan_y_lb)
            ax_pos = (env.right_fan_x, (env.fan_y_lb + env.fan_y_ub) / 2)
        elif side == 2:
            bx, by, w, h = (env.fan_x_lb, env.down_fan_y - env.fan_x_len / 2,
                            env.fan_x_ub - env.fan_x_lb, env.fan_x_len)
            ax_pos = ((env.fan_x_lb + env.fan_x_ub) / 2, env.down_fan_y)
        else:
            bx, by, w, h = (env.fan_x_lb, env.up_fan_y - env.fan_x_len / 2,
                            env.fan_x_ub - env.fan_x_lb, env.fan_x_len)
            ax_pos = ((env.fan_x_lb + env.fan_x_ub) / 2, env.up_fan_y)
        ax.add_patch(
            patches.Rectangle((bx, by),
                              w,
                              h,
                              facecolor="lightsteelblue",
                              edgecolor="steelblue",
                              zorder=2))
        ax.annotate("",
                    xy=(ax_pos[0] + 0.05 * dx, ax_pos[1] + 0.05 * dy),
                    xytext=ax_pos,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "steelblue",
                        "lw": 1.5
                    },
                    zorder=4)
        ax.text(ax_pos[0] - 0.03 * dx,
                ax_pos[1] - 0.03 * dy,
                label,
                fontsize=7,
                ha="center",
                va="center",
                rotation=90 if side in (0, 1) else 0,
                color="steelblue",
                zorder=4)
    # Route.
    if path:
        px = [xs[j] for j, _ in path]
        py = [ys[i] for _, i in path]
        ax.plot(px, py, "-", color="orange", linewidth=2, alpha=0.8, zorder=4)
        for (ja, ia), (jb, ib) in zip(path, path[1:]):
            ax.annotate("",
                        xy=(xs[jb], ys[ib]),
                        xytext=(xs[ja], ys[ia]),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": "orange",
                            "lw": 1.2
                        },
                        zorder=4)
    # Target and ball.
    ax.add_patch(
        patches.Rectangle(
            (state.get(target, "x") - 0.03, state.get(target, "y") - 0.03),
            0.06,
            0.06,
            facecolor="limegreen",
            edgecolor="darkgreen",
            zorder=5))
    ax.add_patch(
        patches.Circle((state.get(ball, "x"), state.get(ball, "y")),
                       env.ball_radius,
                       facecolor="royalblue",
                       edgecolor="navy",
                       zorder=6))
    # Switch bank and robot for orientation.
    for switch in state.get_objects(env._switch_type):  # pylint: disable=protected-access
        ax.plot(state.get(switch, "x"),
                state.get(switch, "y"),
                "s",
                color="0.6",
                markersize=4,
                clip_on=False,
                zorder=2)
    summary = {
        "split": split,
        "seed": seed,
        "task": index,
        "grid": f"{num_x}x{num_y}",
        "walls": len(walls),
        "segments": segments,
        "path_len": (len(path) - 1) if path else None,
        "ball": ball_cell,
        "target": target_cell,
    }
    ax.set_title(
        f"{split} seed {seed} task {index}: {num_x}x{num_y} grid, "
        f"{len(walls)} walls, {segments} fan runs, "
        f"{summary['path_len']} cells",
        fontsize=9)
    return summary


def _render_image(env: PyBulletFanEnv, split: str, index: int) -> np.ndarray:
    env.reset(split, index)
    return env.render()[0]


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--flag",
                        type=_parse_flag,
                        action="append",
                        default=[],
                        help="settings override as name=value (YAML value)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    overrides = dict(args.flag)
    summaries: List[Dict[str, Any]] = []
    schematic_paths: List[str] = []
    for seed in args.seeds:
        config = {
            "env": "pybullet_fan",
            "seed": seed,
            "num_train_tasks": args.num_tasks if args.split == "train" else 1,
            "num_test_tasks": args.num_tasks if args.split == "test" else 1,
            "pybullet_camera_width": 900,
            "pybullet_camera_height": 900,
        }
        config.update(overrides)
        utils.reset_config(config)
        env = PyBulletFanEnv(use_gui=False)
        try:
            tasks = (env.get_train_tasks()
                     if args.split == "train" else env.get_test_tasks())
            for index, task in enumerate(tasks):
                image = _render_image(env, args.split, index)
                fig, (ax_img, ax_map) = plt.subplots(
                    1,
                    2,
                    figsize=(13, 6.2),
                    gridspec_kw={"width_ratios": [1, 1.05]})
                ax_img.imshow(image)
                ax_img.set_axis_off()
                ax_img.set_title(
                    f"PyBullet render ({args.split} seed {seed} "
                    f"task {index})",
                    fontsize=9)
                summary = _draw_schematic(ax_map, env, task, args.split, seed,
                                          index)
                summaries.append(summary)
                fig.tight_layout()
                out_path = os.path.join(
                    args.out, f"fan_{args.split}_seed{seed}_task{index}.png")
                fig.savefig(out_path, dpi=110)
                plt.close(fig)
                # Schematic alone, for the contact sheet.
                fig, ax_map = plt.subplots(figsize=(5.2, 5.2))
                _draw_schematic(ax_map, env, task, args.split, seed, index)
                fig.tight_layout()
                sheet_path = os.path.join(
                    args.out,
                    f"_schematic_{args.split}_seed{seed}_task{index}.png")
                fig.savefig(sheet_path, dpi=100)
                plt.close(fig)
                schematic_paths.append(sheet_path)
                print(summary)
        finally:
            p.disconnect(env._physics_client_id)  # pylint: disable=protected-access

    if schematic_paths:
        images = [Image.open(path) for path in schematic_paths]
        cols = min(3, len(images))
        rows = (len(images) + cols - 1) // cols
        cell_w = max(im.width for im in images)
        cell_h = max(im.height for im in images)
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        for k, im in enumerate(images):
            sheet.paste(im, ((k % cols) * cell_w, (k // cols) * cell_h))
        sheet_out = os.path.join(args.out, f"fan_{args.split}_sheet.png")
        sheet.save(sheet_out)
        for path in schematic_paths:
            os.remove(path)
        print(f"contact sheet: {sheet_out}")


if __name__ == "__main__":
    main()
