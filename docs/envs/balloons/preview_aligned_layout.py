"""Preview a proposed common payload/rack row without changing the task code.

Run on a compute node with the refined hatch checkout on PYTHONPATH.
This produces layout illustrations, not executable-task validation.
"""
import argparse
import pickle
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
from matplotlib.patches import Circle, Rectangle
from render_visuals import AUDIT, PyBulletBalloonsEnv, configure, save_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure("hatch")
    with (AUDIT / "test0-initial.pkl").open("rb") as stream:
        initial = pickle.load(stream)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for aligned, ax in zip((False, True), axes):
        state = initial.copy()
        env = PyBulletBalloonsEnv(use_gui=False)
        try:
            balloons = env._active_balloons(state)
            clips = env._active_clips(state)
            payload_y = float(state.get(env._box, "y"))
            for half, center in env.obstacle_geometry():
                ax.add_patch(
                    Rectangle((center[0] - half[0], center[1] - half[1]),
                              2 * half[0],
                              2 * half[1],
                              facecolor="#e8e8ed",
                              edgecolor="#8a8a96",
                              linestyle="--",
                              zorder=0))
            if aligned:
                for balloon, clip in zip(balloons, clips):
                    state.set(balloon, "y", payload_y)
                    state.set(clip, "y", payload_y - .18)
            env._set_state(state)
            name = "aligned" if aligned else "current"
            imageio.imwrite(args.output / f"{name}-overview.png",
                            env.render()[0])
            half = env.box_half_extents()
            payload_x = float(state.get(env._box, "x"))
            ax.add_patch(
                Rectangle((payload_x - half[0], payload_y - half[1]),
                          2 * half[0],
                          2 * half[1],
                          facecolor="#d2a355",
                          edgecolor="#88621e",
                          zorder=3))
            ax.text(payload_x,
                    payload_y - .065,
                    "payload",
                    ha="center",
                    fontsize=10)
            for i, (balloon, clip) in enumerate(zip(balloons, clips)):
                colour = env.BALLOON_PALETTE[int(state.get(balloon,
                                                           "color"))][1]
                bx, by = (float(state.get(balloon, f)) for f in ("x", "y"))
                cx, cy = (float(state.get(clip, f)) for f in ("x", "y"))
                ax.add_patch(
                    Circle((bx, by),
                           env.balloon_radius,
                           facecolor=colour,
                           edgecolor="#444444",
                           zorder=4))
                ax.add_patch(
                    Rectangle((cx - .035, cy - .025),
                              .07,
                              .05,
                              facecolor=colour,
                              edgecolor="#444444",
                              zorder=4))
                ax.plot([bx, cx], [by, cy], color="#b3bac3", lw=1, ls=":")
                ax.text(cx, cy - .075, f"clip {i}", ha="center", fontsize=9)
            ax.axhline(payload_y, color="#1a7971", lw=1.5, ls="--", zorder=1)
            ax.set(
                title="Proposed aligned row" if aligned else "Current layout",
                xlabel="x (m)",
                ylabel="y (m)",
                xlim=(.27, 1.2),
                ylim=(.88, 1.55),
                aspect="equal")
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(.98,
                    .03,
                    "Robot side (lower y)",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    color="#526475",
                    fontsize=9)
        finally:
            env.dispose()
    fig.suptitle(
        "Plan view: balloons (circles), clips (rectangles)\n"
        "Grey outlines show the overhead hatch panels",
        fontsize=13)
    fig.savefig(args.output / "aligned-layout.png", dpi=160)
    save_svg(fig, args.output / "aligned-layout.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
