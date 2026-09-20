"""Render the actual exposed Fan pilot geometry and annotated plan views."""
# pylint: disable=protected-access,wrong-import-order,ungrouped-imports
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import Circle, Rectangle \
    # pylint: disable=wrong-import-position


def main() -> None:
    """Write a geometry overview and actual simulator views."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/amps/figures"))
    parser.add_argument("--inertial", action="store_true")
    parser.add_argument("--ramp", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": args.seed,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": args.inertial or args.ramp,
        "fan_ramp_transfer": args.ramp,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
        "fan_test_num_pos_x": 3,
        "fan_test_num_pos_y": 3,
    })
    env = PyBulletFanEnv(use_gui=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    scenes, scene_axes = plt.subplots(1, 2, figsize=(12, 5.5))
    try:
        for column, split in enumerate(("train", "test")):
            state = env.reset(split, 0)
            ax = axes[column]
            ax.set_aspect("equal")
            ax.set_xlim(0.26, 1.67 if args.ramp else 1.24)
            ax.set_ylim(1.22, 2.10)
            ax.set_facecolor("#eef2f5")
            for obj in state:
                if obj.type.name not in ("platform", "boundary", "ramp"):
                    continue
                x, y = state.get(obj, "x"), state.get(obj, "y")
                w, h = state.get(obj, "x_len"), state.get(obj, "y_len")
                platform = obj.type.name in ("platform", "ramp")
                fill = "#e6c99e" if platform else "#465365"
                if obj.type.name == "ramp":
                    fill = "#d4a66b"
                ax.add_patch(
                    Rectangle((x - w / 2, y - h / 2),
                              w,
                              h,
                              facecolor=fill,
                              edgecolor="#b38f5e" if platform else "#465365",
                              linewidth=1 if platform else 3,
                              zorder=2 if platform else 4))
            ball, target = env._ball, env._target
            bx, by = state.get(ball, "x"), state.get(ball, "y")
            tx, ty = state.get(target, "x"), state.get(target, "y")
            ax.add_patch(
                Rectangle((tx - .04, ty - .04),
                          .08,
                          .08,
                          facecolor="#4caa83",
                          edgecolor="white",
                          linewidth=1.5,
                          zorder=5))
            ax.add_patch(
                Circle((bx, by),
                       .04,
                       facecolor="#187c9a",
                       edgecolor="white",
                       linewidth=1.5,
                       zorder=5))
            ax.text(bx, by - .075, "Start", ha="center", color="#155a70")
            if split == "train":
                title = "1  Calibrate safely"
                subtitle = "Enclosed tray; learn motion and switch timing"
                ax.annotate("Protective walls",
                            xy=(1.45 if args.ramp else 1.15, 1.65),
                            xytext=(1.22 if args.ramp else .96, 1.96),
                            ha="center",
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#465365"
                            })
                ax.text(tx,
                        ty - .09,
                        "Settled target",
                        ha="center",
                        color="#226a4d")
            else:
                title = "2  Transfer across exposed edges"
                subtitle = "Same physics; overshoot can lose the ball"
                ax.plot([bx, tx, tx], [by, by, ty],
                        "--",
                        color="#ac5b27",
                        linewidth=2,
                        zorder=3)
                ax.annotate("Turn before\nthe edge",
                            xy=(tx + .12, by),
                            xytext=(1.23 if args.ramp else .91, 1.31),
                            ha="center",
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#ac5b27"
                            },
                            color="#8c451c")
                ax.annotate("Open drop",
                            xy=(tx - .12, 1.78),
                            xytext=(.61, 1.86),
                            ha="center",
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#ac5b27"
                            },
                            color="#8c451c")
                ax.text(.51, by + .11, "Safe bay", ha="center", fontsize=10)
                ax.annotate("Stop and settle",
                            xy=(tx, ty),
                            xytext=(.62, 2.025),
                            ha="center",
                            color="#226a4d",
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#226a4d"
                            })
            if args.ramp:
                fan = env._fans[1]
                fx, fy = state.get(fan, "x"), state.get(fan, "y")
                for body in fan.fan_ids:
                    position, _ = p.getBasePositionAndOrientation(
                        body, physicsClientId=env._physics_client_id)
                    ax.scatter(position[0],
                               position[1],
                               marker="<",
                               s=65,
                               color="#465365",
                               zorder=5)
                ax.annotate("Fan bank\noutside deck",
                            xy=(fx, fy),
                            xytext=(fx, 2.045),
                            ha="center",
                            fontsize=8,
                            color="#465365")
                subtitle = ("Wide ramp and fenced landing" if split == "train"
                            else "Downhill momentum before an exposed turn")
                ax.annotate("Downhill ramp",
                            xy=(1.03, by + .075),
                            xytext=(.71, by + .075),
                            fontsize=9,
                            color="#754b25",
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#754b25"
                            },
                            zorder=6)
            ax.set_title(title + "\n" + subtitle,
                         loc="left",
                         fontsize=12,
                         pad=18)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0, labelsize=9, colors="#586777")
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(.9 if args.ramp else .75, 1.65, .25),
                distance=1.85 if args.ramp else 1.35,
                yaw=env._camera_yaw,
                pitch=env._camera_pitch,
                roll=0,
                upAxisIndex=2)
            projection = p.computeProjectionMatrixFOV(50, 1.2, .05, 5)
            frame = p.getCameraImage(1080,
                                     900,
                                     view,
                                     projection,
                                     renderer=p.ER_TINY_RENDERER,
                                     physicsClientId=env._physics_client_id)
            scene_axes[column].imshow(
                np.asarray(frame[2]).reshape(900, 1080, 4))
            scene_axes[column].set_title(title,
                                         loc="left",
                                         fontsize=14,
                                         pad=12)
            scene_axes[column].axis("off")
        variant = "ramp" if args.ramp else (
            "inertial" if args.inertial else "transfer")
        heading = {
            "ramp": "Fan downhill-ramp candidate: preview",
            "inertial": "Fan inertial-transfer pilot",
            "transfer": "Fan exposed-transfer pilot"
        }[variant]
        figure.suptitle(heading,
                        x=.08,
                        ha="left",
                        fontsize=21,
                        fontweight="bold",
                        y=.985)
        figure.text(
            .08,
            .025,
            f"Actual seed-{args.seed} geometry. Dashed route is illustrative, "
            "not an agent trace.\n"
            "Win: fans off, within 4 cm on each axis, settled for 20 steps. "
            "Falling loses the level.",
            fontsize=10,
            color="#586777")
        figure.subplots_adjust(left=.08,
                               right=.98,
                               bottom=.15,
                               top=.83,
                               wspace=.22)
        scenes.tight_layout()
        for fig, name in ((figure, f"fan-{variant}-overview"),
                          (scenes, f"fan-{variant}-scenes")):
            fig.savefig(args.out / f"{name}.png", dpi=180, facecolor="white")
            fig.savefig(args.out / f"{name}.pdf", facecolor="white")
        if args.ramp:
            profile, profile_ax = plt.subplots(figsize=(11, 3.5))
            profile_ax.plot([.35, .67, 1.07, 1.40], [.404, .404, .4, .4],
                            color="#95622f",
                            linewidth=5)
            profile_ax.fill_between([.35, .67, 1.07, 1.40],
                                    [.404, .404, .4, .4],
                                    .395,
                                    color="#e6c99e")
            profile_ax.text(.5, .406, "Safe bay", ha="center")
            profile_ax.text(.87, .406, "4 mm descent over 40 cm", ha="center")
            profile_ax.text(1.25, .402, "Flat landing\nthen turn", ha="center")
            profile_ax.set(xlim=(.30, 1.45),
                           ylim=(.395, .409),
                           xlabel="x (m)",
                           ylabel="Surface height (m)",
                           title=("Side profile through the first leg "
                                  "(vertical scale exaggerated)"))
            profile.tight_layout()
            profile.savefig(args.out / "fan-ramp-profile.png", dpi=180)
            profile.savefig(args.out / "fan-ramp-profile.pdf")
            plt.close(profile)
    finally:
        plt.close(figure)
        plt.close(scenes)
        p.disconnect(env._physics_client_id)


if __name__ == "__main__":
    main()
