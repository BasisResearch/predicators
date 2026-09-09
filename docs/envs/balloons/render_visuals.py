"""Render audited hatch release sequences and a chute geometry illustration.

Run on a compute node with the frozen hatch checkout on PYTHONPATH.
These are mechanical demonstrations, not agent or oracle scorecards.
"""

import argparse
import dataclasses
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / "logs/balloons_followup_20260909/hatch-audit-seed5"


def configure(scene):
    """Match the audited physics and release-skill configuration."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 5,
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "balloons_scene": scene,
        "balloons_hatch_z": .57,
        "balloons_hatch_half_gap": .085,
        "balloons_require_jam_decoy": False,
        "skill_phase_use_motion_planning": False,
        "balloons_probe_max_steps": 500,
        "continual_obs_noise_position": .01,
        "continual_obs_noise_orientation": .02,
        "pybullet_camera_height": 640,
        "pybullet_camera_width": 800,
    })


def close_view(env):
    """Look at the box column from in front of the hatch."""
    client = env._physics_client_id
    view = p.computeViewMatrix((.42, .30, .88), (.42, 1.20, .78), (0, 0, 1),
                               physicsClientId=client)
    projection = p.computeProjectionMatrixFOV(50, 800 / 640, .05, 10)
    pixels = p.getCameraImage(800,
                              640,
                              viewMatrix=view,
                              projectionMatrix=projection,
                              shadow=0,
                              lightDirection=(1, -2, 3),
                              lightAmbientCoeff=.65,
                              lightDiffuseCoeff=.5,
                              lightSpecularCoeff=.1,
                              renderer=p.ER_TINY_RENDERER,
                              physicsClientId=client)[2]
    return np.asarray(pixels, dtype=np.uint8).reshape(640, 800,
                                                      4)[:, :, :3].copy()


def record_sequence(initial, order, name, output, expected):
    """Record the existing executable audit without changing its actions."""
    configure("hatch")
    env = PyBulletBalloonsEnv(use_gui=False)
    frames = []
    samples = []
    actions = []
    try:
        env._set_state(initial)
        frames.append(close_view(env))
        imageio.imwrite(output / "hatch-initial.png", frames[0])
        imageio.imwrite(output / "hatch-overview.png", env.render()[0])
        original_step = env._step_once

        def capture(action, render_obs=False):
            state = original_step(action, render_obs)
            step = len(samples) + 1
            samples.append({
                "step": step,
                "height": float(state.get(env._box, "z")),
                "pitch": float(state.get(env._box, "pitch")),
                "wall_support": bool(env._wall_support()),
            })
            actions.append(np.asarray(action.arr).copy())
            if step % 2 == 0:
                frames.append(close_view(env))
            return state

        env._step_once = capture
        result = env._run_release_sequence(initial, order)
        assert result.status == expected["status"], (name, result, expected)
        assert result.steps == expected["steps"], (name, result, expected)
        assert abs(result.height - expected["height"]) < 1e-4
        assert result.wall_supported == expected["wall_supported"]
        final_frame = close_view(env)
        imageio.imwrite(output / f"{name}-final.png", final_frame)
        imageio.imwrite(output / f"{name}-motion.png",
                        frames[len(frames) // 2])
        with imageio.get_writer(output / f"{name}.mp4",
                                fps=8,
                                codec="libx264",
                                quality=8,
                                macro_block_size=16,
                                ffmpeg_params=["-movflags",
                                               "+faststart"]) as writer:
            for _ in range(8):
                writer.append_data(frames[0])
            for frame in frames:
                writer.append_data(frame)
            for _ in range(16):
                writer.append_data(final_frame)
        np.savez_compressed(output / f"{name}-actions.npz", actions=actions)
        return {
            "name": name,
            "order": order,
            "outcome": dataclasses.asdict(result),
            "samples": samples,
            "video_fps": 8,
            "render_stride": 2,
            "playback": "16 primitive actions per second, with start/end holds"
        }
    finally:
        env.dispose()


def make_plot(runs, band, output):
    """Plot measured box heights from both mechanical replays."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none"
    })
    fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
    ax.axhspan(*band, color="#bfe7d3", alpha=.8, label="Target band")
    ax.axhline(.57,
               color="#6b7280",
               ls="--",
               lw=1.5,
               label="Hatch panel center")
    for run, color, label in zip(runs, ("#b85e13", "#008577"),
                                 ("Red then gold: jam", "Gold then red: win")):
        data = run["samples"]
        ax.plot([s["step"] for s in data], [s["height"] for s in data],
                color=color,
                lw=2.8,
                label=label)
        last = data[-1]
        ax.scatter(last["step"], last["height"], color=color, s=38, zorder=5)
    ax.set(xlabel="Primitive actions in the mechanical replay",
           ylabel="Box-center height (m)",
           ylim=(.39, .93),
           title="Same two balloons, different release order")
    ax.legend(loc="upper left", fontsize=10, framealpha=.96)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(output / "hatch-height.png", dpi=180)
    fig.savefig(output / "hatch-height.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "visuals")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # This trusted pickle was written by our own audited task generator.
    initial_file = AUDIT / "test0-initial.pkl"
    with initial_file.open("rb") as stream:
        initial = pickle.load(stream)
    configure("chute")
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(1, [2, 0, 1, 3], (.5023, .5523))
        env._set_state(state)
        imageio.imwrite(args.output / "chute-initial.png", close_view(env))
        imageio.imwrite(args.output / "chute-overview.png", env.render()[0])
    finally:
        env.dispose()
    if args.preview:
        configure("hatch")
        env = PyBulletBalloonsEnv(use_gui=False)
        try:
            env._set_state(initial)
            imageio.imwrite(args.output / "hatch-initial.png", close_view(env))
            imageio.imwrite(args.output / "hatch-overview.png",
                            env.render()[0])
        finally:
            env.dispose()
        return
    audit = json.loads((AUDIT / "report.json").read_text())
    task = next(row for row in audit["rows"] if row["split"] == "test")
    runs = []
    for order, name in [([0, 2], "hatch-jam"), ([2, 0], "hatch-pass")]:
        expected = next(row for row in task["candidates"]
                        if row["order"] == order)
        run = record_sequence(initial, order, name, args.output, expected)
        runs.append(run)
        print(json.dumps({
            "name": name,
            "outcome": run["outcome"]
        }),
              flush=True)
    band_object = next(obj for obj in initial if obj.type.name == "band")
    band = [float(initial.get(band_object, f)) for f in ("lo", "hi")]
    make_plot(runs, band, args.output)
    manifest = {
        "kind":
        "mechanical illustration, not an agent or oracle result",
        "source_commit":
        subprocess.check_output(["git", "rev-parse", "HEAD"],
                                text=True).strip(),
        "source_initial":
        str(initial_file.relative_to(ROOT)),
        "source_sha256":
        hashlib.sha256(initial_file.read_bytes()).hexdigest(),
        "audit":
        str((AUDIT / "report.json").relative_to(ROOT)),
        "seed":
        5,
        "band":
        band,
        "runs":
        runs,
        "note":
        "Renders show physical states; configured observation noise is not painted into the scene.",
    }
    (args.output /
     "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
