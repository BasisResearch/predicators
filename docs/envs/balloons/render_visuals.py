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
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv, \
    any_popped, box_at_rest
from predicators.settings import CFG
from predicators.structs import Action

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / "logs/balloons_followup_20260909/hatch-audit-seed5"


def save_svg(figure, path):
    """Write portable SVG without generator timestamps or trailing spaces."""
    figure.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip()
                              for line in path.read_text().splitlines()) +
                    "\n")


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
    with env.render_attachments():
        return _close_view_pixels(env)


def _close_view_pixels(env):
    """Capture the custom camera while the environment's ropes are visible."""
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
    """Replay the saved primitive actions and check their physical outcome."""
    configure("hatch")
    env = PyBulletBalloonsEnv(use_gui=False)
    frames = []
    samples = []
    actions_path = HERE / "visuals" / f"{name}-actions.npz"
    with np.load(actions_path) as saved:
        actions = saved["actions"].copy()
    positions = []
    rest = []
    first_win = None
    try:
        env._pybullet_robot.set_joints(
            env._pybullet_robot.initial_joint_positions)
        env._set_state(initial)
        current = env.get_observation()
        frames.append(close_view(env))
        imageio.imwrite(output / "hatch-initial.png", frames[0])
        imageio.imwrite(output / "hatch-overview.png", env.render()[0])
        for action in actions:
            current = env._step_once(Action(action))
            env._current_observation = current
            step = len(samples) + 1
            assert any_popped(current) is None
            if env._InBand_holds(current, [env._box, env._band]):
                if first_win is None:
                    first_win = step
            positions.append(
                [current.get(env._box, f) for f in ("x", "y", "z")])
            angular = p.getBaseVelocity(
                env._box.id, physicsClientId=env._physics_client_id)[1]
            rest.append(
                box_at_rest(current, env._box)
                and np.linalg.norm(angular) < .01)
            samples.append({
                "step": step,
                "height": float(current.get(env._box, "z")),
                "pitch": float(current.get(env._box, "pitch")),
                "wall_support": bool(env._wall_support()),
            })
            if step % 2 == 0:
                frames.append(close_view(env))
        window = CFG.balloons_probe_rest_steps
        stationary = (len(rest) >= window and all(rest[-window:])
                      and np.max(np.ptp(positions[-window:], axis=0)) <=
                      CFG.balloons_probe_rest_tol)
        support = all(s["wall_support"] for s in samples[-window:])
        status = "unresolved"
        if first_win is not None:
            assert first_win == len(actions)
            status = "won"
        elif stationary:
            status = "resting_outside"
        result = env._probe_outcome(current, len(actions), status, stationary
                                    and support)
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
            "source_actions": str(actions_path.relative_to(ROOT)),
            "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
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
    save_svg(fig, output / "hatch-height.svg")
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
        "source_dirty":
        bool(
            subprocess.check_output(["git", "status", "--porcelain"],
                                    text=True).strip()),
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
        "Saved primitive actions are replayed exactly. Renders show physical states; configured observation noise is not painted into the scene.",
    }
    (args.output /
     "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
