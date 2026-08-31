"""Controlled probe of the bridge Place skill's post-release drift.

One PROCESS runs one skill VARIANT (so the CFG-driven skill construction
and the cached env stay cleanly bound): N pick-and-place trials per
placement target, each recording the held block's pose at every env
step. Because the pre-release verify already guarantees the block sits
within 4 mm of the commanded xy while still grasped, the interesting
quantity is the DRIFT: final resting xy minus the last still-held xy -
i.e. everything that happens after the gripper starts opening.

The first ``--video-trials`` trials of each target also record an mp4
(and a per-step timeline CSV) so the drift is visible, not just
tabulated.

Variants:
  baseline   - the skill exactly as production runs it today.
  preloadN   - settle stroke presses to N newtons of support force
               before releasing (skill_place_settle_preload_force=N),
               discharging the arm's position-control sag into the
               table instead of into the released block.

Usage (COMPUTE NODE - this steps pybullet physics):
  python scripts/place_drift_probe.py --variant baseline \
      --trials 10 --video-trials 2 --out logs/place_drift/<tag>
"""
import argparse
import csv
import os
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG

# Placement targets from the actual bridge staging plans (near / mid /
# far reach along the row at y=1.18); the far target is where the agent
# measured the largest systematic drift.
TARGETS = [(0.6477, 1.18), (0.7517, 1.18), (0.8557, 1.18)]
RELEASE_Z = 0.428
PICK_BLOCK = "span1"
MAX_OPTION_STEPS = 400
BASE_SEED = 0

# Env/skill flags mirroring the bridge experiment runs (subset relevant
# to env construction and skill execution).
BRIDGE_FLAGS = {
    "env": "pybullet_bridge",
    "approach": "oracle",
    "seed": BASE_SEED,
    "num_train_tasks": 1,
    "num_test_tasks": 1,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
    "pybullet_birrt_contact_margin": -0.005,
    "horizon": 3000,
    "video_fps": 20,
    "pybullet_camera_width": 640,
    "pybullet_camera_height": 640,
}


def _block_pose(state, block) -> Tuple[float, float, float]:
    return (state.get(block, "x"), state.get(block,
                                             "y"), state.get(block, "z"))


def _run_trial(env, options, target_xy: Tuple[float, float], trial_seed: int,
               record_video: bool) -> Tuple[Optional[dict], List, List[dict]]:
    """One pick-and-place; returns (row, frames, timeline)."""
    CFG.seed = BASE_SEED  # stable task layout through reset
    state = env.reset("train", 0)
    CFG.seed = trial_seed  # motion planning reads CFG.seed at call time
    objs = {o.name: o for o in state}
    robot, block = objs["robot"], objs[PICK_BLOCK]
    opt_by_name = {o.name: o for o in options}
    pick = opt_by_name["PickBlock"].ground([robot, block],
                                           np.array([0.0], dtype=np.float32))
    place = opt_by_name["Place"].ground(
        [robot],
        np.array([target_xy[0], target_xy[1], RELEASE_Z, 0.0],
                 dtype=np.float32))
    policy = utils.option_plan_to_policy([pick, place],
                                         max_option_steps=MAX_OPTION_STEPS)
    frames: List = []
    timeline: List[dict] = []
    failure: Optional[str] = None
    t = 0
    while True:
        try:
            act = policy(state)
        except utils.OptionExecutionFailure as e:
            if not e.info.get("plan_exhausted"):
                failure = str(e)
            break
        state = env.step(act)
        t += 1
        bx, by, bz = _block_pose(state, block)
        timeline.append({
            "t": t,
            "block_x": bx,
            "block_y": by,
            "block_z": bz,
            "is_held": state.get(block, "is_held"),
            "fingers": state.get(robot, "fingers"),
            "robot_z": state.get(robot, "z"),
        })
        if record_video and t % 2 == 0:
            frames.append(env.render()[0])
    if failure is not None:
        print(f"  trial seed {trial_seed}: FAILED - {failure}")
        return None, frames, timeline
    held_steps = [r for r in timeline if r["is_held"] > 0.5]
    if not held_steps:
        print(f"  trial seed {trial_seed}: block was never held")
        return None, frames, timeline
    pre = held_steps[-1]  # last still-grasped step = post-verify pose
    fin = timeline[-1]
    row = {
        "target_x": target_xy[0],
        "target_y": target_xy[1],
        "seed": trial_seed,
        "pre_err_x": pre["block_x"] - target_xy[0],
        "pre_err_y": pre["block_y"] - target_xy[1],
        "final_err_x": fin["block_x"] - target_xy[0],
        "final_err_y": fin["block_y"] - target_xy[1],
        "drift_x": fin["block_x"] - pre["block_x"],
        "drift_y": fin["block_y"] - pre["block_y"],
        "steps": t,
    }
    return row, frames, timeline


def main() -> None:
    """Run the drift trials for one variant and write the CSV report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--video-trials", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    flags = dict(BRIDGE_FLAGS)
    if args.variant.startswith("preload"):
        flags["skill_place_settle_preload_force"] = float(
            args.variant[len("preload"):])
    elif args.variant != "baseline":
        raise ValueError(f"unknown variant {args.variant}")
    utils.reset_config(flags)

    os.makedirs(args.out, exist_ok=True)
    env = create_new_env("pybullet_bridge", do_cache=True, use_gui=False)
    options = get_gt_options(env.get_name())

    rows = []
    for target_xy in TARGETS:
        print(f"[{args.variant}] target {target_xy}")
        for k in range(args.trials):
            trial_seed = 1000 + k
            record = k < args.video_trials
            row, frames, timeline = _run_trial(env, options, target_xy,
                                               trial_seed, record)
            tag = (f"{args.variant}__x{target_xy[0]:.3f}"
                   f"__trial{k}_seed{trial_seed}")
            if frames:
                imageio.mimsave(os.path.join(args.out, f"{tag}.mp4"),
                                frames,
                                fps=CFG.video_fps,
                                macro_block_size=1)
            if record and timeline:
                with open(os.path.join(args.out, f"{tag}__timeline.csv"),
                          "w",
                          newline="",
                          encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(timeline[0]))
                    w.writeheader()
                    w.writerows(timeline)
            if row is not None:
                row["variant"] = args.variant
                row["trial"] = k
                rows.append(row)
                print(f"  trial {k} (seed {trial_seed}): "
                      f"pre_err=({row['pre_err_x']*1000:+.1f}, "
                      f"{row['pre_err_y']*1000:+.1f})mm "
                      f"drift=({row['drift_x']*1000:+.1f}, "
                      f"{row['drift_y']*1000:+.1f})mm "
                      f"final_err=({row['final_err_x']*1000:+.1f}, "
                      f"{row['final_err_y']*1000:+.1f})mm")

    if rows:
        path = os.path.join(args.out, f"results__{args.variant}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for tx, _ in TARGETS:
            sub = [r for r in rows if abs(r["target_x"] - tx) < 1e-6]
            if not sub:
                continue
            dx = np.array([r["drift_x"] for r in sub]) * 1000
            dy = np.array([r["drift_y"] for r in sub]) * 1000
            fx = np.array([r["final_err_x"] for r in sub]) * 1000
            fy = np.array([r["final_err_y"] for r in sub]) * 1000
            print(f"[{args.variant}] x={tx:.3f} n={len(sub)}  "
                  f"drift mm: x {dx.mean():+.1f}+-{dx.std():.1f}, "
                  f"y {dy.mean():+.1f}+-{dy.std():.1f}  |  "
                  f"final err mm: x {fx.mean():+.1f}+-{fx.std():.1f}, "
                  f"y {fy.mean():+.1f}+-{fy.std():.1f}")
    print(f"[{args.variant}] done: {len(rows)} ok trials -> {args.out}")


if __name__ == "__main__":
    main()
