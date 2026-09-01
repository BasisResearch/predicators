"""Controlled probe of the leg-drift-under-glued-span issue (ep0 obs #3).

Scenario: a leg stands at its site; a span is rested centered on the
leg's top. With ``--glue`` the leg's top face is dabbed first, so the
resting contact forms a curing pair: a 0.5 N JOINT_FIXED tack anchors
leg<->span until the cure latches a weld (~25 steps), which then snaps.
Without ``--glue`` the same rest is plain contact - the friction-only
control. The leg's xy displacement and true (geodesic) tilt are
recorded every env step through the rest window, so the timeline
separates: pre-latch tack drag, the latch snap spike, and post-latch
weld-skate, from the friction baseline.

Usage (COMPUTE NODE):
  python scripts/leg_drift_probe.py --tag glue --glue --out logs/leg_drift
  python scripts/leg_drift_probe.py --tag noglue --out logs/leg_drift
"""
import argparse
import csv
import os
from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import GroundAtom, Object, State, _Option

BRIDGE_FLAGS = {
    "env": "pybullet_bridge",
    "approach": "oracle",
    "seed": 0,
    "num_train_tasks": 1,
    "num_test_tasks": 1,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
    "pybullet_birrt_contact_margin": -0.005,
    "horizon": 3000,
    "wait_option_max_steps": 120,
    "wait_option_terminate_on_atom_change": True,
    "video_fps": 20,
    "pybullet_camera_width": 640,
    "pybullet_camera_height": 640,
}

LEG_SITE = (0.6500, 1.3000, 0.4600)
# Leg top at z=0.51 -> dab point 0.515; bottle tip rides 0.03 below the
# commanded center, aimed for the same ~12.5 mm dab margin the proven
# c4 dabs used.
DAB_MOVETO = (0.6500, 1.3000, 0.5575)
SPAN_ON_LEG = (0.6500, 1.3000, 0.5450)
POSE_FEATS = ("x", "y", "z", "roll", "pitch", "yaw")


def main() -> None:
    """Run the leg-drift trials and report per-leg drift/tilt/jump stats."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--glue", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    utils.reset_config(dict(BRIDGE_FLAGS))
    os.makedirs(args.out, exist_ok=True)

    env = create_new_env("pybullet_bridge", do_cache=True, use_gui=False)
    options = {o.name: o for o in get_gt_options(env.get_name())}
    preds = {pr.name: pr for pr in env.predicates}
    attached = preds["Attached"]
    state = env.reset("train", 0)
    objs = {o.name: o for o in state}
    robot, leg, span = objs["robot"], objs["leg0"], objs["span1"]

    def g(name: str, objects: List[Object], params: List[float]) -> _Option:
        return options[name].ground(objects, np.array(params,
                                                      dtype=np.float32))

    f32 = np.float32
    plan = [
        g("PickBlock", [robot, leg], [0.0]),
        g("Place", [robot],
          list(LEG_SITE) + [0.0]),
    ]
    if args.glue:
        plan += [
            g("PickBottle", [robot, objs["bottle"]], [0.0]),
            g("MoveTo", [robot],
              list(DAB_MOVETO) + [0.0]),
            g("Place", [robot], [0.45, 1.12, 0.45, 0.0]),
        ]
    plan += [
        g("PickBlock", [robot, span], [0.0]),
        g("Place", [robot],
          list(SPAN_ON_LEG) + [0.0]),
        options["Wait"].ground([robot], np.array([], dtype=f32)),
        options["Wait"].ground([robot], np.array([], dtype=f32)),
    ]
    if args.glue:
        # First Wait ends at the latch; second observes the welded rest.
        plan[-2].memory["wait_target_atoms"] = {
            GroundAtom(attached, [leg, span])
        }
    rest_start_opt = len(plan) - 2  # the span Place; rest = Waits after

    opt_index = {"i": 0}

    def _option_policy(s: State) -> _Option:
        del s
        if opt_index["i"] >= len(plan):
            raise utils.OptionExecutionFailure("Option plan exhausted!",
                                               info={"plan_exhausted": True})
        opt = plan[opt_index["i"]]
        opt_index["i"] += 1
        print(f"option {opt_index['i']}/{len(plan)}: {opt.name}")
        return opt

    abstract = lambda s: utils.abstract(s, env.predicates)
    policy = utils.option_policy_to_policy(_option_policy,
                                           max_option_steps=400,
                                           abstract_function=abstract)
    timeline: List[dict] = []
    frames: List = []
    failure: Optional[str] = None
    t = 0
    while True:
        try:
            act = policy(state)
        except utils.OptionExecutionFailure as e:
            if not getattr(e, "info", {}).get("plan_exhausted"):
                failure = str(e)
            break
        state = env.step(act)
        t += 1
        row: Dict[str, float] = {"t": t, "opt": opt_index["i"]}
        for name, obj in (("leg", leg), ("span", span)):
            for f in POSE_FEATS:
                row[f"{name}_{f}"] = float(state.get(obj, f))
        timeline.append(row)
        if opt_index["i"] >= rest_start_opt and t % 2 == 0:
            frames.append(env.render()[0])

    latched = attached.holds(state, [leg, span])
    print(f"[{args.tag}] end: steps={t} failure={failure} "
          f"Attached(leg0,span1)={latched}")

    # Leg drift over the rest window (everything after the span Place
    # completes), referenced to the first rest step. Per-step timeline
    # is saved; here print the max and the worst single-step jump.
    rest = [r for r in timeline if r["opt"] > rest_start_opt]
    if rest:
        ref = rest[0]

        def leg_rot(r: Dict[str, float]) -> R:
            return R.from_euler("xyz",
                                [r["leg_roll"], r["leg_pitch"], r["leg_yaw"]])

        ref_rot = leg_rot(ref)
        max_d, max_tilt, max_jump, jump_t = 0.0, 0.0, 0.0, -1
        prev: Optional[Dict[str, float]] = None
        for r in rest:
            d = float(
                np.hypot(r["leg_x"] - ref["leg_x"], r["leg_y"] - ref["leg_y"]))
            tilt = (ref_rot.inv() * leg_rot(r)).magnitude()
            max_d, max_tilt = max(max_d, d), max(max_tilt, tilt)
            if prev is not None:
                # pylint: disable=unsubscriptable-object
                # (guarded by the None check; astroid misses the
                # narrowing here)
                jump = float(
                    np.hypot(r["leg_x"] - prev["leg_x"],
                             r["leg_y"] - prev["leg_y"]))
                # pylint: enable=unsubscriptable-object
                if jump > max_jump:
                    max_jump, jump_t = jump, r["t"]
            prev = r
        print(f"[{args.tag}] leg drift over {len(rest)} rest steps: "
              f"max displacement {max_d*1000:.2f} mm, max tilt "
              f"{np.degrees(max_tilt):.2f} deg, worst single-step jump "
              f"{max_jump*1000:.2f} mm at t={jump_t}")
    with open(os.path.join(args.out, f"{args.tag}__timeline.csv"),
              "w",
              newline="",
              encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(timeline[0]))
        w.writeheader()
        w.writerows(timeline)
    if frames:
        imageio.mimsave(os.path.join(args.out, f"{args.tag}__rest.mp4"),
                        frames,
                        fps=CFG.video_fps,
                        macro_block_size=1)
    print(f"[{args.tag}] done -> {args.out}")


if __name__ == "__main__":
    main()
