"""Quantify weld compliance ("sag") of a carried bonded assembly.

Bonds the three bridge spans into a beam using the known-good staging
sequence (row at y=1.18, dab both joints, Wait for each cure), then
picks the middle span and carries the beam on a long elevated move -
recording every span's pose each env step. The sag metric is the
relative orientation/position of each outer span w.r.t. the held span,
referenced to their relative pose at the moment the carry began: a
rigid weld keeps it constant; the 2026-08-26 ep0 video showed ~40 deg
of hinge sag.

``--solver-iters N`` raises pybullet's numSolverIterations (pybullet
default: 50), and ``--weld-erp E`` sets the weld constraints' error
reduction parameter at creation (monkeypatched onto _create_weld) -
the two candidate stiffness levers for constraint-chain compliance.

Usage (COMPUTE NODE):
  python scripts/weld_sag_probe.py --tag baseline --out logs/weld_sag
  python scripts/weld_sag_probe.py --tag iters150 --solver-iters 150 \
      --out logs/weld_sag
"""
import argparse
import csv
import os
from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG

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

# Known-good c4-style bonding sequence, then a long elevated carry.
ROW_Y, ROW_Z = 1.18, 0.43
SPAN_X = {"span0": 0.6500, "span1": 0.7550, "span2": 0.8600}
DAB_Z = 0.5025
CARRY_TARGET = (0.7550, 1.4500, 0.6000)

POSE_FEATS = ("x", "y", "z", "roll", "pitch", "yaw")


def _pose(state, obj):
    return np.array([state.get(obj, f) for f in POSE_FEATS])


def _ang(d: float) -> float:
    """Wrap an angle difference into [-pi, pi]."""
    return float((d + np.pi) % (2 * np.pi) - np.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--solver-iters", type=int, default=None)
    parser.add_argument("--weld-erp", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video", action="store_true", default=True)
    args = parser.parse_args()
    utils.reset_config(dict(BRIDGE_FLAGS))
    os.makedirs(args.out, exist_ok=True)

    env = create_new_env("pybullet_bridge", do_cache=True, use_gui=False)
    if args.solver_iters is not None:
        p.setPhysicsEngineParameter(numSolverIterations=args.solver_iters,
                                    physicsClientId=env._physics_client_id)  # pylint: disable=protected-access
    if args.weld_erp is not None:
        from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
        orig_create = PyBulletBridgeEnv._create_weld

        def _create_weld_erp(self, body_a, body_b, ideal_dz=None):
            orig_create(self, body_a, body_b, ideal_dz=ideal_dz)
            cid = self._weld_constraints.get(frozenset({body_a, body_b}))
            if cid is not None:
                p.changeConstraint(cid,
                                   maxForce=self.weld_max_force,
                                   erp=args.weld_erp,
                                   physicsClientId=self._physics_client_id)

        PyBulletBridgeEnv._create_weld = _create_weld_erp
    options = {o.name: o for o in get_gt_options(env.get_name())}
    preds = {pr.name: pr for pr in env.predicates}
    state = env.reset("train", 0)
    objs = {o.name: o for o in state}
    robot = objs["robot"]
    spans = [objs["span0"], objs["span1"], objs["span2"]]

    def g(name, objects, params):
        return options[name].ground(objects, np.array(params,
                                                      dtype=np.float32))

    f32 = np.float32
    from predicators.structs import GroundAtom
    attached = preds["Attached"]
    plan = [
        g("PickBlock", [robot, objs["span1"]], [0.0]),
        g("Place", [robot], [SPAN_X["span1"], ROW_Y, ROW_Z, 0.0]),
        g("PickBottle", [robot, objs["bottle"]], [0.0]),
        g("MoveTo", [robot], [SPAN_X["span1"] - 0.05, ROW_Y, DAB_Z, 0.0]),
        g("MoveTo", [robot], [SPAN_X["span1"] + 0.05, ROW_Y, DAB_Z, 0.0]),
        g("Place", [robot], [0.45, 1.12, 0.45, 0.0]),
        g("PickBlock", [robot, objs["span0"]], [0.0]),
        g("Place", [robot], [SPAN_X["span0"], ROW_Y, ROW_Z, 0.0]),
        options["Wait"].ground([robot], np.array([], dtype=f32)),
        g("PickBlock", [robot, objs["span2"]], [0.0]),
        g("Place", [robot], [SPAN_X["span2"], ROW_Y, ROW_Z, 0.0]),
        options["Wait"].ground([robot], np.array([], dtype=f32)),
        g("PickBlock", [robot, objs["span1"]], [0.01]),
        g("Place", [robot],
          list(CARRY_TARGET) + [0.0]),
    ]
    carry_start_opt = len(plan) - 2  # PickBlock(span1) of the carry
    # Cure Waits: explicit targets, or the rich env abstraction's
    # unrelated atom flaps (Resting/Loose churn) end the Wait instantly.
    plan[8].memory["wait_target_atoms"] = {
        GroundAtom(attached, [spans[0], spans[1]])
    }
    plan[11].memory["wait_target_atoms"] = {
        GroundAtom(attached, [spans[1], spans[2]])
    }

    abstract = lambda s: utils.abstract(s, env.predicates)
    opt_index = {"i": 0}
    orig_plan = list(plan)

    def _option_policy(s):
        del s
        if opt_index["i"] >= len(orig_plan):
            raise utils.OptionExecutionFailure("Option plan exhausted!",
                                               info={"plan_exhausted": True})
        opt = orig_plan[opt_index["i"]]
        opt_index["i"] += 1
        print(f"option {opt_index['i']}/{len(orig_plan)}: {opt.name}")
        return opt

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
        row = {"t": t, "opt": opt_index["i"]}
        for sp in spans:
            for f, v in zip(POSE_FEATS, _pose(state, sp)):
                row[f"{sp.name}_{f}"] = float(v)
        row["held"] = float(state.get(objs["span1"], "is_held"))
        timeline.append(row)
        if args.video and opt_index["i"] >= carry_start_opt and t % 2 == 0:
            frames.append(env.render()[0])

    b01 = attached.holds(state, [spans[0], spans[1]])
    b12 = attached.holds(state, [spans[1], spans[2]])
    bond_report = f"Attached(span0,span1)={b01} Attached(span1,span2)={b12}"
    print(f"[{args.tag}] end: steps={t} failure={failure} {bond_report}")

    # Sag: relative pose of outer spans w.r.t. span1, referenced to the
    # first carry step (span1 held, final PickBlock onwards).
    carry = [
        r for r in timeline
        if r["opt"] >= carry_start_opt + 1 and r["held"] > 0.5
    ]
    if carry:
        ref = carry[0]
        worst = {}
        for outer in ("span0", "span2"):
            max_ang, max_pos = 0.0, 0.0
            for r in carry:
                dang = max(
                    abs(
                        _ang((r[f"{outer}_{a}"] - r[f"span1_{a}"]) -
                             (ref[f"{outer}_{a}"] - ref[f"span1_{a}"])))
                    for a in ("roll", "pitch", "yaw"))
                rel = [r[f"{outer}_{a}"] - r[f"span1_{a}"] for a in "xyz"]
                rel0 = [ref[f"{outer}_{a}"] - ref[f"span1_{a}"] for a in "xyz"]
                dpos = float(np.linalg.norm(np.subtract(rel, rel0)))
                max_ang, max_pos = max(max_ang, dang), max(max_pos, dpos)
            worst[outer] = (max_ang, max_pos)
            print(f"[{args.tag}] carry sag {outer} vs span1: "
                  f"max angle {np.degrees(max_ang):.1f} deg, "
                  f"max relative displacement {max_pos*1000:.1f} mm "
                  f"over {len(carry)} steps")
    with open(os.path.join(args.out, f"{args.tag}__timeline.csv"),
              "w",
              newline="",
              encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(timeline[0]))
        w.writeheader()
        w.writerows(timeline)
    if frames:
        imageio.mimsave(os.path.join(args.out, f"{args.tag}__carry.mp4"),
                        frames,
                        fps=CFG.video_fps,
                        macro_block_size=1)
    print(f"[{args.tag}] done -> {args.out}")


if __name__ == "__main__":
    main()
