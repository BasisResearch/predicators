"""Ground-truth oracle for the three-leg plug domain, and its feasibility gate.

The oracle has perfect state access, which the agent does not: it reads the
plug and outlet poses from the environment, turns the plug into the outlet's
frame, and drives in along the plate's normal. Its only purpose is to answer
"is this clearance insertable at all" on a slanted outlet, so the sweep never
ships a task nobody can do.

    uv run python -m agent_robot_control.experiments.plug_oracle \
        --clearances 0.004 0.003 0.0025 0.002 0.0015 --seeds 5
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Dict, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402


def run_oracle(clearance: float, seed: int, lateral_offset: float = 0.0,
               yaw_offset_deg: float = 0.0, tilt_offset_deg: float = 0.0,
               verbose: bool = False) -> Dict:
    """Grasp the plug, turn it to the outlet's frame, and insert along the
    plate's normal.

    The outlet is pitched off horizontal and yawed by a per-task angle, and
    the plug starts at neither, so this is three rotations and a translation
    rather than a drop. ``lateral_offset``, ``yaw_offset_deg`` and
    ``tilt_offset_deg`` inject a deliberate error to probe how much
    misalignment the fit tolerates.
    """
    from predicators.envs.pybullet_plug_outlet import PyBulletPlugOutletEnv
    from agent_robot_control.sim.session import SessionConfig, SimSession
    PyBulletPlugOutletEnv.clearance = clearance
    session = SimSession(SessionConfig(env_name="pybullet_plug_outlet",
                                       task_idx=0, seed=seed,
                                       interaction_cap=100000,
                                       camera_width=224, camera_height=126,
                                       log_transitions=False))
    env, ctl = session.env, session.controller
    plug, outlet = env._plug, env._outlet
    st = env._current_observation
    px, py, pz = [float(st.get(plug, f)) for f in "xyz"]
    plug_yaw_deg = float(np.degrees(st.get(plug, "yaw")))
    o_pos, o_rot = env.outlet_frame(st, outlet)
    o_axis = o_rot[:, 2]  # the direction the holes run

    # Grasp: the jaws close along world x at yaw 0, so square them to the
    # plug's own yaw, whatever the collar happens to be turned to.
    grasp_z = pz + 0.004
    q_grasp = ctl.quat_from_rpy_deg(0, 0, plug_yaw_deg)
    ctl.move_to((px, py, pz + 0.12), q_grasp, gripper="open")
    ctl.move_to((px, py, grasp_z), q_grasp)
    ctl.move_to((px, py, grasp_z), q_grasp, gripper="close")
    ctl.move_to((px, py, grasp_z + 0.14), q_grasp)
    # Read the grasp after the lift: the pads finish closing on the move
    # that follows the close command, so checking any earlier reports False
    # on a pick that in fact succeeded.
    held = ctl.is_holding()

    # Turn the held plug to the outlet's orientation, clear of everything.
    # The grasp is square to the plug, so the end effector's rotation on top
    # of home IS the plug's rotation: Rz(outlet yaw) . Ry(tilt).
    o_yaw_deg = float(np.degrees(st.get(outlet, "yaw")))
    q_ins = ctl.quat_from_rpy_deg(0.0,
                                  env.outlet_tilt_deg + tilt_offset_deg,
                                  o_yaw_deg + yaw_offset_deg)
    ee = ctl.ee_position()
    ctl.move_to(ee, q_ins)

    # Where the plug frame has to end up: on the outlet's axis, deep enough
    # that the blade tips clear insertion_depth below the top face. All in
    # the outlet's frame, because "down" is the plate's normal now.
    half_h = env.outlet_height / 2.0
    z_local = (half_h - env.insertion_depth - 0.004 + env.prong_tip_offset())
    seated = o_pos + o_rot @ np.array([lateral_offset, 0.0, z_local])
    standoff = seated + o_axis * 0.09

    def plug_pos() -> np.ndarray:
        s = env._current_observation
        return np.array([float(s.get(plug, f)) for f in "xyz"])

    # The plug sits in the jaws with some fixed offset; measure it and aim
    # the end effector so the PLUG, not the gripper, lands on the axis.
    ctl.move_to(standoff - (plug_pos() - ctl.ee_position()), q_ins)
    for _ in range(3):
        err = standoff - plug_pos()
        if np.linalg.norm(err) < 0.0004:
            break
        ctl.move_to(ctl.ee_position() + err, q_ins)

    # Drive in along the plate's normal.
    offset = plug_pos() - ctl.ee_position()
    fine = type(ctl)(env, step_fn=session.step, step_size=0.008,
                     max_contact_force=ctl.max_contact_force)
    fine.finger_target = ctl.finger_target
    res = fine.move_to(seated - offset, q_ins, max_steps=160)

    st = env._current_observation
    tips, up = env.leg_tips_and_axis(st, plug)
    o_pos, o_rot = env.outlet_frame(st, outlet)
    # Depth below the top face, measured down the outlet's own axis.
    depths = {k: float(half_h - (o_rot.T @ (v - o_pos))[2])
              for k, v in tips.items()}
    force, bodies = ctl.contact_force()
    out = dict(clearance=clearance, seed=seed, lateral_offset=lateral_offset,
               yaw_offset_deg=yaw_offset_deg,
               tilt_offset_deg=tilt_offset_deg, grasped=bool(held),
               plugged=bool(env.goal_reached()),
               min_depth=min(depths.values()), depths=depths,
               tilt_deg=float(np.degrees(np.arccos(
                   np.clip(up.dot(o_rot[:, 2]), -1, 1)))),
               interactions=session.interactions, contact_force=force,
               contact_with=bodies, move_msg=res.message[:90])
    if verbose:
        print(json.dumps(out, indent=2, default=str))
    p.disconnect(physicsClientId=env._physics_client_id)
    return out


def main() -> None:
    """Sweep clearances and report the tightest reliable one."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearances", type=float, nargs="+",
                    default=[0.004, 0.003, 0.0025, 0.002, 0.0015])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--require", type=int, default=None,
                    help="successes out of --seeds needed to accept a "
                         "clearance (default: 80%% of --seeds, rounded up, "
                         "which is the historical 4-of-5 rule; a fixed 4 "
                         "silently failed every run with --seeds 3)")
    ap.add_argument("--offsets", type=float, nargs="*", default=[],
                    help="also probe these lateral errors at the accepted clearance")
    args = ap.parse_args()
    if args.require is None:
        args.require = int(np.ceil(0.8 * args.seeds))
    print(f"{'clearance':>10s} {'inserted':>9s} {'min depth mm':>13s} "
          f"{'tilt deg':>9s} {'steps':>6s} {'force N':>8s}")
    accepted: Optional[float] = None
    rows = []
    for c in sorted(args.clearances, reverse=True):
        res = [run_oracle(c, seed) for seed in range(args.seeds)]
        n = sum(r["plugged"] for r in res)
        depth = np.mean([r["min_depth"] for r in res]) * 1000
        tilt = np.mean([r["tilt_deg"] for r in res])
        steps = int(np.mean([r["interactions"] for r in res]))
        force = np.mean([r["contact_force"] for r in res])
        rows.append(dict(clearance=c, inserted=n, of=args.seeds,
                         mean_min_depth_mm=round(float(depth), 2),
                         mean_tilt_deg=round(float(tilt), 2), mean_steps=steps))
        print(f"{c*1000:9.2f}mm {n:>4d}/{args.seeds:<4d} {depth:13.2f} "
              f"{tilt:9.2f} {steps:6d} {force:8.0f}")
        if n >= args.require and accepted is None:
            accepted = c
    print()
    print(f"ACCEPTED CLEARANCE: {accepted*1000:.2f} mm" if accepted else
          "ACCEPTED CLEARANCE: none passed the gate")
    if accepted and args.offsets:
        print("\nmisalignment probe at the accepted clearance:")
        for off in args.offsets:
            r = [run_oracle(accepted, s, lateral_offset=off) for s in range(3)]
            print(f"  lateral {off*1000:5.1f} mm -> inserted "
                  f"{sum(x['plugged'] for x in r)}/3, mean min depth "
                  f"{np.mean([x['min_depth'] for x in r])*1000:5.2f} mm")
    print("\n" + json.dumps(rows))


if __name__ == "__main__":
    main()
