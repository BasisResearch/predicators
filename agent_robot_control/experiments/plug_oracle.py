"""Ground-truth oracle for the three-leg plug domain, and its feasibility gate.

The oracle has perfect state access, which the agent does not: it reads the
plug and outlet poses from the environment to align the legs, then lowers. Its
only purpose is to answer "is this clearance insertable at all", so the sweep
never ships a task nobody can do.

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
               yaw_offset_deg: float = 0.0, verbose: bool = False) -> Dict:
    """Grasp the plug, align all three legs over their holes, and lower.

    ``lateral_offset`` and ``yaw_offset_deg`` inject a deliberate error to
    probe how much misalignment the fit tolerates.
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
    ox, oy = float(st.get(outlet, "x")), float(st.get(outlet, "y"))
    # Grasp the block: the jaws close along world x at yaw 0.
    grasp_z = pz + 0.004
    q = ctl.quat_from_rpy_deg(0, 0, yaw_offset_deg)
    ctl.move_to((px, py, pz + 0.12), q, gripper="open")
    ctl.move_to((px, py, grasp_z), q)
    ctl.move_to((px, py, grasp_z), q, gripper="close")
    held = ctl.is_holding()
    ctl.move_to((px, py, grasp_z + 0.12), q)

    # Align: move the end effector so the plug's own frame sits over the
    # outlet centre, correcting for however the plug sits in the jaws.
    above = np.array([ox, oy, grasp_z + 0.10])
    ee = ctl.ee_position()
    st = env._current_observation
    above[:2] += ee[:2] - np.array([float(st.get(plug, "x")),
                                    float(st.get(plug, "y"))])
    ctl.move_to(above, q)
    for _ in range(3):
        st = env._current_observation
        err = np.array([ox - float(st.get(plug, "x")),
                        oy - float(st.get(plug, "y")), 0.0])
        if np.linalg.norm(err[:2]) < 0.0004:
            break
        above = above + err
        ctl.move_to(above, q)
    above[0] += lateral_offset
    if lateral_offset:
        ctl.move_to(above, q)

    # Lower until the blade tips are insertion_depth below the outlet top.
    st = env._current_observation
    ee = ctl.ee_position()
    plug_z = float(st.get(plug, "z"))
    ee_to_plug = plug_z - ee[2]
    target_plug_z = (env.outlet_top_z() - env.insertion_depth
                     + env.prong_tip_offset() - 0.004)
    insert_ee_z = target_plug_z - ee_to_plug
    fine = type(ctl)(env, step_fn=session.step, step_size=0.008,
                     max_contact_force=ctl.max_contact_force)
    fine.finger_target = ctl.finger_target
    res = fine.move_to((above[0], above[1], insert_ee_z), q, max_steps=120)

    st = env._current_observation
    tips, up = env.leg_tips_and_axis(st, plug)
    top = env.outlet_top_z()
    depths = {k: float(top - v[2]) for k, v in tips.items()}
    force, bodies = ctl.contact_force()
    out = dict(clearance=clearance, seed=seed, lateral_offset=lateral_offset,
               yaw_offset_deg=yaw_offset_deg, grasped=bool(held),
               plugged=bool(env.goal_reached()),
               min_depth=min(depths.values()), depths=depths,
               tilt_deg=float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))),
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
    ap.add_argument("--require", type=int, default=4,
                    help="successes out of --seeds needed to accept a clearance")
    ap.add_argument("--offsets", type=float, nargs="*", default=[],
                    help="also probe these lateral errors at the accepted clearance")
    args = ap.parse_args()
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
