"""Feasibility gate for the airport domain: measure the button press window.

Pressing the button extends the pusher, but only after ``pusher_delay_steps``,
and the pusher shoves items only while it sweeps. So a press works only when
the goal item is a particular distance upstream. This sweeps that distance and
reports which leads succeed, which is the window the agent has to hit.

    uv run python -m agent_robot_control.experiments.airport_oracle --leads 0.3 0.4 0.5 0.6
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Dict  # noqa: E402

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402


def run_oracle(lead: float, seed: int = 0, task_idx: int = 2,
               hold_steps: int = 60, settle_steps: int = 250) -> Dict:
    """Press the button when the goal item is ``lead`` metres upstream."""
    from agent_robot_control.sim.session import SessionConfig, SimSession
    session = SimSession(SessionConfig(env_name="pybullet_airport",
                                       task_idx=task_idx, seed=seed,
                                       interaction_cap=100000,
                                       camera_width=224, camera_height=126,
                                       log_transitions=False))
    env, ctl = session.env, session.controller
    goal_item = next(iter(session.task.goal)).objects[0]
    q = ctl.quat_from_rpy_deg(0, 0, 0)
    bx, by = env.button_stand_x, env.button_stand_y
    bz = env.button_stand_z + env.button_height
    ctl.move_to((bx, by, bz + 0.10), q, gripper="close")

    def item_x() -> float:
        return float(env._current_observation.get(goal_item, "x"))

    waited = 0
    while waited < 800:
        if -lead - 0.011 < item_x() - env.pusher_init_x < -lead + 0.011:
            break
        session.step(ctl.hold_action())
        waited += 1
    press_x = item_x()
    ctl.move_to((bx, by, bz - 0.01), q, max_steps=20)
    pressed = env._current_observation.get(env._button, "is_pressed") > 0.5
    for _ in range(hold_steps):
        session.step(ctl.hold_action())
    ctl.move_to((bx, by, bz + 0.10), q, max_steps=20)  # release
    solved_at = None
    for k in range(settle_steps):
        session.step(ctl.hold_action())
        if env.goal_reached() and solved_at is None:
            solved_at = k
            break
    st = env._current_observation
    out = dict(lead=lead, seed=seed, pressed=bool(pressed),
               press_at_item_x=round(press_x, 3),
               solved=bool(env.goal_reached()), solved_after=solved_at,
               item_xy=[round(float(st.get(goal_item, f)), 3) for f in "xy"],
               interactions=session.interactions,
               delay=env.pusher_delay_steps)
    p.disconnect(physicsClientId=env._physics_client_id)
    return out


def main() -> None:
    """Sweep press leads and report the window."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--hold", type=int, default=60)
    args = ap.parse_args()
    rows = []
    print(f"{'lead m':>7s} {'pressed':>8s} {'solved':>7s} {'item xy after':>22s}")
    for lead in args.leads:
        res = [run_oracle(lead, seed, hold_steps=args.hold)
               for seed in range(args.seeds)]
        ok = sum(r["solved"] for r in res)
        rows.append(dict(lead=lead, solved=ok, of=args.seeds))
        print(f"{lead:7.2f} {str(res[0]['pressed']):>8s} {ok:>3d}/{args.seeds:<3d} "
              f"{str(res[0]['item_xy']):>22s}")
    good = [r["lead"] for r in rows if r["solved"] == args.seeds]
    if good:
        step = 0.01  # belt speed per env step
        print(f"\nWINDOW: leads {min(good):.2f} to {max(good):.2f} m succeed, "
              f"about {(max(good) - min(good)) / step + 1:.0f} env steps wide "
              f"(delay {rows and res[0]['delay']} steps)")
    else:
        print("\nWINDOW: no lead succeeded")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
