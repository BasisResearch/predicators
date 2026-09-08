"""Feasibility gate for the donut push domain.

Ground-truth oracle: repeatedly place the closed gripper behind the goal disc
and shove it toward the target, re-approaching between strokes. Its only
purpose is to show the metre-long push is possible for a fixed-base arm before
any agent is asked to do it.

    uv run python -m agent_robot_control.experiments.donut_oracle --seeds 3
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Dict  # noqa: E402

import numpy as np  # noqa: E402
import pybullet as p  # noqa: E402


def run_oracle(seed: int = 0, max_strokes: int = 30,
               verbose: bool = False) -> Dict:
    """Push the goal disc into the target, one stroke at a time."""
    from agent_robot_control.sim.session import SessionConfig, SimSession
    session = SimSession(SessionConfig(env_name="pybullet_donut", task_idx=0,
                                       seed=seed, interaction_cap=100000,
                                       camera_width=224, camera_height=126,
                                       log_transitions=False))
    env, ctl = session.env, session.controller
    disc, target = env._donuts[0], env._target
    q = ctl.quat_from_rpy_deg(0, 0, 0)
    # The finger pads hang about 32 mm below the EE frame; at table+20 mm
    # they would be driven into the table and the 'push' becomes a skid.
    push_z = env.table_height + 0.045
    r = env.donut_radius

    def xy(obj) -> np.ndarray:
        st = env._current_observation
        return np.array([float(st.get(obj, "x")), float(st.get(obj, "y"))])

    start = xy(disc).copy()
    strokes = 0
    for _ in range(max_strokes):
        if env.goal_reached():
            break
        d, t = xy(disc), xy(target)
        direction = t - d
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            break
        direction /= dist
        # Stand behind the disc, lower to push height, then shove.
        behind = d - direction * (r + 0.06)
        # Keep the approach inside the arm's reachable band.
        behind[1] = max(behind[1], 0.22)
        ctl.move_to((behind[0], behind[1], push_z + 0.16), q, gripper="close")
        ctl.move_to((behind[0], behind[1], push_z), q)
        stroke = min(0.22, dist + r * 0.5)
        ahead = behind + direction * stroke
        ctl.move_to((ahead[0], ahead[1], push_z), q, max_steps=120)
        ctl.move_to((ahead[0], ahead[1], push_z + 0.16), q)
        strokes += 1
    st = env._current_observation
    out = dict(seed=seed, solved=bool(env.goal_reached()), strokes=strokes,
               interactions=session.interactions,
               start_xy=np.round(start, 3).tolist(),
               end_xy=np.round(xy(disc), 3).tolist(),
               target_xy=np.round(xy(target), 3).tolist(),
               disc_z=round(float(st.get(disc, "z")), 3),
               interventions=int(getattr(env, "num_interventions", 0)))
    if verbose:
        print(json.dumps(out))
    p.disconnect(physicsClientId=env._physics_client_id)
    return out


def main() -> None:
    """Run the gate over several seeds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rows = [run_oracle(s) for s in range(args.seeds)]
    for r in rows:
        print(f"seed {r['seed']}: solved={r['solved']} strokes={r['strokes']} "
              f"interactions={r['interactions']} disc {r['start_xy']} -> "
              f"{r['end_xy']} target {r['target_xy']}")
    n = sum(r["solved"] for r in rows)
    print(f"\nPUSH GATE: {n}/{len(rows)} solved, mean "
          f"{np.mean([r['interactions'] for r in rows]):.0f} interactions")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
