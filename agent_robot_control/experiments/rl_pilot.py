"""RL pilot without a harness: hand-written rewards, SAC vs PPO (PLAN 7).

    uv run python -m agent_robot_control.experiments.rl_pilot --task donut_push --algo sac --budget 20000 --seed 0 --out $HOME/arc_outputs/rl_pilot

Tasks
  donut_push  : from an anchor next to donut_0, push it into the target
                (reward 1 when the donut centroid is inside the target square).
  plug_insert : from an anchor holding the plug 3 cm above the socket, insert
                it (reward 1 when the plug's lowest points are >= 1.5 cm below
                the outlet top and inside the socket footprint).
Both use only particles + EE state, like an agent-written reward would.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from agent_robot_control.rl.backend import RLRequest  # noqa: E402
from agent_robot_control.rl.reward_loader import load_reward  # noqa: E402
from agent_robot_control.rl.sb3_backend import SB3Backend  # noqa: E402
from agent_robot_control.sim.session import SessionConfig, SimSession  # noqa: E402

DONUT_REWARD = """
def reward(particles, visible, ee_pos, ee_quat, gripper):
    d = particles["donut_0"]; t = particles["target"]
    if len(d) == 0 or len(t) == 0:
        return -2.0
    dc = d.mean(0); tc = t.mean(0)
    dist = float(np.linalg.norm(dc[:2] - tc[:2]))
    if dist < 0.035 and dc[2] < tc[2] + 0.04:
        return 1.0
    # shaped: closeness of donut to target, plus gripper staying near the donut
    ee_d = float(np.linalg.norm(ee_pos[:2] - dc[:2]))
    return -dist - 0.3 * ee_d
"""

PLUG_REWARD = """
def reward(particles, visible, ee_pos, ee_quat, gripper):
    plug = particles["plug"]; outlet = particles["outlet"]
    if len(plug) == 0 or len(outlet) == 0:
        return -2.0
    # Socket centre: centre of the outlet's top-face points (ring around the hole).
    top = outlet[outlet[:, 2] > outlet[:, 2].max() - 0.008]
    sc = top.mean(0) if len(top) else outlet.mean(0)
    top_z = float(outlet[:, 2].max())
    low = plug[plug[:, 2] < plug[:, 2].min() + 0.01]  # prong region
    lc = low.mean(0)
    lateral = float(np.linalg.norm(lc[:2] - sc[:2]))
    depth = top_z - float(plug[:, 2].min())
    if lateral < 0.006 and depth >= 0.012:
        return 1.0
    return -lateral * 10.0 - max(0.0, 0.03 - depth) * 5.0
"""


def make_session(task: str, seed: int, out: Path, cap: int) -> SimSession:
    env = "pybullet_donut" if task == "donut_push" else "pybullet_plug_outlet"
    cfg = SessionConfig(env_name=env, task_idx=0, seed=seed, interaction_cap=cap,
                        camera_width=335, camera_height=180, particles_per_object=32,
                        log_transitions=False, run_dir=str(out))
    return SimSession(cfg)


def move_to_anchor(session: SimSession, task: str) -> None:
    """Scripted coarse phase (what the agent would do with move_to)."""
    ctl = session.controller
    s = session.env._current_observation
    if task == "donut_push":
        d, t = session.env._donuts[0], session.env._target
        dx, dy, dz = [s.get(d, f) for f in "xyz"]
        tx, ty = s.get(t, "x"), s.get(t, "y")
        # Stand behind the donut (opposite side from the target), gripper closed, low.
        v = np.array([dx - tx, dy - ty]); v = v / (np.linalg.norm(v) + 1e-9)
        ctl.move_to((dx + 0.06 * v[0], dy + 0.06 * v[1], 0.35), gripper="close")
        ctl.move_to((dx + 0.06 * v[0], dy + 0.06 * v[1], dz + 0.005))
    else:
        env = session.env
        plug, outlet = env._plug, env._outlet
        px, py, pz = [s.get(plug, f) for f in "xyz"]
        gz = pz + 0.01
        # The plug stands at its collar's yaw, not at 0, so square the jaws
        # to it or the grasp misses.
        q_grasp = ctl.quat_from_rpy_deg(
            0, 0, float(np.degrees(s.get(plug, "yaw"))))
        ctl.move_to((px, py, pz + 0.12), q_grasp, gripper="open")
        ctl.move_to((px, py, gz), q_grasp)
        ctl.move_to((px, py, gz), q_grasp, gripper="close")
        ctl.move_to((px, py, gz + 0.14), q_grasp)
        # Turn the plug to the outlet's frame and stand off along the
        # plate's normal; the plate is slanted, so this is not straight up.
        o_pos, o_rot = env.outlet_frame(env._current_observation, outlet)
        q_ins = ctl.quat_from_rpy_deg(0.0, env.outlet_tilt_deg,
                                      float(np.degrees(s.get(outlet, "yaw"))))
        ctl.move_to(ctl.ee_position(), q_ins)
        st = env._current_observation
        offset = np.array([float(st.get(plug, f)) for f in "xyz"]) \
            - ctl.ee_position()
        # Deliberately imperfect alignment (1 cm off) so RL has work to do.
        aim = o_pos + o_rot @ np.array(
            [0.01, 0.005, 0.06 + env.prong_tip_offset()])
        ctl.move_to(aim - offset, q_ins)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["donut_push", "plug_insert"], required=True)
    ap.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--episode-length", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path.home() / "arc_outputs" / "rl_pilot")
    ap.add_argument("--half-extent", type=float, default=0.12)
    ap.add_argument("--action-mode", default="xyz")
    args = ap.parse_args()
    out = args.out / f"{args.task}_{args.algo}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    session = make_session(args.task, args.seed, out, cap=args.budget + 5000)
    t0 = time.time()
    move_to_anchor(session, args.task)
    coarse = session.interactions
    code = DONUT_REWARD if args.task == "donut_push" else PLUG_REWARD
    req = RLRequest(reward_fn=load_reward(code), reward_source=code,
                    budget_interactions=args.budget, episode_length=args.episode_length,
                    action_mode=args.action_mode, workspace_half_extent=args.half_extent,
                    points_per_object=32, out_dir=out / "rl", seed=args.seed, algo=args.algo,
                    control_gripper=False)
    res = SB3Backend().run(session, req)
    session.close()
    summary = {"task": args.task, "algo": args.algo, "seed": args.seed, "budget": args.budget,
               "coarse_interactions": coarse, "rl_interactions": res.interactions_used,
               "episodes": res.episodes, "train_successes": res.successes_during_training,
               "early_stopped": res.early_stopped, "final_exec_rewards": res.final_exec_rewards,
               "final_success": res.final_exec_success, "goal_reached_env": session.results(),
               "wall_s": time.time() - t0}
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
