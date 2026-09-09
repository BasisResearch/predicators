"""Feasibility gate for the shared domino domain (predicatorv3's `domino`).

Ground-truth oracle: pick up the one blue movable domino, stand it in the gap
between the green start block and the purple target, then push the green block
with the closed gripper. Its purpose is to show that a bridge-and-push solve is
reachable for our fixed-base arm with plain ``move_to`` calls before an agent
is asked to find one, and to check that such a solve passes the domain's own
cascade certificate (which our success metric, ``goal_reached``, ignores).

    uv run python -m agent_robot_control.experiments.domino_oracle --seeds 3
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Dict, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

CFG_OVERRIDES = {
    "domino_use_domino_blocks_as_target": True,
    "domino_use_continuous_place": True,
    "domino_has_glued_dominos": False,
    "domino_initialize_at_finished_state": False,
}

# Approximate colours the task generator paints the three roles with.
_GREEN = (0.56, 0.93, 0.56)
_BLUE = (0.60, 0.80, 1.00)


def _classify(env) -> Tuple[list, list, list]:
    """Split the dominoes into (green start blocks, blues, targets)."""
    comp = env._components[0]
    state = env._current_observation
    greens, blues, targets = [], [], []
    for d in state.get_objects(comp._domino_type):
        rgb = tuple(float(state.get(d, c)) for c in "rgb")
        if np.allclose(rgb, _GREEN, atol=0.05):
            greens.append(d)
        elif np.allclose(rgb, _BLUE, atol=0.05):
            blues.append(d)
        else:
            targets.append(d)
    return greens, blues, targets


def run_oracle(seed: int = 0,
               place_offset: float = 0.0,
               verbose: bool = False,
               mode: str = "solve") -> Dict:
    """Bridge the gap with the blue domino and push the green one.

    ``place_offset`` shifts the placement along the chain axis (metres), so the
    gate can also show that a sloppy placement fails to cascade. ``mode``
    selects the negative controls:

    ``solve``      place the blue, then push the green (the intended solve);
    ``push_only``  skip the blue: the green falls into a gap too wide to
                   bridge, so the target must stay standing;
    ``no_push``    place the blue but never push, so nothing topples;
    ``cheat``      leave the blue alone and shove the target over with the
                   arm. This reaches the goal atom, so it must expose the
                   difference between our metric and the domain's own
                   certificate.
    """
    assert mode in ("solve", "push_only", "no_push", "cheat"), mode
    from agent_robot_control.sim.session import SessionConfig, SimSession
    session = SimSession(
        SessionConfig(env_name="pybullet_domino", task_idx=0, seed=seed,
                      interaction_cap=100000, camera_width=224,
                      camera_height=126, log_transitions=False,
                      record_states=True, cfg_overrides=CFG_OVERRIDES))
    env, ctl = session.env, session.controller
    greens, blues, targets = _classify(env)
    assert greens and blues and targets, (greens, blues, targets)
    green, blue, target = greens[0], blues[0], targets[0]

    def pose(obj) -> np.ndarray:
        st = env._current_observation
        return np.array([float(st.get(obj, f)) for f in ("x", "y", "z")])

    top_z = env.table_height + env.domino_height  # top face of a standing block
    # The finger pads hang about 32 mm below the EE frame, so an EE at the top
    # face grips the top third of the block and clears the table entirely.
    grasp_z = top_z
    # High enough that a carried block (which hangs 150 mm below the jaws)
    # clears the tops of the blocks it is carried over.
    lift_z = top_z + 0.20

    def wrist_for(domino_yaw: float) -> np.ndarray:
        """The wrist angle whose jaws close along a block's thin (15 mm) axis.

        A block at yaw t has its thin axis along (-sin t, cos t); the jaws
        close along world x at wrist yaw 0, so the wrist wants t + 90 deg.
        """
        return ctl.quat_from_rpy_deg(
            0, 0, float(np.degrees(domino_yaw)) + 90.0)

    st0 = env._current_observation
    green_yaw = float(st0.get(green, "yaw"))
    # A placed block must present its wide face to the oncoming one, i.e.
    # stand at the same yaw as the staged row.
    q_place = wrist_for(green_yaw)
    g, t = pose(green), pose(target)
    axis = np.array([1.0, 0.0]) if abs(g[0] - t[0]) > abs(g[1] - t[1]) \
        else np.array([0.0, 1.0])
    span = float(np.dot(t[:2] - g[:2], axis))
    n_gaps = max(1, int(round(abs(span) / env.pos_gap)))
    step = span / n_gaps
    # One blue per interior gap; this scene stages exactly one.
    slots = [g[:2] + axis * step * (i + 1) for i in range(n_gaps - 1)]
    if verbose:
        print(f"green {np.round(g, 3)} target {np.round(t, 3)} "
              f"span {span:+.3f} gaps {n_gaps} slots {np.round(slots, 3)}")

    placed = 0
    for slot, src in zip(slots if mode in ("solve", "no_push") else [],
                         [blue] + blues[1:]):
        b = pose(src)
        q_grasp = wrist_for(float(env._current_observation.get(src, "yaw")))
        ctl.move_to((b[0], b[1], lift_z), q_grasp, gripper="open")
        ctl.move_to((b[0], b[1], grasp_z), q_grasp)
        ctl.set_gripper("close")
        ctl.move_to((b[0], b[1], lift_z), q_grasp, gripper="keep")
        dest = slot + axis * place_offset
        # Turn the block to the row's yaw on the way over.
        ctl.move_to((dest[0], dest[1], lift_z), q_place, gripper="keep")
        res = ctl.move_to((dest[0], dest[1], grasp_z), q_place, gripper="keep")
        ctl.set_gripper("open")
        ctl.move_to((dest[0], dest[1], lift_z), q_place, gripper="open")
        placed += 1
        if verbose:
            print(f"placed {src.name} at {np.round(pose(src), 3)} "
                  f"(wanted {np.round(dest, 3)}, drop reached={res.reached})")

    # Push the green block along the chain axis, contacting its upper half.
    push_dir = axis * np.sign(step)
    push_z = env.table_height + env.domino_height * 0.75
    # The cheat shoves the target itself instead of the start block.
    victim = t[:2] if mode == "cheat" else g[:2]
    behind = victim - push_dir * 0.075
    ahead = victim + push_dir * 0.03
    push = None
    if mode != "no_push":
        ctl.move_to((behind[0], behind[1], push_z + 0.15), q_place,
                    gripper="close")
        ctl.move_to((behind[0], behind[1], push_z), q_place)
        push = ctl.move_to((ahead[0], ahead[1], push_z), q_place,
                           max_steps=200)
        ctl.move_to((ahead[0], ahead[1], push_z + 0.15), q_place)
    # Let the cascade finish.
    for _ in range(240):
        session.step(ctl.hold_action())

    states = session.state_history
    evaluator = getattr(session.task, "evaluator", None)
    reward: Optional[float] = None
    certified: Optional[bool] = None
    reason = ""
    if evaluator is not None:
        # pylint: disable=protected-access
        certified, reason = evaluator._certify(states, None, sim_env=env)
        reward = float(evaluator.reward(states, None, sim_env=env))
    st = env._current_observation
    out = dict(seed=seed,
               place_offset=place_offset,
               placed=placed,
               goal_reached=bool(env.goal_reached()),
               certified=certified,
               reward=reward,
               reason=reason[:300],
               interactions=session.interactions,
               mode=mode,
               push_reached=bool(push.reached) if push else None,
               push_message=push.summary()[:160] if push else "",
               blue_xy=np.round(pose(blue)[:2], 3).tolist(),
               target_tilt_deg=round(
                   float(np.degrees(abs(st.get(target, "pitch")))) if
                   "pitch" in target.type.feature_names else float("nan"), 1),
               states_recorded=len(states))
    session.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--place-offset", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--mode", default="solve",
                    choices=["solve", "push_only", "no_push", "cheat"])
    args = ap.parse_args()
    results = [run_oracle(s, args.place_offset, args.verbose, args.mode)
               for s in range(args.seeds)]
    for r in results:
        print(json.dumps(r))
    n = sum(bool(r["goal_reached"]) for r in results)
    c = sum(bool(r["certified"]) for r in results)
    print(f"goal_reached {n}/{len(results)}  certified {c}/{len(results)}")


if __name__ == "__main__":
    main()
