"""Compute-node diagnostics for exposed Fan physics and skill feasibility."""
# pylint: disable=protected-access
import argparse
import json

import numpy as np
import pybullet as p
from PIL import Image

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action, State


def main() -> None:
    """Measure the physical response or execute the reference skill policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--inertial", action="store_true")
    parser.add_argument("--search-bursts", action="store_true")
    parser.add_argument("--ramp", action="store_true")
    parser.add_argument("--landing-extension", type=float, default=0.0)
    parser.add_argument("--ramp-rise", type=float, default=0.004)
    parser.add_argument("--brake-search", action="store_true")
    parser.add_argument("--brake-max", type=int, default=60)
    parser.add_argument("--contacts", action="store_true")
    parser.add_argument(
        "--replay-bursts",
        type=str,
        default="",
        help="Replay side:wait:brake triples separated by commas")
    args = parser.parse_args()
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": args.seed,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": args.inertial,
        "fan_ramp_transfer": args.ramp,
        "fan_ramp_landing_extension": args.landing_extension,
        "fan_ramp_rise": args.ramp_rise,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
        "fan_test_num_pos_x": 3,
        "fan_test_num_pos_y": 3,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "pybullet_birrt_path_subsample_ratio": 2,
    })
    env = PyBulletFanEnv(use_gui=False)
    try:
        if args.render:
            for split in ("train", "test"):
                env.reset(split, 0)
                Image.fromarray(
                    env.render()[0]).save(  # type: ignore[no-untyped-call]
                        f"logs/fan-transfer-{split}-seed{args.seed}.png")
            return

        def noop(state: State) -> Action:
            assert isinstance(state, utils.PyBulletState)
            return Action(np.asarray(state.joint_positions, dtype=np.float32))

        if args.contacts:
            state = env.reset("train", 0).copy()
            state.set(env._switches[0], "is_on", 1.0)
            for _ in range(160):
                state = env.simulate(state, noop(state))
            print("CONTACT_POSITION", state.get(env._ball, "x"), flush=True)
            for contact in p.getContactPoints(
                    bodyA=env._ball.id,
                    physicsClientId=env._physics_client_id):
                other = contact[2]
                print("CONTACT_BODY",
                      other,
                      p.getBodyInfo(other,
                                    physicsClientId=env._physics_client_id),
                      p.getAABB(other, physicsClientId=env._physics_client_id),
                      flush=True)
            return

        if not args.skills:
            for force in (0.0016, 0.002, 0.003, 0.004):
                setattr(env, "exposed_wind_force_magnitude", force)
                state = env.reset("train", 0).copy()
                state.set(env._switches[0], "is_on", 1.0)
                samples = []
                pulse_steps = 10 if args.inertial else 60
                for tick in range(120):
                    if tick == pulse_steps:
                        state = state.copy()
                        state.set(env._switches[0], "is_on", 0.0)
                    state = env.simulate(state, noop(state))
                    if tick in (9, 19, 29, 39, 59, 69, 89, 119):
                        samples.append((
                            tick + 1, state.get(env._ball, "x"),
                            p.getBaseVelocity(
                                env._ball.id,
                                physicsClientId=env._physics_client_id)[0][0]))
                print("FORCE_PROBE", force, json.dumps(samples), flush=True)
            return

        options = {o.name: o for o in get_gt_options("pybullet_fan")}
        state = env.reset("test", 0)
        states = [state]
        target_x = state.get(env._target, "x")
        target_y = state.get(env._target, "y")

        if args.search_bursts or args.replay_bursts:
            # Privileged feasibility diagnostic, never an agent score.
            # Search with real switch controllers, then replay on the live env.
            def burst(start: State,
                      side: int,
                      wait: int,
                      live: bool,
                      brake: int = -1) -> list[State]:
                current = start.copy()
                trace = []

                def advance(action: Action) -> None:
                    nonlocal current
                    current = (env.step(action) if live else env.simulate(
                        current, action))
                    trace.append(current)
                    if current.get(env._ball, "z") < 0.30:
                        raise RuntimeError("ball fell off")

                sequence = [("SwitchOn", side, wait),
                            ("SwitchOff", side, 120 if brake < 0 else 0)]
                if brake >= 0:
                    sequence.extend([("SwitchOn", side ^ 1, brake),
                                     ("SwitchOff", side ^ 1, 120)])
                for name, fan_side, pause in sequence:
                    option = options[name].ground(
                        [env._robot, env._fans[fan_side]],
                        np.array([0.05, 0.11], dtype=np.float32))
                    assert option.initiable(current)
                    for _ in range(200):
                        if option.terminal(current):
                            break
                        advance(option.policy(current))
                    else:
                        raise RuntimeError("switch failed")
                    for _ in range(pause):
                        advance(noop(current))
                return trace

            selected = []
            if args.replay_bursts:
                selected = [
                    tuple(map(int, entry.split(":")))
                    for entry in args.replay_bursts.split(",")
                ]
                if any(len(entry) != 3 for entry in selected):
                    raise ValueError("Expected side:wait:brake triples")
            searches = [] if selected else [(0, "x", target_x),
                                            (2, "y", target_y)]
            for side, axis, target in searches:
                candidates = []
                timings = [(wait, -1) for wait in range(0, 51, 2)]
                if args.brake_search and side == 0:
                    timings += [(wait, brake) for wait in range(0, 21, 4)
                                for brake in range(0, args.brake_max + 1, 10)]
                failures = 0
                for wait, brake in timings:
                    try:
                        trace = burst(state, side, wait, False, brake)
                    except RuntimeError:
                        failures += 1
                        continue
                    error = abs(trace[-1].get(env._ball, axis) - target)
                    candidates.append((error, wait, brake, trace))
                print("SEARCH_COVERAGE",
                      axis,
                      len(timings),
                      failures,
                      flush=True)
                if not candidates:
                    raise RuntimeError("No feasible burst")
                error, wait, brake, trace = min(candidates, key=lambda c: c[0])
                print("SELECTED_BURST", axis, wait, brake, error, flush=True)
                if error > 0.03:
                    raise RuntimeError("No accurate burst")
                selected.append((side, wait, brake))
                state = trace[-1]
            state = env.reset("test", 0)
            states = [state]
            for side, wait, brake in selected:
                trace = burst(state, side, wait, True, brake)
                states.extend(trace)
                state = trace[-1]
            evaluator = env.get_test_tasks()[0].evaluator
            assert evaluator is not None
            solved = evaluator.solved(states, None)
            print("SEARCH_SKILL_RESULT",
                  args.seed,
                  len(states) - 1,
                  solved,
                  flush=True)
            assert solved
            return

        def step(action: Action) -> None:
            nonlocal state
            state = env.step(action)
            states.append(state)
            if state.get(env._ball, "z") < 0.30:
                raise RuntimeError("ball fell off")

        def switch(name: str, side: int) -> None:
            option = options[name].ground([env._robot, env._fans[side]],
                                          np.array([0.06, 0.105],
                                                   dtype=np.float32))
            assert option.initiable(state)
            for _ in range(200):
                if option.terminal(state):
                    break
                step(option.policy(state))
            else:
                raise RuntimeError("switch failed")
            print("SWITCH",
                  name,
                  side,
                  len(states) - 1,
                  state.get(env._ball, "x"),
                  state.get(env._ball, "y"),
                  state.get(env._fans[side], "is_on"),
                  flush=True)

        # Stop early to account for the physical switch stroke and coasting.
        for side, axis, target in [(0, "x", target_x), (2, "y", target_y)]:
            switch("SwitchOn", side)
            for _ in range(400):
                if state.get(env._ball, axis) >= target - 0.10:
                    break
                step(noop(state))
            switch("SwitchOff", side)
            for _ in range(70):
                step(noop(state))
            print("SETTLED",
                  axis,
                  target,
                  state.get(env._ball, axis),
                  flush=True)
        evaluator = env.get_test_tasks()[0].evaluator
        assert evaluator is not None
        print("SKILL_RESULT",
              args.seed,
              len(states) - 1,
              evaluator.solved(states, None),
              flush=True)
        assert evaluator.solved(states, None)
    finally:
        p.disconnect(env._physics_client_id)


if __name__ == "__main__":
    main()
