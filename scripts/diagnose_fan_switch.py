"""Replay historical Fan switch approaches on a compute node only."""
import argparse
import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pybullet as p

from predicators.envs import create_new_env
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.skill_factories.base import PhaseSkill
from predicators.settings import CFG
from predicators.structs import Action
from scripts.replay_oracle_certificate import load_run_config

# Historical replay deliberately inspects private engine/controller state.
# pylint: disable=protected-access


def main() -> None:
    """Replay a saved switch decision and report physical diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--arm-from", type=Path)
    parser.add_argument("--arm-step", type=int)
    parser.add_argument("--validate-ik", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--trace-controller", action="store_true")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--direct-descent", action="store_true")
    args = parser.parse_args()
    load_run_config(args.run)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.validate_ik:
        CFG.pybullet_ik_validate = True
    if args.trace_controller or args.geometry or args.direct_descent:
        for method_name in ("_execute_move", "_execute_move_birrt",
                            "_execute_move_ik", "_plan_with_simulator"):
            original = getattr(PhaseSkill, method_name)

            def wrapped(self: Any,
                        *values: Any,
                        _original: Any = original,
                        _name: str = method_name,
                        **kwargs: Any) -> Any:
                if _name == "_plan_with_simulator":
                    result = _original(self, *values, **kwargs)
                    if args.geometry and values[2] == "Waypoint_1" and result:
                        sim = self._config.simulator

                        def snapshot(world: Any, joints: Any) -> dict:
                            robot = world._pybullet_robot
                            robot.set_joints(joints)
                            cid = world._physics_client_id
                            body = world._switches[1].id
                            return {
                                "base":
                                p.getBasePositionAndOrientation(
                                    body, physicsClientId=cid),
                                "joints": [
                                    p.getJointState(body,
                                                    i,
                                                    physicsClientId=cid)[0]
                                    for i in range(
                                        p.getNumJoints(body,
                                                       physicsClientId=cid))
                                ],
                                "closest": [(c[3], c[4], c[8])
                                            for c in p.getClosestPoints(
                                                robot.robot_id,
                                                body,
                                                .03,
                                                physicsClientId=cid)],
                            }

                        saved_world = p.saveState(
                            physicsClientId=env._physics_client_id)
                        try:
                            print(
                                "GEOMETRY " +
                                json.dumps({
                                    "real": snapshot(env, result[-1]),
                                    "planner": snapshot(sim, result[-1]),
                                    "goal_joints": result[-1],
                                }),
                                flush=True)
                        finally:
                            p.restoreState(
                                saved_world,
                                physicsClientId=env._physics_client_id)
                            p.removeState(
                                saved_world,
                                physicsClientId=env._physics_client_id)
                    print("CONTROLLER " + repr(
                        (_name, values[2],
                         None if result is None else len(result))),
                          flush=True)
                    return result
                phase = values[0]
                if args.direct_descent and phase.name == "Waypoint_1":
                    phase.direct_descend = True
                print("CONTROLLER " + repr(
                    (_name, phase.name, phase.use_motion_planning,
                     self._config.ik_validate)),
                      flush=True)
                if _name == "_execute_move":
                    state, memory = values[1], values[2]
                    trajectory = memory.get(f"birrt_traj_{id(phase)}")
                    print("TRACK " + json.dumps(
                        {
                            "phase": phase.name,
                            "joints": list(state.joint_positions),
                            "index": memory.get(f"birrt_step_{id(phase)}"),
                            "length": len(trajectory) if trajectory else None,
                        }),
                          flush=True)
                return _original(self, *values, **kwargs)

            setattr(PhaseSkill, method_name, wrapped)
    records = [
        json.loads(line)
        for line in (args.run / "L02/actions.jsonl").read_text().splitlines()
    ]
    actions = [
        Action(np.asarray(r["a"], dtype=np.float32)) for r in records
        if "a" in r and r["ep"] == 0
    ]
    with (args.run / "L02/episodes.pkl").open("rb") as stream:
        saved = pickle.load(stream)[0]["states"]
    for height in (None, 0.10, 0.105, 0.11):
        env = cast(PyBulletFanEnv,
                   create_new_env("pybullet_fan", do_cache=False))
        state = env.reset("test", 0)
        for action in actions[:args.step]:
            state = env.step(action)
        joint_error = float(
            np.max(
                np.abs(
                    np.asarray(state.joint_positions) -
                    np.asarray(saved[args.step].joint_positions))))
        if args.restore:
            env._set_state(state.copy())
            state = env._get_state()
        if args.arm_from:
            with (args.arm_from / "L02/episodes.pkl").open("rb") as stream:
                donor = pickle.load(stream)[0]["states"][args.arm_step]
            changed = state.copy()
            robot = next(o for o in changed if o.name == "robot")
            donor_robot = next(o for o in donor if o.name == "robot")
            changed.data[robot] = donor[donor_robot].copy()
            changed.simulator_state = copy.deepcopy(state.simulator_state)
            changed.simulator_state["joint_positions"] = list(
                donor.joint_positions)
            env._set_state(changed)
            state = env._get_state()
        objects = {o.name: o for o in state}
        result = {
            "run": str(args.run),
            "step": args.step,
            "height": height,
            "prefix_joint_error": joint_error,
            "restored": args.restore,
            "arm_from": str(args.arm_from),
            "validate_ik": CFG.pybullet_ik_validate
        }
        contacts = []

        def trace_contacts() -> None:
            # Called synchronously within this iteration, never retained.
            # pylint: disable=cell-var-from-loop
            rid = env._pybullet_robot.robot_id
            cid = env._physics_client_id
            for contact in p.getContactPoints(bodyA=rid, physicsClientId=cid):
                if contact[2] != env._switches[1].id:
                    continue
                link = contact[3]
                label = (p.getJointInfo(rid, link,
                                        physicsClientId=cid)[12].decode()
                         if link >= 0 else "base")
                contacts.append((count, label, contact[8]))

        count = 0
        try:
            if height is None:
                for action in actions[args.step:args.step + 58]:
                    state = env.step(action)
                    count += 1
                    trace_contacts()
                result["status"] = "recorded_actions"
            else:
                options = get_gt_options("pybullet_fan")
                parent = next(o for o in options if o.name == "SwitchOn")
                option = parent.ground([objects["robot"], objects["fan_1"]],
                                       np.array([0.05, height],
                                                dtype=np.float32))
                assert option.initiable(state)
                for _ in range(150):
                    if option.terminal(state):
                        result["status"] = "completed"
                        break
                    state = env.step(option.policy(state))
                    count += 1
                    trace_contacts()
                else:
                    result["status"] = "step_limit"
        except Exception as error:  # pylint: disable=broad-except
            # Diagnostic output, never acceptance.
            result["status"] = "error"
            result["error"] = str(error)
        result.update(steps=count,
                      fan_on=state.get(objects["fan_1"], "is_on"),
                      ball_x=state.get(objects["ball"], "x"))
        result["first_contacts"] = contacts[:4]
        result["deepest_contacts"] = sorted(contacts, key=lambda c: c[2])[:4]
        print("DIAGNOSTIC " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
