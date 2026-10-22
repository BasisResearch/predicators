"""The quasi-static settle of a state draw keeps what the robot holds in its
grasp."""
# pylint: disable=protected-access
import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action


def test_settling_a_draw_keeps_a_held_jug_in_the_gripper():
    """A rehearsal draw is settled before it runs; the settle holds the robot
    still, and with it the jug the robot holds, so a draw taken while carrying
    the jug starts with the jug where the draw put it."""
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = PyBulletBoilEnv(use_gui=False)
    try:
        options = {o.name: o for o in get_gt_options(env.get_name())}
        state = env.reset("train", 0)
        robot = next(o for o in state if o.type.name == "robot")
        jug = next(o for o in state if o.type.name == "jug")
        pick = options["PickJug"].ground([robot, jug],
                                         np.array([0.0], dtype=np.float32))
        assert pick.initiable(state)
        for _ in range(300):
            if pick.terminal(state):
                break
            state = env.step(pick.policy(state))
        assert state.get(jug, "is_held") > 0.5
        settled = env.settle_state(state)
        assert settled.get(jug, "is_held") > 0.5
        for feature in ("x", "y", "z"):
            assert abs(settled.get(jug, feature) -
                       state.get(jug, feature)) < 1e-6, feature
        # Engine steps that hold the arm still keep the jug in the gripper.
        held = env._held_obj_id
        assert held is not None
        before = np.array(
            p.getBasePositionAndOrientation(
                held, physicsClientId=env._physics_client_id)[0])
        hold = Action(np.asarray(settled.joint_positions, dtype=np.float32))
        for _ in range(5):
            env.step(hold)
        after = np.array(
            p.getBasePositionAndOrientation(
                held, physicsClientId=env._physics_client_id)[0])
        assert np.linalg.norm(after - before) < 0.005
    finally:
        p.disconnect(env._physics_client_id)
