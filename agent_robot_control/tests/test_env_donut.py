"""Donut push domain: ungraspable discs, target at the table's far edge."""
import numpy as np
import pytest

from predicators.envs.pybullet_donut import PyBulletDonutEnv
from predicators.structs import Action

from agent_robot_control.sim.ee_control import EEController
from agent_robot_control.tests.conftest import make_env


def test_discs_are_too_wide_to_grasp_and_solid():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    robot = env._pybullet_robot
    # Jaw span at full open, from the finger link positions.
    from predicators.pybullet_helpers.link import get_link_state
    cid = env._physics_client_id
    lf = get_link_state(robot.robot_id, robot.left_finger_id,
                        physics_client_id=cid).worldLinkFramePosition
    rf = get_link_state(robot.robot_id, robot.right_finger_id,
                        physics_client_id=cid).worldLinkFramePosition
    span = abs(lf[0] - rf[0])
    assert 2 * PyBulletDonutEnv.donut_radius > span, "disc must not fit the jaws"
    # Solid: the alias kept for extent must equal the real radius, and the
    # body is a cylinder rather than a torus mesh.
    assert PyBulletDonutEnv.donut_major_radius == PyBulletDonutEnv.donut_radius
    assert hasattr(PyBulletDonutEnv, "_create_pybullet_disc")
    assert not hasattr(PyBulletDonutEnv, "_create_pybullet_donut")


def test_target_sits_at_the_far_edge_of_the_table():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    s = env.reset("train", 0)
    disc, target = env._donuts[0], env._target
    push = float(s.get(target, "y")) - float(s.get(disc, "y"))
    assert push > 0.7, f"push distance only {push:.2f} m"
    # Both ends inside the lane the arm can actually reach.
    assert abs(float(s.get(disc, "x")) - env.push_lane_x) < 0.05
    assert abs(float(s.get(target, "x")) - env.push_lane_x) < 0.05
    # One slab, and the target square is fully on it: the far edge of the
    # 0.9 m URDF box centred at _table_pose is the end of the table.
    assert not hasattr(env, "_table_id2")
    table_far_y = PyBulletDonutEnv._table_pose[1] + 0.45
    ty = float(s.get(target, "y"))
    assert ty + env.target_width / 2 < table_far_y
    assert table_far_y - (ty + env.target_width / 2) < 0.03, "not at the edge"
    assert not env.goal_reached()


def test_every_disc_is_on_the_table_from_the_start():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    s = env.reset("train", 0)
    assert not hasattr(PyBulletDonutEnv, "spawn_interval")
    assert PyBulletDonutEnv.num_donuts == 4
    resting_z = env.table_height + env.donut_half_height
    for i, donut in enumerate(env._donuts):
        assert not env._is_out_of_view(env._donut_ids[i]), donut.name
        assert float(s.get(donut, "z")) == pytest.approx(resting_z, abs=1e-3)
    # The three distractors are in the lane, between the disc and the target.
    for donut in env._donuts[1:]:
        y = float(s.get(donut, "y"))
        assert env.lane_y_lb - 0.01 <= y <= env.lane_y_ub + 0.01, y
    # Nothing new arrives, ever.
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    for _ in range(600):
        env.step(hold)
    live = sum(not env._is_out_of_view(d) for d in env._donut_ids)
    assert live == env.num_donuts, "a disc left the table on its own"


def test_discs_can_never_be_grasped():
    """The pick is not merely hard, it is not in the simulator.

    In sweep 2 the base class's pinch test fired on these discs in eight of
    nine runs despite the jaws being too narrow to span one, so the domain
    now declines to offer any graspable body at all.
    """
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    env.reset("train", 0)
    assert env._get_object_ids_for_held_check() == []
    ctl = EEController(env)
    disc = env._donuts[0]
    s = env._current_observation
    dx, dy = float(s.get(disc, "x")), float(s.get(disc, "y"))
    top = env.table_height + 2 * env.donut_half_height
    # Close the jaws straight onto the disc, from directly above.
    ctl.move_to((dx, dy, top + 0.12), gripper="open")
    ctl.move_to((dx, dy, top - 0.005))
    ctl.move_to((dx, dy, top - 0.005), gripper="close")
    ctl.move_to((dx, dy, top + 0.15))
    s = env._current_observation
    assert env._held_obj_id is None
    assert float(s.get(disc, "is_held")) == 0.0
    assert float(s.get(disc, "z")) < env.table_height + 0.05, "disc was lifted"


def test_goal_needs_a_resting_disc_in_the_target():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    s = env.reset("train", 0)
    disc, target = env._donuts[0], env._target
    tx, ty, tz = [float(s.get(target, f)) for f in "xyz"]
    inside = s.copy()
    inside.set(disc, "x", tx)
    inside.set(disc, "y", ty)
    inside.set(disc, "z", env.table_height + env.donut_half_height)
    assert env._InTarget_holds(inside, [disc, target])
    hovering = inside.copy()
    hovering.set(disc, "z", tz + 0.1)
    assert not env._InTarget_holds(hovering, [disc, target])
    outside = inside.copy()
    outside.set(disc, "y", ty - 0.12)
    assert not env._InTarget_holds(outside, [disc, target])
