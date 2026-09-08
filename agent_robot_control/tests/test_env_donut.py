"""Donut push domain: ungraspable discs, long table, capped spawns."""
import numpy as np

from predicators.envs.pybullet_donut import PyBulletDonutEnv
from predicators.structs import Action

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


def test_long_table_puts_the_target_a_metre_away():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    s = env.reset("train", 0)
    disc, target = env._donuts[0], env._target
    push = float(s.get(target, "y")) - float(s.get(disc, "y"))
    assert push > 0.85, f"push distance only {push:.2f} m"
    # Both ends inside the lane the arm can actually reach.
    assert abs(float(s.get(disc, "x")) - env.push_lane_x) < 0.05
    assert abs(float(s.get(target, "x")) - env.push_lane_x) < 0.05
    assert env._table_id2 != env._table_id
    assert not env.goal_reached()


def test_spawns_are_capped_and_land_in_the_lane():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    env.reset("train", 0)
    assert PyBulletDonutEnv.spawn_interval == 40
    assert PyBulletDonutEnv.num_donuts == 4
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    max_live = 0
    for _ in range(600):
        s = env.step(hold)
        live = sum(not env._is_out_of_view(d) for d in env._donut_ids)
        max_live = max(max_live, live)
    assert max_live == env.num_donuts
    assert not env._is_out_of_view(env._donut_ids[0]), "goal disc was recycled"
    for i, d in enumerate(env._donut_ids[1:], start=1):
        if env._is_out_of_view(d):
            continue
        y = float(s.get(env._donuts[i], "y"))
        assert env.spawn_y_lb - 0.1 <= y <= env.spawn_y_ub + 0.2, y


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
