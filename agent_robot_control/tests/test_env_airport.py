"""Airport: looping belt, OnTable goal, button/pusher route."""
import numpy as np
import pybullet as p

from predicators.envs.pybullet_airport import PyBulletAirportEnv
from predicators.structs import Action

from agent_robot_control.sim.ee_control import EEController
from agent_robot_control.tests.conftest import make_env


def _hold(env):
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def test_belt_loops_and_goal_is_on_table():
    env = make_env("pybullet_airport", PyBulletAirportEnv)
    s = env.reset("train", 2)
    task = env.get_train_tasks()[2]
    assert len(task.goal) == 1
    atom = next(iter(task.goal))
    assert atom.predicate.name == "OnTable"
    assert atom.objects[0].name == "item_2"
    assert not env.goal_reached()
    items = env._items[:5]
    xs = []
    for _ in range(400):
        s = env.step(_hold(env))
        xs.append([float(s.get(it, "x")) for it in items])
    xs = np.array(xs)
    assert (np.diff(xs, axis=0) < -0.5).sum() >= 5, "items never wrapped"
    assert all(env._OnConveyor_holds(s, [it, env._conveyor]) for it in items)
    # Teleporting the goal item onto the table satisfies the goal.
    goal_item = atom.objects[0]
    p.resetBasePositionAndOrientation(
        goal_item.id, [env.table_x, env.table_y, env.table_height + 0.03],
        [0, 0, 0, 1], physicsClientId=env._physics_client_id)
    for _ in range(10):
        env.step(_hold(env))
    assert env.goal_reached()


def test_button_press_extends_pusher_and_pushes_an_item():
    env = make_env("pybullet_airport", PyBulletAirportEnv)
    env.reset("train", 0)
    ctl = EEController(env)
    bx, by, bz = env.button_stand_x, env.button_stand_y, \
        env.button_stand_z + env.button_height
    ctl.move_to((bx, by, bz + 0.15), gripper="close")
    ctl.move_to((bx, by, bz - 0.03), max_steps=60)
    s = env._current_observation
    assert s.get(env._button, "is_pressed") > 0.5
    for _ in range(300):
        s = env.step(ctl.hold_action())
    assert s.get(env._pusher, "y") > env.pusher_init_y + 0.3
    on_table = [it for it in env._items[:5]
                if env._OnTable_holds(s, [it, env._table])]
    assert on_table, "pusher did not push any item onto the table"


def test_timed_button_press_puts_goal_item_on_table():
    """Oracle for the button route: press when item_2 is ~0.42 m upstream of
    the pusher (the pusher only shoves items during its sweep across the
    belt). Gates that the task is feasible under the force-limited controller."""
    from agent_robot_control.sim.session import SessionConfig, SimSession
    s = SimSession(SessionConfig(env_name="pybullet_airport", task_idx=2,
                                 interaction_cap=3000, camera_width=224,
                                 camera_height=126, log_transitions=False))
    ctl, env = s.controller, s.env
    q = ctl.quat_from_rpy_deg(0, 0, 0)
    item = env._items[2]
    bx, by, bz = env.button_stand_x, env.button_stand_y, \
        env.button_stand_z + env.button_height
    ctl.move_to((bx, by, bz + 0.10), q, gripper="close")
    for _ in range(600):
        dx = float(env._current_observation.get(item, "x")) - env.pusher_init_x
        if -0.44 < dx < -0.42:
            break
        s.step(ctl.hold_action())
    r = ctl.move_to((bx, by, bz - 0.01), q, max_steps=20)
    assert env._current_observation.get(env._button, "is_pressed") > 0.5, r.summary()
    for _ in range(250):
        s.step(ctl.hold_action())
        if env.goal_reached():
            break
    assert env.goal_reached()
    assert s.interactions < 600
