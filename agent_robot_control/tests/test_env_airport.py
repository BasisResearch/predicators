"""Airport: looping belt, OnTable goal, button/pusher route."""
import numpy as np
import pytest
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


def test_button_press_extends_pusher_after_the_lag():
    env = make_env("pybullet_airport", PyBulletAirportEnv)
    env.reset("train", 0)
    ctl = EEController(env)
    bx, by, bz = env.button_stand_x, env.button_stand_y, \
        env.button_stand_z + env.button_height
    ctl.move_to((bx, by, bz + 0.15), gripper="close")
    ctl.move_to((bx, by, bz - 0.03), max_steps=60)
    s = env._current_observation
    assert s.get(env._button, "is_pressed") > 0.5
    # Nothing moves during the lag, then the pusher sweeps.
    y_before = float(s.get(env._pusher, "y"))
    for _ in range(env.pusher_delay_steps - 2):
        s = env.step(ctl.hold_action())
    assert float(s.get(env._pusher, "y")) == pytest.approx(y_before, abs=1e-6)
    for _ in range(300):
        s = env.step(ctl.hold_action())
    assert s.get(env._pusher, "y") > env.pusher_init_y + 0.3


def test_the_pusher_is_small_late_and_fast():
    """The three sweep-3 numbers, pinned, with what each one actually did.

    Doubling the delay and the speed cancel for the moment the pusher reaches
    the belt (20 + 40 == 40 + 20 steps), and the faster paddle imparts more
    impulse, so on their own they measured as a WIDER press window: 14 belt
    steps in sweep 2, 22 after. Halving the paddle's length is what bought
    precision, taking the window to 6 steps (measured 2026-09-09).
    """
    assert PyBulletAirportEnv.pusher_delay_steps == 40
    assert PyBulletAirportEnv.pusher_speed == 0.02
    assert PyBulletAirportEnv.pusher_length == 0.15
    # The paddle's x extent is the timing tolerance: an item is swept only
    # while its centre is within this of the pusher, and it covers that at
    # belt_speed per step.
    reach = PyBulletAirportEnv.pusher_length / 2.0 + 0.03
    window_steps = 2 * reach / PyBulletAirportEnv.belt_speed
    assert window_steps == pytest.approx(21.0), window_steps
    env = make_env("pybullet_airport", PyBulletAirportEnv)
    env.reset("train", 0)
    ctl = EEController(env)
    bx, by, bz = env.button_stand_x, env.button_stand_y, \
        env.button_stand_z + env.button_height
    ctl.move_to((bx, by, bz + 0.15), gripper="close")
    ctl.move_to((bx, by, bz - 0.03), max_steps=60)
    assert env._current_observation.get(env._button, "is_pressed") > 0.5
    ys = []
    for _ in range(env.pusher_delay_steps + 40):
        s = env.step(ctl.hold_action())
        ys.append(float(s.get(env._pusher, "y")))
    moved = np.diff(ys)
    moving = moved[moved > 1e-9]
    assert len(moving) > 0, "the pusher never moved"
    assert moving.max() == pytest.approx(PyBulletAirportEnv.pusher_speed,
                                         abs=1e-6)
    # The full 0.6 m stroke in 30 steps, not 60.
    stroke = env.table_y - env.table_width / 2.0 - env.pusher_init_y
    assert stroke / PyBulletAirportEnv.pusher_speed == pytest.approx(30.0)


def test_the_belt_is_out_of_reach_and_the_button_is_not():
    """The button is the only way in.

    Sweep 3 measured the shortcut: all three airport agents ignored the
    button, reached over the belt at roll -60..-75 deg and took the cube by
    hand, 21 to 39 reaches into the belt lane against 1 to 3 presses. Belt,
    pusher and table were then shifted 0.30 m away from the arm. The arm's
    envelope ends between 1.00 and 1.03 m; the nearest item now sits 1.30 m
    out. This pins that, through the public API an agent actually calls, at
    the orientations the agents actually used.
    """
    env = make_env("pybullet_airport", PyBulletAirportEnv)
    s = env.reset("train", 2)
    ctl = EEController(env)
    base = PyBulletAirportEnv.robot_base_pos
    item_z = PyBulletAirportEnv.conveyor_height + 0.03
    # Straight in front of the base is the closest the belt ever comes.
    for x in (base[0], PyBulletAirportEnv.conveyor_x):
        target = (x, PyBulletAirportEnv.conveyor_y, item_z)
        dist = np.hypot(target[0] - base[0], target[1] - base[1])
        assert dist > 1.15, f"belt only {dist:.2f} m from the base"
        for roll, pitch in ((0, 0), (-45, 0), (-75, 0), (-90, 0), (-60, -45)):
            res = ctl.move_to(target, ctl.quat_from_rpy_deg(roll, pitch, 0),
                              max_steps=40)
            assert res.steps == 0, \
                f"arm moved toward the belt at rpy ({roll},{pitch},0)"
            assert res.extra.get("refused"), res.summary()
    # The button is still comfortably in reach, or the task is impossible.
    bx, by = PyBulletAirportEnv.button_stand_x, PyBulletAirportEnv.button_stand_y
    bz = PyBulletAirportEnv.button_stand_z + PyBulletAirportEnv.button_height
    reached = ctl.move_to((bx, by, bz + 0.10), gripper="close", max_steps=200)
    assert reached.reached, reached.summary()


def test_pusher_obeys_the_button_only_after_the_delay():
    """The pusher sees the button as it was pusher_delay_steps ago."""
    env = make_env("pybullet_airport", PyBulletAirportEnv)
    env.reset("train", 0)
    assert env.pusher_delay_steps > 0
    for _ in range(env.pusher_delay_steps):
        assert env._delayed_button_state(True) is False, "acted before the lag"
    assert env._delayed_button_state(True) is True
    for _ in range(env.pusher_delay_steps):
        assert env._delayed_button_state(False) is True, "stopped too early"
    assert env._delayed_button_state(False) is False


def test_timed_button_press_puts_goal_item_on_table():
    """Oracle for the button route. The working press lead is 0.62-0.67 m of
    belt travel, six belt steps wide (gate measurement, 2026-09-09). The lead
    barely moved from sweep 2 -- the 40-step lag plus the 20 steps the faster
    pusher takes to reach the belt is the same 60 steps as before -- but the
    half-length paddle cut the window from 14 steps to 6."""
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
    for _ in range(900):
        dx = float(env._current_observation.get(item, "x")) - env.pusher_init_x
        if -0.655 < dx < -0.645:  # mid-window; the window is only 6 cm wide
            break
        s.step(ctl.hold_action())
    r = ctl.move_to((bx, by, bz - 0.01), q, max_steps=20)
    assert env._current_observation.get(env._button, "is_pressed") > 0.5, r.summary()
    for _ in range(400):
        s.step(ctl.hold_action())
        if env.goal_reached():
            break
    assert env.goal_reached()
    assert s.interactions < 1200
