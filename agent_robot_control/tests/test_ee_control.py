"""Behaviour of the Cartesian controller the agents drive the arm with."""
import numpy as np

from agent_robot_control.experiments.domino_oracle import CFG_OVERRIDES
from agent_robot_control.sim.session import SessionConfig, SimSession


def _session():
    return SimSession(
        SessionConfig(env_name="pybullet_domino", task_idx=0, seed=2,
                      interaction_cap=5000, camera_width=64, camera_height=36,
                      log_transitions=False, cfg_overrides=CFG_OVERRIDES))


def test_unreachable_target_is_refused_without_moving():
    """``move_to`` is a pose command: with no joint solution it must move
    nothing rather than traverse as far along the line as it can.

    Before this check the arm walked the straight line waypoint by waypoint
    and only failed partway, sweeping whatever lay on the line - which on the
    domino domain dragged a staged block 32 cm and permanently failed the task
    (two of three no-particles runs died this way, 2026-09-09).
    """
    session = _session()
    env, ctl = session.env, session.controller
    q = ctl.quat_from_rpy_deg(0, 0, 0)
    dominoes = env._current_observation.get_objects(
        env._components[0]._domino_type)

    def poses():
        st = env._current_observation
        return {d.name: np.array([st.get(d, "x"), st.get(d, "y")])
                for d in dominoes}

    before = poses()
    used = session.interactions
    ee = ctl.ee_position().copy()
    # Well outside a fixed-base arm's envelope, on a line through the scene.
    res = ctl.move_to((0.05, 1.25, 0.45), q, max_steps=60)
    assert res.ik_failed and not res.reached
    assert res.steps == 0
    assert session.interactions == used
    assert np.linalg.norm(ctl.ee_position() - ee) < 1e-6
    for name, xy in poses().items():
        assert np.linalg.norm(xy - before[name]) < 1e-6, name
    assert "did not move" in res.message
    session.close()


def test_a_blocked_but_reachable_target_still_presses():
    """The refusal is kinematic only. A pose inside the table is solvable and
    must still be attempted, or press-into-contact moves (plug insertion, disc
    pushes) would stop working."""
    session = _session()
    ctl = session.controller
    q = ctl.quat_from_rpy_deg(0, 0, 0)
    ctl.move_to((0.60, 1.10, 0.55), q, gripper="close", max_steps=120)
    res = ctl.move_to((0.60, 1.10, 0.30), q, max_steps=80)
    assert not res.ik_failed
    assert res.steps > 0
    assert not res.reached  # the table is in the way
    session.close()


def test_a_branch_flip_does_not_wreck_the_arm():
    """The sweep-3 airport crash: a reachable pose, an unreachable path.

    Airport seed 0 asked for roll = -90 deg to reach past the belt. IK
    returned a solution in a different branch, one env step commanded it, and
    the hand went through the belt/table gap at 41,610 N; the arm never
    recovered and the run ended at 424 of 100,000 interactions. The pose
    passes the reachability pre-flight -- a joint configuration does reach it
    -- so only a continuity check catches this.

    Recovery is route-dependent, and the test says so: rotating home works
    from the pose the sweep ends at, but at (1.5, 1.0, 0.60) the next
    increment has no IK solution at all and is refused outright. Translating
    first is the way back, which is what an agent has to do.
    """
    from agent_robot_control.sim.session import SessionConfig, SimSession
    session = SimSession(SessionConfig(env_name="pybullet_airport",
                                       task_idx=2, seed=0,
                                       interaction_cap=10000,
                                       camera_width=224, camera_height=126,
                                       log_transitions=False))
    ctl = session.controller
    sweep_pos = (1.5, 0.9, 0.55)
    peak = 0.0
    for roll in (0.0, -30.0, -60.0, -90.0):
        ctl.move_to(sweep_pos, ctl.quat_from_rpy_deg(roll, 0, 0),
                    gripper="close", max_steps=120)
        peak = max(peak, ctl.contact_force()[0])
    # Whatever the arm did, it never generated the crash's forces.
    assert peak < 5000.0, f"peak contact force {peak:.0f} N"
    assert ctl.rpy_relative_to_home_deg()[0] < -60.0, "never got near -90"
    # Still controllable: translation at the orientation it already holds
    # tracks normally, there and back.
    out = ctl.move_to((1.5, 1.0, 0.60), max_steps=200)
    assert out.reached, out.summary()
    assert not out.extra["configuration_jump"], out.summary()
    back = ctl.move_to(sweep_pos, max_steps=200)
    assert back.reached, back.summary()
    # And not stuck: from here one command rotates all the way home.
    home = ctl.move_to(ctl.ee_position(), ctl.quat_from_rpy_deg(0, 0, 0),
                       max_steps=300)
    assert abs(ctl.rpy_relative_to_home_deg()[0]) < 10.0, home.summary()
    assert ctl.contact_force()[0] < 100.0


def test_every_move_result_reports_the_jump_flag():
    """Both of move_to's return paths carry configuration_jump.

    The reachability refusal builds its own extra dict, so the flag was
    missing there and an unguarded read raised KeyError.
    """
    from agent_robot_control.sim.session import SessionConfig, SimSession
    session = SimSession(SessionConfig(env_name="pybullet_donut", task_idx=0,
                                       seed=0, interaction_cap=10000,
                                       camera_width=224, camera_height=126,
                                       log_transitions=False))
    ctl = session.controller
    refused = ctl.move_to((5.0, 5.0, 5.0))  # far out of reach
    assert refused.extra["refused"] is True
    assert refused.extra["configuration_jump"] is False
    assert refused.steps == 0
    ordinary = ctl.move_to((1.20, 0.60, 0.40))
    assert ordinary.extra["configuration_jump"] is False


def test_the_guard_lets_ordinary_moves_through():
    """The continuity guard must not refuse normal tracking.

    A straight Cartesian move turns each joint by a few hundredths of a
    radian per step, two orders below the 0.35 rad threshold, so an ordinary
    reach reports no configuration jump and still reaches its target.
    """
    from agent_robot_control.sim.session import SessionConfig, SimSession
    session = SimSession(SessionConfig(env_name="pybullet_donut", task_idx=0,
                                       seed=0, interaction_cap=10000,
                                       camera_width=224, camera_height=126,
                                       log_transitions=False))
    ctl = session.controller
    res = ctl.move_to((1.20, 0.60, 0.40), ctl.quat_from_rpy_deg(0, 0, 0))
    assert not res.extra["configuration_jump"], res.summary()
    assert res.reached, res.summary()
