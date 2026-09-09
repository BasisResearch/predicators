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
