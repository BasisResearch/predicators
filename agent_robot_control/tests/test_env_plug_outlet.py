"""Gate tests for the plug-outlet insertion domain (Section 4.3 of PLAN.md)."""
import numpy as np
import pybullet as p
import pytest

from predicators.envs.pybullet_plug_outlet import PyBulletPlugOutletEnv

from agent_robot_control.sim.ee_control import EEController
from agent_robot_control.tests.conftest import make_env


def _oracle_insert(env, lateral_offset: float = 0.0):
    """Grasp the plug, lift, align over the socket (with an optional lateral
    error), and lower. Returns (env, controller, total steps)."""
    s = env.reset("train", 0)
    ctl = EEController(env)
    steps = [0]
    inner = ctl.step_fn

    def counting(a):
        steps[0] += 1
        inner(a)

    ctl.step_fn = counting
    plug, outlet = env._plug, env._outlet
    px, py, pz = [s.get(plug, f) for f in "xyz"]
    ox, oy = s.get(outlet, "x"), s.get(outlet, "y")
    grasp_z = pz + 0.01
    ctl.move_to((px, py, pz + 0.12), gripper="open")
    ctl.move_to((px, py, grasp_z))
    ctl.move_to((px, py, grasp_z), gripper="close")
    assert ctl.is_holding(), "oracle failed to grasp the plug"
    ctl.move_to((px, py, grasp_z + 0.12))
    st = env._current_observation
    ee = ctl.ee_position()
    dpx, dpy = st.get(plug, "x") - ee[0], st.get(plug, "y") - ee[1]
    above = np.array([ox - dpx, oy - dpy, grasp_z + 0.10])
    ctl.move_to(above)
    # Ground-truth alignment correction (the oracle has perfect perception):
    # bring the hanging plug's true xy onto the socket centre, then add the
    # requested lateral error.
    for _ in range(3):
        st = env._current_observation
        err = np.array([ox - st.get(plug, "x"), oy - st.get(plug, "y"), 0.0])
        if np.linalg.norm(err[:2]) < 0.0005:
            break
        above = above + err
        ctl.move_to(above)
    above = above + np.array([lateral_offset, 0.0, 0.0])
    if lateral_offset:
        ctl.move_to(above)
    insert_z = env.outlet_top_z() - env.insertion_depth + \
        env.prong_tip_offset() + (grasp_z - pz) - 0.004
    fine = EEController(env, step_size=0.01)
    fine.step_fn = counting
    fine.finger_target = ctl.finger_target
    fine.move_to((above[0], above[1], insert_z), max_steps=100)
    return env, ctl, steps[0]


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_oracle_inserts_at_tier(tier):
    PyBulletPlugOutletEnv.clearance = PyBulletPlugOutletEnv.clearance_tiers[tier]
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    env, ctl, steps = _oracle_insert(env)
    assert env.goal_reached(), f"oracle did not plug in at tier {tier}"
    assert steps < 200
    # Releasing keeps it plugged in.
    ctl.set_gripper("open")
    for _ in range(20):
        env.step(ctl.hold_action())
    assert env.goal_reached()
    PyBulletPlugOutletEnv.clearance = PyBulletPlugOutletEnv.clearance_tiers["medium"]


def test_offset_oracle_jams_on_rim():
    """An 8 mm lateral error at 2 mm clearance must not insert: bounded
    torques and the force limit make the prong jam on the rim instead of
    tunnelling. (Errors up to ~4 mm self-align through grasp compliance.)"""
    PyBulletPlugOutletEnv.clearance = PyBulletPlugOutletEnv.clearance_tiers["medium"]
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    env, ctl, _ = _oracle_insert(env, lateral_offset=0.008)
    assert not env.goal_reached()
    st = env._current_observation
    tip, _ = env.plug_tip_and_axis(st, env._plug)
    depth = env.outlet_top_z() - tip[2]
    assert depth < 0.008, f"prong penetrated {depth*1000:.1f} mm despite offset"


def test_plug_respawns_after_leaving_table():
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    s = env.reset("train", 0)
    ctl = EEController(env)
    holder_xy = (s.get(env._holder, "x"), s.get(env._holder, "y"))
    # Knock the plug off the table by teleporting it past the edge.
    p.resetBasePositionAndOrientation(env._plug_id, [1.35, 0.1, 0.25],
                                      [0, 0, 0, 1],
                                      physicsClientId=env._physics_client_id)
    for _ in range(env.topple_patience_steps + 30):
        s = env.step(ctl.hold_action())
    assert env.num_interventions == 1
    assert np.allclose([s.get(env._plug, "x"), s.get(env._plug, "y")],
                       holder_xy, atol=0.01)
    assert abs(s.get(env._plug, "z") - env.plug_rest_z()) < 0.01


def test_plugged_in_predicate_geometry():
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    s = env.reset("train", 0)
    plug, outlet = env._plug, env._outlet
    ox, oy = s.get(outlet, "x"), s.get(outlet, "y")
    good = s.copy()
    good.set(plug, "x", ox)
    good.set(plug, "y", oy)
    good.set(plug, "z", env.outlet_top_z() - 0.02 + env.prong_tip_offset())
    assert env._PluggedIn_holds(good, [plug, outlet])
    shallow = good.copy()
    shallow.set(plug, "z", env.outlet_top_z() - 0.005 + env.prong_tip_offset())
    assert not env._PluggedIn_holds(shallow, [plug, outlet])
    tilted = good.copy()
    tilted.set(plug, "roll", np.radians(20))
    assert not env._PluggedIn_holds(tilted, [plug, outlet])
    off = good.copy()
    off.set(plug, "x", ox + 0.01)
    assert not env._PluggedIn_holds(off, [plug, outlet])
