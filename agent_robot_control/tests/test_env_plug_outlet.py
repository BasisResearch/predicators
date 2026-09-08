"""Gate tests for the three-leg plug domain (PLAN.md section 4.3).

The clearance the sweep runs at is whatever the ground-truth oracle can insert
reliably; these tests pin that behaviour so a geometry change that quietly
makes the task impossible fails here instead of in a sweep.
"""
import numpy as np
import pybullet as p
import pytest

from predicators.envs.pybullet_plug_outlet import PyBulletPlugOutletEnv

from agent_robot_control.experiments.plug_oracle import run_oracle
from agent_robot_control.sim.ee_control import EEController
from agent_robot_control.tests.conftest import make_env

DEFAULT_CLEARANCE = PyBulletPlugOutletEnv.clearance


def test_geometry_is_three_legged_and_graspable():
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    legs = PyBulletPlugOutletEnv.leg_specs()
    assert [name for name, *_ in legs] == ["blade_left", "blade_right",
                                           "ground_pin"]
    # Blades side by side, ground pin offset in y: that is what fixes yaw.
    (_, (lx, ly), _, _), (_, (rx, _), _, _), (_, (px, py), _, _) = legs
    assert lx < 0 < rx and px == pytest.approx(0.0) and py < ly
    # The grip block must fit the jaws (about 70 mm) and the plate must be
    # decomposed into boxes leaving three holes.
    assert 2 * PyBulletPlugOutletEnv.plug_block_half[0] < 0.07
    assert len(PyBulletPlugOutletEnv._outlet_boxes()) == 8
    # Legs long enough that the finger pads clear the plate at full depth.
    plug_z = (env.outlet_top_z() - env.insertion_depth
              + env.prong_tip_offset())
    assert plug_z + 0.004 - 0.032 > env.outlet_top_z()


def test_oracle_inserts_at_the_configured_clearance():
    res = [run_oracle(DEFAULT_CLEARANCE, seed) for seed in range(3)]
    PyBulletPlugOutletEnv.clearance = DEFAULT_CLEARANCE
    assert sum(r["plugged"] for r in res) >= 2, res
    best = max(res, key=lambda r: r["min_depth"])
    assert best["min_depth"] >= PyBulletPlugOutletEnv.insertion_depth
    assert best["contact_force"] < 200.0  # a clean seat, not a jam


def test_yaw_error_defeats_insertion():
    """Three legs constrain rotation: a plug 15 degrees off square cannot enter,
    which is the precision a single prong never demanded."""
    res = run_oracle(DEFAULT_CLEARANCE, 0, yaw_offset_deg=15.0)
    PyBulletPlugOutletEnv.clearance = DEFAULT_CLEARANCE
    assert not res["plugged"]
    assert res["min_depth"] < PyBulletPlugOutletEnv.insertion_depth


def test_lateral_error_defeats_insertion():
    res = run_oracle(DEFAULT_CLEARANCE, 0, lateral_offset=0.008)
    PyBulletPlugOutletEnv.clearance = DEFAULT_CLEARANCE
    assert not res["plugged"]


def test_plugged_in_predicate_geometry():
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    s = env.reset("train", 0)
    plug, outlet = env._plug, env._outlet
    ox, oy = s.get(outlet, "x"), s.get(outlet, "y")
    seated = s.copy()
    seated.set(plug, "x", ox)
    seated.set(plug, "y", oy)
    seated.set(plug, "z",
               env.outlet_top_z() - 0.014 + env.prong_tip_offset())
    assert env._PluggedIn_holds(seated, [plug, outlet])
    shallow = seated.copy()
    shallow.set(plug, "z", env.outlet_top_z() - 0.004 + env.prong_tip_offset())
    assert not env._PluggedIn_holds(shallow, [plug, outlet])
    yawed = seated.copy()
    yawed.set(plug, "yaw", np.radians(15))
    assert not env._PluggedIn_holds(yawed, [plug, outlet])
    tilted = seated.copy()
    tilted.set(plug, "roll", np.radians(20))
    assert not env._PluggedIn_holds(tilted, [plug, outlet])
    off = seated.copy()
    off.set(plug, "x", ox + 0.01)
    assert not env._PluggedIn_holds(off, [plug, outlet])


def test_plug_respawns_after_leaving_table():
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    s = env.reset("train", 0)
    ctl = EEController(env)
    holder_xy = (s.get(env._holder, "x"), s.get(env._holder, "y"))
    p.resetBasePositionAndOrientation(env._plug_id, [1.35, 0.1, 0.25],
                                      [0, 0, 0, 1],
                                      physicsClientId=env._physics_client_id)
    for _ in range(env.topple_patience_steps + 30):
        s = env.step(ctl.hold_action())
    assert env.num_interventions == 1
    assert np.allclose([s.get(env._plug, "x"), s.get(env._plug, "y")],
                       holder_xy, atol=0.01)
