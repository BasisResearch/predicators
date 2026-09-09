"""Gate tests for the three-leg plug domain (PLAN.md section 4.3).

The clearance the sweep runs at is whatever the ground-truth oracle can insert
reliably; these tests pin that behaviour so a geometry change that quietly
makes the task impossible fails here instead of in a sweep.

They also pin the slant: the outlet is pitched and yawed and the plug starts
off-square, so the sweep-2 solution -- grasp square, lower straight down --
has to fail, and does, in test_holding_the_plug_upright_defeats_insertion.
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
    # Measured down the outlet's own axis, since the plate is not level: the
    # plug frame rides at this local z, the end effector 4 mm above it, and
    # the pads hang about 32 mm below that.
    half_h = PyBulletPlugOutletEnv.outlet_height / 2.0
    plug_local_z = half_h - env.insertion_depth + env.prong_tip_offset()
    assert plug_local_z + 0.004 - 0.032 > half_h


def test_the_outlet_is_slanted_and_the_plug_starts_off_square():
    """No task can be solved by lowering a squarely-grasped plug.

    Sweep 2 shipped both bodies at yaw 0 on a level table, so the three-leg
    yaw constraint cost the agents nothing: none of the nine runs issued a
    rotation command. The plate is now pitched and yawed, and the plug's
    collar is turned away from it.
    """
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    assert PyBulletPlugOutletEnv.outlet_tilt_deg > \
        PyBulletPlugOutletEnv.alignment_max_deg, "tilt must break alignment"
    for task_idx in range(3):
        s = env.reset("train", task_idx)
        outlet, plug, holder = env._outlet, env._plug, env._holder
        assert float(s.get(outlet, "pitch")) == pytest.approx(
            np.radians(PyBulletPlugOutletEnv.outlet_tilt_deg))
        rel = abs(np.degrees(float(s.get(plug, "yaw")) -
                             float(s.get(outlet, "yaw"))))
        assert rel >= PyBulletPlugOutletEnv.min_yaw_offset_deg
        assert rel > PyBulletPlugOutletEnv.yaw_max_deg, "yaw is free again"
        # The collar turns with the plug it holds.
        assert float(s.get(holder, "yaw")) == pytest.approx(
            float(s.get(plug, "yaw")))
        # The plate rests on the table rather than sinking into it.
        o_pos, o_rot = PyBulletPlugOutletEnv.outlet_frame(s, outlet)
        a = PyBulletPlugOutletEnv.outlet_half_xy
        hz = PyBulletPlugOutletEnv.outlet_height / 2.0
        corners = [o_pos + o_rot @ np.array([sx * a, sy * a, sz * hz])
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        assert min(c[2] for c in corners) == pytest.approx(
            PyBulletPlugOutletEnv.table_height, abs=1e-6)


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


def test_holding_the_plug_upright_defeats_insertion():
    """The sweep-2 strategy, run against the slanted outlet.

    Cancelling the tilt is exactly what an agent that never rotates does:
    the plug arrives vertical, out of square with the plate by the full
    tilt, and jams on the rim.
    """
    res = run_oracle(DEFAULT_CLEARANCE, 0,
                     tilt_offset_deg=-PyBulletPlugOutletEnv.outlet_tilt_deg)
    PyBulletPlugOutletEnv.clearance = DEFAULT_CLEARANCE
    assert not res["plugged"]
    assert res["min_depth"] < PyBulletPlugOutletEnv.insertion_depth
    assert res["tilt_deg"] > PyBulletPlugOutletEnv.alignment_max_deg


def _seat_plug(env, s, depth: float = 0.014, lateral: float = 0.0,
               extra_yaw_deg: float = 0.0, upright: bool = False):
    """A state with the plug placed in the outlet's frame.

    ``depth`` is how far the blade tips sit below the top face, ``lateral``
    shifts along the outlet's local x, ``extra_yaw_deg`` turns the plug about
    the outlet's own axis, and ``upright`` holds it vertical instead.
    """
    from scipy.spatial.transform import Rotation
    plug, outlet = env._plug, env._outlet
    o_pos, o_rot = PyBulletPlugOutletEnv.outlet_frame(s, outlet)
    half_h = PyBulletPlugOutletEnv.outlet_height / 2.0
    z_local = half_h - depth + env.prong_tip_offset()
    pos = o_pos + o_rot @ np.array([lateral, 0.0, z_local])
    if upright:
        rot = Rotation.from_euler("z", float(s.get(outlet, "yaw")))
    else:
        rot = Rotation.from_matrix(o_rot) * Rotation.from_euler(
            "z", extra_yaw_deg, degrees=True)
    roll, pitch, yaw = rot.as_euler("xyz")
    out = s.copy()
    for feat, val in zip(("x", "y", "z", "roll", "pitch", "yaw"),
                         (*pos, roll, pitch, yaw)):
        out.set(plug, feat, float(val))
    return out


def test_plugged_in_predicate_geometry():
    """The goal is measured in the outlet's frame, not the world's."""
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    s = env.reset("train", 0)
    plug, outlet = env._plug, env._outlet
    seated = _seat_plug(env, s)
    assert env._PluggedIn_holds(seated, [plug, outlet])
    shallow = _seat_plug(env, s, depth=0.004)
    assert not env._PluggedIn_holds(shallow, [plug, outlet])
    yawed = _seat_plug(env, s, extra_yaw_deg=15.0)
    assert not env._PluggedIn_holds(yawed, [plug, outlet])
    off = _seat_plug(env, s, lateral=0.01)
    assert not env._PluggedIn_holds(off, [plug, outlet])
    # A plug held vertical over a slanted outlet is out of square by the
    # full tilt, however deep it is driven.
    upright = _seat_plug(env, s, upright=True)
    assert not env._PluggedIn_holds(upright, [plug, outlet])


def test_plug_respawns_into_the_collar_it_came_from():
    env = make_env("pybullet_plug_outlet", PyBulletPlugOutletEnv)
    s = env.reset("train", 0)
    ctl = EEController(env)
    holder = env._holder
    holder_xy = (s.get(holder, "x"), s.get(holder, "y"))
    holder_yaw = float(s.get(holder, "yaw"))
    p.resetBasePositionAndOrientation(env._plug_id, [1.35, 0.1, 0.25],
                                      [0, 0, 0, 1],
                                      physicsClientId=env._physics_client_id)
    for _ in range(env.topple_patience_steps + 30):
        s = env.step(ctl.hold_action())
    assert env.num_interventions == 1
    assert np.allclose([s.get(env._plug, "x"), s.get(env._plug, "y")],
                       holder_xy, atol=0.01)
    # Turned to the collar, not to yaw 0: a square respawn would hand the
    # agent the start the slant is meant to deny it.
    assert float(s.get(env._plug, "yaw")) == pytest.approx(holder_yaw,
                                                           abs=0.02)
