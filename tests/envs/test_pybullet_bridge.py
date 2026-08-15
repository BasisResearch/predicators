"""Unit tests for pybullet_bridge's glue / cure / weld machinery.

Covers the planner-backtrack invariant that the physical weld constraint
set always tracks the state's ``attached_*`` features through
``_set_state`` (fresh reset tears welds down, restoring a post-weld
state recreates them), plus glue application, cure ticking, and the
attachment latch.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

from predicators import utils
from predicators.structs import Action


@pytest.fixture(scope="module", name="env_and_task")
def _env_and_task():
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
    })
    from predicators.envs.pybullet_bridge import \
        PyBulletBridgeEnv  # pylint: disable=import-outside-toplevel
    env = PyBulletBridgeEnv(use_gui=False)
    task = env._generate_train_tasks()[0]
    return env, task


def _hold_action(env):
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def test_glue_cure_weld_lifecycle(env_and_task):
    """Wet a face, cure a stacked joint, latch + weld, then check the weld set
    tracks _set_state in both directions."""
    env, task = env_and_task
    env._set_state(task.init)
    state = env._get_state()
    legs = sorted((b for b in state.get_objects(env._block_type)
                   if b.name.startswith("leg")),
                  key=lambda b: b.name)
    leg0, leg1 = legs[0], legs[1]

    # 1. Glue application: hold the bottle with its tip at leg0's
    # world-top dab point. Blocks are ONE shape; a standing leg's
    # world-top face is its local ``end_b`` face (local +x up).
    s = state.copy()
    dab = env._face_dab_point(s, leg0, "end_b")
    s.set(env._bottle, "x", dab[0])
    s.set(env._bottle, "y", dab[1])
    s.set(env._bottle, "z", dab[2] + env.bottle_half_extents[2])
    s.set(env._bottle, "is_held", 1.0)
    s.set(env._robot, "x", dab[0])
    s.set(env._robot, "y", dab[1])
    s.set(env._robot, "z", dab[2] + 2 * env.bottle_half_extents[2] + 0.005)
    s.set(env._robot, "fingers", env.closed_fingers)
    env._set_state(s)
    env.step(_hold_action(env))
    s2 = env._get_state()
    assert s2.get(leg0, "glue_end_b") > 0.5

    # 2. Curing: stack leg1 (also standing, so its world-bottom is its
    # local ``end_a`` face) on wet-topped leg0, bottle back down.
    s3 = s2.copy()
    for feat in ("x", "y", "z"):
        s3.set(env._bottle, feat, state.get(env._bottle, feat))
    s3.set(env._bottle, "is_held", 0.0)
    s3.set(env._robot, "x", env.robot_init_x)
    s3.set(env._robot, "y", env.robot_init_y)
    s3.set(env._robot, "z", env.robot_init_z)
    s3.set(env._robot, "fingers", env.open_fingers)
    s3.set(leg1, "x", s3.get(leg0, "x"))
    s3.set(leg1, "y", s3.get(leg0, "y"))
    s3.set(leg1, "z", s3.get(leg0, "z") + 2 * env.leg_half_extents[2])
    env._set_state(s3)
    for _ in range(env.cure_threshold + 5):
        env.step(_hold_action(env))

    final = env._get_state()
    assert final.get(leg0,
                     "attached_end_b") == float(env._block_index[leg1.name])
    assert final.get(leg1,
                     "attached_end_a") == float(env._block_index[leg0.name])
    assert final.get(leg0, "glue_end_b") < 0.5  # consumed
    assert len(env._weld_constraints) == 1
    assert env._Attached_holds(final, [leg0, leg1])
    assert env._Attached_holds(final, [leg1, leg0])
    assert not env._Loose_holds(final, [leg0])
    assert env.get_welded_partner_ids(leg0.id) == {leg1.id}

    # 3. Fresh reset removes the weld and restores default features.
    env._set_state(task.init)
    assert len(env._weld_constraints) == 0
    fresh = env._get_state()
    assert fresh.get(leg0, "attached_end_b") == -1.0
    assert fresh.get(leg0, "glue_end_b") == 0.0

    # 4. Restoring the post-weld state recreates the weld (the planner
    # backtrack invariant).
    env._set_state(final)
    assert len(env._weld_constraints) == 1


def test_seat_weld_holds_pose(env_and_task):
    """A cured seat joint (lying span welded onto a STANDING leg's top) must
    hold the assembly rigidly at the seated pose.

    Regression: the weld snap (zero relative roll/pitch, ideal dz) is
    a world-frame concept; applying it to the LOCAL relative transform
    was only correct for lying parents, and a standing-parent weld
    re-posed the child by ~pi/2 -- the solver then hurled the whole
    assembly off the table at weld_max_force one step after latch.
    """
    env, task = env_and_task
    env._set_state(task.init)
    state = env._get_state()
    blocks = state.get_objects(env._block_type)
    leg = next(b for b in blocks if b.name == "leg0")
    span = next(b for b in blocks if b.name == "span0")

    s = state.copy()
    # Seat the span flat on the standing leg's top; wet the leg's
    # world-top face (its local end_b). Park everything else far away.
    s.set(span, "x", s.get(leg, "x"))
    s.set(span, "y", s.get(leg, "y"))
    s.set(span, "z",
          s.get(leg, "z") + env.leg_half_extents[2] +
          env.span_half_extents[2])
    for feat in ("roll", "pitch", "rot"):
        s.set(span, feat, 0.0)
    s.set(leg, "glue_end_b", 1.0)
    for i, blk in enumerate(blocks):
        if blk not in (leg, span):
            s.set(blk, "x", 2.0 + 0.2 * i)
            s.set(blk, "y", 2.0)
    env._set_state(s)

    for _ in range(env.cure_threshold + 5):
        env.step(_hold_action(env))
    latched = env._get_state()
    assert latched.get(leg,
                       "attached_end_b") == float(env._block_index[span.name])
    assert len(env._weld_constraints) == 1

    # The weld must HOLD the seated pose, not fight it.
    for _ in range(30):
        env.step(_hold_action(env))
    final = env._get_state()
    for obj in (leg, span):
        for feat in ("x", "y", "z"):
            assert abs(final.get(obj, feat) - latched.get(obj, feat)) < 0.01
    assert abs(final.get(leg, "pitch") + np.pi / 2) < 0.05
    assert abs(final.get(span, "pitch")) < 0.05
