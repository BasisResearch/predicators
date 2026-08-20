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
import pybullet as p
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
    # Wetting takes a SUSTAINED dwell: one in-range step only advances
    # the streak (partial <= 0.5 reads as dry), the wet_streak_steps-th
    # consecutive step latches the face wet. A one-step drive-by can
    # never glue.
    env.step(_hold_action(env))
    partial = env._get_state().get(leg0, "glue_end_b")
    assert 0.0 < partial <= 0.5
    for _ in range(env.wet_streak_steps - 1):
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
    # The ideal partner transform comes from the weld's SNAPPED frame:
    # leg1 stacked on standing leg0 sits one block-length up (0.1 m),
    # independent of any pendulum transient in the live poses.
    transforms = env.get_welded_partner_transforms(leg0.id)
    assert set(transforms) == {leg1.id}
    rel_pos, _ = transforms[leg1.id]
    assert abs(np.linalg.norm(rel_pos) - 2 * env.leg_half_extents[2]) < 0.02

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


def test_drive_by_graze_never_wets(env_and_task):
    """An interrupted dwell must NOT wet a face: the wet streak resets the
    moment the tip leaves the radius.

    Regression: wetting used to be instantaneous, so a one-step
    crossing of the apply radius (e.g. a bottle retreat clipping the
    sphere on its way up) could wet a face -- a step-phasing coin flip
    that let marginal glue targets validate in the sandbox and then
    miss in the real rollout.
    """
    env, task = env_and_task
    env._set_state(task.init)
    state = env._get_state()
    leg0 = next(b for b in state.get_objects(env._block_type)
                if b.name == "leg0")

    def _tip_at_dab(s):
        dab = env._face_dab_point(s, leg0, "end_b")
        s.set(env._bottle, "x", dab[0])
        s.set(env._bottle, "y", dab[1])
        s.set(env._bottle, "z", dab[2] + env.bottle_half_extents[2])
        s.set(env._bottle, "is_held", 1.0)
        s.set(env._robot, "x", dab[0])
        s.set(env._robot, "y", dab[1])
        s.set(env._robot, "z", dab[2] + 2 * env.bottle_half_extents[2] + 0.005)
        s.set(env._robot, "fingers", env.closed_fingers)
        return s

    # Two in-range steps: a partial streak, still dry.
    env._set_state(_tip_at_dab(state.copy()))
    env.step(_hold_action(env))
    env.step(_hold_action(env))
    s = env._get_state()
    assert 0.0 < s.get(leg0, "glue_end_b") <= 0.5
    # Leave the radius for one step: the streak resets to zero.
    s.set(env._bottle, "z", s.get(env._bottle, "z") + 0.1)
    s.set(env._robot, "z", s.get(env._robot, "z") + 0.1)
    env._set_state(s)
    env.step(_hold_action(env))
    s = env._get_state()
    assert s.get(leg0, "glue_end_b") == 0.0
    # A fresh sustained dwell still wets.
    env._set_state(_tip_at_dab(s))
    for _ in range(env.wet_streak_steps):
        env.step(_hold_action(env))
    assert env._get_state().get(leg0, "glue_end_b") > 0.5


def test_place_settles_to_contact():
    """Place must release at first contact instead of free-falling.

    With a release_z several mm above resting height, the settle phase
    lowers the block to the support before opening, so the block lands
    at resting height with no bounce spin. Regression coverage for two
    failure modes: the old open-loop drop bounced and slid (mm-scale
    scatter that flipped this domain's tight tolerances), and an early
    settle implementation took the whole stroke in one IK step whose
    wrist-flipped branch batted the released block across the table (~9
    cm slide, 0.16 rad spin).
    """
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": True,
        "pybullet_birrt_contact_margin": -0.005,
        "pybullet_birrt_path_subsample_ratio": 1,
    })
    from predicators.envs.pybullet_bridge import \
        PyBulletBridgeEnv  # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import \
        get_gt_options  # pylint: disable=import-outside-toplevel
    env = PyBulletBridgeEnv(use_gui=False)
    try:
        task = env._generate_train_tasks()[0]
        env._set_state(task.init)
        state = env._get_state()
        options = {o.name: o for o in get_gt_options(env.get_name())}
        span1 = next(b for b in env._blocks if b.name == "span1")

        def run_option(opt, objs, params):
            nonlocal state
            ground = opt.ground(objs, np.array(params, dtype=np.float32))
            assert ground.initiable(state)
            for _ in range(200):
                env.step(ground.policy(state))
                state = env._get_state()
                if ground.terminal(state):
                    return
            raise AssertionError(f"{opt.name} did not terminate")

        run_option(options["PickBlock"], [env._robot, span1], [0.002])
        resting_z = env.table_height + env.span_half_extents[2]
        tx, ty = 0.75, 1.25
        run_option(options["Place"], [env._robot],
                   [tx, ty, resting_z + 0.008, 0.0])
        # Landed at resting height (no residual drop), near the target,
        # without spinning. The 6 mm xy bound covers the verified
        # release: plant sag walks an unverified settle stroke ~15 mm
        # toward the robot base, and the verify-and-re-aim retry (see
        # create_place_skill's verify_xy_tol) is what keeps landings
        # within tolerance.
        assert abs(state.get(span1, "z") - resting_z) < 0.002
        assert abs(state.get(span1, "x") - tx) < 0.006
        assert abs(state.get(span1, "y") - ty) < 0.006
        assert abs(state.get(span1, "yaw")) < 0.03
    finally:
        p.disconnect(env._physics_client_id)


def test_sim_data_isolated_between_env_instances(env_and_task):
    """Glue/cure/attached written by one env instance must never leak into
    another env instance through shared State Object instances.

    Regression: those features live in ``Object.sim_data`` (stored on
    the instance), and states routinely cross env instances carrying
    the source env's objects (option-model resets, refinement
    rollouts). ``_set_domain_specific_state`` used to write through the
    incoming instances, so a sim rollout's glue/cure values bled into
    the real env's next observation (observed as impossible soft-cure
    floats in a real env's post-mortem state dump).
    """
    env, task = env_and_task
    env._set_state(task.init)
    src_state = env._get_state()
    leg0 = next(b for b in src_state.get_objects(env._block_type)
                if b.name == "leg0")
    assert src_state.get(leg0, "glue_end_b") == 0.0

    from predicators.envs.pybullet_bridge import \
        PyBulletBridgeEnv  # pylint: disable=import-outside-toplevel
    other = PyBulletBridgeEnv(use_gui=False)
    try:
        glued = src_state.copy()
        glued.set(leg0, "glue_end_b", 1.0)
        glued.set(leg0, "cure_end_b", 3.0)
        # The other env must import the features into its OWN blocks...
        other._set_state(glued)
        other_state = other._get_state()
        assert other_state.get(leg0, "glue_end_b") == 1.0
        assert other_state.get(leg0, "cure_end_b") == 3.0
        # ...without touching this env's blocks (src_state's Object
        # instances belong to ``env``).
        fresh = env._get_state()
        assert fresh.get(leg0, "glue_end_b") == 0.0
        assert fresh.get(leg0, "cure_end_b") == 0.0
    finally:
        p.disconnect(other._physics_client_id)


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
          s.get(leg, "z") + env.leg_half_extents[2] + env.span_half_extents[2])
    for feat in ("roll", "pitch", "yaw"):
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


def test_welded_pair_does_not_creep(env_and_task):
    """A freshly welded resting pair must stay put while the scene idles.

    Regression: a PyBullet JOINT_FIXED constraint between two
    table-resting bodies accumulates sub-mm error as each body settles
    into its own contact, and the correction impulses rectify into a
    steady skate -- 7-9 mm and up to 0.13 rad of yaw per 200 idle steps
    (unwelded pairs move < 1.5 mm), enough to invalidate every
    downstream open-loop placement parameter and bend every row. The
    quiescent re-anchoring in _relax_resting_welds must hold the pair
    still.
    """
    env, task = env_and_task
    env._set_state(task.init)
    state = env._get_state()
    blocks = state.get_objects(env._block_type)
    span0 = next(b for b in blocks if b.name == "span0")
    span1 = next(b for b in blocks if b.name == "span1")

    s = state.copy()
    table_z = s.get(span0, "z")
    for blk, x in ((span0, 0.45), (span1, 0.45 + 0.1 + 0.0001)):
        s.set(blk, "x", x)
        s.set(blk, "y", 1.14)
        s.set(blk, "z", table_z)
        for feat in ("roll", "pitch", "yaw"):
            s.set(blk, feat, 0.0)
    s.set(span0, "glue_end_b", 1.0)
    for i, blk in enumerate(blocks):
        if blk not in (span0, span1):
            s.set(blk, "x", 2.0 + 0.2 * i)
            s.set(blk, "y", 2.0)
    env._set_state(s)

    for _ in range(env.cure_threshold + 5):
        env.step(_hold_action(env))
    latched = env._get_state()
    assert latched.get(span0, "attached_end_b") == \
        float(env._block_index[span1.name])

    for _ in range(200):
        env.step(_hold_action(env))
    final = env._get_state()
    for obj in (span0, span1):
        drift = np.hypot(
            final.get(obj, "x") - latched.get(obj, "x"),
            final.get(obj, "y") - latched.get(obj, "y"))
        assert drift < 0.002, f"{obj.name} skated {drift * 1000:.1f} mm"
        dyaw = abs(final.get(obj, "yaw") - latched.get(obj, "yaw"))
        assert dyaw < 0.01, f"{obj.name} rotated {dyaw:.4f} rad"


def _stage_flush_pair(env, task):
    """Two spans butted flush on the table with the joint face wet.

    Returns (span0, span1) with everything else parked far away.
    """
    env._set_state(task.init)
    state = env._get_state()
    blocks = state.get_objects(env._block_type)
    span0 = next(b for b in blocks if b.name == "span0")
    span1 = next(b for b in blocks if b.name == "span1")
    s = state.copy()
    table_z = s.get(span0, "z")
    for blk, x in ((span0, 0.45), (span1,
                                   0.45 + 2 * env.span_half_extents[0])):
        s.set(blk, "x", x)
        s.set(blk, "y", 1.14)
        s.set(blk, "z", table_z)
        for feat in ("roll", "pitch", "yaw"):
            s.set(blk, feat, 0.0)
    for i, blk in enumerate(blocks):
        if blk not in (span0, span1):
            s.set(blk, "x", 2.0 + 0.2 * i)
            s.set(blk, "y", 2.0)
    env._set_state(s)
    env._set_attr(span0, "glue_end_b", 1.0)
    return span0, span1


def test_wet_joint_is_tacked_until_it_welds(env_and_task):
    """A curing joint carries a weak tack constraint, replaced by the weld."""
    env, task = env_and_task
    span0, span1 = _stage_flush_pair(env, task)
    key = frozenset({span0.id, span1.id})

    env.step(_hold_action(env))
    assert set(env._tack_constraints) == {key}
    assert not env._weld_constraints

    for _ in range(env.cure_threshold + 5):
        env.step(_hold_action(env))
    # The latch hands the joint over to the rigid weld; no tack lingers.
    assert not env._tack_constraints
    assert set(env._weld_constraints) == {key}

    # Restoring any state re-derives the welds and drops stale tacks
    # (they are anchored to the poses they were created at).
    env._set_state(task.init)
    assert not env._tack_constraints
    assert not env._weld_constraints


def test_wet_joint_survives_a_release_impulse(env_and_task):
    """A flush joint takes the arm's parting shove as a unit, and still cures.

    Regression: a placement that ended flush against its neighbor could
    fling an already-placed span ~5 cm and ~90 degrees during the cure
    wait (~30% of flush placements), which is what pushed agents onto a
    narrow 3-8 mm assembly gap. The wet-glue tack cannot cancel the
    impulse -- momentum is momentum, the assembly still slides -- but
    the joint must not come apart while it cures. The newcomer is
    joined to an already-welded pair, the mass asymmetry that makes an
    untacked joint separate (~11 mm here) rather than slide together.
    """
    env, task = env_and_task
    _, span1 = _stage_flush_pair(env, task)
    for _ in range(env.cure_threshold + 5):
        env.step(_hold_action(env))
    assert env._weld_constraints

    state = env._get_state()
    span2 = next(b for b in state.get_objects(env._block_type)
                 if b.name == "span2")
    s = state.copy()
    s.set(span2, "x", s.get(span1, "x") + 2 * env.span_half_extents[0])
    s.set(span2, "y", s.get(span1, "y"))
    s.set(span2, "z", s.get(span1, "z"))
    for feat in ("roll", "pitch", "yaw"):
        s.set(span2, feat, 0.0)
    env._set_state(s)
    env._set_attr(span1, "glue_end_b", 1.0)
    env.step(_hold_action(env))

    before = env._get_state()
    rel_before = np.array(
        [before.get(span2, f) - before.get(span1, f) for f in ("x", "y", "z")])
    # The shove the arm leaves behind when it releases and retreats.
    p.resetBaseVelocity(span2.id, (-2.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                        physicsClientId=env._physics_client_id)
    for _ in range(env.cure_threshold + 5):
        env.step(_hold_action(env))
    after = env._get_state()
    rel_after = np.array(
        [after.get(span2, f) - after.get(span1, f) for f in ("x", "y", "z")])
    assert abs(after.get(span1, "x") - before.get(span1, "x")) > 0.005, \
        "the shove should still move the assembly"
    assert np.linalg.norm(rel_after - rel_before) < 0.001
    assert abs(after.get(span2, "yaw") - after.get(span1, "yaw")) < 0.01
    assert after.get(span1, "attached_end_b") == \
        float(env._block_index[span2.name])


def test_degenerate_top_edge_grasp_fails_honestly():
    """A pick that never wraps the block must fail, not report success.

    Regression for seed0 run_20260819_053515: PickBlock(leg0)[0.01] on a
    standing 10 cm leg put the pads' grip band a hair above the leg top.
    The closing fingers cammed over the top corners (shoving the leg
    ~18 mm into the table), held detection latched a constraint off a
    single finger's corner graze, the 3 cm lift left the leg still on
    the table, and the downstream place jammed it -- episode dead. Two
    guards cover it:

    1. ``_detect_held_object`` requires an aligned touch on BOTH
       fingers (a single-finger touch is not a pinch).
    2. The pick skill's lift verification (``verify_lift``): the object
       must gain at least half of ``lift_dz``, else the option raises
       ``OptionExecutionFailure`` -- catching top-EDGE pinches whose
       both-finger contact normals look like a real grasp.
    """
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "pybullet_birrt_contact_margin": -0.005,
        "pybullet_birrt_path_subsample_ratio": 1,
    })
    from predicators.envs.pybullet_bridge import \
        PyBulletBridgeEnv  # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import \
        get_gt_options  # pylint: disable=import-outside-toplevel
    env = PyBulletBridgeEnv(use_gui=False)
    try:
        env.reset("test", 0)
        state = env._get_state()
        options = {o.name: o for o in get_gt_options(env.get_name())}
        leg0 = next(b for b in env._blocks if b.name == "leg0")
        robot = env._robot
        staged_z = state.get(leg0, "z")

        # --- 1) Detector: a single-finger touch is not a grasp. -------
        # Stage the gripper at grip height beside the standing leg so
        # that exactly one finger's inner face overlaps the leg (the
        # cam-over contact geometry). Strip the joint hint so the pose
        # features drive IK.
        def _stage_gripper(dy: float, fingers: float) -> None:
            s = state.copy()
            sim_state = getattr(s, "simulator_state", None)
            if isinstance(sim_state, dict):
                sim_state = dict(sim_state)
                sim_state.pop("joint_positions", None)
                s.simulator_state = sim_state
            s.set(robot, "x", state.get(leg0, "x"))
            s.set(robot, "y", state.get(leg0, "y") + dy)
            s.set(robot, "z", staged_z + 0.03)
            s.set(robot, "wrist", 0.0)
            s.set(robot, "fingers", fingers)
            env._set_state(s)

        def _aligned_fingers() -> list:
            normals = env._get_expected_finger_normals()
            aligned = []
            for fid, normal in normals.items():
                pts = p.getClosestPoints(
                    bodyA=env._pybullet_robot.robot_id,
                    bodyB=leg0.id,
                    distance=env.grasp_tol_small,
                    linkIndexA=fid,
                    physicsClientId=env._physics_client_id)
                aligned.append(
                    any(abs(float(normal.dot(pt[7]))) >= 0.9 for pt in pts))
            return aligned

        # One pad overlaps the leg (aligned touch) while the partner pad
        # is ~37 mm from any leg surface -- beyond grasp_partner_tol, so
        # nothing is closing in on the other side.
        _stage_gripper(dy=0.0225, fingers=env.open_fingers)
        # The staging is the old detector's grant condition: one finger
        # has an aligned touch...
        assert sorted(_aligned_fingers()) == [False, True]
        # ...and the pinch rule refuses it.
        assert env._detect_held_object() is None

        # An off-center pre-pinch stays a legitimate capture: one pad
        # touches while the partner pad FACES the leg from ~15 mm
        # (within grasp_partner_tol) -- the jug-handle pattern.
        _stage_gripper(dy=-0.009, fingers=0.032)
        assert env._detect_held_object() == leg0.id

        # Positive control: a genuine straddle (both pads on the leg's
        # side faces) still detects. 0.0235 leaves both pads slightly
        # inside the 5 cm leg even with IK centering the gripper ~1 mm
        # off the commanded xy.
        _stage_gripper(dy=0.0, fingers=0.0235)
        assert _aligned_fingers() == [True, True]
        assert env._detect_held_object() == leg0.id

        # --- 2) Skill: the seed0 pick must fail honestly. -------------
        def run_pick(grasp_z_offset: float):
            env.reset("test", 0)
            ground = options["PickBlock"].ground([robot, leg0],
                                                 np.array([grasp_z_offset],
                                                          dtype=np.float32))
            st = env._get_state()
            assert ground.initiable(st)
            for _ in range(200):
                env.step(ground.policy(st))
                st = env._get_state()
                if ground.terminal(st):
                    return st
            raise AssertionError("PickBlock did not terminate")

        try:
            st = run_pick(0.01)
        except utils.OptionExecutionFailure:
            # The honest outcome: the pick reports its own failure
            # (lift verification, or a collision abort from the
            # crushed-in gripper).
            pass
        else:
            # Physics drift may one day land this knife-edge pick as a
            # genuine grasp; that is success, not regression. What must
            # never happen again is the silent middle: option "done",
            # state claims held, leg still (near) the table.
            assert st.get(leg0, "is_held") > 0.5
            assert st.get(leg0, "z") > staged_z + 0.015

        # --- 3) Control: the reliable offset still picks properly. ----
        st = run_pick(0.0)
        assert st.get(leg0, "is_held") > 0.5
        assert st.get(leg0, "z") > staged_z + 0.015
    finally:
        p.disconnect(env._physics_client_id)
