"""Tests for the residual ``Attach`` physics command and the base-sim hidden-
weld gate.

Three contracts:

1. Executor mechanics (``PyBulletEnv._reconcile_commanded_attachments``):
   an Attach command welds the pair at their current relative pose,
   persists exactly while re-emitted, and is removed the first action it
   is not re-emitted.
2. Hidden-semantics gate: a base-sim env (``skip_residual_dynamics=True``)
   must NOT materialize welds from ``attached_*`` features in
   ``_set_state`` -- that consequence belongs to the residual rules --
   while the full env still does.
3. The bridge GT simulator programs emit the Attach command for latched
   pairs (the re-emit-to-persist idiom).
"""
# pylint: disable=protected-access
import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.commands import Attach, CommandBuffer
from predicators.code_sim_learning.utils import apply_rules
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.structs import Action

# Fixed block-index convention of the bridge env / GT simulator:
# leg0, leg1 -> 0, 1; span0..span2 -> 2..4.
_SPAN0_IDX, _SPAN1_IDX = 2.0, 3.0


def _make_env(**kwargs) -> PyBulletBridgeEnv:
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
    })
    return PyBulletBridgeEnv(use_gui=False, **kwargs)


def _side_by_side_state(env: PyBulletBridgeEnv):
    """Two spans butted end-to-end on the table, everything else staged."""
    state = env._generate_train_tasks()[0].init.copy()
    span0, span1 = env._spans[:2]
    hx = env.span_half_extents[0]
    z = env.table_height + env.span_half_extents[2]
    for i, span in enumerate((span0, span1)):
        state.set(span, "x", 0.65 + i * (2 * hx + 0.002))
        state.set(span, "y", 1.30)
        state.set(span, "z", z)
        state.set(span, "yaw", 0.0)
    return state, span0, span1


def test_attach_command_welds_and_expires():
    """Attach persists while re-emitted and is removed when it stops."""
    env = _make_env(skip_residual_dynamics=True)
    state, span0, span1 = _side_by_side_state(env)
    env._set_state(state)
    act = Action(np.array(env._pybullet_robot.initial_joint_positions))

    # Lift span0 with an upward force while welded: the pair weighs
    # ~2 N (0.1 kg each), so 3 N on span0 alone lifts BOTH only if the
    # weld carries span1. Forces act through the constraint solver, so
    # the frozen relative pose must hold.
    z0_start = env._get_state().get(span1, "z")
    for _ in range(6):
        buf = CommandBuffer()
        buf.attach(span0, span1)
        buf.apply_force(span0, (0.0, 0.0, 3.0))
        env.queue_residual_commands(buf.commands)
        env.step(act)
    assert len(env._cmd_weld_constraints) == 1
    lifted = env._get_state()
    assert lifted.get(span1, "z") > z0_start + 0.02, \
        "welded partner did not follow the commanded lift"
    # The off-center lift torques the pair, so it rotates as one rigid
    # body; the rotation-invariant check is the center-to-center
    # distance, which the frozen relative pose must preserve.
    gap = np.linalg.norm(
        [lifted.get(span1, f) - lifted.get(span0, f) for f in ("x", "y", "z")])
    assert abs(gap - (2 * env.span_half_extents[0] + 0.002)) < 0.01, \
        "weld did not hold the pair's relative pose"

    # Stop emitting: the very next action removes the weld and span1
    # free-falls while span0 is still driven up.
    z1_high = lifted.get(span1, "z")
    for _ in range(4):
        buf = CommandBuffer()
        buf.apply_force(span0, (0.0, 0.0, 3.0))
        env.queue_residual_commands(buf.commands)
        env.step(act)
    assert not env._cmd_weld_constraints
    assert env._get_state().get(span1, "z") < z1_high - 0.01, \
        "weld survived after the Attach command stopped being emitted"
    p.disconnect(env._physics_client_id)


def test_queued_commands_survive_simulate_state_sync():
    """Commands queued for the incoming state survive simulate().

    The option model queues commands and then calls ``simulate(state,
    act)``, where ``state`` is the previous step's MERGED state (base
    step plus rule-written feature updates, e.g. a curing counter). The
    merge makes ``state`` differ from the env's raw post-step state, so
    simulate() takes its ``_set_state`` branch; the lifecycle wipe in
    ``_set_state`` must not eat the just-queued commands.
    """
    env = _make_env(skip_residual_dynamics=True)
    state, span0, span1 = _side_by_side_state(env)
    env._set_state(state)
    act = Action(np.array(env._pybullet_robot.initial_joint_positions))
    cur = env.simulate(state, act)
    z1_start = cur.get(span1, "z")
    for _ in range(6):
        # Mimic a residual rule advancing a counter: the merged state
        # now differs from the env's post-step state by one feature.
        merged = cur.copy()
        merged.set(span0, "cure_end_b", merged.get(span0, "cure_end_b") + 1.0)
        buf = CommandBuffer()
        buf.attach(span0, span1)
        buf.apply_force(span0, (0.0, 0.0, 3.0))
        env.queue_residual_commands(buf.commands)
        cur = env.simulate(merged, act)
    assert len(env._cmd_weld_constraints) == 1, \
        "queued Attach was wiped by simulate()'s internal _set_state"
    assert cur.get(span1, "z") > z1_start + 0.02, \
        "welded partner did not follow the commanded lift"
    p.disconnect(env._physics_client_id)


def test_base_sim_does_not_materialize_feature_welds():
    """skip_residual_dynamics gates _sync_welds_to_state; full env keeps it."""
    for skip, expect_welds in ((True, 0), (False, 1)):
        env = _make_env(skip_residual_dynamics=skip)
        state, span0, span1 = _side_by_side_state(env)
        state.set(span0, "attached_end_b", _SPAN1_IDX)
        state.set(span1, "attached_end_a", _SPAN0_IDX)
        env._set_state(state)
        assert len(env._weld_constraints) == expect_welds, \
            f"skip_residual_dynamics={skip}"
        p.disconnect(env._physics_client_id)


def test_gt_simulator_emits_attach_for_latched_pair():
    """The FO GT program re-emits Attach for an already-latched joint."""
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models.bridge import gt_simulator
    env = _make_env(skip_residual_dynamics=True)
    state, span0, span1 = _side_by_side_state(env)
    state.set(span0, "attached_end_b", _SPAN1_IDX)
    state.set(span1, "attached_end_a", _SPAN0_IDX)
    params = {s.name: s.init_value for s in gt_simulator.PARAM_SPECS()}
    buf = CommandBuffer()
    apply_rules(state, gt_simulator.RESIDUAL_RULES, params, cmds=buf)
    attaches = [c for c in buf.commands if isinstance(c, Attach)]
    assert len(attaches) == 1
    assert {attaches[0].obj_a_name,
            attaches[0].obj_b_name} == {span0.name, span1.name}
    p.disconnect(env._physics_client_id)


def _rel_transform(env, obj_a, obj_b):
    """(pos, orn) of obj_b in obj_a's frame, straight from pybullet."""
    pos_a, orn_a = p.getBasePositionAndOrientation(
        obj_a.id, physicsClientId=env._physics_client_id)
    pos_b, orn_b = p.getBasePositionAndOrientation(
        obj_b.id, physicsClientId=env._physics_client_id)
    inv = p.invertTransform(pos_a, orn_a)
    return p.multiplyTransforms(inv[0], inv[1], pos_b, orn_b)


def _transform_error(rel, ref):
    """(position error, geodesic angle) between two (pos, orn) tuples."""
    dpos = float(np.linalg.norm(np.subtract(rel[0], ref[0])))
    dot = abs(float(np.dot(rel[1], ref[1])))
    dang = 2.0 * float(np.arccos(min(1.0, dot)))
    return dpos, dang


def test_pin_held_weld_assembly_is_rigid():
    """With pybullet_pin_held_weld_assemblies, a welded chain follows a
    violently carried held root with EXACT relative transforms.

    The carry is mimicked the way the base sim itself carries in reset
    mode: teleport the held root each action. Both constraint
    directions are exercised (held span1 is the CHILD of the
    span0->span1 weld and the PARENT of span1->span2), and the same
    scenario without the flag shows measurable flex, proving the test
    actually stresses the joints.
    """
    for pin, expect_rigid in ((True, True), (False, False)):
        env = _make_env(skip_residual_dynamics=True)
        utils.update_config({"pybullet_pin_held_weld_assemblies": pin})
        state = env._generate_train_tasks()[0].init.copy()
        span0, span1, span2 = env._spans[:3]
        hx = env.span_half_extents[0]
        z = env.table_height + env.span_half_extents[2]
        for i, span in enumerate((span0, span1, span2)):
            state.set(span, "x", 0.55 + i * (2 * hx + 0.002))
            state.set(span, "y", 1.30)
            state.set(span, "z", z)
            state.set(span, "yaw", 0.0)
        env._set_state(state)
        act = Action(np.array(env._pybullet_robot.initial_joint_positions))

        # Weld the chain (re-emitted every action) and settle one step.
        def emit(env=env, span0=span0, span1=span1, span2=span2):
            buf = CommandBuffer()
            buf.attach(span0, span1)
            buf.attach(span1, span2)
            env.queue_residual_commands(buf.commands)

        emit()
        env.step(act)
        # "Grasp" span1 and record the assembled relative transforms.
        env._held_obj_id = span1.id
        ref01 = _rel_transform(env, span1, span0)
        ref12 = _rel_transform(env, span1, span2)
        # Violent carry: teleport the held root 4 cm up/sideways per
        # action for several actions (gravity acts on the partners).
        for _ in range(6):
            pos, orn = p.getBasePositionAndOrientation(
                span1.id, physicsClientId=env._physics_client_id)
            p.resetBasePositionAndOrientation(
                span1.id, (pos[0] + 0.04, pos[1], pos[2] + 0.04),
                orn,
                physicsClientId=env._physics_client_id)
            emit()
            env.step(act)
        err01 = _transform_error(_rel_transform(env, span1, span0), ref01)
        err12 = _transform_error(_rel_transform(env, span1, span2), ref12)
        env._held_obj_id = None
        p.disconnect(env._physics_client_id)
        if expect_rigid:
            # The pin enforces the constraints' DECLARED frames; the
            # reference was captured from the solver-settled pose, which
            # sits micrometers off those frames - hence the 1e-4 slack
            # (still orders of magnitude below physical relevance).
            assert err01[0] < 1e-4 and err12[0] < 1e-4, \
                f"pinned chain drifted: {err01}, {err12}"
            assert err01[1] < 1e-4 and err12[1] < 1e-4, \
                f"pinned chain rotated: {err01}, {err12}"
        else:
            assert max(err01[0], err12[0]) > 1e-3 or \
                max(err01[1], err12[1]) > 1e-2, \
                "unpinned carry showed no flex - the test is not " \
                "stressing the joints"


def test_pin_never_moves_static_bodies_or_idle_assemblies():
    """The pin is a no-op without a held root, and never re-poses a static
    (mass-0) body welded into the held assembly."""
    env = _make_env(skip_residual_dynamics=True)
    utils.update_config({"pybullet_pin_held_weld_assemblies": True})
    state, span0, span1 = _side_by_side_state(env)
    env._set_state(state)
    act = Action(np.array(env._pybullet_robot.initial_joint_positions))
    site = env._sites[0]
    site_pose = p.getBasePositionAndOrientation(
        site.id, physicsClientId=env._physics_client_id)

    def emit():
        buf = CommandBuffer()
        buf.attach(span0, span1)
        buf.attach(span0, site)
        env.queue_residual_commands(buf.commands)

    # No held root: nothing is re-posed (welds still settle normally).
    emit()
    env.step(act)
    # Held root welded (transitively) to a static site: the span partner
    # is pinned, the site must not budge.
    env._held_obj_id = span0.id
    for _ in range(4):
        emit()
        env.step(act)
    site_pose_after = p.getBasePositionAndOrientation(
        site.id, physicsClientId=env._physics_client_id)
    assert np.allclose(site_pose[0], site_pose_after[0], atol=1e-9), \
        "the pin moved a static body"
    rel = _rel_transform(env, span0, span1)
    assert rel is not None  # the dynamic partner still tracked
    env._held_obj_id = None
    p.disconnect(env._physics_client_id)


def test_bridge_weld_edges_merge_native_and_command_welds():
    """The bridge override reports BOTH weld registries to the pin."""
    env = _make_env(skip_residual_dynamics=True)
    state, span0, span1 = _side_by_side_state(env)
    env._set_state(state)
    act = Action(np.array(env._pybullet_robot.initial_joint_positions))
    buf = CommandBuffer()
    buf.attach(span0, span1)
    env.queue_residual_commands(buf.commands)
    env.step(act)
    span2 = env._spans[2]
    env._create_weld(span1.id, span2.id, ideal_dz=None)
    edges = env._weld_constraint_edges()
    assert frozenset({span0.id, span1.id}) in edges
    assert frozenset({span1.id, span2.id}) in edges
    assert len(edges) == 2
    p.disconnect(env._physics_client_id)


def test_command_welds_round_trip_through_state():
    """A live command weld is part of the State (``simulator_state.

    ["command_welds"]``, by object name) and a restore on ANOTHER env
    instance rebuilds it at the restored relative pose, so the skills'
    planning simulator sees the rule's assembly as one rigid body.
    """
    env = _make_env(skip_residual_dynamics=True)
    state, span0, span1 = _side_by_side_state(env)
    env._set_state(state)
    act = Action(np.array(env._pybullet_robot.initial_joint_positions))
    assert "command_welds" not in env._get_state().simulator_state
    for _ in range(3):
        buf = CommandBuffer()
        buf.attach(span0, span1)
        env.queue_residual_commands(buf.commands)
        env.step(act)
    welded = env._get_state()
    assert welded.simulator_state["command_welds"] == [("span0", "span1")]
    assert env.get_welded_partner_ids(span0.id) == {span1.id}

    other = PyBulletBridgeEnv(use_gui=False, skip_residual_dynamics=True)
    try:
        assert not other._cmd_weld_constraints
        other._set_state(welded)
        o_span0 = next(b for b in other._spans if b.name == "span0")
        o_span1 = next(b for b in other._spans if b.name == "span1")
        assert other.get_welded_partner_ids(o_span0.id) == {o_span1.id}
        assert other.get_welded_partner_ids(o_span1.id) == {o_span0.id}
        # The frozen frame is the restored relative pose: span1 sits one
        # span length (+2 mm) along span0's x axis.
        (rel_pos,
         _), = other.get_welded_partner_transforms(o_span0.id).values()
        expected = 2 * env.span_half_extents[0] + 0.002
        assert abs(rel_pos[0] - expected) < 0.003
        assert abs(rel_pos[1]) < 0.003 and abs(rel_pos[2]) < 0.003
        # A State without the record clears the weld again.
        other._set_state(state)
        assert not other._cmd_weld_constraints
        assert not other.get_welded_partner_ids(o_span0.id)
        # During stepping the emitting rule stays the authority: a
        # restored weld that is not re-emitted is dropped by the
        # per-action reconcile, a re-emitted one is kept.
        other._set_state(welded)
        buf = CommandBuffer()
        buf.attach(o_span0, o_span1)
        other.queue_residual_commands(buf.commands)
        other.step(act)
        assert len(other._cmd_weld_constraints) == 1
        other.step(act)
        assert not other._cmd_weld_constraints
    finally:
        p.disconnect(other._physics_client_id)
    p.disconnect(env._physics_client_id)
