"""Tests for the residual ``Attach`` physics command and the base-sim
hidden-weld gate.

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


def test_base_sim_does_not_materialize_feature_welds():
    """skip_residual_dynamics gates _sync_welds_to_state; full env keeps
    it."""
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
