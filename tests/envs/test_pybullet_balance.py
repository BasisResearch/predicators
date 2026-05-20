"""Regression tests for PyBulletBalanceEnv.

Currently covers `_update_balance_beam`'s delta-tracking behavior: when
``|diff|`` (the imbalance between left and right block counts) shrinks
between successive calls, blocks on each plate must continue to track
the plate's z so they neither sink into the plate nor float above it.

The prior implementation read each block's z via ``state.get(block,
"z")`` (the live pybullet z, which already includes any prior shift)
and added ``sign * abs(diff) * shift_per_block``. That made each call
re-add the *absolute* shift on top of the cumulative one, so a sequence
like ``diff = -4, -2, 0`` left blocks displaced by ``-4 - 2 - 0 = -6``
shifts even though plates snapped back to base.
"""

import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_balance import PyBulletBalanceEnv


@pytest.fixture(name="balance_env")
def _balance_env():
    utils.reset_config({
        "env": "pybullet_balance",
        "approach": "oracle",
        "use_gui": False,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 0,
        "pybullet_control_mode": "reset",
    })
    env = PyBulletBalanceEnv(use_gui=False)
    env.reset("test", 0)
    return env


def _z(env, body_id):
    pos, _ = p.getBasePositionAndOrientation(
        body_id, physicsClientId=env._physics_client_id)  # noqa: SLF001
    return pos[2]


def test_update_balance_beam_blocks_track_plates(balance_env):
    env = balance_env
    state = env._get_state()  # noqa: SLF001

    # Identify blocks currently in play (z >= 0; out-of-view blocks live
    # at high z but well off the workspace).
    block_objs = [
        b for b in state.get_objects(env._block_type)  # noqa: SLF001
        if state.get(b, "z") >= 0
    ]
    # state.get(b, "y") chooses side via the same midpoint test
    # _update_balance_beam uses.
    midpoint_y = env._table2_y  # noqa: SLF001
    left_blocks = [b for b in block_objs if state.get(b, "y") < midpoint_y]
    right_blocks = [b for b in block_objs if state.get(b, "y") > midpoint_y]
    assert left_blocks and right_blocks, (
        "Default seed=0 task should split 1:5 between the two plates")

    plate1_id = env._plate1.id  # noqa: SLF001
    plate3_id = env._plate3.id  # noqa: SLF001
    base_p1 = env._plate1_pose[2]  # noqa: SLF001
    base_p3 = env._plate3_pose[2]  # noqa: SLF001

    # The reset run already triggered _update_balance_beam with the
    # natural diff, so pybullet is already shifted. Reset both _prev_diff
    # and the live pybullet positions back to base before driving a
    # controlled sequence — otherwise the first synthetic call mixes the
    # reset's shift into the delta.
    env._prev_diff = 0  # noqa: SLF001
    for body_id, base_z in [(plate1_id, base_p1), (plate3_id, base_p3)]:
        pos, orn = p.getBasePositionAndOrientation(
            body_id, physicsClientId=env._physics_client_id)
        p.resetBasePositionAndOrientation(
            body_id, [pos[0], pos[1], base_z], orn,
            physicsClientId=env._physics_client_id)
    # Snap blocks to a clean "everyone on their plate top" baseline.
    block_base_z = {}
    for b in block_objs:
        pos, orn = p.getBasePositionAndOrientation(
            b.id, physicsClientId=env._physics_client_id)
        block_base_z[b.id] = pos[2]

    # Drive the balance-beam updates by monkey-patching count_num_blocks
    # (the recursive predicate-based counter), so we don't have to
    # actually move blocks between plates.
    sequence = [(1, 5), (2, 4), (3, 3), (2, 4), (1, 5), (3, 3)]
    shift_per_block = 0.007

    def fake_counter(left, right):
        def _count(_state, plate):
            if plate is env._plate1:  # noqa: SLF001
                return left
            if plate is env._plate3:  # noqa: SLF001
                return right
            return 0

        return _count

    for left, right in sequence:
        env.count_num_blocks = fake_counter(left, right)
        # Refresh state each iteration — _domain_specific_step does
        # `state = self._get_state()` per tick, so the buggy version
        # reads each block's *live* z (post-prior-shift) and compounds
        # the offset. A frozen state would mask the bug.
        state = env._get_state()  # noqa: SLF001
        env._update_balance_beam(state)  # noqa: SLF001

        diff = left - right
        # Plate offset matches the diff exactly (signed: positive offset
        # = up; left side rises when right is heavier, i.e. diff < 0).
        expected_p1 = base_p1 + (-diff * shift_per_block)
        expected_p3 = base_p3 + (diff * shift_per_block)
        assert _z(env, plate1_id) == pytest.approx(expected_p1, abs=1e-6)
        assert _z(env, plate3_id) == pytest.approx(expected_p3, abs=1e-6)

        # Each block's z should equal its base + the same offset as the
        # plate it sits on — no compounding across calls.
        for b in left_blocks:
            expected = block_base_z[b.id] + (-diff * shift_per_block)
            assert _z(env, b.id) == pytest.approx(expected, abs=1e-6), (
                f"left block {b.name} drifted at (left={left}, "
                f"right={right}): got {_z(env, b.id)}, want {expected}")
        for b in right_blocks:
            expected = block_base_z[b.id] + (diff * shift_per_block)
            assert _z(env, b.id) == pytest.approx(expected, abs=1e-6), (
                f"right block {b.name} drifted at (left={left}, "
                f"right={right}): got {_z(env, b.id)}, want {expected}")
