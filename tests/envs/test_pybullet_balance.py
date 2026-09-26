"""Tests for the balance beam in PyBulletBalanceEnv."""

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_balance import PyBulletBalanceEnv
from predicators.structs import Action, State


@pytest.fixture(name="env")
def _env():
    utils.reset_config({
        "env": "pybullet_balance",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    return PyBulletBalanceEnv(use_gui=False)


# pylint: disable=protected-access
def _plate_top(env, state: State, block) -> float:
    plate = env._plate1 if state.get(block, "y") < env._table2_y \
        else env._plate3
    return state.get(plate, "z") + env._plate_height


def _rest_gaps(env, state: State) -> list:
    """Per in-view block, how far it sits above resting on its pile."""
    gaps = []
    size = env._block_size
    for block in state.get_objects(env._block_type):
        z = state.get(block, "z")
        level = (z - _plate_top(env, state, block) - size / 2) / size
        gaps.append(abs(level - round(level)) * size)
    return gaps


def _hold(env) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def _move_top_block_across(env, state: State) -> State:
    """Put the top block of the heavier plate's pile on the lighter one."""
    blocks = state.get_objects(env._block_type)
    left = [b for b in blocks if state.get(b, "y") < env._table2_y]
    right = [b for b in blocks if b not in left]
    heavy, light = (left, right) if len(left) > len(right) else (right, left)
    top = max(heavy, key=lambda b: state.get(b, "z"))
    dest = max(light, key=lambda b: state.get(b, "z"))
    moved = state.copy()
    moved.set(top, "x", state.get(dest, "x"))
    moved.set(top, "y", state.get(dest, "y"))
    moved.set(top, "z", state.get(dest, "z") + env._block_size)
    return moved


def test_blocks_ride_the_beam_without_bouncing(env):
    """When the count difference changes, stacks move with their plate and stay
    at rest instead of being pushed into or lifted off it."""
    state = env.reset("test", 0)
    before = env._prev_diff
    assert before != 0
    env._set_state(_move_top_block_across(env, state))
    state = env._get_state()
    assert env._prev_diff == before - 2 * np.sign(before)
    assert max(_rest_gaps(env, state)) < 1e-3
    for _ in range(20):
        state = env.step(_hold(env))
    for block in state.get_objects(env._block_type):
        speed = np.linalg.norm(
            p.getBaseVelocity(block.id,
                              physicsClientId=env._physics_client_id)[0])
        assert speed < 0.02, (block, speed)
    assert max(_rest_gaps(env, state)) < 2e-3


def test_setting_a_tilted_state_does_not_shift_blocks_again(env):
    """A mid-episode state already encodes the tilt; re-setting it must leave
    the blocks where they were."""
    state = env.reset("test", 0)
    assert env._prev_diff != 0
    for _ in range(5):
        state = env.step(_hold(env))
    env._set_state(state)
    again = env._get_state()
    for block in state.get_objects(env._block_type):
        assert abs(again.get(block, "z") - state.get(block, "z")) < 1e-3
