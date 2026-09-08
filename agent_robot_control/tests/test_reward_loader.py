"""Reward code compilation and validation."""
import numpy as np
import pytest

from agent_robot_control.rl.reward_loader import RewardError, call_reward, \
    load_reward

GOOD = """
import numpy as np
def reward(particles, visible, ee_pos, ee_quat, gripper):
    d = particles["donut_0"]
    t = particles["target"]
    if len(d) == 0 or len(t) == 0:
        return -1.0
    dist = np.linalg.norm(d.mean(0)[:2] - t.mean(0)[:2])
    return 1.0 if dist < 0.03 else -float(dist)
"""

def _inputs(dist):
    particles = {"donut_0": np.zeros((8, 3)), "target": np.zeros((8, 3)) + np.array([dist, 0, 0])}
    visible = {k: np.ones(8, dtype=bool) for k in particles}
    return particles, visible, np.zeros(3), np.array([0, 0, 0, 1.0]), 0.04


def test_good_reward_runs():
    fn = load_reward(GOOD)
    assert call_reward(fn, *_inputs(0.5)) == pytest.approx(-0.5)
    assert call_reward(fn, *_inputs(0.01)) == 1.0


def test_missing_function_rejected():
    with pytest.raises(RewardError):
        load_reward("x = 1")


def test_arbitrary_import_rejected():
    with pytest.raises(RewardError):
        load_reward("import os\ndef reward(*a): return 0.0")


def test_non_finite_rejected():
    fn = load_reward("def reward(p, v, e, q, g): return float('nan')")
    with pytest.raises(RewardError):
        call_reward(fn, *_inputs(0.1))


def test_exception_wrapped():
    fn = load_reward("def reward(p, v, e, q, g): return p['missing'].mean()")
    with pytest.raises(RewardError):
        call_reward(fn, *_inputs(0.1))


def test_numpy_scalar_ok():
    fn = load_reward("def reward(p, v, e, q, g): return np.float32(0.25)")
    assert call_reward(fn, *_inputs(0.1)) == pytest.approx(0.25)
