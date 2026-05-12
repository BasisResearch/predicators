"""Tests for ``PyBulletEnv._reconstruction_diff`` angle-modulo handling.

Regression coverage for commit 222680da9 ("Compare angle features
modulo 2π in reconstruction diff"). Before the fix, a wrist of 4.68
(legal, but outside the canonical (-π, π] range that PyBullet reports
back from ``_get_state``) would diff against a reconstructed -1.60 and
trip the reconstruction warning even though the two represent the
same physical orientation.

These tests don't spin up PyBullet — they just exercise the
classmethod on hand-built ``State`` instances.
"""
# pylint: disable=protected-access
from __future__ import annotations

import math

import numpy as np
import pytest

# Bootstrap circular imports.
import predicators.utils  # noqa: F401
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import Object, State, Type


@pytest.fixture(name="robot_type")
def _robot_type():
    """Type with one angle feature and one position feature."""
    return Type("robot", ["wrist", "x"])


def _state(robot_type: Type, wrist: float, x: float) -> State:
    obj = robot_type("robot0")
    return State({obj: np.array([wrist, x], dtype=np.float64)})


def test_reconstruction_diff_angle_wraps_modulo_2pi(robot_type):
    """Values that differ by an exact multiple of 2π represent the
    same physical orientation and must not appear in the diff."""
    requested = _state(robot_type, wrist=0.0, x=0.5)
    reconstructed = _state(robot_type, wrist=2 * math.pi, x=0.5)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert diff == "", diff
    # Also: a near-2π offset under atol should round-trip cleanly.
    requested = _state(robot_type, wrist=4.68, x=0.5)
    reconstructed = _state(robot_type, wrist=4.68 - 2 * math.pi, x=0.5)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert diff == "", diff


def test_reconstruction_diff_angle_pi_vs_negative_pi(robot_type):
    """+π and -π are the same orientation — shortest-arc delta is 0."""
    requested = _state(robot_type, wrist=math.pi, x=0.0)
    reconstructed = _state(robot_type, wrist=-math.pi, x=0.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert diff == ""


def test_reconstruction_diff_angle_real_mismatch_is_reported(robot_type):
    """π/2 vs -π/2 are opposite orientations — the shortest-arc delta
    is π, which exceeds atol and must surface in the diff."""
    requested = _state(robot_type, wrist=math.pi / 2, x=0.0)
    reconstructed = _state(robot_type, wrist=-math.pi / 2, x=0.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.wrist" in diff


def test_reconstruction_diff_non_angle_feature_uses_raw_delta(robot_type):
    """Non-angle features (``x`` here) compare with raw subtraction, no
    modulo wrap-around — a 1.0-unit delta is reported as 1.0."""
    requested = _state(robot_type, wrist=0.0, x=0.0)
    reconstructed = _state(robot_type, wrist=0.0, x=1.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.x" in diff
    assert "robot0.wrist" not in diff


def test_reconstruction_diff_object_set_mismatch(robot_type):
    """Objects present in only one state surface as a top-level diff
    line — unrelated to the angle-modulo logic but the same helper
    handles it."""
    o0 = robot_type("robot0")
    o1 = robot_type("robot1")
    requested = State({o0: np.array([0.0, 0.0])})
    reconstructed = State({o1: np.array([0.0, 0.0])})
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "only in requested" in diff
    assert "only in reconstructed" in diff
