"""Tests for speed_support."""
import numpy as np
import pytest

from predicators.agent_sdk.speed_support import SPEED_WINDOW, \
    recorded_speeds, scored_bodies, switch_features, switch_pattern, \
    window_speeds
from predicators.structs import Object, State, Type

_BALL_TYPE = Type("ball", ["x", "y", "radius"])
_FAN_TYPE = Type("fan", ["slot", "is_on"])
_ROBOT_TYPE = Type("robot", ["x", "busy"])
_BALL = Object("ball", _BALL_TYPE)
_FAN = Object("fan_0", _FAN_TYPE)
_OTHER_FAN = Object("fan_1", _FAN_TYPE)
_ROBOT = Object("robot", _ROBOT_TYPE)
_BODIES = {"ball": ["x", "y"]}
_SWITCHES = {("fan", "is_on")}


def _state(x: float, on: float = 0.0) -> State:
    return State({
        _BALL: np.array([x, 0.0, 0.04], dtype=np.float32),
        _FAN: np.array([0.0, on], dtype=np.float32),
        _OTHER_FAN: np.array([1.0, 0.0], dtype=np.float32),
        _ROBOT: np.array([0.0, on], dtype=np.float32),
    })


def test_scored_bodies_keeps_position_features():
    """A scored type counts as a body by its position features, and the robot
    never does."""
    assert scored_bodies({
        "ball": ["z", "x", "radius"],
        "fan": ["is_on"],
        "robot": ["x"]
    }) == {
        "ball": ["x", "z"]
    }


def test_switch_features_need_both_values_on_one_object():
    """A switch reads only 0 and 1, and some object shows both: a slot number
    that differs between objects is not one, nor is the robot's flag."""
    assert switch_features([[_state(0.0, 0.0),
                             _state(0.0, 1.0)]]) == _SWITCHES
    assert switch_features([[_state(0.0), _state(0.5)]]) == set()


def test_switch_pattern_names_what_is_on():
    """A pattern holds the switches that read on."""
    assert switch_pattern(_state(0.0, 1.0), _SWITCHES) == {"fan_0.is_on"}
    assert switch_pattern(_state(0.0), _SWITCHES) == frozenset()


def test_window_speeds_skip_a_window_that_changes_pattern():
    """A speed comes from a whole window under one pattern."""
    steps = SPEED_WINDOW + 1
    same = [_state(0.1 * i) for i in range(steps)]
    speeds = list(window_speeds(same, _BODIES, _SWITCHES))
    assert len(speeds) == 1
    assert speeds[0][0] == _BALL
    assert speeds[0][2] == pytest.approx(0.1, abs=1e-5)
    changed = same[:-1] + [_state(0.1 * steps, 1.0)]
    assert not list(window_speeds(changed, _BODIES, _SWITCHES))


def test_recorded_speeds_keep_the_top_speed_of_each_pattern():
    """Each pattern keeps its own top speed, over all recordings."""
    length = SPEED_WINDOW + 1
    slow = [_state(0.01 * i) for i in range(length)]
    fast = [_state(0.05 * i, 1.0) for i in range(length)]
    faster = [_state(0.02 * i) for i in range(length)]
    top = recorded_speeds([slow, fast, faster], _BODIES, _SWITCHES)
    assert top[("ball", frozenset())] == pytest.approx(0.02, abs=1e-5)
    assert top[("ball",
                frozenset({"fan_0.is_on"}))] == pytest.approx(0.05, abs=1e-5)
