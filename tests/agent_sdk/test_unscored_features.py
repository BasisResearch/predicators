"""Tests for the sim.fit report's list of moving features the fit does not
score.

Regression for round fixes_r8 Balloons seed 0: its model scored the
box's height but not the balloons', so its fit never saw that the tied
balloons stack, and the top balloon burst on the test level.
"""

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.tools.synthesis import _unscored_moving_features
from predicators.structs import Object, State, Type

_BOX = Type("box", ["z", "speed"])
_BALLOON = Type("balloon", ["z", "colour"])
_ROBOT = Type("robot", ["x", "fingers"])


def _recording(steps: int, noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    box, balloon, still = Object("box", _BOX), Object("b0", _BALLOON), \
        Object("b1", _BALLOON)
    robot = Object("robot", _ROBOT)
    states = []
    for t in range(steps):
        rise = 0.01 * t
        states.append(
            State({
                box: np.array([0.4 + rise + rng.normal(0.0, noise), 0.01]),
                # A tied balloon rises with the box; another stays put.
                balloon: np.array([0.6 + rise + rng.normal(0.0, noise), 2.0]),
                still: np.array([0.43 + rng.normal(0.0, noise), 1.0]),
                robot: np.array([0.1 * t, 0.04]),
            }))
    return states, [None] * (steps - 1)


@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_moving_features_the_fit_does_not_score_are_listed(noise) -> None:
    """A feature that changes beyond the noise and is not scored is listed;
    scored features, still ones and the robot's own are not."""
    utils.reset_config({"continual_obs_noise_position": noise})
    rollouts = [_recording(40, noise)]
    assert _unscored_moving_features(rollouts, {"box": ["z"]}) == {
        "balloon": ["z"]
    }
    # Scoring the balloon's height leaves nothing unchecked.
    assert _unscored_moving_features(rollouts, {
        "box": ["z"],
        "balloon": ["z"]
    }) == {}
    utils.reset_config({})
