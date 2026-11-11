"""A Fan switch's button is observed where it sits along its travel, and a
state puts it back there."""
# pylint: disable=protected-access
import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv


def test_switch_travel_round_trips_and_keeps_the_reading() -> None:
    """A partial travel is restored as given; ``is_on`` follows it; a state
    whose reading disagrees with its travel snaps the button to the reading's
    end."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = PyBulletFanEnv(use_gui=False)
    client = env._physics_client_id
    try:
        state = env.reset("train", 0).copy()
        switch = env._switches[0]
        assert "travel" in switch.type.feature_names
        assert "travel" in switch.type.sensor_features
        assert state.get(switch, "travel") == pytest.approx(0.0, abs=1e-6)

        for travel, on in ((0.3, 0.0), (0.8, 1.0)):
            staged = state.copy()
            staged.set(switch, "travel", travel)
            staged.set(switch, "is_on", on)
            env._set_state(staged)
            observed = env._get_state()
            assert observed.get(switch, "travel") == pytest.approx(travel,
                                                                   abs=1e-4)
            assert observed.get(switch, "is_on") == on

        # A learned rule flips the reading and leaves the travel stale.
        staged = state.copy()
        staged.set(switch, "travel", 0.2)
        staged.set(switch, "is_on", 1.0)
        env._set_state(staged)
        observed = env._get_state()
        assert observed.get(switch, "is_on") == 1.0
        assert observed.get(switch, "travel") == pytest.approx(1.0, abs=1e-4)
    finally:
        p.disconnect(client)
