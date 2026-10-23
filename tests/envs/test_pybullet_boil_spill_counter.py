"""The spill counter lets 20 overflow steps pass and spills from the 21st."""
# pylint: disable=protected-access
import pybullet as p
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_boil import PyBulletBoilEnv


def test_twenty_overflow_steps_spill_nothing_and_the_next_one_spills() -> None:
    """The counter starts 20 fill steps below zero, so 20 steps of water
    running over a full jug or onto the table leave ``spilled_level`` at
    exactly zero and NoWaterSpilled true; the 21st spills one fill step.

    The float sum of 20 fill steps of 0.02 from -0.4 is 1.5e-16, which
    NoWaterSpilled, testing ``spilled_level > 0``, read as a spill.
    """
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    assert isinstance(env, PyBulletBoilEnv)
    try:
        env.reset("train", 0)
        state = env.get_observation().copy()
        faucet = next(o for o in state if o.type.name == "faucet")
        step = env.water_fill_speed
        assert env._faucet._spilled_level == pytest.approx(-20 * step)
        for _ in range(20):
            env._increment_spillage(env._faucet._spilled_level, state)
        assert state.get(faucet, "spilled_level") == 0.0
        assert env._NoWaterSpilled_holds(state, [])
        env._increment_spillage(env._faucet._spilled_level, state)
        assert state.get(faucet, "spilled_level") == pytest.approx(step)
        assert not env._NoWaterSpilled_holds(state, [])
    finally:
        p.disconnect(env._physics_client_id)
