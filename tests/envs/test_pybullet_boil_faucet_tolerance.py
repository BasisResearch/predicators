"""JugAtFaucet follows ``CFG.boil_faucet_align_threshold``: the jug must sit
under the spout, not merely beside it."""

import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.settings import CFG


def test_jug_at_faucet_follows_the_flag() -> None:
    """A jug 7 cm from the outlet is not at the faucet under the default 5 cm
    tolerance, and is under the old 10 cm one."""
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    assert CFG.boil_faucet_align_threshold == 0.05
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    assert isinstance(env, PyBulletBoilEnv)
    try:
        env.reset("train", 0)
        state = env.get_observation().copy()
        jug = next(o for o in state if o.type.name == "jug")
        faucet = next(o for o in state if o.type.name == "faucet")
        ox, oy = env._faucet_outlet_xy(state, faucet)  # pylint: disable=protected-access
        rot = state.get(faucet, "rot")
        # 7 cm further along the spout direction, away from the body.
        state.set(
            jug, "x", ox + 0.07 * np.cos(rot) *
            np.sign(PyBulletBoilEnv.faucet_outlet_local_dx))
        state.set(
            jug, "y", oy + 0.07 * np.sin(rot) *
            np.sign(PyBulletBoilEnv.faucet_outlet_local_dx))
        state.set(jug, "is_held", 0.0)
        holds = env._JugAtFaucet_holds  # pylint: disable=protected-access
        assert env.faucet_align_threshold == 0.05
        assert not holds(state, [jug, faucet])
        utils.update_config({"boil_faucet_align_threshold": 0.1})
        assert env.faucet_align_threshold == 0.1
        assert holds(state, [jug, faucet])
        # At the outlet itself the jug is at the faucet under either.
        utils.update_config({"boil_faucet_align_threshold": 0.05})
        state.set(jug, "x", ox)
        state.set(jug, "y", oy)
        assert holds(state, [jug, faucet])
    finally:
        utils.update_config({"boil_faucet_align_threshold": 0.05})
