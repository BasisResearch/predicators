"""The boil env keeps each jug's true heat per env instance.

A State hands the same jug Objects to every env that is set to it, so
heat stored on the Object let a second env (the model arm's base-sim
env, a validation env) overwrite the live env's running heat whenever it
was set to a state without the privileged block (an observed view, a
sanitized recording state): the live jug fell off the boil at every
skill invocation under observation noise.
"""

# pylint: disable=protected-access

import numpy as np

from predicators import utils
from predicators.code_sim_learning.rollout_env import dispose_env
from predicators.envs import create_new_env
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.observation_noise import ObservationNoise
from predicators.run.recording import sanitize_state
from predicators.structs import Action


def _heat(env, jug) -> float:
    return env.get_observation().privileged[jug.name]["heat_level"]


def test_a_second_env_never_touches_another_envs_heat() -> None:
    """Setting a second env to a privileged-less copy of the live env's state
    leaves the live env's heat alone, and only the live env's own reset from
    the true state (privileged block included) restores it."""
    utils.reset_config({
        "env": "pybullet_boil",
        "seed": 0,
        "partially_observable": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "continual_obs_noise_position": 0.0125,
        "continual_obs_noise_orientation": 0.05,
    })
    live = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    other = create_new_env("pybullet_boil",
                           do_cache=False,
                           use_gui=False,
                           skip_residual_dynamics=True)
    assert isinstance(live, PyBulletBoilEnv)
    assert isinstance(other, PyBulletBoilEnv)
    try:
        live.reset("train", 0)
        state = live.get_observation()
        jug = next(o for o in state if o.type.name == "jug")
        live._heat_levels[jug.name] = 0.95
        state = live.get_observation()
        assert state.privileged == {jug.name: {"heat_level": 0.95}}
        # The derived observable follows the env's own record.
        assert np.isclose(state.get(jug, "bubbling_level"),
                          (0.95 - live.BUBBLING_THRESHOLD) *
                          live.BUBBLING_RAMP)
        zero = Action(np.zeros(live.action_space.shape, dtype=np.float32))

        view = ObservationNoise.from_cfg().perturb(state,
                                                   np.random.default_rng(0))
        assert view.privileged is None
        assert any(o is jug for o in view), "the view shares the Objects"
        other.simulate(view, zero)
        assert _heat(live, jug) == 0.95
        other.simulate(sanitize_state(state), zero)
        assert _heat(live, jug) == 0.95
        # The second env's own heat is whatever its state carried: none.
        assert _heat(other, jug) == 0.0
        # The true state (privileged block included) restores a heat.
        other.simulate(state, zero)
        assert _heat(other, jug) == 0.95
        assert _heat(live, jug) == 0.95

        # The live env's own round trip through its true state keeps it.
        live._set_state(live.get_observation())
        assert _heat(live, jug) == 0.95
        live.step(zero)
        assert _heat(live, jug) == 0.95
    finally:
        dispose_env(live)
        dispose_env(other)
