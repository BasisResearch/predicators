"""The balloons model in the SUBCLASS form: it loads, exposes its parameters
for system-ID, matches the real env step for step, and its parameters are
identifiable from a recorded trajectory.

The subclass form (``gt_simulator_env.py``) is an alternative to the
rule form (``gt_simulator.py``): a subclass of the visible base-sim env
that overrides ``_domain_specific_step`` with full engine access and
declares its learnable constants in ``AGENT_PARAM_SPECS``. These tests
prove the form can express the domain and be fit, without touching the
registered rule-form simulator.
"""
# pylint: disable=protected-access
from __future__ import annotations

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.rollout_env import rollout_states
from predicators.code_sim_learning.utils import read_residual_env
from predicators.settings import CFG
from predicators.structs import Action


@pytest.fixture(scope="module", name="setup")
def _setup():
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "skill_phase_use_motion_planning": False,
    })
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
    from predicators.ground_truth_models.balloons import gt_simulator_env

    # pylint: enable=import-outside-toplevel
    cls = read_residual_env(vars(gt_simulator_env))
    real = PyBulletBalloonsEnv(use_gui=False)
    true_params = {
        gt_simulator_env._lift_param_name(i): float(CFG.balloons_lifts[i])
        for i in range(len(real.BALLOON_PALETTE))
    }
    true_params["fade_height"] = float(CFG.balloons_fade_height)
    for i in range(2):
        true_params[gt_simulator_env._mass_param_name(i)] = \
            real.true_box_mass(i)
    true_params["air_drag"] = float(CFG.balloons_drag)
    return real, cls, true_params


def _hold(state, env):
    arr = np.array(list(state.joint_positions), dtype=np.float32)
    n = env.action_space.shape[0]
    if arr.shape[0] < n:
        arr = np.concatenate(
            [arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
    return Action(arr)


def _level(real, colors, band, open_clips):
    state = real.level_state(1, colors, band)
    real._pybullet_robot.set_joints(
        real._pybullet_robot.initial_joint_positions)
    real._set_state(state)
    init = real._get_state()
    real._current_observation = init
    init = init.copy()
    for i in open_clips:
        init.set(real._clips[i], "is_on", 1.0)
    return init


def test_subclass_form_loads_and_exposes_its_parameters(setup):
    """read_residual_env finds the class, and its AGENT_PARAM_SPECS are
    surfaced for system-ID and round-trip through agent_param."""
    real, cls, true_params = setup
    assert cls is not None and issubclass(cls, type(real).__bases__[0])
    inst = cls(use_gui=False, skip_residual_dynamics=False)
    info = inst.get_physical_param_info()
    assert set(true_params).issubset(set(info))
    inst.apply_physical_param_overrides({"fade_height": 1.23})
    assert abs(inst.agent_param("fade_height") - 1.23) < 1e-9
    # A name it does not own still raises, exactly as a stock env would.
    with pytest.raises((ValueError, NotImplementedError)):
        inst.apply_physical_param_overrides({"not_a_param": 1.0})


def test_subclass_form_matches_the_real_env(setup):
    """With the true parameters, the subclass env's box tracks the real env's
    to a few millimetres through an all-at-once release."""
    real, cls, true_params = setup
    model = cls(use_gui=False, skip_residual_dynamics=False)
    model.apply_physical_param_overrides(true_params)
    colors = [0, 3, 1]  # red, gold, blue
    expected = real.hover_height(1, colors)
    assert expected is not None
    band = (expected - 0.025, expected + 0.025)
    init = _level(real, colors, band, open_clips=[0, 1, 2])
    box = real._box
    s_real, s_model = init, init
    max_gap = 0.0
    for _ in range(250):
        s_real = real.simulate(s_real, _hold(s_real, real))
        s_model = model.simulate(s_model, _hold(s_model, model))
        max_gap = max(max_gap,
                      abs(s_real.get(box, "z") - s_model.get(box, "z")))
    assert max_gap < 0.005, max_gap
    assert abs(s_model.get(box, "z") - expected) < 0.03


def test_subclass_form_parameters_are_identifiable(setup):
    """A recorded release is reproduced far better at the true parameters than
    at wrong ones, so the rollout system-ID has signal to fit."""
    real, cls, true_params = setup
    colors = [0, 3, 1]
    expected = real.hover_height(1, colors)
    init = _level(real,
                  colors, (expected - 0.025, expected + 0.025),
                  open_clips=[0, 1, 2])
    box = real._box
    states = [init]
    actions = []
    s = init
    for _ in range(120):
        a = _hold(s, real)
        s = real.simulate(s, a)
        states.append(s)
        actions.append(a)

    def sse_at(params):
        factory = lambda: cls(use_gui=False, skip_residual_dynamics=False)
        sim_states = rollout_states(factory, states[0], actions, params)
        return float(
            np.mean([
                (sim_states[i].get(box, "z") - states[i + 1].get(box, "z"))**2
                for i in range(len(actions))
            ]))

    true_sse = sse_at(true_params)
    wrong = dict(true_params)
    wrong["lift_gold"] = true_params["lift_gold"] * 0.5
    wrong["mass_oak"] = true_params["mass_oak"] * 2.0
    wrong_sse = sse_at(wrong)
    assert true_sse < 1e-4, true_sse
    assert wrong_sse > 20 * max(true_sse, 1e-9), (true_sse, wrong_sse)
