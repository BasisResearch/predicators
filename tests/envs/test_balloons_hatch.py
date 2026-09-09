"""Executable hatch passages and their observable simulator state."""
# pylint: disable=protected-access
import pytest

from predicators import utils
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.ground_truth_models.balloons.gt_simulator_env import \
    BalloonsResidualEnv
from predicators.ground_truth_models.balloons.options import \
    PyBulletBalloonsGroundTruthOptionFactory, probe_release_option, \
    release_params
from predicators.observation_noise import ObservationNoise, step_rng
from predicators.settings import CFG


def _config():
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 0,
        "balloons_scene": "hatch",
        "balloons_hatch_z": .57,
        "balloons_hatch_half_gap": .085,
        "balloons_probe_max_steps": 500,
        "skill_phase_use_motion_planning": False,
        "continual_obs_noise_position": .01,
        "continual_obs_noise_orientation": .02,
    })


def test_hatch_orientation_is_observable_and_noisy():
    """Tilt survives state reconstruction and uses the declared noise class."""
    _config()
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(0, [0, 1, 3, 2], (.8044, .8544))
        state.set(env._box, "pitch", .3)
        state.set(env._box, "z", .8)
        env._set_state(state)
        actual = env._get_state()
        assert abs(actual.get(env._box, "pitch") - .3) < 1e-6
        noise = ObservationNoise.from_cfg()
        observed = noise.perturb(actual, step_rng(0, 0, 0, 0))
        assert noise.feature_sigma(env._box.type, "pitch") == .02
        assert observed.get(env._box, "pitch") != actual.get(env._box, "pitch")
        assert abs(actual.get(env._box, "pitch") - .3) < 1e-6
        assert observed.get(env._box, "color") == actual.get(env._box, "color")
    finally:
        env.dispose()


def test_hatch_release_orders_are_executable():
    """Both immediate orders of the example reference clear the hatch."""
    _config()
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(0, [0, 1, 3, 2], (.8044, .8544))
        for order in ((1, 3), (3, 1)):
            result = env.release_sequence_outcome(state, order)
            assert result.won, (order, result)
            assert result.steps < 150
    finally:
        env.dispose()


@pytest.mark.parametrize("motion_planning", [False, True])
def test_hatch_certification_matches_public_release(motion_planning):
    """Certifying seed 4 must reproduce the controller's release timing."""
    _config()
    utils.update_config({
        "seed": 4,
        "skill_phase_use_motion_planning": motion_planning,
    })
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(1, [2, 3, 1], (.70412, .75412))
        # Realized initial pose from the failed continual oracle, including
        # the small IK offset from the nominal home pose.
        for feature, value in {
                "x": .7498313188552856,
                "y": 1.1003972291946411,
                "z": .8499622941017151,
                "fingers": .03999999910593033,
                "wrist": -1.570920132493705,
        }.items():
            state.set(env._robot, feature, value)
        env._set_state(state)
        current = env._get_state()
        env._current_observation = current
        options = PyBulletBalloonsGroundTruthOptionFactory.get_options(
            env.get_name(), {t.name: t
                             for t in env.types}, {}, env.action_space)
        release = next(o for o in options if o.name == "Release")
        steps = 0
        won = False
        for index in (1, 2):
            option = release.ground([env._robot, env._clips[index]],
                                    release_params())
            assert option.initiable(current)
            for _ in range(200):
                if option.terminal(current) or won:
                    break
                current = env.step(option.policy(current))
                steps += 1
                won = env._InBand_holds(current, [env._box, env._band])
            else:
                raise AssertionError("Public Release did not terminate")
        for _ in range(200):
            if won:
                break
            current = env.step(env._hold_action())
            steps += 1
            won = env._InBand_holds(current, [env._box, env._band])
        assert won == (not motion_planning)
        certified = env.release_sequence_outcome(state, (1, 2))
        assert certified.won == won, (certified, steps)
        if won:
            assert certified.steps == steps
        else:
            assert abs(certified.height - current.get(env._box, "z")) < .001
        assert env.release_sequence_outcome(state, (2, 1)).won
    finally:
        env.dispose()


def test_hatch_contact_decoy_is_causal_and_order_dependent():
    """A witnessed contact jam can recover when the release order changes."""
    _config()
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(0, [0, 1, 3, 2], (.8044, .8544))
        for order in ((0, 2), ):
            result = env.release_sequence_outcome(state, order)
            assert result.jammed, result
            assert result.height < CFG.balloons_hatch_z
        assert env.release_sequence_outcome(state, (2, 0)).won
    finally:
        env.dispose()


@pytest.mark.parametrize("release_order,expect_clear", [((1, 3), True),
                                                        ((0, 2), False)])
def test_hatch_subclass_matches_actual_release_actions(release_order,
                                                       expect_clear):
    """Fresh real and subclass replay reproduce the recorded passage."""
    _config()
    real = PyBulletBalloonsEnv(use_gui=False)
    model = BalloonsResidualEnv(use_gui=False)
    replay = PyBulletBalloonsEnv(use_gui=False)
    try:
        params = {
            f"mass_{name}": real.true_box_mass(i)
            for i, (name, _) in enumerate(real.BOX_PALETTE)
        }
        params["air_drag"] = CFG.balloons_drag
        values = {
            spec.name: spec.init_value
            for spec in model.AGENT_PARAM_SPECS
        }
        values.update({
            f"lift_{name}": CFG.balloons_lifts[i]
            for i, (name, _) in enumerate(real.BALLOON_PALETTE)
        })
        values["fade_height"] = CFG.balloons_fade_height
        values.update({k: v for k, v in params.items() if k in values})
        model.apply_physical_param_overrides(values)
        real._set_state(real.level_state(0, [0, 1, 3, 2], (.8044, .8544)))
        truth = real._get_state()
        real._current_observation = truth
        prediction = truth
        recorded = [truth]
        max_position_error = 0.0
        max_pitch_error = 0.0
        actions = []
        for index in release_order:
            option = probe_release_option().ground(
                [real._robot, real._clips[index]], release_params())
            assert option.initiable(truth)
            for _ in range(200):
                if option.terminal(truth):
                    break
                action = option.policy(truth)
                actions.append(action)
                truth = real.simulate(truth, action)
                recorded.append(truth)
            else:
                raise AssertionError("Release did not terminate")
        for _ in range(100):
            action = real._hold_action()
            actions.append(action)
            truth = real.simulate(truth, action)
            recorded.append(truth)
        # Keep the original continuous trajectory as the reference.
        replayed = prediction
        for truth, action in zip(recorded[1:], actions):
            replayed = replay.simulate(replayed, action)
            prediction = model.simulate(prediction, action)
            for feature in ("x", "y", "z", "pitch"):
                assert abs(
                    replayed.get(real._box, feature) -
                    truth.get(real._box, feature)) < 1e-4
            max_position_error = max(
                max_position_error,
                max(
                    abs(
                        truth.get(real._box, f) - prediction.get(real._box, f))
                    for f in ("x", "y", "z")))
            max_pitch_error = max(
                max_pitch_error,
                abs(
                    truth.get(real._box, "pitch") -
                    prediction.get(real._box, "pitch")))
        assert max_position_error < .005, max_position_error
        assert max_pitch_error < .05, max_pitch_error
        assert (truth.get(real._box, "z") > .7) == expect_clear
        assert (prediction.get(real._box, "z") > .7) == expect_clear
    finally:
        model.dispose()
        replay.dispose()
        real.dispose()
