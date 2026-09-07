"""Tests for the observation-noise channel
(predicators/observation_noise.py)."""

import numpy as np

from predicators import utils
from predicators.observation_noise import ObservationNoise, noise_or_none, \
    step_rng
from predicators.structs import Object, State, Type
from predicators.utils import PyBulletState


def _scene() -> State:
    block = Type("block", ["x", "y", "z", "yaw", "is_held"],
                 angular_features=["yaw"])
    robot = Type("robot", ["x", "y", "z", "fingers"])
    b = Object("b", block)
    r = Object("r", robot)
    return State(
        {
            b: np.array([0.1, 0.2, 0.3, 1.0, 1.0]),
            r: np.array([0.5, 0.6, 0.7, 0.04]),
        },
        simulator_state={"handle": 7},
        latent={"cure": 3},
        privileged={"truth": 1})


def test_perturb_touches_only_the_noisy_classes() -> None:
    """Positions and orientations of non-robot objects move; discrete features,
    the robot and the latent block are copied; the env-only channels are
    dropped."""
    noise = ObservationNoise(position=0.01, orientation=0.1)
    truth = _scene()
    b, r = sorted(truth, key=lambda o: o.name)
    view = noise.perturb(truth, step_rng(1, 0, 0, 0))
    assert view is not truth
    for feat in ("x", "y", "z", "yaw"):
        assert view.get(b, feat) != truth.get(b, feat)
    assert abs(view.get(b, "x") - truth.get(b, "x")) < 0.1
    assert view.get(b, "is_held") == truth.get(b, "is_held")
    assert np.array_equal(view[r], truth[r])
    assert view.latent == {"cure": 3} and view.latent is not truth.latent
    # An opaque simulator state (a handle) is dropped with the privileged
    # block; the robot's joint data would be kept (see the PyBullet case).
    assert view.simulator_state is None and view.privileged is None
    # The truth was not touched in place.
    assert truth.get(b, "x") == 0.1 and truth.simulator_state == {"handle": 7}
    # One class alone leaves the other exact.
    only_pos = ObservationNoise(position=0.01).perturb(truth,
                                                       step_rng(1, 0, 0, 0))
    assert only_pos.get(b, "yaw") == 1.0 and only_pos.get(b, "x") != 0.1


def test_perturb_keeps_the_pybullet_joint_data() -> None:
    """A PyBullet state stays a PyBullet state with the robot's joints (exact
    proprioception, which the base simulator reads every step) and without the
    engine handles."""
    noise = ObservationNoise(position=0.01)
    truth = _scene()
    b = sorted(truth, key=lambda o: o.name)[0]
    live = PyBulletState(dict(truth.data),
                         simulator_state={
                             "joint_positions": [0.1, 0.2, 0.3],
                             "physics_client_id": 4,
                             "body_ids": {
                                 "b": 9
                             },
                         })
    view = noise.perturb(live, step_rng(0, 0, 0, 0))
    assert isinstance(view, PyBulletState)
    assert view.joint_positions == [0.1, 0.2, 0.3]
    assert set(view.simulator_state) == {"joint_positions"}
    assert view.get(b, "x") != truth.get(b, "x")
    assert live.simulator_state["physics_client_id"] == 4


def test_draws_are_keyed_by_the_step() -> None:
    """The same (seed, level, episode, step) always observes the same frame;
    any other coordinate observes a different one."""
    noise = ObservationNoise(position=0.01)
    truth = _scene()
    b = sorted(truth, key=lambda o: o.name)[0]
    a1 = noise.perturb(truth, step_rng(5, 2, 1, 9))
    a2 = noise.perturb(truth, step_rng(5, 2, 1, 9))
    assert np.array_equal(a1[b], a2[b])
    for other in ((6, 2, 1, 9), (5, 3, 1, 9), (5, 2, 0, 9), (5, 2, 1, 8)):
        assert not np.array_equal(a1[b],
                                  noise.perturb(truth, step_rng(*other))[b])


def test_feature_sigma_and_residual_scale() -> None:
    """The residual scale folds the feature's sigma into the fit's.

    variance model: std^2 = (noise_sigma * motion)^2 + sigma_f^2.
    """
    noise = ObservationNoise(position=0.02, orientation=0.1)
    block = Type("block", ["x", "yaw", "spin", "is_held"],
                 angular_features=["spin"])
    robot = Type("robot", ["x"])
    assert noise.feature_sigma(block, "x") == 0.02
    assert noise.feature_sigma(block, "yaw") == 0.1
    # A Type's own angular features are orientations too.
    assert noise.feature_sigma(block, "spin") == 0.1
    assert noise.feature_sigma(block, "is_held") == 0.0
    assert noise.feature_sigma(robot, "x") == 0.0
    # Exact features keep the motion scale.
    assert noise.residual_scale(block, "is_held", 0.4, 0.05) == 0.4
    assert noise.residual_scale(robot, "x", 0.4, 0.05) == 0.4
    expected = np.sqrt(0.4**2 + (0.02 / 0.05)**2)
    assert abs(noise.residual_scale(block, "x", 0.4, 0.05) - expected) < 1e-12
    expected = np.sqrt(np.pi**2 + (0.1 / 0.05)**2)
    assert abs(noise.residual_scale(block, "spin", np.pi, 0.05) -
               expected) < 1e-12


def test_flags_text_and_disabled_channel() -> None:
    """from_cfg reads the flags; zeros mean exact observations."""
    utils.reset_config({
        "continual_obs_noise_position": 0.005,
        "continual_obs_noise_orientation": 0.0,
        "continual_obs_noise_declared": False,
    })
    noise = ObservationNoise.from_cfg()
    assert noise.enabled and not noise.declared
    assert noise.position == 0.005 and noise.orientation == 0.0
    assert "0.005 m" in noise.describe() and "orientations" not in \
        noise.describe()
    assert "position sigma 0.005 m" in noise.summary()
    assert noise_or_none(noise) is noise
    utils.reset_config({})
    exact = ObservationNoise.from_cfg()
    assert not exact.enabled and exact.declared
    assert exact.describe() == "observations are exact"
    assert noise_or_none(exact) is None and noise_or_none(None) is None
    truth = _scene()
    view = exact.perturb(truth, step_rng(0, 0, 0, 0))
    assert view.allclose(State(dict(truth.data)))
