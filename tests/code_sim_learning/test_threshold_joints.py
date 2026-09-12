"""Threshold observations retain endpoint mass and uncertain joint motion."""
import math

import numpy as np
import pytest

from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_joints import \
    IncompatibleJointObservation, JointCoordinateBoundary, RestingJointPrior
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch


@pytest.mark.parametrize("observed", [False, True])
def test_threshold_conditional_moments(observed: bool) -> None:
    """The exact event changes both the endpoint and motion mixture weights."""
    prior = RestingJointPrior("lever", -1., 2., (-1., 2.), .6, .4)
    conditional = prior.condition_above(-.3, observed)
    lower, upper = (-.3, 2.) if observed else (-1., -.3)
    probability = .3 + .4 * (upper - lower) / 3
    rest = .3 / probability
    endpoint = 2. if observed else -1.
    expected_mean = rest * endpoint + (1 - rest) * (lower + upper) / 2
    expected_second = rest * endpoint**2 + \
        (1 - rest) * (lower**2 + lower * upper + upper**2) / 3
    assert math.exp(conditional.log_observation_factor) == \
        pytest.approx(probability)
    values = np.array([
        conditional.lift(point)
        for point in np.random.default_rng(42).random((8192, 2))
    ])
    assert np.all((values[:, 0] > -.3) == observed)
    assert values[:, 0].mean() == pytest.approx(expected_mean, abs=.03)
    assert (values[:, 0]**2).mean() == pytest.approx(expected_second, abs=.05)
    assert np.mean(values[:, 0] == endpoint) == pytest.approx(rest, abs=.02)
    assert values[:, 1].mean() == pytest.approx(0., abs=.01)
    assert (values[:, 1]**2).mean() == \
        pytest.approx((1 - rest) * .4**2 / 3, abs=.003)


def test_parameter_dependent_threshold_keeps_event_evidence() -> None:
    """The joint posterior matches analytic moments when the cut is unknown."""
    lever = RestingJointPrior("lever", 0., 1., (0., 1.), .6, .4)
    box = BoxPrior(("threshold", "position_unit", "velocity_unit"),
                   ((0., 1.), ) * 3)
    digest = content_digest(b"parameter dependent threshold observation")
    prior = ConditionedPrior(("threshold", "position", "velocity"),
                             lever.digest, digest, box)
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)

    def condition(point: np.ndarray) -> PriorPoint:
        distribution = lever.condition_above(float(point[0]), True)
        position, velocity = distribution.lift(point[1:])
        return PriorPoint((float(point[0]), position, velocity),
                          distribution.log_observation_factor)

    result = sample_batch(prior,
                          identity,
                          lambda _: 0.,
                          SamplerConfig(particles=2048,
                                        temperatures=8,
                                        moves=4,
                                        max_evaluations=70000),
                          19,
                          condition=condition)
    assert result.status == "complete"
    values = np.asarray(result.samples)
    assert np.all(values[:, 1] > values[:, 0])
    means = np.average(values, axis=0, weights=result.weights)
    np.testing.assert_allclose(means, [13 / 30, 13 / 15, 0.], atol=.02)
    endpoint_mass = np.dot(values[:, 1] == 1., result.weights)
    assert endpoint_mass == pytest.approx(.6, abs=.03)


def test_threshold_prior_limits_and_boundaries() -> None:
    """Deterministic rests and pure moving components remain explicit."""
    for mass in (0., 1.):
        prior = RestingJointPrior("lever", 0., 1., (0., 1.), mass, .4)
        off, on = (prior.condition_above(.2, flag) for flag in (False, True))
        assert math.exp(off.log_observation_factor) + \
            math.exp(on.log_observation_factor) == pytest.approx(1.)
        point = np.array([.5, .75])
        if mass == 1.:
            assert off.lift(point) == (0., 0.)
            assert on.lift(point) == (1., 0.)
        else:
            assert on.lift(point) == pytest.approx((.6, .2))
        with pytest.raises(JointCoordinateBoundary):
            on.lift(np.array([0., .5]))
        for threshold in (0., 1., float("nan")):
            with pytest.raises(ValueError, match="interior"):
                prior.condition_above(threshold, True)
    with pytest.raises(ValueError, match="Invalid resting"):
        RestingJointPrior("lever", 0., 1., (0., 1.), 1.1, .4)


def test_controller_poses_inside_mechanical_travel() -> None:
    """Conditioning can select zero, one or both interior resting poses."""
    prior = RestingJointPrior("lever", 0., 1., (.2, .8), .6, .4)
    on = prior.condition_above(.9, True)
    assert math.exp(on.log_observation_factor) == pytest.approx(.04)
    assert on.rest_probability == 0.
    assert on.lift(np.array([.5, .75])) == pytest.approx((.95, .2))
    off = prior.condition_above(.9, False)
    assert math.exp(off.log_observation_factor) == pytest.approx(.96)
    assert off.rest_probability == pytest.approx(.625)
    assert off.lift(np.array([.1, .75])) == (.2, 0.)
    assert off.lift(np.array([.5, .75])) == (.8, 0.)
    resting = RestingJointPrior("lever", 0., 1., (.2, .8), 1., .4)
    with pytest.raises(IncompatibleJointObservation):
        resting.condition_above(.9, True)
