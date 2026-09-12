"""Causal future scores match independent conditional probability
references."""
import math

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    UnsupportedConditioning
from predicators.code_sim_learning.inference_data import Observation, \
    SensorFeature, SensorModel
from predicators.code_sim_learning.inference_observation import \
    CheckedReadoutFactor, EulerOutputFactor, OutputObservationModel, \
    ScalarOutputFactor
from predicators.code_sim_learning.inference_orientation import \
    QuaternionOutputError
from predicators.code_sim_learning.inference_output_error import \
    GaussianOutputError
from predicators.code_sim_learning.inference_readout import ExactReadout

POSITION = ("box", "body", "x")
DISPLAY = ("box", "body", "display")
EVENT = ("box", "body", "attached")
ANGLES = (("robot", "robot", "roll"), ("robot", "robot", "pitch"),
          ("robot", "robot", "yaw"))


@pytest.mark.parametrize("sensor_sigma", [0., .07])
@pytest.mark.parametrize("persistence", [-.6, 0., .8, 1.])
def test_future_density_matches_dense_gaussian(sensor_sigma, persistence):
    """A joint suffix density retains both prefix information and
    covariance."""
    model = OutputObservationModel(
        SensorModel((SensorFeature(POSITION, sensor_sigma), )),
        scalars=(ScalarOutputFactor(POSITION,
                                    GaussianOutputError(persistence, .12,
                                                        .2)), ))
    native = np.array([0., .1, .2, .3, .4])
    values = np.array([.3, -.1, .1, .35, .5])
    predictions = tuple(
        Observation(i, ((POSITION, v), )) for i, v in enumerate(native))
    # A missing reading still advances the process by one physical step.
    observations = tuple(
        Observation(i, () if i == 1 else ((POSITION, v), ))
        for i, v in enumerate(values))
    transition = np.array(
        [[persistence**(t - k) if k <= t else 0. for k in range(5)]
         for t in range(5)])
    covariance = transition @ np.diag([.2**2] + [.12**2] * 4) @ transition.T
    covariance += sensor_sigma**2 * np.eye(5)
    cross = covariance[2:, :1]
    mean = native[2:] + cross[:, 0] * (values[0] - native[0]) / covariance[0,
                                                                           0]
    conditional = covariance[2:, 2:] - cross @ cross.T / covariance[0, 0]
    delta = values[2:] - mean
    sign, logdet = np.linalg.slogdet(conditional)
    assert sign == 1.
    expected = -.5 * (3 * math.log(2 * math.pi) + logdet +
                      delta @ np.linalg.solve(conditional, delta))
    score = model.log_future_likelihood(predictions, observations[:2],
                                        observations[2:])
    assert score == pytest.approx(expected, abs=1e-12)
    assert score == pytest.approx(
        model.log_likelihood(predictions, observations) -
        model.log_likelihood(predictions[:2], observations[:2]),
        abs=1e-12)
    assert model.log_future_likelihood(predictions, (), observations) == \
        model.log_likelihood(predictions, observations)
    assert model.log_future_likelihood(predictions, observations, ()) == 0.


@pytest.mark.parametrize("use_error", [False, True])
def test_small_future_score_is_not_lost_to_prefix_cancellation(use_error):
    """Large finite prefix evidence must not round a future log density to
    0."""
    factors = (ScalarOutputFactor(POSITION,
                                  GaussianOutputError(0., 1., 1.)), ) \
        if use_error else ()
    model = OutputObservationModel(SensorModel(
        (SensorFeature(POSITION, 0. if use_error else 1.), )),
                                   scalars=factors)
    predictions = tuple(Observation(i, ((POSITION, 0.), )) for i in range(2))
    prefix = (Observation(0, ((POSITION, 1e12), )), )
    future = (Observation(1, ((POSITION, .3), )), )
    expected = -.5 * .3**2 - .5 * math.log(2 * math.pi)
    assert model.log_future_likelihood(predictions, prefix, future) == \
        pytest.approx(expected, abs=1e-14)
    assert model.log_likelihood(predictions, prefix + future) - \
        model.log_likelihood(predictions[:1], prefix) == 0.


def test_exact_future_failure_is_distinct_from_unsupported_prefix():
    """A rejected future is a prediction failure, not undefined
    conditioning."""
    sensor = SensorModel(
        tuple(SensorFeature(k, 0.) for k in (POSITION, DISPLAY, EVENT)))
    model = OutputObservationModel(sensor,
                                   scalars=(ScalarOutputFactor(
                                       POSITION,
                                       GaussianOutputError(1., 0., 0.)), ),
                                   readouts=(CheckedReadoutFactor(
                                       ExactReadout(POSITION, DISPLAY,
                                                    sensor.digest, "a" * 64),
                                       lambda value: 2 * value), ))
    predictions = tuple(
        Observation(i, ((POSITION, 0.), (EVENT, 0.))) for i in range(2))
    valid = tuple(
        Observation(i, ((POSITION, 0.), (DISPLAY, 0.), (EVENT, 0.)))
        for i in range(2))
    assert model.log_future_likelihood(predictions, valid[:1], valid[1:]) == 0.
    for key in (POSITION, DISPLAY, EVENT):
        values = dict(valid[1].values)
        values[key] = 1.
        if key == POSITION:
            values[DISPLAY] = 2.
        bad = (Observation(1, tuple(values.items())), )
        assert model.log_future_likelihood(predictions, valid[:1], bad) == \
            -math.inf
        bad_prefix = (Observation(0, bad[0].values), )
        with pytest.raises(UnsupportedConditioning, match="zero-likelihood"):
            model.log_future_likelihood(predictions, bad_prefix, valid[1:])
    with pytest.raises(ValueError, match="Matching nonempty"):
        model.log_future_likelihood(predictions, valid[:1], ())
    with pytest.raises(ValueError, match="each step"):
        model.log_future_likelihood(predictions, valid[:1], valid[:1])


def test_coupled_euler_scores_use_only_future_factors():
    """Native pole outputs retain their mixture factor without prefix reuse."""
    model = OutputObservationModel(
        SensorModel(tuple(SensorFeature(k, 0.) for k in ANGLES)),
        eulers=(EulerOutputFactor(ANGLES, QuaternionOutputError(.1)), ))
    values = tuple(zip(ANGLES, (0., math.pi / 2, 4.7)))
    predictions = tuple(Observation(i, values) for i in range(3))
    observations = tuple(
        Observation(i, tuple(zip(ANGLES, (0., math.pi / 2, 4.5 + .1 * i))))
        for i in range(3))
    # Euler errors are independent across time conditional on native poses.
    expected = math.fsum(
        model.log_likelihood((Observation(0, predictions[i].values), ), (
            Observation(0, observations[i].values), )) for i in (1, 2))
    assert math.isfinite(expected)
    assert model.log_future_likelihood(predictions, observations[:1],
                                       observations[1:]) == expected
    with pytest.raises(UnsupportedConditioning, match="Partial Euler"):
        model.log_future_likelihood(
            predictions, observations[:1],
            (Observation(1, ((ANGLES[0], 0.), )), observations[2]))
