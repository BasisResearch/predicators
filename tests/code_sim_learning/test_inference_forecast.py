"""Future draws match conditional laws without receiving future readings."""
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
ANGLES = (("robot", "robot", "roll"), ("robot", "robot", "tilt"),
          ("robot", "robot", "wrist"))


@pytest.mark.parametrize("persistence", [0., .8])
@pytest.mark.parametrize("sensor_sigma", [0., .07])
def test_forecast_matches_dense_conditional_gaussian(persistence,
                                                     sensor_sigma):
    """Both cross-time covariance and means match a Schur-complement
    reference."""
    process = GaussianOutputError(persistence, .12, .2)
    model = OutputObservationModel(SensorModel(
        (SensorFeature(POSITION, sensor_sigma), )),
                                   scalars=(ScalarOutputFactor(
                                       POSITION, process), ))
    native = np.array([.0, .1, .2, .3, .4])
    predictions = tuple(
        Observation(i, ((POSITION, v), )) for i, v in enumerate(native))
    readings = np.array([.3, -.1])
    prefix = tuple(
        Observation(i, ((POSITION, v), )) for i, v in enumerate(readings))
    transition = np.array(
        [[persistence**(t - k) if k <= t else 0. for k in range(5)]
         for t in range(5)])
    covariance = transition @ np.diag([.2**2] + [.12**2] * 4) @ transition.T
    covariance += sensor_sigma**2 * np.eye(5)
    cross = covariance[2:, :2]
    mean = native[2:] + cross @ np.linalg.solve(covariance[:2, :2],
                                                readings - native[:2])
    conditional_cov = covariance[2:, 2:] - \
        cross @ np.linalg.solve(covariance[:2, :2], cross.T)
    rng = np.random.default_rng(52)
    draws = np.array([[
        dict(o.values)[POSITION]
        for o in model.sample_future(predictions, prefix, rng)
    ] for _ in range(6000)])
    assert draws.mean(axis=0) == pytest.approx(mean, abs=.012)
    assert np.cov(draws.T) == pytest.approx(conditional_cov, abs=.004)
    first = model.sample_future(predictions, prefix, np.random.default_rng(3))
    repeated = model.sample_future(predictions, prefix,
                                   np.random.default_rng(3))
    assert first == repeated
    assert [o.step for o in model.sample_future(predictions, prefix, rng)] == \
        [2, 3, 4]


def test_unconditioned_draw_retains_original_initial_error():
    """No prefix uses b0's original variance without an extra transition."""
    model = OutputObservationModel(SensorModel((SensorFeature(POSITION,
                                                              0.), )),
                                   scalars=(ScalarOutputFactor(
                                       POSITION,
                                       GaussianOutputError(.8, .01, .3)), ))
    predictions = (Observation(0, ((POSITION, .4), )), )
    rng = np.random.default_rng(71)
    values = np.array([
        dict(model.sample_future(predictions, (), rng)[0].values)[POSITION]
        for _ in range(6000)
    ])
    assert values.mean() == pytest.approx(.4, abs=.015)
    assert values.var() == pytest.approx(.09, abs=.005)


def test_forecast_preserves_events_and_derived_observation_relationship():
    """A checked display follows its noisy-error source, not native state."""
    sensor = SensorModel(
        tuple(SensorFeature(k, 0.) for k in (POSITION, DISPLAY, EVENT)))
    model = OutputObservationModel(sensor,
                                   scalars=(ScalarOutputFactor(
                                       POSITION,
                                       GaussianOutputError(.8, .1, .2)), ),
                                   readouts=(CheckedReadoutFactor(
                                       ExactReadout(POSITION, DISPLAY,
                                                    sensor.digest, "a" * 64),
                                       lambda value: 2 * value + 1), ))
    predictions = (Observation(0, ((POSITION, 0.), (EVENT, 0.))),
                   Observation(1, ((POSITION, 0.), (EVENT, 1.))))
    prefix = (Observation(0, ((POSITION, .3), (DISPLAY, 1.6), (EVENT, 0.))), )
    draw = model.sample_future(predictions, prefix, np.random.default_rng(2))
    values = dict(draw[0].values)
    assert values[EVENT] == 1.
    assert values[DISPLAY] == 2 * values[POSITION] + 1
    assert values[POSITION] != 0.
    assert math.isfinite(model.log_likelihood(predictions, prefix + draw))
    assert dict(predictions[1].values)[POSITION] == 0.
    false_prefix = (Observation(0, ((POSITION, .3), (DISPLAY, 1.6),
                                    (EVENT, 1.))), )
    with pytest.raises(UnsupportedConditioning,
                       match="zero-likelihood prefix"):
        model.sample_future(predictions, false_prefix,
                            np.random.default_rng(2))


def test_euler_forecast_uses_unnormalized_antipodal_readout():
    """Near a pitch pole, raw draws retain pole mass and both yaw branches."""
    model = OutputObservationModel(
        SensorModel(tuple(SensorFeature(k, 0.) for k in ANGLES)),
        eulers=(EulerOutputFactor(ANGLES, QuaternionOutputError(.03)), ))
    prediction = (Observation(0, tuple(zip(ANGLES, (0., math.pi / 2, 0.)))), )
    rng = np.random.default_rng(14)
    values = [
        dict(model.sample_future(prediction, (), rng)[0].values)
        for _ in range(6000)
    ]
    poles = [row for row in values if row[ANGLES[1]] == math.pi / 2]
    # Normalizing raw quaternions would remove almost all of this pole mass.
    assert .4 < len(poles) / len(values) < .6
    assert all(row[ANGLES[0]] == 0. for row in poles)
    assert .4 < np.mean([abs(row[ANGLES[2]]) > math.pi for row in poles]) < .6
    for row in values[:8]:
        assert math.isfinite(
            model.log_likelihood(prediction,
                                 (Observation(0, tuple(row.items())), )))
    changed = OutputObservationModel(model.sensor,
                                     eulers=(EulerOutputFactor(
                                         ANGLES,
                                         QuaternionOutputError(.03, .99)), ))
    with pytest.raises(UnsupportedConditioning, match="pole threshold"):
        changed.sample_future(prediction, (), rng)


def test_forecast_requires_explicit_input_and_complete_prediction():
    """Known inputs are copied; absent inputs and malformed histories fail."""
    sensor = SensorModel(
        (SensorFeature(POSITION, 0.,
                       conditioned=True), SensorFeature(EVENT, 0.)))
    model = OutputObservationModel(sensor)
    prediction = (Observation(0, ((POSITION, .4), (EVENT, 1.))), )
    rng = np.random.default_rng(1)
    assert model.sample_future(prediction, (), rng) == prediction
    assert not model.sample_future(prediction, prediction, rng)
    for incomplete in ((Observation(0, ((EVENT, 1.), )), ),
                       (Observation(1, prediction[0].values), ), ()):
        with pytest.raises(ValueError):
            model.sample_future(incomplete, (), rng)
    with pytest.raises(ValueError):
        model.sample_future(prediction, prediction * 2, rng)
