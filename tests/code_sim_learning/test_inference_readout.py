"""A redundant readout must neither add independent evidence nor hide
errors."""
import math
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    UnsupportedConditioning
from predicators.code_sim_learning.inference_data import Observation, \
    SensorFeature, SensorModel, content_digest
from predicators.code_sim_learning.inference_output_error import \
    GaussianOutputError, output_error_likelihood
from predicators.code_sim_learning.inference_readout import ExactReadout, \
    reduce_exact_readout

SOURCE = ("robot", "joint_positions", "7")
OUTPUT = ("robot", "robot", "fingers")


def _sensor() -> SensorModel:
    return SensorModel((SensorFeature(SOURCE, 0.), SensorFeature(OUTPUT, 0.)))


def _rule(sensor: SensorModel) -> ExactReadout:
    return ExactReadout(SOURCE, OUTPUT, sensor.digest,
                        content_digest(b"specified deterministic readout"))


def test_source_density_is_retained_once_without_readout_jacobian() -> None:
    """Changing a redundant display's scale must not change source
    inference."""
    sensor = _sensor()
    process = GaussianOutputError(.8, .1, .2)
    reference = output_error_likelihood(process, (0., ), (.4, ), 0.)
    for scale in (2., 50.):

        def transform(value: float, factor: float = scale) -> float:
            return factor * value

        observation = Observation(0, ((SOURCE, .4), (OUTPUT, scale * .4)))
        result = reduce_exact_readout(observation, sensor, _rule(sensor),
                                      transform)
        assert result.status == "verified"
        assert result.observation == Observation(0, ((SOURCE, .4), ))
        assert result.log_factor == 0
        value = dict(result.observation.values)[SOURCE]
        likelihood = output_error_likelihood(process, (0., ), (value, ), 0.)
        assert likelihood.log_likelihood + result.log_factor == \
            reference.log_likelihood
        assert dict(observation.values)[OUTPUT] == scale * .4


def test_precision_and_inconsistent_readout_are_not_softened() -> None:
    """A float32 display is a deterministic map, not an independent sensor."""
    sensor = _sensor()
    joint = .0123456789
    displayed = float(np.float32(joint))
    observation = Observation(3, ((SOURCE, joint), (OUTPUT, displayed)))
    result = reduce_exact_readout(observation, sensor, _rule(sensor),
                                  lambda value: float(np.float32(value)))
    assert result.status == "verified"
    wrong = reduce_exact_readout(observation, sensor, _rule(sensor),
                                 lambda value: value)
    assert wrong.status == "exact_contradiction"
    assert wrong.log_factor == -math.inf
    assert wrong.observation is None


def test_missing_noisy_or_unknown_sources_cannot_be_discarded() -> None:
    """A readout may still inform an absent or noisy source quantity."""
    sensor = _sensor()
    observation = Observation(0, ((OUTPUT, .3), ))
    with pytest.raises(UnsupportedConditioning, match="missing"):
        reduce_exact_readout(observation, sensor, _rule(sensor),
                             lambda value: value)
    noisy = SensorModel((SensorFeature(SOURCE, .1), SensorFeature(OUTPUT, 0.)))
    with pytest.raises(UnsupportedConditioning, match="exact"):
        reduce_exact_readout(observation, noisy, _rule(noisy),
                             lambda value: value)
    with pytest.raises(ValueError, match="sensor"):
        reduce_exact_readout(observation, noisy, _rule(sensor),
                             lambda value: value)
    absent = Observation(1, ((SOURCE, .4), ))
    result = reduce_exact_readout(absent, sensor, _rule(sensor),
                                  lambda value: value)
    assert result.status == "not_observed"
    assert result.observation is absent
    with pytest.raises(ValueError):
        replace(_rule(sensor), output=SOURCE)


def test_malformed_callback_is_not_an_observation_contradiction() -> None:
    """Invalid mapping implementations propagate as errors."""
    sensor = _sensor()
    observation = Observation(0, ((SOURCE, .3), (OUTPUT, .3)))
    with pytest.raises(ValueError, match="nonfinite"):
        reduce_exact_readout(observation, sensor, _rule(sensor),
                             lambda _: float("nan"))
