"""Dense Gaussian integration checks for the marginalized discrepancy
process."""
import math

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_output_error import \
    ConstantOutputLikelihood, GaussianOutputError, \
    constant_output_likelihood, output_error_likelihood


@pytest.mark.parametrize("persistence", [0., -.4, .8, 1.])
@pytest.mark.parametrize("sensor_sigma", [0., .07])
def test_filter_matches_dense_history_integration(persistence: float,
                                                  sensor_sigma: float) -> None:
    """Marginal density and each causal conditional agree with a joint
    normal."""
    process = GaussianOutputError(persistence, .12, .2)
    predictions = (.1, .12, .17, .3, .45, .5)
    observations = (.15, None, .3, .38, None, .6)
    size = len(predictions)
    transition = np.array(
        [[persistence**(t - k) if k <= t else 0. for k in range(size)]
         for t in range(size)])
    covariance = transition @ np.diag([.2**2] + [.12**2] *
                                      (size - 1)) @ transition.T
    result = output_error_likelihood(process, predictions, observations,
                                     sensor_sigma)
    assert result.status == "complete"
    assert result.failed_step is None
    indices = [i for i, value in enumerate(observations) if value is not None]
    observed = np.array([observations[i] for i in indices])
    residual = observed - np.array(predictions)[indices]
    marginal_cov = covariance[np.ix_(
        indices, indices)] + sensor_sigma**2 * np.eye(len(indices))
    expected = -.5 * (len(indices) * math.log(2 * math.pi) +
                      np.linalg.slogdet(marginal_cov)[1] +
                      residual @ np.linalg.solve(marginal_cov, residual))
    assert result.log_likelihood == pytest.approx(expected, abs=1e-12)
    for step, state in enumerate(result.steps):
        used = [i for i in indices if i <= step]
        cov = covariance[np.ix_(used,
                                used)] + sensor_sigma**2 * np.eye(len(used))
        cross = covariance[step, used]
        values = np.array([observations[i]
                           for i in used]) - np.array(predictions)[used]
        mean = cross @ np.linalg.solve(cov, values)
        variance = covariance[step, step] - cross @ np.linalg.solve(cov, cross)
        assert state.filtered_mean == pytest.approx(mean, abs=1e-12)
        assert state.filtered_sigma**2 == pytest.approx(variance, abs=1e-12)


def test_exact_output_retains_density_and_unobserved_future_is_causal(
) -> None:
    """Exact readings eliminate error uncertainty without deleting the error
    law."""
    process = GaussianOutputError(.8, .1, .2)
    result = output_error_likelihood(process, (0., ) * 4, (.3, .4, None, None),
                                     0.)
    assert result.steps[0].filtered_sigma == 0
    assert result.steps[1].filtered_sigma == 0
    assert result.steps[2].predicted_mean == pytest.approx(.32)
    assert result.steps[3].predicted_mean == pytest.approx(.256)
    assert result.steps[3].predicted_sigma == pytest.approx(math.hypot(
        .08, .1))
    assert result.steps[2].log_observation_factor is None
    changed = output_error_likelihood(process, (0., ) * 4, (.3, .4, 20., -20.),
                                      0.)
    assert changed.steps[:2] == result.steps[:2]
    assert output_error_likelihood(process, (0., ) * 4, (.3, .4, None, None),
                                   0.) == result


def test_deterministic_reference_keeps_exact_contradictions() -> None:
    """An absent discrepancy process does not silently introduce a noise
    floor."""
    process = GaussianOutputError(.8, 0., 0.)
    result = output_error_likelihood(process, (1., 2., 3.), (1., 2.1, 3.), 0.)
    assert result.status == "exact_contradiction"
    assert result.log_likelihood == -math.inf
    assert result.failed_step == 1
    assert len(result.steps) == 1
    noisy = output_error_likelihood(process, (1., 2.), (1.1, 1.9), .2)
    expected = -(.1 / .2)**2 - 2 * math.log(.2 * math.sqrt(2 * math.pi))
    assert noisy.log_likelihood == pytest.approx(expected)
    assert all(step.filtered_sigma == 0 for step in noisy.steps)


def test_invalid_inputs_and_numeric_overflow_are_explicit() -> None:
    """Unsupported numerical ranges are not ordinary zero-likelihood states."""
    for args in ((1.1, .1, .1), (.9, -.1, .1), (.9, .1, float("inf"))):
        with pytest.raises(ValueError):
            GaussianOutputError(*args)
    process = GaussianOutputError(.8, .1)
    with pytest.raises(ValueError):
        output_error_likelihood(process, (0., ), (None, None), .1)
    with pytest.raises(ValueError):
        output_error_likelihood(process, (0., ), (float("nan"), ), .1)
    with pytest.raises(ValueError):
        output_error_likelihood(process, (0., ), (None, ), -.1)
    with pytest.raises(ConditioningNumericalError):
        output_error_likelihood(process, (-1e308, ), (1e308, ), .1)
    assert process.digest != GaussianOutputError(.7, .1).digest
    assert process.digest != GaussianOutputError(.8, .2).digest


@pytest.mark.parametrize("persistence", [-1., 0., .9, 1.])
@pytest.mark.parametrize("initial_sigma", [0., .12])
def test_constant_output_matches_dense_gaussian(persistence: float,
                                                initial_sigma: float) -> None:
    """Unknown static location has the independently integrated GLS density."""
    process = GaussianOutputError(persistence, .08, initial_sigma)
    readings = (None, .7, .8, None, .6, 1.)
    sensor_sigma = .05
    size = len(readings)
    transition = np.array(
        [[persistence**(t - k) if k <= t else 0. for k in range(size)]
         for t in range(size)])
    covariance = transition @ np.diag([initial_sigma**2] + [.08**2] *
                                      (size - 1)) @ transition.T
    indices = [i for i, v in enumerate(readings) if v is not None]
    observed = np.array([readings[i] for i in indices], dtype=float)
    covariance = covariance[np.ix_(indices, indices)] + \
        sensor_sigma**2 * np.eye(len(indices))
    ones = np.ones(len(indices))
    precision = ones @ np.linalg.solve(covariance, ones)
    center = ones @ np.linalg.solve(covariance, observed) / precision
    result = constant_output_likelihood(process, readings, sensor_sigma)
    assert result.center == pytest.approx(center, abs=1e-12)
    assert result.sigma == pytest.approx(precision**-.5, abs=1e-12)
    for location in (-.2, result.center, 1.3):
        residual = observed - location
        expected = -.5 * (len(indices) * math.log(2 * math.pi) +
                          np.linalg.slogdet(covariance)[1] +
                          residual @ np.linalg.solve(covariance, residual))
        assert result.log_likelihood(location) == pytest.approx(expected,
                                                                abs=1e-10)
        original = output_error_likelihood(process, (location, ) * size,
                                           readings, sensor_sigma)
        assert result.log_likelihood(location) == pytest.approx(
            original.log_likelihood, abs=1e-10)


def test_constant_output_independent_readings_and_translation() -> None:
    """The independent special case is an average; offsets preserve its law."""
    readings = (.9, 1.1, 1., 1.2)
    process = GaussianOutputError(.9, 0., 0.)
    result = constant_output_likelihood(process, readings, .2)
    assert result.center == pytest.approx(1.05)
    assert result.sigma == pytest.approx(.1)
    shifted = constant_output_likelihood(process,
                                         tuple(v + 1e6 for v in readings), .2)
    assert shifted.center - 1e6 == pytest.approx(result.center, abs=1e-10)
    assert shifted.sigma == result.sigma
    assert shifted.log_peak == pytest.approx(result.log_peak, abs=1e-8)
    correlated = constant_output_likelihood(GaussianOutputError(.9, .2, 0.),
                                            readings, .2)
    assert correlated.sigma > result.sigma


def test_constant_output_requires_usable_noisy_evidence() -> None:
    """Exact constraints and unsupported numeric scales get explicit errors."""
    process = GaussianOutputError(.9, .005, 0.)
    for readings in ((), (None, None), (float("nan"), )):
        with pytest.raises(ValueError):
            constant_output_likelihood(process, readings, .005)
    for sigma in (0., -.1, float("inf")):
        with pytest.raises(ValueError):
            constant_output_likelihood(process, (1., ), sigma)
    for sigma in (1e-300, 1e300):
        with pytest.raises(ConditioningNumericalError):
            constant_output_likelihood(process, (1., ), sigma)
    with pytest.raises(ConditioningNumericalError):
        constant_output_likelihood(process, (1e308, -1e308), .005)
    with pytest.raises(ValueError):
        ConstantOutputLikelihood(0., 0., 1.)
    with pytest.raises(ValueError):
        ConstantOutputLikelihood(0., 1., 0.).log_likelihood(float("inf"))
    with pytest.raises(ConditioningNumericalError):
        ConstantOutputLikelihood(0., 1e-300, 0.).log_likelihood(1e300)
