"""Conditional path integration retains density factors and sampling error."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_discrepancy import \
    VelocityDiscrepancy
from predicators.code_sim_learning.inference_path_integral import \
    integrate_conditional_paths, summarize_path_integral


@pytest.mark.parametrize("constant", [-1e6, 0., 1e6])
def test_constant_integrand_and_log_scale_stability(constant):
    """Constant densities need no exponentiation at their original scale."""
    result = summarize_path_integral((constant, ) * 16)
    assert result.log_density == constant
    assert result.relative_standard_error == 0.
    assert result.effective_terms == 16.
    assert result.status == "finite_estimate"


def test_zero_contributions_remain_in_the_integral_and_error():
    """Deleting zero paths would inflate density and disguise uncertainty."""
    factors = (0., math.log(3.), -math.inf, -math.inf)
    result = summarize_path_integral(factors)
    values = np.array([1., 3., 0., 0.])
    assert result.log_density == pytest.approx(math.log(values.mean()))
    assert result.relative_standard_error == pytest.approx(
        values.std(ddof=1) / math.sqrt(4) / values.mean())
    assert result.effective_terms == pytest.approx(1.6)
    assert result.log_factors == factors
    shifted = summarize_path_integral(tuple(v - 10000 for v in factors))
    assert shifted.log_density == pytest.approx(result.log_density - 10000)
    assert shifted.relative_standard_error == pytest.approx(
        result.relative_standard_error)
    assert shifted.effective_terms == pytest.approx(result.effective_terms)


@pytest.mark.parametrize("speed", [0., .4])
def test_conditional_speed_future_density_matches_quadrature(speed):
    """Radial density times a conditional downstream integral scores once."""
    law = VelocityDiscrepancy(.2, .3)
    predicted = (0., 0., .3)
    reading, sensor_sigma = .05, .2

    def log_integrand(rng):
        direction = tuple(rng.random(2)) if speed else ()
        conditional = law.condition_on_speed(predicted, speed, direction)
        residual = (reading - conditional.velocity[2]) / sensor_sigma
        output_factor = -.5 * residual**2 - math.log(sensor_sigma) - \
            .5 * math.log(2 * math.pi)
        return conditional.log_observation_factor + output_factor

    result = integrate_conditional_paths(log_integrand, 16000, 86)
    assert result == integrate_conditional_paths(log_integrand, 16000, 86)
    radial = math.exp(
        law.condition_on_speed(predicted, speed, (.5, .5) if speed else
                               ()).log_observation_factor)
    if speed:
        concentration = speed * predicted[2] / law.sigma**2

        def integrand(cosine):
            directional = concentration * math.exp(concentration * cosine) / \
                (2 * math.sinh(concentration))
            output = math.exp(-.5 * ((reading - speed * cosine) /
                                    sensor_sigma)**2) / \
                (math.sqrt(2 * math.pi) * sensor_sigma)
            return directional * output

        expected = radial * quad(integrand, -1., 1., epsabs=1e-12)[0]
        assert result.relative_standard_error is not None
        assert result.relative_standard_error > 0.
    else:
        expected = radial * math.exp(-.5 * (reading / sensor_sigma)**2) / \
            (math.sqrt(2 * math.pi) * sensor_sigma)
        assert result.relative_standard_error == 0.
    assert math.exp(result.log_density) == pytest.approx(expected, rel=.02)


def test_no_sample_support_does_not_claim_model_inconsistency():
    """An unseen rare supported path leaves the empirical error undefined."""
    result = integrate_conditional_paths(
        lambda rng: 0. if rng.random() < 1e-12 else -math.inf, 16, 0)
    assert result.status == "no_sample_support"
    assert result.relative_standard_error is None
    assert result.log_density == -math.inf
    assert result.effective_terms == 0.


def test_invalid_integrands_and_callback_errors_are_not_dropped():
    """Numerical and replay failures propagate instead of becoming zeros."""
    for value in (math.nan, math.inf):
        with pytest.raises(ConditioningNumericalError):
            summarize_path_integral((0., value))
    for count, seed in ((1, 0), (True, 0), (2, -1), (2, True)):
        with pytest.raises(ValueError):
            integrate_conditional_paths(lambda rng: rng.random(), count, seed)
    calls = []

    def fail(rng):
        calls.append(rng.random())
        if len(calls) == 3:
            raise RuntimeError("replay failed")
        return 0.

    with pytest.raises(RuntimeError, match="replay failed"):
        integrate_conditional_paths(fail, 8, 0)
    assert len(calls) == 3
