"""Exact Gaussian proposals preserve original-prior and sensor evidence."""
import math

import pytest

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError, condition_gaussian_coordinate


def _log_density(x: float, mean: float, sigma: float) -> float:
    return -.5 * ((x - mean) / sigma)**2 - math.log(sigma) - \
        .5 * math.log(2 * math.pi)


def test_proposal_ratio_is_the_original_marginal_density() -> None:
    """p0(x) p(y|x) / q(x|y) is constant, including off-center readings."""
    for observed in (-2., .3, 4.):
        point = condition_gaussian_coordinate(.1, .7, observed, .2)
        for offset in (-2., -.5, 0., .5, 2.):
            value = point.mean + offset * point.sigma
            ratio = (_log_density(value, .1, .7) +
                     _log_density(observed, value, .2) -
                     _log_density(value, point.mean, point.sigma))
            assert ratio == pytest.approx(point.log_observation_factor,
                                          abs=1e-12)
        expected = _log_density(observed, .1, math.hypot(.7, .2))
        assert point.log_observation_factor == pytest.approx(expected)


def test_exact_coordinate_retains_its_prior_density() -> None:
    """An exact reading eliminates the coordinate without a tiny variance."""
    point = condition_gaussian_coordinate(.1, .7, 2., 0.)
    assert point.mean == 2. and point.sigma == 0.
    assert point.log_observation_factor == pytest.approx(
        _log_density(2., .1, .7))


def test_fixed_prior_repeated_fit_and_independent_new_reading() -> None:
    """Repeated fitting is identical; new independent evidence adds once."""
    first = condition_gaussian_coordinate(0., 1., 1., .5)
    repeated = condition_gaussian_coordinate(0., 1., 1., .5)
    assert first == repeated
    second = condition_gaussian_coordinate(first.mean, first.sigma, -.5, .5)
    expected_variance = 1 / (1 + 4 + 4)
    assert second.mean == pytest.approx(expected_variance * (4 - 2))
    assert second.sigma**2 == pytest.approx(expected_variance)
    # Joint evidence equals prior*both likelihoods/posterior everywhere.
    for value in (-.3, .2, 1.):
        joint_ratio = (_log_density(value, 0., 1.) +
                       _log_density(1., value, .5) +
                       _log_density(-.5, value, .5) -
                       _log_density(value, second.mean, second.sigma))
        assert joint_ratio == pytest.approx(first.log_observation_factor +
                                            second.log_observation_factor)


def test_scale_and_numerical_failures_remain_explicit() -> None:
    """Stable arithmetic handles large sigmas and rejects lost support."""
    point = condition_gaussian_coordinate(0., 1e200, 1e200, 1e200)
    assert point.mean == pytest.approx(5e199)
    assert point.sigma == pytest.approx(1e200 / math.sqrt(2))
    with pytest.raises(ConditioningNumericalError, match="overflow"):
        condition_gaussian_coordinate(0., 1.7e308, 0., 1.7e308)
    with pytest.raises(ConditioningNumericalError, match="range"):
        condition_gaussian_coordinate(0., 1e-300, 1e300, 1e-300)
    with pytest.raises(ValueError, match="finite"):
        condition_gaussian_coordinate(0., 1., math.nan, .1)
    with pytest.raises(ValueError, match="sigma"):
        condition_gaussian_coordinate(0., 0., 0., .1)
