"""Independent integration references for exact-speed transition
conditioning."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ncx2

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError, RestOrGaussianVelocityPrior, \
    UnsupportedConditioning
from predicators.code_sim_learning.inference_discrepancy import \
    VelocityDiscrepancy


@pytest.mark.parametrize("mean", [0., 1e-6, .2, 2.])
def test_radial_density_integrates_gaussian_shells(mean: float) -> None:
    """The retained factor equals an independent spherical Gaussian
    integral."""
    sigma = .3
    model = VelocityDiscrepancy(.2, sigma)
    for speed in (.05, .4, 1.8):
        result = model.condition_on_speed((0., 0., mean), speed, (.3, .7))

        def shell(cosine: float, radius: float = speed) -> float:
            exponent = -(radius**2 + mean**2 - 2 * radius * mean * cosine) / \
                (2 * sigma**2)
            return (2 * math.pi * radius**2 * math.exp(exponent) /
                    (math.sqrt(2 * math.pi) * sigma)**3)

        expected = .8 * quad(shell, -1., 1., epsabs=1e-40, epsrel=1e-10)[0]
        assert math.exp(result.log_observation_factor) == \
            pytest.approx(expected, rel=1e-9, abs=1e-40)

    def radial(speed: float) -> float:
        return math.exp(
            model.condition_on_speed((0., 0., mean), speed,
                                     (.3, .7)).log_observation_factor)

    upper = mean + 12 * sigma
    assert quad(radial, 0., upper, epsabs=1e-10)[0] == pytest.approx(.8)
    second_moment = quad(lambda r: r * r * radial(r), 0., upper,
                         epsabs=1e-10)[0]
    assert second_moment == pytest.approx(.8 * (mean**2 + 3 * sigma**2))


@pytest.mark.parametrize("concentration", [0., 1e-5, .1, 1., 50., 1000.])
def test_direction_quantiles_match_conditional_sphere(
        concentration: float) -> None:
    """The eliminated speed retains directional bias rather than zeroing it."""
    model = VelocityDiscrepancy(0., 1.)
    # With speed=1 and sigma=1, the predicted magnitude is concentration.
    for uniform in (0., .001, .2, .5, .9, .999, 1.):
        result = model.condition_on_speed((0., 0., concentration), 1.,
                                          (uniform, .31))
        cosine = result.velocity[2]
        assert result.speed_residual < 1e-14
        if concentration == 0:
            cdf = (cosine + 1) / 2
        elif concentration < .5:
            cdf = math.expm1(concentration * (cosine + 1)) / \
                math.expm1(2 * concentration)
        else:
            cdf = (math.exp(concentration * (cosine - 1)) -
                   math.exp(-2 * concentration)) / \
                -math.expm1(-2 * concentration)
        assert cdf == pytest.approx(uniform, abs=2e-12)


def test_rotation_and_large_concentration_do_not_change_radial_evidence(
) -> None:
    """Stable formulas avoid sinh overflow and orient the law in world
    space."""
    model = VelocityDiscrepancy(.1, 1e-4)
    predicted = (3., 4., 0.)
    result = model.condition_on_speed(predicted, 5., (.5, .3))
    axis_result = model.condition_on_speed((0., 0., 5.), 5., (.5, .3))
    assert result.log_observation_factor == axis_result.log_observation_factor
    np.testing.assert_allclose(np.dot(result.velocity, predicted) / 5.,
                               axis_result.velocity[2],
                               atol=1e-14)
    assert math.isfinite(result.log_observation_factor)
    assert result.speed_residual < 1e-14


def test_central_case_and_rest_mass_match_original_velocity_prior() -> None:
    """The extension retains the existing rest atom and central reference."""
    model = VelocityDiscrepancy(.3, .2)
    prior = RestOrGaussianVelocityPrior(.3, .2)
    for speed, coords in ((0., ()), (.01, (.2, .7)), (.5, (.4, .3))):
        result = model.condition_on_speed((0., 0., 0.), speed, coords)
        reference = prior.condition_on_speed(speed, coords)
        np.testing.assert_allclose(result.velocity,
                                   reference.velocity,
                                   atol=1e-15)
        assert result.log_observation_factor == pytest.approx(
            reference.log_observation_factor)
        assert result.free_dimensions == reference.free_dimensions
    rest = model.condition_on_speed((4., 2., 1.), 0.)
    assert rest.velocity == (0., 0., 0.)
    assert rest.log_observation_factor == math.log(.3)
    assert VelocityDiscrepancy(1., .2).condition_on_speed(
        (0., 0., 1.), 1., (.2, .3)).log_observation_factor == -math.inf
    with pytest.raises(UnsupportedConditioning):
        VelocityDiscrepancy(0., .2).condition_on_speed((1., 0., 0.), 0.)


def test_invalid_laws_and_numerical_failure_are_explicit() -> None:
    """No variance floor or clipping rescues a malformed correction model."""
    for rho, sigma in ((-.1, .1), (1.1, .1), (.1, 0.), (.1, float("nan"))):
        with pytest.raises(ValueError):
            VelocityDiscrepancy(rho, sigma)
    model = VelocityDiscrepancy(.1, .2)
    with pytest.raises(ValueError):
        model.condition_on_speed((float("nan"), 0., 0.), 1., (.5, .5))
    with pytest.raises(ValueError):
        model.condition_on_speed((0., 0., 0.), 1., (1.1, .5))
    with pytest.raises(ValueError):
        model.condition_on_speed((0., 0., 0.), 0., (.5, .5))
    with pytest.raises(ConditioningNumericalError):
        VelocityDiscrepancy(.1, 1e-300).condition_on_speed((1., 0., 0.), 1.,
                                                           (.5, .5))
    assert model.digest != VelocityDiscrepancy(.2, .2).digest
    assert model.digest != VelocityDiscrepancy(.1, .3).digest


@pytest.mark.parametrize("rest_probability", [0., .3, 1.])
@pytest.mark.parametrize("predicted", [(0., 0., 0.), (.4, -.3, .8)])
def test_unconditional_draws_match_mixture_moments_and_speed_law(
        rest_probability, predicted):
    """Generated transitions retain rest mass and the noncentral speed law."""
    sigma = .2
    model = VelocityDiscrepancy(rest_probability, sigma)
    rng = np.random.default_rng(82)
    draws = np.array([model.sample(predicted, rng) for _ in range(24000)])
    mean = np.array(predicted)
    expected_mean = (1 - rest_probability) * mean
    covariance = (1 - rest_probability) * sigma**2 * np.eye(3) + \
        rest_probability * (1 - rest_probability) * np.outer(mean, mean)
    assert draws.mean(axis=0) == pytest.approx(expected_mean, abs=.012)
    assert np.cov(draws.T) == pytest.approx(covariance, abs=.012)
    speeds = np.linalg.norm(draws, axis=1)
    assert np.mean(speeds == 0.) == pytest.approx(rest_probability, abs=.012)
    for speed in (.2, .6, 1.2):
        expected = rest_probability + (1 - rest_probability) * \
            ncx2.cdf((speed / sigma)**2, 3, float(mean @ mean) / sigma**2)
        assert np.mean(speeds <= speed) == pytest.approx(expected, abs=.012)


def test_transition_draw_preserves_existing_random_stream():
    """Extracting the existing native branch leaves generated paths intact."""
    model = VelocityDiscrepancy(.3, .2)
    predicted = (.4, -.3, .8)
    old_rng = np.random.default_rng(23)
    new_rng = np.random.default_rng(23)
    for _ in range(100):
        old = tuple(float(v) for v in old_rng.normal(predicted, model.sigma)) \
            if old_rng.random() >= model.rest_probability else (0., 0., 0.)
        assert model.sample(predicted, new_rng) == old
    assert old_rng.random() == new_rng.random()
    for invalid in ((math.nan, 0., 0.), (math.inf, 0., 0.), (0., 0.)):
        with pytest.raises(ValueError, match="three finite"):
            model.sample(invalid, new_rng)
    with np.errstate(over="ignore"), \
            pytest.raises(ConditioningNumericalError, match="overflow"):
        VelocityDiscrepancy(0., 1e308).sample((1.7e308, 1.7e308, 1.7e308),
                                              np.random.default_rng(4))
