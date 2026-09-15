"""Physical velocity conditioning references with explicit rest support."""
import math

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError, RestOrGaussianVelocityPrior, \
    UnsupportedConditioning


def test_speed_distribution_has_unit_mass() -> None:
    """The atom and integrated positive radial density form one proper
    prior."""
    prior = RestOrGaussianVelocityPrior(.3, .7)
    rest = prior.condition_on_speed(0.)
    assert rest.velocity == (0., 0., 0.)
    assert rest.free_dimensions == 0
    assert math.exp(rest.log_observation_factor) == pytest.approx(.3)
    step = 8 * prior.moving_sigma / 4000
    radii = (np.arange(4000) + .5) * step
    moving_mass = sum(
        math.exp(
            prior.condition_on_speed(float(r), (.5,
                                                .5)).log_observation_factor)
        for r in radii) * step
    assert moving_mass == pytest.approx(.7, abs=1e-10)


def test_positive_speed_preserves_uncertain_direction() -> None:
    """Correct sphere coordinates are isotropic and keep the observed speed."""
    prior = RestOrGaussianVelocityPrior(.4, 2.)
    count = 64
    grid = (np.arange(count) + .5) / count
    points = [
        prior.condition_on_speed(1.7, (float(u), float(v))) for u in grid
        for v in grid
    ]
    velocities = np.asarray([p.velocity for p in points])
    np.testing.assert_allclose(np.mean(velocities, axis=0), 0., atol=1e-12)
    np.testing.assert_allclose(velocities.T @ velocities / len(points),
                               np.eye(3) * 1.7**2 / 3,
                               atol=3e-4)
    assert all(p.free_dimensions == 2 for p in points)
    assert max(p.speed_residual for p in points) < 1e-15
    assert len({p.log_observation_factor for p in points}) == 1


def test_rest_evidence_informs_mixture_weight_not_moving_sigma() -> None:
    """A uniform prior on rest mass becomes density 2*rho after exact rest."""
    mass = (np.arange(1000) + .5) / 1000
    weights = np.asarray([
        math.exp(
            RestOrGaussianVelocityPrior(
                float(rho), 1.).condition_on_speed(0.).log_observation_factor)
        for rho in mass
    ])
    assert np.average(mass, weights=weights) == pytest.approx(2 / 3, abs=1e-6)
    for sigma in (.01, 1., 100.):
        assert RestOrGaussianVelocityPrior(.3, sigma).condition_on_speed(
            0.).log_observation_factor == pytest.approx(math.log(.3))


def test_speed_zero_is_not_an_epsilon_band() -> None:
    """Zero-density boundary, zero prior support and invalid input differ."""
    with pytest.raises(UnsupportedConditioning, match="extension"):
        RestOrGaussianVelocityPrior(0., 1.).condition_on_speed(0.)
    point = RestOrGaussianVelocityPrior(1.,
                                        1.).condition_on_speed(1., (.2, .7))
    assert point.log_observation_factor == -math.inf
    prior = RestOrGaussianVelocityPrior(.3, 1.)
    tiny = prior.condition_on_speed(1e-150, (.2, .7))
    assert tiny.free_dimensions == 2
    assert math.isfinite(tiny.log_observation_factor)
    assert tiny.velocity != (0., 0., 0.)
    assert prior.digest != RestOrGaussianVelocityPrior(.4, 1.).digest
    with pytest.raises(ConditioningNumericalError, match="range"):
        prior.condition_on_speed(1e200, (.2, .7))
    for mass, sigma in ((-.1, 1.), (1.1, 1.), (.2, 0.), (.2, math.inf)):
        with pytest.raises(ValueError):
            RestOrGaussianVelocityPrior(mass, sigma)
    for speed, direction in ((-1., ()), (math.nan, ()), (0., (.2, .3)),
                             (1., ()), (1., (.2, math.nan)), (1., (.2, 1.1))):
        with pytest.raises(ValueError):
            prior.condition_on_speed(speed, direction)
