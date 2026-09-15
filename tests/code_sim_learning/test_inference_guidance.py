"""Guided conditional integration retains the original probability model."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, ncx2

from predicators.code_sim_learning.inference_discrepancy import \
    VelocityDiscrepancy
from predicators.code_sim_learning.inference_guidance import \
    condition_guided_velocity
from predicators.code_sim_learning.inference_path_integral import \
    integrate_conditional_paths


@pytest.mark.parametrize("selector", [.1, .95])
def test_full_mixture_density_matches_independent_reference(selector):
    """Both selected components must use the same full mixture density."""
    law = VelocityDiscrepancy(.2, .3)
    original, proposal, speed = (0., 0., .3), (.5, .2, -.4), .4
    probability = .8
    result = condition_guided_velocity(law, original, speed, proposal,
                                       probability, (selector, .4, .6))

    def radial(mean):
        return ncx2.pdf((speed / law.sigma)**2, 3,
                        np.dot(mean, mean) / law.sigma**2) * \
            2 * speed / law.sigma**2

    original_radial, guided_radial = radial(original), radial(proposal)
    original_pdf = multivariate_normal.pdf(result.velocity, original,
                                           np.eye(3) * law.sigma**2)
    guided_pdf = multivariate_normal.pdf(result.velocity, proposal,
                                         np.eye(3) * law.sigma**2)
    relative = guided_pdf / original_pdf * original_radial / guided_radial
    correction = -math.log(1 - probability + probability * relative)
    assert result.log_proposal_correction == pytest.approx(correction,
                                                           abs=1e-12)
    assert result.log_factor == pytest.approx(math.log(.8 * original_radial) +
                                              correction,
                                              abs=1e-12)
    assert result.log_proposal_correction <= -math.log1p(-probability)
    assert result.component == ("guided"
                                if selector < probability else "original")
    assert math.hypot(*result.velocity) == pytest.approx(speed, abs=1e-15)


@pytest.mark.parametrize("probability", [.2, .8])
def test_guided_future_integral_matches_original_sphere_quadrature(
        probability):
    """Off-axis guidance leaves the original downstream integral unchanged."""
    law = VelocityDiscrepancy(.2, .3)
    original, proposal, speed = (0., 0., .3), (.25, .1, .1), .4
    reading, noise = .05, .2

    def integrand(rng):
        result = condition_guided_velocity(law, original, speed, proposal,
                                           probability, tuple(rng.random(3)))
        output = -.5 * ((reading - result.velocity[2]) / noise)**2 - \
            math.log(noise * math.sqrt(2 * math.pi))
        return result.log_factor + output

    result = integrate_conditional_paths(integrand, 12000, 97)
    concentration = speed * original[2] / law.sigma**2

    def reference(cosine):
        direction = concentration * math.exp(concentration * cosine) / \
            (2 * math.sinh(concentration))
        output = math.exp(-.5 * ((reading - speed * cosine) / noise)**2) / \
            (noise * math.sqrt(2 * math.pi))
        return direction * output

    radial = math.exp(
        law.condition_on_speed(original, speed,
                               (.5, .5)).log_observation_factor)
    expected = radial * quad(reference, -1, 1, epsabs=1e-12)[0]
    assert math.exp(result.log_density) == pytest.approx(expected, rel=.03)
    assert result.relative_standard_error is not None
    assert result.relative_standard_error < .02


def test_no_guidance_rest_and_zero_physical_support():
    """Disabled guidance is exact, and rest retains its original atom."""
    law = VelocityDiscrepancy(.2, .3)
    mean, other, units = (0., 0., .3), (.2, .1, .1), (.4, .3, .7)
    original = law.condition_on_speed(mean, .4, units[1:])
    for proposal, probability in ((other, 0.), (mean, .8)):
        result = condition_guided_velocity(law, mean, .4, proposal,
                                           probability, units)
        assert result.velocity == original.velocity
        assert result.log_factor == original.log_observation_factor
        assert result.log_proposal_correction == 0.
    rest = condition_guided_velocity(law, mean, 0., other, .8, ())
    assert rest.log_factor == math.log(.2)
    assert rest.velocity == (0., 0., 0.) and rest.component == "rest"
    zero = condition_guided_velocity(VelocityDiscrepancy(1., .3), mean, .4,
                                     other, .8, units)
    assert zero.log_factor == -math.inf
    for probability in (-.1, 1., math.nan):
        with pytest.raises(ValueError):
            condition_guided_velocity(law, mean, .4, other, probability, units)
    for invalid_units in ((), (.1, .2), (.1, .2, math.nan)):
        with pytest.raises(ValueError):
            condition_guided_velocity(law, mean, .4, other, .8, invalid_units)


@pytest.mark.parametrize("selector", [.1, .95])
def test_opposite_concentrated_means_preserve_direction_ratio(selector):
    """Cancel shared radial constants without losing directional evidence."""
    law = VelocityDiscrepancy(.2, 1.)
    result = condition_guided_velocity(law, (0., 0., 1e150), 1.,
                                       (0., 0., -1e150), .8,
                                       (selector, .5, .5))
    # Equal concentration and opposite axes cancel vMF normalizers exactly.
    relative = -2e150 * result.velocity[2]
    expected = -float(np.logaddexp(math.log(.2), math.log(.8) + relative))
    assert result.log_proposal_correction == pytest.approx(expected, rel=1e-15)
