"""Independent integration and predictive checks for shared variances."""
import math
from typing import Tuple

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import invgamma, norm, t

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_variance import \
    GaussianVariancePosterior, GaussianVariancePrior


@pytest.mark.parametrize("residuals", [(), (.12, ), (.12, -.3, .05)])
def test_normalized_history_matches_variance_quadrature(
        residuals: Tuple[float, ...]) -> None:
    """Integrate the original variance density, with its log-coordinate
    Jacobian, independently of the conjugate implementation."""
    prior = GaussianVariancePrior(2.5, .08)
    posterior = prior.condition(residuals)

    def integrand(log_variance: float) -> float:
        variance = math.exp(log_variance)
        density = invgamma.logpdf(variance, prior.alpha, scale=prior.beta)
        density += sum(
            norm.logpdf(r, scale=math.sqrt(variance)) for r in residuals)
        return math.exp(density + log_variance)

    actual, error = quad(integrand, -25, 25, epsabs=1e-11)
    assert error < 1e-8
    assert posterior.log_evidence == pytest.approx(math.log(actual), abs=1e-10)


def test_causal_updates_equal_batch_and_student_prediction() -> None:
    """Every causal factor is normalized; their product equals one batch
    marginal, with no repeated-prior update."""
    prior = GaussianVariancePrior(2., 1e-6)
    residuals = (.0001, -.002, .0003, .0005)
    state = prior.condition(())
    density = 0.
    for residual in residuals:
        expected = t.logpdf(residual,
                            df=2 * state.alpha,
                            scale=math.sqrt(state.beta / state.alpha))
        assert state.log_predictive(residual) == pytest.approx(expected)
        density += state.log_predictive(residual)
        state = state.advance(residual)
    batch = prior.condition(residuals)
    assert density == pytest.approx(batch.log_evidence, abs=1e-12)
    assert state.alpha == batch.alpha
    assert state.beta == pytest.approx(batch.beta, abs=1e-20)
    assert prior.condition(residuals) == batch
    assert prior.digest == GaussianVariancePrior(2., 1e-6).digest
    assert prior.digest != GaussianVariancePrior(2., 1e-5).digest


@pytest.mark.parametrize("multiplier", [.001, 1000.])
def test_change_of_physical_units_retains_density(multiplier: float) -> None:
    """Scaling channel units transforms the prior and all density factors."""
    residuals = (.1, -.25, .05)
    original = GaussianVariancePrior(3., .01).condition(residuals)
    transformed = GaussianVariancePrior(3., .01 * multiplier**2).condition(
        tuple(r * multiplier for r in residuals))
    assert transformed.log_evidence == pytest.approx(
        original.log_evidence - len(residuals) * math.log(multiplier),
        abs=1e-11)
    assert transformed.log_predictive(.2 * multiplier) == pytest.approx(
        original.log_predictive(.2) - math.log(multiplier), abs=1e-11)


def test_future_draws_share_the_conditioned_variance() -> None:
    """A common variance induces dependence in squared future residuals;
    independent Student draws would lose that history dependence."""
    state = GaussianVariancePrior(4., .03).condition((.1, -.2))
    rng = np.random.default_rng(501)
    variance = np.asarray([state.draw_variance(rng) for _ in range(40000)])
    pair = rng.normal(size=(len(variance), 2)) * np.sqrt(variance[:, None])
    expected_mean = state.beta / (state.alpha - 1)
    expected_second = state.beta**2 / ((state.alpha - 1) * (state.alpha - 2))
    assert variance.mean() == pytest.approx(expected_mean, rel=.02)
    assert np.mean(pair[:, 0]**2 * pair[:, 1]**2) == pytest.approx(
        expected_second, rel=.08)
    assert np.mean(pair[:, 0]**2 * pair[:, 1]**2) > expected_mean**2 * 1.15


@pytest.mark.parametrize("alpha,beta", [(0., 1.), (-1., 1.), (1., 0.),
                                        (math.inf, 1.), (1., math.nan)])
def test_invalid_prior_rejected(alpha: float, beta: float) -> None:
    """A degenerate variance prior is not silently repaired."""
    with pytest.raises(ValueError):
        GaussianVariancePrior(alpha, beta)


def test_invalid_history_and_numerical_overflow_rejected() -> None:
    """Setup errors and unrepresentable arithmetic remain explicit."""
    prior = GaussianVariancePrior(2., .01)
    for count, energy in [(-1, 0.), (True, 0.), (0, .1), (1, -.1),
                          (1, math.inf)]:
        with pytest.raises(ValueError):
            GaussianVariancePosterior(prior, count, energy)
    with pytest.raises(ValueError):
        prior.condition((math.nan, ))
    with pytest.raises(ConditioningNumericalError):
        prior.condition((1e300, ))
    with pytest.raises(ConditioningNumericalError):
        prior.condition(()).log_predictive(1e300)
