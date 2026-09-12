"""Integrated exact conditioning, noisy dynamics and posterior sampling."""
import math
from dataclasses import replace
from typing import Callable, Tuple

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    AffineConditioning
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    ConditionedPrior, PriorPoint, SamplerConfig, sample_batch


def _problem(
    upper: float = 2.
) -> Tuple[ConditionedPrior, InferenceIdentity, Callable[[np.ndarray],
                                                         PriorPoint]]:
    original = BoxPrior(("theta", "start", "unused"),
                        ((1., upper), (0., 1.), (-1., 1.)))
    chart = AffineConditioning(
        original, ("start", ), (.5, ),
        content_digest(b"x(t)=start*theta**t; exact x(1)=.5"))
    prior = ConditionedPrior(original.names, original.digest, chart.digest,
                             BoxPrior(chart.free_names, chart.free_bounds))
    digest = content_digest(b"synthetic full-data reference")
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)

    def condition(free: np.ndarray) -> PriorPoint:
        point = chart.lift(free, np.array([[free[0]]]), np.zeros(1))
        return PriorPoint(point.joint, point.log_base_weight)

    return prior, identity, condition


def test_base_density_survives_every_metropolis_move() -> None:
    """Zero remaining likelihood must preserve the conditional base, not q."""
    prior, identity, condition = _problem(8.)
    result = sample_batch(prior,
                          identity,
                          lambda _: 0.,
                          SamplerConfig(particles=8192,
                                        temperatures=12,
                                        moves=4,
                                        proposal_scale=.12,
                                        max_evaluations=600000),
                          4,
                          condition=condition)
    assert result.status == "complete"
    samples = np.asarray(result.samples)
    assert np.average(samples[:, 0],
                      weights=result.weights) == pytest.approx(7 / math.log(8),
                                                               abs=.07)
    assert abs(np.average(samples[:, 2], weights=result.weights)) < .07
    np.testing.assert_allclose(samples[:, 0] * samples[:, 1], .5, atol=1e-15)
    assert isinstance(result.prior, ConditionedPrior)
    assert result.prior.original_prior == prior.original_prior
    assert result.estimator == "offline_tempered_smc_conditional"


def test_noisy_growth_predictions_against_quadrature() -> None:
    """Joint initial state and dynamics predict a held-out time from one
    fit."""
    prior, identity, condition = _problem()
    times = np.array([0, 2, 3])
    observed = np.array([.38, .71, 1.07])
    sigma = .2

    def likelihood(joint: np.ndarray) -> float:
        return float(-.5 * np.sum(
            ((joint[1] * joint[0]**times - observed) / sigma)**2))

    axis = 1 + (np.arange(10000) + .5) / 10000
    log_grid = -np.log(axis)
    for time, value in zip(times, observed):
        log_grid -= .5 * ((.5 * axis**(time - 1) - value) / sigma)**2
    grid_weights = np.exp(log_grid - log_grid.max())
    grid_weights /= grid_weights.sum()
    result = sample_batch(prior,
                          identity,
                          likelihood,
                          SamplerConfig(particles=1600,
                                        temperatures=24,
                                        moves=4,
                                        max_evaluations=180000),
                          11,
                          condition=condition)
    assert result.status == "complete"
    samples = np.asarray(result.samples)
    weights = np.asarray(result.weights)
    assert np.average(samples[:, 0], weights=weights) == pytest.approx(
        float(grid_weights @ axis), abs=.012)
    assert np.average(samples[:, 1], weights=weights) == pytest.approx(
        float(grid_weights @ (.5 / axis)), abs=.005)
    prediction = samples[:, 1] * samples[:, 0]**4
    assert np.average(prediction, weights=weights) == pytest.approx(float(
        grid_weights @ (.5 * axis**3)),
                                                                    abs=.04)
    covariance = np.cov(samples[:, :2].T, aweights=weights)
    assert covariance[0, 1] / math.sqrt(
        covariance[0, 0] * covariance[1, 1]) < -.95
    assert result.marginal_quantiles("start")[0] < result.marginal_quantiles(
        "start")[2]


def test_tiny_base_mass_can_be_selected_by_exact_discrete_evidence() -> None:
    """Do not underflow a rare component before incorporating its evidence."""
    box = BoxPrior(("x", ), ((0., 1.), ))
    digest = content_digest(b"rare conditional support")
    prior = ConditionedPrior(box.names, box.digest, digest, box)
    identity = InferenceIdentity(digest, digest, digest, prior.digest, digest)
    result = sample_batch(
        prior,
        identity,
        lambda x: 0. if x[0] >= .5 else -math.inf,
        SamplerConfig(particles=100, temperatures=4, moves=1),
        0,
        condition=lambda x: PriorPoint(tuple(x), -1000. if x[0] >= .5 else 0.))
    assert result.status == "complete"
    assert all(row[0] >= .5 for row, w in zip(result.samples, result.weights)
               if w > 0)
    assert sum(result.weights) == pytest.approx(1.)


def test_conditional_failures_are_not_posterior_samples() -> None:
    """Map errors, zero searched support and budget exhaustion remain
    distinct."""
    prior, identity, condition = _problem()
    config = SamplerConfig(particles=20, temperatures=3, moves=1)
    with pytest.raises(ValueError, match="coordinate map"):
        sample_batch(prior, identity, lambda _: 0., config, 0)
    with pytest.raises(ValueError, match="Base weight"):
        sample_batch(prior,
                     identity,
                     lambda _: 0.,
                     config,
                     0,
                     condition=lambda _: PriorPoint((1., .5, 0.), math.inf))
    with pytest.raises(ValueError, match="joint values"):
        sample_batch(prior,
                     identity,
                     lambda _: 0.,
                     config,
                     0,
                     condition=lambda _: PriorPoint((math.nan, .5, 0.), 0.))
    result = sample_batch(prior,
                          identity,
                          lambda _: 0.,
                          config,
                          0,
                          condition=lambda _: PriorPoint(
                              (1., .5, 0.), -math.inf))
    assert result.status == "no_particle_support" and not result.samples
    result = sample_batch(prior,
                          identity,
                          lambda _: 0.,
                          replace(config, max_evaluations=2),
                          0,
                          condition=condition)
    assert result.status == "budget_exhausted" and not result.samples
    result = sample_batch(prior,
                          identity,
                          lambda _: 0.,
                          config,
                          0,
                          condition=condition)
    assert result == sample_batch(prior,
                                  identity,
                                  lambda _: 0.,
                                  config,
                                  0,
                                  condition=condition)
