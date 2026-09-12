"""Statistical references for the offline fixed-prior batch sampler."""
import math
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_data import EpisodeData, \
    InferenceData, InferenceIdentity, Observation, SensorModel, \
    content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    SamplerConfig, sample_batch
from predicators.observation_noise import ObservationNoise, step_rng
from predicators.structs import Object, State, Type


def _identity(prior: BoxPrior) -> InferenceIdentity:
    digest = content_digest(b"fixed synthetic reference")
    return InferenceIdentity(digest, digest, digest, prior.digest, digest)


def test_correlated_dynamics_and_initial_state_against_grid() -> None:
    """A velocity and uncertain starting position retain their tradeoff.

    The third coordinate has no observations and must retain its
    original uniform prior. Repeating the entire batch gives identical
    output, not a second use of the data with the previous posterior as
    prior.
    """
    prior = BoxPrior(("theta.velocity", "episode0.position", "theta.unused"),
                     ((-1, 1), (-1, 1), (-1, 1)))
    times = np.arange(4)
    data = np.array([.19, .48, .68, 1.03])
    sigma = .15

    def likelihood(v: np.ndarray) -> float:
        return float(-.5 * np.sum(((data - v[1] - v[0] * times) / sigma)**2))

    axis = np.linspace(-1, 1, 401)
    velocity, start = np.meshgrid(axis, axis, indexing="ij")
    grid_log = np.zeros_like(velocity)
    for t, observed in zip(times, data):
        grid_log -= .5 * ((observed - start - velocity * t) / sigma)**2
    grid_weights = np.exp(grid_log - grid_log.max())
    grid_weights /= grid_weights.sum()
    means = np.array([(grid_weights * velocity).sum(),
                      (grid_weights * start).sum()])
    covariance = np.array([[(grid_weights * (a - ma) * (b - mb)).sum()
                            for b, mb in zip((velocity, start), means)]
                           for a, ma in zip((velocity, start), means)])
    config = SamplerConfig(particles=1800,
                           temperatures=32,
                           moves=5,
                           proposal_scale=.04,
                           max_evaluations=300000)
    result = sample_batch(prior, _identity(prior), likelihood, config, seed=8)
    assert result.status == "complete"
    samples = np.array(result.samples)
    weights = np.asarray(result.weights)
    np.testing.assert_allclose(np.average(samples[:, :2],
                                          axis=0,
                                          weights=weights),
                               means,
                               atol=.018)
    np.testing.assert_allclose(np.cov(samples[:, :2].T, aweights=weights),
                               covariance,
                               rtol=.25,
                               atol=.001)
    cov = np.cov(samples[:, :2].T, aweights=weights)
    assert cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]) < -.7
    unused_mean = np.average(samples[:, 2], weights=weights)
    assert abs(unused_mean) < .1
    assert abs(
        np.average((samples[:, 2] - unused_mean)**2, weights=weights) -
        1 / 3) < .06
    assert sum(result.weights) == pytest.approx(1)
    quantiles = result.marginal_quantiles("theta.velocity")
    for quantile, probability in zip(quantiles, (.05, .5, .95)):
        assert weights[samples[:, 0] < quantile].sum() < probability
        assert weights[samples[:, 0] <= quantile].sum() >= probability
    again = sample_batch(prior, _identity(prior), likelihood, config, seed=8)
    assert again == result


@pytest.mark.parametrize("seed", [19, 0, 1, 2])
def test_bimodal_reference_and_prior_support(seed: int) -> None:
    """Broad initialization can represent two modes without clipping bounds."""
    prior = BoxPrior(("theta", ), ((-2, 2), ))
    result = sample_batch(
        prior, _identity(prior), lambda x: -.5 * ((x[0]**2 - 1) / .12)**2,
        SamplerConfig(particles=1400, moves=4, max_evaluations=200000), seed)
    assert result.status == "complete"
    values = np.array(result.samples)[:, 0]
    assert np.all(np.abs(values) <= 2)
    assert .35 < np.average(values > 0, weights=result.weights) < .65
    assert .95 < np.average(np.abs(values), weights=result.weights) < 1.04
    assert min(result.effective_sample_sizes) < result.config.particles


def test_failure_results_do_not_publish_partial_posteriors() -> None:
    """Finite support failure and budget exhaustion are numerical outcomes."""
    prior = BoxPrior(("x", ), ((0, 1), ))
    identity = _identity(prior)
    config = SamplerConfig(particles=16, max_evaluations=3)
    result = sample_batch(prior, identity, lambda _: 0., config, 0)
    assert result.status == "budget_exhausted" and result.evaluations == 3
    assert not result.samples and not result.weights
    assert result.initial_finite == 3
    with pytest.raises(ValueError, match="No completed"):
        result.marginal_quantiles("x")
    config = replace(config, max_evaluations=19)
    result = sample_batch(prior, identity, lambda _: 0., config, 0)
    assert result.status == "budget_exhausted" and result.evaluations == 19
    assert result.completed_temperature == 0
    assert not result.samples
    config = replace(config, max_evaluations=5000)
    result = sample_batch(prior, identity, lambda _: -math.inf, config, 0)
    assert result.status == "no_particle_support"
    assert result.initial_finite == 0 and result.evaluations == 16
    assert not result.samples

    def broken(_: np.ndarray) -> float:
        raise RuntimeError("simulator setup failed")

    with pytest.raises(RuntimeError, match="setup failed"):
        sample_batch(prior, identity, broken, config, 0)
    with pytest.raises(ValueError, match="Prior differs"):
        sample_batch(prior, replace(identity, prior=content_digest(b"edit")),
                     lambda _: 0., config, 0)
    with pytest.raises(ValueError, match="positive widths"):
        BoxPrior(("x", ), ((1, 1), ))
    with pytest.raises(ValueError, match="positive integers"):
        SamplerConfig(moves=0)


def test_stationary_injector_to_posterior() -> None:
    """Recorded noise plus the ledger recovers the stationary Gaussian mean.

    This uses the actual sensor injector and full batch likelihood. The
    starting position is unknown, not pinned to the first noisy reading.
    A broad uniform prior makes its boundary correction negligible here.
    """
    obj = Object("box", Type("box", ["x"]))
    truth = State({obj: np.array([.35])})
    noise = ObservationNoise(position=.2)
    sensor = SensorModel.from_state(truth, noise)
    frames = tuple(
        Observation.from_state(t, noise.perturb(truth, step_rng(42, 0, 0, t)))
        for t in range(12))
    ledger = InferenceData((EpisodeData("reset0", ((0., ), ) * 11,
                                        frames + (frames[0], )), ))
    prior = BoxPrior(("reset0.position", ), ((-2., 2.), ))
    identity = replace(_identity(prior),
                       data=ledger.digest,
                       sensor=sensor.digest)
    key = ("box", "box", "x")

    def likelihood(candidate: np.ndarray) -> float:
        return ledger.log_likelihood(
            sensor, {"reset0": [{
                key: float(candidate[0])
            }] * len(frames)})

    result = sample_batch(prior,
                          identity,
                          likelihood,
                          SamplerConfig(particles=1000,
                                        temperatures=24,
                                        moves=3),
                          seed=7)
    assert result.status == "complete"
    samples = np.array(result.samples)[:, 0]
    expected_mean = np.mean([dict(frame.values)[key] for frame in frames])
    mean = np.average(samples, weights=result.weights)
    variance = np.average((samples - mean)**2, weights=result.weights)
    assert abs(mean - expected_mean) < .012
    assert math.sqrt(variance) == pytest.approx(.2 / math.sqrt(12), rel=.15)


def test_small_complete_batch_and_owned_candidates() -> None:
    """Exercise completed-result plumbing without a statistical accuracy
    claim."""
    prior = BoxPrior(("x", ), ((0., 1.), ))
    identity = _identity(prior)
    config = SamplerConfig(particles=32,
                           temperatures=3,
                           moves=2,
                           max_evaluations=300)

    def constant(candidate: np.ndarray) -> float:
        candidate[:] = math.nan  # Callback cannot mutate retained samples.
        return 0.

    result = sample_batch(prior, identity, constant, config, 0)
    assert result.status == "complete"
    assert result.completed_temperature == 1.
    assert 0 < result.accepted_moves <= result.attempted_moves == 192
    assert np.all(np.isfinite(result.samples))
    assert np.all((np.array(result.samples) >= 0)
                  & (np.array(result.samples) <= 1))
    assert len(result.weights) == len(result.samples) == 32
    assert sum(result.weights) == 1
    assert result.effective_sample_sizes == (32., ) * 3
    assert result.resampling_count == 0
    assert result.surviving_ancestors == 32
    assert result.prior == prior
    lo, mid, hi = result.marginal_quantiles("x")
    assert 0 <= lo <= mid <= hi <= 1
    with pytest.raises(ValueError, match="Quantile probabilities"):
        result.marginal_quantiles("x", (1.5, ))
    with pytest.raises(ValueError):
        result.marginal_quantiles("unknown")


@pytest.mark.parametrize("value", [math.inf, math.nan])
def test_invalid_likelihood_values(value: float) -> None:
    """Invalid arithmetic is not a zero-likelihood model candidate."""
    prior = BoxPrior(("x", ), ((0., 1.), ))
    with pytest.raises(ValueError, match="Likelihood returned"):
        sample_batch(prior, _identity(prior), lambda _: value,
                     SamplerConfig(particles=2), 0)
