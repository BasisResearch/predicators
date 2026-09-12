"""Offline tempered batch sampling for small continuous reference problems.

The current implementation uses an independent uniform box prior over a
joint vector of parameters and episode initial states. It is not a
feasible physical-state prior for the five domains, and is not used by
the agent.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, List, Literal, Tuple

import numpy as np

from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest


@dataclass(frozen=True)
class BoxPrior:
    """Fixed normalized independent uniforms in named coordinates.

    Names distinguish shared parameters from individual episode initial
    states. Known exact values should be conditioned and excluded from
    this vector. More general supports require a different prior
    implementation, not rejection followed by an unrecorded change in
    prior density.
    """
    names: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        bounds = tuple((float(lo), float(hi)) for lo, hi in self.bounds)
        if not names or len(names) != len(bounds) or len(
                set(names)) != len(names):
            raise ValueError(
                "Prior requires distinct names and matching bounds")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Prior names must be nonempty strings")
        if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
               or not math.isfinite(hi - lo) for lo, hi in bounds):
            raise ValueError("Prior bounds require finite positive widths")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "bounds", bounds)

    @property
    def digest(self) -> str:
        """Include support, parameter meanings and normalized prior family."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "independent_uniform",
                    "prior": asdict(self)
                },
                sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class SamplerConfig:
    """Explicit simulator budget and deterministic temperature schedule."""
    particles: int = 512
    temperatures: int = 32
    moves: int = 4
    proposal_scale: float = 0.05  # fraction of each prior width
    max_evaluations: int = 100000
    resample_ess_fraction: float = 0.5

    def __post_init__(self) -> None:
        for value in (self.particles, self.temperatures, self.moves,
                      self.max_evaluations):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Sampler counts must be positive integers")
        if not math.isfinite(self.proposal_scale) or self.proposal_scale <= 0:
            raise ValueError("Proposal scale must be finite and positive")
        if not 0 < self.resample_ess_fraction <= 1:
            raise ValueError("Resampling ESS fraction must lie in (0, 1]")


@dataclass(frozen=True)
class BatchPosterior:
    """Versioned candidate result; never a publication or adequacy certificate.

    Samples are joint vectors, including uncertain initial states.
    Failure results contain no posterior samples, even if a partial
    temperature was reached. ESS is measured before each resampling;
    retained weights must not disguise earlier weight collapse or
    missing modes.
    """
    identity: InferenceIdentity
    prior: BoxPrior
    config: SamplerConfig
    seed: int
    status: Literal["complete", "budget_exhausted", "no_particle_support"]
    samples: Tuple[Tuple[float, ...], ...]
    weights: Tuple[float, ...]
    evaluations: int
    completed_temperature: float
    initial_finite: int
    effective_sample_sizes: Tuple[float, ...]
    accepted_moves: int
    attempted_moves: int
    surviving_ancestors: int
    resampling_count: int
    schema_version: Literal[1] = 1
    estimator: Literal["offline_tempered_smc_box"] = "offline_tempered_smc_box"

    def marginal_quantiles(
        self, name: str, probabilities: Tuple[float, ...] = (.05, .5, .95)
    ) -> Tuple[float, ...]:
        """Summarize the same joint samples, without a separate width fit.

        These are empirical posterior quantiles, not a calibration
        claim. Failed or incomplete numerical results have no credible
        intervals.
        """
        if self.status != "complete" or not self.samples:
            raise ValueError("No completed posterior samples")
        if not probabilities or any(not math.isfinite(p) or p < 0 or p > 1
                                    for p in probabilities):
            raise ValueError("Quantile probabilities must lie in [0, 1]")
        index = self.prior.names.index(name)
        values = np.asarray(self.samples)[:, index]
        weights = np.asarray(self.weights)
        keep = weights > 0
        values, weights = values[keep], weights[keep]
        order = np.argsort(values, kind="stable")
        values, weights = values[order], weights[order]
        cumulative = np.cumsum(weights / weights.sum())
        cumulative[-1] = 1.0
        # Inverse empirical CDF, including endpoints, with zero-mass samples
        # excluded. This convention does not interpolate across mode gaps.
        return tuple(
            float(values[np.searchsorted(cumulative, p)])
            for p in probabilities)


class _BudgetExceeded(Exception):
    """Internal control flow for an exhausted evaluation allowance."""


def sample_batch(prior: BoxPrior, identity: InferenceIdentity,
                 log_likelihood: Callable[[np.ndarray], float],
                 config: SamplerConfig, seed: int) -> BatchPosterior:
    """Sample a fixed-prior target from scratch, without carried fit weights.

    The callable must evaluate the immutable complete dataset under the
    identified simulator and sensor model. It receives an owned candidate;
    it must not use future observations, mutable hidden caches, or random
    transitions. Program/setup exceptions propagate rather than becoming
    zero likelihood. Negative infinity is legitimate zero support, whereas
    NaN and positive infinity are evaluation errors.

    Initialization draws from the actual prior. Each fixed temperature
    reweights by the likelihood increment, resamples only below the declared
    ESS threshold, then applies symmetric random-walk Metropolis moves on the
    uniform support. Without resampling, importance weights are retained
    through the target-invariant moves. Out-of-support proposals are rejected,
    never clipped. Final weighted particles target beta=1, but diagnostics and
    repeatability
    checks remain necessary; no ESS threshold certifies undiscovered modes.
    """
    if identity.prior != prior.digest:
        raise ValueError("Prior differs from immutable inference identity")
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(prior.bounds).T
    count = config.particles
    particles = rng.uniform(lower, upper, size=(count, len(prior.names)))
    ancestors = np.arange(count)
    likelihoods = np.full(count, -np.inf)
    weights = np.full(count, 1.0 / count)
    resampling_count = 0
    evaluations = 0
    completed = 0.0
    initial_finite = 0
    ess_values: List[float] = []
    accepted = 0
    attempted = 0

    def evaluate(candidate: np.ndarray) -> float:
        nonlocal evaluations
        if evaluations >= config.max_evaluations:
            raise _BudgetExceeded
        evaluations += 1
        value = float(log_likelihood(candidate.copy()))
        if math.isnan(value) or value == math.inf:
            raise ValueError("Likelihood returned NaN or positive infinity")
        return value

    def result(
        status: Literal["complete", "budget_exhausted", "no_particle_support"]
    ) -> BatchPosterior:
        complete = status == "complete"
        return BatchPosterior(
            identity=identity,
            prior=prior,
            config=config,
            seed=seed,
            status=status,
            samples=tuple(tuple(float(v) for v in row)
                          for row in particles) if complete else (),
            weights=tuple(float(w) for w in weights) if complete else (),
            evaluations=evaluations,
            completed_temperature=completed,
            initial_finite=initial_finite,
            effective_sample_sizes=tuple(ess_values),
            accepted_moves=accepted,
            attempted_moves=attempted,
            surviving_ancestors=len(set(ancestors.tolist())),
            resampling_count=resampling_count)

    try:
        for i in range(count):
            likelihoods[i] = evaluate(particles[i])
            initial_finite += int(math.isfinite(likelihoods[i]))
        if not initial_finite:
            # This is not proof the model/data are impossible: a finite
            # initialization may simply have missed valid support.
            return result("no_particle_support")
        for stage in range(1, config.temperatures + 1):
            beta = stage / config.temperatures
            # Center before multiplication to avoid loss of stability from
            # large normalizing constants common to every candidate.
            log_weights = np.full(count, -np.inf)
            np.log(weights, out=log_weights, where=weights > 0)
            log_weights += (beta - completed) * (likelihoods -
                                                 np.max(likelihoods))
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= weights.sum()
            ess = float(1.0 / np.dot(weights, weights))
            ess_values.append(ess)
            if ess < config.resample_ess_fraction * count:
                indices = rng.choice(count,
                                     size=count,
                                     replace=True,
                                     p=weights)
                particles = particles[indices].copy()
                likelihoods = likelihoods[indices].copy()
                ancestors = ancestors[indices]
                weights.fill(1.0 / count)
                resampling_count += 1
            for _ in range(config.moves):
                for i in range(count):
                    attempted += 1
                    proposal = particles[i] + rng.normal(
                        size=len(prior.names)) * (upper - lower) * \
                        config.proposal_scale
                    if not np.all(np.isfinite(proposal)) or np.any(
                            proposal < lower) or np.any(proposal > upper):
                        continue
                    proposed_likelihood = evaluate(proposal)
                    if proposed_likelihood == -math.inf:
                        continue
                    log_ratio = beta * (proposed_likelihood - likelihoods[i])
                    # 1-random lies in (0, 1], so log never sees zero.
                    if math.log(1.0 - rng.random()) < log_ratio:
                        particles[i] = proposal
                        likelihoods[i] = proposed_likelihood
                        accepted += 1
            completed = beta
        return result("complete")
    except _BudgetExceeded:
        return result("budget_exhausted")
