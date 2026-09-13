"""Monte Carlo density integration over explicitly conditioned future paths.

The caller supplies independent complete-path log integrands, including
all retained observation densities and any proposal correction. Exact
conditioning belongs in that path construction, not in a tolerance band
or equality test against unconditional draws. This module summarizes the
integral; it does not certify support coverage or numerical adequacy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError


@dataclass(frozen=True)
class PathIntegral:
    """A density estimate and empirical diagnostics, not posterior weights.

    Relative standard error assumes independent draws with finite second
    moment. Zero empirical error does not exclude unsampled rare paths.
    No sampled support leaves the error undefined, not falsely zero.
    Effective terms describe concentration of sampled contributions
    only.
    """
    log_factors: Tuple[float, ...]
    log_density: float
    relative_standard_error: Optional[float]
    effective_terms: float
    status: Literal["finite_estimate", "no_sample_support"]


def summarize_path_integral(log_factors: Tuple[float, ...]) -> PathIntegral:
    """Average complete-path densities, retaining zero-support draws.

    These are log target/proposal factors from the same normalized
    sampling law, not normalized posterior weights or log densities from
    independent time-step mixtures. At least two draws are needed for
    the empirical error calculation. NaN and positive infinity are
    numerical errors; negative infinity is a legitimate zero integrand.
    """
    factors = tuple(float(value) for value in log_factors)
    if len(factors) < 2:
        raise ValueError("At least two independent path draws required")
    if any(math.isnan(value) or value == math.inf for value in factors):
        raise ConditioningNumericalError("Invalid path log integrand")
    peak = max(factors)
    if peak == -math.inf:
        return PathIntegral(factors, -math.inf, None, 0., "no_sample_support")
    scaled = tuple(math.exp(value - peak) for value in factors)
    total = math.fsum(scaled)
    count = len(scaled)
    mean = total / count
    squares = math.fsum(value * value for value in scaled)
    centered = math.fsum((value - mean)**2 for value in scaled)
    error = math.sqrt(centered / (count * (count - 1))) / mean
    return PathIntegral(factors, peak + math.log(mean), error,
                        total * total / squares, "finite_estimate")


def integrate_conditional_paths(log_integrand: Callable[[np.random.Generator],
                                                        float], count: int,
                                seed: int) -> PathIntegral:
    """Draw independent conditional paths from an explicit local generator.

    Each call must rebuild the same supported prefix and sample one new
    future history. Evaluated future observations may guide a normalized
    conditional proposal only when its density accounting is retained.
    This scoring operation must remain separate from future generation.
    Callback exceptions abort the computation; no failed path is
    dropped.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError("At least two independent path draws required")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Nonnegative integer integration seed required")
    rng = np.random.default_rng(seed)
    factors = []
    for _ in range(count):
        value = float(log_integrand(rng))
        if math.isnan(value) or value == math.inf:
            raise ConditioningNumericalError("Invalid path log integrand")
        factors.append(value)
    return summarize_path_integral(tuple(factors))
