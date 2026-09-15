"""Collapsed Gaussian discrepancy variance for offline trajectory inference.

One inverse-gamma variance is shared across a channel's residual
history. This is a separately declared dynamics or output discrepancy,
not a change to sensor noise. Callers must identify the channel,
primitive time step, units, original prior and episode-sharing policy in
their model identity. No production estimator or physical state is
modified by this component.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_data import content_digest


def _square(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Residuals must be finite")
    result = value * value
    if not math.isfinite(result):
        raise ConditioningNumericalError("Residual square overflow")
    return result


@dataclass(frozen=True)
class GaussianVariancePrior:
    """Variance v has density beta**alpha/Gamma(alpha) v**(-alpha-1)
    exp(-beta/v); residuals given v are independent Normal(0, v).

    beta has squared channel units. The prior is fixed before seeing the
    fitted residual history. It is not replaced by a previous posterior
    when fitting that same history again.
    """
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0
               for v in (self.alpha, self.beta)):
            raise ValueError("Positive finite shape and scale required")

    @property
    def digest(self) -> str:
        """Identify the normalized shared-variance law and original prior."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family":
                    "zero_mean_gaussian_shared_inverse_gamma_variance",
                    "alpha": float(self.alpha),
                    "beta": float(self.beta)
                },
                sort_keys=True).encode("utf-8"))

    def condition(self,
                  residuals: Sequence[float]) -> GaussianVariancePosterior:
        """Condition once on a complete sequence, retaining its density."""
        try:
            squared = math.fsum(_square(r) for r in residuals)
        except OverflowError as exc:
            raise ConditioningNumericalError("Residual sum overflow") from exc
        return GaussianVariancePosterior(self, len(residuals), squared)


@dataclass(frozen=True)
class GaussianVariancePosterior:
    """Sufficient statistics for one originally declared variance.

    Sequential predictive densities integrate the same variance, rather
    than independently mixing a new variance at every step. For a future
    simulation, draw one variance and retain it for that entire history.
    """
    prior: GaussianVariancePrior
    count: int
    sum_squares: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or \
                self.count < 0:
            raise ValueError("Residual count must be a nonnegative integer")
        if not math.isfinite(self.sum_squares) or self.sum_squares < 0:
            raise ValueError("Finite nonnegative residual sum required")
        if self.count == 0 and self.sum_squares != 0:
            raise ValueError("Empty history cannot have residual energy")

    @property
    def alpha(self) -> float:
        """Conditional inverse-gamma shape."""
        return self.prior.alpha + self.count / 2

    @property
    def beta(self) -> float:
        """Conditional inverse-gamma scale in squared channel units."""
        result = self.prior.beta + self.sum_squares / 2
        if not math.isfinite(result):
            raise ConditioningNumericalError("Variance scale overflow")
        return result

    @property
    def log_evidence(self) -> float:
        """Normalized complete residual-history density under the prior."""
        if not self.count:
            return 0.
        result = (math.lgamma(self.alpha) - math.lgamma(self.prior.alpha) +
                  self.prior.alpha * math.log(self.prior.beta) -
                  self.alpha * math.log(self.beta) -
                  self.count / 2 * math.log(2 * math.pi))
        if not math.isfinite(result):
            raise ConditioningNumericalError("Variance evidence overflow")
        return result

    def log_predictive(self, residual: float) -> float:
        """Student-t density for one additional exactly observed residual."""
        energy = _square(residual)
        # Separate logarithms to avoid overflow in 2*pi*beta.
        ratio = energy / self.beta / 2
        result = (math.lgamma(self.alpha + .5) - math.lgamma(self.alpha) - .5 *
                  (math.log(2 * math.pi) + math.log(self.beta)) -
                  (self.alpha + .5) * math.log1p(ratio))
        if not math.isfinite(result):
            raise ConditioningNumericalError("Variance predictive overflow")
        return result

    def advance(self, residual: float) -> GaussianVariancePosterior:
        """Append new evidence while preserving the original prior."""
        try:
            energy = math.fsum((self.sum_squares, _square(residual)))
        except OverflowError as exc:
            raise ConditioningNumericalError("Residual sum overflow") from exc
        return GaussianVariancePosterior(self.prior, self.count + 1, energy)

    def draw_variance(self, rng: np.random.Generator) -> float:
        """Draw one shared variance for a complete future continuation."""
        gamma = float(rng.gamma(self.alpha))
        if not math.isfinite(gamma) or gamma <= 0:
            raise ConditioningNumericalError("Unrepresentable gamma draw")
        result = self.beta / gamma
        if not math.isfinite(result) or result <= 0:
            raise ConditioningNumericalError("Unrepresentable variance draw")
        return result
