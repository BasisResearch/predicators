"""Normalized defensive proposals for a fixed uniform inference prior.

Guidance changes where candidates are evaluated, not the prior or the
likelihood. The caller must retain all observation factors used to build
the guide and separately account for physical support or conditioning.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
from scipy.stats import truncnorm

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior, \
    PriorPoint


@dataclass(frozen=True)
class GaussianBoxProposal:
    """Mix a box prior with an independent truncated Gaussian guide.

    The guide has the same box support as the original prior. A single
    mixture draw selects the entire vector, so the proposal density is a
    mixture of products, not a product of marginal mixtures. Centers may
    come from a declared observation prefix; they never redefine the
    prior.
    """
    prior: BoxPrior
    centers: Tuple[float, ...]
    scales: Tuple[float, ...]
    prior_mass: float = .1

    def __post_init__(self) -> None:
        centers = tuple(float(v) for v in self.centers)
        scales = tuple(float(v) for v in self.scales)
        if len(centers) != len(self.prior.names) or \
                len(scales) != len(centers):
            raise ValueError("Guide coordinates must match the prior")
        if not 0 < self.prior_mass < 1 or any(
                not math.isfinite(v)
                for v in centers) or any(not math.isfinite(v) or v <= 0
                                         for v in scales):
            raise ValueError("Finite guide centers, positive scales and a "
                             "strictly interior mixture mass are required")
        for (lo, hi), center, scale in zip(self.prior.bounds, centers, scales):
            if not all(math.isfinite((v - center) / scale) for v in (lo, hi)):
                raise ValueError("Standardized guide bounds must be finite")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "scales", scales)

    @property
    def digest(self) -> str:
        """Identify proposal choices separately from the original prior."""
        return content_digest(
            json.dumps({
                "schema": 1,
                "proposal": asdict(self)
            }, sort_keys=True).encode("utf-8"))

    def log_density(self, point: Tuple[float, ...]) -> float:
        """Return the normalized mixture density in physical coordinates."""
        if len(point) != len(self.prior.names) or any(not math.isfinite(v)
                                                      for v in point):
            raise ValueError("A finite point matching the prior is required")
        if any(not lo <= v <= hi
               for v, (lo, hi) in zip(point, self.prior.bounds)):
            return -math.inf
        gaussian_terms = []
        for value, (lo, hi), center, scale in zip(point, self.prior.bounds,
                                                  self.centers, self.scales):
            term = float(
                truncnorm.logpdf(value, (lo - center) / scale,
                                 (hi - center) / scale,
                                 loc=center,
                                 scale=scale))
            if not math.isfinite(term):
                raise ConditioningNumericalError("Guide density overflow")
            gaussian_terms.append(term)
        uniform = -math.fsum(math.log(hi - lo) for lo, hi in self.prior.bounds)
        result = float(
            np.logaddexp(
                math.log(self.prior_mass) + uniform,
                math.log1p(-self.prior_mass) + math.fsum(gaussian_terms)))
        if not math.isfinite(result):
            raise ConditioningNumericalError("Mixture density overflow")
        return result

    def transform(self, unit: Tuple[float, ...]) -> PriorPoint:
        """Map independent uniforms to a candidate and log(prior/proposal).

        One extra coordinate selects the mixture component. This
        returned weight contains no observation likelihood and no scene-
        feasibility normalizer. Those factors remain the caller's
        responsibility.
        """
        if len(unit) != len(self.prior.names) + 1 or any(
                not math.isfinite(v) or not 0 <= v <= 1 for v in unit):
            raise ValueError("Expected one unit coordinate per dimension "
                             "plus one mixture coordinate")
        values = []
        for value, (lo, hi), center, scale in zip(unit, self.prior.bounds,
                                                  self.centers, self.scales):
            if unit[-1] < self.prior_mass:
                physical = lo + value * (hi - lo)
            else:
                physical = float(
                    truncnorm.ppf(value, (lo - center) / scale,
                                  (hi - center) / scale,
                                  loc=center,
                                  scale=scale))
            # Exact unit endpoints denote the declared finite box endpoints.
            if value == 0:
                physical = lo
            elif value == 1:
                physical = hi
            if not math.isfinite(physical) or not lo <= physical <= hi:
                raise ConditioningNumericalError("Guide quantile left support")
            values.append(physical)
        joint = tuple(values)
        uniform = -math.fsum(math.log(hi - lo) for lo, hi in self.prior.bounds)
        return PriorPoint(joint, uniform - self.log_density(joint))
