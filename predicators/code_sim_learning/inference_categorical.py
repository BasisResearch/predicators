"""Defensive proposals for discrete cases of an unchanged physical prior.

Observation guidance changes sampling frequencies only. Retain the
returned prior/proposal correction and every observation factor used by
the guide.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Tuple

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_data import content_digest


@dataclass(frozen=True)
class CategoricalPoint:
    """A selected case and its original-prior/proposal log ratio."""
    index: int
    log_weight: float


@dataclass(frozen=True)
class CategoricalProposal:
    """Mix a positive categorical prior with a likelihood-guided proposal.

    The guide is proportional to prior mass times the supplied guide
    likelihood. Impossible guide cases still receive defensive prior
    mass. This class does not condition the physical prior on the guide.
    """
    prior: Tuple[float, ...]
    log_guide: Tuple[float, ...]
    prior_mass: float = .25

    def __post_init__(self) -> None:
        prior = tuple(float(p) for p in self.prior)
        guide = tuple(float(v) for v in self.log_guide)
        if not prior or len(prior) != len(guide) or any(
                not math.isfinite(p) or p <= 0 or p > 1 for p in prior):
            raise ValueError("Positive normalized prior masses are required")
        total = math.fsum(prior)
        if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError("Prior masses must sum to one")
        if any(math.isnan(v) or v == math.inf for v in guide) or \
                all(v == -math.inf for v in guide):
            raise ValueError("Guide needs a finite case and no NaN or +inf")
        if not 0 < self.prior_mass < 1:
            raise ValueError("Defensive prior mass must lie strictly in (0,1)")
        object.__setattr__(self, "prior", tuple(p / total for p in prior))
        object.__setattr__(self, "log_guide", guide)
        # A positive mathematical mass may have no representable interval
        # after cumulative rounding. Do not silently omit such a case.
        probabilities = self.probabilities
        previous = 0.
        for i in range(len(prior)):
            boundary = math.fsum(probabilities[:i + 1])
            if boundary <= previous or (i < len(prior) - 1 and boundary >= 1):
                raise ConditioningNumericalError(
                    "Categorical probability interval is not representable")
            previous = boundary

    @property
    def probabilities(self) -> Tuple[float, ...]:
        """Normalized proposal probabilities, including defensive mass."""
        largest = max(self.log_guide)
        scaled = tuple(p * math.exp(v - largest)
                       for p, v in zip(self.prior, self.log_guide))
        total = math.fsum(scaled)
        masses = tuple(self.prior_mass * p + (1 - self.prior_mass) * w / total
                       for p, w in zip(self.prior, scaled))
        normalizer = math.fsum(masses)
        return tuple(p / normalizer for p in masses)

    @property
    def digest(self) -> str:
        """Identify prior, guide and proposal separately from data factors."""
        return content_digest(
            json.dumps({
                "schema": 1,
                "proposal": asdict(self)
            }, sort_keys=True).encode("utf-8"))

    def transform(self, unit: float) -> CategoricalPoint:
        """Map one unit uniform to a case, retaining its density correction."""
        if not math.isfinite(unit) or not 0 <= unit <= 1:
            raise ValueError("Expected a finite unit coordinate")
        probabilities = self.probabilities
        index = len(probabilities) - 1
        for i in range(len(probabilities) - 1):
            if unit < math.fsum(probabilities[:i + 1]):
                index = i
                break
        return CategoricalPoint(
            index,
            math.log(self.prior[index]) - math.log(probabilities[index]))
