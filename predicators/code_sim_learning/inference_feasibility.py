"""Offline whole-candidate rejection for a declared conditional prior.

Drawing a complete base candidate and accepting exactly when C holds
samples p0(x) I[C(x)] / Z, where Z is its acceptance probability. This
procedure does not estimate a normalized density or model evidence. The
same construction across mixture cases changes their accepted masses in
proportion to feasibility. Resampling only failed parts or normalizing
each case separately would define a different prior.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Generic, List, Literal, Optional, Tuple, TypeVar

import numpy as np

from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_sampling import \
    ConditionedPrior, PriorPoint

T = TypeVar("T")


@dataclass(frozen=True)
class FeasibleConditioning:
    """Compose exact conditioning and geometric support without losing mass.

    global_joint declares p0(theta, s) I[C(theta, s)] / Z, with one
    constant Z for the full joint distribution. conditional_state instead
    preserves p0(theta) and normalizes the state law separately for each
    theta. It requires the original, pre-observation log acceptance
    probability log Z(theta). The two laws generally give different
    parameter posteriors, even with identical feasible sets.

    The base map retains exact-observation density and proposal factors.
    A support check operates on its complete lifted candidate. It must
    not independently redraw parts or recondition on noisy observations.
    Dependencies, geometric tolerances and callback source/closure must
    be covered by the supplied identities. This is an offline contract,
    not automatic normalization or a physical scene implementation.
    """
    base: ConditionedPrior
    support: str
    normalization: Literal["global_joint", "conditional_state"]
    normalizer_identity: Optional[str] = None

    def __post_init__(self) -> None:
        if self.normalization not in ("global_joint", "conditional_state"):
            raise ValueError("Declare global_joint or conditional_state")
        if (self.normalization == "conditional_state") != \
                (self.normalizer_identity is not None):
            raise ValueError("Conditional state support needs a normalizer "
                             "identity; global joint support must omit it")
        for digest in (self.support, self.normalizer_identity):
            if digest is not None and (len(digest) != 64
                                       or any(c not in "0123456789abcdef"
                                              for c in digest)):
                raise ValueError("Support identities must be SHA256 digests")

    @property
    def prior(self) -> ConditionedPrior:
        """Identify the supported original law separately from observations.

        Changing data in the base conditioning leaves original_prior
        unchanged. Changing the support or normalization changes that
        original law. The omitted global constant permits posterior
        ratios for this fixed target, never model-evidence comparisons.
        """
        original = content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "feasible_original_prior",
                    "original_prior": self.base.original_prior,
                    "support": self.support,
                    "normalization": self.normalization,
                    "normalizer": self.normalizer_identity
                },
                sort_keys=True).encode())
        conditioning = content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "base_conditioning": self.base.conditioning,
                    "supported_prior": original
                },
                sort_keys=True).encode())
        return ConditionedPrior(self.base.names, original, conditioning,
                                self.base.proposal)

    def lift(
        self,
        free: np.ndarray,
        condition: Callable[[np.ndarray], PriorPoint],
        feasible: Callable[[np.ndarray], bool],
        *,
        log_normalizer: Optional[Callable[[np.ndarray], float]] = None
    ) -> PriorPoint:
        """Return the full conditional-base weight before likelihood tempering.

        conditional_state requires a deterministic declared normalizer,
        not an observed finite rejection rate. Its callback may depend
        only on variables retained by the original conditional state
        law, such as theta, not the sampled state or current data.
        Missing normalization is unsupported, never silently set to one.
        Exceptions remain setup/numerical failures, not zero likelihood.
        Each callback receives an owned array to prevent shared
        mutation.
        """
        if (self.normalization == "conditional_state") != \
                (log_normalizer is not None):
            raise ValueError("Supply a log normalizer exactly for "
                             "conditional_state")
        point = condition(np.array(free, dtype=float, copy=True))
        joint = np.asarray(point.joint, dtype=float)
        if joint.shape != (len(self.base.names), ) or \
                not np.isfinite(joint).all():
            raise ValueError("Base map returned invalid joint values")
        weight = float(point.log_weight)
        if math.isnan(weight) or weight == math.inf:
            raise ValueError("Base map returned invalid log weight")
        values = tuple(float(v) for v in joint)
        if weight == -math.inf:
            return PriorPoint(values, weight)
        valid = feasible(joint.copy())
        if not isinstance(valid, (bool, np.bool_)):
            raise TypeError("Feasibility predicate must return a boolean")
        if not valid:
            return PriorPoint(values, -math.inf)
        if log_normalizer is not None:
            normalizer = float(log_normalizer(joint.copy()))
            if not math.isfinite(normalizer) or normalizer > 0:
                raise ValueError("Log support probability must be finite "
                                 "and nonpositive on feasible candidates")
            weight -= normalizer
            if not math.isfinite(weight):
                raise ArithmeticError("Supported base weight overflow")
        return PriorPoint(values, weight)


@dataclass(frozen=True)
class FeasibleDraws(Generic[T]):
    """Complete prior draws or an explicit exhausted rejection budget.

    accepted counts successes seen even when fewer than requested; no
    partial batch is exposed as a completed result. A failed search does
    not prove empty support. These are prior, not posterior draws.
    """
    original_prior: str
    support: str
    seed: int
    requested: int
    max_draws: int
    draws: int
    accepted: int
    samples: Tuple[T, ...]
    status: Literal["complete", "budget_exhausted"]


def draw_feasible(original_prior: str, support: str,
                  draw: Callable[[np.random.Generator],
                                 T], feasible: Callable[[T], bool], *,
                  count: int, max_draws: int, seed: int) -> FeasibleDraws[T]:
    """Sample whole candidates with a fixed, explicitly identified predicate.

    draw must return independently drawn owned values from the declared
    normalized base prior. The predicate must be deterministic and its
    geometry, runtime and numerical policy must be included in support.
    Errors propagate instead of being classified as collision rejection.

    The acceptance rate is not an exact normalizer. Fixed-target
    posterior ratios can omit a common Z, but changing parameters or
    mixture-specific normalization may change Z and must not silently
    omit it. This function makes no parameter-independence assertion.
    """
    for digest in (original_prior, support):
        if len(digest) != 64 or any(c not in "0123456789abcdef"
                                    for c in digest):
            raise ValueError("Feasibility identities must be SHA256 digests")
    if any(not isinstance(v, int) or v <= 0 for v in (count, max_draws)):
        raise ValueError("Rejection counts must be positive integers")
    rng = np.random.default_rng(seed)
    accepted: List[T] = []
    attempts = 0
    while attempts < max_draws and len(accepted) < count:
        candidate = draw(rng)
        attempts += 1
        valid = feasible(candidate)
        if not isinstance(valid, (bool, np.bool_)):
            raise TypeError("Feasibility predicate must return a boolean")
        if valid:
            accepted.append(candidate)
    complete = len(accepted) == count
    return FeasibleDraws(original_prior, support, seed, count, max_draws,
                         attempts, len(accepted),
                         tuple(accepted) if complete else (),
                         "complete" if complete else "budget_exhausted")
