"""Offline whole-candidate rejection for a declared conditional prior.

Drawing a complete base candidate and accepting exactly when C holds
samples p0(x) I[C(x)] / Z, where Z is its acceptance probability. This
procedure does not estimate a normalized density or model evidence. The
same construction across mixture cases changes their accepted masses in
proportion to feasibility. Resampling only failed parts or normalizing
each case separately would define a different prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Literal, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


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
