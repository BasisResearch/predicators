"""Sequential integration of complete stochastic histories, for offline use.

Fixed-size multinomial resampling operates between declared blocks. Each
block retains its conditional target/proposal density ratio, including
exact observation factors. The accumulated normalizer estimates a joint
history density; terminal weights alone do not provide that density or
certify a usable posterior. Independent runs are needed to assess error.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Generic, List, Literal, Optional, Tuple, TypeVar

import numpy as np

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError

History = TypeVar("History")


@dataclass(frozen=True)
class SequentialPathIntegral(Generic[History]):
    """A joint density estimate with concentration and ancestry diagnostics.

    Histories are owned by this result, but their payloads may be
    mutable. Within-run paths share ancestors and are not independent
    estimates. A large terminal ESS cannot erase past collapse or
    missing support.
    """
    log_density: float
    log_increments: Tuple[float, ...]
    effective_terms: Tuple[float, ...]
    surviving_ancestors: Tuple[int, ...]
    histories: Tuple[History, ...]
    weights: Tuple[float, ...]
    status: Literal["finite_estimate", "no_sample_support"]


def integrate_sequential_paths(advance: Callable[
    [Optional[History], int, np.random.Generator], Tuple[History, float]],
                               count: int, stages: int,
                               seed: int) -> SequentialPathIntegral[History]:
    """Extend histories and multiply block normalizer estimates.

    advance(None, 0, rng) starts from the same fixed supported prefix.
    Later calls receive a deep copy of a resampled parent history and
    the zero-based block index. A callback must preserve that history,
    draw its extension from a normalized conditional proposal and return
    only the new block's log target/proposal factor. Deterministic
    history reconstruction may use a native simulator; live simulator
    handles are not suitable history payloads.

    Resampling occurs at every nonfinal boundary under the normalized
    block factors, including zero-support paths with zero probability.
    Independent random draws extend repeated parents. Exceptions and
    invalid factors abort, never discard or retry a failed callback.
    No sampled support does not prove a model is inconsistent.
    """
    for name, argument, minimum in (("count", count, 2), ("stages", stages, 1),
                                    ("seed", seed, 0)):
        if not isinstance(argument, int) or isinstance(argument, bool) or \
                argument < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = np.random.default_rng(seed)
    parents: Tuple[Optional[History], ...] = (None, ) * count
    ancestors = np.arange(count)
    increments: List[float] = []
    effective: List[float] = []
    lineage: List[int] = []
    for stage in range(stages):
        children = []
        factors = []
        for parent in parents:
            child, factor = advance(copy.deepcopy(parent), stage, rng)
            factor = float(factor)
            if math.isnan(factor) or factor == math.inf:
                raise ConditioningNumericalError("Invalid block log factor")
            children.append(copy.deepcopy(child))
            factors.append(factor)
        peak = max(factors)
        if peak == -math.inf:
            return SequentialPathIntegral(-math.inf,
                                          tuple(increments + [-math.inf]),
                                          tuple(effective + [0.]),
                                          tuple(lineage + [0]), (), (),
                                          "no_sample_support")
        scaled = [math.exp(factor - peak) for factor in factors]
        total = math.fsum(scaled)
        weights = np.asarray(scaled) / total
        # Explicit final normalization for the categorical sampling API.
        weights /= weights.sum()
        increment = peak + math.log(total / count)
        if not math.isfinite(increment):
            raise ConditioningNumericalError("Block normalizer overflow")
        increments.append(increment)
        effective.append(total * total / math.fsum(value * value
                                                   for value in scaled))
        lineage.append(len(set(ancestors[weights > 0].tolist())))
        if stage + 1 < stages:
            indices = rng.choice(count, size=count, p=weights)
            parents = tuple(children[i] for i in indices)
            ancestors = ancestors[indices]
    try:
        value = math.fsum(increments)
    except OverflowError as exc:
        raise ConditioningNumericalError("Joint normalizer overflow") from exc
    if not math.isfinite(value):
        raise ConditioningNumericalError("Joint normalizer overflow")
    return SequentialPathIntegral(value, tuple(increments), tuple(effective),
                                  tuple(lineage), tuple(children),
                                  tuple(float(w) for w in weights),
                                  "finite_estimate")
