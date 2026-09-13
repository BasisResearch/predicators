"""Ordered, indivisible target evaluations for offline inference workers.

Each worker must own the entire conditional map and likelihood
calculation. In particular, native worlds cannot be shared between
concurrent evaluations. The caller owns process isolation, executor
lifetime and runtime provenance.
"""
import math
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class TargetEvaluation:
    """Proposal identity, full joint point and its two distinct log factors."""
    proposal: Tuple[float, ...]
    joint: Tuple[float, ...]
    log_base: float
    log_likelihood: float

    def __post_init__(self) -> None:
        for name in ("proposal", "joint"):
            values = tuple(float(v) for v in getattr(self, name))
            if not values or not all(math.isfinite(v) for v in values):
                raise ValueError("Target coordinates must be finite")
            object.__setattr__(self, name, values)
        for name in ("log_base", "log_likelihood"):
            value = float(getattr(self, name))
            if math.isnan(value) or value == math.inf:
                raise ValueError("Target factor is NaN or positive infinity")
            object.__setattr__(self, name, value)
        if self.log_base == -math.inf and self.log_likelihood != -math.inf:
            raise ValueError("Zero base support requires zero target support")


@dataclass(frozen=True)
class BatchedTarget:
    """Evaluate immutable proposals in order, with no dropped or retried rows.

    A synchronous map and a process executor map use the same contract.
    Completion order must not change returned order. Setup and worker
    errors propagate; zero support is a returned density, not a caught
    exception. All dependencies must be identified by the sampler's
    runtime identity.
    """
    evaluate_many: Callable[[Tuple[Tuple[float, ...], ...]],
                            Tuple[TargetEvaluation, ...]]

    def evaluate(self, proposals: Tuple[Tuple[float, ...],
                                        ...], joint_dimension: int,
                 conditional: bool) -> Tuple[TargetEvaluation, ...]:
        """Validate identities before the sampler can consume any result."""
        results = tuple(self.evaluate_many(proposals))
        if len(results) != len(proposals):
            raise ValueError("Target batch returned the wrong number of rows")
        for proposal, result in zip(proposals, results):
            if result.proposal != proposal:
                raise ValueError(
                    "Target batch changed proposal identity/order")
            if len(result.joint) != joint_dimension:
                raise ValueError(
                    "Target batch returned invalid joint dimension")
            if not conditional and (result.joint != proposal
                                    or result.log_base != 0.):
                raise ValueError(
                    "Box target must preserve coordinates and base")
        return results
