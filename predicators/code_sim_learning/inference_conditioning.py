"""Offline change-of-variables reference for exact affine observations.

This is a conditional-coordinate construction, not a contact simulator
constraint solver or a replacement for the production fitter. For free
coordinates u and eliminated coordinates z, the declared observation is
y = A(u) z + b(u). A must be square and nonsingular at the evaluated u.
The conditional density in u is proportional to p(u, z(u)) / |det A(u)|.
Callers must supply the actual declared affine equation, not a local
linearization of a nonlinear simulator output.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior


class UnsupportedConditioning(ValueError):
    """This elimination chart cannot represent the requested constraint.

    A singular chart does not establish that the model is inconsistent;
    another coordinate choice or a different representation may work.
    """


class ConditioningNumericalError(ValueError):
    """Linear algebra failed to resolve this otherwise declared chart."""


@dataclass(frozen=True)
class ConditionalPoint:
    """Lifted coordinates and a base importance factor, without noisy data.

    The weight is relative to a uniform proposal on the original free
    coordinate box. It is unnormalized: a single point neither defines a
    posterior nor proves that the entire constraint has support.
    Negative infinity means this point is outside the original prior.
    """
    joint: Tuple[float, ...]
    log_base_weight: float
    max_constraint_residual: float
    numerical_residual_bound: float


@dataclass(frozen=True)
class AffineConditioning:
    """Eliminate observed coordinates under an immutable original box prior.

    The equation identity must hash the source and closure that
    construct A(u) and b(u); these values are evaluated by the caller.
    This module does not verify runtime closure. Observations and
    eliminated-coordinate order are included in the digest. No first
    noisy reading defines a new prior. Discrete cases and nonlinear or
    redundant constraints need separate representations.
    """
    prior: BoxPrior
    eliminated: Tuple[str, ...]
    observed: Tuple[float, ...]
    equation_identity: str

    def __post_init__(self) -> None:
        eliminated = tuple(self.eliminated)
        observed = tuple(float(v) for v in self.observed)
        if not eliminated or len(set(eliminated)) != len(eliminated) or \
                not set(eliminated) <= set(self.prior.names):
            raise ValueError("Eliminate distinct coordinates in the prior")
        if len(observed) != len(eliminated) or not all(
                math.isfinite(v) for v in observed):
            raise ValueError(
                "One finite observation per eliminated coordinate")
        if len(self.equation_identity) != 64 or any(
                c not in "0123456789abcdef" for c in self.equation_identity):
            raise ValueError("Equation identity must be a SHA256 digest")
        object.__setattr__(self, "eliminated", eliminated)
        object.__setattr__(self, "observed", observed)

    @property
    def free_names(self) -> Tuple[str, ...]:
        """Original order with exactly conditioned coordinates removed."""
        return tuple(n for n in self.prior.names if n not in self.eliminated)

    @property
    def free_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """Proposal bounds, not a claim of uniform conditional density."""
        return tuple(b for n, b in zip(self.prior.names, self.prior.bounds)
                     if n not in self.eliminated)

    @property
    def digest(self) -> str:
        """Identify the original prior, equation, observation and chart."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "affine_elimination_box",
                    "prior": self.prior.digest,
                    "eliminated": self.eliminated,
                    "observed": self.observed,
                    "equation": self.equation_identity,
                    "arithmetic": "float64_solve_backward_error_64eps"
                },
                sort_keys=True).encode("utf-8"))

    def lift(self, free: np.ndarray, matrix: np.ndarray,
             offset: np.ndarray) -> ConditionalPoint:
        """Solve the declared equation and retain its density correction.

        A backward-error bound detects numerical failures only. It does
        not admit an epsilon-wide observation band or change sensor
        noise. Residuals are returned for audit, never used as noisy
        likelihoods. A caller must not score the eliminated equality a
        second time.
        """
        free = np.asarray(free, dtype=np.float64)
        matrix = np.asarray(matrix, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        size = len(self.eliminated)
        if free.shape != (len(self.free_names), ) or \
                matrix.shape != (size, size) or offset.shape != (size, ):
            raise ValueError(
                "Affine chart coordinate or equation shape mismatch")
        if not all(np.all(np.isfinite(v)) for v in (free, matrix, offset)):
            raise ValueError("Affine chart requires finite inputs")
        sign, log_det = np.linalg.slogdet(matrix)
        if sign == 0:
            raise UnsupportedConditioning("Singular affine elimination chart")
        rhs = np.asarray(self.observed) - offset
        try:
            solved = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as err:
            raise ConditioningNumericalError("Affine solve failed") from err
        if not np.all(np.isfinite(solved)) or not math.isfinite(log_det):
            raise ConditioningNumericalError("Nonfinite affine solution")
        residual = float(np.max(np.abs(matrix @ solved - rhs)))
        bound = float(
            64 * np.finfo(np.float64).eps *
            (np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(
                solved, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)))
        if not math.isfinite(residual) or not math.isfinite(bound) or \
                residual > bound:
            raise ConditioningNumericalError("Affine backward error exceeded")
        values = dict(zip(self.free_names, free))
        values.update(zip(self.eliminated, solved))
        joint = tuple(float(values[n]) for n in self.prior.names)
        # p(u,z)/q(u) cancels the free-coordinate uniform widths. Retain the
        # eliminated widths and the observation-to-coordinate Jacobian.
        log_weight = -float(log_det) - sum(
            math.log(hi - lo)
            for name, (lo, hi) in zip(self.prior.names, self.prior.bounds)
            if name in self.eliminated)
        if any(v < lo or v > hi
               for v, (lo, hi) in zip(joint, self.prior.bounds)):
            log_weight = -math.inf
        return ConditionalPoint(joint, log_weight, residual, bound)
