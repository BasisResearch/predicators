"""Offline posterior availability, separate from model predictive adequacy.

The caller supplies an identified numerical assessment protocol and its
checks, including any repeat-run or budget comparisons that it requires.
Finishing a sampler or reporting high ESS alone does not establish
adequacy. This boundary neither publishes a fit nor approves an action.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from predicators.code_sim_learning.inference_data import InferenceIdentity
from predicators.code_sim_learning.inference_sampling import BatchPosterior


@dataclass(frozen=True)
class InferenceCheck:
    """One named check, with an explanation and identified supporting report.

    Predictive names should identify the episode, feature, event or time
    interval being assessed. Unevaluated checks carry an explanation of
    what is missing instead of being counted as passing.
    """
    name: str
    status: Literal["pass", "fail", "unevaluated"]
    detail: str
    evidence: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.detail:
            raise ValueError("Checks need a name and explanation")
        if self.status not in ("pass", "fail", "unevaluated"):
            raise ValueError("Invalid check status")
        if self.status != "unevaluated" and self.evidence is None:
            raise ValueError("Evaluated checks need evidence")
        if self.evidence is not None:
            _validate_digest(self.evidence)


@dataclass(frozen=True)
class AssessmentProtocol:
    """Identify declared numerical criteria, budgets and required checks.

    The source digest identifies their actual definitions, thresholds
    and information boundary. Required names prevent accidentally
    treating a subset of passing checks as a complete assessment.
    Choosing a scientifically adequate protocol remains the caller's
    responsibility; this class does not infer one from sampler output.
    """
    source: str
    required_numerical_checks: Tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.source)
        names = tuple(self.required_numerical_checks)
        if not names or any(not name for name in names) or \
                len(set(names)) != len(names):
            raise ValueError("Protocol needs distinct required checks")
        object.__setattr__(self, "required_numerical_checks", names)


@dataclass(frozen=True)
class AssessedInference:
    """Returned inference and diagnostics without publication side effects.

    An unavailable result has no posterior samples or intervals. Raw
    sampler artifacts can be retained separately for investigation. A
    predictive failure does not remove an otherwise numerically adequate
    posterior. It also does not certify a decision or excuse a
    contradictory exact constraint in the fitted target.
    """
    identity: InferenceIdentity
    protocol: AssessmentProtocol
    numerical_checks: Tuple[InferenceCheck, ...]
    predictive_checks: Tuple[InferenceCheck, ...]
    sampler_status: str
    availability: Literal["available", "numerical_failure", "unevaluated"]
    posterior: Optional[BatchPosterior]


def assess_inference(
    candidate: BatchPosterior,
    protocol: AssessmentProtocol,
    numerical_checks: Tuple[InferenceCheck, ...],
    predictive_checks: Tuple[InferenceCheck, ...] = ()
) -> AssessedInference:
    """Apply explicit numerical checks without gating on prediction quality.

    Missing required checks are recorded as unevaluated. Duplicate and
    undeclared numerical checks are errors rather than silently changed
    protocol definitions. Prediction diagnostics remain attached on all
    paths. No old fit is selected and no posterior weights are changed.
    """
    numerical = tuple(numerical_checks)
    predictive = tuple(predictive_checks)
    for checks in (numerical, predictive):
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("Duplicate assessment check")
    required = protocol.required_numerical_checks
    supplied = {check.name: check for check in numerical}
    if set(supplied) - set(required):
        raise ValueError("Numerical check is absent from the protocol")
    ordered = tuple(
        supplied.get(
            name,
            InferenceCheck(name, "unevaluated",
                           "Required check was not supplied"))
        for name in required)
    availability: Literal["available", "numerical_failure", "unevaluated"]
    if candidate.status != "complete" or \
            any(check.status == "fail" for check in ordered):
        availability = "numerical_failure"
    elif any(check.status == "unevaluated" for check in ordered):
        availability = "unevaluated"
    else:
        availability = "available"
    if candidate.status == "complete":
        _validate_samples(candidate)
    return AssessedInference(
        candidate.identity, protocol, ordered, predictive, candidate.status,
        availability, candidate if availability == "available" else None)


def validated_posterior(result: AssessedInference) -> Optional[BatchPosterior]:
    """Recheck a consumer's assessment, including manually built artifacts.

    Unavailable inference returns None while retaining its diagnostics.
    Contradictory availability, identities or assessment claims raise.
    This applies the declared protocol, not an implicit adequacy test.
    """
    if result.availability not in ("available", "numerical_failure",
                                   "unevaluated"):
        raise ValueError("Invalid posterior availability")
    if result.posterior is None:
        if result.availability == "available":
            raise ValueError("Available assessment needs a posterior")
        return None
    if result.availability != "available" or \
            result.identity != result.posterior.identity or \
            result.identity.prior != result.posterior.prior.digest:
        raise ValueError("Assessment and posterior identity disagree")
    checked = assess_inference(result.posterior, result.protocol,
                               result.numerical_checks,
                               result.predictive_checks)
    if checked.availability != "available" or \
            checked.sampler_status != result.sampler_status:
        raise ValueError("Assessment does not supply an available posterior")
    return result.posterior


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Assessment identities must be SHA256 digests")


def _validate_samples(candidate: BatchPosterior) -> None:
    """Reject malformed artifacts; this is not a statistical adequacy test."""
    if candidate.completed_temperature != 1 or not candidate.samples or \
            len(candidate.samples) != len(candidate.weights):
        raise ValueError("Malformed completed posterior")
    if any(
            len(row) != len(candidate.prior.names) or any(
                not math.isfinite(value) for value in row)
            for row in candidate.samples):
        raise ValueError("Invalid joint posterior samples")
    if any(not math.isfinite(weight) or weight < 0
           for weight in candidate.weights) or \
            not math.isclose(math.fsum(candidate.weights), 1.,
                             rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Posterior weights must be normalized")
