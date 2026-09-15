"""Parameter consumers for a sampled marginal times an analytic prior factor.

Factorization is an explicit, checked model/data claim. Flat numerical
slices do not establish it. This module neither discovers independence
nor fits, publishes, or silently changes a posterior's prior.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, Tuple

import numpy as np

from predicators.code_sim_learning.inference_assessment import \
    AssessedInference, InferenceCheck, validated_posterior
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_parameters import \
    ParameterEnsemble, UnavailableParameterPosterior
from predicators.code_sim_learning.inference_sampling import BatchPosterior, \
    BoxPrior


@dataclass(frozen=True)
class IndependentPriorFactorization:
    """Declare p(full | data) = p(retained | data) times a fixed box prior.

    Full and reduced targets must share data, sensor and program. Their
    prior and runtime identities can differ because the reduced target
    integrates out coordinates. The declaration identifies the complete
    retained schema, including episode state, and an independent uniform
    factor in explicit physical coordinates. It is not a proof.
    """

    full_identity: InferenceIdentity
    reduced_identity: InferenceIdentity
    retained_coordinates: Tuple[str, ...]
    independent_prior: BoxPrior

    def __post_init__(self) -> None:
        coordinates = tuple(self.retained_coordinates)
        if not coordinates or len(set(coordinates)) != len(coordinates) or \
                any(not isinstance(name, str) or not name
                    for name in coordinates):
            raise ValueError("Retained coordinates must be distinct names")
        if set(coordinates).intersection(self.independent_prior.names):
            raise ValueError("Independent and retained coordinates overlap")
        for name in ("data", "sensor", "program"):
            if getattr(self.full_identity, name) != \
                    getattr(self.reduced_identity, name):
                raise ValueError(
                    "Factorization changes data, sensor or program")
        object.__setattr__(self, "retained_coordinates", coordinates)

    @property
    def digest(self) -> str:
        """Bind evidence to the identities, complete schema and factor law."""
        return content_digest(
            json.dumps({
                "schema": 1,
                "factorization": asdict(self)
            },
                       sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class CheckedFactorization:
    """An identified independence check for one exact declaration.

    The evidence must establish the factorization over the target's
    support, not just at selected fitted states. A new data prefix or
    model requires a new declaration and check. The caller is
    responsible for that evidence; a digest match alone does not prove
    independence.
    """

    declaration: IndependentPriorFactorization
    checked_declaration: str
    check: InferenceCheck

    def __post_init__(self) -> None:
        if self.checked_declaration != self.declaration.digest:
            raise ValueError("Factorization evidence has a different scope")


@dataclass(frozen=True)
class FactorizedParameterPosterior:
    """Project one product posterior into parameter intervals and draws.

    The reduced numerical assessment and factorization check must both
    permit use. Analytic factors never turn an unavailable sampled
    marginal into an available full posterior. Predictive diagnostics
    retain their full or reduced target scope and do not approve
    actions.
    """

    reduced: AssessedInference
    factorization: CheckedFactorization
    names: Tuple[str, ...]
    coordinates: Tuple[str, ...] = ()
    predictive_checks: Tuple[InferenceCheck, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.names)
        coordinates = tuple(self.coordinates) if self.coordinates else names
        if len(set(names)) != len(names) or any(
                not isinstance(name, str) or not name for name in names):
            raise ValueError("Parameter names must be distinct strings")
        if len(coordinates) != len(names) or any(
                not isinstance(name, str) or not name for name in coordinates):
            raise ValueError("One joint coordinate per parameter required")
        declaration = self.factorization.declaration
        available = set(declaration.retained_coordinates) | \
            set(declaration.independent_prior.names)
        if not set(coordinates) <= available:
            raise ValueError("Unknown full posterior coordinate")
        checks = tuple(self.predictive_checks)
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("Duplicate full posterior predictive check")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "predictive_checks", checks)
        self._validate_source()

    def _validate_source(self) -> BatchPosterior | None:
        checked = self.factorization
        declaration = checked.declaration
        if checked.checked_declaration != declaration.digest or \
                self.reduced.identity != declaration.reduced_identity:
            raise ValueError("Factorization and reduced source disagree")
        posterior = validated_posterior(self.reduced)
        if posterior is not None and \
                posterior.prior.names != declaration.retained_coordinates:
            raise ValueError("Reduced posterior coordinate schema differs")
        return posterior

    def _posterior(self) -> BatchPosterior:
        posterior = self._validate_source()
        if posterior is None:
            raise UnavailableParameterPosterior("Reduced posterior is " +
                                                self.reduced.availability)
        if self.factorization.check.status != "pass":
            raise UnavailableParameterPosterior(
                "Factorization is " + self.factorization.check.status)
        return posterior

    @property
    def identity(self) -> InferenceIdentity:
        """Identify the full target, separately from its sampled marginal."""
        return self.factorization.declaration.full_identity

    def marginal_quantiles(
        self, probabilities: Tuple[float, ...] = (.05, .5, .95)
    ) -> Dict[str, Tuple[float, ...]]:
        """Keep analytic quantiles exact and sampled quantiles weighted."""
        posterior = self._posterior()
        if not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1
                                    for p in probabilities):
            raise ValueError("Quantile probabilities must lie in [0, 1]")
        prior = self.factorization.declaration.independent_prior
        bounds = dict(zip(prior.names, prior.bounds))
        result = {}
        for name, coordinate in zip(self.names, self.coordinates):
            if coordinate in bounds:
                lo, hi = bounds[coordinate]
                result[name] = tuple(lo if p == 0 else hi if p == 1 else lo +
                                     p * (hi - lo) for p in probabilities)
            else:
                result[name] = posterior.marginal_quantiles(
                    coordinate, probabilities)
        return result

    def resample(self, count: int, seed: int) -> ParameterEnsemble:
        """Draw retained rows jointly and the certified factor independently.

        A repeated source coordinate, including an analytic coordinate,
        shares one value across its aliases. Output rows have equal
        Monte Carlo weights and retain their sampled marginal's original
        particle indices. This does not create new fitting evidence.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Resampling count must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Resampling seed must be a nonnegative integer")
        posterior = self._posterior()
        rng = np.random.default_rng(seed)
        weights = np.asarray(posterior.weights, dtype=float)
        weights /= math.fsum(posterior.weights)
        indices = tuple(
            int(i) for i in rng.choice(len(weights), size=count, p=weights))
        prior = self.factorization.declaration.independent_prior
        lower, upper = np.asarray(prior.bounds).T
        independent = rng.uniform(lower, upper, size=(count, len(prior.names)))
        rows = []
        for index, draw in zip(indices, independent):
            values = dict(zip(posterior.prior.names, posterior.samples[index]))
            values.update(zip(prior.names, draw))
            rows.append(tuple(
                float(values[name]) for name in self.coordinates))
        protocol = content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "reduced_protocol": self.reduced.protocol.source,
                    "factorization": self.factorization.checked_declaration,
                    "check": asdict(self.factorization.check)
                },
                sort_keys=True).encode("utf-8"))
        checks = tuple(
            replace(check, name="reduced/" + check.name)
            for check in self.reduced.predictive_checks) + tuple(
                replace(check, name="full/" + check.name)
                for check in self.predictive_checks)
        return ParameterEnsemble(self.identity, protocol,
                                 checks, self.names, self.coordinates,
                                 tuple(rows), (1. / count, ) * count, indices,
                                 seed)
