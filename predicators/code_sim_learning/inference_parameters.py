"""Parameter summaries and ensembles from one assessed joint posterior.

Projection preserves particle weights and parameter correlations. It
neither refits marginal widths nor publishes parameters or approves a
plan. Initial-state inference and execution-state estimation remain
separate from this parameter-only consumer view.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.active_experiment import \
    noisy_read_information
from predicators.code_sim_learning.inference_assessment import \
    AssessedInference, InferenceCheck, assess_inference
from predicators.code_sim_learning.inference_data import InferenceIdentity
from predicators.code_sim_learning.inference_sampling import BatchPosterior


class UnavailableParameterPosterior(ValueError):
    """Numerical assessment does not supply usable posterior parameters."""


@dataclass(frozen=True)
class ParameterEnsemble:
    """Weighted joint parameter rows with source-particle provenance.

    Resampled rows have equal weights and an explicit sampling seed;
    they add Monte Carlo error to the source approximation. Neither
    source weights nor equal resampling weights are stress-test weights.
    """

    identity: InferenceIdentity
    assessment_protocol: str
    predictive_checks: Tuple[InferenceCheck, ...]
    names: Tuple[str, ...]
    source_coordinates: Tuple[str, ...]
    values: Tuple[Tuple[float, ...], ...]
    weights: Tuple[float, ...]
    source_indices: Tuple[int, ...]
    resampling_seed: Optional[int] = None

    def __post_init__(self) -> None:
        names = tuple(self.names)
        coordinates = tuple(self.source_coordinates)
        values = tuple(
            tuple(float(value) for value in row) for row in self.values)
        weights = tuple(float(weight) for weight in self.weights)
        indices = tuple(self.source_indices)
        if len(set(names)) != len(names) or any(
                not isinstance(name, str) or not name for name in names):
            raise ValueError("Ensemble names must be distinct strings")
        if len(coordinates) != len(names) or any(
                not isinstance(name, str) or not name for name in coordinates):
            raise ValueError("One source coordinate per parameter required")
        if not values or len(weights) != len(values) or \
                len(indices) != len(values) or any(
                    len(row) != len(names) or any(not math.isfinite(value)
                                                 for value in row)
                    for row in values):
            raise ValueError("Invalid ensemble dimensions or values")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) \
                or not math.isclose(math.fsum(weights), 1., rel_tol=1e-12,
                                    abs_tol=1e-12):
            raise ValueError("Ensemble weights must be normalized")
        if any(not isinstance(index, int) or isinstance(index, bool)
               or index < 0 for index in indices):
            raise ValueError("Invalid source particle index")
        if self.resampling_seed is not None and (
                not isinstance(self.resampling_seed, int) or isinstance(
                    self.resampling_seed, bool) or self.resampling_seed < 0):
            raise ValueError("Invalid resampling seed")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "source_coordinates", coordinates)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "source_indices", indices)
        object.__setattr__(self, "predictive_checks",
                           tuple(self.predictive_checks))

    def as_dicts(self) -> Tuple[Dict[str, float], ...]:
        """Return owned parameter maps for simulator consumers."""
        return tuple(dict(zip(self.names, row)) for row in self.values)

    def expectation(self, outcomes: Sequence[float]) -> float:
        """Weight one finite outcome per row without changing decision rules.

        Boolean outcomes yield a represented success probability, not a
        guarantee, calibration statement or action approval.
        """
        values = tuple(float(value) for value in outcomes)
        if len(values) != len(self.values) or any(not math.isfinite(value)
                                                  for value in values):
            raise ValueError("One finite outcome per ensemble row required")
        return math.fsum(weight * value
                         for weight, value in zip(self.weights, values))

    def atom_information(self, read_probabilities: np.ndarray) -> float:
        """Score noisy atom reads using this ensemble's posterior weights.

        Supply one row per parameter map and one column per atom, with
        each entry the probability of reading that atom true under the
        declared observation channel. Binary entries describe exact
        reads. This uses the incumbent per-atom information criterion;
        it does not approve a probe or erase predictive failures.
        """
        return noisy_read_information(read_probabilities, weights=self.weights)


@dataclass(frozen=True)
class ParameterPosterior:
    """An explicit parameter projection of an assessed inference result.

    Unavailable results retain their assessment but expose no samples or
    quantiles. Predictive failures remain visible without suppressing an
    otherwise numerically adequate posterior. Parameter names are an
    explicit subset of joint coordinates, never inferred by position.
    """

    assessment: AssessedInference
    names: Tuple[str, ...]
    coordinates: Tuple[str, ...] = ()
    schema_version: Literal[1] = 1
    uncertainty_kind: Literal["joint_posterior"] = "joint_posterior"

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if len(set(names)) != len(names) or any(
                not isinstance(name, str) or not name for name in names):
            raise ValueError("Parameter names must be distinct strings")
        object.__setattr__(self, "names", names)
        coordinates = tuple(self.coordinates) if self.coordinates else names
        if len(coordinates) != len(names) or any(
                not isinstance(name, str) or not name for name in coordinates):
            raise ValueError("One joint coordinate per parameter required")
        object.__setattr__(self, "coordinates", coordinates)
        result = self.assessment
        if result.availability not in ("available", "numerical_failure",
                                       "unevaluated"):
            raise ValueError("Invalid posterior availability")
        if result.posterior is None:
            if result.availability == "available":
                raise ValueError("Available assessment needs a posterior")
            return
        if result.availability != "available" or \
                result.identity != result.posterior.identity or \
                result.identity.prior != result.posterior.prior.digest:
            raise ValueError("Assessment and posterior identity disagree")
        # Recheck structural validity and the declared protocol even when
        # callers construct AssessedInference directly instead of its helper.
        checked = assess_inference(result.posterior, result.protocol,
                                   result.numerical_checks,
                                   result.predictive_checks)
        if checked.availability != "available" or \
                checked.sampler_status != result.sampler_status:
            raise ValueError(
                "Assessment does not supply an available posterior")
        if not set(coordinates) <= set(result.posterior.prior.names):
            raise ValueError("Unknown joint posterior parameter")

    def _posterior(self) -> BatchPosterior:
        result = self.assessment
        if result.availability != "available" or result.posterior is None:
            raise UnavailableParameterPosterior("Parameter posterior is " +
                                                result.availability)
        return result.posterior

    def marginal_quantiles(
        self, probabilities: Tuple[float, ...] = (.05, .5, .95)
    ) -> Dict[str, Tuple[float, ...]]:
        """Use the same weighted empirical CDF as the joint approximation."""
        posterior = self._posterior()
        if not probabilities or any(not math.isfinite(p) or p < 0 or p > 1
                                    for p in probabilities):
            raise ValueError("Quantile probabilities must lie in [0, 1]")
        return {
            name: posterior.marginal_quantiles(coordinate, probabilities)
            for name, coordinate in zip(self.names, self.coordinates)
        }

    def weighted_samples(self) -> ParameterEnsemble:
        """Project positive-weight rows, retaining weights and source
        indices."""
        posterior = self._posterior()
        columns = tuple(
            posterior.prior.names.index(name) for name in self.coordinates)
        indices = tuple(i for i, weight in enumerate(posterior.weights)
                        if weight > 0)
        return ParameterEnsemble(
            posterior.identity, self.assessment.protocol.source,
            self.assessment.predictive_checks, self.names, self.coordinates,
            tuple(
                tuple(posterior.samples[i][column]
                      for column in columns) for i in indices),
            tuple(posterior.weights[i] for i in indices), indices)

    def resample(self, count: int, seed: int) -> ParameterEnsemble:
        """Resample complete parameter rows; never combine marginal draws."""
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Resampling count must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Resampling seed must be a nonnegative integer")
        weighted = self.weighted_samples()
        probabilities = np.asarray(weighted.weights, dtype=float)
        probabilities /= math.fsum(weighted.weights)
        indices = tuple(
            int(i) for i in np.random.default_rng(seed).choice(
                len(weighted.values), size=count, p=probabilities))
        return ParameterEnsemble(
            weighted.identity, weighted.assessment_protocol,
            weighted.predictive_checks, weighted.names,
            weighted.source_coordinates,
            tuple(weighted.values[index]
                  for index in indices), (1. / count, ) * count,
            tuple(weighted.source_indices[index] for index in indices), seed)
