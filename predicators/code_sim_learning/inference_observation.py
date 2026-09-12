"""Compose an identified full observation likelihood for offline inference.

Each measurement belongs to one declared factor or the original sensor
likelihood. Checked readouts retain their source factor. Unsupported
partial coupled readings are errors, and unmodeled exact contradictions
remain zero likelihood. No production fitting or execution path changes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import pybullet as p

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError, UnsupportedConditioning
from predicators.code_sim_learning.inference_data import FeatureKey, \
    Observation, SensorModel, content_digest
from predicators.code_sim_learning.inference_orientation import \
    QuaternionOutputError
from predicators.code_sim_learning.inference_output_error import \
    GaussianOutputError, output_error_likelihood
from predicators.code_sim_learning.inference_readout import ExactReadout, \
    reduce_exact_readout


@dataclass(frozen=True)
class ScalarOutputFactor:
    """An explicitly selected real-valued discrepancy channel."""
    key: FeatureKey
    process: GaussianOutputError


@dataclass(frozen=True)
class EulerOutputFactor:
    """An exact (roll, pitch, yaw) readout with coupled output discrepancy.

    The Gaussian center is the representative quaternion constructed
    from the native predicted Euler triple, not hidden recorded state.
    The declared mixture retains both quaternion signs. This choice is
    part of the model and loses any information absent from that triple.
    The inference runtime identity must capture the PyBullet conversion.
    """
    keys: Tuple[FeatureKey, FeatureKey, FeatureKey]
    process: QuaternionOutputError


@dataclass(frozen=True)
class CheckedReadoutFactor:
    """A reviewed parameter-independent readout and its identified callback."""
    declaration: ExactReadout
    evaluate: Callable[[float], float]


@dataclass(frozen=True)
class OutputObservationModel:
    """A complete measurement partition, conditional on native predictions.

    Discrepancy processes are independent across declared factors,
    conditional on the physical candidate and their hyperparameters.
    Within a scalar factor the full temporal error history is integrated.
    Every unassigned measurement uses the unchanged sensor model.
    In particular, event errors do not disappear when continuous outputs
    obtain an explicit discrepancy law.

    This model can be called from a joint parameter/initial-state target,
    but does not itself provide a physical prior, simulator replay, or
    evidence that the numerical posterior has been explored adequately.
    """
    sensor: SensorModel
    scalars: Tuple[ScalarOutputFactor, ...] = ()
    eulers: Tuple[EulerOutputFactor, ...] = ()
    readouts: Tuple[CheckedReadoutFactor, ...] = ()

    def __post_init__(self) -> None:
        schema = {f.key: f for f in self.sensor.features}
        assigned = [f.key for f in self.scalars]
        for factor in self.eulers:
            if len(factor.keys) != 3:
                raise ValueError("A coupled Euler factor requires three keys")
            assigned.extend(factor.keys)
            if any(k not in schema or schema[k].sigma != 0
                   for k in factor.keys):
                raise UnsupportedConditioning(
                    "Euler discrepancy currently requires exact readings")
        outputs = [r.declaration.output for r in self.readouts]
        assigned.extend(outputs)
        if len(set(assigned)) != len(assigned):
            raise ValueError("A measurement cannot have multiple factors")
        if any(k not in schema or schema[k].conditioned for k in assigned):
            raise ValueError("Factors require predicted sensor fields")
        for readout in self.readouts:
            declaration = readout.declaration
            if declaration.source in outputs:
                raise UnsupportedConditioning(
                    "Chained readouts require an explicit joint reduction")
            # Validate the declared map/schema even in an empty episode.
            reduce_exact_readout(Observation(0, ()), self.sensor, declaration,
                                 readout.evaluate)

    @property
    def digest(self) -> str:
        """Identify all factors and the representative-quaternion
        convention."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "sensor": self.sensor.digest,
                    "scalars": [(f.key, f.process.digest)
                                for f in self.scalars],
                    "eulers": [(f.keys, f.process.digest)
                               for f in self.eulers],
                    "readouts": [r.declaration.digest for r in self.readouts],
                    "orientation_center":
                    "pybullet_quaternion_from_predicted_euler",
                    "cross_factor_dependence": "conditionally_independent",
                    "unassigned_fields": "original_sensor_likelihood"
                },
                sort_keys=True).encode("utf-8"))

    def log_likelihood(self, predictions: Tuple[Observation, ...],
                       observations: Tuple[Observation, ...]) -> float:
        """Score a complete reset episode, including its initial observation.

        Missing readings use empty/partial Observation values, with
        every primitive step represented. A repeated full-data fit
        starts the original discrepancy laws again. The caller must
        account for an observation-informed initial-state proposal
        separately; this routine never removes the first frame to hide
        double counting.
        """
        return self._log_likelihood_from(predictions, observations, 0)

    def log_future_likelihood(
            self, predictions: Tuple[Observation, ...],
            observed_prefix: Tuple[Observation,
                                   ...], observed_future: Tuple[Observation,
                                                                ...]) -> float:
        """Score a future history conditional on its supported observed prefix.

        Predictions cover the entire history and must be generated
        without future readings. Future observations enter only this
        evaluation, never the physical replay or prefix fit. Temporal
        error factors condition on earlier readings through the chain
        rule, retaining the joint future density. Summing future factors
        directly avoids subtracting two large full-history log scores.

        An impossible prefix has no conditional distribution and raises;
        an impossible future has zero density and returns negative
        infinity. An empty future has log density zero, provided that
        the prefix is supported. No parameter or particle weight changes.
        """
        if observed_prefix:
            prefix_score = self.log_likelihood(
                predictions[:len(observed_prefix)], observed_prefix)
            if prefix_score == -math.inf:
                raise UnsupportedConditioning(
                    "Cannot score a future from a zero-likelihood prefix")
            if not math.isfinite(prefix_score):
                raise ConditioningNumericalError("Nonfinite prefix likelihood")
        return self._log_likelihood_from(predictions,
                                         observed_prefix + observed_future,
                                         len(observed_prefix))

    def _log_likelihood_from(self, predictions: Tuple[Observation, ...],
                             observations: Tuple[Observation, ...],
                             first_step: int) -> float:
        """Condition on a prefix, accumulating only factors at or after
        start."""
        if not predictions or len(predictions) != len(observations):
            raise ValueError("Matching nonempty histories required")
        expected = list(range(len(predictions)))
        if [o.step for o in observations] != expected or \
                [o.step for o in predictions] != expected:
            raise ValueError(
                "Histories must contain each step starting at zero")
        declared = {feature.key for feature in self.sensor.features}
        if any(key not in declared for observation in observations
               for key, _ in observation.values):
            raise ValueError("Unknown observed feature")
        observed_values = []
        for observation in observations:
            for readout in self.readouts:
                reduced = reduce_exact_readout(observation, self.sensor,
                                               readout.declaration,
                                               readout.evaluate)
                if reduced.observation is None:
                    return -math.inf
                observation = reduced.observation
            observed_values.append(dict(observation.values))
        predicted_values = [dict(o.values) for o in predictions]
        sensor = {f.key: f for f in self.sensor.features}
        claimed = {f.key for f in self.scalars}
        claimed.update(k for f in self.eulers for k in f.keys)
        terms = []
        # All other readings retain their original likelihood, including
        # exact events, and unknown measurement keys still cause an error.
        for step in range(first_step, len(observed_values)):
            values = observed_values[step]
            residual = Observation(
                step,
                tuple((k, v) for k, v in values.items() if k not in claimed))
            terms.append(
                self.sensor.log_likelihood(residual, predicted_values[step]))
        for scalar in self.scalars:
            if any(scalar.key not in values for values in predicted_values):
                raise ValueError("Missing scalar prediction")
            result = output_error_likelihood(
                scalar.process,
                tuple(values[scalar.key] for values in predicted_values),
                tuple(values.get(scalar.key) for values in observed_values),
                sensor[scalar.key].sigma)
            if first_step == 0:
                terms.append(result.log_likelihood)
            elif result.status == "exact_contradiction":
                return -math.inf
            else:
                terms.append(
                    math.fsum(step.log_observation_factor
                              for step in result.steps[first_step:]
                              if step.log_observation_factor is not None))
        for euler in self.eulers:
            for observed, predicted in zip(observed_values[first_step:],
                                           predicted_values[first_step:]):
                present = sum(k in observed for k in euler.keys)
                if not present:
                    continue
                if present != 3:
                    raise UnsupportedConditioning(
                        "Partial Euler readings need a marginal likelihood")
                if any(k not in predicted for k in euler.keys):
                    raise ValueError("Missing coupled orientation prediction")
                angles = [predicted[k] for k in euler.keys]
                mean = p.getQuaternionFromEuler(angles)
                roll, pitch, yaw = euler.keys
                reading = (observed[roll], observed[pitch], observed[yaw])
                terms.append(euler.process.log_density(mean, reading))
        return math.fsum(terms)

    def sample_future(self, predictions: Tuple[Observation, ...],
                      observed_prefix: Tuple[Observation, ...],
                      rng: np.random.Generator) -> Tuple[Observation, ...]:
        """Draw a joint future observation history using only a fitted prefix.

        Predictions include the initial frame and every future primitive
        step. They must come from causal simulator continuation, without
        using future readings to correct physical state. This method
        receives no future observations and never changes predictions.

        Scalar discrepancy is filtered on the prefix, then sampled as a
        correlated error history. Sensor noise remains independent and
        separate. Euler errors use the declared raw quaternion mixture,
        and checked displays are derived from their sampled sources.
        Conditioned inputs must be supplied in the future prediction
        frames; they are copied as given inputs, not assigned a density.

        An empty prefix draws from the original output-error law. A
        zero-likelihood prefix has no conditional forecast under this
        supplied physical history. This is an output-model sampler, not
        a posterior over physical parameters or current execution state.
        """
        count = len(observed_prefix)
        if not predictions or count > len(predictions) or \
                [o.step for o in predictions] != list(range(len(predictions))):
            raise ValueError("Predictions must contain each step from zero")
        if count:
            score = self.log_likelihood(predictions[:count], observed_prefix)
            if score == -math.inf:
                raise UnsupportedConditioning(
                    "No conditional forecast for a zero-likelihood prefix")
            if not math.isfinite(score):
                raise ConditioningNumericalError("Nonfinite prefix score")
        if any(f.process.pole_threshold != .99999 for f in self.eulers):
            raise UnsupportedConditioning(
                "Native Euler sampling requires pole threshold .99999")
        if count == len(predictions):
            return ()
        predicted = [dict(o.values) for o in predictions]
        observed = [dict(o.values) for o in observed_prefix]
        sensor = {f.key: f for f in self.sensor.features}
        displays = {r.declaration.output for r in self.readouts}
        for values in predicted[count:]:
            if any(key not in values for key in sensor if key not in displays):
                raise ValueError(
                    "Missing future prediction or conditioned input")
        # One boundary draw per scalar retains temporal dependence within
        # the suffix, unlike drawing each filtered marginal independently.
        errors = {}
        for scalar in self.scalars:
            process = scalar.process
            mean, sigma = 0., process.initial_sigma
            if count:
                filtered = output_error_likelihood(
                    process,
                    tuple(row[scalar.key] for row in predicted[:count]),
                    tuple(row.get(scalar.key) for row in observed),
                    sensor[scalar.key].sigma)
                mean = filtered.steps[-1].filtered_mean
                sigma = filtered.steps[-1].filtered_sigma
            errors[scalar.key] = float(rng.normal(mean, sigma))
        draws = []
        for index in range(count, len(predictions)):
            values = {
                key: predicted[index][key]
                for key in sensor if key not in displays
            }
            for scalar in self.scalars:
                process = scalar.process
                if index:
                    errors[scalar.key] = float(
                        process.persistence * errors[scalar.key] +
                        rng.normal(0., process.innovation_sigma))
                values[scalar.key] += errors[scalar.key]
            for euler in self.eulers:
                angles = [predicted[index][key] for key in euler.keys]
                mean_quaternion = np.asarray(p.getQuaternionFromEuler(angles))
                sign = 1. if rng.integers(2) else -1.
                raw = sign * mean_quaternion + rng.normal(
                    0., euler.process.sigma, size=4)
                # Do not normalize: the likelihood models native Euler
                # readout of raw quaternion components, including poles.
                values.update(zip(euler.keys, p.getEulerFromQuaternion(raw)))
            for key, feature in sensor.items():
                if feature.sigma > 0 and not feature.conditioned:
                    values[key] += float(rng.normal(0., feature.sigma))
            for readout in self.readouts:
                declaration = readout.declaration
                values[declaration.output] = readout.evaluate(
                    values[declaration.source])
            draws.append(Observation(index, tuple(values.items())))
        return tuple(draws)
