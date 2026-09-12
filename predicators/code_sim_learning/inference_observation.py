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

import pybullet as p

from predicators.code_sim_learning.inference_conditioning import \
    UnsupportedConditioning
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
        for step, values in enumerate(observed_values):
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
            terms.append(result.log_likelihood)
        for euler in self.eulers:
            for observed, predicted in zip(observed_values, predicted_values):
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
