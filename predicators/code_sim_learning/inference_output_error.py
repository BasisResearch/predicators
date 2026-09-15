"""Analytically marginalized scalar output discrepancy for offline inference.

This is a declared statistical discrepancy model, not a physical state
correction or extra sensor noise. It is suitable only for explicitly
chosen real-valued outputs. Events, bounded quantities and coupled
kinematic constraints need their own observation models. No production
fitter or execution estimator uses this module.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError, condition_gaussian_coordinate
from predicators.code_sim_learning.inference_data import content_digest


@dataclass(frozen=True)
class GaussianOutputError:
    """A zero-mean initial error followed by scalar AR(1) transitions.

    b_0 ~ Normal(0, initial_sigma**2)
    b_t = persistence * b_(t-1) + innovation_sigma * Normal(0, 1)
    o_t = simulator_output_t + b_t + declared_sensor_noise_t

    Parameters describe one primitive time step. Zero scales represent
    deterministic quantities, not tiny Gaussian approximations. The
    caller fixes these parameters or gives them an identified original
    prior when fitting; a previous error estimate is not a new prior.
    """
    persistence: float
    innovation_sigma: float
    initial_sigma: float = 0.

    def __post_init__(self) -> None:
        if not math.isfinite(self.persistence) or \
                not -1 <= self.persistence <= 1:
            raise ValueError("Persistence must lie in [-1, 1]")
        if any(not math.isfinite(value) or value < 0
               for value in (self.innovation_sigma, self.initial_sigma)):
            raise ValueError("Error scales must be finite and nonnegative")

    @property
    def digest(self) -> str:
        """Identify the normalized process and primitive-step convention."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "scalar_gaussian_ar1_output_discrepancy",
                    "parameters":
                    {key: float(value)
                     for key, value in asdict(self).items()},
                    "step": "one_primitive_action",
                    "sensor": "separate_declared_additive_gaussian"
                },
                sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class ErrorFilterStep:
    """Causal error moments before and after this step's optional reading.

    These moments describe discrepancy, not a reconstructed physical
    state or a smoothed history. Before-reading moments supply an honest
    predictive distribution for that reading.
    """
    predicted_mean: float
    predicted_sigma: float
    filtered_mean: float
    filtered_sigma: float
    log_observation_factor: Optional[float]


@dataclass(frozen=True)
class OutputErrorLikelihood:
    """Marginal likelihood with the entire stochastic error history integrated.

    The returned steps are filtering marginals. They are not independent
    samples of the joint error history. An exact contradiction stops the
    calculation before inventing a conditional distribution at that
    step. A contradiction concerns this supplied prediction history, not
    every parameter or initial state of the simulator program.
    """
    process: str
    status: Literal["complete", "exact_contradiction"]
    log_likelihood: float
    steps: Tuple[ErrorFilterStep, ...]
    failed_step: Optional[int]


def output_error_likelihood(process: GaussianOutputError,
                            predictions: Tuple[float, ...],
                            observations: Tuple[Optional[float], ...],
                            sensor_sigma: float) -> OutputErrorLikelihood:
    """Score one contiguous reset episode or forecast its unobserved suffix.

    Index zero is the initial frame. Every later index is one primitive
    transition; missing observations are None and still advance the error
    process. A new call starts from the same original error law. Carrying
    a fitted error mean into a repeated full-data fit would double-count
    evidence and is intentionally not an argument to this function.

    With positive innovation variance, exactly observed continuous
    outputs condition a Gaussian latent error and retain its density.
    Zero sensor sigma remains exact: it eliminates error uncertainty at
    that frame. The simulator prediction is never changed or scored as
    though its discrepancy were measurement noise.
    """
    if not predictions or len(predictions) != len(observations):
        raise ValueError(
            "Matching nonempty prediction and observation histories required")
    if any(not math.isfinite(value) for value in predictions) or \
            any(value is not None and not math.isfinite(value)
                for value in observations):
        raise ValueError(
            "History values must be finite; missing readings are None")
    if not math.isfinite(sensor_sigma) or sensor_sigma < 0:
        raise ValueError("Sensor sigma must be finite and nonnegative")
    mean, sigma = 0., process.initial_sigma
    steps: List[ErrorFilterStep] = []
    factors: List[float] = []
    for index, (prediction,
                observed) in enumerate(zip(predictions, observations)):
        if index:
            mean *= process.persistence
            sigma = math.hypot(process.persistence * sigma,
                               process.innovation_sigma)
        if not math.isfinite(mean) or not math.isfinite(sigma):
            raise ConditioningNumericalError(
                "Output-error prediction overflow")
        before_mean, before_sigma = mean, sigma
        factor = None
        if observed is not None:
            residual = observed - prediction
            if not math.isfinite(residual):
                raise ConditioningNumericalError("Output residual overflow")
            if sigma > 0:
                conditional = condition_gaussian_coordinate(
                    mean, sigma, residual, sensor_sigma)
                mean, sigma = conditional.mean, conditional.sigma
                factor = conditional.log_observation_factor
            elif sensor_sigma > 0:
                scaled = (residual - mean) / sensor_sigma
                factor = (-.5 * scaled * scaled - math.log(sensor_sigma) -
                          .5 * math.log(2 * math.pi))
                if not math.isfinite(factor):
                    raise ConditioningNumericalError(
                        "Observation log density overflow")
            elif residual != mean:
                return OutputErrorLikelihood(process.digest,
                                             "exact_contradiction", -math.inf,
                                             tuple(steps), index)
            else:
                factor = 0.
            factors.append(factor)
        steps.append(
            ErrorFilterStep(before_mean, before_sigma, mean, sigma, factor))
    total = math.fsum(factors)
    if not math.isfinite(total):
        raise ConditioningNumericalError("Marginal log likelihood overflow")
    return OutputErrorLikelihood(process.digest, "complete", total,
                                 tuple(steps), None)
