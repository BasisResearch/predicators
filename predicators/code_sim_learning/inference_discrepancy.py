"""An explicit stochastic velocity transition for offline model diagnostics.

This is a different dynamics model, not sensor-noise inflation or a
numerical tolerance. A predicted velocity is followed by a declared
mixture of rest and an isotropic Gaussian correction. Exact speed can
then be conditioned analytically while direction remains uncertain. It
is not installed in the acting agent or the deterministic reference.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

from predicators.code_sim_learning.inference_conditioning import \
    ConditionedVelocity, ConditioningNumericalError, UnsupportedConditioning
from predicators.code_sim_learning.inference_data import content_digest


@dataclass(frozen=True)
class VelocityDiscrepancy:
    """Post-transition rest mass plus Gaussian velocity around a prediction.

    Conditional on the moving case, v_next = v_predicted + sigma * N(0,I).
    The alternative rest case sets v_next to zero with its stated mass.
    sigma is in velocity units per modeled transition, not observation
    units or per-second diffusion units. Changing the time resolution
    requires a separately specified model. The caller must declare where
    the correction occurs relative to contacts, events and observations.

    Hyperparameters are model assumptions to evaluate or infer with an
    identified original prior. They must not be selected to manufacture
    support for each candidate or changed after inspecting a residual.
    """
    rest_probability: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rest_probability) or \
                not 0 <= self.rest_probability <= 1:
            raise ValueError("Rest probability must lie in [0, 1]")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Correction sigma must be finite and positive")

    @property
    def digest(self) -> str:
        """Identify the transition law, conditional measure and units."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "rest_or_gaussian_velocity_transition",
                    "parameters": asdict(self),
                    "speed_measure": "delta_zero_plus_positive_lebesgue",
                    "direction": "von_mises_fisher_unit_square",
                    "time_units": "per_declared_transition"
                },
                sort_keys=True).encode("utf-8"))

    def condition_on_speed(
        self,
        predicted: Tuple[float, float, float],
        speed: float,
        direction: Tuple[float, ...] = ()) -> ConditionedVelocity:
        """Condition a transition and retain its speed mass or radial density.

        Positive speed has a noncentral chi radial density (three
        dimensions). Its conditional direction is von Mises-Fisher,
        centered on the predicted direction with concentration
        speed * norm(predicted) / sigma**2. Two independent unit uniforms
        map to this normalized direction law, so no additional proposal
        weight is required. At zero speed the rest component is selected.

        A retained direction affects subsequent positions and contacts;
        it cannot be replaced by its mean without changing the model.
        The reported speed residual is floating-point reconstruction
        error, not an observation acceptance threshold.
        """
        if len(predicted) != 3 or any(not math.isfinite(v) for v in predicted):
            raise ValueError("Predicted velocity needs three finite values")
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Speed must be finite and nonnegative")
        if speed == 0:
            if direction:
                raise ValueError("Rest has no direction coordinates")
            if self.rest_probability == 0:
                raise UnsupportedConditioning(
                    "Zero speed without a rest atom needs another "
                    "representation")
            return ConditionedVelocity((0., 0., 0.), 0,
                                       math.log(self.rest_probability), 0.)
        if len(direction) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1
                                      for v in direction):
            raise ValueError(
                "Moving speed requires two unit-square coordinates")
        magnitude = math.hypot(*predicted)
        if not math.isfinite(magnitude):
            raise ConditioningNumericalError("Predicted speed overflow")
        concentration, log_radial = _radial_law(speed, magnitude, self.sigma)
        cosine = _direction_cosine(direction[0], concentration)
        azimuth = 2 * math.pi * direction[1]
        radius = math.sqrt(max(0., (1 - cosine) * (1 + cosine)))
        if magnitude == 0:
            unit = np.array([
                radius * math.cos(azimuth), radius * math.sin(azimuth), cosine
            ])
        else:
            axis = np.asarray(predicted) / magnitude
            reference = np.zeros(3)
            reference[int(np.argmin(np.abs(axis)))] = 1.
            tangent = np.cross(axis, reference)
            tangent /= math.hypot(*tangent)
            bitangent = np.cross(axis, tangent)
            unit = (
                cosine * axis + radius *
                (math.cos(azimuth) * tangent + math.sin(azimuth) * bitangent))
        velocity = (float(speed * unit[0]), float(speed * unit[1]),
                    float(speed * unit[2]))
        if any(not math.isfinite(v) for v in velocity):
            raise ConditioningNumericalError("Conditional velocity overflow")
        log_factor = (-math.inf if self.rest_probability == 1 else
                      math.log1p(-self.rest_probability) + log_radial)
        return ConditionedVelocity(velocity, 2, log_factor,
                                   abs(math.hypot(*velocity) - speed))

    def sample(self, predicted: Tuple[float, float, float],
               rng: np.random.Generator) -> Tuple[float, float, float]:
        """Draw the original transition law without observing future speed.

        The rest branch sets velocity to zero; the moving branch adds
        isotropic Gaussian noise around the native prediction. The
        caller supplies a local generator and applies the draw at the
        declared physical transition boundary, retaining any angular
        velocity. This does not score exact speed or condition on a
        future reading.
        """
        if len(predicted) != 3 or any(not math.isfinite(v) for v in predicted):
            raise ValueError("Predicted velocity needs three finite values")
        if rng.random() < self.rest_probability:
            return (0., 0., 0.)
        values = rng.normal(predicted, self.sigma)
        if any(not math.isfinite(v) for v in values):
            raise ConditioningNumericalError("Sampled velocity overflow")
        return (float(values[0]), float(values[1]), float(values[2]))


def _radial_law(speed: float, mean: float,
                sigma: float) -> Tuple[float, float]:
    """Stable noncentral chi density without exponentiating a large square."""
    scaled_speed, scaled_mean = speed / sigma, mean / sigma
    concentration = scaled_speed * scaled_mean
    if not math.isfinite(concentration):
        raise ConditioningNumericalError("Directional concentration overflow")
    if concentration < 1e-4:
        # log(sinh(k)/k) = k**2/6 - k**4/180 + O(k**6).
        # The omitted term is below 4e-28 at this branch boundary.
        log_radial = (
            .5 * math.log(2 / math.pi) + 2 * math.log(speed) -
            3 * math.log(sigma) - .5 *
            (scaled_speed * scaled_speed + scaled_mean * scaled_mean) +
            concentration**2 / 6 - concentration**4 / 180)
    else:
        difference = (speed - mean) / sigma
        log_radial = (math.log(speed) - math.log(mean) - math.log(sigma) -
                      .5 * math.log(2 * math.pi) -
                      .5 * difference * difference +
                      math.log(-math.expm1(-2 * concentration)))
    if not math.isfinite(log_radial):
        raise ConditioningNumericalError("Radial log density overflow")
    return concentration, log_radial


def _direction_cosine(uniform: float, concentration: float) -> float:
    """Invert the normalized axial CDF, retaining antipodal endpoints."""
    if concentration == 0:
        return 2 * uniform - 1
    if uniform == 0:
        return -1.
    if uniform == 1:
        return 1.
    if concentration < .5:
        return math.log1p(uniform * math.expm1(2 * concentration)) / \
            concentration - 1
    log_mixture = float(
        np.logaddexp(math.log(uniform),
                     math.log1p(-uniform) - 2 * concentration))
    return 1 + log_mixture / concentration
