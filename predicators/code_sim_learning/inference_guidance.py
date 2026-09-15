"""Defensive direction proposals for exact-speed conditional integration.

Guidance changes the sampling law only. The returned factor includes the
original radial density and the original-direction/mixture-proposal
ratio. It is not a replacement velocity law or a forecast generator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_discrepancy import \
    VelocityDiscrepancy


@dataclass(frozen=True)
class GuidedVelocity:
    """An importance draw, including the complete proposal correction."""
    velocity: Tuple[float, float, float]
    log_factor: float
    log_proposal_correction: float
    component: Literal["original", "guided", "rest"]
    speed_residual: float


def condition_guided_velocity(law: VelocityDiscrepancy,
                              predicted: Tuple[float, float, float],
                              speed: float, proposal_mean: Tuple[float, float,
                                                                 float],
                              guided_probability: float,
                              unit: Tuple[float, ...]) -> GuidedVelocity:
    """Draw from a mixture of original and guided conditional directions.

    At positive speed, three independent unit uniforms select a mixture
    component and its two direction coordinates. Both components are
    normalized conditional Gaussian direction laws at the same speed.
    The original component has strictly positive mixture probability,
    bounding the direction importance ratio by 1/(1-guided_probability).
    The mixture density, not the selected component density, determines
    the correction. A zero speed uses the original rest mass and no
    proposal coordinates.

    proposal_mean may depend on the scored observations and parent
    history. The caller must keep it fixed while drawing this extension,
    preserve its reconstruction, and retain this full factor exactly
    once. It must never use these observation-guided draws as generated
    forecasts or reinterpret the factors as posterior probabilities.
    """
    if any(
            len(mean) != 3 or any(not math.isfinite(v) for v in mean)
            for mean in (predicted, proposal_mean)):
        raise ValueError(
            "Original and proposal means need three finite values")
    if not math.isfinite(guided_probability) or \
            not 0 <= guided_probability < 1:
        raise ValueError("Guided probability must lie in [0, 1)")
    if speed == 0:
        if unit:
            raise ValueError("Rest has no proposal coordinates")
        rest = law.condition_on_speed(predicted, speed)
        return GuidedVelocity(rest.velocity, rest.log_observation_factor, 0.,
                              "rest", rest.speed_residual)
    if len(unit) != 3 or any(not math.isfinite(v) or not 0 <= v <= 1
                             for v in unit):
        raise ValueError("Positive speed requires three unit uniforms")
    if guided_probability == 0 or predicted == proposal_mean:
        original = law.condition_on_speed(predicted, speed, unit[1:])
        return GuidedVelocity(original.velocity,
                              original.log_observation_factor, 0., "original",
                              original.speed_residual)
    # Moving-only radial laws avoid subtracting two -inf masses when the
    # physical model assigns probability one to rest.
    moving = VelocityDiscrepancy(0., law.sigma)
    original = moving.condition_on_speed(predicted, speed, unit[1:])
    guided = moving.condition_on_speed(proposal_mean, speed, unit[1:])
    selected = guided if unit[0] < guided_probability else original
    scaled_shift = tuple(
        (b - a) / law.sigma for a, b in zip(predicted, proposal_mean))
    scaled_offset = tuple(
        (v - (.5 * a + .5 * b)) / law.sigma
        for v, a, b in zip(selected.velocity, predicted, proposal_mean))
    terms = tuple(d * offset for d, offset in zip(scaled_shift, scaled_offset))
    if any(not math.isfinite(v) for v in terms):
        raise ConditioningNumericalError("Guidance density ratio overflow")
    try:
        # q(direction)/p(direction) = N_q(v)/N_p(v) * radial_p/radial_q.
        relative = math.fsum(terms + (original.log_observation_factor,
                                      -guided.log_observation_factor))
    except OverflowError as exc:
        raise ConditioningNumericalError("Guidance density ratio overflow") \
            from exc
    if not math.isfinite(relative):
        raise ConditioningNumericalError("Guidance density ratio overflow")
    correction = -float(
        np.logaddexp(math.log1p(-guided_probability),
                     math.log(guided_probability) + relative))
    radial = (-math.inf if law.rest_probability == 1 else
              math.log1p(-law.rest_probability) +
              original.log_observation_factor)
    factor = radial + correction
    if math.isnan(factor) or factor == math.inf or \
            (math.isfinite(radial) and not math.isfinite(factor)):
        raise ConditioningNumericalError("Guided factor overflow")
    return GuidedVelocity(selected.velocity, factor, correction,
                          "guided" if selected is guided else "original",
                          selected.speed_residual)
