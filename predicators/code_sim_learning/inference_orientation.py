"""Marginalize explicit quaternion-output error through the native Euler map.

The latent readout input is an antipodal mixture of four-dimensional
Gaussians, not a normalized rotation or a physical state correction. The
Euler map has a continuous ordinary branch and two collapsed pole
branches. Their densities use different dimensions of one mixed measure.
Only exact complete Euler readings are supported here. No production
estimator uses this separately declared discrepancy model.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
# SciPy exports log_ndtr through a compiled ufunc.
# pylint: disable=no-name-in-module
from scipy.special import log_ndtr, logsumexp

# pylint: enable=no-name-in-module
from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_data import content_digest

Quaternion = Tuple[float, float, float, float]
Euler = Tuple[float, float, float]


@dataclass(frozen=True)
class QuaternionOutputError:
    """Independent latent readout inputs, with a fixed original scale law.

    Q ~ .5 Normal(mu, sigma**2 I_4) + .5 Normal(-mu, sigma**2 I_4).
    Observation = native_euler(Q), with no added sensor noise.

    Q is deliberately unnormalized: the native readout uses its raw
    products for the pitch test and asin. Normalizing Q changes this
    probability model. Antipodal mixing accounts for an unmodeled
    quaternion sign without dropping the observed native yaw branch.
    This is not isotropic angular noise on SO(3), a transition model,
    or an exact model of float32 quantization.
    """
    sigma: float
    pole_threshold: float = .99999

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Quaternion discrepancy scale must be positive")
        if not math.isfinite(self.pole_threshold) or \
                not 0 < self.pole_threshold < 1:
            raise ValueError("Pole threshold must lie strictly in (0, 1)")

    @property
    def digest(self) -> str:
        """Identify the mixture, readout convention and observation measure."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "raw_quaternion_gaussian_euler_output_error",
                    "parameters": asdict(self),
                    "antipodal_weights": [.5, .5],
                    "time": "independent_given_native_predictions",
                    "measure": "ordinary_dr_dp_dy_plus_two_pole_dy_branches"
                },
                sort_keys=True).encode("utf-8"))

    def log_density(self,
                    mean: Quaternion,
                    observed: Euler,
                    relative_tolerance: float = 1e-7) -> float:
        """Retain the density induced by an exact three-field reading.

        Ordinary outputs integrate over unknown quaternion radius and
        both lifts of a rotation. A pole output integrates its unknown
        x/y radius and analytically integrates the z/w half-space.
        Native pole yaw spans [-2*pi, 2*pi]; it is never wrapped here.
        Invalid support gives zero density, while numerical failure is
        explicit and must not be interpreted as a model contradiction.
        """
        if len(mean) != 4 or len(observed) != 3 or any(
                not math.isfinite(x) for x in (*mean, *observed)):
            raise ValueError("Finite quaternion and complete Euler required")
        if not math.isfinite(relative_tolerance) or \
                not 1e-11 <= relative_tolerance <= 1e-3:
            raise ValueError("Quadrature relative tolerance out of range")
        roll, pitch, yaw = observed
        if pitch in (-math.pi / 2, math.pi / 2):
            if roll != 0 or not -2 * math.pi <= yaw <= 2 * math.pi:
                return -math.inf
            return self._pole_density(mean, pitch, yaw, relative_tolerance)
        if not -math.pi / 2 < pitch < math.pi / 2 or \
                abs(math.sin(pitch)) >= self.pole_threshold or \
                not -math.pi <= roll <= math.pi or \
                not -math.pi <= yaw <= math.pi:
            return -math.inf
        return self._ordinary_density(mean, observed, relative_tolerance)

    def _ordinary_density(self, mean: Quaternion, observed: Euler,
                          tolerance: float) -> float:
        roll, pitch, yaw = observed
        sine = math.sin(pitch)
        center = math.hypot(*mean)
        constant = -2 * math.log(2 * math.pi) - 4 * math.log(self.sigma)
        log_jacobian = math.log(math.cos(pitch) / 8)

        def integrand(transverse: float) -> float:
            if transverse <= 0:
                return -math.inf
            # r^2=abs(sin(pitch))+u^2 makes r dr=u du. Retaining u
            # avoids subtracting nearly equal rounded squared radii.
            radius = math.hypot(math.sqrt(abs(sine)), transverse)
            cosine_part = transverse * math.sqrt(transverse * transverse +
                                                 2 * abs(sine))
            beta = math.atan2(sine, cosine_part)
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(beta / 2), math.sin(beta / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            unit = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
            exponents = [
                -.5 * math.fsum(((radius * q - sign * m) / self.sigma)**2
                                for q, m in zip(unit, mean))
                for sign in (-1, 1)
            ]
            # Sum the two lifts of the antipodal mixture: no extra .5.
            return (math.log(transverse) + log_jacobian + constant +
                    float(logsumexp(exponents)))

        tail_constant = log_jacobian + constant + math.log(2)
        return _log_radial_integral(integrand, center, self.sigma,
                                    tail_constant, tolerance, abs(sine))

    def _pole_density(self, mean: Quaternion, pitch: float, yaw: float,
                      tolerance: float) -> float:
        theta = yaw / 2
        sine, cosine = math.sin(theta), math.cos(theta)
        sign = 1 if pitch > 0 else -1
        direction = (-sign * sine, sign * cosine)
        center = math.hypot(mean[0], mean[1])
        projection = sine * mean[2] + cosine * mean[3]
        constant = -math.log(2 * math.pi) - 2 * math.log(self.sigma)

        def integrand(radius: float) -> float:
            if radius <= 0:
                return -math.inf
            exponents = [
                -.5 * math.fsum(
                    ((radius * q - lift * m) / self.sigma)**2
                    for q, m in zip(direction, mean[:2])) + float(
                        log_ndtr((lift * projection - self.pole_threshold /
                                  (2 * radius)) / self.sigma))
                for lift in (-1, 1)
            ]
            # Half for theta=yaw/2, and half for the antipodal mixture.
            return (math.log(radius) + constant - math.log(4) +
                    float(logsumexp(exponents)))

        return _log_radial_integral(integrand, center, self.sigma,
                                    constant - math.log(2), tolerance)


def _log_radial_integral(evaluate: Callable[[float], float],
                         center: float,
                         sigma: float,
                         tail_constant: float,
                         tolerance: float,
                         squared_radius_offset: float = 0.) -> float:
    """Scale adaptive quadrature and bound the omitted Gaussian radial tail.

    The bound is constant * integral r exp(-(r-center)^2/(2 sigma^2)) dr.
    Quadrature error estimates are numerical diagnostics, not proofs
    against undiscovered modes. Independent reference checks remain
    necessary before accepting an inference configuration.
    """
    radial_lower = math.sqrt(squared_radius_offset)
    radial_upper = max(center, radial_lower) + 8 * sigma
    upper = math.sqrt(
        (radial_upper - radial_lower) * (radial_upper + radial_lower))
    for _ in range(12):
        knots = np.linspace(0., upper, 9).tolist()
        candidates = [x for x in knots if x > 0]
        for left, right in zip(knots[:-1], knots[1:]):
            optimum = minimize_scalar(lambda r: -evaluate(float(r)),
                                      bounds=(left, right),
                                      method="bounded",
                                      options={"xatol": sigma * 1e-5})
            candidates.append(float(optimum.x))
        mode = max(candidates, key=evaluate)
        scale = evaluate(mode)
        if not math.isfinite(scale):
            raise ConditioningNumericalError("Quaternion density overflow")
        points = sorted({
            x
            for x in [*knots, mode - sigma, mode, mode + sigma]
            if 0 < x < upper
        })

        def scaled_density(value: float, log_scale: float = scale) -> float:
            return math.exp(evaluate(value) - log_scale)

        result = quad(scaled_density,
                      0.,
                      upper,
                      points=points,
                      epsabs=0.,
                      epsrel=tolerance / 4,
                      limit=250,
                      full_output=1)
        value, error = result[:2]
        if len(result) != 3 or not math.isfinite(value) or value <= 0 or \
                error > tolerance * value:
            raise ConditioningNumericalError(
                "Quaternion marginal quadrature did not converge")
        answer = scale + math.log(value)
        distance = (math.hypot(radial_lower, upper) - center) / sigma
        tail_terms = [2 * math.log(sigma) - .5 * distance**2]
        if center > 0:
            tail_terms.append(
                math.log(center * sigma) + .5 * math.log(2 * math.pi) +
                float(log_ndtr(-distance)))
        log_tail = tail_constant + float(logsumexp(tail_terms))
        if log_tail <= answer + math.log(tolerance / 4):
            return answer
        upper += max(8 * sigma, upper / 2)
    raise ConditioningNumericalError("Quaternion marginal tail unresolved")
