"""Explicit offline joint priors, conditioned on exact initial positions.

Joint bounds and motion assumptions belong to the declared model. They
are not estimated from the observed extrema or filled from evaluator
metadata. This component does not establish robot/scene collision
feasibility or solve exact constraints at subsequent trajectory steps.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Mapping, Optional, Tuple, Union

import numpy as np

from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior


class IncompatibleJointObservation(ValueError):
    """An exact initial measurement lies outside this component's support.

    This is a prior-specific contradiction, not a finite sampler miss or
    proof that all possible initial-state models are inconsistent.
    """


class JointCoordinateBoundary(ValueError):
    """A zero-measure coordinate boundary has no supported conditional lift."""


@dataclass(frozen=True)
class GaussianJointPosition:
    """A declared Gaussian over reset angles/positions, without wrapping.

    This describes a simulator initialization law, not an ideal hard
    mechanical limit. Its parameters must be fixed independently of the
    readings used for a fit. It does not certify collision feasibility.
    """
    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean) or not math.isfinite(
                self.sigma) or self.sigma <= 0:
            raise ValueError(
                "Gaussian position needs a finite mean and positive scale")

    def log_density(self, value: float) -> float:
        """Retain exact-position information under the original reset law."""
        ratio = (value - self.mean) / self.sigma
        result = -.5 * ratio * ratio - math.log(
            self.sigma) - .5 * math.log(2 * math.pi)
        if not math.isfinite(result):
            raise ArithmeticError(
                "Gaussian position density is not representable")
        return result

    def quantile(self, unit: float) -> float:
        """Push uniform open-unit coordinates to the whole real line."""
        if not 0 < unit < 1:
            raise JointCoordinateBoundary(
                "Gaussian quantiles require an interior coordinate")
        result = self.mean + self.sigma * NormalDist().inv_cdf(unit)
        if not math.isfinite(result):
            raise ArithmeticError(
                "Gaussian position quantile is not representable")
        return result


PositionPrior = Union[Tuple[float, float], GaussianJointPosition]


@dataclass(frozen=True)
class JointStatePrior:
    """Independent position priors and explicit rest/motion choices.

    None denotes a mechanically fixed joint with state (0, 0). Movable
    joints have either finite uniform bounds or Gaussian reset
    positions. Finite winding support and Gaussian tails are explicit
    assumptions, not mechanical limits or angle-wrapping rules. Positive
    velocity half-widths give normalized uniform velocities; zero
    denotes a prior atom at zero velocity. Dependencies and mixture
    masses require a separately specified full initial-state model.
    """
    names: Tuple[str, ...]
    position_priors: Tuple[Optional[PositionPrior], ...]
    velocity_half_widths: Tuple[float, ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        bounds = tuple(
            b if b is None or isinstance(b, GaussianJointPosition) else tuple(
                float(v) for v in b) for b in self.position_priors)
        widths = tuple(float(v) for v in self.velocity_half_widths)
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n for n in names)
                or len(bounds) != len(names) or len(widths) != len(names)):
            raise ValueError(
                "Joint prior requires distinct names and matching fields")
        for bound, width in zip(bounds, widths):
            if not math.isfinite(width) or width < 0 or not math.isfinite(
                    2 * width):
                raise ValueError(
                    "Joint velocity width must be finite and nonnegative")
            if bound is None:
                if width != 0:
                    raise ValueError(
                        "Fixed joint cannot have uncertain velocity")
            elif isinstance(bound, GaussianJointPosition):
                continue
            elif (len(bound) != 2 or not all(math.isfinite(v) for v in bound)
                  or bound[0] >= bound[1]
                  or not math.isfinite(bound[1] - bound[0])):
                raise ValueError(
                    "Movable joint requires finite position bounds")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "position_priors", bounds)
        object.__setattr__(self, "velocity_half_widths", widths)

    @property
    def digest(self) -> str:
        """Identify mechanical reductions and normalized component measures."""
        return content_digest(
            json.dumps(
                {
                    "schema": 2,
                    "family":
                    "joint_position_priors_and_uniform_or_rest_velocities",
                    "prior": asdict(self)
                },
                sort_keys=True).encode())

    def condition_positions(
            self, observations: Mapping[str, float]) -> ConditionedJointPrior:
        """Eliminate measured initial coordinates and retain their density.

        The exact measurements must be initial positions, not future
        values injected into a rollout. Movable positions contribute
        their original position density. A fixed joint's zero
        contributes unit mass. Out-of-support observations raise a
        distinct error.
        """
        return ConditionedJointPrior(self, tuple(observations.items()))


@dataclass(frozen=True)
class ConditionedJointPrior:
    """A fixed-prior conditional component, retaining all URDF joint states.

    Construct via JointStatePrior.condition_positions. An empty free
    coordinate space is a deterministic conditional, not a fake
    interval. Its observation factor remains available even when no
    sampling is needed. This result does not assert feasibility of later
    outputs.
    """
    prior: JointStatePrior
    observations: Tuple[Tuple[str, float], ...]

    def __post_init__(self) -> None:
        values = tuple(sorted((n, float(v)) for n, v in self.observations))
        names = [n for n, _ in values]
        if len(set(names)) != len(names) or not set(names) <= set(
                self.prior.names):
            raise ValueError("Unknown or repeated observed joint")
        for name, value in values:
            if not math.isfinite(value):
                raise ValueError("Joint observation must be finite")
            bound = self.prior.position_priors[self.prior.names.index(name)]
            if (bound is None
                    and value != 0) or (isinstance(bound, tuple)
                                        and not bound[0] <= value <= bound[1]):
                raise IncompatibleJointObservation(
                    f"Exact position for {name} is outside the declared prior")
        object.__setattr__(self, "observations", values)

    @property
    def coordinates(self) -> Optional[BoxPrior]:
        """Normalized free position/velocity coordinates, or a point mass."""
        measured = dict(self.observations)
        names = []
        bounds = []
        for name, bound in zip(self.prior.names, self.prior.position_priors):
            if bound is not None and name not in measured:
                names.append(name + ".position")
                bounds.append((
                    0.,
                    1.) if isinstance(bound, GaussianJointPosition) else bound)
        for name, width in zip(self.prior.names,
                               self.prior.velocity_half_widths):
            if width:
                names.append(name + ".velocity")
                bounds.append((-width, width))
        return BoxPrior(tuple(names), tuple(bounds)) if names else None

    @property
    def log_observation_factor(self) -> float:
        """Density/mass of the conditioned readings under the original
        prior."""
        measured = dict(self.observations)
        return sum(
            bound.log_density(measured[name]) if isinstance(
                bound, GaussianJointPosition) else -math.log(bound[1] -
                                                             bound[0]) for
            name, bound in zip(self.prior.names, self.prior.position_priors)
            if name in measured and bound is not None)

    @property
    def digest(self) -> str:
        """Bind the original prior and exact initial conditioning values."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "prior": self.prior.digest,
                    "initial_positions": self.observations
                },
                sort_keys=True).encode())

    def lift(self, point: np.ndarray) -> Tuple[Tuple[float, float], ...]:
        """Return position/velocity for every joint in declared URDF order."""
        space = self.coordinates
        values = np.asarray(point, dtype=float)
        size = 0 if space is None else len(space.names)
        if values.shape != (size, ) or not np.isfinite(values).all():
            raise ValueError("Invalid free joint coordinates")
        free = {}
        if space is not None:
            bounds = np.asarray(space.bounds)
            if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
                raise ValueError(
                    "Free joint coordinates outside prior support")
            free = dict(zip(space.names, values))
        measured = dict(self.observations)
        joints = []
        for name, bound in zip(self.prior.names, self.prior.position_priors):
            position = measured.get(name, free.get(name + ".position", 0.))
            if isinstance(bound,
                          GaussianJointPosition) and name not in measured:
                position = bound.quantile(position)
            velocity = free.get(name + ".velocity", 0.)
            joints.append((float(position) if bound is not None else 0.,
                           float(velocity)))
        return tuple(joints)


@dataclass(frozen=True)
class RestingJointPrior:
    """A declared rest/motion law for an articulated joint.

    Rest mass is divided equally between two declared controller poses,
    with zero velocity. The remaining mass has independent uniform
    position and velocity. This engineering prior is not implied by a
    Boolean reading or by the absence of recorded joint motion.
    """
    name: str
    lower: float
    upper: float
    rest_positions: Tuple[float, float]
    rest_probability: float
    velocity_half_width: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or any(
                not math.isfinite(value) for value in (
                self.lower, self.upper, self.rest_probability,
                self.velocity_half_width)) or self.lower >= self.upper or \
                not math.isfinite(self.upper - self.lower) or \
                not 0 <= self.rest_probability <= 1 or \
                self.velocity_half_width <= 0:
            raise ValueError("Invalid resting joint prior")
        poses = tuple(float(value) for value in self.rest_positions)
        if len(poses) != 2 or not self.lower <= poses[0] < poses[1] <= \
                self.upper:
            raise ValueError("Rest poses must lie inside the joint support")
        object.__setattr__(self, "rest_positions", poses)

    @property
    def digest(self) -> str:
        """Identify controller-pose atoms and the normalized continuous
        component."""
        return content_digest(
            json.dumps(
                {
                    "family": "controller_rest_uniform_motion",
                    "schema": 1,
                    "prior": asdict(self)
                },
                sort_keys=True).encode("utf-8"))

    def condition_above(self, threshold: float,
                        observed: bool) -> ThresholdJointPrior:
        """Condition on position > threshold without selecting one angle."""
        return ThresholdJointPrior(self, threshold, observed)

    @property
    def coordinates(self) -> BoxPrior:
        """Unit coordinates for the original, unobserved joint law."""
        return BoxPrior(
            (self.name + ".position_mixture", self.name + ".velocity"),
            ((0., 1.), ) * 2)

    def lift(self, point: np.ndarray) -> Tuple[float, float]:
        """Draw the original rest/motion prior without inventing a reading.

        The second coordinate is an unused uniform auxiliary in either
        rest case. The original law is already normalized, so this map
        contributes no observation or proposal-density correction.
        """
        values = np.asarray(point, dtype=float)
        if values.shape != (2, ) or not np.isfinite(values).all():
            raise ValueError("Invalid original joint coordinates")
        if np.any(values <= 0) or np.any(values >= 1):
            raise JointCoordinateBoundary(
                "Original joint coordinates must be interior")
        for index, position in enumerate(self.rest_positions):
            if values[0] < self.rest_probability * (index + 1) / 2:
                return position, 0.
        position = self.lower + (self.upper - self.lower) * \
            (float(values[0]) - self.rest_probability) / \
            (1 - self.rest_probability)
        if not self.lower < position < self.upper:
            raise JointCoordinateBoundary(
                "Interior position rounded to an original-prior boundary")
        velocity = self.velocity_half_width * (2 * float(values[1]) - 1)
        return position, velocity


@dataclass(frozen=True)
class ThresholdJointPrior:
    """Conditional position/motion with the Boolean observation mass retained.

    The two unit coordinates encode the rest/moving mixture and
    velocity. They are auxiliary uniforms in each rest case, which has
    no continuous physical dimensions; the moving case has two.
    """
    prior: RestingJointPrior
    threshold: float
    observed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observed, bool) or not \
                self.prior.lower < self.threshold < self.prior.upper:
            raise ValueError("Threshold must be interior and reading Boolean")
        if self.prior.rest_probability == 1 and not self._rest_positions:
            raise IncompatibleJointObservation("No rest pose matches reading")

    @property
    def _rest_positions(self) -> Tuple[float, ...]:
        return tuple(q for q in self.prior.rest_positions
                     if (q > self.threshold) == self.observed)

    @property
    def _interval(self) -> Tuple[float, float]:
        if self.observed:
            return self.threshold, self.prior.upper
        return self.prior.lower, self.threshold

    @property
    def log_observation_factor(self) -> float:
        """Probability mass, including any parameter-dependent event cut."""
        lower, upper = self._interval
        rest = self.prior.rest_probability
        atom = math.log(rest) + math.log(len(self._rest_positions)) - \
            math.log(2) if rest and self._rest_positions else -math.inf
        motion = math.log1p(-rest) + math.log(upper - lower) - \
            math.log(self.prior.upper - self.prior.lower) if rest < 1 else \
            -math.inf
        larger, smaller = max(atom, motion), min(atom, motion)
        return larger + math.log1p(math.exp(smaller - larger))

    @property
    def rest_probability(self) -> float:
        """Posterior mass on the compatible resting controller poses."""
        rest = self.prior.rest_probability
        return math.exp(
            math.log(rest) + math.log(len(self._rest_positions)) - math.log(2)
            - self.log_observation_factor) if rest and \
            self._rest_positions else 0.

    @property
    def coordinates(self) -> BoxPrior:
        """Normalized proposal coordinates with unused rest-case
        auxiliaries."""
        return BoxPrior((self.prior.name + ".position_mixture",
                         self.prior.name + ".velocity"), ((0., 1.), ) * 2)

    def lift(self, point: np.ndarray) -> Tuple[float, float]:
        """Draw from the conditional measure; never project an invalid draw."""
        values = np.asarray(point, dtype=float)
        if values.shape != (2, ) or not np.isfinite(values).all():
            raise ValueError("Invalid conditional joint coordinates")
        if np.any(values <= 0) or np.any(values >= 1):
            raise JointCoordinateBoundary(
                "Threshold joint coordinates must be interior")
        endpoint = self.rest_probability
        for index, position in enumerate(self._rest_positions):
            if values[0] < endpoint * (index + 1) / len(self._rest_positions):
                return position, 0.
        lower, upper = self._interval
        position = lower + (upper - lower) * \
            (float(values[0]) - endpoint) / (1 - endpoint)
        if not lower < position < upper:
            raise JointCoordinateBoundary(
                "Interior position rounded to a conditional boundary")
        velocity = self.prior.velocity_half_width * (2 * float(values[1]) - 1)
        return position, velocity
