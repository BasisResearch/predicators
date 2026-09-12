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
from typing import Mapping, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior


class IncompatibleJointObservation(ValueError):
    """An exact initial measurement lies outside this component's support.

    This is a prior-specific contradiction, not a finite sampler miss or
    proof that all possible initial-state models are inconsistent.
    """


@dataclass(frozen=True)
class JointStatePrior:
    """Independent bounded joint positions and explicit rest/motion choices.

    None position bounds denote a mechanically fixed joint with state
    (0, 0). Every movable joint needs finite positive-width position
    bounds, including continuous URDF joints: finite winding support is
    a prior assumption, not a mechanical limit or angle-wrapping rule.
    Positive velocity half-widths give normalized uniform velocities;
    zero denotes a prior atom at zero velocity. Dependencies and mixture
    masses require a separately specified full initial-state model.
    """
    names: Tuple[str, ...]
    position_bounds: Tuple[Optional[Tuple[float, float]], ...]
    velocity_half_widths: Tuple[float, ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        bounds = tuple(None if b is None else tuple(float(v) for v in b)
                       for b in self.position_bounds)
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
            elif (len(bound) != 2 or not all(math.isfinite(v) for v in bound)
                  or bound[0] >= bound[1]
                  or not math.isfinite(bound[1] - bound[0])):
                raise ValueError(
                    "Movable joint requires finite position bounds")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "position_bounds", bounds)
        object.__setattr__(self, "velocity_half_widths", widths)

    @property
    def digest(self) -> str:
        """Identify mechanical reductions and normalized component measures."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family":
                    "bounded_joint_positions_and_uniform_or_rest_velocities",
                    "prior": asdict(self)
                },
                sort_keys=True).encode())

    def condition_positions(
            self, observations: Mapping[str, float]) -> ConditionedJointPrior:
        """Eliminate measured initial coordinates and retain their density.

        The exact measurements must be initial positions, not future
        values injected into a rollout. Movable positions contribute
        their original uniform density. A fixed joint's zero contributes
        unit mass. Out-of-support observations raise a distinct error.
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
            bound = self.prior.position_bounds[self.prior.names.index(name)]
            if (bound is None
                    and value != 0) or (bound is not None
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
        for name, bound in zip(self.prior.names, self.prior.position_bounds):
            if bound is not None and name not in measured:
                names.append(name + ".position")
                bounds.append(bound)
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
        return -sum(
            math.log(bound[1] - bound[0])
            for name, bound in zip(self.prior.names, self.prior.position_bounds
                                   ) if name in measured and bound is not None)

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
        for name, bound in zip(self.prior.names, self.prior.position_bounds):
            position = measured.get(name, free.get(name + ".position", 0.))
            velocity = free.get(name + ".velocity", 0.)
            joints.append((float(position) if bound is not None else 0.,
                           float(velocity)))
        return tuple(joints)
