"""Hidden as-built deviations of the live world from its nominal model.

With ``CFG.sim_gap`` on, the environment an agent acts in (built with
``skip_residual_dynamics=False``) differs from the scene it is described
by: every movable body is a little larger or smaller than its nominal
shape, every body's mass and lateral friction are off by a factor, and
the engine may integrate with other solver settings. The planning twin,
the scene manifest and the asset files keep the nominal values, so every
arm starts from the same imperfect description of the world, the way a
real robot's CAD models and calibration differ from the real scene.

The deviations are drawn from ``CFG.seed + CFG.sim_gap_seed_offset``,
once per body and quantity, and hold for the whole run. Geometry is
applied when a body is created (a PyBullet shape cannot be resized);
masses and frictions are applied before every simulated action, over
whatever the domain code last set, so a domain that resets a body's
dynamics at a task boundary is deviated again rather than restored.
"""
import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pybullet as p

from predicators.settings import CFG

# Salts separating the per-quantity draws of one body.
_GEOMETRY, _MASS, _FRICTION = 1, 2, 3


class WorldGap:
    """The deviations of one live world, keyed by body."""

    def __init__(self, seed: int, geometry: float, mass: float,
                 friction: float, solver_iterations: int,
                 substeps: int) -> None:
        self.seed = seed
        self.geometry = geometry
        self.mass = mass
        self.friction = friction
        self.solver_iterations = solver_iterations
        self.substeps = substeps
        # (body, link) -> (mass, lateral friction) this gap last set, to
        # tell its own values from a domain's reset of the nominal ones.
        self._applied: Dict[Tuple[int, int], Tuple[Optional[float],
                                                   Optional[float]]] = {}

    @classmethod
    def from_cfg(cls) -> "WorldGap":
        """The gap the configuration asks for."""
        return cls(seed=int(CFG.seed) + int(CFG.sim_gap_seed_offset),
                   geometry=float(CFG.sim_gap_geometry),
                   mass=float(CFG.sim_gap_mass),
                   friction=float(CFG.sim_gap_friction),
                   solver_iterations=int(CFG.sim_gap_solver_iterations),
                   substeps=int(CFG.sim_gap_substeps))

    def describe(self) -> str:
        """One log line naming the configured magnitudes (not the draws)."""
        return (f"seed {self.seed}, movable-body size +-{self.geometry:g}, "
                f"mass x/{1 + self.mass:g}, friction x/{1 + self.friction:g}, "
                f"solver iterations {self.solver_iterations or 'default'}, "
                f"substeps {self.substeps or 'default'}")

    def _uniform(self, salt: int, key: int) -> float:
        rng = np.random.default_rng([self.seed, salt, key])
        return float(rng.uniform(-1.0, 1.0))

    def _factor(self, salt: int, key: int, magnitude: float) -> float:
        """A multiplier log-uniform in [1 / (1 + magnitude), 1 + magnitude]."""
        if magnitude <= 0.0:
            return 1.0
        return float(np.exp(self._uniform(salt, key) * np.log1p(magnitude)))

    def size_scale(self, body_key: int) -> float:
        """The size multiplier of the movable body created as ``body_key``."""
        if self.geometry <= 0.0:
            return 1.0
        return 1.0 + self.geometry * self._uniform(_GEOMETRY, body_key)

    def _deviate(self, current: float, last: Optional[float], salt: int,
                 key: int, magnitude: float) -> float:
        """The deviated value of one quantity: ``current`` itself when it is
        the value this gap last set, else ``current`` read as the new nominal
        value and scaled (a zero mass stays static)."""
        if last is not None and np.isclose(current, last):
            return current
        if salt == _MASS and current <= 0.0:
            return current
        return current * self._factor(salt, key, magnitude)

    def configure_engine(self, physics_client_id: int) -> None:
        """Apply the solver settings the gap names; zero keeps a default."""
        settings = {}
        if self.solver_iterations > 0:
            settings["numSolverIterations"] = self.solver_iterations
        if self.substeps > 0:
            settings["numSubSteps"] = self.substeps
        if settings:
            p.setPhysicsEngineParameter(physicsClientId=physics_client_id,
                                        **settings)

    def apply_dynamics(self, physics_client_id: int,
                       skip_bodies: Iterable[int]) -> None:
        """Deviate every link's mass and lateral friction from the value the
        domain last set; a value this gap set itself is left alone."""
        skip = set(skip_bodies)
        for index in range(p.getNumBodies(physicsClientId=physics_client_id)):
            body = p.getBodyUniqueId(index, physicsClientId=physics_client_id)
            if body in skip:
                continue
            links = [-1] + list(
                range(p.getNumJoints(body, physicsClientId=physics_client_id)))
            for link in links:
                info = p.getDynamicsInfo(body,
                                         link,
                                         physicsClientId=physics_client_id)
                key = body * 1000 + link + 1
                last = self._applied.get((body, link), (None, None))
                # Each quantity separately: a domain that resets one of
                # them leaves the other's deviation in place.
                mass = self._deviate(float(info[0]), last[0], _MASS, key,
                                     self.mass)
                friction = self._deviate(float(info[1]), last[1], _FRICTION,
                                         key, self.friction)
                if (mass, friction) != (float(info[0]), float(info[1])):
                    p.changeDynamics(body,
                                     link,
                                     mass=mass,
                                     lateralFriction=friction,
                                     physicsClientId=physics_client_id)
                self._applied[(body, link)] = (mass, friction)


# Live worlds with a gap, by physics client. While an env builds its
# world (its client is not known until its first body is created), the
# build's gap, or None for a nominal world, decides every body's size.
_WORLDS: Dict[int, WorldGap] = {}
_BUILDING = False
_PENDING: Optional[WorldGap] = None


def begin_world(gap: Optional[WorldGap]) -> None:
    """Start building a world deviated by ``gap`` (None: nominal)."""
    global _BUILDING, _PENDING  # pylint: disable=global-statement
    _BUILDING, _PENDING = True, gap


def bind_world(physics_client_id: int) -> Optional[WorldGap]:
    """End the build of the world owned by ``physics_client_id`` and return its
    gap.

    A nominal world clears any record a disconnected world left on the
    same client id.
    """
    global _BUILDING, _PENDING  # pylint: disable=global-statement
    gap = _PENDING
    _BUILDING, _PENDING = False, None
    if gap is None:
        _WORLDS.pop(physics_client_id, None)
        return None
    _WORLDS[physics_client_id] = gap
    logging.info("[world gap] %s", gap.describe())
    return gap


def release_world(physics_client_id: int) -> None:
    """Forget a disconnected world's gap."""
    _WORLDS.pop(physics_client_id, None)


def size_scale(physics_client_id: int, movable: bool) -> float:
    """The size multiplier for a body about to be created in a world.

    Static bodies (tables, walls, fixtures) keep their nominal shape.
    """
    if not movable:
        return 1.0
    gap = _PENDING if _BUILDING else _WORLDS.get(physics_client_id)
    if gap is None:
        return 1.0
    body_key = p.getNumBodies(physicsClientId=physics_client_id)
    return gap.size_scale(body_key)
