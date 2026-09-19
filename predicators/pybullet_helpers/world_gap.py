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
from typing import Callable, Dict, Iterable, Optional, Tuple

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
        self._scaler = DynamicsScaler()

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
        domain last set."""
        self._scaler.apply(
            physics_client_id, skip_bodies, lambda body, link: self._factor(
                _MASS, body * 1000 + link + 1, self.mass), lambda body, link:
            self._factor(_FRICTION, body * 1000 + link + 1, self.friction))


class DynamicsScaler:
    """Scales every link's mass and lateral friction over its nominal value.

    The nominal value is whatever the domain code last set: a value that
    differs from the one this scaler set is read as a new nominal (a
    domain reset at a task boundary), per quantity, so a domain that
    resets one quantity leaves the other's scaling in place and nothing
    compounds. A zero mass stays static.
    """

    def __init__(self) -> None:
        # (body, link) -> (mass set, friction set, nominal mass, nominal
        # friction).
        self._record: Dict[Tuple[int, int], Tuple[float, float, float,
                                                  float]] = {}

    def apply(self, physics_client_id: int, skip_bodies: Iterable[int],
              mass_factor: Callable[[int, int], float],
              friction_factor: Callable[[int, int], float]) -> None:
        """Set every non-skipped link to its nominal values times the
        factors."""
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
                current = (float(info[0]), float(info[1]))
                record = self._record.get((body, link))
                nominal_mass, nominal_friction = current
                if record is not None:
                    if np.isclose(current[0], record[0]):
                        nominal_mass = record[2]
                    if np.isclose(current[1], record[1]):
                        nominal_friction = record[3]
                mass = nominal_mass
                if mass > 0.0:
                    mass *= mass_factor(body, link)
                friction = nominal_friction * friction_factor(body, link)
                if not np.allclose((mass, friction), current):
                    p.changeDynamics(body,
                                     link,
                                     mass=mass,
                                     lateralFriction=friction,
                                     physicsClientId=physics_client_id)
                self._record[(body, link)] = (mass, friction, nominal_mass,
                                              nominal_friction)


class CalibrationMenu:
    """Per-type mass and friction scales a planning twin exposes for fitting.

    With ``CFG.sim_calibration_menu`` on, a world without a gap (the
    planning twin) offers, for every object type with a movable body,
    ``mass_scale_<type>`` and ``friction_scale_<type>``, plus
    ``friction_scale_support`` for every static body (tables, walls,
    fixtures). Each is a multiplier on the nominal value, 1.0 by
    default, fitted like any other physical parameter, so the harness's
    system identification and uncertainty cover the quantities a sim gap
    deviates.
    """

    LO, HI = 0.5, 2.0

    def __init__(self, movable_types: Iterable[str]) -> None:
        self.types = sorted(set(movable_types))
        self.values: Dict[str, float] = {name: 1.0 for name in self.names()}
        self._scaler = DynamicsScaler()

    def names(self) -> list:
        """The parameter names, per type then the support scale."""
        names = []
        for type_name in self.types:
            names += [f"mass_scale_{type_name}", f"friction_scale_{type_name}"]
        return names + ["friction_scale_support"]

    def info(self) -> Dict[str, Dict]:
        """The menu entries, in get_physical_param_info's format."""
        out: Dict[str, Dict] = {}
        for name in self.names():
            if name == "friction_scale_support":
                what = ("lateral friction of every static body (tables, "
                        "walls, fixtures)")
            else:
                quantity, type_name = name.split("_scale_")
                what = (
                    f"{'mass' if quantity == 'mass' else 'lateral friction'}"
                    f" of every {type_name} body")
            out[name] = {
                "default": 1.0,
                "lo": self.LO,
                "hi": self.HI,
                "scale": "log",
                "description": f"multiplier on the nominal {what}",
            }
        return out

    def take(self, params: Dict[str, float]) -> Dict[str, float]:
        """Store the menu's values from ``params`` and return the rest."""
        rest = {}
        for name, value in params.items():
            if name in self.values:
                self.values[name] = float(value)
            else:
                rest[name] = value
        return rest

    def apply(self, physics_client_id: int, skip_bodies: Iterable[int],
              body_types: Dict[int, str]) -> None:
        """Scale every body's mass and friction by its type's values."""
        movable = set(self.types)

        def mass_factor(body: int, _link: int) -> float:
            return self.values.get(f"mass_scale_{body_types.get(body)}", 1.0)

        static: Dict[int, bool] = {}

        def friction_factor(body: int, _link: int) -> float:
            type_name = body_types.get(body)
            if type_name in movable:
                return self.values[f"friction_scale_{type_name}"]
            if body not in static:
                static[body] = p.getDynamicsInfo(
                    body, -1, physicsClientId=physics_client_id)[0] == 0.0
            # A movable body of a type the domain's own menu covers keeps
            # the domain's value.
            return self.values["friction_scale_support"] if static[
                body] else 1.0

        self._scaler.apply(physics_client_id, skip_bodies, mass_factor,
                           friction_factor)


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
