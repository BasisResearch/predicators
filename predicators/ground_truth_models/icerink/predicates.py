"""Helper predicates exposing the ice rink's physics to the oracle planner.

Where a pushed tile stops is the thing the domain hides, so it cannot be
a feature of any State. The oracle needs it to plan, and the codebase's
channel for exactly that is a ground-truth HELPER predicate: only
oracle / process-planning approaches receive these, and agent
approaches never see them.

* ``Reachable(?tile, ?direction, ?target)`` - the target lies ahead of
  the tile along the direction, and a push at the top of the speed
  range from this state carries the tile at least that far without
  leaving the rink. Some speed in the range then lands the tile on the
  target; the process sampler searches for it.
* ``WouldLeave(?tile, ?direction)`` - a push at the top of the speed
  range slides the tile off the rink.

Both are answered by running the push skill on a dedicated env
instance (:func:`pybullet_icerink.probe_env`), hidden dynamics
included, until the rink settles. Results are memoised on the tiles'
poses, since the planner abstracts the same state many times.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.envs.pybullet_icerink import PLANNED_DIRECTIONS, \
    PyBulletIceRinkEnv, probe_env, tile_lost
from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type

_PUSH_CACHE: Dict[Tuple, Optional[Tuple[float, float, bool]]] = {}
_PUSH_CACHE_MAX = 4096


def _pose_key(state: State) -> Tuple:
    tiles = sorted((o for o in state if o.type.name == "tile"),
                   key=lambda o: o.name)
    return tuple(
        (t.name, round(state.get(t, "x"), 3), round(state.get(t, "y"), 3),
         int(round(state.get(t, "color")))) for t in tiles)


def push_outcome(state: State, tile: Object, direction: str,
                 speed: float) -> Optional[Tuple[float, float, bool]]:
    """Memoised probe of where ``tile`` stops after a push along ``direction``
    at ``speed`` from ``state``."""
    key = (_pose_key(state), tile.name, direction, round(float(speed), 4))
    if key not in _PUSH_CACHE:
        if len(_PUSH_CACHE) >= _PUSH_CACHE_MAX:
            _PUSH_CACHE.clear()
        _PUSH_CACHE[key] = probe_env().push_outcome(state, tile, direction,
                                                    float(speed))
    return _PUSH_CACHE[key]


def along_direction(state: State, tile: Object, direction: Object,
                    target: Object) -> Optional[float]:
    """The target's distance ahead of the tile along the direction, or None
    when it does not lie on the tile's line within the goal tolerance."""
    yaw = state.get(direction, "yaw")
    fx, fy = float(np.sin(yaw)), float(np.cos(yaw))
    dx = state.get(target, "x") - state.get(tile, "x")
    dy = state.get(target, "y") - state.get(tile, "y")
    ahead = dx * fx + dy * fy
    lateral = abs(-dx * fy + dy * fx)
    if lateral >= CFG.icerink_on_tol or ahead <= 0.0:
        return None
    return float(ahead)


# Speeds the oracle probes across the range. Travel is not quite
# monotone in the commanded speed at the top of the range (the arm's
# tracking of a fast stroke lags), so a grid rather than the endpoints
# says what the range covers, and the sampler refines between grid
# neighbours.
SPEED_GRID_SIZE = 5


def speed_grid() -> List[float]:
    """The probed speeds, lowest first."""
    lo, hi = (float(v) for v in CFG.icerink_push_speed_range)
    return [float(v) for v in np.linspace(lo, hi, SPEED_GRID_SIZE)]


def travel_along(state: State, tile: Object, direction: Object,
                 speed: float) -> Optional[float]:
    """How far along the direction a push at ``speed`` carries the tile;
    infinity when it leaves the rink, None when the skill cannot run."""
    outcome = push_outcome(state, tile, direction.name, speed)
    if outcome is None:
        return None
    if outcome[2]:
        return float("inf")
    yaw = state.get(direction, "yaw")
    return float((outcome[0] - state.get(tile, "x")) * np.sin(yaw) +
                 (outcome[1] - state.get(tile, "y")) * np.cos(yaw))


def _reachable_holds(state: State, objects: Sequence[Object]) -> bool:
    tile, direction, target = objects
    if direction.name not in PLANNED_DIRECTIONS or tile_lost(state, tile):
        return False
    ahead = along_direction(state, tile, direction, target)
    if ahead is None:
        return False
    travels = [travel_along(state, tile, direction, s) for s in speed_grid()]
    tol = CFG.icerink_on_tol
    if any(t is not None and abs(t - ahead) <= tol for t in travels):
        return True
    # A bracket: some push stops short of the target and a faster one
    # goes past it, so a speed in between lands on it.
    shorter = any(t is not None and t < ahead for t in travels)
    longer = any(t is not None and t > ahead for t in travels)
    return shorter and longer


def _would_leave_holds(state: State, objects: Sequence[Object]) -> bool:
    tile, direction = objects
    if tile_lost(state, tile):
        return True
    return any(
        travel_along(state, tile, direction, s) == float("inf")
        for s in speed_grid())


class PyBulletIceRinkGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Push-outcome helper predicates for the ice rink environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {PyBulletIceRinkEnv.get_name()}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """The physics predicates, for oracle approaches only."""
        del env_name  # unused
        tile_type = types["tile"]
        target_type = types["target"]
        direction_type = types["direction"]
        Reachable = Predicate(
            "Reachable", [tile_type, direction_type, target_type],
            _reachable_holds,
            natural_language_assertion=lambda os:
            f"some push of {os[0]} {os[1]} lands it on {os[2]}")
        WouldLeave = Predicate(
            "WouldLeave", [tile_type, direction_type],
            _would_leave_holds,
            natural_language_assertion=lambda os:
            f"a hard push of {os[0]} {os[1]} slides it off the rink")
        return {Reachable, WouldLeave}
