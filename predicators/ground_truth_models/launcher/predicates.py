"""Helper predicates exposing the launcher's physics to the oracle planner.

Whether some compression takes the top block down and leaves the rest
of the tower standing is the thing the domain hides. The oracle needs
it to plan, and the channel for that is a ground-truth HELPER
predicate, which only oracle / process-planning approaches receive:

* ``Hittable(?launcher, ?block)`` - from this state, some compression
  in the scanned range topples the block while every other block of
  the tower stays standing.

Answered by running the Cock skill and the flight on a dedicated env
instance (:func:`pybullet_launcher.probe_env`) over the scan, memoised
on the tower's pose, since the planner abstracts the same state many
times.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from predicators.envs.pybullet_launcher import PyBulletLauncherEnv, \
    ball_loaded, block_toppled, probe_env
from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.structs import Object, Predicate, State, Type

_SCAN_CACHE: Dict[Tuple, List[float]] = {}
_SCAN_CACHE_MAX = 512


def _tower_key(state: State) -> Tuple:
    blocks = sorted((o for o in state if o.type.name == "block"),
                    key=lambda o: o.name)
    launcher = next(o for o in state if o.type.name == "launcher")
    return (tuple(
        (b.name, round(state.get(b, "x"), 3), round(state.get(b, "y"), 3),
         round(state.get(b, "z"), 3), int(round(state.get(b, "color"))))
        for b in blocks), round(state.get(launcher, "balls_left")))


def working_depths(state: State) -> List[float]:
    """Memoised scan of the compressions that reach the goal from ``state``."""
    key = _tower_key(state)
    if key not in _SCAN_CACHE:
        if len(_SCAN_CACHE) >= _SCAN_CACHE_MAX:
            _SCAN_CACHE.clear()
        _SCAN_CACHE[key] = probe_env().working_depths(state)
    return _SCAN_CACHE[key]


def _hittable_holds(state: State, objects: Sequence[Object]) -> bool:
    launcher, block = objects
    del launcher
    if block_toppled(state, block) or not ball_loaded(state):
        return False
    blocks = sorted((o for o in state if o.type.name == "block"),
                    key=lambda o: o.name)
    if block != blocks[-1]:
        # Only the top block can be taken alone.
        return False
    return bool(working_depths(state))


class PyBulletLauncherGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Launch-outcome helper predicates for the launcher environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {PyBulletLauncherEnv.get_name()}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        del env_name  # unused
        Hittable = Predicate(
            "Hittable", [types["launcher"], types["block"]],
            _hittable_holds,
            natural_language_assertion=lambda os:
            f"some shot from {os[0]} topples {os[1]} and nothing else")
        return {Hittable}
