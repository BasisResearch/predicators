"""Helper predicates exposing the crane's physics to the oracle planner.

Whether some pull lands the crate on the bin is the thing the domain
hides. The oracle needs it to plan, and the channel for that is a
ground-truth HELPER predicate, which only oracle / process-planning
approaches receive:

* ``Hittable(?ram, ?crate, ?bin)`` - from this state, some pull in the
  scanned range lands the crate on the bin.

Answered by running the Pull skill and the swing on a dedicated env
instance (:func:`pybullet_crane.probe_env`) over the scan, memoised on
the lane's pose, since the planner abstracts the same state many
times.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from predicators.envs.pybullet_crane import PyBulletCraneEnv, crate_fallen, \
    crate_reachable, probe_env
from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type

_SCAN_CACHE: Dict[Tuple, List[float]] = {}
_SCAN_CACHE_MAX = 512


def _lane_key(state: State) -> Tuple:
    """The settled lane's pose, at the resolution the probe cares about."""
    crane = next(o for o in state if o.type.name == "crane")
    crate = next(o for o in state if o.type.name == "crate")
    bin_ = next(o for o in state if o.type.name == "bin")
    return (round(state.get(crane, "y"),
                  3), round(state.get(crane, "length"),
                            3), round(state.get(crate, "x"),
                                      2), round(state.get(crate, "y"), 2),
            round(state.get(crate, "z"),
                  2), int(round(state.get(crate, "color"))),
            round(state.get(bin_, "x"), 3), round(state.get(bin_, "half"), 3))


def working_pulls(state: State) -> List[float]:
    """Memoised scan of the pulls that land the crate on the bin from
    ``state``."""
    key = _lane_key(state)
    if key not in _SCAN_CACHE:
        if len(_SCAN_CACHE) >= _SCAN_CACHE_MAX:
            _SCAN_CACHE.clear()
        _SCAN_CACHE[key] = probe_env().working_pulls(state)
    return _SCAN_CACHE[key]


def _hittable_holds(state: State, objects: Sequence[Object]) -> bool:
    ram, crate, bin_ = objects
    del bin_
    if crate_fallen(state, crate) or not crate_reachable(state, ram, crate):
        return False
    # Only a settled lane is worth a scan: while the ram swings or the
    # crate slides, every step is a new state and the probe would run
    # at each of them.
    if state.get(crate, "speed") >= CFG.crane_settle_speed or \
            state.get(ram, "speed") >= CFG.crane_settle_speed:
        return False
    return bool(working_pulls(state))


class PyBulletCraneGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Swing-outcome helper predicates for the crane environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {PyBulletCraneEnv.get_name()}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        del env_name  # unused
        Hittable = Predicate(
            "Hittable", [types["ram"], types["crate"], types["bin"]],
            _hittable_holds,
            natural_language_assertion=lambda os:
            f"some swing of ram {os[0]} lands {os[1]} on {os[2]}")
        return {Hittable}
