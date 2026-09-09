"""Helper predicates exposing the balloons' lift law to the oracle planner.

Which balloons hang the box in the band is the thing the domain hides;
the oracle receives it as helper predicates, which agent approaches
never see:

* ``Needed(?balloon, ?band)`` - the balloon is in the unique subset
  whose lift, by the analytic law, hangs the box inside the band;
* ``Holds(?clip, ?balloon)`` - the clip is the one in front of the
  balloon, the one that frees it;
* ``AllNeededTied(?band)`` (derived) - every needed balloon is freed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv, probe_env
from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.structs import DerivedPredicate, GroundAtom, Object, \
    Predicate, State, Type


def _index(obj: Object) -> int:
    """Rack position of a ``balloon<N>`` / ``clip<N>`` object."""
    return int(obj.name[len(obj.type.name):])


def _needed_holds(state: State, objects: Sequence[Object]) -> bool:
    balloon, _ = objects
    subset = probe_env().solution_subset(state)
    return subset is not None and _index(balloon) in subset


def _holds_holds(state: State, objects: Sequence[Object]) -> bool:
    del state
    clip, balloon = objects
    return _index(clip) == _index(balloon)


def _named(atoms: Iterable[GroundAtom], name: str) -> List[GroundAtom]:
    return [a for a in atoms if a.predicate.name == name]


def _all_needed_tied_holds(atoms: Set[GroundAtom],
                           objects: Sequence[Object]) -> bool:
    band, = objects
    needed = [
        a.objects[0] for a in _named(atoms, "Needed") if a.objects[1] == band
    ]
    if not needed:
        return False
    tied = {a.objects[0] for a in _named(atoms, "Tied")}
    return all(b in tied for b in needed)


def _never(state: State, objects: Sequence[Object]) -> bool:
    del state, objects
    return False


class PyBulletBalloonsGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Lift-law helper predicates for the balloons environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {PyBulletBalloonsEnv.get_name()}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        del env_name  # unused
        balloon_type = types["balloon"]
        clip_type = types["clip"]
        band_type = types["band"]
        Needed = Predicate(
            "Needed", [balloon_type, band_type],
            _needed_holds,
            natural_language_assertion=lambda os:
            f"balloon {os[0]} is one of those that hang the box in {os[1]}")
        Holds = Predicate("Holds", [clip_type, balloon_type],
                          _holds_holds,
                          natural_language_assertion=lambda os:
                          f"clip {os[0]} holds balloon {os[1]}")
        Tied = Predicate("Tied", [balloon_type], _never)
        AllNeededTied = DerivedPredicate("AllNeededTied", [band_type],
                                         _all_needed_tied_holds,
                                         auxiliary_predicates={Needed, Tied})
        return {Needed, Holds, AllNeededTied}
