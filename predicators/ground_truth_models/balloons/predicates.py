"""Helper predicates exposing the balloons' lift law to the oracle planner.

Which clips hang the box in the band is the thing the domain hides;
the oracle receives it as helper predicates, which agent approaches
never see:

* ``Needed(?clip, ?band)`` - the clip is in the reference subset whose
  balloons, by the witnessed rollouts, hang the box inside the band;
* ``Holds(?clip, ?balloon)`` - the clip is the one that frees the
  balloon (its ``clip`` feature; a bundle shares one clip);
* ``AllNeededOpen(?band)`` (derived) - every needed clip is open.

The helpers are over clips, not balloons, so one Release frees a whole
bundle without the planner having to know how many balloons it holds.
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
    clip, _ = objects
    subset = probe_env().solution_subset(state)
    return subset is not None and _index(clip) in subset


def _holds_holds(state: State, objects: Sequence[Object]) -> bool:
    clip, balloon = objects
    return _index(clip) == PyBulletBalloonsEnv.clip_index(state, balloon)


def _named(atoms: Iterable[GroundAtom], name: str) -> List[GroundAtom]:
    return [a for a in atoms if a.predicate.name == name]


def _all_needed_open_holds(atoms: Set[GroundAtom],
                           objects: Sequence[Object]) -> bool:
    band, = objects
    needed = [
        a.objects[0] for a in _named(atoms, "Needed") if a.objects[1] == band
    ]
    if not needed:
        return False
    on = {a.objects[0] for a in _named(atoms, "ClipOn")}
    return all(c in on for c in needed)


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
            "Needed", [clip_type, band_type],
            _needed_holds,
            natural_language_assertion=lambda os:
            f"clip {os[0]} frees balloons that hang the box in {os[1]}")
        Holds = Predicate("Holds", [clip_type, balloon_type],
                          _holds_holds,
                          natural_language_assertion=lambda os:
                          f"clip {os[0]} holds balloon {os[1]}")
        ClipOn = Predicate("ClipOn", [clip_type], _never)
        AllNeededOpen = DerivedPredicate("AllNeededOpen", [band_type],
                                         _all_needed_open_holds,
                                         auxiliary_predicates={Needed, ClipOn})
        return {Needed, Holds, AllNeededOpen}
