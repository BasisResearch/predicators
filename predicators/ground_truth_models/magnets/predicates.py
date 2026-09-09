"""Helper predicates exposing the magnets' field to the oracle planner.

Which colours the wand pulls is the thing the domain hides, so it
cannot be a feature of any State; the oracle receives it as a helper
predicate, which agent approaches never see:

* ``Pulled(?piece)`` - the wand attracts this piece's colour;
* ``Pushed(?piece)`` - the wand repels it.
"""

from __future__ import annotations

from typing import Dict, Sequence, Set

from predicators.envs.pybullet_magnets import PyBulletMagnetsEnv
from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.structs import Object, Predicate, State, Type


def _pulled_holds(state: State, objects: Sequence[Object]) -> bool:
    piece, = objects
    return PyBulletMagnetsEnv.polarity(state.get(piece, "color")) > 0


def _pushed_holds(state: State, objects: Sequence[Object]) -> bool:
    return not _pulled_holds(state, objects)


class PyBulletMagnetsGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Polarity helper predicates for the magnets environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {PyBulletMagnetsEnv.get_name()}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        del env_name  # unused
        piece_type = types["piece"]
        Pulled = Predicate("Pulled", [piece_type],
                           _pulled_holds,
                           natural_language_assertion=lambda os:
                           f"the wand pulls piece {os[0]}")
        Pushed = Predicate("Pushed", [piece_type],
                           _pushed_holds,
                           natural_language_assertion=lambda os:
                           f"the wand pushes piece {os[0]} away")
        return {Pulled, Pushed}
