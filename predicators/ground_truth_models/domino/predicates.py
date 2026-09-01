"""Helper predicates for the domino environment.

The grid predicates (DominoAtPos, DominoAtRot, PosClear,
InFrontDirection, InFront, AdjacentTo) are defined canonically by
``GridComponent``; this factory simply delegates to it so there is a
single source of truth.
"""

from typing import Dict, Sequence, Set

from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type


class PyBulletDominoGroundTruthPredicateFactory(GroundTruthPredicateFactory):
    """Ground-truth helper predicates for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry", "pybullet_domino_fan",
            "pybullet_domino_declare",
            "pybullet_domino_blow"
        }

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """Get helper predicates for the domino environment.

        Delegates to ``GridComponent``, the canonical definition of the
        grid predicates. Only oracle / process-planning approaches
        consume these helpers; agent approaches run grid-free.
        """
        if env_name == "pybullet_domino_blow":
            return _blow_helper_predicates(types)

        from predicators.envs.pybullet_domino.components.grid_component import \
            GridComponent  # pylint: disable=import-outside-toplevel
        return GridComponent(domino_type=types["domino"]).get_predicates()


# ── Blow task: the one thing the oracle knows and a learner must not ──


def _blow_slide_distance() -> float:
    """How far this gust carries the block, in metres.

    A ground-truth constant, fitted from the env's own wind force by the
    curve measured in settings.py (1.8 N -> 5.8 cm, 2.5 -> 11.5, 3.2 ->
    19.3): slide grows steeply and monotonically with force, which is
    the property that makes this task's parameter learnable at all.

    This lives in the ORACLE's helper predicates, never in the env's, so
    an agent approach cannot read it off the state. Knowing it is the
    whole content of the task.
    """
    # A quadratic least-squares fit to the measured curve, accurate to
    # under a millimetre from 1.8 to 3.2 N (5.75 / 8.79 / 11.53 / 14.61
    # / 19.33 cm). Cheaper and clearer than shipping a table, and it
    # extrapolates sensibly for a task generator that varies the force.
    force = CFG.domino_blow_wind_force
    return max(0.0, 0.02081 * force * force - 0.00703 * force + 0.00273)


def _blow_helper_predicates(types: Dict[str, Type]) -> Set[Predicate]:
    """``ReadyToBlow``: the block is where the gust will deliver it.

    The oracle plans Place -> DeclareFinished -> Wait, and this is the
    predicate that makes the Place worth doing: it is true exactly when
    the block sits one slide-length upwind of the goal patch, within the
    patch's own tolerance. The wind process then turns it into InGoal.
    """
    domino_type = types["domino"]
    region_type = types["region"]

    def _ready_holds(state: State, objects: Sequence[Object]) -> bool:
        domino, region = objects
        if state.get(domino, "is_held") > 0.5:
            return False
        # The fan blows +x, so the block must start upwind by one slide.
        want_x = float(state.get(region, "x")) - _blow_slide_distance()
        dx = abs(float(state.get(domino, "x")) - want_x)
        dy = abs(float(state.get(domino, "y")) - float(state.get(region, "y")))
        return (dx <= float(state.get(region, "half_x"))
                and dy <= float(state.get(region, "half_y")))

    return {
        Predicate("ReadyToBlow", [domino_type, region_type], _ready_holds)
    }
