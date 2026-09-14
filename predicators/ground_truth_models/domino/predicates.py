"""Helper predicates for the domino environment.

The grid predicates (DominoAtPos, DominoAtRot, PosClear,
InFrontDirection, InFront, AdjacentTo) are defined canonically by
``GridComponent``; this factory simply delegates to it so there is a
single source of truth.
"""

from typing import Dict, Sequence, Set, Tuple

import numpy as np

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
            "pybullet_domino_declare", "pybullet_domino_blow",
            "pybullet_domino_blow_real"
        }

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """Get helper predicates for the domino environment.

        Delegates to ``GridComponent``, the canonical definition of the
        grid predicates. Only oracle / process-planning approaches
        consume these helpers; agent approaches run grid-free.
        """
        if env_name in BLOW_ENVS:
            return _blow_helper_predicates(types)

        from predicators.envs.pybullet_domino.components.grid_component import \
            GridComponent  # pylint: disable=import-outside-toplevel
        return GridComponent(domino_type=types["domino"]).get_predicates()


# ── Blow task: the one thing the oracle knows and a learner must not ──

# The generated blow task and its real-bench twin share one model: the
# same block, the same gust, the same patch. They differ in where the fan
# stands and which way it blows, which every reader below takes from the
# fan object rather than assuming +x.
BLOW_ENVS = frozenset({"pybullet_domino_blow", "pybullet_domino_blow_real"})


def blow_wind_dir(state: State) -> Tuple[float, float]:
    """Unit vector the blow task's fan blows along, in world xy.

    Read off the fan's ``rot``, which is the heading the wind force is
    applied along (``FanComponent._apply_wind_force`` rotates local +x
    by the fan body's orientation, and the body is posed at ``rot``).
    The generated task's fan has rot 0 and blows +x; the real bench's
    blows wherever the cameras saw it pointing. Falls back to +x when
    the state has no fan, which only a unit test's stub state does.
    """
    fans = [o for o in state if o.type.name == "fan"]
    if not fans:
        return (1.0, 0.0)
    yaw = float(state.get(fans[0], "rot"))
    return (float(np.cos(yaw)), float(np.sin(yaw)))


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
    # A quadratic least-squares fit to the measured curve over the
    # 30-step gust, from 1.5 to 3.5 N (12.02 / 14.35 / 16.33 / 20.33 /
    # 26.36 cm). Cheaper and clearer than shipping a table, and it
    # extrapolates sensibly for a task generator that varies the force.
    force = CFG.domino_blow_wind_force
    return max(0.0, 0.02691 * force * force - 0.06525 * force + 0.16024)


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
        """The block is somewhere the gust can still deliver it.

        A CORRIDOR, from one slide-length upwind of the patch through to
        the patch's far edge, rather than the single point the placement
        aims at. Written as a point it was true only at the instant of
        release: the wind then moved the block, the atom flipped, and
        the Wait terminated after three steps with "atom change during
        Wait" - killing the gust a twentieth of the way through its own
        flight. It has to stay true while the thing it describes is
        happening.

        Loosening it costs nothing, because it is not what makes the
        task hard. The goal demands the block end up FLAT in the patch,
        and no placement anywhere in this corridor achieves that on its
        own.
        """
        domino, region = objects
        if state.get(domino, "is_held") > 0.5:
            return False
        half_x = float(state.get(region, "half_x"))
        half_y = float(state.get(region, "half_y"))
        # The block travels from upwind toward the patch, along the
        # fan's axis: ``along`` is its offset from the patch centre in
        # that direction, ``across`` the offset off the axis. Half a
        # patch of slack at the upwind end is the placement tolerance;
        # the far edge closes the corridor.
        dx, dy = blow_wind_dir(state)
        rx = float(state.get(domino, "x")) - float(state.get(region, "x"))
        ry = float(state.get(domino, "y")) - float(state.get(region, "y"))
        along = rx * dx + ry * dy
        across = abs(-rx * dy + ry * dx)
        lo = -_blow_slide_distance() - half_x
        return lo <= along <= half_x and across <= half_y

    return {Predicate("ReadyToBlow", [domino_type, region_type], _ready_holds)}
