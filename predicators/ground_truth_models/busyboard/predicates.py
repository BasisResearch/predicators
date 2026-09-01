"""Helper predicates exposing the busyboard's wiring to the oracle planner.

A lamp's drive condition is the thing the domain hides, so it cannot be
a feature of any State. But a symbolic planner has to know it to plan at
all, and the codebase's designated channel for exactly that is a
ground-truth HELPER predicate: only oracle / process-planning approaches
receive these (``get_gt_helper_predicates``), and agent approaches never
see them.

The classifiers reconstruct the wiring rather than being told it. Under
``CFG.busyboard_fixed_wiring`` the wiring is a pure function of the board
size (``canonical_wiring``), and the board size is readable off any state
by counting buttons and lamps - so these predicates need no privileged
channel into the env, and the planner and the env cannot drift apart.
With per-task wiring that reconstruction is not available, and the
factory withdraws rather than silently answering with the wrong board's
wiring.

Two predicates, because a drive condition is either a single button or a
conjunction of two, and a lifted process cannot branch on which:

* ``SoleDriver(?button, ?lamp)`` - the lamp needs exactly this button.
* ``JointDrivers(?b1, ?b2, ?lamp)`` - the lamp needs both, with ``?b1``
  the lower-indexed one. Conjunction is symmetric, so emitting only the
  canonical order keeps one ground atom per real hypothesis instead of
  two spurious ones.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type


def _board_wiring(state: State) -> Optional[List[Tuple[int, int]]]:
    """This board's (driver, enabler) per lamp, or None if unavailable."""
    if not CFG.busyboard_fixed_wiring:
        return None
    from predicators.envs.pybullet_busyboard import \
        canonical_wiring  # pylint: disable=import-outside-toplevel
    num_buttons = sum(1 for o in state if o.type.name == "button")
    num_lamps = sum(1 for o in state if o.type.name == "lamp")
    if num_buttons == 0 or num_lamps == 0:
        return None
    driver, enabler = canonical_wiring(num_buttons, num_lamps)
    return list(zip(driver, enabler))


def _index(obj: Object) -> int:
    """Board position of a ``button<N>`` / ``lamp<N>`` object."""
    return int(obj.name[len(obj.type.name):])


def _sole_driver_holds(state: State, objects: Sequence[Object]) -> bool:
    """Whether ``lamp`` is driven by ``button`` alone."""
    button, lamp = objects
    wiring = _board_wiring(state)
    if wiring is None:
        return False
    from predicators.envs.pybullet_busyboard import \
        NO_ENABLER  # pylint: disable=import-outside-toplevel
    lamp_idx = _index(lamp)
    if lamp_idx >= len(wiring):
        return False
    driver, enabler = wiring[lamp_idx]
    return enabler == NO_ENABLER and _index(button) == driver


def _joint_drivers_holds(state: State, objects: Sequence[Object]) -> bool:
    """Whether ``lamp`` needs both buttons, lower index first."""
    button_a, button_b, lamp = objects
    wiring = _board_wiring(state)
    if wiring is None:
        return False
    from predicators.envs.pybullet_busyboard import \
        NO_ENABLER  # pylint: disable=import-outside-toplevel
    lamp_idx = _index(lamp)
    if lamp_idx >= len(wiring):
        return False
    driver, enabler = wiring[lamp_idx]
    if enabler == NO_ENABLER:
        return False
    lo, hi = min(driver, enabler), max(driver, enabler)
    return (_index(button_a), _index(button_b)) == (lo, hi)


class PyBulletBusyBoardGroundTruthPredicateFactory(GroundTruthPredicateFactory
                                                   ):
    """Wiring helper predicates for the busyboard environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_busyboard"}

    @classmethod
    def get_helper_predicates(cls, env_name: str,
                              types: Dict[str, Type]) -> Set[Predicate]:
        """The wiring predicates, for oracle approaches only."""
        del env_name  # unused
        button_type = types["button"]
        lamp_type = types["lamp"]

        SoleDriver = Predicate(
            "SoleDriver", [button_type, lamp_type],
            _sole_driver_holds,
            natural_language_assertion=lambda os:
            f"lamp {os[1]} is driven by {os[0]} alone")
        JointDrivers = Predicate(
            "JointDrivers", [button_type, button_type, lamp_type],
            _joint_drivers_holds,
            natural_language_assertion=lambda os:
            f"lamp {os[2]} needs both {os[0]} and {os[1]}")
        return {SoleDriver, JointDrivers}
