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

Static wiring facts, one per role a button can play for a lamp:

* ``Drives(?button, ?lamp)`` - the button is the lamp's driver;
* ``Enables(?button, ?lamp)`` - the button is one of its enablers;
* ``Inhibits(?button, ?lamp)`` - the button is its inhibitor.

Hidden dynamic state the process model needs:

* ``Armed(?lamp)`` / ``Disarmed(?lamp)`` - the lamp's arming latch,
  read from the fully-observable ``armed`` feature (the oracle runs the
  board fully observed; absent the feature the lamp is taken as
  disarmed, which only ever makes a plan re-press a driver);
* ``JustPressed(?button)`` - a one-tick pulse the model's PressButton
  process emits so that arming can be keyed to the driver's rising
  edge. It is never true of a real state.

And derived predicates over those, so a lifted process can say "all of
the lamp's enablers are on" without knowing how many there are:
``Enabled``, ``Uninhibited``, ``Driven``, ``Undriven``, the nullary
``Overloaded`` (more lamps driven than the breaker allows) and
``AllButtonsOff`` (what closes a tripped breaker). Every one of them is
MONOTONE in the positive atoms it reads (``ButtonOff`` rather than "not
``ButtonOn``", ``Disarmed`` rather than "not ``Armed``"), because the
planner's reachability analysis is a delete relaxation: a derived
predicate that read a negation would be false wherever its positive
side is reachable, and the goal would look unreachable from the start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set

from predicators.ground_truth_models import GroundTruthPredicateFactory
from predicators.settings import CFG
from predicators.structs import DerivedPredicate, GroundAtom, Object, \
    Predicate, State, Type

if TYPE_CHECKING:
    from predicators.envs.pybullet_busyboard import Condition


def _board_wiring(state: State) -> Optional[List[Condition]]:
    """This board's condition per lamp, or None if unavailable."""
    if not CFG.busyboard_fixed_wiring:
        return None
    from predicators.envs.pybullet_busyboard import \
        canonical_wiring  # pylint: disable=import-outside-toplevel
    num_buttons = sum(1 for o in state if o.type.name == "button")
    num_lamps = sum(1 for o in state if o.type.name == "lamp")
    if num_buttons == 0 or num_lamps == 0:
        return None
    return canonical_wiring(num_buttons, num_lamps)


def _index(obj: Object) -> int:
    """Board position of a ``button<N>`` / ``lamp<N>`` object."""
    return int(obj.name[len(obj.type.name):])


def _condition(state: State, lamp: Object) -> Optional[Condition]:
    wiring = _board_wiring(state)
    if wiring is None:
        return None
    lamp_idx = _index(lamp)
    if lamp_idx >= len(wiring):
        return None
    return wiring[lamp_idx]


def _drives_holds(state: State, objects: Sequence[Object]) -> bool:
    """Whether ``button`` is ``lamp``'s driver."""
    button, lamp = objects
    cond = _condition(state, lamp)
    return cond is not None and _index(button) == cond.driver


def _enables_holds(state: State, objects: Sequence[Object]) -> bool:
    """Whether ``button`` is one of ``lamp``'s enablers."""
    button, lamp = objects
    cond = _condition(state, lamp)
    return cond is not None and _index(button) in cond.enablers


def _inhibits_holds(state: State, objects: Sequence[Object]) -> bool:
    """Whether ``button`` is ``lamp``'s inhibitor."""
    button, lamp = objects
    cond = _condition(state, lamp)
    return cond is not None and _index(button) == cond.inhibitor


def _armed_holds(state: State, objects: Sequence[Object]) -> bool:
    """The lamp's arming latch, when the state carries it."""
    lamp, = objects
    if "armed" not in lamp.type.feature_names:
        return False
    return state.get(lamp, "armed") > 0.5


def _disarmed_holds(state: State, objects: Sequence[Object]) -> bool:
    """The latch's complement, as a positive fact."""
    return not _armed_holds(state, objects)


def _just_pressed_holds(state: State, objects: Sequence[Object]) -> bool:
    """A model-only pulse; never true of a real state."""
    del state, objects
    return False


# ── Derived predicates (over atoms) ──────────────────────────────


def _named(atoms: Iterable[GroundAtom], name: str) -> List[GroundAtom]:
    return [a for a in atoms if a.predicate.name == name]


def _holds(atoms: Set[GroundAtom], name: str, *objects: Object) -> bool:
    return any(a.objects == list(objects) for a in _named(atoms, name))


def _roles(atoms: Set[GroundAtom], name: str, lamp: Object) -> List[Object]:
    """The buttons in role ``name`` (Drives / Enables / Inhibits) for
    ``lamp``."""
    return [a.objects[0] for a in _named(atoms, name) if a.objects[1] == lamp]


def _enabled(atoms: Set[GroundAtom], lamp: Object) -> bool:
    return all(
        _holds(atoms, "ButtonOn", b) for b in _roles(atoms, "Enables", lamp))


def _uninhibited(atoms: Set[GroundAtom], lamp: Object) -> bool:
    return all(
        _holds(atoms, "ButtonOff", b) for b in _roles(atoms, "Inhibits", lamp))


def _driven(atoms: Set[GroundAtom], lamp: Object) -> bool:
    if not _holds(atoms, "Armed", lamp):
        return False
    if not any(
            _holds(atoms, "ButtonOn", b)
            for b in _roles(atoms, "Drives", lamp)):
        return False
    if not _named(atoms, "BreakerClosed"):
        return False
    return _enabled(atoms, lamp) and _uninhibited(atoms, lamp)


def _undriven(atoms: Set[GroundAtom], lamp: Object) -> bool:
    """The positive spelling of "not driven": some part of the drive is visibly
    missing."""
    if _holds(atoms, "Disarmed", lamp) or _named(atoms, "BreakerTripped"):
        return True
    if any(
            _holds(atoms, "ButtonOff", b)
            for b in _roles(atoms, "Drives", lamp)):
        return True
    if any(
            _holds(atoms, "ButtonOff", b)
            for b in _roles(atoms, "Enables", lamp)):
        return True
    return any(
        _holds(atoms, "ButtonOn", b) for b in _roles(atoms, "Inhibits", lamp))


def _enabled_holds(atoms: Set[GroundAtom], objects: Sequence[Object]) -> bool:
    lamp, = objects
    return _enabled(atoms, lamp)


def _uninhibited_holds(atoms: Set[GroundAtom],
                       objects: Sequence[Object]) -> bool:
    lamp, = objects
    return _uninhibited(atoms, lamp)


def _driven_holds(atoms: Set[GroundAtom], objects: Sequence[Object]) -> bool:
    lamp, = objects
    return _driven(atoms, lamp)


def _undriven_holds(atoms: Set[GroundAtom], objects: Sequence[Object]) -> bool:
    lamp, = objects
    return _undriven(atoms, lamp)


def _overloaded_holds(atoms: Set[GroundAtom],
                      objects: Sequence[Object]) -> bool:
    del objects
    limit = int(CFG.busyboard_breaker_limit)
    if limit <= 0:
        return False
    lamps = {a.objects[1] for a in _named(atoms, "Drives")}
    return sum(1 for lamp in lamps if _driven(atoms, lamp)) > limit


def _all_buttons_off_holds(atoms: Set[GroundAtom],
                           objects: Sequence[Object]) -> bool:
    del objects
    # Every button carries ButtonOn or ButtonOff at all times, so the
    # two together enumerate the board's buttons.
    buttons = {a.objects[0] for a in _named(atoms, "ButtonOn")}
    buttons |= {a.objects[0] for a in _named(atoms, "ButtonOff")}
    return all(_holds(atoms, "ButtonOff", b) for b in buttons)


def _never(state: State, objects: Sequence[Object]) -> bool:
    """Classifier of a stand-in predicate that is never evaluated."""
    del state, objects
    return False


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
        breaker_type = types["breaker"]

        Drives = Predicate("Drives", [button_type, lamp_type],
                           _drives_holds,
                           natural_language_assertion=lambda os:
                           f"{os[0]} is the driver of lamp {os[1]}")
        Enables = Predicate("Enables", [button_type, lamp_type],
                            _enables_holds,
                            natural_language_assertion=lambda os:
                            f"{os[0]} must be on for lamp {os[1]} to respond")
        Inhibits = Predicate(
            "Inhibits", [button_type, lamp_type],
            _inhibits_holds,
            natural_language_assertion=lambda os:
            f"{os[0]} must be off for lamp {os[1]} to respond")
        Armed = Predicate(
            "Armed", [lamp_type],
            _armed_holds,
            natural_language_assertion=lambda os: f"lamp {os[0]} is armed")
        Disarmed = Predicate(
            "Disarmed", [lamp_type],
            _disarmed_holds,
            natural_language_assertion=lambda os: f"lamp {os[0]} is not armed")
        JustPressed = Predicate("JustPressed", [button_type],
                                _just_pressed_holds,
                                natural_language_assertion=lambda os:
                                f"{os[0]} was pressed on this tick")
        # The env's own predicates the derived ones read. A Predicate is
        # equal to another by name and types, so these stand-ins key the
        # planner's dependency index to the env's atoms without this
        # module holding the env's objects; they are never evaluated.
        ButtonOn = Predicate("ButtonOn", [button_type], _never)
        ButtonOff = Predicate("ButtonOff", [button_type], _never)
        BreakerClosed = Predicate("BreakerClosed", [breaker_type], _never)
        BreakerTripped = Predicate("BreakerTripped", [breaker_type], _never)

        drive_facts = {
            Drives, Enables, Inhibits, Armed, Disarmed, ButtonOn, ButtonOff,
            BreakerClosed, BreakerTripped
        }
        Enabled = DerivedPredicate("Enabled", [lamp_type],
                                   _enabled_holds,
                                   auxiliary_predicates={Enables, ButtonOn})
        Uninhibited = DerivedPredicate(
            "Uninhibited", [lamp_type],
            _uninhibited_holds,
            auxiliary_predicates={Inhibits, ButtonOff})
        Driven = DerivedPredicate("Driven", [lamp_type],
                                  _driven_holds,
                                  auxiliary_predicates=set(drive_facts))
        Undriven = DerivedPredicate("Undriven", [lamp_type],
                                    _undriven_holds,
                                    auxiliary_predicates=set(drive_facts))
        Overloaded = DerivedPredicate("Overloaded", [],
                                      _overloaded_holds,
                                      auxiliary_predicates=set(drive_facts))
        AllButtonsOff = DerivedPredicate(
            "AllButtonsOff", [],
            _all_buttons_off_holds,
            auxiliary_predicates={ButtonOn, ButtonOff})
        return {
            Drives, Enables, Inhibits, Armed, Disarmed, JustPressed, Enabled,
            Uninhibited, Driven, Undriven, Overloaded, AllButtonsOff
        }
