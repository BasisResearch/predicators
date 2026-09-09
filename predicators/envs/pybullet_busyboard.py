"""The busyboard environment: hidden wiring, latent charge, tasks, predicates.

The observable simulation core (board geometry, body construction,
state read/write, button mechanics) lives in
:mod:`predicators.envs.pybullet_busyboard_base`, which may be surfaced
to learning agents as reference source. This module holds everything an
agent must LEARN or must not see:

* the WIRING - each lamp's drive condition: the button that drives it,
  up to two further buttons that must also be on (its enablers), and
  optionally one button that must be OFF (its inhibitor);
* the ARMING LATCH - a lamp only responds to its driver if the driver
  was pressed while its enablers were already on;
* the BREAKER - driving more lamps at once than the breaker allows
  trips it, which kills every lamp until the board is power-cycled;
* the CHARGE dynamics (``_domain_specific_step`` and its constants) -
  how long a lamp must be driven before it lights, and how fast it
  fades once the drive is removed;
* task generation (the train/test distribution) and goal semantics.

Design notes, and how this domain differs from the rest of the suite.

**Discrete structure, not just parameters.** Every other domain in the
suite (fan wind, boil heating, bridge curing, domino friction) fixes
the *form* of the hidden process and leaves a handful of real-valued
knobs to identify. Here the form itself is unknown: a model of this
board is a branching program over a wiring relation, and no setting of
a continuous parameter vector expresses it. A learner that can only fit
scalars cannot represent this domain's answer at all.

**Conjunctive drives (many-to-one).** A lamp may require up to THREE
buttons at once: its ``driver`` and one or two ``enablers``. Prior
busyboard-style environments for robot learning exclude many-to-one
relations specifically to keep the relations identifiable from
undirected play (Liu et al., CoRL 2022, "BusyBot", Sec. 3: "We exclude
many-to-one relations to eliminate possible ambiguities"). Including
them is the point: an interlock is exactly the structure that a passive
observer confounds and a well-chosen experiment separates. An
``inhibitor`` is the dual: a button that must stay OFF for the lamp to
respond, so that "press everything" is never a solution and a button
the agent knows nothing about is a hazard rather than a decoy.

**The arming latch (order matters).** A conjunctive lamp is *armed* on
the rising edge of its driver, and only if every enabler is already on
at that moment; it is disarmed when the driver goes off. Only an armed
lamp charges. So pressing the driver first and the enablers afterwards
lights nothing until the driver is released and pressed again, while
the same buttons pressed in the other order light the lamp. The board
is therefore not a function of the button setting: two identical
settings can behave differently, and telling them apart needs a model
with state, not a table of settings.

**The breaker.** Driving more than ``busyboard_breaker_limit`` lamps at
the same time trips the breaker: every charge drops to zero, nothing
charges until every button has been released (a power cycle), and the
breaker tile on the board turns red. Undirected play that latches
buttons broadly trips it; a plan that knows which buttons feed which
lamps never does.

**A latent that delays the evidence.** A lamp does not respond to its
drive immediately. It accumulates hidden ``charge`` while driven, and
lights only once the charge crosses an onset - by which time the robot
has typically pressed something else. So a naive "press it and see"
policy systematically mis-attributes causes, and telling two candidate
wirings apart requires *designing* the interaction (press one button,
hold, wait), not merely covering the button set. The charge is never
observable. What is observable is ``brightness``, which stays flat at
zero through the early accumulation and only then ramps - the same
latent-plus-monotone-readout shape as boil's ``bubbling_level``, so the
rate is recoverable from the ramp while the onset stays hidden.

**Training wiring extends to test.** The hidden wiring is fixed for a
run and the test boards *extend* the training board: every lamp the
agent trained on keeps its drive condition, and what a test board adds
is buttons and lamps. The added lamps draw their drive from the added
buttons, and an added button may be the inhibitor of a lamp the agent
trained on. So the relation learned in training is true at test as long
as the new buttons stay off, the way glue chemistry or fan thrust is in
the other domains; what test asks is whether the agent can light a lamp
it never saw (``busyboard_test_extension_lit``) without tripping a
hazard among buttons it never saw. Test goals also ask for at least two
lamps lit (``busyboard_min_lit_test``), so they need two learned
conditions composed rather than one training goal repeated.

Example commands::

    # Watch a board with the GUI, no agent.
    python predicators/envs/pybullet_busyboard.py

    # Oracle demo via bilevel process planning.
    python predicators/main.py --env pybullet_busyboard \
        --approach oracle_process_planning --seed 0 \
        --num_train_tasks 0 --num_test_tasks 10 \
        --sesame_check_expected_atoms False

    # Learning run (partially observable).
    python predicators/main.py --env pybullet_busyboard \
        --approach agent_sim_predicate_invention --seed 0 \
        --partially_observable True

``--sesame_check_expected_atoms False`` is required for the same reason
``pybullet_bridge`` needs it. A lamp's lighting is a delayed effect of a
drive condition, so the exact tick on which it lands depends on how many
low-level steps the surrounding options happen to take. The symbolic
delay places it on one tick; physics may deliver it on the one before or
after, and the per-step atom check then rejects a plan that reaches the
goal.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_busyboard_base import PyBulletBusyBoardBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

# Sentinel for "no button" in a condition's enabler or inhibitor slots and
# in the flat float encoding carried on EnvironmentTask.offline_task_metrics.
NO_ENABLER: int = -1
NO_BUTTON: int = NO_ENABLER
# Most enablers a condition may have (three-input conditions at most).
MAX_ENABLERS: int = 2


@dataclass(frozen=True, order=True)
class Condition:
    """One lamp's drive condition.

    The lamp charges while ``driver`` and every button in ``enablers``
    are on, ``inhibitor`` (if any) is off, and the lamp is armed: its
    driver was last pressed while the enablers were already on.
    """
    driver: int
    enablers: Tuple[int, ...] = ()
    inhibitor: int = NO_BUTTON

    @property
    def drive_set(self) -> Tuple[int, ...]:
        """The driver and the enablers, driver first."""
        return (self.driver, ) + self.enablers

    @property
    def buttons(self) -> Tuple[int, ...]:
        """Every button the condition mentions."""
        if self.inhibitor == NO_BUTTON:
            return self.drive_set
        return self.drive_set + (self.inhibitor, )

    def drive_key(self) -> Tuple[int, Tuple[int, ...]]:
        """What makes two conditions the same drive: driver and enablers.

        Two lamps with the same drive are indistinguishable by any
        experiment (the inhibitor can only ever keep a lamp dark), so
        this is the identity the sampler keeps distinct.
        """
        return (self.driver, self.enablers)

    def canonical(self) -> "Condition":
        """Sorted, deduplicated enablers; no button in two roles."""
        enablers = tuple(
            sorted({
                e
                for e in self.enablers if e not in (NO_BUTTON, self.driver)
            }))
        inhibitor = self.inhibitor
        if inhibitor == self.driver or inhibitor in enablers:
            inhibitor = NO_BUTTON
        return Condition(self.driver, enablers, inhibitor)

    def to_metrics(self, index: int) -> Dict[str, float]:
        """Flatten into the float-valued metrics dict, as lamp ``index``."""
        enablers = list(self.enablers) + [NO_BUTTON] * MAX_ENABLERS
        return {
            f"wiring_driver_{index}": float(self.driver),
            f"wiring_enabler_{index}": float(enablers[0]),
            f"wiring_enabler2_{index}": float(enablers[1]),
            f"wiring_inhibitor_{index}": float(self.inhibitor),
        }

    @classmethod
    def from_metrics(cls, metrics: Dict[str, float],
                     index: int) -> "Condition":
        """Inverse of :meth:`to_metrics`."""
        enablers = tuple(
            int(metrics.get(key, NO_BUTTON))
            for key in (f"wiring_enabler_{index}", f"wiring_enabler2_{index}"))
        return cls(int(metrics[f"wiring_driver_{index}"]), enablers,
                   int(metrics.get(f"wiring_inhibitor_{index}",
                                   NO_BUTTON))).canonical()

    def __str__(self) -> str:
        text = f"b{self.driver}"
        if self.enablers:
            text += " & " + " & ".join(f"b{e}" for e in self.enablers)
        if self.inhibitor != NO_BUTTON:
            text += f" & not b{self.inhibitor}"
        return text


def canonical_pair(driver: int, enabler: int) -> Tuple[int, int]:
    """Put a one-enabler drive condition in canonical form.

    Kept for callers that still think in (driver, enabler) pairs. The
    order of a driver and its enabler IS observable now (the arming
    latch), so this no longer sorts the two; it only normalises the "no
    enabler" spellings.
    """
    if enabler in (NO_ENABLER, driver):
        return (driver, NO_ENABLER)
    return (driver, enabler)


def legal_conditions(num_buttons: int,
                     max_enablers: int = MAX_ENABLERS) -> List[Condition]:
    """Every distinct drive (driver plus enablers) on a board of this size.

    This list is the domain's discrete hypothesis space per lamp, before
    the inhibitor: ``num_buttons`` choices of driver times the subsets
    of at most ``max_enablers`` other buttons. For a 4-button board that
    is 4 * (1 + 3 + 3) = 28 drives; for an 8-button board 8 * (1 + 7 +
    21) = 232, times up to 6 inhibitor choices each. The driver is
    distinguished from the enablers because the arming latch makes the
    order observable: a lamp needing button 1 pressed after button 2 is
    a different lamp from one needing button 2 pressed after button 1.
    """
    conditions = []
    for driver in range(num_buttons):
        others = [b for b in range(num_buttons) if b != driver]
        for k in range(0, min(max_enablers, len(others)) + 1):
            for enablers in combinations(others, k):
                conditions.append(Condition(driver, tuple(enablers)))
    return conditions


def legal_pairs(num_buttons: int) -> List[Tuple[int, int]]:
    """The one-enabler slice of :func:`legal_conditions`, as pairs."""
    return [(c.driver, c.enablers[0] if c.enablers else NO_ENABLER)
            for c in legal_conditions(num_buttons, max_enablers=1)]


def _nearest_first(candidates: List[Condition],
                   driver: int) -> List[Condition]:
    """``candidates`` with those keeping ``driver`` first, then in order."""
    return sorted(candidates, key=lambda c: (c.driver != driver, c))


def _dedupe_wiring(wiring: Sequence[Condition],
                   num_buttons: int) -> List[Condition]:
    """Force every lamp's drive to be distinct.

    Two lamps wired to the same drive are indistinguishable by
    construction: no experiment separates them and no goal can ask for
    one lit and the other dark, which would make the board unsolvable
    for reasons that have nothing to do with inference. A colliding lamp
    moves to the nearest unused drive with the same number of enablers
    (the same driver where possible), and keeps its inhibitor unless
    the new drive uses that button.
    """
    legal = legal_conditions(num_buttons)
    used: Set[Tuple[int, Tuple[int, ...]]] = set()
    out: List[Condition] = []
    for cond in wiring:
        cond = cond.canonical()
        if cond.drive_key() in used:
            candidates = [
                c for c in legal if c.drive_key() not in used
                and len(c.enablers) == len(cond.enablers)
            ]
            candidates = _nearest_first(candidates, cond.driver)
            if not candidates:
                candidates = [c for c in legal if c.drive_key() not in used]
            if not candidates:
                raise RuntimeError(
                    f"A {num_buttons}-button board has only {len(legal)} "
                    f"distinct drives, fewer than the number of lamps "
                    f"requested.")
            cond = Condition(candidates[0].driver, candidates[0].enablers,
                             cond.inhibitor).canonical()
        used.add(cond.drive_key())
        out.append(cond)
    return out


def core_board() -> Tuple[int, int]:
    """The smallest training board: (num_buttons, num_lamps).

    The core board is the part of every board that training reveals. Its
    lamps are the *core lamps* and its buttons the *core buttons*; every
    other lamp and button a board may carry is an *extension*.
    """
    return (min(CFG.busyboard_num_buttons_train),
            min(CFG.busyboard_num_lamps_train))


def _remap_extension(index: int, core_buttons: int, num_buttons: int) -> int:
    """Map a max-board extension button onto a smaller board.

    Extension buttons are the indices at or above ``core_buttons``. On a
    board with fewer of them the index folds into the extension range
    that board does have, so an extension lamp still draws its drive
    from a button training never showed. A board with no extension
    buttons at all has nowhere else to put it and folds into the core.
    """
    if num_buttons > core_buttons:
        return core_buttons + (index - core_buttons) % (num_buttons -
                                                        core_buttons)
    return index % num_buttons


def project_wiring(wiring_full: Sequence[Condition], num_buttons: int,
                   num_lamps: int) -> List[Condition]:
    """Project a max-board wiring onto a board of the requested size.

    The projection is an EXTENSION: the core board's wiring is a subset
    of every larger board's wiring. Concretely, for the first
    ``num_lamps`` lamps:

    * a core lamp keeps its drive verbatim, which is well defined
      because ``canonical_wiring`` wires core lamps to core buttons
      only; its inhibitor, when it is an extension button, folds into
      whatever extension buttons this board has and is dropped on a
      board that has none;
    * an extension lamp's drive folds into the extension buttons this
      board has (see ``_remap_extension``), core buttons staying put.

    Then the ways folding can degrade a condition are repaired: an
    enabler landing on its driver or on another enabler, an inhibitor
    landing inside the drive, and two lamps landing on the same drive.

    So a rule learned about a core lamp on the training board is true of
    that lamp on every test board while the added buttons stay off. This
    is a pure function of the max-board wiring and the board size, which
    is what lets ONE parameter vector be a correct model of every board
    in a run: the ground-truth simulator carries the max-board wiring in
    its params and applies this same projection to whatever board the
    observation shows it.
    """
    core_buttons, core_lamps = core_board()
    core_buttons = min(core_buttons, num_buttons)

    def _fold(index: int, is_core_lamp: bool) -> int:
        if is_core_lamp or index < core_buttons:
            return index % core_buttons
        return _remap_extension(index, core_buttons, num_buttons)

    out: List[Condition] = []
    for i in range(num_lamps):
        full = wiring_full[i]
        is_core = i < core_lamps
        limit = core_buttons if is_core else num_buttons
        driver = _fold(full.driver, is_core)
        enablers: List[int] = []
        for e in full.enablers:
            if len(enablers) + 2 > limit:
                break  # no room for another distinct enabler
            e = _fold(e, is_core)
            while e == driver or e in enablers:
                e = (e + 1) % limit
            enablers.append(e)
        inhibitor = full.inhibitor
        if inhibitor != NO_BUTTON and inhibitor >= num_buttons:
            if num_buttons > core_buttons:
                inhibitor = _remap_extension(inhibitor, core_buttons,
                                             num_buttons)
            elif is_core:
                inhibitor = NO_BUTTON
            else:
                inhibitor = inhibitor % num_buttons
        if inhibitor == driver or inhibitor in enablers:
            inhibitor = NO_BUTTON
        out.append(Condition(driver, tuple(sorted(enablers)), inhibitor))
    return _dedupe_wiring(out, num_buttons)


def _draw_full_wiring(rng: np.random.Generator, max_buttons: int,
                      max_lamps: int) -> List[Condition]:
    """One draw of the max-board wiring under the extension contract."""
    core_buttons, core_lamps = core_board()
    extension = list(range(core_buttons, max_buttons))
    wiring: List[Condition] = []
    for i in range(max_lamps):
        is_core = i < core_lamps
        # A core lamp lives entirely on the core board; an extension lamp
        # is driven by a button the core board does not have (falling
        # back to any button when the distribution adds lamps but no
        # buttons), and may take its enablers from anywhere.
        if is_core:
            driver_pool = list(range(core_buttons))
            enabler_pool = list(range(core_buttons))
        else:
            driver_pool = extension or list(range(max_buttons))
            enabler_pool = list(range(max_buttons))
        driver = int(rng.choice(driver_pool))
        pool = [b for b in enabler_pool if b != driver]
        k = 0
        if pool and rng.random() < CFG.busyboard_interlock_prob:
            k = 1
            if len(pool) >= 2 and \
                    rng.random() < CFG.busyboard_double_enabler_prob:
                k = 2
        enablers = tuple(
            sorted(
                int(b)
                for b in rng.choice(pool, size=k, replace=False))) if k else ()
        drive = {driver, *enablers}
        inhibitor = NO_BUTTON
        if is_core and extension and \
                rng.random() < CFG.busyboard_extension_inhibitor_prob:
            # The hazard a test board adds: a button training never
            # showed that must stay off for a lamp training did show.
            inhibitor = int(rng.choice(extension))
        else:
            free = [
                b for b in (
                    range(core_buttons) if is_core else range(max_buttons))
                if b not in drive
            ]
            if free and rng.random() < CFG.busyboard_inhibitor_prob:
                inhibitor = int(rng.choice(free))
        wiring.append(Condition(driver, enablers, inhibitor))
    return wiring


def _board_sizes(train: bool) -> List[Tuple[int, int]]:
    buttons = CFG.busyboard_num_buttons_train if train else \
        CFG.busyboard_num_buttons_test
    lamps = CFG.busyboard_num_lamps_train if train else \
        CFG.busyboard_num_lamps_test
    return [(int(b), int(l)) for b in buttons for l in lamps]


def _lit_candidates(num_lamps: int, train: bool) -> int:
    """How many of a board's lamps a goal may ask to be lit.

    Only core lamps in training and, unless
    ``busyboard_test_extension_lit``, at test too: an extension lamp is
    then only ever asked to stay dark.
    """
    if not CFG.busyboard_fixed_wiring:
        return num_lamps
    if not train and CFG.busyboard_test_extension_lit:
        return num_lamps
    return min(num_lamps, core_board()[1])


def _wiring_supports_distribution(wiring_full: Sequence[Condition]) -> bool:
    """Whether every board size of the run has a non-trivial goal."""
    for train in (True, False):
        min_lit = int(CFG.busyboard_min_lit_train if train else CFG.
                      busyboard_min_lit_test)
        for num_buttons, num_lamps in _board_sizes(train):
            wiring = tuple(project_wiring(wiring_full, num_buttons, num_lamps))
            if not realizable_targets(wiring, num_buttons,
                                      _lit_candidates(num_lamps, train),
                                      min_lit):
                return False
    return True


def _wiring_cache_key() -> Tuple[Any, ...]:
    return (
        int(CFG.seed),
        int(CFG.busyboard_wiring_salt),
        tuple(CFG.busyboard_num_buttons_train),
        tuple(CFG.busyboard_num_buttons_test),
        tuple(CFG.busyboard_num_lamps_train),
        tuple(CFG.busyboard_num_lamps_test),
        int(CFG.busyboard_min_lit_train),
        int(CFG.busyboard_min_lit_test),
        float(CFG.busyboard_interlock_prob),
        float(CFG.busyboard_double_enabler_prob),
        float(CFG.busyboard_inhibitor_prob),
        float(CFG.busyboard_extension_inhibitor_prob),
        int(CFG.busyboard_breaker_limit),
        bool(CFG.busyboard_latch),
        bool(CFG.busyboard_test_extension_lit),
        int(CFG.busyboard_max_sampling_attempts),
    )


_FULL_WIRING_CACHE: Dict[Tuple[Any, ...], Tuple[Condition, ...]] = {}


def full_wiring() -> Tuple[Condition, ...]:
    """The run's max-board wiring.

    Sampled once per seed over the largest board the task distribution
    can produce, redrawn until every board size of the run admits a
    non-trivial goal (see ``realizable_targets``), and cached for the
    process: the oracle's wiring predicates ask for it on every atom
    evaluation.
    """
    key = _wiring_cache_key()
    cached = _FULL_WIRING_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng([CFG.seed, int(CFG.busyboard_wiring_salt)])
    max_buttons = max(
        list(CFG.busyboard_num_buttons_train) +
        list(CFG.busyboard_num_buttons_test))
    max_lamps = max(
        list(CFG.busyboard_num_lamps_train) +
        list(CFG.busyboard_num_lamps_test))
    for _ in range(int(CFG.busyboard_max_sampling_attempts)):
        wiring = tuple(_draw_full_wiring(rng, max_buttons, max_lamps))
        if _wiring_supports_distribution(wiring):
            _FULL_WIRING_CACHE[key] = wiring
            return wiring
    raise RuntimeError(
        f"No wiring found in {CFG.busyboard_max_sampling_attempts} draws "
        "that gives every board size of the run a non-trivial goal.")


def canonical_wiring(num_buttons: int, num_lamps: int) -> List[Condition]:
    """The run's wiring, reduced to a board of the requested size.

    ``full_wiring`` projected by extension (see ``project_wiring``). The
    draw respects the extension contract: core lamps are wired to core
    buttons only, and an extension lamp's driver is an extension button,
    so the core board's wiring is literally a sub-relation of every
    larger board's. So one wiring describes every board in a run, and
    what an agent learns about the lamps on a 4-button training board
    stays true of those lamps on an 8-button test board while the added
    buttons stay off.

    Why a run-level constant rather than a fresh draw per task: the
    residual-simulator contract in this codebase resolves ``PARAM_SPECS``
    once, after CFG is final and before any task is chosen, so a hidden
    quantity that varied per task would have no home in a fitted model.
    Both the env and the ground-truth simulator call this function, so
    they agree on the answer without either reading the other.
    """
    return project_wiring(full_wiring(), num_buttons, num_lamps)


def press_sequence_outcome(wiring: Sequence[Condition], order: Sequence[int],
                           latch: bool) -> Tuple[Tuple[bool, ...], int]:
    """Which lamps are driven after pressing ``order`` from all-off, and the
    most lamps driven at once along the way.

    Every button in ``order`` is pressed once, in that order, and held.
    With the latch a lamp is armed only if its driver is pressed after
    every one of its enablers; without it the driver's state alone
    counts. This is the model the goal sampler and the docs' solving
    sequences share with the env's dynamics.
    """
    on: Set[int] = set()
    armed = [False] * len(wiring)
    max_driven = 0
    driven: List[bool] = [False] * len(wiring)
    for button in order:
        on.add(button)
        for i, cond in enumerate(wiring):
            if cond.driver == button:
                armed[i] = all(e in on for e in cond.enablers) or not latch
        driven = [
            armed[i] and cond.driver in on and all(e in on
                                                   for e in cond.enablers)
            and cond.inhibitor not in on for i, cond in enumerate(wiring)
        ]
        max_driven = max(max_driven, sum(driven))
    return tuple(driven), max_driven


@lru_cache(maxsize=None)
def _realizable_targets_cached(wiring: Tuple[Condition, ...], num_buttons: int,
                               num_lit_candidates: int, min_lit: int,
                               breaker_limit: int,
                               latch: bool) -> Tuple[Tuple[bool, ...], ...]:
    relevant = sorted({b for c in wiring for b in c.buttons})
    relevant = [b for b in relevant if 0 <= b < num_buttons]
    targets = set()
    for mask in range(1 << len(relevant)):
        on = [relevant[j] for j in range(len(relevant)) if mask >> j & 1]
        # Only the order of the buttons some drivable lamp needs can
        # change which lamps end up armed; every other pressed button
        # (an inhibitor, a button of a lamp this setting cannot drive)
        # goes first, where it can arm nothing.
        drivable = [c for c in wiring if all(b in on for b in c.drive_set)]
        ordered = sorted({b for c in drivable for b in c.drive_set})
        rest = [b for b in on if b not in ordered]
        orders = permutations(ordered) if latch else [tuple(ordered)]
        for order in orders:
            target, max_driven = press_sequence_outcome(
                wiring, rest + list(order), latch)
            if breaker_limit and max_driven > breaker_limit:
                continue
            if sum(target) < min_lit:
                continue
            if len(target) >= 2 and all(target):
                continue
            if any(target[num_lit_candidates:]):
                continue
            targets.add(target)
    return tuple(sorted(targets))


def realizable_targets(wiring: Sequence[Condition],
                       num_buttons: int,
                       num_lit_candidates: Optional[int] = None,
                       min_lit: int = 1) -> List[Tuple[bool, ...]]:
    """Every lamp assignment some press sequence realizes exactly.

    Exhaustive over the button settings and, under the latch, over the
    press orders of the buttons that matter for arming, so goals are
    drawn from the exact set of achievable ones rather than rejection-
    sampled against a solver. A sequence that would trip the breaker on
    its way is not a realization. A target is only kept if it is non-
    trivial: at least ``min_lit`` lamps lit, and with two or more lamps
    at least one that must stay dark. That off-target is what the whole
    domain rests on - it is the reason "latch every button" is not a
    policy, and the reason an agent has to know which button feeds
    which lamp rather than merely which buttons do something.

    Only the first ``num_lit_candidates`` lamps may be asked to be lit
    (default: all of them).
    """
    if num_lit_candidates is None:
        num_lit_candidates = len(wiring)
    return list(
        _realizable_targets_cached(tuple(wiring), int(num_buttons),
                                   int(num_lit_candidates), max(1, min_lit),
                                   int(CFG.busyboard_breaker_limit),
                                   bool(CFG.busyboard_latch)))


class PyBulletBusyBoardEnv(PyBulletBusyBoardBaseEnv):
    """A busy board whose button-to-lamp wiring must be discovered.

    Subclass of the observable sim core (see
    :mod:`predicators.envs.pybullet_busyboard_base`); this class adds the
    hidden wiring, the arming latch, the breaker, the charge dynamics,
    task generation, and predicates.
    """

    # =========================================================================
    # HIDDEN DYNAMICS CONSTANTS
    # =========================================================================
    # Charge accumulates per low-level step while a lamp is driven, and
    # bleeds away faster than it builds - a lamp is slow to light and quick
    # to die, so plans stay short while the evidence stays delayed. The
    # rates are CFG-driven because the useful values depend on how many
    # low-level actions a push skill takes on the robot in use.
    @classmethod
    def charge_rate(cls) -> float:
        """Charge gained per step while driven."""
        return CFG.busyboard_charge_rate

    @classmethod
    def decay_rate(cls) -> float:
        """Charge lost per step while not driven."""
        return CFG.busyboard_decay_rate

    # Observable projection of the hidden charge:
    #   brightness = clip((charge - BRIGHTNESS_ONSET) * BRIGHTNESS_RAMP, 0, 1)
    # Brightness is flat at zero through the whole early accumulation, which
    # is what makes the charge genuinely latent rather than merely rescaled.
    BRIGHTNESS_ONSET: ClassVar[float] = 0.6
    BRIGHTNESS_RAMP: ClassVar[float] = 1.0 / (1.0 - BRIGHTNESS_ONSET)  # 2.5

    # A lamp counts as lit at half brightness, which the ramp reaches
    # strictly after the onset - so LampOn is never true while the lamp is
    # still in its invisible accumulation phase.
    LAMP_ON_THRESHOLD: ClassVar[float] = 0.5

    # =========================================================================
    # TYPES
    # =========================================================================
    # Fully observable: the charge and the arming latch are visible
    # features, so a learner sees the accumulation and the latch directly
    # and only the wiring is hidden.
    _lamp_type_fo = Type(
        "lamp",
        ["x", "y", "z", "rot", "color", "brightness", "charge", "armed"])
    # Partially observable: charge and latch are dropped. The learner sees
    # only the brightness readout and must postulate both.
    _lamp_type_po = Type("lamp", ["x", "y", "z", "rot", "color", "brightness"])

    @classmethod
    def _lamp_type_for_run(cls) -> Type:
        return cls._lamp_type_po if CFG.partially_observable \
            else cls._lamp_type_fo

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Bind the run's lamp type before super().__init__ builds the lamp
        # objects and the predicates off it.
        self._lamp_type = self._lamp_type_for_run()

        # The active task's wiring, one condition per lamp. Installed by
        # reset() and never present in any State - this is the learning
        # target.
        self._wiring: List[Condition] = []
        # How much of the (max-size) board this task actually uses.
        self._num_active_buttons: int = 0
        self._num_active_lamps: int = 0
        # Hidden per-lamp charge and arming latch, keyed by object NAME
        # rather than held in each lamp Object's sim_data. Bilevel planning
        # runs a second env instance built by the option model, and that
        # instance owns its own Object instances: identical in name and
        # type, but with separate sim_data. Charge kept there accumulated
        # on one set of objects while the state was read off the other, so
        # the planner saw a board whose lamps never charged. A name-keyed
        # store is instance-independent, which is what the two env copies
        # need.
        self._charges: Dict[str, float] = {}
        self._armed: Dict[str, bool] = {}
        # The breaker, and the button states at the previous step (the
        # arming latch is edge-triggered on the driver).
        self._tripped: bool = False
        self._prev_button_on: List[bool] = []

        super().__init__(use_gui, **kwargs)

        self._ButtonOn = Predicate(
            "ButtonOn", [self._button_type],
            self._ButtonOn_holds,
            natural_language_assertion=lambda os: f"button {os[0]} is pressed")
        self._ButtonOff = Predicate("ButtonOff", [self._button_type],
                                    self._ButtonOff_holds,
                                    natural_language_assertion=lambda os:
                                    f"button {os[0]} is released")
        self._LampOn = Predicate(
            "LampOn", [self._lamp_type],
            self._LampOn_holds,
            natural_language_assertion=lambda os: f"lamp {os[0]} is lit")
        self._LampOff = Predicate(
            "LampOff", [self._lamp_type],
            self._LampOff_holds,
            natural_language_assertion=lambda os: f"lamp {os[0]} is dark")
        self._BreakerTripped = Predicate(
            "BreakerTripped", [self._breaker_type],
            self._BreakerTripped_holds,
            natural_language_assertion=lambda os:
            f"the breaker {os[0]} has tripped: no lamp can charge until "
            "every button is released")
        self._BreakerClosed = Predicate(
            "BreakerClosed", [self._breaker_type],
            self._BreakerClosed_holds,
            natural_language_assertion=lambda os:
            f"the breaker {os[0]} is closed: the board is powered")
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds,
                                    natural_language_assertion=lambda os:
                                    f"robot {os[0]} is not holding anything")

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_busyboard"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._ButtonOn, self._ButtonOff, self._LampOn, self._LampOff,
            self._BreakerTripped, self._BreakerClosed, self._HandEmpty
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._LampOn, self._LampOff}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._button_type, self._lamp_type,
            self._breaker_type
        }

    # =========================================================================
    # WIRING
    # =========================================================================
    def _install_wiring(self, metrics: Dict[str, float]) -> None:
        """Load a task's wiring from its (experimenter-only) metrics block.

        ``offline_task_metrics`` is the codebase's designated home for
        per-task oracle quantities: it is merged into the per-task
        results by ``main.py`` and is never propagated into the agent-
        facing ``Task``, so recording the true wiring there both keeps
        it away from the learner and makes it available for scoring how
        much of the wiring an agent actually recovered.
        """
        num_lamps = int(metrics.get("wiring_num_lamps", 0))
        self._wiring = [
            Condition.from_metrics(metrics, i) for i in range(num_lamps)
        ]

    @staticmethod
    def _wiring_to_metrics(wiring: Sequence[Condition],
                           num_buttons: int) -> Dict[str, float]:
        """Flatten a wiring into the float-valued metrics dict."""
        metrics: Dict[str, float] = {
            "wiring_num_lamps": float(len(wiring)),
            "wiring_num_buttons": float(num_buttons),
        }
        for i, cond in enumerate(wiring):
            metrics.update(cond.to_metrics(i))
        return metrics

    @staticmethod
    def _driven(button_on: Sequence[bool],
                cond: Condition,
                armed: bool = True) -> bool:
        """Whether a lamp's drive holds under a button assignment.

        The conjunction is the interlock: with an enabler present, the
        driver alone does nothing, which is precisely the many-to-one
        relation that undirected play confounds. The inhibitor is the
        dual, a button that must be off, and ``armed`` is the latch.
        """
        if not armed:
            return False
        if not 0 <= cond.driver < len(button_on) or \
                not button_on[cond.driver]:
            return False
        for e in cond.enablers:
            if not 0 <= e < len(button_on) or not button_on[e]:
                return False
        if cond.inhibitor != NO_BUTTON and \
                0 <= cond.inhibitor < len(button_on) and \
                button_on[cond.inhibitor]:
            return False
        return True

    @classmethod
    def _realizable_targets(cls,
                            wiring: Sequence[Condition],
                            num_buttons: int,
                            num_lit_candidates: Optional[int] = None,
                            min_lit: int = 1) -> List[Tuple[bool, ...]]:
        """See :func:`realizable_targets`."""
        return realizable_targets(wiring, num_buttons, num_lit_candidates,
                                  min_lit)

    def _sample_board(self, num_buttons: int, num_lamps: int,
                      rng: np.random.Generator, min_lit: int,
                      train: bool) -> Tuple[List[Condition], List[bool]]:
        """Pick this task's wiring and a goal assignment it can realize.

        With ``busyboard_fixed_wiring`` (the default) the wiring is the
        run's canonical one at this board size, drawn so that every
        board size has a non-trivial goal; otherwise a fresh wiring is
        drawn per task. Either way the goal is drawn uniformly from the
        exact set of realizable non-trivial assignments. ``min_lit``
        (the split's ``busyboard_min_lit_*``) narrows the draw to
        targets with at least that many lamps lit.
        """
        num_lit_candidates = _lit_candidates(num_lamps, train)
        if CFG.busyboard_fixed_wiring:
            wiring = canonical_wiring(num_buttons, num_lamps)
        else:
            wiring = _dedupe_wiring(
                _draw_full_wiring(rng, num_buttons, num_lamps), num_buttons)

        targets = realizable_targets(wiring, num_buttons, num_lit_candidates,
                                     min_lit)
        if not targets:
            raise RuntimeError(
                f"No non-trivial realizable goal with at least {min_lit} "
                f"lamp(s) lit on a {num_buttons}-button, {num_lamps}-lamp "
                f"board with wiring {[str(c) for c in wiring]}.")
        target = list(targets[int(rng.integers(0, len(targets)))])
        return wiring, target

    # =========================================================================
    # LABELS
    # =========================================================================
    @classmethod
    def button_label(cls, button_idx: int) -> str:
        """How to name a button in text: colour first, then its id."""
        return (f"the {cls.color_name(cls.button_color_index(button_idx))} "
                f"button (button{button_idx})")

    @classmethod
    def lamp_label(cls, lamp_idx: int) -> str:
        """How to name a lamp in text: colour first, then its id."""
        return (f"the {cls.color_name(cls.lamp_color_index(lamp_idx))} "
                f"lamp (lamp{lamp_idx})")

    # =========================================================================
    # STATE READ / WRITE
    # =========================================================================
    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type.name == "button":
            if feature == "is_on":
                return float(self._is_button_on(obj))
            if feature == "color":
                return float(self.button_color_index(self._buttons.index(obj)))
        if obj.type.name == "lamp":
            if feature == "color":
                return float(self.lamp_color_index(self._lamps.index(obj)))
            charge = self._charges.get(obj.name, 0.0)
            if feature == "charge":
                return charge
            if feature == "brightness":
                return self._brightness(charge)
            if feature == "armed":
                return float(self._armed.get(obj.name, False))
        if obj.type.name == "breaker" and feature == "tripped":
            return float(self._tripped)
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Restore button latches, lamp charges, the arming latches and the
        breaker, then repaint the board.

        Also re-derives the wiring from the board in front of it. That
        matters because an env instance is not always driven through
        ``reset``: bilevel planning forward-simulates candidate plans on a
        SEPARATE env built by the option model, which only ever calls
        ``_set_state``. With the wiring installed solely at reset, that
        simulator ran a board wired to nothing, no lamp ever charged in
        the planner's head, and refinement backtracked forever against a
        goal it could not reach. Deriving it here makes any instance
        correct for any state it is handed.
        """
        buttons, lamps = self._active_objects(state)
        self._num_active_buttons = len(buttons)
        self._num_active_lamps = len(lamps)
        if CFG.busyboard_fixed_wiring:
            self._wiring = canonical_wiring(len(buttons), len(lamps))

        button_on = []
        for button in buttons:
            is_on = state.get(button, "is_on") > 0.5
            self._set_button_on(button, is_on)
            button_on.append(is_on)
        # The latch is edge-triggered on the driver, so the restored
        # button states are the baseline the next step's edges are read
        # against.
        self._prev_button_on = button_on

        for i, lamp in enumerate(lamps):
            if "charge" in lamp.type.feature_names:
                charge = float(state.get(lamp, "charge"))
            else:
                # Partially observable: charge is not in the state, so it is
                # inverted from the observable brightness. The inversion is
                # exact above the onset and pins everything below it to the
                # onset itself - the dark band is precisely the information
                # the observation does not carry, and a reset lands at its
                # top so a restored dark lamp is never further along than it
                # looked.
                brightness = float(state.get(lamp, "brightness"))
                charge = self.BRIGHTNESS_ONSET + \
                    brightness / self.BRIGHTNESS_RAMP
            self._charges[lamp.name] = float(np.clip(charge, 0.0, 1.0))
            if "armed" in lamp.type.feature_names:
                self._armed[lamp.name] = state.get(lamp, "armed") > 0.5
            elif i < len(self._wiring) and not (
                    0 <= self._wiring[i].driver < len(button_on)
                    and button_on[self._wiring[i].driver]):
                # A lamp whose driver is off cannot be armed; with the
                # driver on the latch is unobservable and the instance's
                # own memory of it stands.
                self._armed[lamp.name] = False
            self._set_lamp_brightness_visual(
                lamp, self._brightness(self._charges[lamp.name]))

        self._tripped = any(
            state.get(o, "tripped") > 0.5 for o in state
            if o.type.name == "breaker")
        self._set_breaker_visual(self._tripped)
        self._seat_lamp_bases(state, lamps)
        self._park_unused_bodies(len(buttons), len(lamps))

    @classmethod
    def _brightness(cls, charge: float) -> float:
        """Observable readout of the hidden charge (flat, then a ramp)."""
        return float(
            np.clip((charge - cls.BRIGHTNESS_ONSET) * cls.BRIGHTNESS_RAMP, 0.0,
                    1.0))

    # =========================================================================
    # HIDDEN DYNAMICS
    # =========================================================================
    def _domain_specific_step(self) -> None:
        """Arm or disarm each lamp on its driver's edges, trip or reset the
        breaker, accumulate or bleed each lamp's hidden charge, then repaint.

        Skipped when the env is constructed with
        ``skip_residual_dynamics=True`` - that is the base sim the
        learning agent rolls out, and it must show a board on which no
        button ever lights anything.
        """
        buttons = self._buttons[:self._num_active_buttons]
        lamps = self._lamps[:self._num_active_lamps]
        button_on = [self._is_button_on(b) for b in buttons]
        prev = self._prev_button_on
        if len(prev) != len(button_on):
            prev = list(button_on)
        self._prev_button_on = list(button_on)

        # A power cycle - every button released - closes a tripped breaker.
        if self._tripped and not any(button_on):
            self._tripped = False

        driven: List[bool] = []
        for i, lamp in enumerate(lamps):
            if i >= len(self._wiring):
                driven.append(False)
                continue
            cond = self._wiring[i]
            d = cond.driver
            driver_on = 0 <= d < len(button_on) and button_on[d]
            if not driver_on:
                self._armed[lamp.name] = False
            elif not prev[d] or not CFG.busyboard_latch:
                # Rising edge of the driver: armed iff every enabler is
                # already on (always, with the latch ablated).
                self._armed[lamp.name] = all(
                    0 <= e < len(button_on) and button_on[e]
                    for e in cond.enablers)
            driven.append(not self._tripped and self._driven(
                button_on, cond, self._armed.get(lamp.name, False)))

        limit = int(CFG.busyboard_breaker_limit)
        if 0 < limit < sum(driven) and not self._tripped:
            self._tripped = True
            driven = [False] * len(driven)
            for lamp in lamps:
                self._charges[lamp.name] = 0.0
                self._armed[lamp.name] = False

        for i, lamp in enumerate(lamps):
            charge = self._charges.get(lamp.name, 0.0)
            if driven[i]:
                charge = min(1.0, charge + self.charge_rate())
            else:
                charge = max(0.0, charge - self.decay_rate())
            self._charges[lamp.name] = charge
            self._set_lamp_brightness_visual(lamp, self._brightness(charge))
        self._set_breaker_visual(self._tripped)

    def reset(self,
              train_or_test: str,
              task_idx: int,
              render: bool = False) -> Any:
        """Install this task's wiring before any state is applied.

        Redundant under ``busyboard_fixed_wiring``, where
        ``_set_domain_specific_state`` re-derives the same wiring from
        the board; load-bearing when the wiring varies per task, which
        is the only route by which a per-task wiring reaches the env at
        all.
        """
        task = self.get_task(train_or_test, task_idx)
        self._install_wiring(task.offline_task_metrics)
        return super().reset(train_or_test, task_idx, render=render)

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _ButtonOn_holds(state: State, objects: Sequence[Object]) -> bool:
        button, = objects
        return state.get(button, "is_on") > 0.5

    @staticmethod
    def _ButtonOff_holds(state: State, objects: Sequence[Object]) -> bool:
        button, = objects
        return state.get(button, "is_on") <= 0.5

    @classmethod
    def _LampOn_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        lamp, = objects
        return state.get(lamp, "brightness") >= cls.LAMP_ON_THRESHOLD

    @classmethod
    def _LampOff_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        lamp, = objects
        return state.get(lamp, "brightness") < cls.LAMP_ON_THRESHOLD

    @staticmethod
    def _BreakerTripped_holds(state: State, objects: Sequence[Object]) -> bool:
        breaker, = objects
        return state.get(breaker, "tripped") > 0.5

    @staticmethod
    def _BreakerClosed_holds(state: State, objects: Sequence[Object]) -> bool:
        breaker, = objects
        return state.get(breaker, "tripped") <= 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    # =========================================================================
    # TASK GENERATION
    # =========================================================================
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng,
                                train=True)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng,
                                train=False)

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        button_counts = list(CFG.busyboard_num_buttons_train if train else CFG.
                             busyboard_num_buttons_test)
        lamp_counts = list(CFG.busyboard_num_lamps_train if train else CFG.
                           busyboard_num_lamps_test)
        min_lit = int(CFG.busyboard_min_lit_train if train else CFG.
                      busyboard_min_lit_test)

        tasks = []
        wiring: List[Condition] = []
        for _ in range(num_tasks):
            num_buttons = int(rng.choice(button_counts))
            num_lamps = int(rng.choice(lamp_counts))
            wiring, target = self._sample_board(num_buttons, num_lamps, rng,
                                                min_lit, train)

            init_dict: Dict[Object, Dict[str, float]] = {
                self._robot: {
                    "x": self.robot_init_x,
                    "y": self.robot_init_y,
                    "z": self.robot_init_z,
                    "fingers": self.open_fingers,
                    "roll": self.robot_init_roll,
                    "tilt": self.robot_init_tilt,
                    "wrist": self.robot_init_wrist,
                }
            }

            # Every board starts fully off: all buttons released, every lamp
            # dark, uncharged and unarmed, the breaker closed. The agent's
            # information about this board therefore comes entirely from
            # what it does to it.
            for i, ((x, y), button) in enumerate(
                    zip(self.button_layout(num_buttons),
                        self._buttons[:num_buttons])):
                init_dict[button] = {
                    "x": x,
                    "y": y,
                    "z": self.board_top,
                    "rot": self.button_rot,
                    "color": float(self.button_color_index(i)),
                    "is_on": 0.0,
                }

            lamp_z = self.lamp_z
            for i, (x, lamp) in enumerate(
                    zip(self._row_xs(num_lamps, self.lamp_x_gap),
                        self._lamps[:num_lamps])):
                lamp_dict = {
                    "x": x,
                    "y": self.lamp_y,
                    "z": lamp_z,
                    "rot": 0.0,
                    "color": float(self.lamp_color_index(i)),
                    "brightness": 0.0,
                }
                if "charge" in self._lamp_type.feature_names:
                    lamp_dict["charge"] = 0.0
                if "armed" in self._lamp_type.feature_names:
                    lamp_dict["armed"] = 0.0
                init_dict[lamp] = lamp_dict

            init_dict[self._breaker] = {
                "x": self.breaker_pos[0],
                "y": self.breaker_pos[1],
                "z": self.breaker_z,
                "tripped": 0.0,
            }

            init_state = utils.create_state_from_dict(init_dict)

            goal_atoms = set()
            for lamp, want_on in zip(self._lamps[:num_lamps], target):
                pred = self._LampOn if want_on else self._LampOff
                goal_atoms.add(GroundAtom(pred, [lamp]))

            lit = [self.lamp_label(i) for i, t in enumerate(target) if t]
            dark = [self.lamp_label(i) for i, t in enumerate(target) if not t]
            goal_nl = ("Use the buttons to leave " + f"{', '.join(lit)} lit" +
                       (f" and {', '.join(dark)} dark." if dark else "."))

            tasks.append(
                EnvironmentTask(init_state,
                                goal_atoms,
                                goal_nl=goal_nl,
                                offline_task_metrics=self._wiring_to_metrics(
                                    wiring, num_buttons)))

        # _add_pybullet_state_to_tasks replays every init state through the
        # simulator. Install the last sampled wiring so that replay runs
        # against a well-formed board; init states are all-off and all-dark,
        # so no wiring is observable in them either way. Guarded because a
        # run may ask for zero tasks of a split (num_train_tasks=0 is the
        # normal way to evaluate a non-learning approach), in which case
        # nothing was sampled and the installed wiring must stand.
        if tasks:
            self._wiring = wiring
        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    # Watch a board: latch every button on in turn and let the lamps
    # respond. Useful for eyeballing geometry and the charge delay.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_busyboard"
    CFG.num_train_tasks = 1
    CFG.num_test_tasks = 0
    env = PyBulletBusyBoardEnv(use_gui=True)
    _task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
    env._install_wiring(_task.offline_task_metrics)  # pylint: disable=protected-access
    env._set_state(_task.init)  # pylint: disable=protected-access
    print("wiring:", [str(c) for c in env._wiring])  # pylint: disable=protected-access
    print("goal:", _task.goal_description)

    _joints = env._pybullet_robot.initial_joint_positions  # pylint: disable=protected-access
    for _b in env._buttons[:env._num_active_buttons]:  # pylint: disable=protected-access
        env._set_button_on(_b, True)  # pylint: disable=protected-access
        for _ in range(12):
            env.step(Action(np.array(_joints)))
            time.sleep(0.05)
        _lit = env._lamps[:env._num_active_lamps]  # pylint: disable=protected-access
        print(_b.name, "on ->", [round(float(l.charge), 2) for l in _lit])
    while True:
        env.step(Action(np.array(_joints)))
        time.sleep(0.05)
