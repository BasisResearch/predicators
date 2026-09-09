"""The busyboard environment: hidden wiring, latent charge, tasks, predicates.

The observable simulation core (board geometry, body construction,
state read/write, button mechanics) lives in
:mod:`predicators.envs.pybullet_busyboard_base`, which may be surfaced
to learning agents as reference source. This module holds everything an
agent must LEARN or must not see:

* the WIRING - which button drives which lamp, and which second button
  must also be on for that drive to take effect;
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

**Conjunctive drives (many-to-one).** A lamp may require TWO buttons at
once: its ``driver`` and its ``enabler``. Prior busyboard-style
environments for robot learning exclude many-to-one relations
specifically to keep the relations identifiable from undirected play
(Liu et al., CoRL 2022, "BusyBot", Sec. 3: "We exclude many-to-one
relations to eliminate possible ambiguities"). Including them is the
point: an interlock is exactly the structure that a passive observer
confounds and a well-chosen experiment separates.

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
agent trained on keeps its drive condition, and the buttons and lamps a
test board adds are either decoys or a new lamp that goals only ever
ask to keep dark. So the relation learned in training is true at test,
the way glue chemistry or fan thrust is in the other domains, and what
test asks is whether the agent trusts it on a bigger board with more
distractors - and leaves alone the buttons it knows nothing about. Test
goals also ask for at least two lamps lit (``busyboard_min_lit_test``),
so they need two learned conditions composed, with any button the two
share held, rather than one training goal repeated on a bigger board.

**Decoys.** Some buttons drive nothing. Their existence is what makes
"press everything" uninformative, and it is why goals require some
lamps to stay OFF: a policy that simply latches every button on lights
every lamp it can and fails every task with an off-target.

Example commands::

    # Watch a board with the GUI, no agent.
    python predicators/envs/pybullet_busyboard.py

    # Oracle demo via bilevel process planning (solves 10/10).
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
goal. Delay tuning does not fix this - the check is off-by-one against
physics in both directions at once, and a sweep over delay values and
option durations moves which tasks fail without ever clearing all ten.
"""
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_busyboard_base import PyBulletBusyBoardBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

# Sentinel for "this lamp has no enabler" in the wiring vectors and in the
# flat float encoding carried on EnvironmentTask.offline_task_metrics.
NO_ENABLER: int = -1


def canonical_pair(driver: int, enabler: int) -> Tuple[int, int]:
    """Put a drive condition in canonical form.

    A conjunctive drive is symmetric - a lamp needing button 1 AND
    button 2 is the same lamp whichever of the two is called the
    "driver" - so no experiment can order them and the two labellings
    must not be treated as different hypotheses. Canonical form puts the
    lower button index first.
    """
    if enabler in (NO_ENABLER, driver):
        return (driver, NO_ENABLER)
    return (min(driver, enabler), max(driver, enabler))


def legal_pairs(num_buttons: int) -> List[Tuple[int, int]]:
    """Every distinct drive condition on a board of this size.

    This list is the domain's discrete hypothesis space per lamp:
    ``num_buttons`` plain drives plus ``num_buttons choose 2``
    conjunctive ones, the latter unordered because conjunction is
    symmetric. For a 4-button board that is 10 conditions per lamp, so a
    3-lamp board has 10 * 9 * 8 = 720 distinct wirings once the lamps'
    conditions are required to be distinct - small enough to
    enumerate exactly, which is what makes information gain here
    measurable in bits against an optimal splitter.
    """
    pairs = [(d, NO_ENABLER) for d in range(num_buttons)]
    pairs += [(d, e) for d in range(num_buttons)
              for e in range(d + 1, num_buttons)]
    return pairs


def _dedupe_wiring(driver: List[int], enabler: List[int],
                   num_buttons: int) -> Tuple[List[int], List[int]]:
    """Force every lamp's drive condition to be distinct.

    Two lamps wired identically are indistinguishable by construction:
    no experiment separates them and no goal can ask for one lit and the
    other dark, which would make the board unsolvable for reasons that
    have nothing to do with inference. Colliding lamps are moved to the
    nearest unused pair of the same kind (conjunctive stays conjunctive
    where possible), which keeps the interlock mix the sampler chose.
    """
    pairs = legal_pairs(num_buttons)
    used: set = set()
    out_d, out_e = [], []
    for d, e in zip(driver, enabler):
        pair = canonical_pair(d, e)
        if pair in used:
            want_conjunctive = e != NO_ENABLER
            candidates = [
                p for p in pairs
                if p not in used and (p[1] != NO_ENABLER) == want_conjunctive
            ] or [p for p in pairs if p not in used]
            if not candidates:
                raise RuntimeError(
                    f"A {num_buttons}-button board has only {len(pairs)} "
                    f"distinct drive conditions, fewer than the number of "
                    f"lamps requested.")
            pair = candidates[0]
        used.add(pair)
        out_d.append(pair[0])
        out_e.append(pair[1])
    return out_d, out_e


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


def project_wiring(driver_full: Sequence[int], enabler_full: Sequence[int],
                   num_buttons: int,
                   num_lamps: int) -> Tuple[List[int], List[int]]:
    """Project a max-board wiring onto a board of the requested size.

    The projection is an EXTENSION: the core board's wiring is a subset
    of every larger board's wiring. Concretely, for the first
    ``num_lamps`` lamps:

    * a core lamp keeps its drive condition verbatim, which is well
      defined because ``canonical_wiring`` wires core lamps to core
      buttons only;
    * an extension lamp's driver folds into whatever extension buttons
      this board has (see ``_remap_extension``), and its enabler folds
      the same way if it is an extension button or stays put if it is a
      core one.

    Then the two ways folding can degrade a drive condition are
    repaired: an enabler landing on its own driver (which would silently
    turn a conjunctive drive into a plain one) and two lamps landing on
    the same condition (which would make them indistinguishable by any
    experiment).

    So a rule learned about a core lamp on the training board is true of
    that lamp on every test board, and the buttons a test board adds
    either do nothing or feed a lamp the goals only ever ask to keep
    dark. This is a pure function of the max-board wiring and the board
    size, which is what lets ONE parameter vector be a correct model of
    every board in a run: the ground-truth simulator carries the
    max-board wiring in its params and applies this same projection to
    whatever board the observation shows it.
    """
    core_buttons, core_lamps = core_board()
    core_buttons = min(core_buttons, num_buttons)

    def _fold(index: int, is_core_lamp: bool) -> int:
        if is_core_lamp or index < core_buttons:
            return index % core_buttons
        return _remap_extension(index, core_buttons, num_buttons)

    driver = [
        _fold(int(driver_full[i]), i < core_lamps) for i in range(num_lamps)
    ]
    enabler: List[int] = []
    for i in range(num_lamps):
        e = int(enabler_full[i])
        if e == NO_ENABLER or num_buttons < 2:
            enabler.append(NO_ENABLER)
            continue
        e = _fold(e, i < core_lamps)
        if e == driver[i]:
            # Bump within the range the lamp is allowed to use: core
            # buttons for a core lamp, any button for an extension lamp.
            limit = core_buttons if i < core_lamps else num_buttons
            e = (e + 1) % limit
        enabler.append(e)
    return _dedupe_wiring(driver, enabler, num_buttons)


def canonical_wiring(num_buttons: int,
                     num_lamps: int) -> Tuple[List[int], List[int]]:
    """The run's wiring, reduced to a board of the requested size.

    Sampled once per seed over the largest board the task distribution
    can produce, then projected onto smaller boards by extension (see
    ``project_wiring``). The draw respects the extension contract: core
    lamps are wired to core buttons only, and an extension lamp's driver
    is an extension button, so the core board's wiring is literally a
    sub-relation of every larger board's. So one wiring describes every
    board in a run, and what an agent learns about the lamps on a
    4-button training board stays true of those lamps on a 6-button
    test board; the test board differs by the buttons and lamps it adds.

    Why a run-level constant rather than a fresh draw per task: the
    residual-simulator contract in this codebase resolves ``PARAM_SPECS``
    once, after CFG is final and before any task is chosen, so a hidden
    quantity that varied per task would have no home in a fitted model.
    Making the wiring vary per task (``busyboard_fixed_wiring = False``)
    is the more interesting setting and the one this domain exists to
    motivate - a model whose STRUCTURE is re-identified per episode -
    but it needs a per-task parameter scope that the fitting stack does
    not yet have. Both the env and the ground-truth simulator call this
    function, so they agree on the answer without either reading the
    other.
    """
    rng = np.random.default_rng([CFG.seed, int(CFG.busyboard_wiring_salt)])
    max_buttons = max(
        list(CFG.busyboard_num_buttons_train) +
        list(CFG.busyboard_num_buttons_test))
    max_lamps = max(
        list(CFG.busyboard_num_lamps_train) +
        list(CFG.busyboard_num_lamps_test))
    core_buttons, core_lamps = core_board()

    driver_full: List[int] = []
    enabler_full: List[int] = []
    for i in range(max_lamps):
        # A core lamp lives entirely on the core board; an extension lamp
        # is driven by a button the core board does not have (falling
        # back to any button when the distribution adds lamps but no
        # buttons), and may take its enabler from anywhere.
        if i < core_lamps:
            driver_pool = list(range(core_buttons))
            enabler_pool = list(range(core_buttons))
        else:
            driver_pool = list(range(core_buttons, max_buttons)) or list(
                range(max_buttons))
            enabler_pool = list(range(max_buttons))
        d = int(rng.choice(driver_pool))
        driver_full.append(d)
        enabler_pool = [b for b in enabler_pool if b != d]
        if not enabler_pool or rng.random() >= CFG.busyboard_interlock_prob:
            enabler_full.append(NO_ENABLER)
            continue
        enabler_full.append(int(rng.choice(enabler_pool)))

    return project_wiring(driver_full, enabler_full, num_buttons, num_lamps)


class PyBulletBusyBoardEnv(PyBulletBusyBoardBaseEnv):
    """A busy board whose button-to-lamp wiring must be discovered.

    Subclass of the observable sim core (see
    :mod:`predicators.envs.pybullet_busyboard_base`); this class adds the
    hidden wiring, the charge dynamics, task generation, and predicates.
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
    # Fully observable: the charge is a visible feature, so a learner sees
    # the accumulation directly and only the wiring is hidden.
    _lamp_type_fo = Type(
        "lamp", ["x", "y", "z", "rot", "color", "brightness", "charge"])
    # Partially observable: charge is dropped. The learner sees only the
    # brightness readout and must postulate the accumulator itself.
    _lamp_type_po = Type("lamp", ["x", "y", "z", "rot", "color", "brightness"])

    @classmethod
    def _lamp_type_for_run(cls) -> Type:
        return cls._lamp_type_po if CFG.partially_observable \
            else cls._lamp_type_fo

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Bind the run's lamp type before super().__init__ builds the lamp
        # objects and the predicates off it.
        self._lamp_type = self._lamp_type_for_run()

        # The active task's wiring. Index i is lamp i; the value is a button
        # index (or NO_ENABLER). Installed by reset() and never present in
        # any State - this is the learning target.
        self._driver: List[int] = []
        self._enabler: List[int] = []
        # How much of the (max-size) board this task actually uses.
        self._num_active_buttons: int = 0
        self._num_active_lamps: int = 0
        # Hidden per-lamp charge, keyed by object NAME rather than held in
        # each lamp Object's sim_data. Bilevel planning runs a second env
        # instance built by the option model, and that instance owns its
        # own Object instances: identical in name and type, but with
        # separate sim_data. Charge kept there accumulated on one set of
        # objects while the state was read off the other, so the planner
        # saw a board whose lamps never charged. A name-keyed store is
        # instance-independent, which is what the two env copies need.
        self._charges: Dict[str, float] = {}

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
            self._HandEmpty
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._LampOn, self._LampOff}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._button_type, self._lamp_type}

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
        self._driver = [
            int(metrics[f"wiring_driver_{i}"]) for i in range(num_lamps)
        ]
        self._enabler = [
            int(metrics[f"wiring_enabler_{i}"]) for i in range(num_lamps)
        ]

    @staticmethod
    def _wiring_to_metrics(driver: Sequence[int], enabler: Sequence[int],
                           num_buttons: int) -> Dict[str, float]:
        """Flatten a wiring into the float-valued metrics dict."""
        metrics: Dict[str, float] = {
            "wiring_num_lamps": float(len(driver)),
            "wiring_num_buttons": float(num_buttons),
        }
        for i, (d, e) in enumerate(zip(driver, enabler)):
            metrics[f"wiring_driver_{i}"] = float(d)
            metrics[f"wiring_enabler_{i}"] = float(e)
        return metrics

    @staticmethod
    def _driven(button_on: Sequence[bool], driver: int, enabler: int) -> bool:
        """Whether a lamp's drive condition holds under a button assignment.

        The conjunction is the interlock: with an enabler present, the
        driver alone does nothing, which is precisely the many-to-one
        relation that undirected play confounds.
        """
        if not 0 <= driver < len(button_on) or not button_on[driver]:
            return False
        if enabler == NO_ENABLER:
            return True
        return 0 <= enabler < len(button_on) and button_on[enabler]

    @classmethod
    def _realizable_targets(cls,
                            driver: Sequence[int],
                            enabler: Sequence[int],
                            num_buttons: int,
                            num_lit_candidates: Optional[int] = None,
                            min_lit: int = 1) -> List[Tuple[bool, ...]]:
        """Every lamp assignment some button setting realizes exactly.

        Exhaustive over the 2**num_buttons settings, which is a handful
        at these board sizes, so goals are drawn from the exact set of
        achievable ones rather than rejection-sampled against a solver.
        A target is only kept if it is non-trivial: at least one lamp
        lit, and with two or more lamps at least one that must stay
        dark. That off-target is what the whole domain rests on - it is
        the reason "latch every button" is not a policy, and the reason
        an agent has to know which button feeds which lamp rather than
        merely which buttons do something.

        Only the first ``num_lit_candidates`` lamps may be asked to be
        lit (default: all of them). Under the run-level wiring these are
        the core lamps, the ones training reveals; an extension lamp is
        only ever asked to stay dark, so what a test board adds is a
        button that must be left alone rather than a relation the agent
        never had a chance to learn.

        ``min_lit`` is the fewest lamps a target may ask to be lit (at
        least one either way). Above one it selects the goals that need
        two or more drive conditions satisfied at once, which is how
        the test split asks for composition rather than recall.
        """
        if num_lit_candidates is None:
            num_lit_candidates = len(driver)
        min_lit = max(1, min_lit)
        targets = set()
        for mask in range(1 << num_buttons):
            button_on = [bool(mask >> b & 1) for b in range(num_buttons)]
            target = tuple(
                cls._driven(button_on, d, e) for d, e in zip(driver, enabler))
            if sum(target) < min_lit:
                continue
            if len(target) >= 2 and all(target):
                continue
            if any(target[num_lit_candidates:]):
                continue
            targets.add(target)
        return sorted(targets)

    def _sample_board(self, num_buttons: int, num_lamps: int,
                      rng: np.random.Generator,
                      min_lit: int) -> Tuple[List[int], List[int], List[bool]]:
        """Pick this task's wiring and a goal assignment it can realize.

        With ``busyboard_fixed_wiring`` (the default) the wiring is the
        run's canonical one at this board size; otherwise a fresh wiring
        is drawn per task. Either way the goal is drawn uniformly from
        the exact set of realizable non-trivial assignments, which is
        never empty: distinct drive conditions are monotone boolean
        functions that differ somewhere, and at any input where two of
        them differ one lamp is lit and another is dark. Under the run-
        level wiring only core lamps may be lit targets, which leaves
        that argument intact because an extension lamp's driver is a
        button no core lamp uses. ``min_lit`` (the split's
        ``busyboard_min_lit_*``) narrows the draw to targets with at
        least that many lamps lit; up to the number of core lamps it is
        always satisfiable under the run-level wiring, since latching
        every core button lights every core lamp and no extension lamp.
        """
        num_lit_candidates = num_lamps
        if CFG.busyboard_fixed_wiring:
            driver, enabler = canonical_wiring(num_buttons, num_lamps)
            num_lit_candidates = min(num_lamps, core_board()[1])
        else:
            driver = [
                int(rng.integers(0, num_buttons)) for _ in range(num_lamps)
            ]
            enabler = []
            for d in driver:
                if num_buttons < 2 or \
                        rng.random() >= CFG.busyboard_interlock_prob:
                    enabler.append(NO_ENABLER)
                    continue
                enabler.append(
                    int(rng.choice([b for b in range(num_buttons) if b != d])))
            driver, enabler = _dedupe_wiring(driver, enabler, num_buttons)

        targets = self._realizable_targets(driver, enabler, num_buttons,
                                           num_lit_candidates, min_lit)
        if not targets:
            raise RuntimeError(
                f"No non-trivial realizable goal with at least {min_lit} "
                f"lamp(s) lit on a {num_buttons}-button, {num_lamps}-lamp "
                f"board with wiring {list(zip(driver, enabler))}.")
        target = list(targets[int(rng.integers(0, len(targets)))])
        return driver, enabler, target

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
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Restore button latches and lamp charges, then repaint the lamps.

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
            self._driver, self._enabler = canonical_wiring(
                len(buttons), len(lamps))

        for button in buttons:
            self._set_button_on(button, state.get(button, "is_on") > 0.5)

        for lamp in lamps:
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
            self._set_lamp_brightness_visual(
                lamp, self._brightness(self._charges[lamp.name]))

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
        """Accumulate or bleed each lamp's hidden charge, then repaint it.

        Skipped when the env is constructed with
        ``skip_residual_dynamics=True`` - that is the base sim the
        learning agent rolls out, and it must show a board on which no
        button ever lights anything.
        """
        buttons = self._buttons[:self._num_active_buttons]
        lamps = self._lamps[:self._num_active_lamps]
        button_on = [self._is_button_on(b) for b in buttons]

        for i, lamp in enumerate(lamps):
            if i >= len(self._driver):
                continue
            charge = self._charges.get(lamp.name, 0.0)
            if self._driven(button_on, self._driver[i], self._enabler[i]):
                charge = min(1.0, charge + self.charge_rate())
            else:
                charge = max(0.0, charge - self.decay_rate())
            self._charges[lamp.name] = charge
            self._set_lamp_brightness_visual(lamp, self._brightness(charge))

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
        for _ in range(num_tasks):
            num_buttons = int(rng.choice(button_counts))
            num_lamps = int(rng.choice(lamp_counts))
            driver, enabler, target = self._sample_board(
                num_buttons, num_lamps, rng, min_lit)

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
            # dark and uncharged. The agent's information about this board
            # therefore comes entirely from what it does to it.
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
                init_dict[lamp] = lamp_dict

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
                                    driver, enabler, num_buttons)))

        # _add_pybullet_state_to_tasks replays every init state through the
        # simulator. Install the last sampled wiring so that replay runs
        # against a well-formed board; init states are all-off and all-dark,
        # so no wiring is observable in them either way. Guarded because a
        # run may ask for zero tasks of a split (num_train_tasks=0 is the
        # normal way to evaluate a non-learning approach), in which case
        # nothing was sampled and the installed wiring must stand.
        if tasks:
            self._driver, self._enabler = driver, enabler
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
    print("wiring:", list(zip(env._driver, env._enabler)))  # pylint: disable=protected-access
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
