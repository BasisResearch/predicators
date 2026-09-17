"""The balloons environment: release, lift, pop, masses, tasks, predicates.

The observable simulation core (table, box, clipped balloons, band,
ceiling) lives in :mod:`predicators.envs.pybullet_balloons_base`, which
may be surfaced to learning agents as reference source. This module
holds everything an agent must LEARN or must not see:

* the RELEASE - opening a balloon's clip frees it, and its string
  pulls it onto the box's top, where it stays;
* the LIFT - a freed balloon pulls the box up with a force set by its
  colour, which fades with the box's height, so a box with enough
  balloons rises to the height where the pull just matches its weight
  and hangs there;
* the POP - a balloon that reaches the ceiling bursts, and pulls no
  more;
* the MATERIALS - how heavy each box colour is, and how much the air
  drags the assembly;
* task generation (the train/test distribution) and goal semantics.

Task selection checks executable release sequences against the evaluator.
A test level has an order-robust winning reference subset and a witnessed
losing release sequence whose subset has an in-band analytic equilibrium.
Other subsets or release orders may also win; uniqueness is not claimed.
The underdamped ascent and irreversible bursts can make release order
matter even when equilibrium heights are similar.
Whether this benefits MB over MF is an experimental question.

**Test extends train.** A test level holds the whole palette, one more
balloon than any train level, on a box material training showed; the
train levels together cover every colour and both materials, so the
test composes known lifts, a known mass, and the drag learned from the
train ascents into a rack never seen.

**Composition test levels** close the lookup this leaves open: with
every colour shown on both boxes, a test band centred on a rest height
training measured is answered from memory. Every test level is a
composition level. It admits no winning in-band subset that is a single
balloon or a colour set some train rack on the same box already held.
And every in-band subset bursts when freed weakest lift first with the
box settling between releases, the order and timing an agent that
reads rest heights alone and plays safe would use, while the reference
wins in another order (the last increment must be small enough not to
overshoot into the ceiling). Ordering the ascent takes the transient.

**The dwell** (``balloons_goal_dwell_steps``): the level is won only
once the box has hung at rest inside the band for that many consecutive
environment steps. A single-frame check lets a swinging box win at a
turning point whose peak pokes into the band while its rest height lies
outside it. The generator's probes judge with the same evaluator.

**Bundle test levels** (``balloons_test_bundle_sizes``) take away the
safe first release. The composition levels above are still solved by
arithmetic on rest heights: free the strongest balloon training already
measured, watch it settle below the band, then trim with the weakest.
On a bundle level the test rack ties its balloons into bundles, one
clip per bundle (every balloon's ``clip`` feature names its clip), so
one cut frees two or more balloons at once and no cut is a small trim.
Every first cut is a union training never showed, launched from the
table, and whether it survives is a question about the transient, not
the rest height. The generator accepts a bundle level only when

* the reference wins with two or more cuts, settling between them, and
  its first cut is not the weakest bundle;
* every winning union is a colour multiset no train rack on the same
  box held;
* some decoy bundle rests inside the band by the analytic law yet
  bursts when cut first from the table (the arithmetic answer loses);
* every winning order starts with the same clip, so the first cut
  decides the level; and, when the draws allow it,
* some other bundle rests below the band, above the reference's first
  cut, with an analytic in-band continuation of its own, which loses.

Cutting the weakest bundle, cutting the bundle whose rest is in band,
and cutting the highest safe bundle then adding are the three readings
of the rest heights; the first two always lose here, the third loses
on a level with such a tempting bundle (the generator prefers one and
falls back to a level without, recorded as ``bundle_tempting_clip``
-1), and the winning order is found by rolling the ascent forward.
"""
import logging
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, permutations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce, Attach, \
    PhysicsCommand
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, StepOption, TaskEvaluator

GRAVITY = 9.81


# =============================================================================
# CLASSIFIERS
# =============================================================================
def box_in_band(state: State, box: Object, band: Object) -> bool:
    """Whether the box's centre lies within the band's heights."""
    z = state.get(box, "z")
    return state.get(band, "lo") <= z <= state.get(band, "hi")


def box_at_rest(state: State, box: Object) -> bool:
    """Whether the box has stopped moving."""
    return state.get(box, "speed") < CFG.balloons_settle_speed


def any_popped(state: State) -> Optional[Object]:
    """The first burst balloon, if any."""
    for obj in sorted((o for o in state if o.type.name == "balloon"),
                      key=lambda o: o.name):
        if state.get(obj, "popped") > 0.5:
            return obj
    return None


class BalloonsEvaluator(TaskEvaluator):
    """Win when the box hangs at rest inside the band for a dwell of
    consecutive steps; a burst balloon loses the level."""

    def __init__(self, goal: Set[GroundAtom]) -> None:
        super().__init__(goal)
        self.dwell_steps = int(CFG.balloons_goal_dwell_steps)
        if self.dwell_steps < 1:
            raise ValueError("balloons_goal_dwell_steps must be positive")

    def terminated_trajectory(self, states: Sequence[State]) -> bool:
        if not states:
            return False
        if any_popped(states[-1]) is not None:
            return True
        # N complete real step intervals require N+1 endpoint observations.
        window = states[-self.dwell_steps - 1:]
        return len(window) == self.dwell_steps + 1 and all(
            self.terminated(state) and any_popped(state) is None
            for state in window)

    def terminated(self, state: State) -> bool:
        if any_popped(state) is not None:
            return True
        box = next(o for o in state if o.type.name == "box")
        return all(atom.holds(state)
                   for atom in self.goal) and box_at_rest(state, box)

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        del step_options, sim_env  # unused
        popped = any_popped(states[-1])
        if popped is not None:
            return False, f"{popped.name} burst on the ceiling"
        return True, ""

    def objective_description(self) -> str:
        speed_limit = CFG.balloons_settle_speed
        return (
            "The level is won when the box hangs at rest with its centre "
            f"inside the band for {self.dwell_steps} consecutive environment "
            f"steps, with speed below {speed_limit:g} m/s throughout. "
            "A balloon that reaches the ceiling bursts "
            "and ends the level.")


@dataclass(frozen=True)
class BalloonsProbeOutcome:
    """A witnessed success/failure, or an unresolved finite rollout."""
    status: str
    steps: int
    height: float
    speed: float
    wall_supported: bool = False
    wall_free_won: bool = False

    @property
    def won(self) -> bool:
        """Whether the evaluator's success condition was witnessed."""
        return self.status == "won"

    @property
    def burst(self) -> bool:
        """Whether an irreversible ceiling burst was witnessed first."""
        return self.status == "burst"

    @property
    def jammed(self) -> bool:
        """A wall-supported failure whose identical wall-free replay wins."""
        return (self.status == "resting_outside" and self.wall_supported
                and self.wall_free_won)


class PyBulletBalloonsEnv(PyBulletBalloonsBaseEnv):
    """A balloon puzzle whose lifts, fade and box masses must be learned."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._candidate_cache: Dict[Tuple[Any, ...],
                                    Dict[Tuple[int, ...],
                                         List[BalloonsProbeOutcome]]] = {}
        super().__init__(use_gui, **kwargs)
        self._InBand = Predicate(
            "InBand", [self._box_type, self._band_type],
            self._InBand_holds,
            natural_language_assertion=lambda os:
            f"box {os[0]} hangs at rest within band {os[1]}")
        self._Tied = Predicate(
            "Tied", [self._balloon_type],
            self._Tied_holds,
            natural_language_assertion=lambda os:
            f"balloon {os[0]} has been freed and pulled onto the box")
        self._Untied = Predicate("Untied", [self._balloon_type],
                                 self._Untied_holds,
                                 natural_language_assertion=lambda os:
                                 f"balloon {os[0]} is still in its clip")
        self._Intact = Predicate("Intact", [self._balloon_type],
                                 self._Intact_holds,
                                 natural_language_assertion=lambda os:
                                 f"balloon {os[0]} has not burst")
        self._Popped = Predicate(
            "Popped", [self._balloon_type],
            self._Popped_holds,
            natural_language_assertion=lambda os: f"balloon {os[0]} has burst")
        self._ClipOn = Predicate(
            "ClipOn", [self._clip_type],
            self._ClipOn_holds,
            natural_language_assertion=lambda os: f"clip {os[0]} is open")
        self._ClipOff = Predicate(
            "ClipOff", [self._clip_type],
            self._ClipOff_holds,
            natural_language_assertion=lambda os: f"clip {os[0]} is closed")
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds,
                                    natural_language_assertion=lambda os:
                                    f"robot {os[0]} is not holding anything")
        self._AtRest = Predicate(
            "AtRest", [self._box_type],
            self._AtRest_holds,
            natural_language_assertion=lambda os: f"box {os[0]} is not moving")

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_balloons"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InBand, self._Tied, self._Untied, self._Intact, self._Popped,
            self._ClipOn, self._ClipOff, self._HandEmpty, self._AtRest
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InBand}

    # =========================================================================
    # HIDDEN PHYSICS
    # =========================================================================
    @classmethod
    def lift_at_ground(cls, color_index: int) -> float:
        """A balloon colour's pull at table height, in newtons."""
        return float(CFG.balloons_lifts[int(round(color_index))])

    @classmethod
    def lift_at(cls, color_index: int, z: float) -> float:
        """The pull of a colour on a box at height ``z``: fading linearly to
        zero ``balloons_fade_height`` above the table."""
        frac = (z - cls.table_height) / float(CFG.balloons_fade_height)
        return cls.lift_at_ground(color_index) * max(0.0, 1.0 - frac)

    @classmethod
    def true_box_mass(cls, color_index: int) -> float:
        """The mass of a box material, the learning target."""
        return float(CFG.balloons_box_masses[int(round(color_index))])

    def _box_mass_for(self, color_index: int) -> float:
        if self._skip_domain_specific_dynamics:
            return self.believed_box_mass(color_index)
        return self.true_box_mass(color_index)

    def _drag(self) -> float:
        if self._skip_domain_specific_dynamics:
            return float(self._param_overrides.get("air_drag", 0.04))
        return float(CFG.balloons_drag)

    def _domain_specific_step(self) -> None:
        """Release, pop, and pull: an open clip frees every balloon it holds,
        which their strings seat on the box's top; a freed balloon that reaches
        the ceiling bursts; every intact freed balloon pulls the box up with
        its colour's lift at the box's height while its string holds it to the
        box."""
        state = self._get_state()
        box_top = self.box_top_point(state, self._box)
        box_z = float(state.get(self._box, "z"))
        balloons = self._active_balloons(state)
        clips = self._active_clips(state)
        commands: List[PhysicsCommand] = []
        clip_of = [
            self._balloon_clips.get(b.name, i) for i, b in enumerate(balloons)
        ]
        # Tiers: one per bundle freed, in the order they were freed.
        tiers = len({
            clip_of[i]
            for i, b in enumerate(balloons) if self._tied.get(b.name, False)
        })
        new_tiers: Dict[int, int] = {}
        for index, balloon in enumerate(balloons):
            name = balloon.name
            if not self._tied.get(name, False):
                clip = clip_of[index]
                if clip >= len(clips) or not self._is_clip_on(clips[clip]):
                    continue
                # Freed: the string pulls the balloon to the end of its
                # tether above the box's top, a tier above the bundles
                # freed before it, so its pull acts through the box.
                self._tied[name] = True
                if clip not in new_tiers:
                    new_tiers[clip] = tiers
                    tiers += 1
                bundle = [i for i, c in enumerate(clip_of) if c == clip]
                dx, dy, dz = self.bundle_seat(index, len(balloons),
                                              bundle.index(index), len(bundle),
                                              new_tiers[clip])
                seat = (box_top[0] + dx, box_top[1] + dy, box_top[2] + dz)
                p.resetBasePositionAndOrientation(
                    balloon.id,
                    seat, (0.0, 0.0, 0.0, 1.0),
                    physicsClientId=self._physics_client_id)
                p.resetBaseVelocity(balloon.id, (0.0, 0.0, 0.0),
                                    (0.0, 0.0, 0.0),
                                    physicsClientId=self._physics_client_id)
            z_balloon = float(state.get(balloon, "z"))
            if not self._popped.get(name, False) and \
                    z_balloon + self.balloon_radius >= self.ceiling_z - \
                    self.ceiling_half_extents[2] - 0.002:
                self._popped[name] = True
                self._paint_balloon(balloon)
            commands.append(Attach(name, self._box.name))
            if not self._popped.get(name, False):
                # The pull acts at the balloon, above the box, and the
                # string carries it down: pulled from above, the box
                # hangs upright; pushed from below it would tip over.
                lift = self.lift_at(self._balloon_colors[name], box_z)
                commands.append(ApplyForce(name, (0.0, 0.0, lift)))
        if commands:
            self.queue_residual_commands(commands)

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _InBand_holds(state: State, objects: Sequence[Object]) -> bool:
        # Hanging in the band: a box passing through it on its way up
        # does not count, so a Wait keeps waiting for it to settle.
        box, band = objects
        return box_in_band(state, box, band) and box_at_rest(state, box)

    @staticmethod
    def _Tied_holds(state: State, objects: Sequence[Object]) -> bool:
        balloon, = objects
        return state.get(balloon, "tied") > 0.5

    @staticmethod
    def _Untied_holds(state: State, objects: Sequence[Object]) -> bool:
        balloon, = objects
        return state.get(balloon, "tied") <= 0.5

    @staticmethod
    def _Intact_holds(state: State, objects: Sequence[Object]) -> bool:
        balloon, = objects
        return state.get(balloon, "popped") <= 0.5

    @staticmethod
    def _Popped_holds(state: State, objects: Sequence[Object]) -> bool:
        balloon, = objects
        return state.get(balloon, "popped") > 0.5

    @staticmethod
    def _ClipOn_holds(state: State, objects: Sequence[Object]) -> bool:
        clip, = objects
        return state.get(clip, "is_on") > 0.5

    @staticmethod
    def _ClipOff_holds(state: State, objects: Sequence[Object]) -> bool:
        clip, = objects
        return state.get(clip, "is_on") <= 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    @staticmethod
    def _AtRest_holds(state: State, objects: Sequence[Object]) -> bool:
        box, = objects
        return box_at_rest(state, box)

    # =========================================================================
    # THE FLOAT LAW, FOR THE GENERATOR AND THE ORACLE
    # =========================================================================
    @classmethod
    def hover_height(cls, box_color: int,
                     balloon_colors: Sequence[int]) -> Optional[float]:
        """The height the box's centre settles at with these balloons freed, by
        the analytic law; None if they cannot lift it."""
        total_lift = sum(cls.lift_at_ground(c) for c in balloon_colors)
        weight = (cls.true_box_mass(box_color) +
                  len(balloon_colors) * cls.balloon_mass) * GRAVITY
        if total_lift <= weight:
            return None
        return cls.table_height + float(
            CFG.balloons_fade_height) * (1.0 - weight / total_lift)

    @classmethod
    def burst_height(cls, n_tiers: int) -> float:
        """The box-centre height at which the top of a stack of ``n_tiers``
        freed bundles (one tier each; a lone balloon is a bundle of one) meets
        the ceiling plate and bursts."""
        return (cls.ceiling_z - cls.ceiling_half_extents[2] - 0.002 -
                cls.box_half - cls.string_length -
                2 * cls.balloon_radius * n_tiers)

    @classmethod
    def unit_lift(cls, unit: Sequence[int]) -> float:
        """The pull at table height of the balloons one clip frees."""
        return sum(cls.lift_at_ground(c) for c in unit)

    @classmethod
    def union_colors(cls, unit_colors: Sequence[Sequence[int]],
                     subset: Sequence[int]) -> List[int]:
        """The colours the clips of ``subset`` free together."""
        return [c for i in subset for c in unit_colors[i]]

    @classmethod
    def lifting_subsets(
        cls, box_color: int, unit_colors: Sequence[Sequence[int]]
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Every subset of clips that lifts the box, with the hover height of
        the balloons they free together, highest first.

        ``unit_colors`` lists the colours each clip frees; on a rack of
        single balloons that is one colour per clip, so a subset is the
        balloon indices as well.
        """
        out = []
        indices = list(range(len(unit_colors)))
        for size in range(1, len(indices) + 1):
            for subset in combinations(indices, size):
                z = cls.hover_height(box_color,
                                     cls.union_colors(unit_colors, subset))
                if z is not None:
                    out.append((subset, z))
        return sorted(out, key=lambda item: -item[1])

    def unit_colors(self, state: State) -> List[List[int]]:
        """The colours each active clip frees, by clip index."""
        balloons = self._active_balloons(state)
        colors = [int(round(state.get(b, "color"))) for b in balloons]
        return [[colors[i] for i in bundle] for bundle in self.bundles(state)]

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

    def _robot_init_dict(self) -> Dict[str, float]:
        return {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "roll": self.robot_init_roll,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

    def level_state(self,
                    box_color: int,
                    balloon_colors: Sequence[int],
                    band: Tuple[float, float],
                    clip_of: Optional[Sequence[int]] = None) -> State:
        """A level: the box on the table, balloons clipped in the rack, the
        band.

        ``clip_of`` names the clip holding each balloon; by default
        every balloon has its own clip, in rack order. A clip's bundle
        stands in a row behind it.
        """
        init: Dict[Object, Dict[str, float]] = {
            self._robot: self._robot_init_dict()
        }
        init[self._box] = {
            "x": self.box_xy[0],
            "y": self.box_xy[1],
            "z": self.box_z,
            "color": float(box_color),
            "speed": 0.0,
        }
        if clip_of is None:
            clip_of = list(range(len(balloon_colors)))
        assert len(clip_of) == len(balloon_colors)
        n_clips = max(clip_of) + 1 if clip_of else 0
        assert sorted(set(clip_of)) == list(range(n_clips))
        xs = self.rack_xs(n_clips)
        row: Dict[int, int] = {}
        for i, color in enumerate(balloon_colors):
            clip = int(clip_of[i])
            init[self._balloons[i]] = {
                "x": xs[clip],
                "y": self.rack_y + self.bundle_y_gap * row.get(clip, 0),
                "z": self.balloon_rest_z,
                "color": float(color),
                "clip": float(clip),
                "tied": 0.0,
                "popped": 0.0,
            }
            row[clip] = row.get(clip, 0) + 1
        for clip in range(n_clips):
            init[self._clips[clip]] = {
                "x": xs[clip],
                "y": self.clip_y,
                "z": self.table_height,
                "rot": self.clip_rot,
                "is_on": 0.0,
            }
        init[self._band] = {
            "x": self.box_xy[0] + self.band_offset_x,
            "y": self.box_xy[1],
            "lo": band[0],
            "hi": band[1],
        }
        return utils.create_state_from_dict(init)

    def candidate_outcomes(
            self,
            state: State) -> Dict[Tuple[int, ...], List[BalloonsProbeOutcome]]:
        """Executable outcomes for every order of each in-band candidate.

        Candidates and orders are over clips. Reconstruct the clean
        level, so oracle predicates keep a stable reference as execution
        progresses. On a rack of single balloons these are immediate
        Release sequences followed by a hold, not all possible release
        timings; on a bundle level the box settles between cuts, the
        cadence of an agent that watches each cut land.
        """
        box_color = int(round(state.get(self._box, "color")))
        colors = [
            int(round(state.get(b, "color")))
            for b in self._active_balloons(state)
        ]
        clip_of = tuple(
            self.clip_index(state, b) for b in self._active_balloons(state))
        settled = self.is_bundled(state)
        lo, hi = (float(state.get(self._band, f)) for f in ("lo", "hi"))
        key = (box_color, tuple(colors), clip_of, settled, lo, hi,
               tuple(CFG.balloons_lifts), tuple(CFG.balloons_box_masses),
               CFG.balloons_fade_height, CFG.balloons_drag,
               CFG.balloons_probe_max_steps, CFG.balloons_probe_rest_steps,
               CFG.balloons_probe_rest_tol, CFG.balloons_settle_speed,
               CFG.balloons_goal_dwell_steps,
               CFG.skill_phase_use_motion_planning, CFG.seed,
               CFG.balloons_push_approach, CFG.balloons_push_contact_z,
               tuple(sorted(self._param_overrides.items())))
        if key not in self._candidate_cache:
            clean = self.level_state(box_color, colors, (lo, hi), clip_of)
            units = self.unit_colors(clean)
            self._candidate_cache[key] = {
                subset: [
                    self.release_sequence_outcome(clean, order, settled)
                    for order in permutations(subset)
                ]
                for subset, z in self.lifting_subsets(box_color, units)
                if lo <= z <= hi
            }
        return {
            subset: list(results)
            for subset, results in self._candidate_cache[key].items()
        }

    def solution_subset(self, state: State) -> Optional[Tuple[int, ...]]:
        """A canonical reference subset of clips.

        Prefer a subset whose every tested Release order wins, then
        fewer releases, then lexicographic order. A subset with any
        witnessed winning order also qualifies, ranked after the order-
        robust ones, because a composition level may hang on the release
        order (see ``reference_order``). Multiple winning subsets are
        allowed and scored by the same evaluator. On a bundle level the
        reference has at least two cuts (see the module doc).
        """
        candidates = self.candidate_outcomes(state)
        if self.is_bundled(state):
            candidates = {
                subset: outcomes
                for subset, outcomes in candidates.items() if len(subset) >= 2
            }
        ranked = [(0, len(subset), subset)
                  for subset, outcomes in candidates.items()
                  if outcomes and all(outcome.won for outcome in outcomes)]
        robust = {subset for _, _, subset in ranked}
        ranked += [(1, len(subset), subset)
                   for subset, outcomes in candidates.items()
                   if subset not in robust and any(o.won for o in outcomes)]
        return min(ranked)[2] if ranked else None

    def reference_order(self, state: State, subset: Tuple[int,
                                                          ...]) -> List[int]:
        """A release order of ``subset`` the probes witnessed winning: the
        weakest-first order when it wins, else the first winning order in
        ``permutations`` order, else weakest-first (the oracle then fails on
        this level)."""
        order = self.weakest_first_order(self.unit_colors(state), subset)
        outcomes = self.candidate_outcomes(state).get(subset, [])
        if not outcomes or self._order_outcome(outcomes, subset, order).won:
            return order
        for candidate, outcome in zip(permutations(subset), outcomes):
            if outcome.won:
                return list(candidate)
        return order

    @classmethod
    def weakest_first_order(cls, unit_colors: Sequence[Sequence[int]],
                            subset: Tuple[int, ...]) -> List[int]:
        """The subset's clips, weakest pull at table height first: the oracle's
        release order on a rack of singles."""
        return sorted(subset, key=lambda i: cls.unit_lift(unit_colors[i]))

    @staticmethod
    def _order_outcome(outcomes: Sequence[BalloonsProbeOutcome],
                       subset: Tuple[int, ...],
                       order: Sequence[int]) -> BalloonsProbeOutcome:
        """The cached outcome of one release order of ``subset``;
        ``candidate_outcomes`` lists the orders as ``permutations`` does."""
        return outcomes[list(permutations(subset)).index(tuple(order))]

    def _train_racks(self) -> List[Tuple[int, Tuple[int, ...]]]:
        """(box material, sorted balloon colours) of every train level."""
        return [(int(round(task.init.get(self._box, "color"))),
                 tuple(
                     sorted(
                         int(round(task.init.get(b, "color")))
                         for b in self._active_balloons(task.init))))
                for task in self.get_train_tasks()]

    @classmethod
    def _rack_seen(cls, subset: Tuple[int, ...],
                   unit_colors: Sequence[Sequence[int]], box_color: int,
                   train_racks: Sequence[Tuple[int, Tuple[int, ...]]]) -> bool:
        """Whether some train rack on this box held every colour the clips of
        ``subset`` free, as many times over, so training could have measured
        their rest height."""
        palette = Counter(cls.union_colors(unit_colors, subset))
        return any(box == box_color and not palette - Counter(rack)
                   for box, rack in train_racks)

    def _could_compose(
            self, in_band: Sequence[Tuple[int, ...]],
            unit_colors: Sequence[Sequence[int]], box_color: int,
            train_racks: Sequence[Tuple[int, Tuple[int, ...]]]) -> bool:
        """Analytic pre-screen of a composition level, before any rollout:

        some unseen in-band subset frees two or more balloons.
        """
        return any(
            len(self.union_colors(unit_colors, subset)) >= 2 and
            not self._rack_seen(subset, unit_colors, box_color, train_racks)
            for subset in in_band)

    def naive_release_bursts(self, state: State, subset: Tuple[int,
                                                               ...]) -> bool:
        """Whether freeing ``subset`` weakest lift first, letting the box
        settle between releases, bursts a balloon: the order and timing of an
        agent that reads rest heights alone and plays safe."""
        order = self.weakest_first_order(self.unit_colors(state), subset)
        return self._run_release_sequence(state, order, settled=True).burst

    def composition_decoy(
        self, candidates: Dict[Tuple[int, ...], List[BalloonsProbeOutcome]],
        unit_colors: Sequence[Sequence[int]], box_color: int,
        train_racks: Sequence[Tuple[int, Tuple[int, ...]]]
    ) -> Optional[Tuple[int, ...]]:
        """The in-band subset a static reading picks, or None.

        Every in-band subset with a winning order must free at least two
        balloons whose colours never shared a train rack on this box;
        otherwise None. The decoy is the lowest-hanging in-band subset,
        the fewest-lift choice; ``naive_release_bursts`` certifies that
        its weakest-first release (and every other in-band subset's)
        bursts, while the reference wins in another order.
        """

        def seen(subset: Tuple[int, ...]) -> bool:
            return self._rack_seen(subset, unit_colors, box_color, train_racks)

        def hover(subset: Tuple[int, ...]) -> float:
            z = self.hover_height(box_color,
                                  self.union_colors(unit_colors, subset))
            assert z is not None
            return z

        for subset, outcomes in candidates.items():
            if any(o.won for o in outcomes) and (
                    len(self.union_colors(unit_colors, subset)) < 2
                    or seen(subset)):
                return None
        return min(candidates, key=hover)

    def _could_bundle(
            self, in_band: Sequence[Tuple[int, ...]],
            reachable: Sequence[Tuple[Tuple[int, ...], float]],
            unit_colors: Sequence[Sequence[int]], box_color: int,
            band: Tuple[float, float],
            train_racks: Sequence[Tuple[int, Tuple[int, ...]]]) -> bool:
        """Analytic pre-screen of a bundle level, before any rollout: an unseen
        in-band subset of two or more clips whose stack rests clear of the
        ceiling, an in-band single clip (the decoy), and two clips resting
        below the band that each belong to some multi-clip in-band subset (the
        first cut and the tempting clip)."""

        def clears_ceiling(subset: Tuple[int, ...]) -> bool:
            return band[1] + 0.02 <= self.burst_height(len(subset))

        if not any(
                len(subset) >= 2 and clears_ceiling(subset) and not self.
                _rack_seen(subset, unit_colors, box_color, train_racks)
                for subset in in_band):
            return False
        if not any(len(subset) == 1 for subset in in_band):
            return False
        rest = {subset[0]: z for subset, z in reachable if len(subset) == 1}
        below = [
            clip for clip, z in rest.items()
            if z < band[0] and any(clip in subset and len(subset) >= 2
                                   for subset in in_band)
        ]
        return len(below) >= 2

    def _bundle_report(
            self, state: State,
            candidates: Dict[Tuple[int, ...],
                             List[BalloonsProbeOutcome]]) -> str:
        """One line per in-band candidate order: its witnessed status."""
        units = self.unit_colors(state)
        box_color = int(round(state.get(self._box, "color")))
        lines = []
        for subset, outcomes in candidates.items():
            z = self.hover_height(box_color, self.union_colors(units, subset))
            for order, outcome in zip(permutations(subset), outcomes):
                lines.append(f"{list(order)} rest {z:.3f}: {outcome.status}")
        return "; ".join(lines)

    def bundle_level(
        self, state: State, candidates: Dict[Tuple[int, ...],
                                             List[BalloonsProbeOutcome]],
        train_racks: Sequence[Tuple[int, Tuple[int, ...]]]
    ) -> Optional[Tuple[List[int], int, int]]:
        """(reference order, decoy clip, tempting clip) when ``state`` is a
        bundle level by the module doc's conditions, else None.

        ``candidates`` are the settled outcomes of every order of every
        analytic in-band clip subset. The decoy clip rests in the band
        by the analytic law and bursts when cut first; the tempting clip
        rests below the band, above the reference's first cut, and has
        an analytic in-band continuation that loses.
        """
        box_color = int(round(state.get(self._box, "color")))
        units = self.unit_colors(state)
        lo = float(state.get(self._band, "lo"))
        winning = {
            subset: [
                list(order)
                for order, outcome in zip(permutations(subset), outcomes)
                if outcome.won
            ]
            for subset, outcomes in candidates.items()
        }
        winning = {s: orders for s, orders in winning.items() if orders}
        if not winning or any(
                self._rack_seen(s, units, box_color, train_racks)
                for s in winning):
            return None
        firsts = {order[0] for orders in winning.values() for order in orders}
        if len(firsts) != 1:
            return None
        first = firsts.pop()
        if all(len(s) < 2 for s in winning):
            return None
        weakest = min(range(len(units)),
                      key=lambda i: self.unit_lift(units[i]))
        if first == weakest:
            return None
        decoys = sorted(s[0] for s, outcomes in candidates.items()
                        if len(s) == 1 and any(o.burst for o in outcomes))
        if not decoys:
            return None
        rest = {
            i: self.hover_height(box_color, units[i])
            for i in range(len(units))
        }
        first_rest = rest[first]
        if first_rest is None or first_rest >= lo:
            return None
        tempting = sorted(
            i for i, z in rest.items()
            if i != first and z is not None and first_rest < z < lo and any(
                i in s and len(s) >= 2 for s in candidates))
        reference = self.solution_subset(state)
        if reference is None:
            return None
        return (self.reference_order(state, reference), decoys[0],
                tempting[0] if tempting else -1)

    def _hold_action(self) -> Action:
        """A no-op action that holds the robot at its initial joints, for
        rolling the free-and-rise dynamics forward with no arm motion."""
        arr = np.array(self._pybullet_robot.initial_joint_positions,
                       dtype=np.float32)
        n = self.action_space.shape[0]
        if arr.shape[0] < n:
            arr = np.concatenate(
                [arr, np.zeros(n - arr.shape[0], dtype=np.float32)])
        return Action(arr)

    def _probe_outcome(self,
                       state: State,
                       steps: int,
                       status: str,
                       wall_supported: bool = False) -> BalloonsProbeOutcome:
        return BalloonsProbeOutcome(status, steps,
                                    float(state.get(self._box, "z")),
                                    float(state.get(self._box, "speed")),
                                    wall_supported)

    def _wall_support(self) -> bool:
        """Contact alone is not a jam: require vertical load on a wall."""
        for wall in self._chute_ids:
            for contact in p.getContactPoints(
                    self._box.id, wall,
                    physicsClientId=self._physics_client_id):
                vertical_force = (contact[9] * contact[7][2] +
                                  contact[10] * contact[11][2] +
                                  contact[12] * contact[13][2])
                if abs(vertical_force) > 1e-4:
                    return True
        return False

    def _wait_probe(
            self,
            state: State,
            max_steps: int,
            history: Optional[List[State]] = None) -> BalloonsProbeOutcome:
        """Check every frame for success; only sustained rest ends failure.

        ``history`` is the trailing window of states the evaluator's
        dwell needs (the frames of the releases that preceded the wait).
        """
        evaluator = self._evaluator()
        history = list(history) if history is not None else [state]
        positions: List[np.ndarray] = []
        supported = 0
        action = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        for step in range(max_steps + 1):
            if any_popped(state) is not None:
                return self._probe_outcome(state, step, "burst")
            if evaluator.terminated_trajectory(history):
                return self._probe_outcome(state, step, "won")
            angular = p.getBaseVelocity(
                self._box.id, physicsClientId=self._physics_client_id)[1]
            if box_at_rest(state, self._box) and np.linalg.norm(angular) < .01:
                positions.append(
                    np.array(
                        [state.get(self._box, f) for f in ("x", "y", "z")]))
                positions = positions[-CFG.balloons_probe_rest_steps:]
                supported = supported + 1 if self._wall_support() else 0
                if (not self._InBand_holds(state, [self._box, self._band])
                        and len(positions) == CFG.balloons_probe_rest_steps
                        and np.max(np.ptp(positions, axis=0)) <=
                        CFG.balloons_probe_rest_tol):
                    return self._probe_outcome(
                        state, step, "resting_outside",
                        supported >= CFG.balloons_probe_rest_steps)
            else:
                positions.clear()
                supported = 0
            if step < max_steps:
                state = self.simulate(state, action)
                history.append(state)
                history = history[-evaluator.dwell_steps - 1:]
        return self._probe_outcome(state, max_steps, "unresolved")

    def _evaluator(self) -> BalloonsEvaluator:
        """The evaluator of the current band goal, for the probes."""
        return BalloonsEvaluator(
            {GroundAtom(self._InBand, [self._box, self._band])})

    def assess_subset(self, state: State,
                      subset: Tuple[int, ...]) -> BalloonsProbeOutcome:
        """Free a subset simultaneously and classify the observed rollout.

        This is a screening rollout. Task acceptance also checks
        executable release orders. A timeout is unresolved, never a jam
        or a known loss.
        """
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        current = self._get_state().copy()
        for i in subset:
            current.set(self._clips[i], "is_on", 1.0)
        # Apply the releases before inspecting rest at the initial frame.
        current = self.simulate(current, self._hold_action())
        result = self._wait_probe(current,
                                  int(CFG.balloons_probe_max_steps) - 1)
        return BalloonsProbeOutcome(result.status, result.steps + 1,
                                    result.height, result.speed,
                                    result.wall_supported)

    def subset_outcome(self, state: State,
                       subset: Tuple[int, ...]) -> Tuple[bool, bool]:
        """Compatibility view: (witnessed win, burst).

        False/False includes unresolved motion and is NOT evidence of a
        jam. Use assess_subset for task selection or failure
        classification.
        """
        result = self.assess_subset(state, subset)
        return result.won, result.burst

    def release_sequence_outcome(
            self,
            state: State,
            order: Sequence[int],
            settled: bool = False) -> BalloonsProbeOutcome:
        """Classify a sequence, verifying contact failures counterfactually.

        Vertical wall contact by itself does not prove the wall caused a
        failure. Call it a jam only if the same sequence wins when box-
        wall collisions are disabled in a diagnostic replay. Restore
        collisions even if that replay fails; the task's actual physics
        is unchanged. With ``settled`` the box rests between releases.
        """
        result = self._run_release_sequence(state, order, settled)
        if result.status != "resting_outside" or not result.wall_supported:
            return result
        try:
            for wall in self._chute_ids:
                p.setCollisionFilterPair(
                    self._box.id,
                    wall,
                    -1,
                    -1,
                    False,
                    physicsClientId=self._physics_client_id)
            without_wall = self._run_release_sequence(state, order)
        finally:
            for wall in self._chute_ids:
                p.setCollisionFilterPair(
                    self._box.id,
                    wall,
                    -1,
                    -1,
                    True,
                    physicsClientId=self._physics_client_id)
        return replace(result, wall_free_won=without_wall.won)

    def _hold_until_rest(
            self, state: State, history: List[State],
            evaluator: BalloonsEvaluator) -> Tuple[State, int, Optional[str]]:
        """Hold the arm still until the box rests, a balloon bursts or the goal
        is met: (final state, steps, terminal status or None).

        ``history`` is extended in place for the evaluator's dwell.
        """
        action = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        rest = 0
        for step in range(int(CFG.balloons_probe_max_steps)):
            if any_popped(state) is not None:
                return state, step, "burst"
            if evaluator.terminated_trajectory(history):
                return state, step, "won"
            rest = rest + 1 if box_at_rest(state, self._box) else 0
            if rest >= int(CFG.balloons_probe_rest_steps):
                return state, step, None
            state = self.simulate(state, action)
            history.append(state)
            del history[:-evaluator.dwell_steps - 1]
        return state, int(CFG.balloons_probe_max_steps), None

    def _run_release_sequence(self,
                              state: State,
                              order: Sequence[int],
                              settled: bool = False) -> BalloonsProbeOutcome:
        """Execute Release skills, then hold to an outcome.

        Detect wins and bursts during each skill, matching continual
        play. Without ``settled`` the skills run back to back; with it
        the box comes to rest between releases, the timing of an agent
        that waits after each clip. This certifies the supplied order
        and timing only.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.balloons.options import \
            probe_release_option, release_params
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        current = self._get_state()
        self._current_observation = current
        history = [current]
        evaluator = self._evaluator()
        release = probe_release_option()
        steps = 0
        for index in order:
            option = release.ground([self._robot, self._clips[index]],
                                    release_params())
            if not option.initiable(current):
                return self._probe_outcome(current, steps, "skill_failed")
            try:
                for _ in range(int(CFG.balloons_probe_max_steps)):
                    if option.terminal(current):
                        break
                    current = self._step_once(option.policy(current))
                    self._current_observation = current
                    steps += 1
                    history.append(current)
                    history = history[-evaluator.dwell_steps - 1:]
                    if any_popped(current) is not None:
                        return self._probe_outcome(current, steps, "burst")
                    if evaluator.terminated_trajectory(history):
                        return self._probe_outcome(current, steps, "won")
                else:
                    return self._probe_outcome(current, steps, "unresolved")
            except utils.OptionExecutionFailure:
                return self._probe_outcome(current, steps, "skill_failed")
            if settled and index != order[-1]:
                current, held, status = self._hold_until_rest(
                    current, history, evaluator)
                self._current_observation = current
                steps += held
                if status is not None:
                    return self._probe_outcome(current, steps, status)
        result = self._wait_probe(current, int(CFG.balloons_probe_max_steps),
                                  history)
        return BalloonsProbeOutcome(result.status, result.steps + steps,
                                    result.height, result.speed,
                                    result.wall_supported)

    @staticmethod
    def _draw_covering(rng: np.random.Generator, n: int,
                       box_colors: Sequence[int], palette: Sequence[int],
                       seen_boxes: Set[int],
                       seen_colors: Set[int]) -> Tuple[int, List[int]]:
        """A box material and ``n`` distinct balloon colours, preferring the
        ones earlier train levels have not shown."""
        unseen_boxes = [b for b in box_colors if b not in seen_boxes]
        box_color = int(rng.choice(unseen_boxes or list(box_colors)))
        unseen = [c for c in palette if c not in seen_colors]
        rng.shuffle(unseen)
        colors = [int(c) for c in unseen[:n]]
        others = [c for c in palette if c not in colors]
        if len(colors) < n:
            colors += [
                int(c) for c in rng.choice(
                    others, size=n - len(colors), replace=False)
            ]
        rng.shuffle(colors)
        return box_color, colors

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.balloons.oracle import solve_level
        counts = list(CFG.balloons_num_balloons_train if train else CFG.
                      balloons_num_balloons_test)
        box_colors = list(CFG.balloons_box_colors_train if train else CFG.
                          balloons_box_colors_test)
        half = float(CFG.balloons_band_half)
        # Train levels together show every balloon colour and every box
        # material the split allows, so a test level composes lifts and
        # a mass the agent has seen rather than ones it must guess.
        palette = list(range(len(self.BALLOON_PALETTE)))
        seen_colors: Set[int] = set()
        seen_boxes: Set[int] = set()
        attempts = int(CFG.balloons_max_sampling_attempts)
        tasks = []
        # Retry draws can revisit identical candidates. Cache classifications
        # only within this generation call, with exact band bounds.
        rejected_levels: Set[Tuple[Any, ...]] = set()
        # Every test level is a composition level (see the module doc); with
        # bundle sizes configured it is a bundle level instead.
        bundle_sizes = [int(s) for s in CFG.balloons_test_bundle_sizes
                        ] if not train else []
        compose = not train and not bundle_sizes
        train_racks = self._train_racks() if not train else []
        for _ in range(num_tasks):
            found = None
            fallback: Optional[Tuple[State, Tuple[int, ...],
                                     Optional[Tuple[int, ...]]]] = None
            # A bundle level keeps drawing past the nominal attempts while
            # it has nothing at all, and stops at the first fallback then.
            for attempt in range(attempts * 3 if bundle_sizes else attempts):
                if attempt >= attempts and fallback is not None:
                    break
                n = int(rng.choice(counts))
                clip_of: Optional[List[int]] = None
                # The covering draw first; a free draw for the last
                # quarter of the attempts, so a rack the filters below
                # keep rejecting does not sink the level.
                if train and attempt < 3 * attempts // 4:
                    box_color, colors = self._draw_covering(
                        rng, n, box_colors, palette, seen_boxes, seen_colors)
                elif bundle_sizes:
                    # A bundle rack: the configured bundle sizes in a random
                    # rack order, repeating colours once the rack outgrows
                    # the palette.
                    sizes = [int(s) for s in rng.permutation(bundle_sizes)]
                    n = sum(sizes)
                    colors = [
                        int(c) for c in rng.choice(
                            palette, size=n, replace=n > len(palette))
                    ]
                    clip_of = [
                        clip for clip, size in enumerate(sizes)
                        for _ in range(size)
                    ]
                else:
                    box_color = int(rng.choice(box_colors))
                    colors = [
                        int(c)
                        for c in rng.choice(palette, size=n, replace=False)
                    ]
                if bundle_sizes:
                    box_color = int(rng.choice(box_colors))
                units = self.unit_colors(
                    self.level_state(box_color, colors, (0.0, 0.0), clip_of))
                # Reachable equilibria. The transient (not a static stack-
                # clearance) decides whether a subset overshoots into the
                # ceiling, so each equilibrium-in-band candidate is rolled
                # forward below.
                # Bands must sit inside the chute (below its top) so the box
                # is gated by the walls through its whole ascent and settles
                # between them.
                reach_max = min(self.ceiling_z - self.ceiling_half_extents[2],
                                self.chute_z_hi) - 0.06
                reachable = [
                    (subset, z)
                    for subset, z in self.lifting_subsets(box_color, units)
                    if self.table_height + 0.12 <= z <= reach_max
                ]
                # Nearby analytic equilibria provide candidate bands.
                # Executable rollouts below establish the reference and
                # losing sequence; other subsets or orders may also win.
                centers = [
                    (reachable[a][1] + reachable[b][1]) / 2.0
                    for a in range(len(reachable))
                    for b in range(a + 1, len(reachable))
                    if abs(reachable[a][1] - reachable[b][1]) <= 2 * half
                ]
                if train or bundle_sizes:
                    centers += [z for _, z in reachable]
                if not train and CFG.balloons_require_jam_decoy:
                    # The strict contact challenge also tests bands centred
                    # on an equilibrium, requiring witnessed wall-supported
                    # rest away from the goal for its central candidate.
                    centers += [z for _, z in reachable]
                rng.shuffle(centers)
                for center_z in centers:
                    band = (center_z - half, center_z + half)
                    in_band = [
                        subset for subset, z in reachable
                        if band[0] <= z <= band[1]
                    ]
                    if not train and len(in_band) < 2:
                        continue
                    if compose and not self._could_compose(
                            in_band, units, box_color, train_racks):
                        continue
                    if bundle_sizes and not self._could_bundle(
                            in_band, reachable, units, box_color, band,
                            train_racks):
                        continue
                    level_key = (box_color, tuple(colors), tuple(clip_of
                                                                 or ()), band)
                    if level_key in rejected_levels:
                        continue
                    rejected_levels.add(level_key)
                    state = self.level_state(box_color, colors, band, clip_of)
                    if compose and not all(
                            self.naive_release_bursts(state, subset)
                            for subset in in_band):
                        logging.info(
                            "Balloons composition candidate: box %d rack %s "
                            "band %.3f-%.3f in-band %s: a weakest-first "
                            "release survives", box_color, colors, band[0],
                            band[1], in_band)
                        continue
                    # Success/failure is measured through actual skills,
                    # including every intermediate frame and release order.
                    candidates = self.candidate_outcomes(state)
                    reference = self.solution_subset(state)
                    if compose or bundle_sizes:
                        logging.info(
                            "Balloons %s candidate: box %d rack %s clips %s "
                            "band %.3f-%.3f in-band %s reference %s",
                            "bundle" if bundle_sizes else "composition",
                            box_color, colors, clip_of, band[0], band[1],
                            in_band, reference)
                    if reference is None:
                        continue
                    decoys = [(subset, outcome)
                              for subset, results in candidates.items()
                              for outcome in results
                              if outcome.burst or outcome.jammed]
                    if not train and not decoys:
                        continue
                    decoy = None
                    if compose:
                        decoy = self.composition_decoy(candidates, units,
                                                       box_color, train_racks)
                        if decoy is None:
                            continue
                    if bundle_sizes:
                        verdict = self.bundle_level(state, candidates,
                                                    train_racks)
                        if verdict is None:
                            logging.info(
                                "Balloons bundle candidate rejected: %s",
                                self._bundle_report(state, candidates))
                            continue
                        _, decoy_clip, tempting_clip = verdict
                        decoy = (decoy_clip, )
                        if tempting_clip < 0 and solve_level(
                                self, state) is not None:
                            # A level without a tempting bundle serves if
                            # the remaining draws find none with one.
                            if fallback is None:
                                fallback = (state, reference, decoy)
                            logging.info(
                                "Balloons bundle candidate kept as fallback "
                                "(no tempting bundle): %s",
                                self._bundle_report(state, candidates))
                            continue
                    if not train and CFG.balloons_require_jam_decoy:
                        eqz = dict(reachable)
                        distances = {
                            subset: abs(eqz[subset] - center_z)
                            for subset in in_band
                        }
                        central = min(distances, key=distances.__getitem__)
                        if not any(subset == central and outcome.jammed
                                   and abs(eqz[subset] - eqz[reference]) <=
                                   CFG.balloons_contact_height_tol
                                   for subset, outcome in decoys):
                            continue
                    if solve_level(self, state) is None:
                        continue
                    rejected_levels.discard(level_key)
                    found = (state, reference, decoy)
                    break
                if found is not None:
                    break
            if found is None and fallback is not None:
                logging.info(
                    "Balloons bundle level: no draw with a tempting "
                    "bundle in %d attempts; using the fallback", attempts)
                found = fallback
            if found is None:
                raise RuntimeError(
                    "No balloon level with a verified winning reference and "
                    f"verified decoys in {attempts} draws "
                    f"(require_jam_decoy={CFG.balloons_require_jam_decoy}, "
                    f"bundle_sizes={bundle_sizes}). "
                    "Unresolved rollouts are not failures; the requested "
                    "decoy property may be unavailable under this physics.")
            state, subset, decoy = found
            seen_boxes.add(box_color)
            seen_colors.update(colors)
            goal = {GroundAtom(self._InBand, [self._box, self._band])}
            balloons = self._active_balloons(state)
            bundles = self.bundles(state)
            if self.is_bundled(state):
                names = "; ".join(f"clip{clip} frees " + " and ".join(
                    f"{self.balloon_color_name(state.get(b, 'color'))} "
                    f"({b.name})" for b in (balloons[i] for i in bundle))
                                  for clip, bundle in enumerate(bundles))
                holding = (
                    f"The balloons are tied in bundles, one clip per "
                    f"bundle, each bundle racked in a row behind its clip; "
                    f"opening a clip frees every balloon of its bundle at "
                    f"once, and each balloon's clip feature names its clip: "
                    f"{names}.")
            else:
                names = ", ".join(
                    f"{self.balloon_color_name(state.get(b, 'color'))} "
                    f"({b.name}, clip{i})" for i, b in enumerate(balloons))
                holding = (f"Each balloon is held by the clip in front of "
                           f"it: {names}.")
            goal_nl = (
                f"Open clips to free balloons so that the "
                f"{self.box_color_name(box_color)} box floats up and hangs "
                f"still with its centre inside the green band "
                f"({state.get(self._band, 'lo'):.2f} to "
                f"{state.get(self._band, 'hi'):.2f} m). {holding} A balloon "
                f"that reaches the ceiling bursts and the level is lost; a "
                f"freed balloon cannot be clipped back. Success requires "
                f"remaining inside the band at speed below "
                f"{CFG.balloons_settle_speed:g} m/s for "
                f"{CFG.balloons_goal_dwell_steps} consecutive environment "
                f"steps.")
            metrics = {
                f"solution_{b.name}":
                float(self.clip_index(state, b) in subset)
                for b in balloons
            }
            metrics.update({
                f"solution_clip{clip}": float(clip in subset)
                for clip in range(len(bundles))
            })
            metrics["task_generation_version"] = 3.0
            metrics["goal_dwell_steps"] = float(CFG.balloons_goal_dwell_steps)
            if compose or bundle_sizes:
                assert decoy is not None
                metrics.update({
                    f"decoy_{b.name}":
                    float(self.clip_index(state, b) in decoy)
                    for b in balloons
                })
            if bundle_sizes:
                verdict = self.bundle_level(state,
                                            self.candidate_outcomes(state),
                                            train_racks)
                assert verdict is not None
                order, decoy_clip, tempting = verdict
                metrics["bundle_level"] = 1.0
                metrics["bundle_decoy_clip"] = float(decoy_clip)
                metrics["bundle_tempting_clip"] = float(tempting)
                metrics["bundle_reference_cuts"] = float(len(order))
                metrics["bundle_reference_first_clip"] = float(order[0])
            metrics["witnessed_winning_candidate_subsets"] = float(
                sum(
                    any(o.won for o in results)
                    for results in self.candidate_outcomes(state).values()))
            tasks.append(
                EnvironmentTask(state,
                                goal,
                                goal_nl=goal_nl,
                                evaluator=BalloonsEvaluator(goal),
                                offline_task_metrics=metrics))
        return self._add_pybullet_state_to_tasks(tasks)

    # =========================================================================
    # PROBE
    # =========================================================================
    def run_option(self, state: State, option: Any,
                   max_steps: int) -> Optional[State]:
        """Execute a grounded option from ``state`` on this instance, then hold
        until the box has settled; None if it cannot run."""
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        obs = self._get_state()
        self._current_observation = obs
        if not option.initiable(obs):
            return None
        try:
            for _ in range(max_steps):
                if option.terminal(obs):
                    break
                obs = self._step_once(option.policy(obs))
        except utils.OptionExecutionFailure:
            return None
        result = self._wait_probe(self._get_state(), max_steps)
        if result.status == "unresolved":
            return None
        return self._get_state()


_PROBE_ENV: Optional[PyBulletBalloonsEnv] = None


def probe_env() -> PyBulletBalloonsEnv:
    """A dedicated instance for probes."""
    global _PROBE_ENV  # pylint: disable=global-statement
    if _PROBE_ENV is None:
        _PROBE_ENV = PyBulletBalloonsEnv(use_gui=False)
    return _PROBE_ENV
