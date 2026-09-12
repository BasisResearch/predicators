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
"""
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
    """Win when the box hangs at rest inside the band; a burst balloon loses
    the level."""

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
        return ("The level is won when the box hangs at rest with its centre "
                "inside the band. A balloon that reaches the ceiling bursts "
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
        if CFG.balloons_task_generation not in ("original", "validated"):
            raise ValueError("balloons_task_generation must be "
                             "original or validated")
        if (CFG.balloons_task_generation == "original"
                and CFG.balloons_scene != "chute"):
            raise ValueError(
                "Original task generation requires balloons_scene=chute")
        self._original_solution_cache: Dict[Tuple[Any, ...],
                                            Optional[Tuple[int, ...]]] = {}
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
        """Release, pop, and pull: an open clip frees its balloon, which its
        string seats on the box's top; a freed balloon that reaches the ceiling
        bursts; every intact freed balloon pulls the box up with its colour's
        lift at the box's height while its string holds it to the box."""
        state = self._get_state()
        box_top = self.box_top_point(state, self._box)
        box_z = float(state.get(self._box, "z"))
        balloons = self._active_balloons(state)
        clips = self._active_clips(state)
        commands: List[PhysicsCommand] = []
        stacked = sum(1 for b in balloons if self._tied.get(b.name, False))
        for index, balloon in enumerate(balloons):
            name = balloon.name
            if not self._tied.get(name, False):
                if index >= len(clips) or not self._is_clip_on(clips[index]):
                    continue
                # Freed: the string pulls the balloon to the end of its
                # tether above the box's top centre, above any balloon
                # freed before it, so its pull acts through the box.
                self._tied[name] = True
                offset = self._attach_offset(index, len(balloons))
                seat = (box_top[0] + offset, box_top[1],
                        box_top[2] + self.balloon_radius + self.string_length +
                        2 * self.balloon_radius * stacked)
                stacked += 1
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
    def lifting_subsets(
            cls, box_color: int, balloon_colors: Sequence[int]
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Every subset (as balloon indices) that lifts the box, with its hover
        height, highest first."""
        out = []
        indices = list(range(len(balloon_colors)))
        for size in range(1, len(indices) + 1):
            for subset in combinations(indices, size):
                z = cls.hover_height(box_color,
                                     [balloon_colors[i] for i in subset])
                if z is not None:
                    out.append((subset, z))
        return sorted(out, key=lambda item: -item[1])

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

    def level_state(self, box_color: int, balloon_colors: Sequence[int],
                    band: Tuple[float, float]) -> State:
        """A level: the box on the table, balloons clipped in the rack, the
        band."""
        init: Dict[Object, Dict[str, float]] = {
            self._robot: self._robot_init_dict()
        }
        init[self._box] = {
            "x": self.box_xy[0],
            "y": self.box_xy[1],
            "z": self.table_height + self.box_half_extents()[2],
            "color": float(box_color),
            "speed": 0.0,
        }
        xs = self.rack_xs(len(balloon_colors))
        for i, color in enumerate(balloon_colors):
            init[self._balloons[i]] = {
                "x": xs[i],
                "y": self.rack_y,
                "z": self.balloon_rest_z,
                "color": float(color),
                "tied": 0.0,
                "popped": 0.0,
            }
            init[self._clips[i]] = {
                "x": xs[i],
                "y": self.clip_y,
                "z": self.table_height,
                "rot": self.clip_rot,
                "is_on": 0.0,
            }
        if CFG.balloons_scene == "hatch":
            for obj, features in init.items():
                if obj.type.name in {"box", "balloon"}:
                    features.update(roll=0.0, pitch=0.0, yaw=0.0)
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

        Reconstruct the clean level, so oracle predicates keep a stable
        reference as execution progresses. These are immediate Release
        sequences followed by a hold, not all possible release timings.
        """
        box_color = int(round(state.get(self._box, "color")))
        colors = [
            int(round(state.get(b, "color")))
            for b in self._active_balloons(state)
        ]
        lo, hi = (float(state.get(self._band, f)) for f in ("lo", "hi"))
        key = (box_color, tuple(colors), lo, hi, tuple(CFG.balloons_lifts),
               tuple(CFG.balloons_box_masses), CFG.balloons_fade_height,
               CFG.balloons_drag, CFG.balloons_probe_max_steps,
               CFG.balloons_probe_rest_steps, CFG.balloons_probe_rest_tol,
               CFG.balloons_settle_speed, CFG.balloons_scene,
               self.box_half_extents(), tuple(self.obstacle_geometry()),
               CFG.balloons_hatch_attach_span,
               CFG.skill_phase_use_motion_planning, CFG.seed,
               CFG.balloons_push_approach, CFG.balloons_push_contact_z,
               tuple(sorted(self._param_overrides.items())))
        if key not in self._candidate_cache:
            clean = self.level_state(box_color, colors, (lo, hi))
            self._candidate_cache[key] = {
                subset: [
                    self.release_sequence_outcome(clean, order)
                    for order in permutations(subset)
                ]
                for subset, z in self.lifting_subsets(box_color, colors)
                if lo <= z <= hi
            }
        return {
            subset: list(results)
            for subset, results in self._candidate_cache[key].items()
        }

    def solution_subset(self, state: State) -> Optional[Tuple[int, ...]]:
        """Select a reference using the configured task-generation criteria.

        Validated tasks require every tested order of the reference to
        win, preferring fewer releases and then lexicographic order.
        Original tasks retain the historical simultaneous-release
        screen.
        """
        if CFG.balloons_task_generation == "original":
            from predicators.envs.balloons_original_tasks import \
                solution_subset  # pylint: disable=import-outside-toplevel
            return solution_subset(self, state)
        candidates = self.candidate_outcomes(state)
        robust = [
            subset for subset, outcomes in candidates.items()
            if outcomes and all(outcome.won for outcome in outcomes)
        ]
        return min(robust, key=lambda subset: (len(subset), subset)) \
            if robust else None

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

    def _wait_probe(self, state: State,
                    max_steps: int) -> BalloonsProbeOutcome:
        """Check every frame for success; only sustained rest ends failure."""
        positions: List[np.ndarray] = []
        supported = 0
        action = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        for step in range(max_steps + 1):
            if any_popped(state) is not None:
                return self._probe_outcome(state, step, "burst")
            if self._InBand_holds(state, [self._box, self._band]):
                return self._probe_outcome(state, step, "won")
            angular = p.getBaseVelocity(
                self._box.id, physicsClientId=self._physics_client_id)[1]
            if box_at_rest(state, self._box) and np.linalg.norm(angular) < .01:
                positions.append(
                    np.array(
                        [state.get(self._box, f) for f in ("x", "y", "z")]))
                positions = positions[-CFG.balloons_probe_rest_steps:]
                supported = supported + 1 if self._wall_support() else 0
                if (len(positions) == CFG.balloons_probe_rest_steps
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
        return self._probe_outcome(state, max_steps, "unresolved")

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

    def release_sequence_outcome(self, state: State,
                                 order: Sequence[int]) -> BalloonsProbeOutcome:
        """Classify a sequence, verifying contact failures counterfactually.

        Vertical wall contact by itself does not prove the wall caused a
        failure. Call it a jam only if the same sequence wins when box-
        wall collisions are disabled in a diagnostic replay. Restore
        collisions even if that replay fails; the task's actual physics
        is unchanged.
        """
        result = self._run_release_sequence(state, order)
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

    def _run_release_sequence(self, state: State,
                              order: Sequence[int]) -> BalloonsProbeOutcome:
        """Execute Release skills without extra waits, then hold to an outcome.

        Detect wins and bursts during each skill, matching continual
        play. This certifies the supplied order only, not arbitrary
        release timing.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.balloons.options import \
            probe_release_option, release_params
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        current = self._get_state()
        self._current_observation = current
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
                    if any_popped(current) is not None:
                        return self._probe_outcome(current, steps, "burst")
                    if self._InBand_holds(current, [self._box, self._band]):
                        return self._probe_outcome(current, steps, "won")
                else:
                    return self._probe_outcome(current, steps, "unresolved")
            except utils.OptionExecutionFailure:
                return self._probe_outcome(current, steps, "skill_failed")
        result = self._wait_probe(current, int(CFG.balloons_probe_max_steps))
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
        if CFG.balloons_task_generation == "original":
            from predicators.envs.balloons_original_tasks import \
                make_tasks  # pylint: disable=import-outside-toplevel
            return make_tasks(self, num_tasks, rng, train)
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
        for _ in range(num_tasks):
            found = None
            for attempt in range(attempts):
                n = int(rng.choice(counts))
                # The covering draw first; a free draw for the last
                # quarter of the attempts, so a rack the filters below
                # keep rejecting does not sink the level.
                if train and attempt < 3 * attempts // 4:
                    box_color, colors = self._draw_covering(
                        rng, n, box_colors, palette, seen_boxes, seen_colors)
                else:
                    box_color = int(rng.choice(box_colors))
                    colors = [
                        int(c)
                        for c in rng.choice(palette, size=n, replace=False)
                    ]
                # Reachable equilibria. The transient (not a static stack-
                # clearance) decides whether a subset overshoots into the
                # ceiling, so each equilibrium-in-band candidate is rolled
                # forward below.
                # Bands must sit inside the chute (below its top) so the box
                # is gated by the walls through its whole ascent and settles
                # between them.
                reach_max = min(self.ceiling_z - self.ceiling_half_extents[2],
                                self.chute_z_hi) - 0.06
                reach_min = self.table_height + 0.12
                if CFG.balloons_scene == "hatch":
                    # Put the entire payload above the hatch at the goal,
                    # including at its most tilted orientation.
                    reach_min = (CFG.balloons_hatch_z +
                                 CFG.balloons_hatch_half_thickness +
                                 np.linalg.norm(self.box_half_extents()) +
                                 half)
                reachable = [
                    (subset, z)
                    for subset, z in self.lifting_subsets(box_color, colors)
                    if reach_min <= z <= reach_max
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
                if train:
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
                    level_key = (box_color, tuple(colors), band)
                    if level_key in rejected_levels:
                        continue
                    rejected_levels.add(level_key)
                    state = self.level_state(box_color, colors, band)
                    # Success/failure is measured through actual skills,
                    # including every intermediate frame and release order.
                    candidates = self.candidate_outcomes(state)
                    reference = self.solution_subset(state)
                    if reference is None:
                        continue
                    decoys = [(subset, outcome)
                              for subset, results in candidates.items()
                              for outcome in results
                              if outcome.burst or outcome.jammed]
                    if not train and not decoys:
                        continue
                    if not train and CFG.balloons_scene == "hatch" and not any(
                            len(subset) == len(reference) and outcomes and any(
                                outcome.jammed for outcome in outcomes)
                            for subset, outcomes in candidates.items()):
                        # Equal counts prevent a fewest-balloons shortcut.
                        # Contact depends on release order; certify a losing
                        # sequence without claiming all orders of a set lose.
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
                    found = (state, reference)
                    break
                if found is not None:
                    break
            if found is None:
                raise RuntimeError(
                    "No balloon level with a verified winning reference and "
                    f"verified decoys in {attempts} draws "
                    f"(require_jam_decoy={CFG.balloons_require_jam_decoy}). "
                    "Unresolved rollouts are not failures; the requested "
                    "decoy property may be unavailable under this physics.")
            state, subset = found
            seen_boxes.add(box_color)
            seen_colors.update(colors)
            goal = {GroundAtom(self._InBand, [self._box, self._band])}
            balloons = self._active_balloons(state)
            names = ", ".join(
                f"{self.balloon_color_name(state.get(b, 'color'))} "
                f"({b.name}, clip{i})" for i, b in enumerate(balloons))
            goal_nl = (
                f"Open clips to free balloons so that the "
                f"{self.box_color_name(box_color)} box floats up and hangs "
                f"still with its centre inside the green band "
                f"({state.get(self._band, 'lo'):.2f} to "
                f"{state.get(self._band, 'hi'):.2f} m). Each balloon is "
                f"held by the clip in front of it: {names}. A balloon that "
                f"reaches the ceiling bursts and the level is lost; a freed "
                f"balloon cannot be clipped back.")
            if CFG.balloons_scene == "hatch":
                dims = tuple(2 * size for size in self.box_half_extents())
                goal_nl += (
                    f" The payload dimensions are {dims} m. "
                    f"A horizontal hatch at z={CFG.balloons_hatch_z:g} m "
                    f"has an opening {2 * CFG.balloons_hatch_half_gap:g} m "
                    f"wide, centred at x="
                    f"{self.box_xy[0] + CFG.balloons_hatch_offset_x:g} m. "
                    "The hatch panels collide with the payload. "
                    "The target band is above the hatch.")
            metrics = {
                f"solution_{b.name}": float(i in subset)
                for i, b in enumerate(balloons)
            }
            metrics["task_generation_version"] = (4.0 if CFG.balloons_scene
                                                  == "hatch" else 3.0)
            metrics["validated_motion_planning"] = float(
                CFG.skill_phase_use_motion_planning)
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
