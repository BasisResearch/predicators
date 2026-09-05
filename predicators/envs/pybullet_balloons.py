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

Design notes, and why this domain favours a model of the physics.

**The answer is a subset and a height.** The goal is a band of
heights; the box floats to the height where the freed balloons' fading
lift equals its weight. Which clips to open is a small choice, but the
height each choice gives is a quantitative composition of per colour
lifts, the fade, and the box's mass. Open one clip too many and the
box rises to the ceiling, where a balloon bursts and the level is
lost; a freed balloon cannot be clipped back.

**Test extends train.** Test levels bring a heavier box colour and one
more balloon; the balloon colours keep their lift.
"""
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce, Attach, \
    PhysicsCommand
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, StepOption, TaskEvaluator, Type

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


class PyBulletBalloonsEnv(PyBulletBalloonsBaseEnv):
    """A balloon puzzle whose lifts, fade and box masses must be learned."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._param_overrides: Dict[str, float] = {}
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

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._box_type, self._balloon_type,
            self._clip_type, self._band_type
        }

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

    def believed_box_mass(self, color_index: int) -> float:
        """The box mass an instance running the visible physics uses."""
        name = self.box_color_name(color_index)
        return float(
            self._param_overrides.get(f"mass_{name}", self.box_base_mass))

    def _box_mass_for(self, color_index: int) -> float:
        if self._skip_domain_specific_dynamics:
            return self.believed_box_mass(color_index)
        return self.true_box_mass(color_index)

    def _drag(self) -> float:
        if self._skip_domain_specific_dynamics:
            return float(self._param_overrides.get("air_drag", 0.04))
        return float(CFG.balloons_drag)

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """A mass per box material, and the air's drag on moving bodies."""
        info: Dict[str, Dict[str, Any]] = {}
        for index, (name, _) in enumerate(self.BOX_PALETTE):
            info[f"mass_{name}"] = {
                "default": self.believed_box_mass(index),
                "lo": 0.01,
                "hi": 1.0,
                "scale": "log",
                "description": f"mass of a {name} box, in kilograms",
            }
        info["air_drag"] = {
            "default":
            self._drag(),
            "lo":
            0.01,
            "hi":
            40.0,
            "scale":
            "log",
            "description": ("linear damping of the box and the balloons: "
                            "the engine's velocity decay per second"),
        }
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        unknown = set(params) - set(self.get_physical_param_info())
        if unknown:
            raise ValueError(f"Unknown physical params: {sorted(unknown)}")
        self._param_overrides.update({k: float(v) for k, v in params.items()})
        if self._current_observation is not None:
            self._apply_dynamics(self._active_balloons(self._current_state))

    def _apply_dynamics(self, balloons: List[Object]) -> None:
        p.changeDynamics(self._box.id,
                         -1,
                         mass=self._box_mass_for(self._box_color),
                         linearDamping=self._drag(),
                         physicsClientId=self._physics_client_id)
        for balloon in balloons:
            p.changeDynamics(balloon.id,
                             -1,
                             linearDamping=self._drag(),
                             physicsClientId=self._physics_client_id)

    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
        self._apply_dynamics(self._active_balloons(state))

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
                seat = (box_top[0], box_top[1],
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
            "z": self.box_z,
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
        init[self._band] = {
            "x": self.box_xy[0] + self.band_offset_x,
            "y": self.box_xy[1],
            "lo": band[0],
            "hi": band[1],
        }
        return utils.create_state_from_dict(init)

    def solution_subset(self, state: State) -> Optional[Tuple[int, ...]]:
        """The balloons (indices into the level's balloons) whose lift, by the
        analytic law, hangs the box inside the band; None when no subset does
        or more than one does."""
        box_color = int(round(state.get(self._box, "color")))
        balloons = self._active_balloons(state)
        colors = [int(round(state.get(b, "color"))) for b in balloons]
        lo, hi = state.get(self._band, "lo"), state.get(self._band, "hi")
        hits = [
            subset for subset, z in self.lifting_subsets(box_color, colors)
            if lo <= z <= hi
        ]
        if len(hits) != 1:
            return None
        return hits[0]

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.balloons.oracle import solve_level
        counts = list(CFG.balloons_num_balloons_train if train else CFG.
                      balloons_num_balloons_test)
        box_colors = list(CFG.balloons_box_colors_train if train else CFG.
                          balloons_box_colors_test)
        half = float(CFG.balloons_band_half)
        tasks = []
        for _ in range(num_tasks):
            found = None
            for _ in range(int(CFG.balloons_max_sampling_attempts)):
                n = int(rng.choice(counts))
                box_color = int(rng.choice(box_colors))
                colors = [
                    int(c) for c in rng.choice(
                        len(self.BALLOON_PALETTE), size=n, replace=False)
                ]
                subsets = self.lifting_subsets(box_color, colors)
                # Bands that a single subset reaches and that keep its
                # stack of balloons clear of the ceiling.
                candidates = []
                for subset, z in subsets:
                    z_max = self.ceiling_z - self.ceiling_half_extents[2] - \
                        self.box_half - self.string_length - \
                        (2 * len(subset) + 1) * self.balloon_radius - 0.03
                    if not self.table_height + 0.12 <= z <= z_max:
                        continue
                    others = [oz for other, oz in subsets if other != subset]
                    if all(abs(oz - z) > 2 * half + 0.01 for oz in others):
                        candidates.append((subset, z))
                if not candidates:
                    continue
                subset, z = candidates[int(rng.integers(len(candidates)))]
                state = self.level_state(box_color, colors,
                                         (z - half, z + half))
                if self.solution_subset(state) != subset:
                    continue
                if solve_level(self, state) is None:
                    continue
                found = (state, subset)
                break
            if found is None:
                raise RuntimeError(
                    "No balloon level with a unique floating subset the "
                    f"oracle clears in {CFG.balloons_max_sampling_attempts} "
                    "draws.")
            state, subset = found
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
            metrics = {
                f"solution_{b.name}": float(i in subset)
                for i, b in enumerate(balloons)
            }
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
        hold = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        moved = False
        for i in range(max_steps):
            obs = self._step_once(hold)
            moving = self._speed(self._box) > CFG.balloons_settle_speed
            moved |= moving
            if moved and not moving:
                break
            if not moved and i > 20:
                break
        return self._get_state()


_PROBE_ENV: Optional[PyBulletBalloonsEnv] = None


def probe_env() -> PyBulletBalloonsEnv:
    """A dedicated instance for probes."""
    global _PROBE_ENV  # pylint: disable=global-statement
    if _PROBE_ENV is None:
        _PROBE_ENV = PyBulletBalloonsEnv(use_gui=False)
    return _PROBE_ENV
