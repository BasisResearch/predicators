"""The crane environment: materials, swing drag, tasks, predicates.

The observable simulation core (table, crane, hinged ram, crate,
bin) lives in :mod:`predicators.envs.pybullet_crane_base`, which
may be surfaced to learning agents as reference source. This module
holds everything an agent must LEARN or must not see:

* the MATERIALS - how heavy each crate colour is and how it grips the
  table, which together set how far a given blow sends it;
* the HINGE - how much the swing loses on its way to the crate;
* task generation (the train/test distribution) and goal semantics.

Design notes, and why this domain favours a model of the physics.

**One number, many causes.** The robot draws the ram back along its
arc and lets it go. The crate's landing spot is set by the pull, the
arm's length (a longer arm swings slower for the same pull), the
crate's mass (a heavy crate takes less of the ram's speed) and its
grip on the table (a grippy crate stops sooner). Only the pull is the
agent's; the length is visible; the rest is the colour's secret. Too
short a pull and the crate stops short of the bin, where the ram can
no longer reach it; too long and it slides off the far edge of the
table. Either loses the level.

**Test extends train.** Test levels bring a longer arm and a crate
material the training levels never showed; the training materials
keep their properties.
"""
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_crane_base import PyBulletCraneBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, StepOption, TaskEvaluator, Type


# =============================================================================
# CLASSIFIERS
# =============================================================================
def crate_at_rest(state: State, crate: Object) -> bool:
    """Whether the crate has stopped."""
    return state.get(crate, "speed") < CFG.crane_settle_speed


def crate_on_pad(state: State, crate: Object, bin_: Object) -> bool:
    """Whether the crate's centre lies over the bin pad, moving or not."""
    return (abs(state.get(crate, "x") - state.get(bin_, "x")) <= state.get(
        bin_, "half") and abs(state.get(crate, "y") - state.get(bin_, "y")) <=
            PyBulletCraneBaseEnv.bin_half_across
            and state.get(crate, "z") > PyBulletCraneBaseEnv.table_height)


def crate_in_bin(state: State, crate: Object, bin_: Object) -> bool:
    """Whether the crate RESTS on the bin pad: over it and stopped, so a crate
    sliding across the pad does not count."""
    return crate_on_pad(state, crate, bin_) and crate_at_rest(state, crate)


def crate_fallen(state: State, crate: Object) -> bool:
    """Whether the crate has left the table top."""
    return state.get(crate, "z") < PyBulletCraneBaseEnv.table_height - 0.05


def crate_reachable(state: State, ram: Object, crate: Object) -> bool:
    """Whether the ram, drawn back as far as it goes, can still strike the
    crate: the crate's near face within the swing's reach along the lane,
    and the crate still in the lane."""
    reach = state.get(ram, "x") + PyBulletCraneBaseEnv.max_pull() + \
        PyBulletCraneBaseEnv.head_half
    near_face = state.get(crate, "x") - PyBulletCraneBaseEnv.crate_half
    in_lane = abs(state.get(crate, "y") -
                  state.get(ram, "y")) < PyBulletCraneBaseEnv.crate_half + 0.04
    return near_face <= reach and in_lane and not crate_fallen(state, crate)


class CraneEvaluator(TaskEvaluator):
    """Win when the crate rests on the bin; a crate off the table or out of the
    ram's reach loses the level."""

    def _objects(self, state: State) -> Tuple[Object, Object, Object]:
        ram = next(o for o in state if o.type.name == "ram")
        crate = next(o for o in state if o.type.name == "crate")
        bin_ = next(o for o in state if o.type.name == "bin")
        return ram, crate, bin_

    def terminated(self, state: State) -> bool:
        ram, crate, bin_ = self._objects(state)
        if crate_fallen(state, crate):
            return True
        if not crate_at_rest(state, crate):
            return False
        if crate_in_bin(state, crate, bin_):
            return all(atom.holds(state) for atom in self.goal)
        return not crate_reachable(state, ram, crate)

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        del step_options, sim_env  # unused
        ram, crate, bin_ = self._objects(states[-1])
        if crate_fallen(states[-1], crate):
            return False, "the crate fell off the table"
        if not crate_in_bin(states[-1], crate, bin_):
            if not crate_reachable(states[-1], ram, crate):
                return False, "the crate stopped out of the ram's reach"
            return False, "the crate is not on the bin"
        return True, ""

    def objective_description(self) -> str:
        return ("The level is won when the crate rests on the bin pad. A "
                "crate that slides off the table, or stops where the ram "
                "can no longer reach it, ends the level.")


class PyBulletCraneEnv(PyBulletCraneBaseEnv):
    """A wrecking-ram puzzle whose crate materials must be learned."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._param_overrides: Dict[str, float] = {}
        super().__init__(use_gui, **kwargs)
        self._InBin = Predicate("InBin", [self._crate_type, self._bin_type],
                                self._InBin_holds,
                                natural_language_assertion=lambda os:
                                f"crate {os[0]} rests on bin {os[1]}")
        self._AtRest = Predicate(
            "AtRest", [self._crate_type],
            self._AtRest_holds,
            natural_language_assertion=lambda os: f"crate {os[0]} is still")
        self._OnTable = Predicate(
            "OnTable", [self._crate_type],
            self._OnTable_holds,
            natural_language_assertion=lambda os: f"crate {os[0]} is on the "
            "table")
        self._Fallen = Predicate("Fallen", [self._crate_type],
                                 self._Fallen_holds,
                                 natural_language_assertion=lambda os:
                                 f"crate {os[0]} has fallen off the table")
        self._Reachable = Predicate(
            "Reachable", [self._ram_type, self._crate_type],
            self._Reachable_holds,
            natural_language_assertion=lambda os:
            f"ram {os[0]} can still strike crate {os[1]}")
        self._RamStill = Predicate(
            "RamStill", [self._ram_type],
            self._RamStill_holds,
            natural_language_assertion=lambda os: f"ram {os[0]} hangs still")
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds,
                                    natural_language_assertion=lambda os:
                                    f"robot {os[0]} is not holding anything")

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_crane"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InBin, self._AtRest, self._OnTable, self._Fallen,
            self._Reachable, self._RamStill, self._HandEmpty
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InBin}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._ram_type, self._crane_type,
            self._crate_type, self._bin_type
        }

    # =========================================================================
    # HIDDEN PHYSICS
    # =========================================================================
    @classmethod
    def true_crate_mass(cls, color_index: int) -> float:
        """The mass of a crate material, a learning target."""
        return float(CFG.crane_crate_masses[int(round(color_index))])

    @classmethod
    def true_crate_friction(cls, color_index: int) -> float:
        """The grip of a crate material on the table, a learning target."""
        return float(CFG.crane_crate_frictions[int(round(color_index))])

    def believed_crate_mass(self, color_index: int) -> float:
        """The crate mass an instance running the visible physics uses."""
        name = self.color_name(color_index)
        return float(
            self._param_overrides.get(f"mass_{name}", self.crate_base_mass))

    def believed_crate_friction(self, color_index: int) -> float:
        """The crate friction an instance running the visible physics uses."""
        name = self.color_name(color_index)
        return float(
            self._param_overrides.get(f"friction_{name}",
                                      self.crate_base_friction))

    def _crate_mass_for(self, color_index: int) -> float:
        if self._skip_domain_specific_dynamics:
            return self.believed_crate_mass(color_index)
        return self.true_crate_mass(color_index)

    def _crate_friction_for(self, color_index: int) -> float:
        if self._skip_domain_specific_dynamics:
            return self.believed_crate_friction(color_index)
        return self.true_crate_friction(color_index)

    def _swing_damping(self) -> float:
        if self._skip_domain_specific_dynamics:
            return float(
                self._param_overrides.get("swing_damping",
                                          self.hinge_base_damping))
        return float(CFG.crane_swing_damping)

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """A mass and a table friction per crate material, and the hinge's drag
        on the swinging ram."""
        info: Dict[str, Dict[str, Any]] = {}
        for index, (name, _) in enumerate(self.COLOR_PALETTE):
            info[f"mass_{name}"] = {
                "default": self.believed_crate_mass(index),
                "lo": 0.01,
                "hi": 2.0,
                "scale": "log",
                "description": f"mass of a {name} crate, in kilograms",
            }
            info[f"friction_{name}"] = {
                "default":
                self.believed_crate_friction(index),
                "lo":
                0.05,
                "hi":
                1.5,
                "scale":
                "log",
                "description": (f"lateral friction coefficient between a "
                                f"{name} crate and the table"),
            }
        info["swing_damping"] = {
            "default":
            self._swing_damping(),
            "lo":
            0.0005,
            "hi":
            0.5,
            "scale":
            "log",
            "description": ("damping of the ram's hinge, the engine's joint "
                            "damping in newton metre seconds per radian"),
        }
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        unknown = set(params) - set(self.get_physical_param_info())
        if unknown:
            raise ValueError(f"Unknown physical params: {sorted(unknown)}")
        self._param_overrides.update({k: float(v) for k, v in params.items()})
        if self._current_observation is not None:
            self._apply_dynamics()

    def _apply_dynamics(self) -> None:
        p.changeDynamics(self._crate.id,
                         -1,
                         mass=self._crate_mass_for(self._crate_color),
                         lateralFriction=self._crate_friction_for(
                             self._crate_color),
                         physicsClientId=self._physics_client_id)
        p.changeDynamics(self._ram.id,
                         0,
                         jointDamping=self._swing_damping(),
                         physicsClientId=self._physics_client_id)

    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
        self._apply_dynamics()

    def _domain_specific_step(self) -> None:
        """Nothing beyond the engine: what this env hides are the numbers the
        materials put into it, applied when a state is set."""

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _InBin_holds(state: State, objects: Sequence[Object]) -> bool:
        crate, bin_ = objects
        return crate_in_bin(state, crate, bin_)

    @staticmethod
    def _AtRest_holds(state: State, objects: Sequence[Object]) -> bool:
        crate, = objects
        return crate_at_rest(state, crate)

    @staticmethod
    def _OnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        crate, = objects
        return not crate_fallen(state, crate)

    @staticmethod
    def _Fallen_holds(state: State, objects: Sequence[Object]) -> bool:
        crate, = objects
        return crate_fallen(state, crate)

    @staticmethod
    def _Reachable_holds(state: State, objects: Sequence[Object]) -> bool:
        ram, crate = objects
        return crate_reachable(state, ram, crate)

    @staticmethod
    def _RamStill_holds(state: State, objects: Sequence[Object]) -> bool:
        ram, = objects
        return state.get(ram, "speed") < CFG.crane_settle_speed

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    # =========================================================================
    # THE SWING, FOR THE GENERATOR AND THE ORACLE
    # =========================================================================
    def swing_outcome(self, state: State, pull: float) -> Optional[State]:
        """The lane after a real Pull of ``pull`` from ``state`` and the swing,
        or None when the skill could not execute.

        Runs the Pull skill itself (no motion planning) on this
        instance, then holds until the crate has stopped and the ram can
        no longer reach it. Overwrites this instance's state: callers
        that own a live episode use :func:`probe_env`.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.crane.options import \
            probe_pull_option

        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        obs = self._get_state()
        self._current_observation = obs
        params = np.array(
            [CFG.crane_push_approach, CFG.crane_push_contact_z, pull],
            dtype=np.float32)
        option = probe_pull_option().ground(
            [self._robot, self._ram, self._crane], params)
        if not option.initiable(obs):
            return None
        try:
            for _ in range(int(CFG.crane_probe_max_steps)):
                if option.terminal(obs):
                    break
                obs = self._step_once(option.policy(obs))
        except utils.OptionExecutionFailure:
            return None
        hold = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        rest_x = obs.get(self._ram, "x")
        still = 0
        window: List[float] = []
        for _ in range(int(CFG.crane_probe_max_steps)):
            obs = self._step_once(hold)
            if crate_fallen(obs, self._crate):
                break
            window.append(self.head_position()[0] - rest_x)
            window = window[-30:]
            still = still + 1 if crate_at_rest(obs, self._crate) else 0
            near_face = obs.get(self._crate, "x") - self.crate_half
            if still >= 15 and len(window) == 30 and \
                    rest_x + max(window) + self.head_half < near_face:
                break
        return self._get_state()

    # Pulls the generator scans for a working window.
    scan_step: ClassVar[float] = 0.01

    @classmethod
    def scan_pulls(cls) -> List[float]:
        """The pulls scanned for a working window."""
        lo, hi = CFG.crane_pull_range
        return [
            float(v)
            for v in np.round(np.arange(lo, hi + 1e-9, cls.scan_step), 3)
        ]

    def swing_outcomes(self,
                       state: State) -> List[Tuple[float, Optional[State]]]:
        """The lane after each scanned pull from ``state``."""
        return [(pull, self.swing_outcome(state, pull))
                for pull in self.scan_pulls()]

    def _lands_on(self, end: Optional[State], bin_x: float, lane_y: float,
                  bin_half: float) -> bool:
        """Whether an outcome has the crate at rest on a pad centred at
        ``bin_x`` (the pad is a picture: it does not touch the dynamics, so one
        scan answers for every pad position)."""
        if end is None or crate_fallen(end, self._crate) or \
                not crate_at_rest(end, self._crate):
            return False
        return abs(end.get(self._crate, "x") - bin_x) <= bin_half and abs(
            end.get(self._crate, "y") - lane_y) <= self.bin_half_across

    def working_pulls(self, state: State) -> List[float]:
        """The scanned pulls whose swing lands the crate on the bin."""
        bin_x = float(state.get(self._bin, "x"))
        lane_y = float(state.get(self._bin, "y"))
        half = float(state.get(self._bin, "half"))
        return [
            pull for pull, end in self.swing_outcomes(state)
            if self._lands_on(end, bin_x, lane_y, half)
        ]

    @classmethod
    def window_middle(cls, pulls: List[float]) -> Optional[float]:
        """The middle of the longest run of consecutive scanned pulls, if the
        run is at least three long: a swing from the middle of a wider window
        survives the arm arriving by a different path than the probe's."""
        if not pulls:
            return None
        runs: List[List[float]] = [[pulls[0]]]
        for d in pulls[1:]:
            if abs(d - runs[-1][-1] - cls.scan_step) < 1e-6:
                runs[-1].append(d)
            else:
                runs.append([d])
        best = max(runs, key=len)
        if len(best) < 3:
            return None
        return float(best[len(best) // 2])

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

    def level_state(self, lane_y: float, length: float, gap: float, color: int,
                    bin_x: float, bin_half: float) -> State:
        """A level: the ram at rest on an arm of ``length`` over lane
        ``lane_y``, a crate ``gap`` along the lane from it, the bin pad centred
        at ``bin_x``."""
        init: Dict[Object, Dict[str, float]] = {
            self._robot: self._robot_init_dict()
        }
        init[self._crane] = {
            "x": self.ram_x,
            "y": lane_y,
            "z": self.ram_rest_z + length,
            "length": length,
        }
        init[self._ram] = {
            "x": self.ram_x,
            "y": lane_y,
            "z": self.ram_rest_z,
            "angle": 0.0,
            "speed": 0.0,
        }
        init[self._crate] = {
            "x": self.ram_x + gap,
            "y": lane_y,
            "z": self.crate_z,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "color": float(color),
            "speed": 0.0,
        }
        init[self._bin] = {"x": bin_x, "y": lane_y, "half": bin_half}
        return utils.create_state_from_dict(init)

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        lengths = list(
            CFG.crane_length_train if train else CFG.crane_length_test)
        colors = list(CFG.crane_crate_colors_train if train else CFG.
                      crane_crate_colors_test)
        gap_lo, gap_hi = CFG.crane_gap_range
        lane_lo, lane_hi = CFG.crane_lane_y_range
        bin_half = float(CFG.crane_bin_half)
        tasks = []
        for _ in range(num_tasks):
            found = None
            for _ in range(int(CFG.crane_max_sampling_attempts)):
                lane_y = float(rng.uniform(lane_lo, lane_hi))
                length = float(rng.choice(lengths))
                gap = float(rng.uniform(gap_lo, gap_hi))
                color = int(rng.choice(colors))
                # The pad is placed where some swing actually sends the
                # crate: scan first, then centre the pad on one outcome
                # from the middle of the range and keep the level if
                # that gives a wide enough working window.
                bin_lo = self.ram_x + gap + self.crate_half + bin_half
                bin_hi = self.table_x_ub - bin_half - 0.04
                state = self.level_state(lane_y, length, gap, color, bin_lo,
                                         bin_half)
                outcomes = self.swing_outcomes(state)
                landing = [
                    end.get(self._crate, "x") for _, end in outcomes
                    if end is not None and crate_at_rest(end, self._crate)
                    and not crate_fallen(end, self._crate)
                    and bin_lo <= end.get(self._crate, "x") <= bin_hi
                ]
                if not landing:
                    continue
                bin_x = float(rng.choice(landing))
                working = [
                    pull for pull, end in outcomes
                    if self._lands_on(end, bin_x, lane_y, bin_half)
                ]
                pull = self.window_middle(working)
                if pull is not None:
                    state = self.level_state(lane_y, length, gap, color, bin_x,
                                             bin_half)
                    found = (state, pull)
                    break
            if found is None:
                raise RuntimeError(
                    "No crane level with a working pull window in "
                    f"{CFG.crane_max_sampling_attempts} draws.")
            state, pull = found
            goal = {GroundAtom(self._InBin, [self._crate, self._bin])}
            goal_nl = (
                f"Draw the crane's ram back along the lane and let it "
                f"swing so that the {self.color_name(color)} crate slides "
                f"along the table and comes to rest on the green bin pad. "
                f"The crate loses the level if it slides off the far edge "
                f"of the table or stops where the ram can no longer reach "
                f"it.")
            tasks.append(
                EnvironmentTask(state,
                                goal,
                                goal_nl=goal_nl,
                                evaluator=CraneEvaluator(goal),
                                offline_task_metrics={"solution_pull": pull}))
        return self._add_pybullet_state_to_tasks(tasks)


_PROBE_ENV: Optional[PyBulletCraneEnv] = None


def probe_env() -> PyBulletCraneEnv:
    """A dedicated instance for probes."""
    global _PROBE_ENV  # pylint: disable=global-statement
    if _PROBE_ENV is None:
        _PROBE_ENV = PyBulletCraneEnv(use_gui=False)
    return _PROBE_ENV
