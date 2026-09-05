"""The launcher environment: the launch law, block masses, tasks, predicates.

The observable simulation core (rail, plunger, ball, stands, blocks,
the reload) lives in :mod:`predicators.envs.pybullet_launcher_base`,
which may be surfaced to learning agents as reference source. This
module holds everything an agent must LEARN or must not see:

* the LAUNCH LAW - how fast the ball leaves the muzzle for a given
  plunger compression (linear in the compression, with a hidden
  spring constant), along the barrel's visible elevation;
* the MATERIALS - how heavy each block colour is;
* task generation (the train/test distribution) and goal semantics.

Design notes, and why this domain favours a model of the physics.

**One knob, a flight, a collision.** The only thing the robot chooses
is how far back it pushes the plunger. What follows is a ballistic
flight along the barrel's elevation and an impact on a tower of
blocks: whether the ball strikes the top block, or the one below it,
or the stand, or nothing, is a matter of centimetres of trajectory
height at the tower's distance, and what the strike does to the tower
depends on the masses. A goal asks for the top block down and the
rest of the tower standing, so the strike has to be high enough to
take the top block and low enough to take nothing else.

**Few shots.** Each level has a handful of balls. A shot that misses
is a spare gone, and on a test level with two balls a trial-and-error
search for the right compression is over before it has bracketed
anything. An agent that has identified the spring constant from a
training shot or two rolls the flight forward in the base simulator,
reads the trajectory height at the tower off it, and picks the
compression that puts the ball on the top block.

**Test extends train.** Test towers are taller and stand farther away,
with fewer spare balls; the launcher and the materials are the same.

Example commands::

    python predicators/main.py --env pybullet_launcher \
        --approach oracle_process_planning --seed 0 \
        --num_train_tasks 0 --num_test_tasks 5 \
        --sesame_check_expected_atoms False
"""
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.commands import SetVelocity
from predicators.envs.pybullet_launcher_base import PyBulletLauncherBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, StepOption, TaskEvaluator, Type

# Tilt (radians) beyond which a block counts as toppled.
TOPPLE_TILT = 0.5


# =============================================================================
# CLASSIFIERS
# =============================================================================
def block_toppled(state: State, block: Object) -> bool:
    """A block that has tilted over, dropped off its tower or slid off its
    stand."""
    if abs(state.get(block, "roll")) > TOPPLE_TILT or \
            abs(state.get(block, "pitch")) > TOPPLE_TILT:
        return True
    stand = next((o for o in state if o.type.name == "stand"), None)
    if stand is None:
        return False
    env_cls = PyBulletLauncherBaseEnv
    stand_top = state.get(stand, "z") + env_cls.stand_half_extents[2]
    if state.get(block, "z") < stand_top + env_cls.block_half - 0.02:
        return True
    dx = state.get(block, "x") - state.get(stand, "x")
    dy = state.get(block, "y") - state.get(stand, "y")
    return bool(np.hypot(dx, dy) > 0.05)


def ball_at_rest(state: State) -> bool:
    """Whether the ball has stopped moving."""
    ball = next(o for o in state if o.type.name == "ball")
    return state.get(ball, "speed") < CFG.launcher_settle_speed


def ball_loaded(state: State) -> bool:
    """Whether the ball rests in the muzzle cup."""
    ball = next(o for o in state if o.type.name == "ball")
    cx, cy, cz = PyBulletLauncherBaseEnv.cup_position()
    dist = np.linalg.norm([
        state.get(ball, "x") - cx,
        state.get(ball, "y") - cy,
        state.get(ball, "z") - cz
    ])
    return bool(dist < PyBulletLauncherBaseEnv.cup_radius) and \
        ball_at_rest(state)


def out_of_balls(state: State) -> bool:
    """No spare left and the loaded ball has flown and stopped."""
    launcher = next(o for o in state if o.type.name == "launcher")
    if state.get(launcher, "balls_left") > 0.5:
        return False
    return ball_at_rest(state) and not ball_loaded(state)


class LauncherEvaluator(TaskEvaluator):
    """Win when the goal holds with the ball at rest; lose when a block the
    goal wants standing has toppled, or when the balls run out.

    Both losses are absorbing, so ``terminated`` fires on them and
    ``_certify`` refuses them, ending the episode as GAME_OVER.
    """

    def __init__(self, goal: Set[GroundAtom]) -> None:
        super().__init__(goal)
        self._protected = [
            atom.objects[0] for atom in goal
            if atom.predicate.name == "Standing"
        ]

    def _protected_toppled(self, state: State) -> Optional[Object]:
        for block in self._protected:
            if block in state.data and block_toppled(state, block):
                return block
        return None

    def terminated(self, state: State) -> bool:
        if self._protected_toppled(state) is not None:
            return True
        if all(atom.holds(state) for atom in self.goal):
            return ball_at_rest(state)
        return out_of_balls(state)

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        del step_options, sim_env  # unused
        final = states[-1]
        block = self._protected_toppled(final)
        if block is not None:
            return False, f"{block.name} was meant to stay standing"
        if not all(atom.holds(final) for atom in self.goal):
            return False, "out of balls"
        return True, ""

    def objective_description(self) -> str:
        return ("The level is won when the marked block has toppled and "
                "every other block of the tower is still standing. Toppling "
                "a block the goal wants standing, or running out of balls, "
                "ends the level.")


class PyBulletLauncherEnv(PyBulletLauncherBaseEnv):
    """A spring launcher whose launch law and block masses must be learned."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._mass_overrides: Dict[str, float] = {}
        # Launch-law state: the compression at the previous step, and the
        # deepest compression since the handle was last home.
        self._peak_compression: float = 0.0
        super().__init__(use_gui, **kwargs)

        self._Toppled = Predicate(
            "Toppled", [self._block_type],
            self._Toppled_holds,
            natural_language_assertion=lambda os: f"block {os[0]} has toppled")
        self._Standing = Predicate("Standing", [self._block_type],
                                   self._Standing_holds,
                                   natural_language_assertion=lambda os:
                                   f"block {os[0]} is standing where it was")
        self._Loaded = Predicate("Loaded", [self._launcher_type],
                                 self._Loaded_holds,
                                 natural_language_assertion=lambda os:
                                 f"a ball rests in the muzzle of {os[0]}")
        self._OutOfBalls = Predicate("OutOfBalls", [self._launcher_type],
                                     self._OutOfBalls_holds,
                                     natural_language_assertion=lambda os:
                                     f"{os[0]} has no ball left to fire")
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds,
                                    natural_language_assertion=lambda os:
                                    f"robot {os[0]} is not holding anything")

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_launcher"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Toppled, self._Standing, self._Loaded, self._OutOfBalls,
            self._HandEmpty
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Toppled, self._Standing}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._launcher_type, self._ball_type,
            self._stand_type, self._block_type
        }

    # =========================================================================
    # HIDDEN PHYSICS
    # =========================================================================
    @classmethod
    def true_mass(cls, color_index: int) -> float:
        """The mass of a block material, the learning target."""
        return float(CFG.launcher_block_masses[int(round(color_index))])

    def believed_mass(self, color_index: int) -> float:
        """The mass an instance running the visible physics uses."""
        name = self.color_name(color_index)
        return float(
            self._mass_overrides.get(f"mass_{name}", self.block_base_mass))

    def _mass_for(self, color_index: int) -> float:
        if self._skip_domain_specific_dynamics:
            return self.believed_mass(color_index)
        return self.true_mass(color_index)

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """One tunable mass per block material."""
        info: Dict[str, Dict[str, Any]] = {}
        for index, (name, _) in enumerate(self.COLOR_PALETTE):
            info[f"mass_{name}"] = {
                "default": self.believed_mass(index),
                "lo": 0.02,
                "hi": 3.0,
                "scale": "log",
                "description": f"mass of a {name} block, in kilograms",
            }
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        unknown = set(params) - set(self.get_physical_param_info())
        if unknown:
            raise ValueError(f"Unknown physical params: {sorted(unknown)}")
        self._mass_overrides.update({k: float(v) for k, v in params.items()})
        if self._current_observation is not None:
            self._apply_block_masses(self._active_blocks(self._current_state))

    def _apply_block_masses(self, blocks: List[Object]) -> None:
        for block in blocks:
            color = self._block_colors.get(block.name, 0)
            p.changeDynamics(block.id,
                             -1,
                             mass=self._mass_for(color),
                             physicsClientId=self._physics_client_id)

    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
        self._apply_block_masses(self._active_blocks(state))
        self._peak_compression = self._compression

    def _domain_specific_step(self) -> None:
        """The launch law: when the handle snaps home from a compression, the
        loaded ball leaves along the barrel at the spring constant times the
        deepest compression reached."""
        # The handle's position at the end of this action's physics is
        # what the camera saw; the snap home (a reset by the visible
        # mechanics) is the release.
        before = self._compression_before_snap
        self._peak_compression = max(self._peak_compression, before)
        released = self._compression <= 0.0 < before
        if released:
            peak = self._peak_compression
            self._peak_compression = 0.0
            if peak >= CFG.launcher_min_compression and self._ball_in_cup():
                speed = float(CFG.launcher_spring_k) * peak
                dx, dy, dz = self.launch_direction(self.launch_angle)
                self.queue_residual_commands([
                    SetVelocity(self._ball.name,
                                (speed * dx, speed * dy, speed * dz), None)
                ])

    # =========================================================================
    # LAUNCH PROBE
    # =========================================================================
    def launch_outcome(self, state: State, depth: float) -> Optional[State]:
        """The rink after a real Cock at ``depth`` from ``state`` and the
        ball's flight, or None when the skill could not execute.

        Runs the Cock skill itself (no motion planning) on this
        instance, then holds until the ball is at rest. Overwrites this
        instance's state: callers that own a live episode use
        :func:`probe_env`.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.launcher.options import \
            probe_cock_option

        # Start from the home joint configuration so the outcome is a
        # function of the state alone (see the ice rink's push probe).
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        obs = self._get_state()
        self._current_observation = obs
        params = np.array(
            [CFG.launcher_push_approach, CFG.launcher_push_contact_z, depth],
            dtype=np.float32)
        option = probe_cock_option().ground([self._robot, self._launcher],
                                            params)
        if not option.initiable(obs):
            return None
        # The snap, and so the launch, happens while the arm retreats,
        # inside the option; the flight may be over before it ends.
        launched = False
        try:
            for _ in range(int(CFG.launcher_probe_max_steps)):
                if option.terminal(obs):
                    break
                obs = self._step_once(option.policy(obs))
                launched |= self._ball_speed() > CFG.launcher_settle_speed
        except utils.OptionExecutionFailure:
            return None
        hold = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        for _ in range(int(CFG.launcher_probe_max_steps)):
            obs = self._step_once(hold)
            moving = self._ball_speed() > CFG.launcher_settle_speed
            launched |= moving
            if launched and not moving:
                break
        return self._get_state()

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _Toppled_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return block_toppled(state, block)

    @staticmethod
    def _Standing_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return not block_toppled(state, block)

    @staticmethod
    def _Loaded_holds(state: State, objects: Sequence[Object]) -> bool:
        del objects
        return ball_loaded(state)

    @staticmethod
    def _OutOfBalls_holds(state: State, objects: Sequence[Object]) -> bool:
        del objects
        return out_of_balls(state)

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    # =========================================================================
    # TASK GENERATION
    # =========================================================================
    # Compressions the generator scans for a working window.
    scan_step: ClassVar[float] = 0.0025
    _scan_depths: ClassVar[np.ndarray] = np.round(
        np.arange(0.02, 0.1001, scan_step), 4)

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

    def level_state(self, stand_x: float, colors: Sequence[int],
                    balls_left: int) -> State:
        """A level: a tower of ``colors`` on a stand at ``stand_x``, the top
        block marked, the ball loaded, ``balls_left`` spares."""
        init: Dict[Object, Dict[str, float]] = {
            self._robot: self._robot_init_dict()
        }
        init[self._launcher] = {
            "x": self.handle_rest_x,
            "y": self.rail_y,
            "z": self.handle_z,
            "angle": float(self.launch_angle),
            "compression": 0.0,
            "balls_left": float(balls_left),
        }
        cx, cy, cz = self.cup_position()
        init[self._ball] = {"x": cx, "y": cy, "z": cz, "speed": 0.0}
        stand_z = self.table_height + self.stand_half_extents[2]
        init[self._stand] = {"x": stand_x, "y": self.rail_y, "z": stand_z}
        stand_top = stand_z + self.stand_half_extents[2]
        for i, color in enumerate(colors):
            init[self._blocks[i]] = {
                "x": stand_x,
                "y": self.rail_y,
                "z": stand_top + (2 * i + 1) * self.block_half,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "color": float(color),
                "is_target": float(i == len(colors) - 1),
            }
        return utils.create_state_from_dict(init)

    def goal_for(self, state: State) -> Set[GroundAtom]:
        """Top block toppled, every other block standing."""
        blocks = self._active_blocks(state)
        goal = {GroundAtom(self._Toppled, [blocks[-1]])}
        for block in blocks[:-1]:
            goal.add(GroundAtom(self._Standing, [block]))
        return goal

    def working_depths(self, state: State) -> List[float]:
        """The scanned compressions whose launch reaches the level's goal from
        ``state``."""
        goal = self.goal_for(state)
        working = []
        for depth in self._scan_depths:
            end = self.launch_outcome(state, float(depth))
            if end is not None and all(atom.holds(end) for atom in goal):
                working.append(float(depth))
        return working

    @classmethod
    def _window_middle(cls, depths: List[float]) -> Optional[float]:
        """The middle of the longest run of consecutive scanned depths, if the
        run is at least three long: a shot from the middle of a wider window
        survives the arm arriving by a different path than the probe's."""
        if not depths:
            return None
        runs: List[List[float]] = [[depths[0]]]
        for d in depths[1:]:
            if abs(d - runs[-1][-1] - cls.scan_step) < 1e-6:
                runs[-1].append(d)
            else:
                runs.append([d])
        best = max(runs, key=len)
        if len(best) < 3:
            return None
        return float(best[len(best) // 2])

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        counts = list(CFG.launcher_num_blocks_train if train else CFG.
                      launcher_num_blocks_test)
        x_lo, x_hi = (CFG.launcher_stand_x_train
                      if train else CFG.launcher_stand_x_test)
        balls_left = int(CFG.launcher_balls_left_train if train else CFG.
                         launcher_balls_left_test)
        tasks = []
        for _ in range(num_tasks):
            num_blocks = int(rng.choice(counts))
            found = None
            for _ in range(int(CFG.launcher_max_sampling_attempts)):
                stand_x = float(rng.uniform(x_lo, x_hi))
                colors = [
                    int(c) for c in rng.integers(
                        0, len(self.COLOR_PALETTE), size=num_blocks)
                ]
                state = self.level_state(stand_x, colors, balls_left)
                depth = self._window_middle(self.working_depths(state))
                if depth is not None:
                    found = (state, depth)
                    break
            if found is None:
                raise RuntimeError(
                    f"No {num_blocks}-block tower with a working compression "
                    f"window in {CFG.launcher_max_sampling_attempts} draws.")
            state, depth = found
            goal = self.goal_for(state)
            blocks = self._active_blocks(state)
            others = ", ".join(b.name for b in blocks[:-1])
            goal_nl = (f"Fire the launcher so that the red top block "
                       f"({blocks[-1].name}) topples off the tower while the "
                       f"block{'s' if len(blocks) > 2 else ''} below it "
                       f"({others}) stay standing. You have "
                       f"{balls_left + 1} ball"
                       f"{'s' if balls_left else ''} in total.")
            tasks.append(
                EnvironmentTask(
                    state,
                    goal,
                    goal_nl=goal_nl,
                    evaluator=LauncherEvaluator(goal),
                    offline_task_metrics={"solution_depth": float(depth)}))
        return self._add_pybullet_state_to_tasks(tasks)


_PROBE_ENV: Optional[PyBulletLauncherEnv] = None


def probe_env() -> PyBulletLauncherEnv:
    """A dedicated instance for launch probes, so a helper predicate never
    disturbs the live episode's world."""
    global _PROBE_ENV  # pylint: disable=global-statement
    if _PROBE_ENV is None:
        _PROBE_ENV = PyBulletLauncherEnv(use_gui=False)
    return _PROBE_ENV
