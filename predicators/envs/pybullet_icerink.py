"""The ice rink environment: material friction, patch drag, tasks, predicates.

The observable simulation core (slab, walls, tiles, targets, the push
directions) lives in :mod:`predicators.envs.pybullet_icerink_base`,
which may be surfaced to learning agents as reference source. This
module holds everything an agent must LEARN or must not see:

* the MATERIALS - what each tile colour's sliding friction is, so how
  far a tile of that colour travels from one push;
* the PATCH - the dark strip across the rink drags a tile crossing it;
* task generation (the train/test distribution) and goal semantics.

Design notes, and why this domain favours a model of the physics.

**A puzzle whose answer is a stopping point.** A tile pushed along a
direction leaves the gripper at the push speed and slides until its
friction, a wall or another tile stops it. The goal is a set of
targets, one per tile, and a push either lands the tile on its target
or it does not. Nothing the robot can do afterwards pulls a tile back:
a tile against the far wall stays there, and a tile that slides past
an open edge has left the rink for good. So every push is a
prediction, and the prediction is the sliding distance of THIS colour
from THIS spot through whatever lies on its path.

**Two hidden scalars per colour, one law.** The friction coefficient of
each material is the whole hidden content per colour, and the patch's
drag is one more. They are ENGINE parameters and a force rule
respectively: an agent that identifies the coefficients from one slide
per colour rolls the base simulator forward and reads the stopping
point off it, walls, patch and collisions included. An agent that
reasons from a table of past slides has to try the push for real, and
a wrong push on a level without resets is the level.

**Test extends train.** Test rinks carry more tiles than training ones,
so more of the slab is occupied and more paths cross a tile or the
patch; the materials keep their colours and their friction. What is
learned about a colour on a two-tile rink is true on a four-tile one.

Example commands::

    # Oracle demo via process planning.
    python predicators/main.py --env pybullet_icerink \
        --approach oracle_process_planning --seed 0 \
        --num_train_tasks 0 --num_test_tasks 5 \
        --sesame_check_expected_atoms False
"""
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce, PhysicsCommand
from predicators.envs.pybullet_icerink_base import PyBulletIceRinkBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, StepOption, TaskEvaluator, Type

# The push directions the task generator and the oracle use. South
# pushes are legal but slide a tile toward the open near edge, and the
# arm's reach over a tile on the far half of the rink is marginal, so
# neither the generator nor the oracle plans one.
PLANNED_DIRECTIONS: Tuple[str, ...] = ("north", "east", "west")


# =============================================================================
# CLASSIFIERS (module level so the evaluator and the env share them)
# =============================================================================
def tile_on_target(state: State, tile: Object, target: Object) -> bool:
    """Whether ``tile`` rests within the goal tolerance of ``target``."""
    dx = abs(state.get(tile, "x") - state.get(target, "x"))
    dy = abs(state.get(tile, "y") - state.get(target, "y"))
    return dx < CFG.icerink_on_tol and dy < CFG.icerink_on_tol


def tile_lost(state: State, tile: Object) -> bool:
    """Whether ``tile`` has left the slab."""
    return not PyBulletIceRinkBaseEnv.on_rink(state.get(tile, "x"),
                                              state.get(tile, "y"))


def tile_at_rest(state: State, tile: Object) -> bool:
    """Whether ``tile`` has stopped sliding."""
    return state.get(tile, "speed") < CFG.icerink_settle_speed


class IceRinkEvaluator(TaskEvaluator):
    """Win when every tile rests on its target; lose a tile, lose the level.

    ``terminated`` fires either when the goal atoms hold with every tile
    at rest (a tile sliding THROUGH its target is not on it), or when
    any tile has left the slab - an absorbing state, since nothing can
    bring it back. ``_certify`` refuses the second case, so it ends the
    episode as GAME_OVER rather than WIN.
    """

    def terminated(self, state: State) -> bool:
        tiles = [o for o in state if o.type.name == "tile"]
        if any(tile_lost(state, t) for t in tiles):
            return True
        if not all(atom.holds(state) for atom in self.goal):
            return False
        return all(tile_at_rest(state, t) for t in tiles)

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        del step_options, sim_env  # unused
        final = states[-1]
        for tile in sorted((o for o in final if o.type.name == "tile"),
                           key=lambda o: o.name):
            if tile_lost(final, tile):
                return False, f"{tile.name} slid off the rink"
        return True, ""

    def objective_description(self) -> str:
        return ("The level is won when every tile rests on the target of "
                "its colour. A tile that slides off an open edge of the "
                "rink cannot be recovered and ends the level.")


class PyBulletIceRinkEnv(PyBulletIceRinkBaseEnv):
    """An ice rink whose tiles' materials must be learned by sliding them.

    Subclass of the observable sim core (see
    :mod:`predicators.envs.pybullet_icerink_base`); this class adds the
    per-material friction, the patch drag, task generation, the goal
    evaluator and the predicates.
    """

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Believed friction per material, for an instance that runs the
        # visible physics only (see ``get_physical_param_info``).
        self._friction_overrides: Dict[str, float] = {}
        super().__init__(use_gui, **kwargs)

        self._On = Predicate("On", [self._tile_type, self._target_type],
                             self._On_holds,
                             natural_language_assertion=lambda os:
                             f"tile {os[0]} rests on target {os[1]}")
        self._OnRink = Predicate("OnRink", [self._tile_type],
                                 self._OnRink_holds,
                                 natural_language_assertion=lambda os:
                                 f"tile {os[0]} is on the rink")
        self._Lost = Predicate("Lost", [self._tile_type],
                               self._Lost_holds,
                               natural_language_assertion=lambda os:
                               f"tile {os[0]} has slid off the rink")
        self._AtRest = Predicate("AtRest", [self._tile_type],
                                 self._AtRest_holds,
                                 natural_language_assertion=lambda os:
                                 f"tile {os[0]} is not moving")
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds,
                                    natural_language_assertion=lambda os:
                                    f"robot {os[0]} is not holding anything")

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_icerink"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnRink, self._Lost, self._AtRest, self._HandEmpty
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._tile_type, self._target_type,
            self._direction_type
        }

    # =========================================================================
    # HIDDEN PHYSICS
    # =========================================================================
    @classmethod
    def true_friction(cls, color_index: int) -> float:
        """The sliding friction of a material, the learning target."""
        return float(CFG.icerink_material_frictions[int(round(color_index))])

    def believed_friction(self, color_index: int) -> float:
        """The friction an instance running the visible physics uses."""
        name = self.color_name(color_index)
        return float(
            self._friction_overrides.get(f"friction_{name}",
                                         self.tile_base_friction))

    def _friction_for(self, color_index: int) -> float:
        if self._skip_domain_specific_dynamics:
            return self.believed_friction(color_index)
        return self.true_friction(color_index)

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """One tunable friction coefficient per material colour."""
        info: Dict[str, Dict[str, Any]] = {}
        for index, (name, _) in enumerate(self.COLOR_PALETTE):
            info[f"friction_{name}"] = {
                "default":
                self.believed_friction(index),
                "lo":
                0.01,
                "hi":
                1.0,
                "scale":
                "log",
                "description": (f"sliding friction coefficient of the "
                                f"{name} tiles on the rink"),
            }
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        unknown = set(params) - set(self.get_physical_param_info())
        if unknown:
            raise ValueError(f"Unknown physical params: {sorted(unknown)}")
        self._friction_overrides.update(
            {k: float(v)
             for k, v in params.items()})
        if self._current_observation is not None:
            tiles, _ = self._active_objects(self._current_state)
            self._apply_tile_frictions(tiles)

    def _apply_tile_frictions(self, tiles: List[Object]) -> None:
        for tile in tiles:
            color = self._tile_colors.get(tile.name, 0)
            self._set_tile_friction(tile, self._friction_for(color))

    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
        tiles, _ = self._active_objects(state)
        self._apply_tile_frictions(tiles)

    def _domain_specific_step(self) -> None:
        """The patch: a tile crossing the dark strip is braked.

        An extra Coulomb friction force against the tile's motion, held
        for the next action and queued through the same channel a
        learned rule uses, so a rule emitting the same command is bit-
        identical to the env. The force is capped at what would bring
        the tile to rest within the action, since a held force outlives
        the velocity it was computed from.
        """
        if not CFG.icerink_patch:
            return
        commands: List[PhysicsCommand] = []
        dt = float(CFG.pybullet_sim_steps_per_action) / 240.0
        for tile in self._tiles:
            if tile.name not in self._tile_colors or tile.id is None:
                continue
            (x, _, _), _ = p.getBasePositionAndOrientation(
                tile.id, physicsClientId=self._physics_client_id)
            if not self.on_patch(x):
                continue
            (vx, vy,
             _), _ = p.getBaseVelocity(tile.id,
                                       physicsClientId=self._physics_client_id)
            speed = float(np.hypot(vx, vy))
            if speed < 1e-4:
                continue
            brake = float(CFG.icerink_patch_friction) * self.tile_mass * 9.81
            brake = min(brake, self.tile_mass * speed / dt)
            commands.append(
                ApplyForce(tile.name,
                           (-brake * vx / speed, -brake * vy / speed, 0.0)))
        if commands:
            self.queue_residual_commands(commands)

    # =========================================================================
    # PUSH PROBE
    # =========================================================================
    def push_outcome(self, state: State, tile: Object, direction: str,
                     speed: float) -> Optional[Tuple[float, float, bool]]:
        """Where ``tile`` stops after a real push along ``direction`` at
        ``speed`` from ``state``.

        Runs the Push skill itself (transit by plain IK, no motion
        planning) on this instance, hidden dynamics included, then holds
        until every tile is at rest. Returns ``(x, y, lost)``, or None
        when the skill could not execute from here (a stroke the arm
        cannot reach). Overwrites this instance's state: callers that
        own a live episode use :func:`probe_env`.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.icerink.options import \
            probe_push_option

        # The arm's joint configuration is not part of a State, and a
        # redundant arm reaches the same pose from many of them; start
        # every probe from the home configuration so the stroke, and so
        # the outcome, is a function of the state alone.
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        # A plain State has no joint positions; stepping needs the
        # engine-backed observation.
        obs = self._get_state()
        self._current_observation = obs
        direction_obj = next(d for d in self._directions
                             if d.name == direction)
        params = np.array(
            [CFG.icerink_push_approach, CFG.icerink_push_contact_z, speed],
            dtype=np.float32)
        option = probe_push_option().ground([self._robot, tile, direction_obj],
                                            params)
        if not option.initiable(obs):
            return None
        tiles, _ = self._active_objects(state)
        try:
            for _ in range(int(CFG.icerink_probe_max_steps)):
                if option.terminal(obs):
                    break
                obs = self._step_once(option.policy(obs))
        except utils.OptionExecutionFailure:
            return None
        hold = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        for _ in range(int(CFG.icerink_probe_max_steps)):
            if all(
                    self._tile_speed(t) < CFG.icerink_settle_speed
                    for t in tiles):
                break
            obs = self._step_once(hold)
        end = self._get_state()
        x, y = end.get(tile, "x"), end.get(tile, "y")
        return float(x), float(y), not self.on_rink(x, y)

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _On_holds(state: State, objects: Sequence[Object]) -> bool:
        tile, target = objects
        return tile_on_target(state, tile, target)

    @staticmethod
    def _OnRink_holds(state: State, objects: Sequence[Object]) -> bool:
        tile, = objects
        return not tile_lost(state, tile)

    @staticmethod
    def _Lost_holds(state: State, objects: Sequence[Object]) -> bool:
        tile, = objects
        return tile_lost(state, tile)

    @staticmethod
    def _AtRest_holds(state: State, objects: Sequence[Object]) -> bool:
        tile, = objects
        return tile_at_rest(state, tile)

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    # =========================================================================
    # TASK GENERATION
    # =========================================================================
    _grid_margin: ClassVar[float] = 0.07
    _grid_cols: ClassVar[int] = 5
    _grid_rows: ClassVar[int] = 4
    # Minimum centre spacing between tiles at the start, and the
    # clearance a slide's path keeps from every other tile and target.
    _tile_spacing: ClassVar[float] = 0.10
    _path_clearance: ClassVar[float] = 0.075
    # A push must move the tile at least this far, or the target sits
    # where the tile already is.
    _min_travel: ClassVar[float] = 0.06

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng,
                                train=True)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng,
                                train=False)

    # Where a tile may START. The far strip of the rink is where tiles
    # slide TO: the arm cannot push a tile that already sits there (a
    # push ends with the gripper at the tile's centre, and the far edge
    # is at the edge of its reach), so it is left to the walls.
    _start_y_max: ClassVar[float] = 1.30

    @classmethod
    def _grid_cells(cls) -> List[Tuple[float, float]]:
        x_min, x_max, y_min, _ = cls.rink_bounds()
        m = cls._grid_margin
        xs = np.linspace(x_min + m, x_max - m, cls._grid_cols)
        ys = np.linspace(y_min + m, cls._start_y_max, cls._grid_rows)
        return [(float(x), float(y)) for x in xs for y in ys]

    @staticmethod
    def _segment_distance(px: float, py: float, ax: float, ay: float,
                          bx: float, by: float) -> float:
        """Distance from point P to segment AB."""
        abx, aby = bx - ax, by - ay
        length_sq = abx * abx + aby * aby
        if length_sq < 1e-12:
            return float(np.hypot(px - ax, py - ay))
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
        return float(np.hypot(px - (ax + t * abx), py - (ay + t * aby)))

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

    def _sample_level(
        self, num_tiles: int, rng: np.random.Generator
    ) -> Optional[Tuple[State, List[Tuple[float, float]], List[int],
                        List[Tuple[int, float]]]]:
        """One draw: tiles, materials, a target per tile that a single push
        from the start realizes, and the (direction index, speed) of that push;
        None if the draw admits none."""
        cells = self._grid_cells()
        order = rng.permutation(len(cells))
        positions: List[Tuple[float, float]] = []
        for idx in order:
            x, y = cells[idx]
            if all(
                    np.hypot(x - px, y - py) >= self._tile_spacing
                    for px, py in positions):
                positions.append((x, y))
            if len(positions) == num_tiles:
                break
        if len(positions) < num_tiles:
            return None
        materials = [
            int(c) for c in rng.choice(
                len(self.COLOR_PALETTE), size=num_tiles, replace=False)
        ]

        init_dict: Dict[Object, Dict[str, float]] = {
            self._robot: self._robot_init_dict()
        }
        for direction in self._directions:
            init_dict[direction] = {"yaw": self.DIRECTIONS[direction.name]}
        for i in range(num_tiles):
            x, y = positions[i]
            init_dict[self._tiles[i]] = {
                "x": x,
                "y": y,
                "z": self.tile_z,
                "rot": 0.0,
                "color": float(materials[i]),
                "speed": 0.0,
            }
            # Targets are placed once the slides are known; park them
            # on the tile for the probe, where they collide with nothing.
            init_dict[self._targets[i]] = {
                "x": x,
                "y": y,
                "z": self.target_z,
                "color": float(materials[i]),
            }
        state = utils.create_state_from_dict(init_dict)

        targets: List[Optional[Tuple[float, float]]] = [None] * num_tiles
        solution: List[Tuple[int, float]] = [(0, 0.0)] * num_tiles
        lo, hi = CFG.icerink_push_speed_range
        for i in rng.permutation(num_tiles):
            start = positions[i]
            found = False
            for direction in rng.permutation(PLANNED_DIRECTIONS):
                speed = float(rng.uniform(lo, hi))
                outcome = self.push_outcome(state, self._tiles[i],
                                            str(direction), speed)
                if outcome is None:
                    continue
                sx, sy, lost = outcome
                if lost or np.hypot(sx - start[0],
                                    sy - start[1]) < self._min_travel:
                    continue
                others = [positions[j] for j in range(num_tiles) if j != i]
                others += [t for j, t in enumerate(targets) if t and j != i]
                if any(
                        self._segment_distance(ox, oy, start[0], start[1], sx,
                                               sy) < self._path_clearance
                        for ox, oy in others):
                    continue
                targets[i] = (sx, sy)
                solution[i] = (PLANNED_DIRECTIONS.index(str(direction)), speed)
                found = True
                break
            if not found:
                return None
        return state, [t for t in targets if t is not None], materials, \
            solution

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        counts = list(CFG.icerink_num_tiles_train if train else CFG.
                      icerink_num_tiles_test)
        tasks = []
        for _ in range(num_tasks):
            num_tiles = int(rng.choice(counts))
            level = None
            for _ in range(int(CFG.icerink_max_sampling_attempts)):
                level = self._sample_level(num_tiles, rng)
                if level is not None:
                    break
            if level is None:
                raise RuntimeError(
                    f"No {num_tiles}-tile rink with single-push targets "
                    f"found in {CFG.icerink_max_sampling_attempts} draws.")
            state, targets, materials, solution = level
            init_dict = {
                obj: {
                    feat: float(state.get(obj, feat))
                    for feat in obj.type.feature_names
                }
                for obj in state
            }
            goal_atoms: Set[GroundAtom] = set()
            names = []
            for i, (tx, ty) in enumerate(targets):
                init_dict[self._targets[i]]["x"] = tx
                init_dict[self._targets[i]]["y"] = ty
                goal_atoms.add(
                    GroundAtom(self._On, [self._tiles[i], self._targets[i]]))
                names.append(self.color_name(materials[i]))
            init_state = utils.create_state_from_dict(init_dict)
            parts = [
                f"the {n} tile ({self._tiles[i].name}) onto the {n} target "
                f"({self._targets[i].name})" for i, n in enumerate(names)
            ]
            goal_nl = "Slide " + ", ".join(parts[:-1]) + \
                (" and " if len(parts) > 1 else "") + parts[-1] + "."
            # The generator's own solution, experimenter-only: which
            # direction and speed landed each tile on its target.
            metrics: Dict[str, float] = {}
            for i, (dir_idx, speed) in enumerate(solution):
                metrics[f"solution_tile{i}_direction"] = float(dir_idx)
                metrics[f"solution_tile{i}_speed"] = float(speed)
            tasks.append(
                EnvironmentTask(init_state,
                                goal_atoms,
                                goal_nl=goal_nl,
                                evaluator=IceRinkEvaluator(goal_atoms),
                                offline_task_metrics=metrics))
        return self._add_pybullet_state_to_tasks(tasks)


# =============================================================================
# PROBE INSTANCE FOR THE ORACLE
# =============================================================================
_PROBE_ENV: Optional[PyBulletIceRinkEnv] = None


def probe_env() -> PyBulletIceRinkEnv:
    """A dedicated instance for slide probes, so a helper predicate never
    disturbs the live episode's world."""
    global _PROBE_ENV  # pylint: disable=global-statement
    if _PROBE_ENV is None:
        _PROBE_ENV = PyBulletIceRinkEnv(use_gui=False)
    return _PROBE_ENV


if __name__ == "__main__":
    # Watch a rink: push every tile along its planned direction in turn.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_icerink"
    CFG.num_train_tasks = 1
    CFG.num_test_tasks = 0
    env = PyBulletIceRinkEnv(use_gui=True)
    _task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
    env._set_state(_task.init)  # pylint: disable=protected-access
    print("goal:", _task.goal_nl)
    _joints = env._pybullet_robot.initial_joint_positions  # pylint: disable=protected-access
    while True:
        env.step(Action(np.array(_joints)))
        time.sleep(0.05)
