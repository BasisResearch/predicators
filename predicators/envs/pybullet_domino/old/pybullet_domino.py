"""Example usage:

python predicators/main.py --approach oracle --env pybullet_domino \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug \
--sesame_check_expected_atoms False --horizon 60 \
--video_not_break_on_exception --pybullet_ik_validate False
"""
import logging
import time
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, \
    Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


@dataclass
class PlacementResult:
    """Result of placing a domino, target, or pivot in the sequence."""
    success: bool
    x: float
    y: float
    rotation: float
    domino_count: int
    pivot_count: int = 0
    target_count: int = 0
    just_turned_90: bool = False
    just_placed_target: bool = False


class PyBulletDominoEnv(PyBulletEnv):
    """A simple PyBullet environment involving M dominoes and N targets.

    Each target is considered 'toppled' if it is significantly tilted
    from its upright orientation. The overall goal is to topple all
    targets.
    """
    # Table / workspace config
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi / 2])
    table_width: ClassVar[float] = 1.0

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2  # 0.95

    # Domino shape
    domino_width: ClassVar[float] = 0.07
    domino_depth: ClassVar[float] = 0.015
    domino_height: ClassVar[float] = 0.15
    domino_x_lb: ClassVar[float] = x_lb
    domino_x_ub: ClassVar[float] = x_ub
    domino_y_lb: ClassVar[float] = y_lb + domino_width
    domino_y_ub: ClassVar[float] = y_ub - 3 * domino_width
    domino_in_upper_half_shift: ClassVar[float] = 0.4
    turn_shift_frac: ClassVar[float] = 0.6
    # domino_mass: ClassVar[float] = 0.3
    domino_mass: ClassVar[float] = 0.1
    domino_friction: ClassVar[float] = 0.5
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.85, 0.7, 0.85, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)
    glued_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (1.0, 0.0, 0.0, 1.0)
    glued_percentage: ClassVar[float] = 0.5
    domino_roll_threshold: ClassVar[float] = np.deg2rad(5)
    fallen_threshold: ClassVar[float] = np.pi * 2 / 5  # 60 degrees in radians

    target_height: ClassVar[float] = 0.2
    pivot_width: ClassVar[float] = 0.2

    # For deciding if a target is toppled: if absolute roll in x or y
    # is bigger than some threshold (e.g. 0.4 rad ~ 23 deg), treat as toppled.
    topple_angle_threshold: ClassVar[float] = 0.4

    # Camera defaults, optional
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = -70
    _camera_pitch: ClassVar[float] = -40
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Debug line settings
    debug_line_height: ClassVar[float] = 0.2

    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub
    robot_base_pos: ClassVar[Optional[Tuple[float, float,
                                            float]]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Optional[Tuple[float, float, float, float]]] =\
        p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    turn_choices: ClassVar[List[str]] = ["straight", "turn90", "pivot180"]

    # Grid configuration
    # num_pos_x and num_pos_y will be set dynamically based on train/test mode
    pos_gap: ClassVar[
        float] = domino_width * 1.4  # Distance between grid positions 0.07 * 1.4=0.098

    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _domino_type = Type(
        "domino",
        ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
    )
    _target_type = Type("target", ["x", "y", "z", "yaw"],
                        sim_features=["id", "joint_id"])
    _pivot_type = Type("pivot", ["x", "y", "z", "yaw"],
                       sim_features=["id", "joint_id"])
    _direction_type = Type("direction", ["dir"])

    def __init__(self, use_gui: bool = True) -> None:
        # Initialize domino count variables from CFG
        # Calculate maximums from train and test configurations
        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))

        assert max_dominos <= 9
        assert max_targets <= 3

        self.num_dominos_max = max_dominos
        self.num_targets_max = max_targets
        self.num_pivots_max = max_pivots

        # Create 'dummy' Objects (they'll be assigned IDs on reset)
        self._robot = Object("robot", self._robot_type)
        # We'll hold references to all domino and target objects in lists
        # after we create them in tasks.
        self.dominos: List[Object] = []
        if CFG.domino_use_domino_blocks_as_target:
            # When true, the number of target objects is 0.
            num_dominos = self.num_dominos_max + self.num_targets_max
            num_targets = 0
        else:
            num_dominos = self.num_dominos_max
            num_targets = self.num_targets_max
        for i in range(num_dominos):
            name = f"domino_{i}"
            obj_type = self._domino_type
            obj = Object(name, obj_type)
            self.dominos.append(obj)
        self.targets: List[Object] = []
        for i in range(num_targets):
            name = f"target_{i}"
            obj_type = self._target_type
            obj = Object(name, obj_type)
            self.targets.append(obj)
        self.pivots: List[Object] = []
        for i in range(self.num_pivots_max):
            name = f"pivot_{i}"
            obj_type = self._pivot_type
            obj = Object(name, obj_type)
            self.pivots.append(obj)

        # Create direction objects
        self.directions: List[Object] = []
        direction_names = ["straight", "left", "right"]
        for i, name in enumerate(direction_names):
            obj = Object(name, self._direction_type)
            self.directions.append(obj)

        self.block_constraints: List[int] = []
        self.fixed_domino_ids: List[int] = []
        self._debug_line_ids: List[int] = []

        super().__init__(use_gui)

        # Define Predicates
        if CFG.domino_use_domino_blocks_as_target:
            self._Toppled = Predicate("Toppled", [self._domino_type],
                                      self._Toppled_holds)
        else:
            self._Toppled = Predicate("Toppled", [self._target_type],
                                      self._Toppled_holds)
        self._Upright = Predicate("Upright", [self._domino_type],
                                  self._Upright_holds)
        self._Tilting = Predicate("Tilting", [self._domino_type],
                                  self._Tilting_holds)
        self._InitialBlock = Predicate("InitialBlock", [self._domino_type],
                                       self._StartBlock_holds)
        self._MovableBlock = Predicate("MovableBlock", [self._domino_type],
                                       self._MovableBlock_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._domino_type],
                                  self._Holding_holds)

        # Add DominoNotGlued predicate
        self._DominoNotGlued = Predicate("DominoNotGlued", [self._domino_type],
                                         self._DominoNotGlued_holds)

        # Grid position predicates (initialized as empty, populated on demand)
        self._grid_position_predicates: Set[Predicate] = set()
        # Rotation predicates (initialized as empty, populated on demand)
        self._rotation_predicates: Set[Predicate] = set()

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_old"

    @property
    def predicates(self) -> Set[Predicate]:
        base_predicates = {
            self._Toppled,
            self._Upright,
            self._Tilting,
            self._InitialBlock,
            self._MovableBlock,
            self._HandEmpty,
            self._Holding,
        }
        if CFG.domino_has_glued_dominos:
            base_predicates.add(self._DominoNotGlued)
        base_predicates.update(self._grid_position_predicates)
        base_predicates.update(self._rotation_predicates)

        return base_predicates

    def generate_grid_position_predicates(self, grid_gap: float,
                                          num_x_cells: int,
                                          num_y_cells: int) -> Set[Predicate]:
        """Generate predicates for grid positions centered on the initial
        domino.

        This creates predicates that represent a discretized grid for domino
        positions. The grid is centered on the initial (start) domino and
        oriented according to its rotation.

        Args:
            grid_gap: Distance that corresponds to one grid step
            num_x_cells: Number of cells in the x direction (forward/backward
                        from initial domino). The range will be
                        [-num_x_cells, num_x_cells].
            num_y_cells: Number of cells in the y direction (left/right from
                        initial domino). The range will be
                        [-num_y_cells, num_y_cells].

        Returns:
            Set of Predicate objects for each grid position.
        """
        predicates = set()

        # Generate predicate for initial position (0, 0)
        def AtX0Y0_holds(state: State, objects: Sequence[Object]) -> bool:
            """Check if domino is at the initial position (the start
            domino)."""
            curr_domino, = objects
            return self._StartBlock_holds(state, [curr_domino])

        predicates.add(Predicate("AtX0Y0", [self._domino_type], AtX0Y0_holds))

        # Generate predicates for other grid positions
        for offset_x in range(-num_x_cells, num_x_cells + 1):
            for offset_y in range(-num_y_cells, num_y_cells + 1):
                if offset_x == 0 and offset_y == 0:
                    continue  # Already handled above

                # Create a closure to capture offset_x, offset_y, and grid_gap
                def make_at_position_holds(
                        ox: int, oy: int, gap: float
                ) -> Callable[[State, Sequence[Object]], bool]:
                    """Create a predicate checking function for a specific grid
                    position."""

                    def at_position_holds(state: State,
                                          objects: Sequence[Object]) -> bool:
                        """Check if domino is at grid position (ox, oy)."""
                        curr_domino, = objects

                        # Find the initial domino
                        initial_domino = None
                        for domino in state.get_objects(self._domino_type):
                            if self._StartBlock_holds(state, [domino]):
                                initial_domino = domino
                                break

                        if initial_domino is None:
                            return False

                        # Get initial domino position and rotation
                        id_x = state.get(initial_domino, "x")
                        id_y = state.get(initial_domino, "y")
                        id_rot = state.get(initial_domino, "yaw")

                        # Get current domino position
                        cd_x = state.get(curr_domino, "x")
                        cd_y = state.get(curr_domino, "y")

                        # Calculate expected position based on initial domino's orientation
                        # Transform from local grid coordinates to world coordinates
                        # ox is forward/backward in the direction the domino faces
                        # oy is left/right perpendicular to the domino's facing direction
                        exp_x = id_x + np.cos(id_rot) * ox * gap - np.sin(
                            id_rot) * oy * gap
                        exp_y = id_y + np.sin(id_rot) * ox * gap + np.cos(
                            id_rot) * oy * gap

                        # Check if current domino is close to expected position
                        return np.isclose(cd_x, exp_x, atol=gap / 2) and \
                               np.isclose(cd_y, exp_y, atol=gap / 2)

                    return at_position_holds

                # Create predicate name
                # Use "Xm" for negative values, "X" for positive
                if offset_x < 0:
                    x_str = f"Xm{abs(offset_x)}"
                else:
                    x_str = f"X{offset_x}"

                if offset_y < 0:
                    y_str = f"Ym{abs(offset_y)}"
                else:
                    y_str = f"Y{offset_y}"

                pred_name = f"At{x_str}{y_str}"

                # Create the predicate
                holds_func = make_at_position_holds(offset_x, offset_y,
                                                    grid_gap)
                predicates.add(
                    Predicate(pred_name, [self._domino_type], holds_func))

        # Store for later access
        self._grid_position_predicates = predicates
        return predicates

    @property
    def grid_position_predicates(self) -> Set[Predicate]:
        """Return the grid position predicates.

        Note: These predicates must be generated first using
        generate_grid_position_predicates() before they can be accessed.
        """
        return self._grid_position_predicates

    def generate_rotation_predicates(self, n: int) -> Set[Predicate]:
        """Generate predicates for discretized yaw rotations.

        This creates predicates that represent discretized yaw rotations
        for dominoes. The rotation range [-π, π] is divided into n equal
        intervals.

        Args:
            n: Number of rotation intervals to create. The rotation range
               [-π, π] will be divided into n equal intervals.

        Returns:
            Set of Predicate objects for each rotation interval.
        """
        predicates = set()

        # Divide the rotation range [-π, π] into n intervals
        interval_size = 2 * np.pi / n

        for i in range(n):
            # Calculate interval bounds first (needed for predicate name)
            lower_bound = -np.pi + i * interval_size
            upper_bound = lower_bound + interval_size

            # Create a closure to capture the rotation interval bounds
            def make_at_rotation_holds(
                interval_idx: int, interval_sz: float
            ) -> Callable[[State, Sequence[Object]], bool]:
                """Create a predicate checking function for a specific rotation
                interval."""

                def at_rotation_holds(state: State,
                                      objects: Sequence[Object]) -> bool:
                    """Check if domino is at rotation interval i."""
                    domino, = objects
                    yaw = state.get(domino, "yaw")

                    # Normalize yaw to [-π, π]
                    yaw = utils.wrap_angle(yaw)

                    # Calculate interval bounds
                    lower_bound = -np.pi + interval_idx * interval_sz
                    upper_bound = lower_bound + interval_sz

                    # Check if yaw is in this interval
                    return lower_bound < yaw <= upper_bound

                return at_rotation_holds

            # Create predicate name with interval range
            pred_name = f"AtRot({lower_bound:.1f},{upper_bound:.1f}]"

            # Create the predicate
            holds_func = make_at_rotation_holds(i, interval_size)
            predicates.add(
                Predicate(pred_name, [self._domino_type], holds_func))

        # Store for later access
        self._rotation_predicates = predicates
        return predicates

    @property
    def rotation_predicates(self) -> Set[Predicate]:
        """Return the rotation predicates.

        Note: These predicates must be generated first using
        generate_rotation_predicates() before they can be accessed.
        """
        return self._rotation_predicates

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # The goal is always to topple all targets
        return {self._Toppled}

    @property
    def target_predicates(self) -> Set[Predicate]:
        target_predicates: Set[Predicate] = set()
        if CFG.domino_has_glued_dominos:
            target_predicates.add(self._DominoNotGlued)
        return target_predicates

    @property
    def types(self) -> Set[Type]:
        base_types = {
            self._robot_type, self._domino_type, self._target_type,
            self._pivot_type, self._direction_type
        }

        return base_types

    def is_task_solvable(self, task: EnvironmentTask) -> bool:
        """Check if the task is solvable."""
        if CFG.domino_has_glued_dominos:
            dominos = task.init.get_objects(self._domino_type)
            for domino in dominos:
                if self._DominoGlued_holds(task.init, [domino]):
                    return False
        return True

    # -------------------------------------------------------------------------
    # Environment Setup

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        # Reuse parent method to create a robot and get a physics client
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # (Optional) Add a simple table
        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id
        # add another table for more space to play dominoes
        create_object(asset_path="urdf/table.urdf",
                      position=(cls.table_pos[0],
                                cls.table_pos[1] + cls.table_width / 2,
                                cls.table_pos[2]),
                      orientation=cls.table_orn,
                      scale=1.0,
                      use_fixed_base=True,
                      physics_client_id=physics_client_id)
        # add a debug line at the end of the first table
        p.addUserDebugLine([
            cls.table_pos[0] + (cls.x_ub - cls.x_lb) / 2, cls.table_pos[1] +
            (cls.y_ub - cls.y_lb) / 2, cls.table_height + 0.001
        ], [
            cls.table_pos[0] - (cls.x_ub - cls.x_lb) / 2, cls.table_pos[1] +
            (cls.y_ub - cls.y_lb) / 2, cls.table_height + 0.001
        ], [1, 0, 0],
                           parentObjectUniqueId=-1,
                           parentLinkIndex=-1)

        # Create a fixed number of dominoes and targets here
        domino_ids = []
        target_ids = []

        # Calculate maximums from train and test configurations
        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))

        if CFG.domino_use_domino_blocks_as_target:
            # If using domino blocks as targets, we create more dominoes
            num_dominos_to_create = max_dominos + max_targets
            num_targets_to_create = 0
        else:
            num_dominos_to_create = max_dominos
            num_targets_to_create = max_targets
        for i in range(num_dominos_to_create):  # e.g. 3 dominoes
            domino_id = create_domino_block(
                color=cls.start_domino_color if i == 0 else cls.domino_color,
                half_extents=(cls.domino_width / 2, cls.domino_depth / 2,
                              cls.domino_height / 2),
                mass=cls.domino_mass,
                friction=cls.domino_friction,
                orientation=(0.0, 0.0, 0.0, 1.0),
                physics_client_id=physics_client_id,
                add_top_triangle=True,
            )
            domino_ids.append(domino_id)
        for _ in range(num_targets_to_create):  # e.g. 2 targets
            tid = create_object("urdf/domino_target.urdf",
                                position=(cls.x_lb, cls.y_lb, cls.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            target_ids.append(tid)
        pivot_ids = []
        for _ in range(max_pivots):
            pid = create_object("urdf/domino_pivot.urdf",
                                position=(cls.x_lb, cls.y_lb, cls.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            pivot_ids.append(pid)
        bodies["pivot_ids"] = pivot_ids
        bodies["domino_ids"] = domino_ids
        bodies["target_ids"] = target_ids

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int, joint_name: str) -> int:
        num_joints = p.getNumJoints(obj_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        # We don't have a single known ID for dominoes or targets, so we'll store
        # them all at runtime. For now, we just keep a reference to the dict.
        for domini, id in zip(self.dominos, pybullet_bodies["domino_ids"]):
            domini.id = id
        for target, id in zip(self.targets, pybullet_bodies["target_ids"]):
            target.id = id
            target.joint_id = self._get_joint_id(id, "flap_hinge_joint")
        for pivot, pid in zip(self.pivots, pybullet_bodies["pivot_ids"]):
            pivot.id = pid
            pivot.joint_id = self._get_joint_id(pid, "flap_hinge_joint")

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        domino_ids = [domino.id for domino in self.dominos]
        pivot_ids = [pivot.id for pivot in self.pivots]
        return domino_ids + pivot_ids

    def _create_task_specific_objects(self, state: State) -> None:
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._direction_type:
            if feature == "dir":
                if obj.name == "straight":
                    return 0.0
                elif obj.name == "left":
                    return 1.0
                elif obj.name == "right":
                    return 2.0

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _create_fixed_constraint(self, bodyA: int, bodyB: int) -> int:
        """Create a fixed joint in PyBullet with a pivot at the midpoint of the
        two bodies (so they remain exactly where they are)."""
        # Get the current global positions/orientations of each domino.
        pA, oA = p.getBasePositionAndOrientation(
            bodyA, physicsClientId=self._physics_client_id)
        pB, oB = p.getBasePositionAndOrientation(
            bodyB, physicsClientId=self._physics_client_id)

        # Compute a midpoint in world space (so the constraint pivot is between them).
        midpoint = [(pA[i] + pB[i]) / 2.0 for i in range(3)]

        # Express this midpoint in the local frames of each body:
        inv_pA, inv_oA = p.invertTransform(pA, oA)
        parentPivot, parentOrn = p.multiplyTransforms(inv_pA, inv_oA, midpoint,
                                                      [0, 0, 0, 1])

        inv_pB, inv_oB = p.invertTransform(pB, oB)
        childPivot, childOrn = p.multiplyTransforms(inv_pB, inv_oB, midpoint,
                                                    [0, 0, 0, 1])

        # Create the constraint at those local pivots, ensuring no sudden jump.
        cid = p.createConstraint(
            parentBodyUniqueId=bodyA,
            parentLinkIndex=-1,
            childBodyUniqueId=bodyB,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=parentPivot,
            parentFrameOrientation=parentOrn,
            childFramePosition=childPivot,
            childFrameOrientation=childOrn,
            physicsClientId=self._physics_client_id,
        )
        return cid

    def _no_target_in_between(self, state: State, domino1: Object,
                              domino2: Object) -> bool:
        for target in state.get_objects(self._target_type):
            x1 = state.get(domino1, "x")
            y1 = state.get(domino1, "y")
            x2 = state.get(domino2, "x")
            y2 = state.get(domino2, "y")
            x = state.get(target, "x")
            y = state.get(target, "y")
            if x1 < x < x2 and y == y1:
                return False
            if y1 < y < y2 and x == x1:
                return False
        return True

    def _reset_custom_env_state(self, state: State) -> None:
        """Reset the custom environment state to match the given state."""
        domino_objs = state.get_objects(self._domino_type)

        for constraint in self.block_constraints:
            p.removeConstraint(constraint)
        self.block_constraints = []

        # Restore normal dynamics to previously fixed dominoes
        for domino_id in self.fixed_domino_ids:
            p.changeDynamics(domino_id,
                             -1,
                             mass=self.domino_mass,
                             physicsClientId=self._physics_client_id)
        self.fixed_domino_ids = []

        if CFG.domino_some_dominoes_are_connected:
            for i in range(len(domino_objs) - 1):
                domino1 = domino_objs[i]
                domino2 = domino_objs[i + 1]
                rot1 = state.get(domino1, "yaw")
                rot2 = state.get(domino2, "yaw")

                if abs(rot1 - rot2) < 1e-5 and self._no_target_in_between(
                        state, domino1, domino2):
                    cid = self._create_fixed_constraint(domino1.id, domino2.id)
                    self.block_constraints.append(cid)
                    break

        # Update domino colors to match the state
        for domino in domino_objs:
            if domino.id is not None:
                r = state.get(domino, "r")
                g = state.get(domino, "g")
                b = state.get(domino, "b")
                update_object(domino.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(domino_objs), len(self.dominos)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.dominos[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        target_objs = state.get_objects(self._target_type)
        for target_obj in target_objs:
            self._set_flat_rotation(target_obj, 0.0)
        for i in range(len(target_objs), len(self.targets)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.targets[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        pivot_objs = state.get_objects(self._pivot_type)
        for pivot_obj in pivot_objs:
            self._set_flat_rotation(pivot_obj, 0.0)
        for i in range(len(pivot_objs), len(self.pivots)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.pivots[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Handle fixed domino constraints when CFG.domino_has_glued_dominoes is True
        if CFG.domino_has_glued_dominos:
            eps = 1e-5
            for domino in domino_objs:
                if domino.id is not None:
                    if self._DominoGlued_holds(state, [domino]):
                        # Make this domino immovable by setting very high mass
                        p.changeDynamics(
                            domino.id,
                            -1,
                            mass=1e10,
                            physicsClientId=self._physics_client_id)
                        self.fixed_domino_ids.append(domino.id)

    def _get_flat_rotation(self, flap_obj: Object) -> float:
        j_pos, _, _, _ = p.getJointState(flap_obj.id, flap_obj.joint_id)
        return j_pos

    def _set_flat_rotation(self, flap_obj: Object, rot: float = 0.0) -> None:
        p.resetJointState(flap_obj.id, flap_obj.joint_id, rot)
        return

    def step(self, action: Action, render_obs: bool = False) -> State:
        """In this domain, stepping might be trivial (we won't do anything
        special aside from the usual robot step)."""
        next_state = super().step(action, render_obs=render_obs)

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Predicates

    @classmethod
    def _Toppled_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Target is toppled if it's significantly tilted from upright in pitch
        or roll.

        For domino targets, we use the roll feature. For regular
        targets, we use rotation threshold.
        """
        obj, = objects

        if CFG.domino_use_domino_blocks_as_target:
            roll_angle = abs(state.get(obj, "roll"))
            return roll_angle >= cls.fallen_threshold
        else:
            # For regular targets, use rotation-based check (currently disabled)
            rot_z = state.get(obj, "yaw")
            if abs(utils.wrap_angle(rot_z)) < 0.8:
                return True
            return False

    @classmethod
    def _Upright_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        tilt_angle = state.get(obj, "roll")
        return abs(tilt_angle) < cls.domino_roll_threshold

    @classmethod
    def _Tilting_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Domino is tilting (in transition, leaning) - roll angle between
        domino_roll_threshold and tilting_threshold (60°)."""
        obj, = objects
        roll_angle = abs(state.get(obj, "roll"))
        return cls.domino_roll_threshold <= roll_angle < cls.fallen_threshold

    @classmethod
    def _StartBlock_holds(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        domino, = objects
        # Check if the domino has the light green color (start domino)
        eps = 1e-3
        if abs(state.get(domino, "r") - cls.start_domino_color[0]) > eps:
            return False
        if abs(state.get(domino, "g") - cls.start_domino_color[1]) > eps:
            return False
        if abs(state.get(domino, "b") - cls.start_domino_color[2]) > eps:
            return False
        return True

    @classmethod
    def _MovableBlock_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        domino, = objects
        # Check if the domino has the regular blue domino color (movable block)
        eps = 1e-3
        if abs(state.get(domino, "r") - cls.domino_color[0]) > eps:
            return False
        if abs(state.get(domino, "g") - cls.domino_color[1]) > eps:
            return False
        if abs(state.get(domino, "b") - cls.domino_color[2]) > eps:
            return False
        return True

    @classmethod
    def _TargetDomino_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        domino, = objects
        # Check if the domino has the pink color (target domino)
        eps = 1e-3
        return (cls._DominoGlued_holds(state, objects)) or (
            abs(state.get(domino, "r") - cls.target_domino_color[0]) < eps
            and abs(state.get(domino, "g") - cls.target_domino_color[1]) < eps
            and abs(state.get(domino, "b") - cls.target_domino_color[2]) < eps)

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, = objects
        dominos = state.get_objects(self._domino_type)
        for domino in dominos:
            if state.get(domino, "is_held"):
                return False
        return True

    @classmethod
    def _Holding_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        _, domino = objects
        return state.get(domino, "is_held") > 0.5

    @classmethod
    def _DominoNotGlued_holds(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        """Check if a domino is NOT glued (i.e., does not have the red glued
        color)."""
        return not cls._DominoGlued_holds(state, objects)

    @classmethod
    def _DominoGlued_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if a domino is glued (i.e., has the red glued color)."""
        eps = 1e-3
        r_val = state.get(objects[0], "r")
        g_val = state.get(objects[0], "g")
        b_val = state.get(objects[0], "b")

        return (abs(r_val - cls.glued_domino_color[0]) < eps
                and abs(g_val - cls.glued_domino_color[1]) < eps
                and abs(b_val - cls.glued_domino_color[2]) < eps)

    # -------------------------------------------------------------------------
    # Task Generation

    def _get_expected_domino_count(self, n_dominos: int,
                                   n_targets: int) -> int:
        """Calculate the expected total number of dominoes."""
        if CFG.domino_use_domino_blocks_as_target:
            return n_dominos + n_targets
        return n_dominos

    def _should_continue_placement(self, domino_count: int, target_count: int,
                                   n_dominos: int, n_targets: int) -> bool:
        """Check if we should continue placing dominoes and targets."""
        expected_domino_count = self._get_expected_domino_count(
            n_dominos, n_targets)
        if CFG.domino_use_domino_blocks_as_target:
            return (domino_count < expected_domino_count
                    or target_count < n_targets)
        return domino_count < n_dominos or target_count < n_targets

    def _check_placement_complete(self, domino_count: int, target_count: int,
                                  pivot_count: int, n_dominos: int,
                                  n_targets: int, n_pivots: int) -> bool:
        """Check if all dominoes, targets, and pivots have been placed."""
        expected_domino_count = self._get_expected_domino_count(
            n_dominos, n_targets)
        if CFG.domino_use_domino_blocks_as_target:
            return (domino_count == expected_domino_count
                    and target_count == n_targets and pivot_count == n_pivots)
        return (domino_count == n_dominos and target_count == n_targets
                and pivot_count == n_pivots)

    def _generate_domino_sequence(
            self,
            rng: np.random.Generator,
            n_dominos: int,
            n_targets: int,
            n_pivots: int,
            log_debug: bool = False,
            task_idx: Optional[int] = None,
            domino_in_upper_half: bool = False) -> Optional[Dict]:
        """Generate a sequence of dominoes, targets, and pivots.

        Returns:
            Dict mapping objects to their placement parameters, or None if failed
        """

        # Initialize placement state
        obj_dict = {}
        domino_count = 0
        target_count = 0
        pivot_count = 0
        just_placed_target = False
        just_turned_90 = False

        y_lb, y_ub = self.y_lb, self.y_ub
        x_lb, x_ub = self.x_lb, self.x_ub
        if domino_in_upper_half:
            y_lb += self.domino_in_upper_half_shift
            y_ub += self.domino_in_upper_half_shift

        def _in_bounds(nx: float, ny: float) -> bool:
            """Check if (nx, ny) is within table boundaries."""
            return x_lb < nx < x_ub and y_lb < ny < y_ub

        # Draw boundary rectangle for domino placement area
        debug_height = self.table_height + 0.002
        # Bottom edge (y_lb)
        p.addUserDebugLine([x_lb, y_lb, debug_height],
                           [x_ub, y_lb, debug_height], [0, 1, 0],
                           2,
                           parentObjectUniqueId=-1,
                           parentLinkIndex=-1)
        # Top edge (y_ub)
        p.addUserDebugLine([x_lb, y_ub, debug_height],
                           [x_ub, y_ub, debug_height], [0, 1, 0],
                           2,
                           parentObjectUniqueId=-1,
                           parentLinkIndex=-1)
        # Left edge (x_lb)
        p.addUserDebugLine([x_lb, y_lb, debug_height],
                           [x_lb, y_ub, debug_height], [0, 1, 0],
                           2,
                           parentObjectUniqueId=-1,
                           parentLinkIndex=-1)
        # Right edge (x_ub)
        p.addUserDebugLine([x_ub, y_lb, debug_height],
                           [x_ub, y_ub, debug_height], [0, 1, 0],
                           2,
                           parentObjectUniqueId=-1,
                           parentLinkIndex=-1)

        # Initial domino position and orientation
        x = rng.uniform(x_lb, x_ub)
        y = rng.uniform(y_lb, y_ub)
        rotation = rng.choice([0, np.pi / 2, -np.pi / 2])
        gap = self.pos_gap

        # Place first domino (start block)
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count,
            x,
            y,
            rotation,
            is_start_block=True,
            rng=rng,
            task_idx=task_idx)
        domino_count += 1

        # Main placement loop
        while self._should_continue_placement(domino_count, target_count,
                                              n_dominos, n_targets):
            # Determine what we can place
            can_place_target = (domino_count >= 2 and target_count < n_targets
                                and not just_placed_target)
            expected_domino_count = self._get_expected_domino_count(
                n_dominos, n_targets)
            can_place_domino = domino_count < expected_domino_count

            # Decide whether to place domino or target
            should_place_domino = (not can_place_target
                                   or rng.random() > 0.5) and can_place_domino

            if should_place_domino:
                # Place domino (or pivot)
                result = self._place_next_domino(
                    rng, obj_dict, x, y, rotation, gap, domino_count,
                    pivot_count, target_count, n_pivots, n_dominos, n_targets,
                    just_placed_target, just_turned_90, _in_bounds, task_idx)
                if not result.success:
                    return None

                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                pivot_count = result.pivot_count
                target_count += result.target_count
                just_turned_90 = result.just_turned_90
                just_placed_target = result.just_placed_target

            else:
                # Place target
                if log_debug:
                    print("Placing target")
                result = self._place_next_target(rng, obj_dict, x, y, rotation,
                                                 gap, domino_count,
                                                 target_count, _in_bounds,
                                                 task_idx)
                if not result.success:
                    return None

                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                target_count = result.target_count
                just_placed_target = True
                just_turned_90 = False

        # Check if we successfully placed everything
        if self._check_placement_complete(domino_count, target_count,
                                          pivot_count, n_dominos, n_targets,
                                          n_pivots):
            return obj_dict
        return None

    def _place_next_domino(self,
                           rng: np.random.Generator,
                           obj_dict: Dict,
                           x: float,
                           y: float,
                           rotation: float,
                           gap: float,
                           domino_count: int,
                           pivot_count: int,
                           target_count: int,
                           n_pivots: int,
                           n_dominos: int,
                           n_targets: int,
                           just_placed_target: bool,
                           just_turned_90: bool,
                           _in_bounds: Callable[[float, float], bool],
                           task_idx: Optional[int] = None) -> PlacementResult:
        """Place the next domino in the sequence by selecting and executing a
        placement strategy.

        Determines available placement strategies based on constraints
        (e.g., avoiding consecutive 90-degree turns, forcing straight
        placement after targets), then randomly chooses and executes one
        of the valid strategies: straight, turn90, or pivot180.
        """
        # Determine available placement strategies
        turn_choices = self.turn_choices.copy()
        if pivot_count >= n_pivots and "pivot180" in turn_choices:
            turn_choices.remove("pivot180")
        if just_turned_90 and "turn90" in turn_choices:
            turn_choices.remove("turn90")
        if just_placed_target:
            turn_choices = ["straight"]

        choice = rng.choice(turn_choices)
        print(f"Choice: {choice}")

        # Determine if the second block in a turn should be a target
        should_place_target_at_end = False
        if CFG.domino_use_domino_blocks_as_target and choice in [
                "turn90", "pivot180"
        ]:
            # Check if we have targets left to place and randomly decide
            if target_count < n_targets and rng.random() > 0.5:
                should_place_target_at_end = True

        # Execute the chosen placement strategy
        if choice == "straight":
            return self._place_straight_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, _in_bounds,
                                               task_idx)
        elif choice == "turn90":
            return self._place_turn90_domino(rng, obj_dict, x, y, rotation,
                                             gap, domino_count, n_dominos,
                                             n_targets, _in_bounds, task_idx,
                                             should_place_target_at_end)
        elif choice == "pivot180":
            return self._place_pivot180_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, pivot_count,
                                               _in_bounds, task_idx,
                                               should_place_target_at_end)
        else:
            # Fallback to straight
            return self._place_straight_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, _in_bounds,
                                               task_idx)

    def _place_straight_domino(
            self,
            rng: np.random.Generator,
            obj_dict: Dict,
            x: float,
            y: float,
            rotation: float,
            gap: float,
            domino_count: int,
            _in_bounds: Callable[[float, float], bool],
            task_idx: Optional[int] = None) -> PlacementResult:
        """Place a domino straight ahead in the current direction.

        Calculates the next position by moving forward along the current
        rotation angle by the specified gap distance. Validates the new
        position is within bounds before placing the domino.
        """
        # Calculate next position
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        new_x, new_y = x + dx, y + dy

        if not _in_bounds(new_x, new_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count)

        # Place the domino
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count,
            new_x,
            new_y,
            rotation,
            is_start_block=False,
            rng=rng,
            task_idx=task_idx)

        return PlacementResult(success=True,
                               x=new_x,
                               y=new_y,
                               rotation=rotation,
                               domino_count=domino_count + 1)

    def _place_turn90_domino(
            self,
            rng: np.random.Generator,
            obj_dict: Dict,
            x: float,
            y: float,
            rotation: float,
            gap: float,
            domino_count: int,
            n_dominos: int,
            n_targets: int,
            _in_bounds: Callable[[float, float], bool],
            task_idx: Optional[int] = None,
            should_place_target_at_end: bool = False) -> PlacementResult:
        """Place two dominoes to create a 90-degree turn in the sequence.

        Executes the turn by placing two dominoes with 45-degree rotations each, resulting
        in a smooth 90-degree curve. The turn direction (left or right) is randomly selected.
        Returns early with straight placement if insufficient dominoes remain.

        Args:
            should_place_target_at_end: If True, the second block will be placed as a target domino.
        """
        # Check if we have enough dominos left for a full turn (needs 2 dominos)
        expected_domino_count = self._get_expected_domino_count(
            n_dominos, n_targets)
        if domino_count + 1 >= expected_domino_count:
            # Not enough dominos for turn, fallback to straight
            return self._place_straight_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, _in_bounds,
                                               task_idx)

        # Turn 45° twice (total 90° turn)
        turn_direction = rng.choice([-1, 1])

        # First domino: one step forward with 45° rotation
        # Calculate base position (one step forward)
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        d1_base_x, d1_base_y = x + dx, y + dy

        # Calculate the first domino's rotation (45° toward turn direction)
        d1_rot = rotation - turn_direction * np.pi / 4

        # Calculate the shift vector to pull the turning domino inward
        shift_magnitude = self.domino_width * self.turn_shift_frac
        shift_dx = shift_magnitude * (turn_direction * np.cos(rotation) -
                                      np.sin(rotation))
        shift_dy = shift_magnitude * (-turn_direction * np.sin(rotation) -
                                      np.cos(rotation))

        # The physical position is the base position plus the shift
        d1_x = d1_base_x + shift_dx
        d1_y = d1_base_y + shift_dy

        if not _in_bounds(d1_x, d1_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count)

        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count,
            d1_x,
            d1_y,
            d1_rot,
            is_start_block=False,
            rng=rng,
            task_idx=task_idx)
        domino_count += 1

        # Second domino: completes the turn with another 45° rotation
        d2_rot = d1_rot - turn_direction * np.pi / 4

        # Calculate d2's physical position relative to d1's
        sin_d1 = np.sin(d1_rot)
        cos_d1 = np.cos(d1_rot)
        disp_x = (gap * turn_direction * cos_d1 +
                  (2 * shift_magnitude - gap) * sin_d1) / np.sqrt(2)
        disp_y = (-gap * turn_direction * sin_d1 +
                  (2 * shift_magnitude - gap) * cos_d1) / np.sqrt(2)
        d2_x = d1_x + disp_x
        d2_y = d1_y + disp_y

        if not _in_bounds(d2_x, d2_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count)

        # Place second block - can be a target if requested
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count,
            d2_x,
            d2_y,
            d2_rot,
            is_start_block=False,
            is_target_block=should_place_target_at_end,
            rng=rng,
            task_idx=task_idx)

        target_count_increment = 1 if should_place_target_at_end else 0
        return PlacementResult(success=True,
                               x=d2_x,
                               y=d2_y,
                               rotation=d2_rot,
                               domino_count=domino_count + 1,
                               target_count=target_count_increment,
                               just_turned_90=True,
                               just_placed_target=should_place_target_at_end)

    def _place_pivot180_domino(
            self,
            rng: np.random.Generator,
            obj_dict: Dict,
            x: float,
            y: float,
            rotation: float,
            gap: float,
            domino_count: int,
            pivot_count: int,
            _in_bounds: Callable[[float, float], bool],
            task_idx: Optional[int] = None,
            should_place_target_at_end: bool = False) -> PlacementResult:
        """Place a pivot followed by a domino to create a 180-degree direction
        reversal.

        Places a pivot object that acts as a turning point, then positions a domino on the
        opposite side with 180-degree rotation. The side offset direction is randomly chosen.
        Both the pivot and domino positions are validated before placement.

        Args:
            should_place_target_at_end: If True, the domino after the pivot will be placed as a target domino.
        """
        pivot_direction = rng.choice([-1, 1])
        side_offset = self.pivot_width / 2

        # Calculate pivot position
        pivot_x = x + gap * (2 / 3) * np.sin(rotation)
        pivot_y = y + gap * (2 / 3) * np.cos(rotation)
        pivot_x -= pivot_direction * side_offset * np.cos(rotation)
        pivot_y -= pivot_direction * side_offset * np.sin(rotation)

        if not _in_bounds(pivot_x, pivot_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   pivot_count=pivot_count)

        # Place the pivot
        obj_dict[self.pivots[pivot_count]] = self._place_pivot_or_target(
            pivot_x, pivot_y, rotation)

        # Calculate domino position after 180° flip
        domino_x = pivot_x - (gap * (2 / 3)) * np.sin(rotation)
        domino_y = pivot_y - (gap * (2 / 3)) * np.cos(rotation)
        domino_x -= pivot_direction * side_offset * np.cos(rotation)
        domino_y += pivot_direction * side_offset * -np.sin(rotation)

        if not _in_bounds(domino_x, domino_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   pivot_count=pivot_count)

        # Place the domino with 180° rotation - can be a target if requested
        new_rotation = rotation + np.pi
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count,
            domino_x,
            domino_y,
            new_rotation,
            is_start_block=False,
            is_target_block=should_place_target_at_end,
            rng=rng,
            task_idx=task_idx)

        target_count_increment = 1 if should_place_target_at_end else 0
        return PlacementResult(success=True,
                               x=domino_x,
                               y=domino_y,
                               rotation=new_rotation,
                               domino_count=domino_count + 1,
                               pivot_count=pivot_count + 1,
                               target_count=target_count_increment,
                               just_placed_target=should_place_target_at_end)

    def _place_next_target(self,
                           rng: np.random.Generator,
                           obj_dict: Dict,
                           x: float,
                           y: float,
                           rotation: float,
                           gap: float,
                           domino_count: int,
                           target_count: int,
                           _in_bounds: Callable[[float, float], bool],
                           task_idx: Optional[int] = None) -> PlacementResult:
        """Place the next target object in the domino sequence.

        Calculates the target position along the current direction and
        places either a pink domino (if using domino blocks as targets)
        or a regular target object. The placement behavior depends on
        the CFG.domino_use_domino_blocks_as_target setting.
        """
        # Calculate target position
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        target_x, target_y = x + dx, y + dy

        if not _in_bounds(target_x, target_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   target_count=target_count)

        if CFG.domino_use_domino_blocks_as_target:
            # Place a pink domino as target
            obj_dict[self.dominos[domino_count]] = self._place_domino(
                domino_count,
                target_x,
                target_y,
                rotation,
                is_target_block=True,
                rng=rng,
                task_idx=task_idx)
            return PlacementResult(success=True,
                                   x=target_x,
                                   y=target_y,
                                   rotation=rotation,
                                   domino_count=domino_count + 1,
                                   target_count=target_count + 1)
        else:
            # Place a regular target
            obj_dict[self.targets[target_count]] = self._place_pivot_or_target(
                target_x, target_y, rotation)
            return PlacementResult(success=True,
                                   x=target_x,
                                   y=target_y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   target_count=target_count + 1)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            possible_num_dominos=CFG.domino_train_num_dominos,
            possible_num_targets=CFG.domino_train_num_targets,
            possible_num_pivots=CFG.domino_train_num_pivots,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            possible_num_dominos=CFG.domino_test_num_dominos,
            possible_num_targets=CFG.domino_test_num_targets,
            possible_num_pivots=CFG.domino_test_num_pivots,
            rng=self._test_rng)

    def _make_tasks(self,
                    num_tasks: int,
                    possible_num_dominos: List[int],
                    possible_num_targets: List[int],
                    possible_num_pivots: List[int],
                    rng: np.random.Generator,
                    log_debug: bool = True) -> List[EnvironmentTask]:
        tasks = []
        total_attempts = 0
        # Suppose we want to create M = 3 dominoes, N = 2 targets for each task

        for i_task in range(num_tasks):
            # 1) Robot initial
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # 2) Dominoes
            init_dict = {self._robot: robot_dict}

            # Add direction objects to initial state
            for i, direction_obj in enumerate(self.directions):
                init_dict[direction_obj] = {"dir": float(i)}

            # Place dominoes (D) and targets (T) in order: D D T D T
            # at fixed positions along the x-axis
            n_dominos = rng.choice(possible_num_dominos)
            n_targets = rng.choice(possible_num_targets)
            n_pivots = rng.choice(possible_num_pivots)

            # Generate sequence using helper function
            obj_dict = None
            max_attempts = 1000
            attempts_for_this_task = 0
            for attempt_num in range(max_attempts):
                attempts_for_this_task = attempt_num + 1
                if log_debug:
                    print(f"\nAttempt {attempt_num} for task {i_task}")
                obj_dict = self._generate_domino_sequence(rng,
                                                          n_dominos,
                                                          n_targets,
                                                          n_pivots,
                                                          log_debug=log_debug,
                                                          task_idx=i_task)
                if obj_dict is not None:
                    if log_debug:
                        print("Found satisfying a task")
                    break

            if obj_dict is None:
                raise RuntimeError("Failed to generate valid domino sequence")
            if log_debug:
                print(f"Found a task")

            # If we want to not initialize at finished state, move inter. objs.
            if not CFG.domino_initialize_at_finished_state:
                obj_dict = self._move_intermediate_objects_to_unfinished_state(
                    obj_dict)

            init_dict.update(obj_dict)
            init_state = utils.create_state_from_dict(init_dict)

            # The goal: topple all targets
            if CFG.domino_use_domino_blocks_as_target:
                # Find target dominoes (pink dominoes) and set them as goals
                goal_atoms = set()
                for domino_obj in init_state.get_objects(self._domino_type):
                    if self._TargetDomino_holds(init_state, [domino_obj]):
                        goal_atoms.add(GroundAtom(self._Toppled, [domino_obj]))
            else:
                # Use regular targets
                goal_atoms = {GroundAtom(self._Toppled, [self.targets[0]])}

            tasks.append(EnvironmentTask(init_state, goal_atoms))
            total_attempts += attempts_for_this_task
        if log_debug:
            print(f"Total attempts: {total_attempts}")

        return self._add_pybullet_state_to_tasks(tasks)

    def _place_domino(self,
                      domino_idx: int,
                      x: float,
                      y: float,
                      rot: float,
                      is_start_block: bool = False,
                      is_target_block: bool = False,
                      rng: Optional[np.random.Generator] = None,
                      task_idx: Optional[int] = None) -> Dict:
        """Create a dictionary containing the placement parameters for a
        domino.

        Returns a dictionary with position, orientation, color, and
        state information. The color is determined by the block type:
        light green for start blocks, pink or red for target blocks
        (depending on glued status), and blue for regular blocks.
        """
        # Choose color based on block type
        if is_start_block:
            color = self.start_domino_color
        elif is_target_block:
            # Check if this target domino should be glued
            should_be_glued = False
            if CFG.domino_has_glued_dominos:
                if task_idx == 0:
                    # First task: target is definitely glued
                    should_be_glued = True
                elif task_idx == 1:
                    # Second task: target is definitely not glued
                    should_be_glued = False
                else:
                    # Other tasks: use random sampling with glued_percentage
                    should_be_glued = (rng is not None and
                                       rng.random() < self.glued_percentage)

            if should_be_glued:
                color = self.glued_domino_color
            else:
                color = self.target_domino_color
        else:
            color = self.domino_color

        return {
            "x": x,
            "y": y,
            "z": self.z_lb + self.domino_height / 2,
            "yaw": rot,
            "roll": 0.0,  # All dominos start upright
            "r": color[0],
            "g": color[1],
            "b": color[2],
            "is_held": 0.0,
        }

    def _place_pivot_or_target(self,
                               x: float,
                               y: float,
                               rot: float = 0.0) -> Dict:
        """Create a dictionary containing the placement parameters for a pivot
        or target.

        Returns a dictionary with position and rotation information. The
        z-coordinate is set to table height since pivots and targets
        rest directly on the table surface.
        """
        return {
            "x": x,
            "y": y,
            "z": self.z_lb,
            "yaw": rot,
        }

    def _move_intermediate_objects_to_unfinished_state(self,
                                                       obj_dict: Dict) -> Dict:
        """Move all intermediate dominoes and pivots to the lower end of the
        table in a row, keeping only the start domino and targets in their
        original positions.

        Args:
            obj_dict: Dictionary containing the original positions of all objects

        Returns:
            Modified dictionary with intermediate objects repositioned
        """
        # Identify which objects to move
        intermediate_objects = []

        # Find all dominoes except the start domino (which has light green color)
        # and target dominoes (which have pink color when CFG option is enabled)
        for domino in self.dominos:
            if domino in obj_dict:
                domino_data = obj_dict[domino]
                # Check if it's not a start domino (not light green)
                eps = 1e-3
                is_start_domino = (abs(
                    domino_data.get("r", 0.0) -
                    self.start_domino_color[0]) < eps and abs(
                        domino_data.get("g", 0.0) - self.start_domino_color[1])
                                   < eps and abs(
                                       domino_data.get("b", 0.0) -
                                       self.start_domino_color[2]) < eps)

                # Check if it's a target domino (pink color) when using domino blocks as targets
                is_target_domino = False
                if CFG.domino_use_domino_blocks_as_target:
                    is_target_domino = (abs(
                        domino_data.get("r", 0.0) - self.target_domino_color[0]
                    ) < eps and abs(
                        domino_data.get("g", 0.0) -
                        self.target_domino_color[1]) < eps and abs(
                            domino_data.get("b", 0.0) -
                            self.target_domino_color[2]) < eps) or (abs(
                                domino_data.get("r", 0.0) -
                                self.glued_domino_color[0]) < eps and abs(
                                    domino_data.get("g", 0.0) -
                                    self.glued_domino_color[1]) < eps and abs(
                                        domino_data.get("b", 0.0) -
                                        self.glued_domino_color[2]) < eps)

                # Only move dominoes that are neither start nor target dominoes
                if not is_start_domino and not is_target_domino:
                    intermediate_objects.append((domino, "domino"))

        # Find all pivots
        for pivot in self.pivots:
            if pivot in obj_dict:
                intermediate_objects.append((pivot, "pivot"))

        if not intermediate_objects:
            return obj_dict

        # Original non-grid positioning
        # Calculate positions for intermediate objects
        # Place them in a row near x_lb with even spacing
        start_x = self.x_lb + self.domino_width  # Start a bit inside the boundary
        spacing = self.domino_width * 1.5  # Space between objects
        y_position = (self.y_lb +
                      self.y_ub) / 2  # Middle of the table in y direction

        # Update positions for intermediate objects
        for i, (obj, obj_type) in enumerate(intermediate_objects):
            new_x = start_x + i * spacing

            if obj_type == "domino":
                obj_dict[obj] = {
                    "x": new_x,
                    "y": y_position,
                    "z": self.z_lb + self.domino_height / 2,
                    "yaw": 0.0,  # Reset rotation to upright
                    "roll": 0.0,  # Reset tilt to upright
                    "r": self.domino_color[0],
                    "g": self.domino_color[1],
                    "b": self.domino_color[2],
                    "is_held": 0.0,
                }
            elif obj_type == "pivot":
                obj_dict[obj] = {
                    "x": new_x,
                    "y": y_position,
                    "z": self.z_lb,
                    "yaw": 0.0,  # Reset rotation
                }

        return obj_dict


def create_domino_block(
        color: Tuple[float, float, float, float],
        half_extents: Tuple[float, float, float],
        mass: float,
        # This is the *lateral* friction you already pass to create_pybullet_block
        friction: float,
        position: Pose3D = (0.0, 0.0, 0.0),
        orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
        physics_client_id: int = 0,
        add_top_triangle: bool = False,
        *,
        # --- Domino-friendly extras (all optional) ---
        restitution: float = 0.02,
        rolling_friction: float = 0.006,
        spinning_friction: Optional[
            float] = None,  # default: reuse `friction` if None
        linear_damping: float = 0.0,
        angular_damping: float = 0.03,
        friction_anchor: bool = True,
        ccd: bool = True,
        ccd_swept_radius: Optional[
            float] = None,  # defaults to 0.5 * min(half_extents)
        _ccd_motion_threshold:
    Optional[
        float] = None,  # defaults to 0.5 * min(half_extents) - currently unused
) -> int:
    """Create a 'domino-tuned' block by calling your original
    create_pybullet_block and then applying additional dynamics
    (rolling/spinning friction, damping, CCD).

    Returns:
        PyBullet body unique ID (int).
    """
    import pybullet as p

    # 1) Create the base block using your original function (kept intact).
    block_id = create_pybullet_block(
        color=color,
        half_extents=half_extents,
        mass=mass,
        friction=friction,
        position=position,
        orientation=orientation,
        physics_client_id=physics_client_id,
        add_top_triangle=add_top_triangle,
    )

    # 2) Domino-friendly dynamics.
    if spinning_friction is None:
        spinning_friction = friction  # reuse user's lateral friction unless specified

    p.changeDynamics(
        block_id,
        linkIndex=-1,
        lateralFriction=friction,
        rollingFriction=rolling_friction,
        spinningFriction=spinning_friction,
        restitution=restitution,
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        frictionAnchor=friction_anchor,
        physicsClientId=physics_client_id,
    )

    # 3) Continuous Collision Detection to prevent tunneling at speed.
    if ccd:
        m = min(half_extents)
        swept = ccd_swept_radius if ccd_swept_radius is not None else 0.5 * m
        # Note: ccdMotionThreshold is commented out but kept for reference
        # thresh = _ccd_motion_threshold if _ccd_motion_threshold is not None else 0.5 * m
        p.changeDynamics(
            block_id,
            linkIndex=-1,
            ccdSweptSphereRadius=swept,
            # ccdMotionThreshold=thresh,
            physicsClientId=physics_client_id,
        )

    return block_id


if __name__ == "__main__":

    CFG.seed = 2
    CFG.env = "pybullet_domino"
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.num_train_tasks = 1
    CFG.num_test_tasks = 2
    env = PyBulletDominoEnv(use_gui=True)

    # Generate grid predicates with a 3x3 grid
    grid_gap = env.pos_gap  # Use the default gap
    grid_predicates = env.generate_grid_position_predicates(grid_gap=grid_gap,
                                                            num_x_cells=2,
                                                            num_y_cells=2)
    rot_predicates = env.generate_rotation_predicates(8)

    print(f"\nGenerated {len(grid_predicates)} grid position predicates with:")
    print(f"  grid_gap: {grid_gap:.4f}")
    print(f"  num_x_cells: 2 (range: [-2, 2])")
    print(f"  num_y_cells: 2 (range: [-2, 2])")
    print(f"\nPredicate names:")
    for pred in sorted(grid_predicates, key=lambda p: p.name):
        print(f"  - {pred.name}")

    tasks = env._generate_train_tasks()

    for task in tasks:
        env._reset_state(task.init)
        print("\n" + "=" * 60)
        print("Task State Information")
        print("=" * 60)
        print(
            f"\ninit state: {pformat(utils.abstract(task.init, env.predicates))}\n"
        )
        print(f"goal: {task.goal}\n")

        # Test grid predicates on the initial state
        print("\n" + "=" * 60)
        print("Grid Position Predicate Evaluation")
        print("=" * 60)
        dominos = task.init.get_objects(env._domino_type)
        print(f"\nEvaluating grid predicates for {len(dominos)} dominoes:")
        for domino in dominos:
            print(f"\n  Domino: {domino.name}")
            print(f"    Position: ({task.init.get(domino, 'x'):.4f}, "
                  f"{task.init.get(domino, 'y'):.4f})")
            print(f"    Rotation: {task.init.get(domino, 'yaw'):.4f}")
            print(f"    Grid predicates that hold:")
            for pred in sorted(grid_predicates, key=lambda p: p.name):
                if pred.holds(task.init, [domino]):
                    print(f"      ✓ {pred.name}")

        print("\n" + "=" * 60)
        print("Running simulation")
        print("=" * 60)
        for i in range(10000):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
