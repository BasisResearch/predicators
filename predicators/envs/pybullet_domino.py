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
from predicators.structs import Action, DerivedPredicate, EnvironmentTask, \
    GroundAtom, Object, Predicate, State, Type


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
    turn_shift_frac: ClassVar[float] = 0.6
    # domino_mass: ClassVar[float] = 0.3
    domino_mass: ClassVar[float] = 0.1
    domino_friction: ClassVar[float] = 0.5
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (1.0, 0.75, 0.8, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)
    start_domino_x: ClassVar[float] = x_lb + domino_width
    start_domino_y: ClassVar[float] = y_lb + domino_width
    domino_roll_threshold: ClassVar[float] = 0.1
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
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
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

        # Conditionally create grid-related types
        if CFG.domino_use_grid:
            self._position_type = Type("loc", ["xx", "yy"],
                                       sim_features=["id", "xx", "yy"])
            self._angle_type = Type("angle", ["angle"])

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

        # Conditionally create position objects for grid
        if CFG.domino_use_grid:
            # Create rotation objects for 8 discrete angles
            self.rotations: List[Object] = []
            angle_values = [-135, -90, -45, 0, 45, 90, 135, 180]  # degrees
            for angle in angle_values:
                name = f"ang_{angle}"
                obj = Object(name, self._angle_type)
                self.rotations.append(obj)
        else:
            # Initialize empty lists when grid is not used
            self.rotations: List[Object] = []
        self.grid_pos = []

        self.block_constraints = []
        self._debug_line_ids = []

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
        self._StartBlock = Predicate("StartBlock", [self._domino_type],
                                     self._StartBlock_holds)
        self._MovableBlock = Predicate("MovableBlock", [self._domino_type],
                                       self._MovableBlock_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._domino_type],
                                  self._Holding_holds)
        # Define DominoAtPos and DominoAtRot first if using grid
        if CFG.domino_use_grid:
            self._DominoAtPos = Predicate(
                "DominoAtPos", [self._domino_type, self._position_type],
                self._DominoAtPos_holds)
            self._DominoAtRot = Predicate(
                "DominoAtRot", [self._domino_type, self._angle_type],
                self._DominoAtRot_holds)
            self._Connected = Predicate(
                "Connected", [self._position_type, self._position_type],
                self._Connected_holds)
            self._PosClear = Predicate("PosClear", [self._position_type],
                                       self._PosClear_holds)

            if CFG.domino_use_grid:
                self._InFrontDirection = DerivedPredicate(
                    "InFrontDirection", [
                        self._domino_type, self._domino_type,
                        self._direction_type
                    ],
                    self._InFrontDirection_holds,
                    auxiliary_predicates=[
                        self._DominoAtPos, self._DominoAtRot
                    ])
                self._InFront = DerivedPredicate(
                    "InFront", [self._domino_type, self._domino_type],
                    self._InFront_holds,
                    auxiliary_predicates=[self._InFrontDirection])
                self._AdjacentTo = DerivedPredicate(
                    "AdjacentTo", [self._position_type, self._domino_type],
                    self._AdjacentTo_holds,
                    auxiliary_predicates=[self._DominoAtPos])

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino"

    @property
    def predicates(self) -> Set[Predicate]:
        base_predicates = {
            self._Toppled,
            self._Upright,
            self._Tilting,
            self._StartBlock,
            self._MovableBlock,
            self._HandEmpty,
            self._Holding,
            self._InFrontDirection,
            self._InFront,
        }

        if CFG.domino_use_grid:
            base_predicates.update({
                self._DominoAtPos,
                self._DominoAtRot,
                self._PosClear,
            })
            if CFG.domino_include_connected_predicate:
                base_predicates.update({self._Connected})
            else:
                base_predicates.update({self._AdjacentTo})

        return base_predicates

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # The goal is always to topple all targets
        return {self._Toppled}

    @property
    def types(self) -> Set[Type]:
        base_types = {
            self._robot_type, self._domino_type, self._target_type,
            self._pivot_type, self._direction_type
        }

        if CFG.domino_use_grid:
            base_types.update({self._position_type, self._angle_type})

        return base_types

    # -------------------------------------------------------------------------
    # Grid Coordinate Generation

    @classmethod
    def _generate_grid_coordinates(
            cls, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates for position objects with specified
        dimensions."""
        # Calculate grid extents based on workspace bounds
        total_x_range = cls.x_ub - cls.x_lb
        total_y_range = cls.y_ub - cls.y_lb

        # Center the grid within the workspace
        x_start = cls.x_lb + (total_x_range -
                              (num_pos_x - 1) * cls.pos_gap) / 2
        y_start = cls.y_lb + (total_y_range -
                              (num_pos_y - 1) * cls.pos_gap) / 2

        x_coords = [
            round(x_start + i * cls.pos_gap, 5) for i in range(num_pos_x)
        ]
        y_coords = [
            round(y_start + i * cls.pos_gap, 5) for i in range(num_pos_y)
        ]

        return x_coords, y_coords

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
                      position=[
                          cls.table_pos[0],
                          cls.table_pos[1] + (cls.y_ub - cls.y_lb) / 2,
                          cls.table_pos[2]
                      ],
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
                orientation=[0.0, 0.0, 0.0],
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

    def _create_task_specific_objects(self, state):
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
        elif CFG.domino_use_grid and obj.type == self._position_type:
            if feature == "xx":
                return obj.xx
            elif feature == "yy":
                return obj.yy
        elif CFG.domino_use_grid and obj.type == self._angle_type:
            if feature == "angle":
                # Extract angle from object name (e.g., "ang_45" -> 45.0)
                angle_str = obj.name.split("_")[1]
                return float(angle_str)

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _create_invisible_link_body(self) -> int:
        """Create a zero-mass, zero-collision 'rod' (link) in PyBullet.

        We'll attach each domino to this rod with a hinge joint.
        """
        # A tiny sphere collision shape (or None) so it doesn't collide / add mass.
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.0001,  # effectively 0
            physicsClientId=self._physics_client_id)

        # Visual shape is also effectively invisible.
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.00001,
            rgbaColor=[0, 0, 0, 0],  # transparent
            physicsClientId=self._physics_client_id)

        # Create the multi-body with mass=0, so it never moves on its own.
        rod_body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, 10],  # spawn out of the way, reposition below
            useMaximalCoordinates=True,  # simpler, no internal links
            physicsClientId=self._physics_client_id)
        return rod_body_id

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

    def _no_target_in_between(self, state, domino1: Object,
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

        # Draw debug lines at grid cell centers based on current task configuration
        if CFG.domino_use_grid:
            # Clear existing debug lines
            for line_id in self._debug_line_ids:
                p.removeUserDebugItem(line_id)
            self._debug_line_ids = []

            # Draw debug lines based on position objects' xx, yy features
            position_objs = state.get_objects(self._position_type)
            for pos_obj in position_objs:
                x = state.get(pos_obj, "xx")
                y = state.get(pos_obj, "yy")
                line_id = p.addUserDebugLine(
                    [x, y, self.table_height],
                    [x, y, self.table_height + self.debug_line_height],
                    [1, 0, 0],
                    parentObjectUniqueId=-1,
                    parentLinkIndex=-1)
                self._debug_line_ids.append(line_id)

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
        if abs(state.get(domino, "r") - cls.target_domino_color[0]) > eps:
            return False
        if abs(state.get(domino, "g") - cls.target_domino_color[1]) > eps:
            return False
        if abs(state.get(domino, "b") - cls.target_domino_color[2]) > eps:
            return False
        return True

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
    def _InFrontDirection_holds(cls, atoms: Set[GroundAtom],
                                objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in the given direction.

        This is an optimized implementation for heuristic evaluation. It
        decouples the positional and rotational checks to be much faster.
        It remains correct for concrete states but may produce false
        positives in relaxed states (which is acceptable for a heuristic).

        The relationship is symmetric: InFrontDirection(d1, d2, "right") is
        true if either:
        - d1 is in the cell in front of d2 with a rotation difference of -π/4, OR
        - d2 is in the cell in front of d1 with a rotation difference of +π/4
          (equivalent to InFrontDirection(d2, d1, "left")).
        """
        domino1, domino2, direction_obj = objects

        if not CFG.domino_use_grid:
            raise ValueError("Grid is not used, this derived predicate cannot "
                             "function")

        # Helper functions to parse object names and cache results
        _pos_coord_cache = {}
        _rot_rad_cache = {}

        def extract_grid_coords(pos_obj):
            if pos_obj in _pos_coord_cache:
                return _pos_coord_cache[pos_obj]
            name_parts = pos_obj.name.split("_")
            y_idx = int(name_parts[1][1:])
            x_idx = int(name_parts[2][1:])
            result = (x_idx, y_idx)
            _pos_coord_cache[pos_obj] = result
            return result

        def extract_rotation_angle_rad(rot_obj):
            if rot_obj in _rot_rad_cache:
                return _rot_rad_cache[rot_obj]
            angle_str = rot_obj.name.split("_")[1]
            result = np.radians(float(angle_str))
            _rot_rad_cache[rot_obj] = result
            return result

        # Step 1: Gather all possible states for each domino.
        d1_positions_coords = {
            extract_grid_coords(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtPos"
            and atom.objects[0] == domino1
        }
        d1_rotations_rad = {
            extract_rotation_angle_rad(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtRot"
            and atom.objects[0] == domino1
        }
        d2_positions_coords = {
            extract_grid_coords(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtPos"
            and atom.objects[0] == domino2
        }
        d2_rotations_rad = {
            extract_rotation_angle_rad(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtRot"
            and atom.objects[0] == domino2
        }

        # Step 2: Define the optimized function to check one directional case.
        def _check_case(front_domino_positions: Set[Tuple[int, int]],
                        front_domino_rotations: Set[float],
                        back_domino_positions: Set[Tuple[int, int]],
                        back_domino_rotations: Set[float],
                        direction_name: str,
                        tolerance: float = 1e-6) -> bool:
            """Perform decoupled checks for positional and rotational
            possibility."""
            # Fail fast if any required sets of states are empty.
            if not all([
                    front_domino_positions, front_domino_rotations,
                    back_domino_positions, back_domino_rotations
            ]):
                return False

            # 2a. Positional Check: Is there ANY valid geometric placement?
            position_possible = False
            for (x_back_idx, y_back_idx) in back_domino_positions:
                for rot_back_rad in back_domino_rotations:
                    # Relationship only holds for cardinal rotations of back domino.
                    if not (abs(np.sin(rot_back_rad)) < tolerance or \
                            abs(np.cos(rot_back_rad)) < tolerance):
                        continue
                    # Calculate expected position and check if it exists.
                    dx_idx = round(np.sin(rot_back_rad))
                    dy_idx = round(np.cos(rot_back_rad))
                    expected_front_coords = (x_back_idx + dx_idx,
                                             y_back_idx + dy_idx)
                    if expected_front_coords in front_domino_positions:
                        position_possible = True
                        break
                if position_possible:
                    break

            # If it's not positionally possible, no need to check rotation.
            if not position_possible:
                return False

            # 2b. Rotational Check: Is there ANY pair of rotations with the correct diff?
            if direction_name == "left":
                expected_rot_diff = np.pi / 4
            elif direction_name == "straight":
                expected_rot_diff = 0
            elif direction_name == "right":
                expected_rot_diff = -np.pi / 4
            else:
                return False  # Should not happen

            for rot_back_rad in back_domino_rotations:
                for rot_front_rad in front_domino_rotations:
                    diff = utils.wrap_angle(rot_front_rad - rot_back_rad)
                    if abs(diff - expected_rot_diff) < tolerance:
                        # Position is possible and rotation is possible, so we're done.
                        return True

            return False

        # Step 3: Check both symmetric cases for the relationship.
        dir_name = direction_obj.name
        if dir_name == "left":
            opposite_dir_name = "right"
        elif dir_name == "right":
            opposite_dir_name = "left"
        else:  # "straight"
            opposite_dir_name = "straight"

        # Case 1: Is domino1 in front of domino2 in `dir_name`?
        if _check_case(front_domino_positions=d1_positions_coords,
                       front_domino_rotations=d1_rotations_rad,
                       back_domino_positions=d2_positions_coords,
                       back_domino_rotations=d2_rotations_rad,
                       direction_name=dir_name):
            return True

        # Case 2: Is domino2 in front of domino1 in `opposite_dir_name`?
        if _check_case(front_domino_positions=d2_positions_coords,
                       front_domino_rotations=d2_rotations_rad,
                       back_domino_positions=d1_positions_coords,
                       back_domino_rotations=d1_rotations_rad,
                       direction_name=opposite_dir_name):
            return True

        return False

    @classmethod
    def _InFront_holds(cls, atoms: Set[GroundAtom],
                       objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in any direction.

        This derived predicate returns True if there exists any
        direction such that InFrontDirection(domino1, domino2,
        direction) is true.
        """
        domino1, domino2 = objects

        # Check if there exists any InFrontDirection atom with these dominos
        for atom in atoms:
            if (atom.predicate.name == "InFrontDirection"
                    and len(atom.objects) == 3 and atom.objects[0] == domino1
                    and atom.objects[1] == domino2):
                return True

        return False

    @classmethod
    def _DominoAtPos_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific position."""
        domino, position = objects
        if state.get(domino, "is_held"):
            return False

        # Get domino's actual position
        domino_x = state.get(domino, "x")
        domino_y = state.get(domino, "y")

        # Get the target position
        target_x = state.get(position, "xx")
        target_y = state.get(position, "yy")

        # Check if domino is close enough to the target position
        position_tolerance = cls.pos_gap * 0.5
        return (abs(domino_x - target_x) <= position_tolerance
                and abs(domino_y - target_y) <= position_tolerance)

    @classmethod
    def _DominoAtRot_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific rotation."""
        domino, rotation = objects
        if state.get(domino, "is_held"):
            return False

        # Get domino's actual rotation (in radians)
        domino_rot = state.get(domino, "yaw")

        # Get the target rotation (convert from degrees to radians)
        target_rot_degrees = state.get(rotation, "angle")
        target_rot_radians = np.radians(target_rot_degrees)

        # Check if domino rotation is close enough to target rotation
        rotation_tolerance = np.radians(15)  # 15 degrees tolerance
        angle_diff = abs(utils.wrap_angle(domino_rot - target_rot_radians))

        return angle_diff <= rotation_tolerance

    @classmethod
    def _Connected_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if two positions are adjacent in cardinal directions only.

        Returns True if positions are adjacent up/down or left/right,
        but False for diagonal adjacencies.
        """
        pos1, pos2 = objects
        if pos1.name == pos2.name:
            return False

        # Get coordinates of both positions
        x1 = state.get(pos1, "xx")
        y1 = state.get(pos1, "yy")
        x2 = state.get(pos2, "xx")
        y2 = state.get(pos2, "yy")

        # Calculate differences
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # Positions are connected if they are exactly one grid step apart
        # in only one direction (either x or y, but not both)
        grid_step = cls.pos_gap
        tolerance = grid_step * 0.1  # Small tolerance for floating point comparison

        # Check if adjacent in x-direction only (same row)
        x_adjacent = abs(dx - grid_step) < tolerance and dy < tolerance

        # Check if adjacent in y-direction only (same column)
        y_adjacent = abs(dy - grid_step) < tolerance and dx < tolerance

        return x_adjacent or y_adjacent

    @classmethod
    def _PosClear_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Check if a position is clear (not occupied by any domino).

        A position is considered clear if no domino is currently at that
        position.
        """
        position, = objects

        # Get the position coordinates
        target_x = state.get(position, "xx")
        target_y = state.get(position, "yy")

        # Check if any domino is at this position
        position_tolerance = cls.pos_gap * 0.5
        for domino in state.get_objects(cls._domino_type):
            domino_x = state.get(domino, "x")
            domino_y = state.get(domino, "y")

            # If domino is close enough to this position, position is not clear
            if (abs(domino_x - target_x) <= position_tolerance
                    and abs(domino_y - target_y) <= position_tolerance
                    and not state.get(domino, "is_held")):
                return False

        return True

    @classmethod
    def _AdjacentTo_holds(cls, atoms: Set[GroundAtom],
                          objects: Sequence[Object]) -> bool:
        """Check if a position is adjacent to a domino in cardinal directions.

        This is similar to _InFrontDirection_holds but checks if a position
        is adjacent to any position where the domino could be placed, considering
        that the domino can be in multiple positions during heuristic computation.

        Adjacent positions are those that are exactly one grid step away in
        cardinal directions (up, down, left, right) but not diagonal.
        """
        position, domino = objects

        if not CFG.domino_use_grid:
            raise ValueError("Grid is not used, this derived predicate cannot "
                             "function")

        # Helper functions to parse object names and cache results
        _pos_coord_cache = {}

        def extract_grid_coords(pos_obj):
            if pos_obj in _pos_coord_cache:
                return _pos_coord_cache[pos_obj]
            name_parts = pos_obj.name.split("_")
            y_idx = int(name_parts[1][1:])
            x_idx = int(name_parts[2][1:])
            result = (x_idx, y_idx)
            _pos_coord_cache[pos_obj] = result
            return result

        # Get coordinates of the target position
        target_coords = extract_grid_coords(position)
        target_x_idx, target_y_idx = target_coords

        # Get all possible positions where the domino could be
        domino_positions_coords = {
            extract_grid_coords(atom.objects[1])
            for atom in atoms if atom.predicate.name == "DominoAtPos"
            and atom.objects[0] == domino
        }

        # Check if the target position is adjacent to any domino position
        # Adjacent means exactly one grid step away in cardinal directions
        for domino_x_idx, domino_y_idx in domino_positions_coords:
            # Calculate the difference in grid coordinates
            dx = abs(target_x_idx - domino_x_idx)
            dy = abs(target_y_idx - domino_y_idx)

            # Adjacent in cardinal directions means:
            # - Exactly 1 step away in one direction AND 0 steps in the other
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                return True

        return False

    # -------------------------------------------------------------------------
    # Task Generation

    def _generate_domino_sequence(self,
                                  rng: np.random.Generator,
                                  n_dominos: int,
                                  n_targets: int,
                                  n_pivots: int,
                                  log_debug: bool = False) -> Optional[Dict]:
        """Generate a sequence of dominoes, targets, and pivots.

        Returns:
            Dict mapping objects to their placement parameters, or None if failed
        """

        def _in_bounds(nx: float, ny: float) -> bool:
            """Check if (nx, ny) is within table boundaries."""
            return self.x_lb < nx < self.x_ub and self.y_lb < ny < self.y_ub

        obj_dict = {}
        domino_count = 0
        target_count = 0
        pivot_count = 0
        just_placed_target = False
        just_turned_90 = False
        success = True

        # Initial domino position and orientation
        x = rng.uniform(self.x_lb, self.x_ub)
        y = rng.uniform(self.y_lb + self.domino_width,
                        self.y_ub - 3 * self.domino_width)
        rot = rng.choice([0, np.pi / 2, -np.pi / 2])
        gap = self.pos_gap

        # Place first domino (start block)
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count, x, y, rot, is_start_block=True)
        domino_count += 1

        turn_choices = self.turn_choices.copy()
        if pivot_count == n_pivots:
            turn_choices.remove("pivot180")

        # Determine loop condition based on CFG
        if CFG.domino_use_domino_blocks_as_target:
            expected_total_dominoes = n_dominos + n_targets
            loop_condition = domino_count < expected_total_dominoes or target_count < n_targets
        else:
            loop_condition = domino_count < n_dominos or target_count < n_targets

        # Main placement loop
        while loop_condition:
            can_place_target = (domino_count >= 2 and target_count < n_targets
                                and not just_placed_target)
            must_place_domino = not can_place_target

            if CFG.domino_use_domino_blocks_as_target:
                expected_total_dominoes = n_dominos + n_targets
                can_place_domino = domino_count + n_targets < expected_total_dominoes
            else:
                can_place_domino = domino_count < n_dominos

            if (must_place_domino or rng.random() > 0.5) and can_place_domino:
                # Place domino (or pivot)
                # Update turn choices first
                turn_choices = self.turn_choices.copy()
                if pivot_count == n_pivots:
                    turn_choices.remove("pivot180")

                result = self._place_next_domino(rng, obj_dict, x, y, rot, gap,
                                                 domino_count, pivot_count,
                                                 n_pivots, n_dominos,
                                                 turn_choices,
                                                 just_placed_target,
                                                 just_turned_90, _in_bounds)
                if not result[0]:
                    return None

                x, y, rot = result[1], result[2], result[3]
                domino_count = result[4]
                pivot_count = result[5]
                just_turned_90 = result[6]
                just_placed_target = False

            else:
                # Place target
                if log_debug:
                    print("Placing target")
                result = self._place_next_target(rng, obj_dict, x, y, rot, gap,
                                                 domino_count, target_count,
                                                 _in_bounds)
                if not result[0]:
                    return None

                x, y, rot = result[1], result[2], result[3]
                domino_count = result[4]
                target_count = result[5]
                just_placed_target = True
                just_turned_90 = False

            # Update loop condition
            if CFG.domino_use_domino_blocks_as_target:
                expected_total_dominoes = n_dominos + n_targets
                loop_condition = domino_count < expected_total_dominoes or target_count < n_targets
            else:
                loop_condition = domino_count < n_dominos or target_count < n_targets

        # Check if we successfully placed everything
        if CFG.domino_use_domino_blocks_as_target:
            expected_domino_count = n_dominos + n_targets
            success = (domino_count == expected_domino_count
                       and target_count == n_targets
                       and pivot_count == n_pivots)
        else:
            success = (domino_count == n_dominos and target_count == n_targets
                       and pivot_count == n_pivots)

        return obj_dict if success else None

    def _generate_domino_sequence_with_grid(
            self,
            rng: np.random.Generator,
            n_dominos: int,
            n_targets: int,
            num_pos_x: int,
            num_pos_y: int,
            log_debug: bool = False) -> Optional[Dict]:
        """Grid-based sequence generator.

        This version implements straight moves and L-shaped 90-degree
        turns, mimicking the logic of the non-grid-based generator. A
        90-degree turn consumes two dominoes and forms an 'L' shape on
        the grid. The turning domino is shifted inward for better
        stability.
        """
        obj_dict: Dict = {}
        domino_count = 0
        target_count = 0
        used_coords = set()

        # Generate grid coordinates for this specific configuration
        x_coords, y_coords = self._generate_grid_coordinates(
            num_pos_x, num_pos_y)
        grid_pos = [(x, y) for y in y_coords for x in x_coords]

        # Use a set for efficient checking of valid grid coordinates.
        grid_coords_set = set(grid_pos)

        # Choose a random starting position and orientation (cardinal directions).
        start_idx = rng.choice(len(grid_pos))
        curr_x, curr_y = grid_pos[start_idx]
        # If in the top row, can't face down because it's unreachable for the
        # robot
        top_row_y = np.max([y for _, y in grid_pos])
        if np.abs(curr_y - top_row_y) < 1e-3:
            curr_rot = rng.choice([np.pi / 2, np.pi / 2])
        else:
            curr_rot = rng.choice([0, np.pi / 2, np.pi, -np.pi / 2])
        used_coords.add((curr_x, curr_y))

        # Place the first domino (start block).
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count, curr_x, curr_y, curr_rot, is_start_block=True)
        domino_count += 1
        if log_debug:
            print(f"Placed first domino at {curr_x}, {curr_y}, {curr_rot}")

        # Determine total domino blocks to place.
        if CFG.domino_use_domino_blocks_as_target:
            total_domino_blocks = n_dominos + n_targets
        else:
            total_domino_blocks = n_dominos

        # Main placement loop.
        while domino_count < total_domino_blocks:
            possible_moves = []
            # A move is defined by: (name, final_x, final_y, final_rot, placements)
            # where placements is a list of (x, y, rot) for each domino in the move.

            # 1. Check for a "straight" move (1 domino).
            dx = round(self.pos_gap * np.sin(curr_rot), 5)
            dy = round(self.pos_gap * np.cos(curr_rot), 5)
            next_x = round(curr_x + dx, 5)
            next_y = round(curr_y + dy, 5)

            if (next_x, next_y) in grid_coords_set and \
            (next_x, next_y) not in used_coords:
                placements = [(next_x, next_y, curr_rot)]
                possible_moves.append(
                    ("straight", next_x, next_y, curr_rot, placements))

            # 2. Check for "turn" moves (2 dominoes).
            if (total_domino_blocks - domino_count) >= 2:
                # turn_dir: -1 for left, 1 for right.
                for turn_dir, name in [(-1, "turn_left"), (1, "turn_right")]:
                    # The first domino (d1) is one step straight on the grid.
                    d1_grid_x, d1_grid_y = next_x, next_y
                    if (d1_grid_x, d1_grid_y) not in grid_coords_set or \
                       (d1_grid_x, d1_grid_y) in used_coords:
                        continue

                    # Its orientation is 45 degrees towards the turn direction.
                    d1_rot = curr_rot - turn_dir * np.pi / 4

                    # Calculate the shift vector to pull the turning domino inward.
                    shift_magnitude = self.domino_width * self.turn_shift_frac
                    shift_dx = shift_magnitude * \
                        (turn_dir * np.cos(curr_rot) - np.sin(curr_rot))
                    shift_dy = shift_magnitude * \
                        (-turn_dir * np.sin(curr_rot) - np.cos(curr_rot))

                    # The physical position is the grid position plus the shift.
                    d1_x = d1_grid_x + shift_dx
                    d1_y = d1_grid_y + shift_dy

                    # The second domino (d2) completes the turn.
                    d2_rot = d1_rot - turn_dir * 1 * np.pi / 4

                    # Calculate d2's physical position relative to d1's.
                    gap = self.pos_gap
                    sin_d1 = np.sin(d1_rot)
                    cos_d1 = np.cos(d1_rot)
                    disp_x = (
                        gap * turn_dir * cos_d1 +
                        (2 * shift_magnitude - gap) * sin_d1) / np.sqrt(2)
                    disp_y = (
                        -gap * turn_dir * sin_d1 +
                        (2 * shift_magnitude - gap) * cos_d1) / np.sqrt(2)
                    d2_x = round(d1_x + disp_x, 5)
                    d2_y = round(d1_y + disp_y, 5)

                    # Check if the grid position of the second domino is valid.
                    # We need to determine where the grid position *would* be.
                    # A left turn from rot=pi should result in rot=pi/2.
                    # This is equivalent to rot + turn_dir * pi/2
                    expected_final_rot = curr_rot + turn_dir * np.pi / 2
                    d2_grid_dx = round(
                        self.pos_gap * np.sin(expected_final_rot), 5)
                    d2_grid_dy = round(
                        self.pos_gap * np.cos(expected_final_rot), 5)
                    d2_grid_x = round(d1_grid_x + d2_grid_dx, 5)
                    d2_grid_y = round(d1_grid_y + d2_grid_dy, 5)

                    if (d2_grid_x, d2_grid_y) in grid_coords_set and \
                       (d2_grid_x, d2_grid_y) not in used_coords:

                        placements = [(d1_x, d1_y, d1_rot),
                                      (d2_x, d2_y, d2_rot)]

                        possible_moves.append(
                            (name, d2_grid_x, d2_grid_y, d2_rot, placements))

            if not possible_moves:
                # No valid moves, generation failed for this attempt.
                return None

            # Choose a random valid move and get its placement plan.
            _move_name, final_x, final_y, final_rot, placements = \
                possible_moves[rng.choice(len(possible_moves))]
            if log_debug:
                print(
                    f"Chose move: {_move_name}, final_x: {final_x}, final_y: {final_y}, final_rot: {final_rot}"
                )

            # Execute the placement plan for the chosen move.
            for (x, y, rot) in placements:
                if log_debug:
                    print(f"Placing domino at {x}, {y}, {rot}")
                if domino_count >= total_domino_blocks:
                    break  # Should not be reached with correct logic.

                # Decide if this domino block should be a target.
                is_target = False
                if CFG.domino_use_domino_blocks_as_target and \
                target_count < n_targets:
                    remaining_blocks = total_domino_blocks - domino_count
                    remaining_targets = n_targets - target_count

                    # Reserve one target for the very last domino in the sequence
                    is_last_block = (domino_count == total_domino_blocks - 1)
                    has_targets_left = remaining_targets > 0

                    if is_last_block and has_targets_left:
                        # Force the last domino to be a target if we still need targets
                        is_target = True
                    elif not is_last_block and remaining_targets > 1:
                        # For non-last dominoes, only consider making them targets if we have more than 1 target left
                        # This ensures at least one target is reserved for the end
                        targets_available_for_placement = remaining_targets - 1
                        if (targets_available_for_placement >= remaining_blocks - 1 or \
                            rng.random() < targets_available_for_placement / (remaining_blocks - 1)) and\
                            domino_count >= 2:
                            is_target = True

                # Place the domino block.
                obj_dict[self.dominos[domino_count]] = self._place_domino(
                    domino_count, x, y, rot, is_target_block=is_target)

                # Use the grid coordinates for tracking used spots.
                # Find the closest grid coordinate to the physical placement
                closest_grid_coord = min(grid_pos,
                                         key=lambda p: (p[0] - x)**2 +
                                         (p[1] - y)**2)
                used_coords.add(closest_grid_coord)

                if is_target:
                    target_count += 1
                domino_count += 1

            # Update state for the next iteration.
            curr_x, curr_y, curr_rot = final_x, final_y, final_rot

        # Place non-block targets if necessary (not on grid centers).
        if not CFG.domino_use_domino_blocks_as_target:
            raise NotImplementedError(
                "Placing non-block targets with the grid generator is not "
                "currently supported.")

        return obj_dict

    def _place_next_domino(
            self, rng: np.random.Generator, obj_dict: Dict, x: float, y: float,
            rot: float, gap: float, domino_count: int, pivot_count: int,
            n_pivots: int, n_dominos: int, turn_choices: List[str],
            just_placed_target: bool, just_turned_90: bool,
            _in_bounds: Callable[[float, float], bool]) -> Tuple:
        """Place the next domino in the sequence."""
        # Choose placement strategy
        choices = turn_choices.copy()
        if just_turned_90:
            choices.remove("turn90")
        if just_placed_target:
            choices = ["straight"]
        choice = rng.choice(choices)
        print(f"Choice: {choice}")

        if choice == "straight":
            return self._place_straight_domino(obj_dict, x, y, rot, gap,
                                               domino_count, _in_bounds)
        elif choice == "turn90":
            return self._place_turn90_domino(rng, obj_dict, x, y, rot, gap,
                                             domino_count, n_dominos,
                                             _in_bounds)
        elif choice == "pivot180" and pivot_count < n_pivots:
            return self._place_pivot180_domino(rng, obj_dict, x, y, rot, gap,
                                               domino_count, pivot_count,
                                               _in_bounds)
        else:
            # Fallback to straight
            return self._place_straight_domino(obj_dict, x, y, rot, gap,
                                               domino_count, _in_bounds)

    def _place_straight_domino(
            self, obj_dict: Dict, x: float, y: float, rot: float, gap: float,
            domino_count: int, _in_bounds: Callable[[float, float],
                                                    bool]) -> Tuple:
        """Place a domino straight ahead."""
        dy = gap * np.cos(rot)
        dx = gap * np.sin(rot)
        nx, ny = x + dx, y + dy
        if not _in_bounds(nx, ny):
            return (False, x, y, rot, domino_count, 0, False)

        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count, nx, ny, rot, is_start_block=False)
        return (True, nx, ny, rot, domino_count + 1, 0, False)

    def _place_turn90_domino(
            self, rng: np.random.Generator, obj_dict: Dict, x: float, y: float,
            rot: float, gap: float, domino_count: int, n_dominos: int,
            _in_bounds: Callable[[float, float], bool]) -> Tuple:
        """Place dominoes in a 90-degree turn."""
        # Check we have enough dominos left
        if domino_count + 1 >= n_dominos:
            # Fallback to straight
            dy = gap * np.cos(rot)
            dx = gap * np.sin(rot)
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny):
                return (False, x, y, rot, domino_count, 0, False)

            obj_dict[self.dominos[domino_count]] = self._place_domino(
                domino_count, nx, ny, rot, is_start_block=False)
            return (True, nx, ny, rot, domino_count + 1, 0, True)
        else:
            # Turn 45° twice
            turn_dir = rng.choice([-1, 1])
            half_turn = np.pi / 4 * turn_dir

            # First 45°
            side_offset = 0  # Original had side_offset = 0
            rot += half_turn
            dx = gap * np.sin(rot)
            dy = gap * np.cos(rot)
            dx -= turn_dir * side_offset * np.cos(rot)
            dy += turn_dir * side_offset * np.sin(rot)
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny):
                return (False, x, y, rot, domino_count, 0, False)

            obj_dict[self.dominos[domino_count]] = self._place_domino(
                domino_count, nx, ny, rot + np.pi / 2, is_start_block=False)
            domino_count += 1

            # Second 45°
            side_offset = self.domino_width / 2
            rot += half_turn
            dx = gap * np.sin(rot)
            dy = gap * np.cos(rot)
            dx -= turn_dir * side_offset * np.cos(rot)
            dy += turn_dir * side_offset * np.sin(rot)
            nx, ny = nx + dx, ny + dy
            if not _in_bounds(nx, ny):
                return (False, x, y, rot, domino_count, 0, False)

            obj_dict[self.dominos[domino_count]] = self._place_domino(
                domino_count, nx, ny, rot, is_start_block=False)
            return (True, nx, ny, rot, domino_count + 1, 0, True)

    def _place_pivot180_domino(
            self, rng: np.random.Generator, obj_dict: Dict, x: float, y: float,
            rot: float, gap: float, domino_count: int, pivot_count: int,
            _in_bounds: Callable[[float, float], bool]) -> Tuple:
        """Place a pivot and domino with 180-degree flip."""
        pivot_dir = rng.choice([-1, 1])
        side_offset = self.pivot_width / 2

        # Place pivot
        pivot_x = x + gap * (2 / 3) * np.sin(rot)
        pivot_y = y + gap * (2 / 3) * np.cos(rot)
        pivot_x -= pivot_dir * side_offset * np.cos(rot)
        pivot_y -= pivot_dir * side_offset * np.sin(rot)

        if not _in_bounds(pivot_x, pivot_y):
            return (False, x, y, rot, domino_count, pivot_count, False)

        obj_dict[self.pivots[pivot_count]] = self._place_pivot_or_target(
            pivot_x, pivot_y, rot)
        pivot_count += 1

        # Place domino after 180° flip
        back_x = pivot_x - (gap * (2 / 3)) * np.sin(rot)
        back_y = pivot_y - (gap * (2 / 3)) * np.cos(rot)
        back_x -= pivot_dir * side_offset * np.cos(rot)
        back_y += pivot_dir * side_offset * -np.sin(rot)

        if not _in_bounds(back_x, back_y):
            return (False, x, y, rot, domino_count, pivot_count, False)

        new_rot = rot + np.pi  # 180° flip
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count, back_x, back_y, new_rot, is_start_block=False)

        return (True, back_x, back_y, new_rot, domino_count + 1, pivot_count,
                False)

    def _place_next_target(
            self, rng: np.random.Generator, obj_dict: Dict, x: float, y: float,
            rot: float, gap: float, domino_count: int, target_count: int,
            _in_bounds: Callable[[float, float], bool]) -> Tuple:
        """Place the next target in the sequence."""
        dy = gap * np.cos(rot)
        dx = gap * np.sin(rot)
        nx, ny = x + dx, y + dy
        if not _in_bounds(nx, ny):
            return (False, x, y, rot, domino_count, target_count)

        if CFG.domino_use_domino_blocks_as_target:
            # Place a pink domino as target
            obj_dict[self.dominos[domino_count]] = self._place_domino(
                domino_count, nx, ny, rot, is_target_block=True)
            domino_count += 1
        else:
            # Place a regular target
            obj_dict[self.targets[target_count]] = self._place_pivot_or_target(
                nx, ny, rot)

        return (True, nx, ny, rot, domino_count, target_count + 1)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            possible_num_dominos=CFG.domino_train_num_dominos,
            possible_num_targets=CFG.domino_train_num_targets,
            possible_num_pivots=CFG.domino_train_num_pivots,
            num_pos_x=CFG.domino_train_num_pos_x,
            num_pos_y=CFG.domino_train_num_pos_y,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            possible_num_dominos=CFG.domino_test_num_dominos,
            possible_num_targets=CFG.domino_test_num_targets,
            possible_num_pivots=CFG.domino_test_num_pivots,
            num_pos_x=CFG.domino_test_num_pos_x,
            num_pos_y=CFG.domino_test_num_pos_y,
            rng=self._test_rng)

    def _make_tasks(self,
                    num_tasks: int,
                    possible_num_dominos: List[int],
                    possible_num_targets: List[int],
                    possible_num_pivots: List[int],
                    num_pos_x: int,
                    num_pos_y: int,
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

            # Add position objects to initial state based on current grid
            if CFG.domino_use_grid:
                # Generate grid coordinates for this specific configuration
                x_coords, y_coords = self._generate_grid_coordinates(
                    num_pos_x, num_pos_y)
                self.grid_pos = [(x, y) for y in y_coords for x in x_coords]

                positions: List[Object] = []
                for i in range(num_pos_x * num_pos_y):
                    name = f"loc_y{i//num_pos_x}_x{i%num_pos_x}"
                    obj = Object(name, self._position_type)
                    positions.append(obj)

                # Create position dictionary for this task configuration
                pos_dict = {}
                pos_index = 0
                for i in range(num_pos_y):
                    for j in range(num_pos_x):
                        if pos_index < len(positions):
                            pos_obj = positions[pos_index]
                            pos_dict[pos_obj] = {
                                "xx": x_coords[j],
                                "yy": y_coords[i]
                            }
                            # Set sim features for position objects
                            pos_obj.xx = x_coords[j]
                            pos_obj.yy = y_coords[i]
                            pos_index += 1

                # Add position objects to initial state
                init_dict.update(pos_dict)

                # Add rotation objects to initial state
                for rotation_obj in self.rotations:
                    angle_str = rotation_obj.name.split("_")[1]
                    init_dict[rotation_obj] = {"angle": float(angle_str)}

            # Place dominoes (D) and targets (T) in order: D D T D T
            # at fixed positions along the x-axis
            n_dominos = rng.choice(possible_num_dominos)
            n_targets = rng.choice(possible_num_targets)
            n_pivots = rng.choice(possible_num_pivots)

            # Generate sequence using helper function
            obj_dict = None
            max_attempts = 1000
            for i in range(max_attempts):
                if log_debug:
                    print(f"\nAttempt {i} for task {i_task}")
                if CFG.domino_use_grid:
                    obj_dict = self._generate_domino_sequence_with_grid(
                        rng,
                        n_dominos,
                        n_targets,
                        num_pos_x,
                        num_pos_y,
                        log_debug=log_debug)
                else:
                    obj_dict = self._generate_domino_sequence(
                        rng,
                        n_dominos,
                        n_targets,
                        n_pivots,
                        log_debug=log_debug)
                if obj_dict is not None:
                    if log_debug:
                        print("Found satisfying a task")
                    break

            if obj_dict is None:
                raise RuntimeError("Failed to generate valid domino sequence")
            if log_debug:
                print(f"Found a task")

            # If we want to initialize at finished state, move intermediate objects
            if not CFG.domino_initialize_at_finished_state:
                obj_dict = self._move_intermediate_objects_to_finished_state(
                    obj_dict, num_pos_x, num_pos_y)

            init_dict.update(obj_dict)
            init_state = utils.create_state_from_dict(init_dict)

            # The goal: topple all targets
            if CFG.domino_use_domino_blocks_as_target:
                # Find target dominoes (pink dominoes) and set them as goals
                goal_atoms = set()
                for domino_obj in init_state.get_objects(self._domino_type):
                    # goal_atoms.add(GroundAtom(self._Toppled, [domino_obj]))
                    if self._TargetDomino_holds(init_state, [domino_obj]):
                        goal_atoms.add(GroundAtom(self._Toppled, [domino_obj]))
            else:
                # Use regular targets
                goal_atoms = {GroundAtom(self._Toppled, [self.targets[0]])}

            tasks.append(EnvironmentTask(init_state, goal_atoms))
            total_attempts += i + 1
        if log_debug:
            print(f"Total attempts: {total_attempts}")

        return self._add_pybullet_state_to_tasks(tasks)

    # A small helper to set up dictionary entries:
    def _place_domino(self,
                      d_idx: int,
                      x: float,
                      y: float,
                      rot: float,
                      is_start_block: bool = False,
                      is_target_block: bool = False) -> Dict:
        # Choose color based on block type
        if is_start_block:
            color = self.start_domino_color
        elif is_target_block:
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

    # Same for pivot or target (note pivot/target is on z_lb):
    def _place_pivot_or_target(self,
                               x: float,
                               y: float,
                               rot: float = 0.0) -> Dict:
        return {
            "x": x,
            "y": y,
            "z": self.z_lb,
            "yaw": rot,
        }

    def _move_intermediate_objects_to_finished_state(
            self,
            obj_dict: Dict,
            num_pos_x: int = None,
            num_pos_y: int = None) -> Dict:
        """Move all intermediate dominoes and pivots to the lower end of the
        table in a row, keeping only the start domino and targets in their
        original positions.

        When CFG.domino_use_grid=True, places intermediate objects on clear
        grid positions, preferably on the bottom side starting from the
        middle and extending to left and right.

        Args:
            domino_dict: Dictionary containing the original positions of all objects

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
                        domino_data.get("r", 0.0) -
                        self.target_domino_color[0]) < eps and abs(
                            domino_data.get("g", 0.0) -
                            self.target_domino_color[1]) < eps and abs(
                                domino_data.get("b", 0.0) -
                                self.target_domino_color[2]) < eps)

                # Only move dominoes that are neither start nor target dominoes
                if not is_start_domino and not is_target_domino:
                    intermediate_objects.append((domino, "domino"))

        # Find all pivots
        for pivot in self.pivots:
            if pivot in obj_dict:
                intermediate_objects.append((pivot, "pivot"))

        if not intermediate_objects:
            return obj_dict

        if CFG.domino_use_grid:
            # Use grid positioning when grid is enabled
            # First, identify which grid positions are already occupied
            occupied_positions = set()
            position_tolerance = self.pos_gap * 0.5

            # Extract just the objects from the intermediate_objects tuples for easier checking
            intermediate_obj_set = {
                obj
                for obj, obj_type in intermediate_objects
            }

            for obj, obj_data in obj_dict.items():
                if obj not in intermediate_obj_set:  # Skip objects we're about to move
                    obj_x = obj_data.get("x", 0.0)
                    obj_y = obj_data.get("y", 0.0)

                    # Check which grid position this object occupies
                    for grid_x, grid_y in self.grid_pos:
                        if (abs(obj_x - grid_x) <= position_tolerance
                                and abs(obj_y - grid_y) <= position_tolerance):
                            occupied_positions.add((grid_x, grid_y))
                            break

            # Find available positions on the bottom side, starting from middle
            # Sort grid positions by y coordinate (ascending) then by distance from x center
            if num_pos_x is not None and num_pos_y is not None:
                x_coords, y_coords = self._generate_grid_coordinates(
                    num_pos_x, num_pos_y)
            else:
                # Fallback to maximum grid size
                max_num_pos_x = max(CFG.domino_train_num_pos_x,
                                    CFG.domino_test_num_pos_x)
                max_num_pos_y = max(CFG.domino_train_num_pos_y,
                                    CFG.domino_test_num_pos_y)
                x_coords, y_coords = self._generate_grid_coordinates(
                    max_num_pos_x, max_num_pos_y)

            x_center = (x_coords[0] + x_coords[-1]) / 2 if x_coords else 0

            # Get bottom row positions first, then other rows if needed
            available_positions = []
            for y in sorted(y_coords):  # Start from bottom (smallest y)
                row_positions = [(x, y) for x in x_coords
                                 if (x, y) not in occupied_positions]
                # Sort by distance from center
                row_positions.sort(key=lambda pos: abs(pos[0] - x_center))
                available_positions.extend(row_positions)

            # Place intermediate objects on available grid positions
            for i, (obj, obj_type) in enumerate(intermediate_objects):
                if i < len(available_positions):
                    new_x, new_y = available_positions[i]
                else:
                    # Fallback to non-grid positioning if we run out of grid positions
                    start_x = self.x_lb + self.domino_width
                    spacing = self.domino_width * 1.5
                    new_x = start_x + i * spacing
                    new_y = (self.y_lb + self.y_ub) / 2

                if obj_type == "domino":
                    obj_dict[obj] = {
                        "x": new_x,
                        "y": new_y,
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
                        "y": new_y,
                        "z": self.z_lb,
                        "yaw": 0.0,  # Reset rotation
                    }
        else:
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
        position: Sequence[Pose3D] = (0, 0, 0),
        orientation: Sequence[Quaternion] = (0, 0, 0, 1),
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
        ccd_motion_threshold: Optional[
            float] = None,  # defaults to 0.5 * min(half_extents)
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
        thresh = ccd_motion_threshold if ccd_motion_threshold is not None else 0.5 * m
        p.changeDynamics(
            block_id,
            linkIndex=-1,
            ccdSweptSphereRadius=swept,
            # ccdMotionThreshold=thresh,
            physicsClientId=physics_client_id,
        )

    return block_id


if __name__ == "__main__":

    CFG.seed = 0
    CFG.env = "pybullet_domino"
    CFG.domino_initialize_at_finished_state = False
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_use_grid = True
    CFG.num_train_tasks = 1
    CFG.num_test_tasks = 2
    env = PyBulletDominoEnv(use_gui=True)
    # # Set up test configurations for the example
    # CFG.domino_test_num_dominos = [3]
    # CFG.domino_test_num_targets = [1]
    # CFG.domino_test_num_pivots = [1]

    tasks = env._generate_train_tasks()

    for task in tasks:
        env._reset_state(task.init)
        print(pformat(utils.abstract(task.init, env.predicates)), '\n')

        for i in range(10000):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
