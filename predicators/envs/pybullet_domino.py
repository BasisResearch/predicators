"""Example usage:

python predicators/main.py --approach oracle --env pybullet_domino \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug \
--sesame_check_expected_atoms False --horizon 60 \
--video_not_break_on_exception --pybullet_ik_validate False
"""
import logging
from re import A
import time
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
    z_ub: ClassVar[float] = 0.75 + table_height / 2

    # Domino shape
    domino_width: ClassVar[float] = 0.07
    domino_depth: ClassVar[float] = 0.02
    domino_height: ClassVar[float] = 0.15
    turn_shift_frac: ClassVar[float] = 0.3
    # domino_mass: ClassVar[float] = 0.3
    domino_mass: ClassVar[float] = 0.8
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (1.0, 0.75, 0.8, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)
    start_domino_x: ClassVar[float] = x_lb + domino_width
    start_domino_y: ClassVar[float] = y_lb + domino_width

    target_height: ClassVar[float] = 0.2
    pivot_width: ClassVar[float] = 0.2

    # For deciding if a target is toppled: if absolute tilt in x or y
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

    num_dominos_max: ClassVar[int] = min(9, 2)
    num_dominos_min: ClassVar[int] = 2
    num_targets_max: ClassVar[int] = min(3, 1)
    num_targets_min: ClassVar[int] = 1
    num_pivots_max: ClassVar[int] = min(2, 0)
    num_pivots_min: ClassVar[int] = 0
    turn_choices: ClassVar[List[str]] = ["straight", "turn90", "pivot180"]

    # Grid configuration
    # num_pos_x: ClassVar[int] = 9
    # num_pos_y: ClassVar[int] = 5
    num_pos_x: ClassVar[int] = 2
    num_pos_y: ClassVar[int] = 2
    pos_gap: ClassVar[
        float] = domino_width * 1.3  # Distance between grid positions

    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _domino_type = Type(
        "domino",
        ["x", "y", "z", "rot", "tilt", "r", "g", "b", "is_held"],
    )
    _target_type = Type("target", ["x", "y", "z", "rot"],
                        sim_features=["id", "joint_id"])
    _pivot_type = Type("pivot", ["x", "y", "z", "rot"],
                       sim_features=["id", "joint_id"])
    _direction_type = Type("direction", ["dir"])

    def __init__(self, use_gui: bool = True) -> None:
        # Conditionally create grid-related types
        if CFG.domino_use_grid:
            self._position_type = Type("loc", ["xx", "yy"])
            self._rotation_type = Type("rot", ["angle"])

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
            self.positions: List[Object] = []
            x_coords, y_coords = self._generate_grid_coordinates()
            self.grid_pos = [(x, y) for y in y_coords for x in x_coords]
            self.pos_dict = dict()
            for i, (x, y) in enumerate(self.grid_pos):
                name = f"pos_y{i//self.num_pos_y}_x{i%self.num_pos_x}"
                obj = Object(name, self._position_type)
                self.positions.append(obj)
                self.pos_dict[obj] = (x, y)

            # Create rotation objects for 8 discrete angles
            self.rotations: List[Object] = []
            angle_values = [-135, -90, -45, 0, 45, 90, 135, 180]  # degrees
            for angle in angle_values:
                name = f"rot_{angle}"
                obj = Object(name, self._rotation_type)
                self.rotations.append(obj)
        else:
            # Initialize empty lists when grid is not used
            self.positions: List[Object] = []
            self.rotations: List[Object] = []
            self.pos_dict = dict()
            self.grid_pos = []

        self.block_constraints = []

        super().__init__(use_gui)

        # Define Predicates
        if CFG.domino_use_domino_blocks_as_target:
            self._Toppled = Predicate("Toppled", [self._domino_type],
                                      self._Toppled_holds)
        else:
            self._Toppled = Predicate("Toppled", [self._target_type],
                                      self._Toppled_holds)
        self._StartBlock = Predicate("StartBlock", [self._domino_type],
                                     self._StartBlock_holds)
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
                "DominoAtRot", [self._domino_type, self._rotation_type],
                self._DominoAtRot_holds)
            
            self._InFrontDirection = DerivedPredicate(
                "InFrontDirection", [self._domino_type, self._domino_type, 
                self._direction_type],
                self._InFrontDirection_holds, 
                auxiliary_predicates=[self._DominoAtPos, self._DominoAtRot] if 
                CFG.domino_use_grid else [])
            self._InFront = DerivedPredicate(
                "InFront", [self._domino_type, self._domino_type],
                self._InFront_holds, auxiliary_predicates=[self._InFrontDirection])
            self._NotInFrontOfAny = DerivedPredicate("NotInFrontOfAny",
                                            [self._domino_type],
                                            self._NotInFrontOfAny_holds,
                                auxiliary_predicates=[self._InFrontDirection])


    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino"

    @property
    def predicates(self) -> Set[Predicate]:
        base_predicates = {
            self._Toppled,
            self._StartBlock,
            self._HandEmpty,
            self._Holding,
            self._InFrontDirection,
            self._InFront,
            # self._NotInFrontOfAny,
            # self._Upright,
            # self._NotUpright
        }

        if CFG.domino_use_grid:
            base_predicates.update({
                self._DominoAtPos,
                self._DominoAtRot,
            })

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
            base_types.update({self._position_type, self._rotation_type})

        return base_types

    # -------------------------------------------------------------------------
    # Grid Coordinate Generation

    @classmethod
    def _generate_grid_coordinates(cls) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates for position objects."""
        # Calculate grid extents based on workspace bounds
        total_x_range = cls.x_ub - cls.x_lb
        total_y_range = cls.y_ub - cls.y_lb

        # Center the grid within the workspace
        x_start = cls.x_lb + (total_x_range -
                              (cls.num_pos_x - 1) * cls.pos_gap) / 2
        y_start = cls.y_lb + (total_y_range -
                              (cls.num_pos_y - 1) * cls.pos_gap) / 2

        x_coords = [
            round(x_start + i * cls.pos_gap, 5) for i in range(cls.num_pos_x)
        ]
        y_coords = [
            round(y_start + i * cls.pos_gap, 5) for i in range(cls.num_pos_y)
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

        # Create a fixed number of dominoes and targets here
        domino_ids = []
        target_ids = []
        if CFG.domino_use_domino_blocks_as_target:
            # If using domino blocks as targets, we create more dominoes
            num_dominos_to_create = cls.num_dominos_max + cls.num_targets_max
            num_targets_to_create = 0
        else:
            num_dominos_to_create = cls.num_dominos_max
            num_targets_to_create = cls.num_targets_max
        for i in range(num_dominos_to_create):  # e.g. 3 dominoes
            domino_id = create_pybullet_block(
                color=cls.start_domino_color if i == 0 else cls.domino_color,
                half_extents=(cls.domino_width / 2, cls.domino_depth / 2,
                              cls.domino_height / 2),
                mass=cls.domino_mass,
                friction=0.5,
                orientation=[0.0, 0.0, 0.0],
                physics_client_id=physics_client_id,
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
        for _ in range(cls.num_pivots_max):
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

        # Draw debug lines at grid cell centers if grid is enabled
        if CFG.domino_use_grid:
            for pos_obj in self.positions:
                x, y = self.pos_dict[pos_obj]
                p.addUserDebugLine(
                    [x, y, self.table_height],
                    [x, y, self.table_height + self.debug_line_height],
                    [1, 0, 0],
                    parentObjectUniqueId=-1,
                    parentLinkIndex=-1)

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
        if obj.type == self._domino_type:
            if feature == "tilt":
                (_, _, _), orn = p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)

                # Convert quaternion to Euler angles
                roll, _, _ = p.getEulerFromQuaternion(orn)

                # The tilt w.r.t. the domino width axis is the roll angle
                # (rotation around the x-axis in the domino's local frame)
                return roll
        elif obj.type == self._direction_type:
            if feature == "dir":
                if obj.name == "straight":
                    return 0.0
                elif obj.name == "left":
                    return 1.0
                elif obj.name == "right":
                    return 2.0
        elif CFG.domino_use_grid and obj.type == self._position_type:
            if feature == "xx":
                return self.pos_dict[obj][0]
            elif feature == "yy":
                return self.pos_dict[obj][1]
        elif CFG.domino_use_grid and obj.type == self._rotation_type:
            if feature == "angle":
                # Extract angle from object name (e.g., "rot_45" -> 45.0)
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
                rot1 = state.get(domino1, "rot")
                rot2 = state.get(domino2, "rot")

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

        For domino targets, we use the tilt feature. For regular
        targets, we use rotation threshold.
        """
        obj, = objects

        if CFG.domino_use_domino_blocks_as_target:
            # For domino targets, check tilt angle
            tilt_angle = state.get(obj, "tilt")
            # Use the same threshold as NotUpright but inverted logic
            tilt_threshold = 0.1  # radians
            return abs(tilt_angle) >= tilt_threshold
        else:
            # For regular targets, use rotation-based check (currently disabled)
            rot_z = state.get(obj, "rot")
            if abs(utils.wrap_angle(rot_z)) < 0.8:
                return True
            return False

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

    @classmethod
    def _HandEmpty_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    @classmethod
    def _Holding_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        _, domino = objects
        return state.get(domino, "is_held") > 0.5

    @classmethod
    def _check_single_direction(cls, x1_idx: int, y1_idx: int, x2_idx: int, y2_idx: int,
                                rot1_rad: float, rot2_rad: float, dir_value: str, 
                                tolerance: float = 1e-6) -> bool:
        """Helper function to check if domino1 is in front of domino2 with the given direction.
        
        This checks a single directional relationship: domino1 must be in the cell in 
        front of domino2, and the rotation difference must match the expected direction.
        
        Args:
            x1_idx, y1_idx: Grid coordinates of domino1
            x2_idx, y2_idx: Grid coordinates of domino2  
            rot1_rad, rot2_rad: Rotations of domino1 and domino2 in radians
            dir_value: Direction string ("left", "straight", "right")
            tolerance: Numerical tolerance for comparisons
            
        Returns:
            True if domino1 is in front of domino2 with the correct rotation difference
        """
        # Check if domino1 is in the cell in front of domino2
        # This check is only valid if domino2 has a cardinal rotation (0, 90, 180, etc.)
        # We can check this by seeing if sin or cos of the angle is close to 0.
        if not (abs(np.sin(rot2_rad)) < tolerance or abs(np.cos(rot2_rad)) < tolerance):
            return False

        dx2_idx = round(np.sin(rot2_rad))
        dy2_idx = round(np.cos(rot2_rad))
        expected_x1 = x2_idx + dx2_idx
        expected_y1 = y2_idx + dy2_idx
        
        # domino1 must be at the expected position
        if not (x1_idx == expected_x1 and y1_idx == expected_y1):
            return False
            
        # Calculate rotation difference (domino1 - domino2)
        rot_diff = rot1_rad - rot2_rad
        
        # Normalize rotation difference to [-π, π] range
        while rot_diff > np.pi:
            rot_diff -= 2*np.pi
        while rot_diff < -np.pi:
            rot_diff += 2*np.pi
            
        # Define expected rotation difference based on direction
        # FIX: Swapped expected rotation for "left" and "right"
        if dir_value == "left":
            expected_rot_diff = np.pi/4
        elif dir_value == "straight":
            expected_rot_diff = 0
        elif dir_value == "right":
            expected_rot_diff = -np.pi/4
        else:
            return False
            
        # Check if rotation difference matches expected value
        return abs(rot_diff - expected_rot_diff) < tolerance

    @classmethod
    def _InFrontDirection_holds(cls, atoms: Set[GroundAtom],
                                objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in the given direction.

        This predicate is symmetric and checks two cases:
        1. Original: domino1 is in front of domino2 with the given direction
        2. Swapped: domino2 is in front of domino1 with the opposite direction
        
        For example, InFrontDirection(d1, d2, "right") returns True if either:
        - d1 is in the cell in front of d2 with rotation difference of -π/4, OR
        - d2 is in the cell in front of d1 with rotation difference of π/4
          (equivalent to InFrontDirection(d2, d1, "left"))
        
        This symmetry ensures that both InFrontDirection(d1, d2, "left") and 
        InFrontDirection(d2, d1, "right") can be true simultaneously.
        """
        domino1, domino2, direction = objects

        if not CFG.domino_use_grid:
            raise ValueError("Grid is not used, this derived predicate cannot "
                             "function")

        # Find positions and rotations using auxiliary predicates
        domino1_pos = None
        domino1_rot = None
        domino2_pos = None
        domino2_rot = None

        # Extract positions and rotations from atoms
        for atom in atoms:
            try:
                if atom.predicate.name == "DominoAtPos":
                    if atom.objects[0] == domino1:
                        domino1_pos = atom.objects[1]
                    elif atom.objects[0] == domino2:
                        domino2_pos = atom.objects[1]
                elif atom.predicate.name == "DominoAtRot":
                    if atom.objects[0] == domino1:
                        domino1_rot = atom.objects[1]
                    elif atom.objects[0] == domino2:
                        domino2_rot = atom.objects[1]
            except:
                breakpoint()

        # All required information must be available
        if not all([domino1_pos, domino1_rot, domino2_pos, domino2_rot]):
            return False

        # Extract coordinates from position object names (e.g., "pos_y1_x2")
        def extract_grid_coords(pos_obj):
            # Position names follow pattern "pos_y{y_idx}_x{x_idx}"
            name_parts = pos_obj.name.split("_")
            y_idx = int(name_parts[1][1:])  # Remove 'y' prefix
            x_idx = int(name_parts[2][1:])  # Remove 'x' prefix
            return x_idx, y_idx

        # Extract rotation from rotation object names (e.g., "rot_45")
        def extract_rotation_angle(rot_obj):
            # Rotation names follow pattern "rot_{angle}"
            angle_str = rot_obj.name.split("_")[1]
            return float(angle_str)

        x1_idx, y1_idx = extract_grid_coords(domino1_pos)
        x2_idx, y2_idx = extract_grid_coords(domino2_pos)
        rot1_angle = extract_rotation_angle(domino1_rot)
        rot2_angle = extract_rotation_angle(domino2_rot)

        # Convert angles to radians
        rot1_rad = utils.wrap_angle(np.radians(rot1_angle))
        rot2_rad = utils.wrap_angle(np.radians(rot2_angle))
        
        # Get direction value
        dir_value = direction.name
        
        # Determine the opposite direction for the swapped case
        if dir_value == "left":
            opposite_dir = "right"
        elif dir_value == "right":
            opposite_dir = "left"
        else:  # "straight"
            opposite_dir = "straight"
        
        # FIX: The original implementation incorrectly used an if/else
        # that only checked one of the two cases. The correct implementation
        # checks both and returns True if either one holds.

        # Case 1: Original - domino1 is in front of domino2 with the given direction
        case1 = cls._check_single_direction(
            x1_idx, y1_idx, x2_idx, y2_idx,
            rot1_rad, rot2_rad, dir_value
        )
        
        # Case 2: Swapped - domino2 is in front of domino1 with the opposite direction
        case2 = cls._check_single_direction(
            x2_idx, y2_idx, x1_idx, y1_idx,
            rot2_rad, rot1_rad, opposite_dir
        )
        
        # Return True if either case holds
        return case1 or case2

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
    def _NotInFrontOfAny_holds(cls, state: State,
                               objects: Sequence[Object]) -> bool:
        """Check if domino1 is not in front of any other domino in any
        direction.

        This predicate returns True if domino1 is not positioned in
        front of any other domino in any direction (straight, left, or
        right).
        """
        domino1, = objects

        # Get all dominos and direction objects in the state
        dominos = state.get_objects(cls._domino_type)
        directions = state.get_objects(cls._direction_type)

        # Check if domino1 is in front of any other domino in any direction
        for domino2 in dominos:
            if domino1 == domino2:
                continue  # Skip self-comparison

            # Check all directions
            for direction in directions:
                if cls._InFrontDirection_holds(state,
                                               [domino1, domino2, direction]):
                    return False  # domino1 is in front of domino2 in this direction

        return True  # domino1 is not in front of any other domino

    @classmethod
    def _DominoAtPos_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific position."""
        domino, position = objects

        # Get domino's actual position
        domino_x = state.get(domino, "x")
        domino_y = state.get(domino, "y")

        # Get the target position
        target_x = state.get(position, "xx")
        target_y = state.get(position, "yy")

        # Check if domino is close enough to the target position
        position_tolerance = cls.pos_gap * 0.5
        return (abs(domino_x - target_x) <= position_tolerance and
                abs(domino_y - target_y) <= position_tolerance)

    @classmethod
    def _DominoAtRot_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is at a specific rotation."""
        domino, rotation = objects

        # Get domino's actual rotation (in radians)
        domino_rot = state.get(domino, "rot")

        # Get the target rotation (convert from degrees to radians)
        target_rot_degrees = state.get(rotation, "angle")
        target_rot_radians = np.radians(target_rot_degrees)

        # Check if domino rotation is close enough to target rotation
        rotation_tolerance = np.radians(15)  # 15 degrees tolerance
        angle_diff = abs(utils.wrap_angle(domino_rot - target_rot_radians))

        return angle_diff <= rotation_tolerance

    # -------------------------------------------------------------------------
    # Task Generation

    def _generate_domino_sequence(self, rng: np.random.Generator,
                                  n_dominos: int, n_targets: int,
                                  n_pivots: int) -> Optional[Dict]:
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
        gap = self.domino_width * 1.3

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

    def _generate_domino_sequence_with_grid(self, rng: np.random.Generator,
                                            n_dominos: int,
                                            n_targets: int) -> Optional[Dict]:
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

        # Use a set for efficient checking of valid grid coordinates.
        grid_coords_set = set(self.grid_pos)

        # Choose a random starting position and orientation (cardinal directions).
        start_idx = rng.choice(len(self.grid_pos))
        curr_x, curr_y = self.grid_pos[start_idx]
        curr_rot = rng.choice([0, np.pi / 2, np.pi, -np.pi / 2])
        used_coords.add((curr_x, curr_y))

        # Place the first domino (start block).
        obj_dict[self.dominos[domino_count]] = self._place_domino(
            domino_count, curr_x, curr_y, curr_rot, is_start_block=True)
        domino_count += 1

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
                    d1_x, d1_y = next_x, next_y
                    if (d1_x, d1_y) not in grid_coords_set or \
                    (d1_x, d1_y) in used_coords:
                        continue

                    # Its orientation is 45 degrees towards the turn direction.
                    d1_rot = curr_rot - turn_dir * np.pi / 4

                    # Calculate the components of a diagonal shift vector.
                    # This vector pulls the turning domino both inward (perpendicular to its original path)
                    # and backward (parallel to its original path) to create a tighter, more stable corner.
                    # It's constructed by summing two unit vectors:
                    #
                    # 1. Perpendicular component (pulls sideways into the turn):
                    #    (turn_dir * np.cos(curr_rot), -turn_dir * np.sin(curr_rot))
                    #
                    # 2. Parallel component (pulls backward along the path):
                    #    (-np.sin(curr_rot), -np.cos(curr_rot))
                    #
                    # The terms are combined below. Note that adding these two unit vectors results in a
                    # direction vector with a magnitude of sqrt(2). The final shift distance will thus be
                    # `shift_magnitude * sqrt(2)`. To get a final distance of exactly `shift_magnitude`,
                    # you would need to normalize this vector by dividing each component by sqrt(2).
                    shift_magnitude = self.domino_width * self.turn_shift_frac
                    # The shift vector is perpendicular to the original direction of movement.
                    shift_dx = shift_magnitude * \
                        (turn_dir * np.cos(curr_rot) - np.sin(curr_rot))
                    shift_dy = shift_magnitude * \
                        (-turn_dir * np.sin(curr_rot) - np.cos(curr_rot))

                    # The final position is the grid position plus the shift.
                    d1_x += shift_dx
                    d1_y += shift_dy

                    # The second domino (d2) is a step from d1's GRID position
                    # in the new, fully turned direction.
                    # 1. Calculate d2's orientation relative to d1's orientation.
                    # A full 90-degree turn (pi/2) from the original orientation (curr_rot)
                    # is equivalent to a 135-degree turn (3*pi/4) from d1's 45-degree angle.
                    d2_rot = d1_rot - turn_dir * 1 * np.pi / 4

                    # 2. Calculate d2's position relative to d1's final, shifted position.
                    # This displacement vector is derived by expressing the original logic
                    # (V_displacement = V_step2 - V_shift) in terms of d1's rotation.
                    gap = self.pos_gap
                    sin_d1 = np.sin(d1_rot)
                    cos_d1 = np.cos(d1_rot)

                    # Components of the displacement vector from (d1_x, d1_y) to (d2_x, d2_y)
                    disp_x = (gap * turn_dir * cos_d1 + (2 * shift_magnitude - 
                                gap) * sin_d1) / np.sqrt(2)
                    disp_y = (-gap * turn_dir * sin_d1 + (2 * shift_magnitude - 
                                gap) * cos_d1) / np.sqrt(2)

                    d2_x = round(d1_x + disp_x, 5)
                    d2_y = round(d1_y + disp_y, 5)

                    if (d2_x, d2_y) in grid_coords_set and \
                    (d2_x, d2_y) not in used_coords:
                        # Use the shifted position for d1 in the final placements.
                        placements = [(d1_x, d1_y, d1_rot),
                                      (d2_x, d2_y, d2_rot)]
                        possible_moves.append(
                            (name, d2_x, d2_y, d2_rot, placements))

            if not possible_moves:
                # No valid moves, generation failed for this attempt.
                return None

            # Choose a random valid move and get its placement plan.
            _move_name, final_x, final_y, final_rot, placements = \
                possible_moves[rng.choice(len(possible_moves))]
            print(f"Chose move: {_move_name}")

            # Execute the placement plan for the chosen move.
            for (x, y, rot) in placements:
                print(f"Placing domino at {x}, {y}, {rot}")
                if domino_count >= total_domino_blocks:
                    break  # Should not be reached with correct logic.

                # Decide if this domino block should be a target.
                is_target = False
                if CFG.domino_use_domino_blocks_as_target and \
                target_count < n_targets:
                    remaining_blocks = total_domino_blocks - domino_count
                    remaining_targets = n_targets - target_count
                    # Force target if we must, otherwise decide randomly.
                    if remaining_targets >= remaining_blocks or \
                    rng.random() < remaining_targets / remaining_blocks:
                        is_target = True

                # Place the domino block.
                obj_dict[self.dominos[domino_count]] = self._place_domino(
                    domino_count, x, y, rot, is_target_block=is_target)

                # Use the grid coordinates for tracking used spots.
                used_coords.add((round(x, 5), round(y, 5)))
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
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        # Suppose we want to create M = 3 dominoes, N = 2 targets for each task

        for _ in range(num_tasks):
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

            # Add position objects to initial state
            if CFG.domino_use_grid:
                for position_obj in self.positions:
                    x, y = self.pos_dict[position_obj]
                    init_dict[position_obj] = {"xx": x, "yy": y}

                # Add rotation objects to initial state
                for rotation_obj in self.rotations:
                    angle_str = rotation_obj.name.split("_")[1]
                    init_dict[rotation_obj] = {"angle": float(angle_str)}

            # Place dominoes (D) and targets (T) in order: D D T D T
            # at fixed positions along the x-axis
            gap = self.domino_width * 1.3

            def _in_bounds(nx: float, ny: float) -> bool:
                """Check if (nx, ny) is within table boundaries."""
                return self.x_lb < nx < self.x_ub and \
                    self.y_lb < ny < self.y_ub

            n_dominos = rng.integers(low=self.num_dominos_min,
                                     high=self.num_dominos_max + 1)
            # n_dominos = len(self.dominos)
            n_targets = rng.integers(low=self.num_targets_min,
                                     high=self.num_targets_max + 1)
            n_pivots = rng.integers(low=self.num_pivots_min,
                                    high=self.num_pivots_max + 1)

            # Generate sequence using helper function
            obj_dict = None
            max_attempts = 1000
            for _ in range(max_attempts):
                print("\nSample again:")
                if CFG.domino_use_grid:
                    obj_dict = self._generate_domino_sequence_with_grid(
                        rng, n_dominos, n_targets)
                else:
                    obj_dict = self._generate_domino_sequence(
                        rng, n_dominos, n_targets, n_pivots)
                if obj_dict is not None:
                    print("Found satisfying a task")
                    break

            if obj_dict is None:
                raise RuntimeError("Failed to generate valid domino sequence")
            print(f"Found a task")

            # If we want to initialize at finished state, move intermediate objects
            if not CFG.domino_initialize_at_finished_state:
                obj_dict = self._move_intermediate_objects_to_finished_state(
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
                # Fallback to first target domino if none found
                if not goal_atoms:
                    target_dominos = [
                        d for d in init_state.get_objects(self._domino_type)
                        if self._TargetDomino_holds(init_state, [d])
                    ]
                    if target_dominos:
                        goal_atoms = {
                            GroundAtom(self._Toppled, [target_dominos[0]])
                        }
            else:
                # Use regular targets
                goal_atoms = {GroundAtom(self._Toppled, [self.targets[0]])}

            tasks.append(EnvironmentTask(init_state, goal_atoms))

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
            "rot": rot,
            "tilt": 0.0,  # All dominos start upright
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
            "rot": rot,
        }

    def _move_intermediate_objects_to_finished_state(self,
                                                     obj_dict: Dict) -> Dict:
        """Move all intermediate dominoes and pivots to the lower end of the
        table in a row, keeping only the start domino and targets in their
        original positions.

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
                    "rot": 0.0,  # Reset rotation to upright
                    "tilt": 0.0,  # Reset tilt to upright
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
                    "rot": 0.0,  # Reset rotation
                }

        return obj_dict


if __name__ == "__main__":

    CFG.seed = 0
    CFG.env = "pybullet_domino"
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_use_grid = True
    env = PyBulletDominoEnv(use_gui=True)
    tasks = env._make_tasks(10, env._train_rng)
    for task in tasks:
        env._reset_state(task.init)

        for i in range(100):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
