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
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

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
        # Create 'dummy' Objects (they'll be assigned IDs on reset)
        self._robot = Object("robot", self._robot_type)
        # We'll hold references to all domino and target objects in lists
        # after we create them in tasks.
        self.dominos: List[Object] = []
        if CFG.domino_use_domino_blocks_as_target:
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
        self._InFrontDirection = Predicate(
            "InFrontDirection",
            [self._domino_type, self._domino_type, self._direction_type],
            self._InFrontDirection_holds)
        self._InFront = DerivedPredicate(
            "InFront", [self._domino_type, self._domino_type],
            self._InFront_holds)
        self._NotInFrontOfAny = Predicate("NotInFrontOfAny",
                                          [self._domino_type],
                                          self._NotInFrontOfAny_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Toppled,
            self._StartBlock,
            self._HandEmpty,
            self._Holding,
            self._InFrontDirection,
            self._InFront,
            self._NotInFrontOfAny,
            # self._Upright,
            # self._NotUpright
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # The goal is always to topple all targets
        return {self._Toppled}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._domino_type, self._target_type,
            self._pivot_type, self._direction_type
        }

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
    def _InFrontDirection_holds(cls, state: State,
                                objects: Sequence[Object]) -> bool:
        """Check if domino1 is in front of domino2 in the given direction.

        This predicate returns True if domino1 is positioned such that
        when domino2 falls in the specified direction, domino1 would
        fall afterwards.
        """
        domino1, domino2, direction = objects

        # Get positions and orientations
        x1, y1 = state.get(domino1, "x"), state.get(domino1, "y")
        x2, y2 = state.get(domino2, "x"), state.get(domino2, "y")
        rot2 = state.get(domino2, "rot")
        dir_value = state.get(direction, "dir")

        # Calculate expected front position based on domino2's orientation and direction
        gap = cls.domino_width * 1.3  # Same gap used in make_tasks

        if dir_value == 0.0:  # straight
            # Domino1 should be directly in front of domino2
            expected_x = x2 + gap * np.sin(rot2)
            expected_y = y2 + gap * np.cos(rot2)
        elif dir_value == 1.0:  # left
            # Domino1 should be to the left of domino2's fall direction
            turn_angle = rot2 - np.pi / 2  # 90 degrees to the left
            expected_x = x2 + gap * np.sin(turn_angle)
            expected_y = y2 + gap * np.cos(turn_angle)
        elif dir_value == 2.0:  # right
            # Domino1 should be to the right of domino2's fall direction
            turn_angle = rot2 + np.pi / 2  # 90 degrees to the right
            expected_x = x2 + gap * np.sin(turn_angle)
            expected_y = y2 + gap * np.cos(turn_angle)
        else:
            return False

        # Check if domino1 is close enough to the expected position
        position_tolerance = cls.domino_width * 0.5
        distance = np.sqrt((x1 - expected_x)**2 + (y1 - expected_y)**2)

        return distance <= position_tolerance

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

    # -------------------------------------------------------------------------
    # Task Generation

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

            while True:
                print("\nSample again:")
                obj_dict = {}
                domino_count = 0
                target_count = 0
                pivot_count = 0
                just_placed_target = False
                just_turned_90 = False
                success = True  # Track whether we placed everything successfully

                # Initial domino
                x = rng.uniform(self.x_lb, self.x_ub)
                y = rng.uniform(self.y_lb + self.domino_width,
                                self.y_ub - 3 * self.domino_width)
                # self.y_lb + self.domino_width * 2)
                rot = rng.uniform(-np.pi / 2, np.pi / 2)
                rot = rng.choice([0, np.pi / 2, -np.pi / 2])
                gap = self.domino_width * 1.3

                # Place first domino
                obj_dict[self.dominos[domino_count]] = self._place_domino(
                    domino_count, x, y, rot, is_start_block=True)
                domino_count += 1

                turn_choices = self.turn_choices.copy()
                if pivot_count == n_pivots:
                    turn_choices.remove("pivot180")

                # Try placing dominos/targets
                if CFG.domino_use_domino_blocks_as_target:
                    # When using domino blocks as targets, all objects are dominoes
                    expected_total_dominoes = n_dominos + n_targets
                    loop_condition = domino_count < expected_total_dominoes or target_count < n_targets
                else:
                    # Regular mode
                    loop_condition = domino_count < n_dominos or target_count < n_targets

                while loop_condition:
                    can_place_target = (domino_count >= 2
                                        and target_count < n_targets
                                        and not just_placed_target)
                    must_place_domino = not can_place_target

                    if CFG.domino_use_domino_blocks_as_target:
                        expected_total_dominoes = n_dominos + n_targets
                        can_place_domino = domino_count + n_targets <\
                            expected_total_dominoes
                        # can_place_domino = domino_count < expected_total_dominoes
                    else:
                        can_place_domino = domino_count < n_dominos

                    if (must_place_domino or rng.random() > 0.5) and \
                        can_place_domino:
                        # If just placed a target, enforce a "straight" choice
                        choices = turn_choices.copy()
                        if just_turned_90:
                            choices.remove("turn90")
                        if just_placed_target:
                            choices = ["straight"]
                        choice = rng.choice(choices)
                        print(f"Choice: {choice}")

                        if choice == "straight":
                            dy = gap * np.cos(rot)
                            dx = gap * np.sin(rot)
                            nx, ny = x + dx, y + dy
                            if not _in_bounds(nx, ny):
                                success = False
                                break
                            x, y = nx, ny
                            obj_dict[self.dominos[
                                    domino_count]] = self._place_domino(
                                        domino_count,
                                        x,
                                        y,
                                        rot,
                                        is_start_block=False)
                            domino_count += 1
                            just_turned_90 = False

                        elif choice == "turn90":
                            # Check we have enough dominos left
                            if domino_count + 1 >= n_dominos:
                                # Fallback to straight
                                dy = gap * np.cos(rot)
                                dx = gap * np.sin(rot)
                                nx, ny = x + dx, y + dy
                                if not _in_bounds(nx, ny):
                                    success = False
                                    break
                                x, y = nx, ny
                                obj_dict[self.dominos[
                                    domino_count]] = self._place_domino(
                                        domino_count,
                                        x,
                                        y,
                                        rot,
                                        is_start_block=False)
                                domino_count += 1
                            else:
                                # Turn 45° twice
                                turn_dir = rng.choice([-1, 1])
                                half_turn = np.pi / 4 * turn_dir

                                # First 45°
                                side_offset = 0  #(self.domino_width / 4)
                                rot += half_turn
                                dx = gap * np.sin(rot)
                                dy = gap * np.cos(rot)

                                dx -= turn_dir * side_offset * np.cos(rot)
                                dy += turn_dir * side_offset * np.sin(rot)
                                nx, ny = x + dx, y + dy
                                if not _in_bounds(nx, ny):
                                    success = False
                                    break
                                x, y = nx, ny
                                obj_dict[self.dominos[
                                    domino_count]] = self._place_domino(
                                        domino_count,
                                        x,
                                        y,
                                        rot + np.pi / 2,
                                        is_start_block=False)
                                domino_count += 1

                                # Second 45°
                                side_offset = (self.domino_width / 2)
                                rot += half_turn
                                dx = gap * np.sin(rot)
                                dy = gap * np.cos(rot)

                                dx -= turn_dir * side_offset * np.cos(rot)
                                dy += turn_dir * side_offset * np.sin(rot)
                                nx, ny = x + dx, y + dy
                                if not _in_bounds(nx, ny):
                                    success = False
                                    break
                                x, y = nx, ny
                                obj_dict[self.dominos[
                                    domino_count]] = self._place_domino(
                                        domino_count,
                                        x,
                                        y,
                                        rot,
                                        is_start_block=False)
                                domino_count += 1
                            just_turned_90 = True

                        elif choice == "pivot180" and pivot_count < n_pivots:

                            pivot_dir = rng.choice(
                                [-1, 1])  # pick left or right offset
                            side_offset = (self.pivot_width / 2)

                            # Parallel movement along orientation rot:
                            #   (cos(rot), sin(rot)) is the unit vector in direction 'rot'
                            pivot_x = x + gap * (2 / 3) * np.sin(rot)  # 0.03
                            pivot_y = y + gap * (2 / 3) * np.cos(rot)

                            # Optional sideways shift:
                            pivot_x -= pivot_dir * side_offset * np.cos(
                                rot)  # 0
                            pivot_y -= pivot_dir * side_offset * np.sin(
                                rot)  # -0.1

                            if not _in_bounds(pivot_x, pivot_y):
                                success = False
                                break

                            obj_dict[self.pivots[
                                pivot_count]] = self._place_pivot_or_target(
                                    pivot_x, pivot_y, rot)
                            pivot_count += 1

                            # Flip orientation
                            # big +y, small -x
                            back_x = pivot_x - (gap * (2 / 3)) * np.sin(rot)
                            back_y = pivot_y - (gap * (2 / 3)) * np.cos(rot)

                            # Optionally keep the same sideways offset so
                            # it's "same side"
                            back_x -= pivot_dir * side_offset * np.cos(rot)
                            back_y += pivot_dir * side_offset * -np.sin(rot)
                            if not _in_bounds(back_x, back_y):
                                success = False
                                break
                            x, y = back_x, back_y
                            rot += np.pi  # 180° flip

                            # Place next domino at this new position
                            obj_dict[self.dominos[
                                domino_count]] = self._place_domino(
                                    domino_count,
                                    x,
                                    y,
                                    rot,
                                    is_start_block=False)
                            domino_count += 1
                            just_turned_90 = False
                        else:
                            # fallback
                            dy = gap * np.cos(rot)
                            dx = gap * np.sin(rot)
                            nx, ny = x + dx, y + dy
                            if not _in_bounds(nx, ny):
                                success = False
                                break
                            x, y = nx, ny
                            obj_dict[self.dominos[
                                domino_count]] = self._place_domino(
                                    domino_count,
                                    x,
                                    y,
                                    rot,
                                    is_start_block=False)
                            domino_count += 1
                            just_turned_90 = False
                        just_placed_target = False

                    else:
                        print("Placing target")
                        # Place a target
                        dy = gap * np.cos(rot)
                        dx = gap * np.sin(rot)
                        nx, ny = x + dx, y + dy
                        if not _in_bounds(nx, ny):
                            success = False
                            break
                        x, y = nx, ny

                        if CFG.domino_use_domino_blocks_as_target:
                            # Place a pink domino as target
                            obj_dict[self.dominos[
                                domino_count]] = self._place_domino(
                                    domino_count,
                                    x,
                                    y,
                                    rot,
                                    is_target_block=True)
                            domino_count += 1
                        else:
                            # Place a regular target
                            obj_dict[self.targets[
                                target_count]] = self._place_pivot_or_target(
                                    x, y, rot)
                        target_count += 1
                        just_placed_target = True
                        just_turned_90 = False

                    # Update loop condition
                    if CFG.domino_use_domino_blocks_as_target:
                        expected_total_dominoes = n_dominos + n_targets
                        loop_condition = domino_count < expected_total_dominoes or target_count < n_targets
                    else:
                        loop_condition = domino_count < n_dominos or target_count < n_targets

                    if not success:
                        break

                if CFG.domino_use_domino_blocks_as_target:
                    # When using domino blocks as targets, domino_count includes both regular and target dominoes
                    expected_domino_count = n_dominos + n_targets
                    if (success and domino_count == expected_domino_count
                            and target_count == n_targets
                            and pivot_count == n_pivots):
                        print("Found satisfying a task")
                        break
                else:
                    # Regular mode: separate domino and target counts
                    if (success and domino_count == n_dominos
                            and target_count == n_targets
                            and pivot_count == n_pivots):
                        print("Found satisfying a task")
                        break

                # Retry if we didn't find a satisfying task
                continue
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

    CFG.seed = 1
    CFG.env = "pybullet_domino"
    env = PyBulletDominoEnv(use_gui=True)
    tasks = env._make_tasks(10, env._train_rng)
    for task in tasks:
        env._reset_state(task.init)

        for i in range(100):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.01)
