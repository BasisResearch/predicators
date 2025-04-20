"""A PyBullet version of Cover."""

import random
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple


import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG

from predicators.structs import Action, Array, EnvironmentTask, Object, \
    Predicate, State, Type
from predicators.utils import BoundingBox, NSPredicate, RawState, VLMQuery



class PyBulletCoverEnv(PyBulletEnv, CoverEnv):
    """PyBullet Cover domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # Object parameters.
    _obj_len_hgt: ClassVar[float] = 0.045
    # _max_obj_width: ClassVar[float] = 0.07  # highest width normalized to this
    _max_obj_width: ClassVar[float] = 0.06  # highest width normalized to this

    # Dimension and workspace parameters.
    _table_height: ClassVar[float] = 0.2
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    _offset: ClassVar[float] = 0.01
    pickplace_z: ClassVar[float] = _table_height + _obj_len_hgt * 0.5 + _offset
    _target_height: ClassVar[float] = 0.0001
    _obj_id_to_obj: Dict[int, Object] = {}

    def _Holding_NSP_holds(self, state: RawState, objects: Sequence[Object]) ->\
            bool:
        """Is the robot holding the block."""
        block, = objects

        # The block can't be held if the robot's hand is open.
        # We know there is only one robot in this environment.
        robot = state.get_objects(self._robot_type)[0]
        if self._GripperOpen_NSP_holds(state, [robot]):
            return False

        # Using simple heuristics to check if they have overlap
        block_bbox = state.get_obj_bbox(block)
        robot_bbox = state.get_obj_bbox(robot)
        if block_bbox.right < robot_bbox.left or \
            block_bbox.left > robot_bbox.right or\
            block_bbox.upper < robot_bbox.lower or\
            block_bbox.lower > robot_bbox.upper:
            return False

        block_name = block.id_name
        attention_image = state.crop_to_objects([block, robot])

        if CFG.save_nsp_image_patch_before_query:
            attention_image.save(f"{CFG.image_dir}/holding({block_name}).png")

        return state.evaluate_simple_assertion(
            f"{block_name} is held by the robot", attention_image)

    def _GripperOpen_NSP_holds(self, state: RawState, objects: Sequence[Object]) ->\
            bool:
        """Is the robots gripper open."""
        robot, = objects
        finger_state = state.get(robot, "fingers")
        assert finger_state in (0.0, 1.0)
        return finger_state == 1.0

    def _Covers_NSP_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        """Determine if the block is covering (directly on top of) the target
        region."""
        block, target = objects
        # Necessary but not sufficient condition for covering: no part of the
        # target region is outside the block.
        if state.get(target, "bbox_left") > state.get(block, "bbox_left") and\
           state.get(target, "bbox_right") < state.get(block, "bbox_right"):
            return True
        else:
            return False

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._block_id_to_block: Dict[int, Object] = {}
        self._target_id_to_target: Dict[int, Object] = {}

        # Create a copy of the pybullet robot for checking forward kinematics
        # in step() without changing the "real" robot state.
        fk_physics_id = p.connect(p.DIRECT)
        self._pybullet_robot_fk = self._create_pybullet_robot(fk_physics_id)

    def simulate(self, state: State, action: Action) -> State:
        # To implement this, need to handle resetting to states where the
        # block is held, and need to take into account the offset between
        # the hand and the held block, which reset_state() doesn't yet.
        raise NotImplementedError("Simulate not implemented for PyBulletCover")

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle cover-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        bodies["table_id"] = table_id
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)

        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))
        num_blocks = max(CFG.cover_num_blocks_train, CFG.cover_num_blocks_test)
        block_ids = []
        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            width = CFG.cover_block_widths[i] / max_width * cls._max_obj_width
            half_extents = (cls._obj_len_hgt / 2.0, width / 2.0,
                            cls._obj_len_hgt / 2.0)
            block_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))
        bodies["block_ids"] = block_ids

        num_targets = max(CFG.cover_num_targets_train,
                          CFG.cover_num_targets_test)
        target_ids = []
        for i in range(num_targets):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            color = (color[0], color[1], color[2], 0.5)  # slightly transparent
            width = CFG.cover_target_widths[i] / max_width * cls._max_obj_width
            half_extents = (cls._obj_len_hgt / 2.0, width / 2.0,
                            cls._target_height / 2.0)
            target_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))

        bodies["target_ids"] = target_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids = pybullet_bodies["block_ids"]
        self._target_ids = pybullet_bodies["target_ids"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.workspace_x, cls.robot_init_y, cls.workspace_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        if self._HandEmpty_holds(state, []):
            fingers = self._pybullet_robot.open_fingers
        else:
            fingers = self._pybullet_robot.closed_fingers
        y_norm = state.get(self._robot, "pose_y_norm")
        # De-normalize robot y to actual coordinates.
        ry = self.y_lb + (self.y_ub - self.y_lb) * y_norm
        rx = state.get(self._robot, "pose_x")
        rz = state.get(self._robot, "pose_z")
        # The orientation is fixed in this environment.
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        return np.array([rx, ry, rz, qx, qy, qz, qw, fingers],
                        dtype=np.float32)

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle cover-specific resetting."""
        super()._reset_state(state)
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._heavy_blocks = set()
        self._obj_id_to_obj = {}
        self._obj_id_to_obj[self._pybullet_robot.robot_id] = self._robot
        self._obj_id_to_obj[self._table_id] = self._table
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert width == state.get(block_obj, "width")
            self._block_id_to_block[block_id] = block_obj
            self._obj_id_to_obj[block_id] = block_obj
            bx = self.workspace_x
            # De-normalize block y to actual coordinates.
            y_norm = state.get(block_obj, "pose_y_norm")
            by = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            if state.get(block_obj, "grasp") != -1:
                # If an object starts out held, it has a different z.
                bz = self.workspace_z - self._offset
            else:
                bz = self._table_height + self._obj_len_hgt * 0.5
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)
            if state.get(block_obj, "grasp") != -1:
                # If an object starts out held, set up the grasp constraint.
                self._held_obj_id = self._detect_held_object()
                assert self._held_obj_id == block_id
                self._create_grasp_constraint()

            # Change the block's color if it's the CoverWeighted env.
            if self.get_name() == "pybullet_cover_weighted":
                if state.get(block_obj, "is_heavy") == 1:
                    self._heavy_blocks.add(block_obj)
                    p.changeVisualShape(block_id,
                                        -1,
                                        rgbaColor=self.heavy_color)
                else:
                    p.changeVisualShape(block_id,
                                        -1,
                                        rgbaColor=self.light_color)

        # For any blocks not involved, put them out of view.
        h = self._obj_len_hgt
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Reset targets based on the state.
        target_objs = state.get_objects(self._target_type)
        self._target_id_to_target = {}
        for i, target_obj in enumerate(target_objs):
            target_id = self._target_ids[i]
            width_unnorm = p.getVisualShapeData(
                target_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert width == state.get(target_obj, "width")
            self._target_id_to_target[target_id] = target_obj
            self._obj_id_to_obj[target_id] = target_obj
            tx = self.workspace_x
            # De-normalize target y to actual coordinates.
            y_norm = state.get(target_obj, "pose_y_norm")
            ty = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            tz = self._table_height  # + self._obj_len_hgt * 0.5
            p.resetBasePositionAndOrientation(
                target_id, [tx, ty, tz],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Draw hand regions as debug lines.
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert self.using_gui, \
                "use_gui must be True to use pybullet_draw_debug."
            p.removeAllUserDebugItems(physicsClientId=self._physics_client_id)
            for hand_lb, hand_rb in self._get_hand_regions(state):
                # De-normalize hand bounds to actual coordinates.
                y_lb = self.y_lb + (self.y_ub - self.y_lb) * hand_lb
                y_rb = self.y_lb + (self.y_ub - self.y_lb) * hand_rb
                p.addUserDebugLine(
                    [self.workspace_x, y_lb, self._table_height + 1e-4],
                    [self.workspace_x, y_rb, self._table_height + 1e-4],
                    [0.0, 0.0, 1.0],
                    lineWidth=5.0,
                    physicsClientId=self._physics_client_id)

    def step(self, action: Action) -> State:
        # In the cover environment, we need to first check the hand region
        # constraint before we can call PyBullet.
        # Use self._pybullet_robot_fk to run forward kinematics, since that
        # method shouldn't be run on the client that is doing simulation.
        joint_positions = action.arr.tolist()
        _, ry, rz = self._pybullet_robot_fk.forward_kinematics(
            joint_positions).position
        hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
        hand_regions = self._get_hand_regions(self._current_state)
        # If we're going down to grasp, we need to be in a hand region.
        # Otherwise, we don't care if we're between hand regions.
        # To decide whether we should care about hand regions, we use a
        # value z_thresh that is the average between the resting z
        # and the z used for picking/placing a block.
        z_thresh = (self.pickplace_z + self.workspace_z) / 2
        if rz < z_thresh and not any(hand_lb <= hand <= hand_rb
                                     for hand_lb, hand_rb in hand_regions):
            # The constraint is violated, so noop.
            state_copy = self._current_state.copy()
            return state_copy

        # If the env is pybullet_cover_weighted, then when the robot goes to
        # grasp, we check if the block is heavy, if it is, then noop.
        # if isinstance(self, PyBulletCoverWeighted):
        #     # Check if hand is on top of a heavy block
        #     if rz < z_thresh:
        #         for block_id in self._heavy_block_ids:
        #             obj = self._block_id_to_block[block_id]
        #             pose_y = self._current_state.get(obj, "pose_y_norm")
        #             width = self._current_state.get(obj, "width")
        #             lb, ub = pose_y - width / 2, pose_y + width / 2
        #             if lb <= hand <= ub:
        #                 # Heavy block is under the hand, so noop
        #                 state_copy = self._current_state.copy()
        #                 return state_copy

        return super().step(action)

    def _get_state(self) -> State:
        state_dict = {}
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # Get robot state.
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
        if "fingers" in self._robot_type.feature_names:
            state_dict[self._robot] = np.array([hand, rx, rz, fingers],
                                               dtype=np.float32)
        else:
            state_dict[self._robot] = np.array([hand, rx, rz],
                                               dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            (_, by, _), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            pose = (by - self.y_lb) / (self.y_ub - self.y_lb)
            held = (block_id == self._held_obj_id)
            if held:
                grasp_unnorm = p.getConstraintInfo(
                    self._held_constraint_id, self._physics_client_id)[7][1]
                # Normalize grasp.
                grasp = grasp_unnorm / (self.y_ub - self.y_lb)
            else:
                grasp = -1
            if "is_heavy" in self._block_type.feature_names:
                is_heavy = int(block in self._heavy_blocks)
                state_dict[block] = np.array(
                    [1.0, 0.0, width, pose, grasp, is_heavy], dtype=np.float32)
            else:
                state_dict[block] = np.array([1.0, 0.0, width, pose, grasp],
                                             dtype=np.float32)

        # Get target states.
        for target_id, target in self._target_id_to_target.items():
            width_unnorm = p.getVisualShapeData(
                target_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            (_, ty, _), _ = p.getBasePositionAndOrientation(
                target_id, physicsClientId=self._physics_client_id)
            pose = (ty - self.y_lb) / (self.y_ub - self.y_lb)
            state_dict[target] = np.array([0.0, 1.0, width, pose],
                                          dtype=np.float32)

        # Get table state.
        state_dict[self._table] = np.array([], dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        # Both fetch and panda have grippers parallel to x-axis
        return {
            self._pybullet_robot.left_finger_id: np.array([1., 0., 0.]),
            self._pybullet_robot.right_finger_id: np.array([-1., 0., 0.]),
        }

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover"

    def _get_tasks(self,
                   num: int,
                   rng: np.random.Generator,
                   is_train: Optional[bool] = True) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num, rng, is_train)
        return self._add_pybullet_state_to_tasks(tasks)

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)


class PyBulletCoverTypedOptionEnv(PyBulletCoverEnv):

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
        self._block_type = Type(
            "block",
            ["is_block", "is_target", "width", "pose_y_norm", "grasp"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._target_type = Type(
            "target", ["is_block", "is_target", "width", "pose_y_norm"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._robot_type = Type(
            "robot", ["pose_y_norm", "pose_x", "pose_z", "fingers"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._table_type = Type(
            "table", bbox_features if CFG.env_include_bbox_features else [])

        self._GripperOpen_NSP = NSPredicate("HandEmpty", [self._robot_type],
                                            self._GripperOpen_NSP_holds)
        self._Holding_NSP = NSPredicate("Holding", [self._block_type],
                                        self._Holding_NSP_holds)
        self._Covers_NSP = NSPredicate("Covers",
                                       [self._block_type, self._target_type],
                                       self._Covers_NSP_holds)

        self.ns_to_sym_predicates: Dict[Tuple[str], Predicate] = {
            ("HandEmpty"): self._HandEmpty,
            ("BlockGrasped"): self._Holding,
            ("Holding"): self._Holding,
        }

    @property
    def ns_predicates(self) -> Set[NSPredicate]:
        return {
            self._GripperOpen_NSP,
            self._Holding_NSP,
            self._Covers_NSP,
        }

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover_typed_options"


class PyBulletCoverWeighted(PyBulletCoverTypedOptionEnv):
    """A variation that have heavy blocks that the robot can't lift.

    This is achieved by keep a track of a listed of heavy blocks, which
    would fell to be lifted by the robot in the simulator. These blocks
    are marked by gold color, while the normal blocks are white. Without
    changing the goals, the intend outcome is for the robot to realize
    the tasks with the heavy blocks are not achievable.
    """
    # Define colors
    light_color = [0, 1, 0, 1]  # light block color -> green
    heavy_color = [1, 0, 0, 1]  # heavy block color -> red
    target_colr = [0, 0, 1, 1]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
        self._block_type = Type("block", [
            "is_block", "is_target", "width", "pose_y_norm", "grasp",
            "is_heavy"
        ] + (bbox_features if CFG.env_include_bbox_features else []))
        self._target_type = Type(
            "target", ["is_block", "is_target", "width", "pose_y_norm"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._robot_type = Type(
            "robot", ["pose_y_norm", "pose_x", "pose_z", "fingers"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._table_type = Type(
            "table", bbox_features if CFG.env_include_bbox_features else [])

        self._heavy_blocks: Set[Object] = set()

        for target_id in self._target_ids:
            p.changeVisualShape(target_id, -1, rgbaColor=self.target_colr)

    # def _reset_state(self, state: State) -> None:
    #     super()._reset_state(state)
    #     # Randomly select some block ids to be heavy from block_ids
    #     num_blocks = len(self._block_ids)
    #     num_heavy_blocks = int(CFG.cover_weighted_ratio * num_blocks)
    #     self._heavy_block_ids = set(random.sample(self._block_ids,
    #                                                 num_heavy_blocks))

    #     # Apply colors to blocks
    #     for block_id in self._block_ids:
    #         if block_id in self._heavy_block_ids:
    #             p.changeVisualShape(block_id, -1, rgbaColor=self.heavy_color)
    #         else:
    #             p.changeVisualShape(block_id, -1, rgbaColor=self.light_color)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover_weighted"

    @property
    def ns_predicates(self) -> Set[NSPredicate]:
        return {
            self._IsLight,
            self._GripperOpen_NSP,
            # self._Holding,
            self._Holding_NSP,
            # self._Covers_NSP,
        }
    
    def check_task_solvable(self, task: EnvironmentTask) -> bool:
        """Check if the task is solvable."""
        solvable = True
        init = task.init
        goal = task.goal
        for atom in list(goal):
            block, _ = atom.objects
            if init.get(block, "is_heavy"):
                solvable = False
                break
        return solvable
