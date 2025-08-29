"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType
from typing import cast

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_domino import PyBulletDominoEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletDominoEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletDominoGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletDominoEnv]] = PyBulletDominoEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _transport_z: ClassVar[float] = env_cls.table_height +\
            env_cls.domino_height * 2.26
    _transport_z_push: ClassVar[float] = env_cls.table_height +\
            env_cls.domino_height * 1.5
    _offset_x: ClassVar[float] = env_cls.domino_depth * 3
    _offset_z: ClassVar[float] = env_cls.domino_height * 0.55
    _place_drop_z: ClassVar[float] = env_cls.table_height +\
            env_cls.domino_height * 1.13

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the grow environment."""
        del env_name, predicates  # unused

        _, pybullet_robot, _ = \
            PyBulletDominoEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        domino_type = types["domino"]
        rotation_type = types["rot"]
        position_type = types["loc"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletDominoEnv._fingers_state_to_joint(
                pybullet_robot, state.get(robot, "fingers"))

        def open_fingers_func(state: State, objects: Sequence[Object],
                              params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.open_fingers - 0.01
            return current, target

        def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.closed_fingers - 0.01
            return current, target

        options: Set[ParameterizedOption] = set()
        # Push
        option_type = [robot_type, domino_type]
        params_space = Box(0, 1, (0, ))
        Push = utils.LinearChainParameterizedOption(
            "Push",
            [
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletEnv.grasp_tol_small),
                cls._create_domino_move_to_push_domino_option(
                    "MoveToAboveDomino",
                    lambda x, rot: x - np.sin(rot) * cls._offset_x,
                    lambda y, rot: y - np.cos(rot) * cls._offset_x,
                    lambda _: cls._transport_z_push, "closed", option_type,
                    params_space),
                cls._create_domino_move_to_push_domino_option(
                    "MoveToBehindDomino",
                    lambda x, rot: x - np.sin(rot) * cls._offset_x,
                    lambda y, rot: y - np.cos(rot) * cls._offset_x,
                    lambda z: z + cls._offset_z, "closed", option_type,
                    params_space),
                cls._create_domino_move_to_push_domino_option(
                    "PushDomino",
                    lambda x, rot: x + np.sin(rot) * cls._offset_x / 4,
                    lambda y, rot: y + np.cos(rot) * cls._offset_x / 4,
                    lambda z: z + cls._offset_z, "closed", option_type,
                    params_space),
                cls._create_domino_move_to_push_domino_option(
                    "BackUp", lambda _1, _2: cls.env_cls.robot_init_x,
                    lambda _1, _2: cls.env_cls.robot_init_y,
                    lambda _: cls.env_cls.robot_init_z, "closed", option_type,
                    params_space),
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_type, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletEnv.grasp_tol_small),
                # cls._create_domino_move_to_push_domino_option(
                #     "MoveToBehindDomino",
                #     lambda _: cls.env_cls.start_domino_x - cls._offset_x,
                #     lambda z: z + cls._offset_z,
                #     "closed",
                #     option_type, params_space),
            ])
        options.add(Push)

        # Pick
        pick_option_types = [robot_type, domino_type]
        pick_params_space = Box(0, 1, (0, ))

        def _Pick_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, domino = objects
            return state.get(robot, "fingers") < PyBulletEnv.grasp_tol

        Pick = utils.LinearChainParameterizedOption("Pick", [
            create_change_fingers_option(
                pybullet_robot, "CloseFingers", pick_option_types,
                pick_params_space, close_fingers_func,
                CFG.pybullet_max_vel_norm, PyBulletEnv.grasp_tol),
            cls._create_domino_move_to_domino_option(
                "MoveToAboveDomino", lambda dx: dx, lambda dy: dy,
                lambda _: cls._transport_z, "closed", pick_option_types,
                pick_params_space),
            cls._create_domino_move_to_domino_option(
                "MoveToGraspDomino", lambda dx: dx, lambda dy: dy,
                lambda dz: dz + cls._offset_z, "open", pick_option_types,
                pick_params_space),
            create_change_fingers_option(pybullet_robot,
                                         "CloseFingers",
                                         pick_option_types,
                                         pick_params_space,
                                         close_fingers_func,
                                         CFG.pybullet_max_vel_norm,
                                         PyBulletEnv.grasp_tol_small,
                                         terminal=_Pick_terminal),
            cls._create_domino_move_to_domino_option(
                "LiftDomino", lambda dx: dx, lambda dy: dy,
                lambda _: cls._transport_z, "closed", pick_option_types,
                pick_params_space),
        ])
        options.add(Pick)

        # Place
        place_option_types = [
            robot_type, domino_type, domino_type, position_type, rotation_type
        ]
        place_params_space = Box(0, 1, (0, ))

        Place = utils.LinearChainParameterizedOption("Place", [
            cls._create_domino_place_option(
                "MoveToAbovePlacement", lambda _: cls._transport_z, "closed",
                place_option_types, place_params_space),
            cls._create_domino_place_option(
                "MoveToPlacement", lambda _: cls._place_drop_z, "closed",
                place_option_types, place_params_space),
            create_change_fingers_option(
                pybullet_robot, "OpenFingers", place_option_types,
                place_params_space, open_fingers_func,
                CFG.pybullet_max_vel_norm, PyBulletEnv.grasp_tol),
            cls._create_domino_place_option(
                "MoveAwayFromPlacement", lambda _: cls._transport_z, "open",
                place_option_types, place_params_space),
        ])
        options.add(Place)

        # NoOp
        noop_params_space = Box(0, 1, (0, ))

        def _create_no_op_policy() -> ParameterizedPolicy:
            nonlocal action_space

            def _policy(state: State, memory: Dict, objects: Sequence[Object],
                        params: Array) -> Action:
                del memory, params
                robot = objects[0]
                # check finger open or closed
                finger = state.get(robot, "fingers")
                mid_point = (pybullet_robot.open_fingers +
                             pybullet_robot.closed_fingers) / 2
                if finger > mid_point:
                    # currently open
                    finger_delta = cls._finger_action_nudge_magnitude
                else:
                    finger_delta = -cls._finger_action_nudge_magnitude

                # nudge finger to the direction of the current state to counter
                state = cast(utils.PyBulletState, state)
                joint_positions = state.joint_positions.copy()
                finger_position = joint_positions[
                    pybullet_robot.left_finger_joint_idx]
                # The finger action is an absolute joint position for the fingers.
                f_action = finger_position + finger_delta
                # Override the meaningless finger values in joint_action.
                joint_positions[
                    pybullet_robot.left_finger_joint_idx] = f_action
                joint_positions[
                    pybullet_robot.right_finger_joint_idx] = f_action
                # slide
                action = np.array(joint_positions, dtype=np.float32)
                action = action.clip(action_space.low,
                                     action_space.high).astype(np.float32)
                return Action(action)

            return _policy

        NoOp = ParameterizedOption(
            "NoOp",
            types=[robot_type],
            params_space=noop_params_space,
            policy=_create_no_op_policy(),
            initiable=lambda _1, _2, _3, _4: True,
            terminal=lambda _1, _2, _3, _4: False,
        )
        options.add(NoOp)

        return options

    @classmethod
    def _create_domino_move_to_push_domino_option(
            cls, name: str, x_func: Callable[[float, float], float],
            y_func: Callable[[float, float], float], z_func: Callable[[float],
                                                                      float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Create a move-to-pose option for the domino environment."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, domino = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            dx = state.get(domino, "x")
            dy = state.get(domino, "y")
            dz = state.get(domino, "z")
            drot = state.get(domino, "rot")
            target_position = (x_func(dx, drot), y_func(dy, drot), z_func(dz))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, drot + np.pi / 2])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(),
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)

    @classmethod
    def _create_domino_move_to_domino_option(
            cls, name: str, x_func: Callable[[float], float],
            y_func: Callable[[float], float], z_func: Callable[[float], float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Create a move-to-pose option for simple domino movement."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, domino = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            dx = state.get(domino, "x")
            dy = state.get(domino, "y")
            dz = state.get(domino, "z")
            drot = state.get(domino, "rot")
            target_position = (x_func(dx), y_func(dy), z_func(dz))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, drot])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(),
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)

    @classmethod
    def _create_domino_place_option(cls, name: str, z_func: Callable[[float],
                                                                     float],
                                    finger_status: str,
                                    option_types: List[Type],
                                    params_space: Box) -> ParameterizedOption:
        """Create a move-to-pose option for placing dominoes."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, domino_f, domino_b, tgt_pos, rotation = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            # Get properties of the reference domino (domino2)
            x2, y2 = state.get(domino_b, "x"), state.get(domino_b, "y")
            rot2 = state.get(domino_b, "rot")
            # Use domino1's current z for reference
            dz = state.get(domino_f, "z")

            # Compute dir_value based on rotation of domino2 and the rotation object
            target_angle = state.get(rotation, "angle")  # degrees
            target_rot_rad = np.radians(target_angle)  # convert to radians

            # Calculate rotation difference (target - domino2)
            rot_diff = target_rot_rad - rot2
            # Normalize rotation difference to [-π, π] range
            rot_diff = utils.wrap_angle(rot_diff)

            # Determine direction based on rotation difference
            angle_tol = 2e-1  # Tolerance for checking cardinal/diagonal angles
            # ~22.5 degrees tolerance
            if abs(rot_diff) < np.pi / 8 or abs(abs(rot_diff) -
                                                np.pi / 2) < angle_tol:
                dir_value = 0.0  # straight or perpendicular
            elif rot_diff > np.pi / 8:
                dir_value = 1.0  # left (positive rotation difference)
            else:
                dir_value = 2.0  # right (negative rotation difference)

            # Get constants from the environment class
            gap = cls.env_cls.pos_gap

            target_angle_is_cardinal = abs(np.sin(
                2 * target_rot_rad)) < angle_tol

            # Case 1: Place straight ahead
            if dir_value == 0.0 or target_angle_is_cardinal:  # straight
                # target_x = x2 + gap * np.sin(rot2)
                # target_y = y2 + gap * np.cos(rot2)
                target_x = state.get(tgt_pos, "xx")
                target_y = state.get(tgt_pos, "yy")
                if abs(rot_diff) < np.pi / 8:
                    target_rot = rot2
                else:
                    target_rot = target_rot_rad
            # Case 2: Place to the left or right (a turn)
            else:
                # Map dir_value to turn_dir from the generator code
                # dir_value: 1.0 -> left, 2.0 -> right
                # turn_dir: -1.0 -> left, 1.0 -> right
                turn_dir = -1.0 if dir_value == 1.0 else 1.0

                # If domino2 is in a cardinal direction (0, 90, 180 deg),
                # we are initiating a turn. This logic mirrors placing d1.
                if abs(np.sin(2 * rot2)) < angle_tol:
                    # The target domino will be turned by 45 degrees.
                    target_rot = rot2 - turn_dir * np.pi / 4

                    # First, calculate the position on the grid, one step forward.
                    # grid_x = x2 + gap * np.sin(rot2)
                    # grid_y = y2 + gap * np.cos(rot2)
                    grid_x = state.get(tgt_pos, "xx")
                    grid_y = state.get(tgt_pos, "yy")

                    # Then, apply the diagonal shift from the generator for stability.
                    shift_magnitude = cls.env_cls.domino_width * cls.env_cls.turn_shift_frac
                    shift_dx = shift_magnitude * (turn_dir * np.cos(rot2) -
                                                  np.sin(rot2))
                    shift_dy = shift_magnitude * (-turn_dir * np.sin(rot2) -
                                                  np.cos(rot2))
                    target_x = grid_x + shift_dx
                    target_y = grid_y + shift_dy

                # If domino2 is in a diagonal direction (45, 135 deg),
                # we are completing a turn. This logic mirrors placing d2.
                elif abs(np.cos(2 * rot2)) < angle_tol:
                    # The target domino completes the 90-degree turn.
                    target_rot = rot2 - turn_dir * np.pi / 4

                    # Calculate position relative to domino2 using the generator's formula.
                    shift_magnitude = cls.env_cls.domino_width * cls.env_cls.turn_shift_frac
                    sin_rot2 = np.sin(rot2)
                    cos_rot2 = np.cos(rot2)

                    disp_x = (
                        gap * turn_dir * cos_rot2 +
                        (2 * shift_magnitude - gap) * sin_rot2) / np.sqrt(2)
                    disp_y = (
                        -gap * turn_dir * sin_rot2 +
                        (2 * shift_magnitude - gap) * cos_rot2) / np.sqrt(2)

                    target_x = x2 + disp_x
                    target_y = y2 + disp_y

                # Fallback for unexpected rotations: default to cardinal logic.
                else:
                    logging.warning(
                        f"Unexpected domino rotation {rot2} in place option. "
                        "Defaulting to cardinal turn logic.")
                    raise ValueError(
                        f"Unexpected domino rotation {rot2} in place option. ")
                    # target_rot = rot2 - turn_dir * np.pi / 4
                    # grid_x = x2 + gap * np.sin(rot2)
                    # grid_y = y2 + gap * np.cos(rot2)
                    # shift_magnitude = cls.env_cls.domino_width * cls.env_cls.turn_shift_frac
                    # shift_dx = shift_magnitude * (turn_dir * np.cos(rot2) - np.sin(rot2))
                    # shift_dy = shift_magnitude * (-turn_dir * np.sin(rot2) - np.cos(rot2))
                    # target_x = grid_x + shift_dx
                    # target_y = grid_y + shift_dy

            target_position = (target_x, target_y, z_func(dz))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, target_rot])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(),
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)
