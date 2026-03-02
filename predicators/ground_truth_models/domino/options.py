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
# from predicators.envs.pybullet_domino.old.pybullet_domino_grid import \
#     PyBulletDominoEnv
from predicators.envs.pybullet_domino import PyBulletDominoEnv
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, create_pick_skill, \
    create_place_skill, create_push_skill, create_wait_option, \
    make_move_to_phase
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
        return {"pybullet_domino_grid", "pybullet_domino"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the domino environment."""
        if CFG.domino_use_skill_factories:
            return cls._get_options_skill_factories(env_name, types,
                                                    predicates, action_space)
        return cls._get_options_legacy(env_name, types, predicates,
                                       action_space)

    @classmethod
    def _get_options_legacy(cls, env_name: str, types: Dict[str, Type],
                            predicates: Dict[str, Predicate],
                            action_space: Box) -> Set[ParameterizedOption]:
        """Original option implementation (legacy path)."""
        del env_name, predicates  # unused

        _, pybullet_robot, _ = \
            PyBulletDominoEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        domino_type = types["domino"]
        rotation_type = types["angle"]
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

        if CFG.domino_restricted_push:
            # PushRestricted - like Push but takes only robot (finds start block from state)
            restricted_option_type = [robot_type]
            restricted_params_space = Box(0, 1, (0, ))
            PushRestricted = utils.LinearChainParameterizedOption(
                "Push", [
                    create_change_fingers_option(
                        pybullet_robot, "CloseFingers", restricted_option_type,
                        restricted_params_space, close_fingers_func,
                        CFG.pybullet_max_vel_norm,
                        PyBulletEnv.grasp_tol_small),
                    cls._create_domino_move_to_push_start_block_option(
                        "MoveToAboveDomino",
                        lambda x, rot: x - np.sin(rot) * cls._offset_x,
                        lambda y, rot: y - np.cos(rot) * cls._offset_x,
                        lambda _: cls._transport_z_push, "closed",
                        restricted_option_type, restricted_params_space,
                        domino_type),
                    cls._create_domino_move_to_push_start_block_option(
                        "MoveToBehindDomino",
                        lambda x, rot: x - np.sin(rot) * cls._offset_x,
                        lambda y, rot: y - np.cos(rot) * cls._offset_x,
                        lambda z: z + cls._offset_z, "closed",
                        restricted_option_type, restricted_params_space,
                        domino_type),
                    cls._create_domino_move_to_push_start_block_option(
                        "PushDomino",
                        lambda x, rot: x + np.sin(rot) * cls._offset_x / 4,
                        lambda y, rot: y + np.cos(rot) * cls._offset_x / 4,
                        lambda z: z + cls._offset_z, "closed",
                        restricted_option_type, restricted_params_space,
                        domino_type),
                    cls._create_domino_move_to_push_start_block_option(
                        "BackUp", lambda _1, _2: cls.env_cls.robot_init_x,
                        lambda _1, _2: cls.env_cls.robot_init_y,
                        lambda _: cls.env_cls.robot_init_z, "closed",
                        restricted_option_type, restricted_params_space,
                        domino_type),
                    create_change_fingers_option(
                        pybullet_robot, "OpenFingers", restricted_option_type,
                        restricted_params_space, open_fingers_func,
                        CFG.pybullet_max_vel_norm,
                        PyBulletEnv.grasp_tol_small),
                ])
            options.add(PushRestricted)
        else:
            # Push
            option_type = [robot_type, domino_type]
            params_space = Box(0, 1, (0, ))
            Push = utils.LinearChainParameterizedOption(
                "Push",
                [
                    create_change_fingers_option(
                        pybullet_robot, "CloseFingers", option_type,
                        params_space, close_fingers_func,
                        CFG.pybullet_max_vel_norm,
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
                        lambda _: cls.env_cls.robot_init_z, "closed",
                        option_type, params_space),
                    create_change_fingers_option(pybullet_robot, "OpenFingers",
                                                 option_type, params_space,
                                                 open_fingers_func,
                                                 CFG.pybullet_max_vel_norm,
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

        # Choose between discrete (Place) or continuous (PlaceContinuous) based on CFG
        if CFG.domino_use_continuous_place:
            # PlaceContinuous - continuous parameters version
            place_continuous_option_types = [robot_type]
            # Parameters: [x, y, rotation_radians]
            place_continuous_params_space = Box(
                low=np.array([cls.env_cls.x_lb, cls.env_cls.y_lb, -np.pi]),
                high=np.array([cls.env_cls.x_ub, cls.env_cls.y_ub, np.pi]),
                shape=(3, ),
                dtype=np.float32)

            Place = utils.LinearChainParameterizedOption(
                "Place", [
                    cls._create_domino_place_continuous_option(
                        "MoveToAbovePlacement", lambda _: cls._transport_z,
                        "closed", place_continuous_option_types,
                        place_continuous_params_space),
                    cls._create_domino_place_continuous_option(
                        "MoveToPlacement", lambda _: cls._place_drop_z,
                        "closed", place_continuous_option_types,
                        place_continuous_params_space),
                    create_change_fingers_option(
                        pybullet_robot, "OpenFingers",
                        place_continuous_option_types,
                        place_continuous_params_space, open_fingers_func,
                        CFG.pybullet_max_vel_norm, PyBulletEnv.grasp_tol),
                    cls._create_domino_place_continuous_option(
                        "MoveAwayFromPlacement", lambda _: cls._transport_z,
                        "open", place_continuous_option_types,
                        place_continuous_params_space),
                ])
        else:
            # Place - discrete version with object parameters
            place_option_types = [
                robot_type, domino_type, domino_type, position_type,
                rotation_type
            ]
            place_params_space = Box(0, 1, (0, ))
            Place = utils.LinearChainParameterizedOption(
                "Place", [
                    cls._create_domino_place_option(
                        "MoveToAbovePlacement", lambda _: cls._transport_z,
                        "closed", place_option_types, place_params_space),
                    cls._create_domino_place_option(
                        "MoveToPlacement", lambda _: cls._place_drop_z,
                        "closed", place_option_types, place_params_space),
                    create_change_fingers_option(
                        pybullet_robot, "OpenFingers", place_option_types,
                        place_params_space, open_fingers_func,
                        CFG.pybullet_max_vel_norm, PyBulletEnv.grasp_tol),
                    cls._create_domino_place_option(
                        "MoveAwayFromPlacement", lambda _: cls._transport_z,
                        "open", place_option_types, place_params_space),
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
            drot = state.get(domino, "yaw")
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
    def _find_start_block(cls, state: State, domino_type: Type) -> Object:
        """Find the start block domino using the InitialBlock classifier."""
        for domino in state.get_objects(domino_type):
            if DominoComponent._StartBlock_holds(state, [domino]):
                return domino
        raise ValueError("No start block found in state")

    @classmethod
    def _create_domino_move_to_push_start_block_option(
            cls, name: str, x_func: Callable[[float, float], float],
            y_func: Callable[[float, float], float], z_func: Callable[[float],
                                                                      float],
            finger_status: str, option_types: List[Type], params_space: Box,
            domino_type: Type) -> ParameterizedOption:
        """Create a push option that automatically finds the start block."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, = objects
            domino = cls._find_start_block(state, domino_type)
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            dx = state.get(domino, "x")
            dy = state.get(domino, "y")
            dz = state.get(domino, "z")
            drot = state.get(domino, "yaw")
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
            drot = state.get(domino, "yaw")
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
            rot2 = state.get(domino_b, "yaw")
            # Use domino1's current z for reference
            dz = state.get(domino_f, "z")

            # Compute dir_value based on rotation of domino2 and the rotation object
            # target_angle = state.get(rotation, "angle")  # degrees
            target_angle = float(
                rotation.name.split("_")[-1])  # extract angle from name
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
                # target_x = state.get(tgt_pos, "xx")
                # target_y = state.get(tgt_pos, "yy")
                target_x = float(
                    tgt_pos.name.split("_")[1])  # extract x from name
                target_y = float(tgt_pos.name.split("_")[2])  # extract y from
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
                    # grid_x = state.get(tgt_pos, "xx")
                    # grid_y = state.get(tgt_pos, "yy")
                    grid_x = float(
                        tgt_pos.name.split("_")[1])  # extract x from name
                    grid_y = float(
                        tgt_pos.name.split("_")[2])  # extract y from

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
                    # raise ValueError(
                    #     f"Unexpected domino rotation {rot2} in place option. ")
                    # The target domino will be turned by 45 degrees.
                    target_rot = rot2 - turn_dir * np.pi / 4
                    # grid_x = state.get(tgt_pos, "xx")
                    # grid_y = state.get(tgt_pos, "yy")
                    grid_x = float(
                        tgt_pos.name.split("_")[1])  # extract x from name
                    grid_y = float(
                        tgt_pos.name.split("_")[2])  # extract y from
                    shift_magnitude = cls.env_cls.domino_width * cls.env_cls.turn_shift_frac
                    shift_dx = shift_magnitude * (turn_dir * np.cos(rot2) -
                                                  np.sin(rot2))
                    shift_dy = shift_magnitude * (-turn_dir * np.sin(rot2) -
                                                  np.cos(rot2))
                    target_x = grid_x + shift_dx
                    target_y = grid_y + shift_dy

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

    @classmethod
    def _create_domino_place_continuous_option(
            cls, name: str, z_func: Callable[[float], float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Create a move-to-pose option for placing dominoes with continuous
        parameters.

        This version accepts continuous parameters [x, y,
        rotation_radians] instead of using position and rotation
        objects.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            # params: [x, y, rotation_radians]
            assert len(params) == 3
            target_x, target_y, target_rot = params

            robot = objects[0]
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            # Use a default z value (could be improved by tracking held domino)
            # For now, use a reasonable z height
            dz = cls._place_drop_z

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

    # ------------------------------------------------------------------
    # Skill-factories-based implementation
    # ------------------------------------------------------------------

    @classmethod
    def _get_options_skill_factories(
            cls, env_name: str, types: Dict[str, Type],
            predicates: Dict[str, Predicate],
            action_space: Box) -> Set[ParameterizedOption]:
        """Option implementation built on skill_factories primitives."""
        del env_name, predicates, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletDominoEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        domino_type = types["domino"]
        rotation_type = types["angle"]
        position_type = types["loc"]

        cfg = cls._build_skill_config(pybullet_robot)

        options: Set[ParameterizedOption] = set()

        if CFG.domino_restricted_push:
            options.add(
                cls._create_sf_push_restricted(cfg, robot_type, domino_type))
        else:
            options.add(cls._create_sf_push(cfg, robot_type, domino_type))

        options.add(cls._create_sf_pick(cfg, robot_type, domino_type))
        options.add(
            cls._create_sf_place(cfg, robot_type))
        options.add(create_wait_option("NoOp", cfg, robot_type))

        return options

    @classmethod
    def _build_skill_config(
            cls,
            pybullet_robot: SingleArmPyBulletRobot) -> SkillConfig:
        """Build the shared SkillConfig for domino skill_factories options."""
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=PyBulletDominoEnv._fingers_state_to_joint,
            move_to_pose_tol=cls._move_to_pose_tol,
            finger_action_nudge_magnitude=cls._finger_action_nudge_magnitude,
            max_vel_norm=CFG.pybullet_max_vel_norm,
            grasp_tol=PyBulletEnv.grasp_tol_small,
            ik_validate=CFG.pybullet_ik_validate,
            robot_init_tilt=cls.env_cls.robot_init_tilt,
        )

    @classmethod
    def _create_sf_push(cls, cfg: SkillConfig, robot_type: Type,
                        domino_type: Type) -> ParameterizedOption:
        """Push option using create_push_skill."""
        option_types = [robot_type, domino_type]
        params_space = Box(0, 1, (0, ))

        def _get_target(state: State, objects: Sequence[Object],
                        params: Array,
                        config: SkillConfig) -> Tuple[float, float, float,
                                                      float]:
            del params, config
            _, domino = objects
            return (state.get(domino, "x"), state.get(domino, "y"),
                    state.get(domino, "z"), state.get(domino, "yaw"))

        offset_x = cls._offset_x
        offset_z = cls._offset_z
        transport_z_push = cls._transport_z_push
        env_cls = cls.env_cls

        def _waypoints(tx: float, ty: float, tz: float, tyaw: float,
                       c: SkillConfig) -> List[Tuple[float, float, float,
                                                     float, str]]:
            del c
            drot = tyaw
            return [
                (tx - np.sin(drot) * offset_x, ty - np.cos(drot) * offset_x,
                 transport_z_push, drot + np.pi / 2, "closed"),
                (tx - np.sin(drot) * offset_x, ty - np.cos(drot) * offset_x,
                 tz + offset_z, drot + np.pi / 2, "closed"),
                (tx + np.sin(drot) * offset_x / 4,
                 ty + np.cos(drot) * offset_x / 4, tz + offset_z,
                 drot + np.pi / 2, "closed"),
                (env_cls.robot_init_x, env_cls.robot_init_y,
                 env_cls.robot_init_z, drot + np.pi / 2, "closed"),
            ]

        return create_push_skill(name="Push",
                                 types=option_types,
                                 params_space=params_space,
                                 config=cfg,
                                 get_target_pose_fn=_get_target,
                                 waypoints_fn=_waypoints)

    @classmethod
    def _create_sf_push_restricted(cls, cfg: SkillConfig, robot_type: Type,
                                   domino_type: Type) -> ParameterizedOption:
        """Push (restricted) option: finds start block from state."""
        option_types = [robot_type]
        params_space = Box(0, 1, (0, ))

        def _get_target(state: State, objects: Sequence[Object],
                        params: Array,
                        config: SkillConfig) -> Tuple[float, float, float,
                                                      float]:
            del objects, params, config
            start = cls._find_start_block(state, domino_type)
            return (state.get(start, "x"), state.get(start, "y"),
                    state.get(start, "z"), state.get(start, "yaw"))

        offset_x = cls._offset_x
        offset_z = cls._offset_z
        transport_z_push = cls._transport_z_push
        env_cls = cls.env_cls

        def _waypoints(tx: float, ty: float, tz: float, tyaw: float,
                       c: SkillConfig) -> List[Tuple[float, float, float,
                                                     float, str]]:
            del c
            drot = tyaw
            return [
                (tx - np.sin(drot) * offset_x, ty - np.cos(drot) * offset_x,
                 transport_z_push, drot + np.pi / 2, "closed"),
                (tx - np.sin(drot) * offset_x, ty - np.cos(drot) * offset_x,
                 tz + offset_z, drot + np.pi / 2, "closed"),
                (tx + np.sin(drot) * offset_x / 4,
                 ty + np.cos(drot) * offset_x / 4, tz + offset_z,
                 drot + np.pi / 2, "closed"),
                (env_cls.robot_init_x, env_cls.robot_init_y,
                 env_cls.robot_init_z, drot + np.pi / 2, "closed"),
            ]

        return create_push_skill(name="Push",
                                 types=option_types,
                                 params_space=params_space,
                                 config=cfg,
                                 get_target_pose_fn=_get_target,
                                 waypoints_fn=_waypoints)

    @classmethod
    def _create_sf_pick(cls, cfg: SkillConfig, robot_type: Type,
                        domino_type: Type) -> ParameterizedOption:
        """Pick option using create_pick_skill."""

        def _get_domino_pose(
            state: State, objects: Sequence[Object], params: Array,
            c: SkillConfig
        ) -> Tuple[float, float, float, float]:
            del params, c
            _, domino = objects
            return (state.get(domino, "x"), state.get(domino, "y"),
                    state.get(domino, "z"), state.get(domino, "yaw"))

        def _grasp_terminal(state: State, objects: Sequence[Object],
                            params: Array, c: SkillConfig) -> bool:
            del params, c
            return bool(
                state.get(objects[0], "fingers") < PyBulletEnv.grasp_tol)

        return create_pick_skill(
            name="Pick",
            types=[robot_type, domino_type],
            params_space=Box(0, 1, (0,)),
            config=cfg,
            get_target_pose_fn=_get_domino_pose,
            transport_z=cls._transport_z,
            grasp_z_offset=cls._offset_z,
            grasp_terminal_fn=_grasp_terminal,
        )

    @classmethod
    def _create_sf_place(cls, cfg: SkillConfig, robot_type: Type
                         ) -> ParameterizedOption:
        """Place option using create_place_skill (discrete or continuous)."""
        return cls._create_sf_place_continuous(cfg, robot_type)

    @classmethod
    def _create_sf_place_continuous(cls, cfg: SkillConfig,
                                    robot_type: Type) -> ParameterizedOption:
        """Continuous place: params = [x, y, rotation_radians]."""
        params_space = Box(
            low=np.array([cls.env_cls.x_lb, cls.env_cls.y_lb, -np.pi]),
            high=np.array([cls.env_cls.x_ub, cls.env_cls.y_ub, np.pi]),
            shape=(3, ),
            dtype=np.float32)

        def _get_placement(state: State, objects: Sequence[Object],
                           params: Array,
                           c: SkillConfig) -> Tuple[float, float, float, float]:
            del state, objects, c
            target_x, target_y, target_rot = params
            return float(target_x), float(target_y), 0.0, float(target_rot)

        return create_place_skill(
            name="Place",
            types=[robot_type],
            params_space=params_space,
            config=cfg,
            get_target_pose_fn=_get_placement,
            transport_z=cls._transport_z,
            drop_z=cls._place_drop_z,
            params_description=("x_position", "y_position", "yaw_angle"),
        )
