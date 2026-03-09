"""Ground-truth options for the boil environment."""

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType
from typing import cast

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_boil import PyBulletBoilEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    create_pick_skill, create_place_skill, create_push_skill, \
    create_wait_option
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
        PyBulletBoilEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletBoilGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletBoilEnv]] = PyBulletBoilEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _hand_empty_move_z: ClassVar[float] = env_cls.z_ub - 0.3
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.35
    _z_offset: ClassVar[float] = 0.1
    _y_offset: ClassVar[float] = 0.03

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_boil"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the boil environment."""
        if CFG.boil_use_skill_factories:
            return cls._get_options_skill_factories(env_name, types,
                                                    predicates, action_space)
        return cls._get_options_legacy(env_name, types, predicates,
                                       action_space)

    @classmethod
    def _get_options_legacy(cls, env_name: str, types: Dict[str, Type],
                            predicates: Dict[str, Predicate],
                            action_space: Box) -> Set[ParameterizedOption]:
        """Legacy option implementations."""
        del env_name  # unused

        _, pybullet_robot, _ = \
            PyBulletBoilEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        switch_type = types["switch"]
        jug_type = types["jug"]
        burner_type = types["burner"]
        faucet_type = types["faucet"]
        # Predicates
        Holding = predicates["Holding"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletBoilEnv._fingers_state_to_joint(
                pybullet_robot, state.get(robot, "fingers"))

        def open_fingers_func(state: State, objects: Sequence[Object],
                              params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.open_fingers
            return current, target

        def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.closed_fingers
            target = 0
            return current, target

        options: Set[ParameterizedOption] = set()

        # SwitchFaucetOn
        option_type = [robot_type, faucet_type]
        params_space = Box(0, 1, (0, ))
        behind_factor = 1.9
        push_factor = 0.3
        push_above_factor = 1.3
        SwitchFaucetOn = utils.LinearChainParameterizedOption(
            "SwitchFaucetOn", [
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol_small),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToAboveAndBehindSwitch",
                    lambda y: y - cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToBehindSwitch",
                    lambda y: y - cls._y_offset * behind_factor, lambda z: z +
                    cls.env_cls.switch_height * push_above_factor, "closed",
                    option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "PushSwitchOn", lambda y: y - cls._y_offset * push_factor,
                    lambda z: z + cls.env_cls.switch_height *
                    push_above_factor, "closed", option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveBack", lambda y: y + cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
            ])
        options.add(SwitchFaucetOn)

        # SwitchFaucetOff
        option_type = [robot_type, faucet_type]
        params_space = Box(0, 1, (0, ))
        SwitchFaucetOff = utils.LinearChainParameterizedOption(
            "SwitchFaucetOff", [
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol_small),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToAboveAndInFrontOfSwitch",
                    lambda y: y - cls._y_offset * push_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToInFrontOfSwitch",
                    lambda y: y + cls._y_offset * behind_factor, lambda z: z +
                    cls.env_cls.switch_height * push_above_factor, "closed",
                    option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "PushSwitchOff", lambda y: y + cls._y_offset * push_factor,
                    lambda z: z + cls.env_cls.switch_height *
                    push_above_factor, "closed", option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveBack", lambda y: y + cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
            ])
        options.add(SwitchFaucetOff)

        # SwitchBurnerOn
        option_type = [robot_type, burner_type]
        params_space = Box(0, 1, (0, ))
        SwitchBurnerOn = utils.LinearChainParameterizedOption(
            "SwitchBurnerOn", [
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol_small),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToAboveAndBehindSwitch",
                    lambda y: y - cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToBehindSwitch",
                    lambda y: y - cls._y_offset * behind_factor, lambda z: z +
                    cls.env_cls.switch_height * push_above_factor, "closed",
                    option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "PushSwitchOn", lambda y: y - cls._y_offset * push_factor,
                    lambda z: z + cls.env_cls.switch_height *
                    push_above_factor, "closed", option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveBack", lambda y: y + cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
            ])
        options.add(SwitchBurnerOn)

        # SwitchBurnerOff
        option_type = [robot_type, burner_type]
        params_space = Box(0, 1, (0, ))
        SwitchBurnerOff = utils.LinearChainParameterizedOption(
            "SwitchBurnerOff", [
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol_small),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToAboveAndInFrontOfSwitch",
                    lambda y: y - cls._y_offset * push_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveToInFrontOfSwitch",
                    lambda y: y + cls._y_offset * behind_factor, lambda z: z +
                    cls.env_cls.switch_height * push_above_factor, "closed",
                    option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "PushSwitchOff", lambda y: y + cls._y_offset * push_factor,
                    lambda z: z + cls.env_cls.switch_height *
                    push_above_factor, "closed", option_type, params_space),
                cls._create_boil_move_to_push_switch_option(
                    "MoveBack", lambda y: y + cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, "closed", option_type,
                    params_space),
            ])
        options.add(SwitchBurnerOff)

        # PickJug
        option_types = [robot_type, jug_type]
        params_space = Box(0, 1, (0, ))

        def _PickJug_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            holds = Holding.holds(state, [robot, jug])
            return holds

        PickJug = utils.LinearChainParameterizedOption(
            "PickJug",
            [
                # Move to far above the jug which we will grasp.
                cls._create_boil_move_to_above_jug_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: cls._hand_empty_move_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol),
                # Move down to grasp.
                cls._create_boil_move_to_above_jug_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda jug_z: (jug_z),
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Close fingers.
                create_change_fingers_option(
                    pybullet_robot,
                    "CloseFingers",
                    option_types,
                    params_space,
                    close_fingers_func,
                    CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol_small / 5,
                    terminal=_PickJug_terminal),
                # # Move down to grasp.
                # cls._create_boil_move_to_above_jug_option(
                #     name="MoveEndEffectorToGrasp",
                #     z_func=lambda jug_z: (jug_z),
                #     finger_status="closed",
                #     pybullet_robot=pybullet_robot,
                #     option_types=option_types,
                #     params_space=params_space),
                # # Move back up.
                # cls._create_boil_move_to_above_jug_option(
                #     name="MoveEndEffectorBackUp",
                #     z_func=lambda _: cls._transport_z,
                #     finger_status="closed",
                #     pybullet_robot=pybullet_robot,
                #     option_types=option_types,
                #     params_space=params_space),
            ])
        options.add(PickJug)

        # PlaceJugFaucet
        option_types = [robot_type, faucet_type]
        params_space = Box(0, 1, (0, ))
        PlaceUnderFaucet = utils.LinearChainParameterizedOption(
            "PlaceUnderFaucet",
            [
                # First move to the air to avoid collision.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToAir",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    under_faucet=True,
                    # move_to_initial_pos=True,
                    move_directly_up=True
                    ),
                # Move to center in the space in the air to avoid collision.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToAir",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    under_faucet=True,
                    move_to_initial_pos=True,
                    # move_directly_up=True
                    ),
                # Move to above the burner on which we will stack.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    under_faucet=True),
                # Move down to place.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda _: cls.env_cls.table_height + \
                                        cls.env_cls.jug_handle_height,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    under_faucet=True),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol),
                # Move back up.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: cls._hand_empty_move_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    under_faucet=True),
            ])
        options.add(PlaceUnderFaucet)

        # PlaceJugBurner
        option_types = [robot_type, burner_type]
        params_space = Box(0, 1, (0, ))
        PlaceOnBurner = utils.LinearChainParameterizedOption(
            "PlaceOnBurner",
            [
                # Move to above the burner on which we will stack.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    move_to_initial_pos=True),
                # Move to directly above to avoid collision.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Move down to place.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda _: cls.env_cls.table_height + \
                                        cls.env_cls.jug_handle_height,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol),
                # Move back up.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: cls._hand_empty_move_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ])
        options.add(PlaceOnBurner)

        # PlaceOutsideBurnerAndFaucet
        option_types = [robot_type]
        params_space = Box(0, 1, (0, ))
        PlaceOutsideBurnerAndFaucet = utils.LinearChainParameterizedOption(
            "PlaceOutsideBurnerAndFaucet",
            [
                # Move to above the burner on which we will stack.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: cls._transport_z,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space,
                    move_to_initial_pos=True),
                # Move down to place.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda _: cls.env_cls.table_height + \
                                        cls.env_cls.jug_handle_height,
                    finger_status="closed",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
                # Open fingers.
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletBoilEnv.grasp_tol),
                # Move back up.
                cls._create_boil_move_to_above_placing_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: cls._hand_empty_move_z,
                    finger_status="open",
                    pybullet_robot=pybullet_robot,
                    option_types=option_types,
                    params_space=params_space),
            ])
        options.add(PlaceOutsideBurnerAndFaucet)

        # Noop
        option_types = [robot_type]
        params_space = Box(0, 1, (0, ))

        def _create_no_op_policy() -> ParameterizedPolicy:
            nonlocal action_space

            def _policy(state: State, memory: Dict, objects: Sequence[Object],
                        params: Array) -> Action:
                del memory, params
                robot = objects[0]
                nonlocal action_space
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
                pybullet_state = cast(utils.PyBulletState, state)
                joint_positions = pybullet_state.joint_positions.copy()
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
            params_space=params_space,
            policy=_create_no_op_policy(),
            initiable=lambda _1, _2, _3, _4: True,
            terminal=lambda _1, _2, _3, _4: False,
        )
        options.add(NoOp)

        if CFG.boil_goal == "task_completed":
            # Declare Complete
            option_types = [robot_type]
            params_space = Box(0, 1, (0, ))
            DeclareComplete = utils.LinearChainParameterizedOption(
                "DeclareComplete",
                [
                    # Open fingers.
                    create_change_fingers_option(
                        pybullet_robot, "OpenFingers", option_types,
                        params_space, open_fingers_func,
                        CFG.pybullet_max_vel_norm, PyBulletBoilEnv.grasp_tol),
                    # Move to initial position.
                    cls._create_boil_move_to_init_option(
                        name="MoveEndEffectorToInit",
                        finger_status="open",
                        pybullet_robot=pybullet_robot,
                        option_types=option_types,
                        params_space=params_space),
                ])
            options.add(DeclareComplete)

        return options

    # ------------------------------------------------------------------
    # Skill-factory path
    # ------------------------------------------------------------------

    @classmethod
    def _get_options_skill_factories(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:
        """Skill-factory-based option implementations for the boil env."""
        del env_name, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletBoilEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        switch_type = types["switch"]
        jug_type = types["jug"]
        burner_type = types["burner"]
        faucet_type = types["faucet"]

        env_cls = cls.env_cls

        simulator = env_cls(use_gui=False) \
            if CFG.skill_phase_use_motion_planning else None
        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=PyBulletBoilEnv._fingers_state_to_joint,
            robot_init_tilt=PyBulletBoilEnv.robot_init_tilt,
            robot_init_wrist=PyBulletBoilEnv.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=cls._transport_z,
            simulator=simulator,
        )

        # ---------------------------------------------------------------
        # Helper: find the switch object associated with a faucet/burner.
        # The env sets obj.switch_id in _reset_state.
        # ---------------------------------------------------------------
        def _get_switch_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            config: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, config
            _, obj = objects
            switch = next((s for s in state.get_objects(switch_type)
                           if s.id == obj.switch_id), None)
            if switch is None:
                raise utils.OptionExecutionFailure(
                    f"No switch found for {obj} (switch_id={obj.switch_id})")
            return (state.get(switch, "x"), state.get(switch, "y"),
                    state.get(switch, "z"), state.get(switch, "rot"))

        # Adjust yaw to match standard facing convention:
        # standard facing = (sin(yaw), cos(yaw)),
        # switch push_dir = (cos(rot), sin(rot)) → yaw = π/2 − rot
        def _get_switch_on_pose(state, objects, params, cfg):
            x, y, z, rot = _get_switch_pose(state, objects, params, cfg)
            return x, y, z, np.pi / 2 - rot

        def _get_switch_off_pose(state, objects, params, cfg):
            x, y, z, rot = _get_switch_pose(state, objects, params, cfg)
            return x, y, z, -np.pi / 2 - rot

        _push_transport_z = cls._hand_empty_move_z
        push_config = replace(config, transport_z=_push_transport_z)

        # ---------------------------------------------------------------
        # PickJug: grasp at handle position with is_held terminal.
        # ---------------------------------------------------------------
        def _get_jug_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            config: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, config
            _, jug = objects
            rot = state.get(jug, "rot")
            gx = (state.get(jug, "x") +
                  np.cos(rot) * env_cls.jug_handle_offset)
            gy = (state.get(jug, "y") +
                  np.sin(rot) * env_cls.jug_handle_offset)
            gz = env_cls.table_height + env_cls.jug_handle_height
            return gx, gy, gz, rot

        PickJug = create_pick_skill(
            name="PickJug",
            types=[robot_type, jug_type],
            config=config,
            get_target_pose_fn=_get_jug_pose,
        )

        # ---------------------------------------------------------------
        # SwitchFaucetOn / SwitchFaucetOff (take [robot, faucet] objects)
        # ---------------------------------------------------------------
        SwitchFaucetOn = create_push_skill(
            name="SwitchFaucetOn",
            types=[robot_type, faucet_type],
            config=push_config,
            get_target_pose_fn=_get_switch_on_pose,
        )
        SwitchFaucetOff = create_push_skill(
            name="SwitchFaucetOff",
            types=[robot_type, faucet_type],
            config=push_config,
            get_target_pose_fn=_get_switch_off_pose,
        )

        # ---------------------------------------------------------------
        # SwitchBurnerOn / SwitchBurnerOff (take [robot, burner] objects)
        # ---------------------------------------------------------------
        SwitchBurnerOn = create_push_skill(
            name="SwitchBurnerOn",
            types=[robot_type, burner_type],
            config=push_config,
            get_target_pose_fn=_get_switch_on_pose,
        )
        SwitchBurnerOff = create_push_skill(
            name="SwitchBurnerOff",
            types=[robot_type, burner_type],
            config=push_config,
            get_target_pose_fn=_get_switch_off_pose,
        )

        # ---------------------------------------------------------------
        # Place options
        # ---------------------------------------------------------------
        PlaceUnderFaucet = create_place_skill(
            name="PlaceUnderFaucet",
            types=[robot_type, faucet_type],
            config=config,
        )

        PlaceOnBurner = create_place_skill(
            name="PlaceOnBurner",
            types=[robot_type, burner_type],
            config=config,
        )

        PlaceOutsideBurnerAndFaucet = create_place_skill(
            name="PlaceOutsideBurnerAndFaucet",
            types=[robot_type],
            config=config,
        )

        # ---------------------------------------------------------------
        # NoOp
        # ---------------------------------------------------------------
        NoOp = create_wait_option("NoOp", config, robot_type)

        options: Set[ParameterizedOption] = {
            PickJug, SwitchFaucetOn, SwitchFaucetOff, SwitchBurnerOn,
            SwitchBurnerOff, PlaceUnderFaucet, PlaceOnBurner,
            PlaceOutsideBurnerAndFaucet, NoOp
        }

        if CFG.boil_goal == "task_completed":
            # DeclareComplete: open fingers + move to initial robot position.
            # Reuse the legacy helper since it depends on the pybullet_robot.
            _, pybullet_robot2, _ = \
                PyBulletBoilEnv.initialize_pybullet(using_gui=False)

            def _open_func(state: State, objects: Sequence[Object],
                           params: Array) -> Tuple[float, float]:
                del objects, params
                robot_obj2, = state.get_objects(robot_type)
                current = PyBulletBoilEnv._fingers_state_to_joint(
                    pybullet_robot2, state.get(robot_obj2, "fingers"))
                return current, pybullet_robot2.open_fingers

            DeclareComplete = utils.LinearChainParameterizedOption(
                "DeclareComplete", [
                    create_change_fingers_option(
                        pybullet_robot2, "OpenFingers", [robot_type],
                        Box(0, 1, (0, )), _open_func,
                        CFG.pybullet_max_vel_norm, PyBulletBoilEnv.grasp_tol),
                    cls._create_boil_move_to_init_option(
                        name="MoveEndEffectorToInit",
                        finger_status="open",
                        pybullet_robot=pybullet_robot2,
                        option_types=[robot_type],
                        params_space=Box(0, 1, (0, ))),
                ])
            options.add(DeclareComplete)

        return options

    @classmethod
    def _create_boil_move_to_above_placing_option(
            cls,
            name: str,
            z_func: Callable[[float], float],
            finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot,
            option_types: List[Type],
            params_space: Box,
            under_faucet: bool = False,
            move_to_initial_pos: bool = False,
            move_directly_up: bool = False) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        burner argument.

        The parameter z_func maps the burner's z position to the target
        z position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            if len(objects) == 2:
                robot, burner = objects
                # Current
                current_position = (state.get(robot,
                                              "x"), state.get(robot, "y"),
                                    state.get(robot, "z"))
                ee_orn = p.getQuaternionFromEuler(
                    [0, state.get(robot, "tilt"),
                     state.get(robot, "wrist")])
                current_pose = Pose(current_position, ee_orn)
                # Target
                target_x = state.get(burner, "x")
                target_y = state.get(burner,
                                     "y") - cls.env_cls.jug_handle_offset
                if under_faucet:
                    target_y -= cls.env_cls.faucet_x_len
            else:
                robot, = objects
                # Current
                current_position = (state.get(robot,
                                              "x"), state.get(robot, "y"),
                                    state.get(robot, "z"))
                ee_orn = p.getQuaternionFromEuler(
                    [0, state.get(robot, "tilt"),
                     state.get(robot, "wrist")])
                current_pose = Pose(current_position, ee_orn)
                target_x = cls.env_cls.x_mid - 0.15
                target_y = cls.env_cls.y_mid + 0.10

            if move_to_initial_pos:
                target_position = (cls.env_cls.robot_init_x,
                                   cls.env_cls.robot_init_y,
                                   cls.env_cls.robot_init_z - 0.1)
            elif move_directly_up:
                target_position = (current_position[0], current_position[1],
                                   cls.env_cls.robot_init_z - 0.1)
            else:
                if len(objects) == 2:
                    target_position = (target_x, target_y,
                                       z_func(state.get(burner, "z")))
                else:
                    target_position = (target_x, target_y,
                                       z_func(cls.env_cls.table_height))

            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, cls.env_cls.robot_init_wrist])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)

    @classmethod
    def _create_boil_move_to_above_jug_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        jug argument.

        The parameter z_func maps the jug's z position to the target z
        position.
        """
        home_orn = PyBulletBoilEnv.get_robot_ee_home_orn()

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, jug = objects
            # Current
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            # Target
            rot = state.get(jug, "rot")
            target_x = state.get(jug, "x") + np.cos(rot) * \
                            cls.env_cls.jug_handle_offset
            target_y = state.get(jug, "y") + np.sin(rot) * \
                            cls.env_cls.jug_handle_offset
            target_z = z_func(cls.env_cls.jug_handle_height +
                              cls.env_cls.table_height)
            target_position = (target_x, target_y, target_z)
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt,
                 state.get(jug, "rot")])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)

    @classmethod
    def _create_boil_move_to_push_switch_option(
            cls, name: str, y_func: Callable[[float],
                                             float], z_func: Callable[[float],
                                                                      float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Create a move-to-pose option for the switch environment."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, obj = objects
            switch = next((s
                           for s in state.get_objects(cls.env_cls._switch_type)
                           if s.id == obj.switch_id), None)
            assert switch is not None
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (y_func(state.get(switch,
                                                "x")), state.get(switch, "y"),
                               z_func(state.get(switch, "z")))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, 0])
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
    def _create_boil_move_to_init_option(
            cls, name: str, finger_status: str,
            pybullet_robot: SingleArmPyBulletRobot, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to the initial position."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (cls.env_cls.robot_init_x,
                               cls.env_cls.robot_init_y,
                               cls.env_cls.robot_init_z)
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, cls.env_cls.robot_init_wrist])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            pybullet_robot,
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)
