"""Ground-truth options for the coffee environment."""

from functools import lru_cache
from typing import ClassVar, Dict, Sequence, Set
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.envs.pybullet_grow import PyBulletGrowEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.coffee.options import \
    PyBulletCoffeeGroundTruthOptionFactory
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletGrowGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletGrowEnv]] = PyBulletGrowEnv
    pick_policy_tol: ClassVar[float] = 1e-3
    pour_policy_tol: ClassVar[float] = 1e-3/2
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        _, pybullet_robot, _ = \
            PyBulletGrowEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        # Predicates
        Holding = predicates["Holding"]
        Grown = predicates["Grown"]
        JugAboveCup = predicates["JugAboveCup"]
        HandTilted = predicates["HandTilted"]

        # PickJug
        def _PickJug_terminal(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            del memory, params  # unused
            robot, jug = objects
            holds = Holding.holds(state, [robot, jug])
            return holds

        PickJug = ParameterizedOption(
            name="PickJug",
            types=[robot_type, jug_type],
            params_space=Box(0, 1, (0, )),
            policy=PyBulletCoffeeGroundTruthOptionFactory.  # pylint: disable=protected-access
            _create_pick_jug_policy(),
            # policy=cls._create_pick_jug_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickJug_terminal)

        # Pour
        def _Pour_terminal(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            robot, jug, cup = objects
            if CFG.grow_weak_pour_terminate_condition:
                if not Holding.holds(state, [robot, jug]):
                    return False
                jug_x = state.get(jug, "x")
                jug_y = state.get(jug, "y")
                jug_z = state.get(robot, "z") -\
                    PyBulletCoffeeEnv.jug_handle_height()
                jug_pos = (jug_x, jug_y, jug_z)
                pour_pos = PyBulletCoffeeEnv._get_pour_position(state, cup)
                sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
                jug_above_cup = sq_dist_to_pour < cls.env_cls.pour_pos_tol/\
                                            (cls.env_cls.pour_pos_tol_factor*2)
                
                cond = jug_above_cup and HandTilted.holds(state, [robot])
            else:
                cond = Grown.holds(state, [cup])
            return cond

        Pour = ParameterizedOption(
            "Pour",
            [robot_type, jug_type, cup_type],
            params_space=Box(0, 1, (0, )),
            policy=PyBulletCoffeeGroundTruthOptionFactory.  # pylint: disable=protected-access
                _create_pour_policy(pour_policy_tol=cls.pour_policy_tol),
            initiable=lambda s, m, o, p: True,
            terminal=_Pour_terminal)

        # Place
        def _Place_terminal(state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
            del memory, params
            robot, jug = objects
            return not Holding.holds(state, [robot, jug])

        if CFG.grow_place_option_no_sampler:
            params_space = Box(0, 1, (0, ))
        else:
            params_space = Box(0, 1, (2, ))
        Place = ParameterizedOption("Place", [robot_type, jug_type],
                                    params_space=params_space,
                                    policy=cls._crete_place_policy(),
                                    initiable=lambda s, m, o, p: True,
                                    terminal=_Place_terminal)

        # Noop
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
            params_space=params_space,
            policy=_create_no_op_policy(),
            initiable=lambda _1, _2, _3, _4: True,
            terminal=lambda _1, _2, _3, _4: False,
        )

        return {PickJug, Pour, Place, NoOp}

    @classmethod
    def _crete_place_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory
            robot, jug = objects

            # Get the current robot position.
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            tilt = state.get(robot, "tilt")
            wrist = state.get(robot, "wrist")
            robot_pos = (x, y, z)

            # Get the difference between the jug location and the target.
            # Use the jug position as the origin.
            jx = state.get(jug, "x")
            jy = state.get(jug, "y")
            jz = state.get(jug, "z")
            # jz = cls.env_cls.z_lb + cls.env_cls.jug_height()
            current_jug_pos = (jx, jy, jz)
            if CFG.grow_place_option_no_sampler:
                target_jug_pos = (jug.init_x, jug.init_y, jug.init_z)
            else:
                x_norm, y_norm = params
                target_jug_pos = (
                    cls.env_cls.x_lb +
                    (cls.env_cls.x_ub - cls.env_cls.x_lb) * x_norm,
                    cls.env_cls.y_lb +
                    (cls.env_cls.y_ub - cls.env_cls.y_lb) * y_norm,
                    cls.env_cls.z_lb + cls.env_cls.jug_height / 2)

            dtilt = cls.env_cls.robot_init_tilt - tilt
            dwrist = cls.env_cls.robot_init_wrist - wrist
            dx, dy, dz = np.subtract(target_jug_pos, current_jug_pos)

            # Get the target robot position.
            target_robot_pos = (x + dx, y + dy, z + dz)
            # If close enough, place.
            sq_dist_to_place = np.sum(
                np.subtract(robot_pos, target_robot_pos)**2)
            if sq_dist_to_place < cls.env_cls.place_jug_tol:
                return PyBulletCoffeeGroundTruthOptionFactory._get_place_action(  # pylint: disable=protected-access
                    state)

            # only move down if it has arrived at target x, y
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                # print("Moving down to place jug")
                return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(  # pylint: disable=protected-access
                    state,
                    target_robot_pos,
                    robot_pos,
                    finger_status="closed",
                    dtilt=dtilt,
                    dwrist=dwrist,
                )

            target_robot_pos = (x + dx, y + dy, z)
            # print("Moving to place jug")
            return PyBulletCoffeeGroundTruthOptionFactory._get_move_action(  # pylint: disable=protected-access
                state,
                target_robot_pos,
                robot_pos,
                finger_status="closed",
                dtilt=dtilt,
                dwrist=dwrist,
            )

        return policy
