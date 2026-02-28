"""Pick skill factory: creates a multi-phase pick-and-lift controller.

This module provides ``create_pick_skill``, which builds a
``ParameterizedOption`` that picks up an object by:

  1. Moving above the object at ``transport_z``.
  2. Descending to the grasp height (object z + ``grasp_z_offset``).
  3. Closing the gripper.
  4. Lifting back to ``transport_z``.

The caller supplies a single callback ``get_target_pose_fn`` that extracts
the object's ``(x, y, z, yaw)`` from the current state.  All environment-
specific logic lives in this callback; the factory handles motion planning,
IK, and phase sequencing.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pick_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
    )

    def _get_jug_pose(state, objects, params, config):
        _, jug = objects
        return (state.get(jug, "x"), state.get(jug, "y"),
                state.get(jug, "z"), state.get(jug, "rot"))

    PickJug = create_pick_skill(
        name="PickJug",
        types=[robot_type, jug_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_jug_pose,
        transport_z=0.8,
    )
"""

from typing import Callable, Optional, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_pick_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    transport_z: float,
    grasp_z_offset: float = 0.0,
    grasp_terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None,
) -> ParameterizedOption:
    """Create a multi-phase pick skill that grasps and lifts an object.

    Phases:
        0. **MoveAbove** -- Move end-effector above the object at
           ``transport_z``, with fingers closed.
        1. **Descend** -- Lower to the object's z + ``grasp_z_offset``,
           with fingers open.
        2. **Grasp** -- Close fingers until contact (or custom terminal).
        3. **Lift** -- Lift back to ``transport_z``, with fingers closed.

    Args:
        name: Option name used for logging and matching (e.g. "Pick",
            "PickJug").
        types: Ordered object types for the option signature.  The first
            element **must** be the robot type.
            Example: ``[robot_type, obj_type]``.
        params_space: Continuous parameter space.  Use ``Box(0, 1, (0,))``
            for zero-dimensional (no sampled parameters).
        config: Shared skill configuration.  See ``SkillConfig``.
        get_target_pose_fn: Callback that returns the grasp target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
            ``(x, y)`` is the horizontal grasp position, ``z`` is the
            object's height (before ``grasp_z_offset`` is applied), and
            ``yaw`` is the EE wrist rotation for approach.
        transport_z: Safe Z height for moving above obstacles before and
            after grasping.
        grasp_z_offset: Added to the z returned by ``get_target_pose_fn``
            to compute the actual descend height.  Default ``0.0``.
        grasp_terminal_fn: Optional override for the Grasp phase terminal.
            Signature: ``(state, objects, params, config) -> bool``.
            If ``None``, terminates when finger joints reach the closed
            target.

    Returns:
        A ``ParameterizedOption`` implementing the pick skill.

    Example::

        def _get_cup_pose(state, objects, params, config):
            _, cup = objects
            return (state.get(cup, "x"), state.get(cup, "y"),
                    state.get(cup, "z"), 0.0)

        pick_cup = create_pick_skill(
            name="PickCup",
            types=[robot_type, cup_type],
            params_space=Box(0, 1, (0,)),
            config=config,
            get_target_pose_fn=_get_cup_pose,
            transport_z=0.8,
            grasp_z_offset=0.02,
        )
    """

    def _close_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        del params  # unused
        robot_obj = objects[0]
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(robot_obj, "fingers"))
        target = cfg.closed_fingers_joint - 0.01
        return current, target

    def _get_current_pose(
        state: State,
        objects: Sequence[Object],
    ) -> Pose:
        robot_obj = objects[0]
        current_position = (state.get(robot_obj,
                                      "x"), state.get(robot_obj, "y"),
                            state.get(robot_obj, "z"))
        ee_orn = p.getQuaternionFromEuler(
            [0, state.get(robot_obj, "tilt"),
             state.get(robot_obj, "wrist")])
        return Pose(current_position, ee_orn)

    def _move_above_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        obj_x, obj_y, _, obj_yaw = get_target_pose_fn(state, objects, params,
                                                       cfg)
        target_orn = p.getQuaternionFromEuler(
            [0, cfg.robot_init_tilt, obj_yaw])
        target_pose = Pose((obj_x, obj_y, transport_z), target_orn)
        return current_pose, target_pose, "closed"

    def _descend_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        obj_x, obj_y, obj_z, obj_yaw = get_target_pose_fn(
            state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler(
            [0, cfg.robot_init_tilt, obj_yaw])
        target_pose = Pose((obj_x, obj_y, obj_z + grasp_z_offset), target_orn)
        return current_pose, target_pose, "open"

    def _lift_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        obj_x, obj_y, _, obj_yaw = get_target_pose_fn(state, objects, params,
                                                       cfg)
        target_orn = p.getQuaternionFromEuler(
            [0, cfg.robot_init_tilt, obj_yaw])
        target_pose = Pose((obj_x, obj_y, transport_z), target_orn)
        return current_pose, target_pose, "closed"

    phases = [
        # Phase 0: Move above object
        Phase(
            name="MoveAbove",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_move_above_target,
        ),
        # Phase 1: Descend to grasp height
        Phase(
            name="Descend",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_descend_target,
        ),
        # Phase 2: Close fingers to grasp
        Phase(
            name="Grasp",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_close_fingers_target,
            terminal_fn=grasp_terminal_fn,
        ),
        # Phase 3: Lift to transport height
        Phase(
            name="Lift",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_lift_target,
        ),
    ]

    return PhaseSkill(name, types, params_space, config, phases).build()
