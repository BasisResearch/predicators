"""Pick skill: a multi-phase pick controller for PyBullet environments."""

from typing import Callable, Optional, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.pybullet_helpers.geometry import Pose
from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig
from predicators.structs import Array, Object, State, Type


def create_pick_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_object_pose_fn: Callable[[State, Sequence[Object], Array],
                                 Tuple[float, float, float, float]],
    transport_z: float,
    grasp_z_offset: float = 0.0,
    grasp_terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None,
) -> PhaseSkill:
    """Create a pick skill.

    Args:
        name: Option name (e.g. "Pick").
        types: Object types for the option signature.
        params_space: Parameter space for the option.
        config: Skill configuration with robot and tolerances.
        get_object_pose_fn: Extracts (x, y, z, yaw) of the object to pick
            from (state, objects, params).
        transport_z: Z height for safe transport above objects.
        grasp_z_offset: Offset added to object z for grasp height.
        grasp_terminal_fn: Optional custom terminal for the grasp phase.
            Signature: (state, objects, params, config) -> bool.

    Returns:
        A PhaseSkill whose .build() produces a ParameterizedOption.
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
        obj_x, obj_y, _, obj_yaw = get_object_pose_fn(state, objects, params)
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
        obj_x, obj_y, obj_z, obj_yaw = get_object_pose_fn(
            state, objects, params)
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
        obj_x, obj_y, _, obj_yaw = get_object_pose_fn(state, objects, params)
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

    return PhaseSkill(name, types, params_space, config, phases)
