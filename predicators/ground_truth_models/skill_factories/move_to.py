"""MoveToTarget skill and reusable move-to-pose phase builder."""

from typing import Callable, Optional, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.pybullet_helpers.geometry import Pose
from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig
from predicators.structs import Array, Object, ParameterizedOption, State, Type

def create_move_to_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_position_fn: Callable[
        [State, Sequence[Object], Array, SkillConfig], Tuple[float, float,
                                                             float, float]],
) -> ParameterizedOption:
    """Create a single-phase move-to-pose skill.

    Preserves the current finger status (open/closed) from state.

    Args:
        name: Option name.
        types: Object types for the option signature.
        params_space: Parameter space for the option.
        config: Skill configuration with robot and tolerances.
        get_target_position_fn: Returns (x, y, z, yaw) for the target
            from (state, objects, params, config).

    Returns:
        A ParameterizedOption implementing the move-to-pose skill.
    """
    phase = make_move_to_phase(name, get_target_position_fn)
    return PhaseSkill(name, types, params_space, config, [phase]).build()

def _get_current_ee_pose(state: State, robot_obj: Object) -> Pose:
    """Extract current end-effector pose from state."""
    position = (state.get(robot_obj,
                          "x"), state.get(robot_obj,
                                          "y"), state.get(robot_obj, "z"))
    orientation = p.getQuaternionFromEuler(
        [0, state.get(robot_obj, "tilt"),
         state.get(robot_obj, "wrist")])
    return Pose(position, orientation)


def _get_finger_status(state: State, robot_obj: Object,
                       cfg: SkillConfig) -> str:
    """Infer 'open' or 'closed' from current finger state."""
    current_fingers = state.get(robot_obj, "fingers")
    finger_joint = cfg.fingers_state_to_joint(cfg.robot, current_fingers)
    if abs(finger_joint - cfg.open_fingers_joint) < \
            abs(finger_joint - cfg.closed_fingers_joint):
        return "open"
    return "closed"


def make_move_to_phase(
    name: str,
    get_target_position_fn: Callable[
        [State, Sequence[Object], Array, SkillConfig], Tuple[float, float,
                                                             float, float]],
    finger_status: Optional[str] = None,
) -> Phase:
    """Create a MOVE_TO_POSE phase.

    Args:
        name: Phase name.
        get_target_position_fn: Returns (x, y, z, yaw) for the target
            from (state, objects, params, config).
        finger_status: "open" or "closed". If None, preserves the current
            finger status from state.

    Returns:
        A Phase that can be included in a PhaseSkill.
    """

    def _target_fn(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        robot_obj = objects[0]
        current_pose = _get_current_ee_pose(state, robot_obj)

        tx, ty, tz, tyaw = get_target_position_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, tyaw])
        target_pose = Pose((tx, ty, tz), target_orn)

        if finger_status is not None:
            status = finger_status
        else:
            status = _get_finger_status(state, robot_obj, cfg)
        return current_pose, target_pose, status

    return Phase(
        name=name,
        action_type=PhaseAction.MOVE_TO_POSE,
        target_fn=_target_fn,
    )
