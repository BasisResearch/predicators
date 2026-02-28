"""Pour skill: a multi-phase pour controller for PyBullet environments."""

from typing import Callable, Optional, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.pybullet_helpers.geometry import Pose
from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_pour_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_pour_position_fn: Callable[
        [State, Sequence[Object], Array, SkillConfig], Tuple[float, float,
                                                             float, float]],
    pour_tilt: float,
    transport_z: float,
    tilt_terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None,
) -> ParameterizedOption:
    """Create a pour skill.

    A multi-phase skill for pouring liquid from a held container (e.g. jug)
    into a target (e.g. cup). The robot moves above the pour position,
    descends, then tilts the end-effector to pour.

    Args:
        name: Option name (e.g. "Pour").
        types: Object types for the option signature.
        params_space: Parameter space for the option.
        config: Skill configuration with robot and tolerances.
        get_pour_position_fn: Computes the robot EE target (x, y, z, wrist)
            for the pour position from (state, objects, params, config).
            Each environment computes this from jug/cup offsets.
        pour_tilt: Target EE tilt (pitch) angle for pouring.
        transport_z: Safe Z height for transit above obstacles.
        tilt_terminal_fn: Optional custom terminal for the tilt phase.
            Signature: (state, objects, params, config) -> bool.
            If None, the tilt phase terminates when the EE reaches the
            target pose (default distance-based terminal).

    Returns:
        A ParameterizedOption implementing the pour skill.
    """

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
        tx, ty, _, twrist = get_pour_position_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler(
            [0, cfg.robot_init_tilt, twrist])
        target_pose = Pose((tx, ty, transport_z), target_orn)
        return current_pose, target_pose, "closed"

    def _descend_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        tx, ty, tz, twrist = get_pour_position_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler(
            [0, cfg.robot_init_tilt, twrist])
        target_pose = Pose((tx, ty, tz), target_orn)
        return current_pose, target_pose, "closed"

    def _tilt_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        tx, ty, tz, twrist = get_pour_position_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, pour_tilt, twrist])
        target_pose = Pose((tx, ty, tz), target_orn)
        return current_pose, target_pose, "closed"

    phases = [
        # Phase 0: Move above pour position at normal tilt
        Phase(
            name="MoveAbovePour",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_move_above_target,
        ),
        # Phase 1: Descend to pour height at normal tilt
        Phase(
            name="DescendToPour",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_descend_target,
        ),
        # Phase 2: Tilt EE to pour angle (incremental IK for fine control)
        Phase(
            name="Tilt",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_tilt_target,
            terminal_fn=tilt_terminal_fn,
            use_motion_planning=False,
        ),
    ]

    return PhaseSkill(name, types, params_space, config, phases).build()
