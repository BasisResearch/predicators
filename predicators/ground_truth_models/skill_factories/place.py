"""Place skill: a multi-phase place controller for PyBullet environments."""

from typing import Callable, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.pybullet_helpers.geometry import Pose
from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_place_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_placement_pose_fn: Callable[
        [State, Sequence[Object], Array, SkillConfig], Tuple[float, float,
                                                             float, float]],
    transport_z: float,
    drop_z: float,
) -> ParameterizedOption:
    """Create a place skill.

    Args:
        name: Option name (e.g. "Place").
        types: Object types for the option signature.
        params_space: Parameter space for the option.
        config: Skill configuration with robot and tolerances.
        get_placement_pose_fn: Computes (x, y, z, yaw) for the placement
            target from (state, objects, params, config).
        transport_z: Z height for safe transport above placement.
        drop_z: Z height at which to release the object.

    Returns:
        A ParameterizedOption implementing the place skill.
    """

    def _open_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        del params  # unused
        robot_obj = objects[0]
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(robot_obj, "fingers"))
        target = cfg.open_fingers_joint - 0.01
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
        tx, ty, _, tyaw = get_placement_pose_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, tyaw])
        target_pose = Pose((tx, ty, transport_z), target_orn)
        return current_pose, target_pose, "closed"

    def _descend_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        tx, ty, _, tyaw = get_placement_pose_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, tyaw])
        target_pose = Pose((tx, ty, drop_z), target_orn)
        return current_pose, target_pose, "closed"

    def _retreat_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        current_pose = _get_current_pose(state, objects)
        tx, ty, _, tyaw = get_placement_pose_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, tyaw])
        target_pose = Pose((tx, ty, transport_z), target_orn)
        return current_pose, target_pose, "open"

    phases = [
        # Phase 0: Move above placement
        Phase(
            name="MoveAbove",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_move_above_target,
        ),
        # Phase 1: Descend to drop height
        Phase(
            name="Descend",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_descend_target,
        ),
        # Phase 2: Open fingers to release
        Phase(
            name="OpenFingers",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_open_fingers_target,
        ),
        # Phase 3: Retreat upward
        Phase(
            name="Retreat",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_retreat_target,
        ),
    ]

    return PhaseSkill(name, types, params_space, config, phases).build()
