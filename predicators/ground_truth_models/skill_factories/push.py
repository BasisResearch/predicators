"""Push skill: a multi-phase push controller for PyBullet environments."""

from typing import Callable, List, Sequence, Tuple

from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig
from predicators.ground_truth_models.skill_factories.move_to_pose import \
    make_move_to_pose_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_push_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: Callable[[State, Sequence[Object], Array],
                                 Tuple[float, float, float, float]],
    waypoints_fn: Callable[[float, float, float, float, SkillConfig],
                           List[Tuple[float, float, float, float, str]]],
) -> ParameterizedOption:
    """Create a push skill from target extraction and waypoint functions.

    Args:
        name: Option name (e.g. "Push").
        types: Object types for the option signature.
        params_space: Parameter space for the option.
        config: Skill configuration with robot and tolerances.
        get_target_pose_fn: Extracts (x, y, z, yaw) of the push target
            from (state, objects, params).
        waypoints_fn: Computes a list of (x, y, z, yaw, finger_status)
            waypoints from (target_x, target_y, target_z, target_yaw,
            config). The robot will move through these sequentially.

    Returns:
        A ParameterizedOption implementing the push skill.
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

    def _make_waypoint_position_fn(
        waypoint_idx: int,
    ) -> Callable[[State, Sequence[Object], Array, SkillConfig], Tuple[
            float, float, float, float], ]:
        """Create a get_target_position_fn for a specific waypoint."""

        def _get_target(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, yaw = get_target_pose_fn(state, objects, params)
            waypoints = waypoints_fn(x, y, z, yaw, cfg)
            wx, wy, wz, wyaw, _ = waypoints[waypoint_idx]
            return wx, wy, wz, wyaw

        return _get_target

    # Determine waypoint count and finger statuses from dummy call.
    dummy_waypoints = waypoints_fn(0.0, 0.0, 0.0, 0.0, config)
    num_waypoints = len(dummy_waypoints)

    phases: List[Phase] = []

    # Phase 0: Close fingers.
    phases.append(
        Phase(
            name="CloseFingers",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_close_fingers_target,
        ))

    # Phases 1..N: Move through waypoints.
    for i in range(num_waypoints):
        _, _, _, _, finger_status = dummy_waypoints[i]
        phases.append(
            make_move_to_pose_phase(
                name=f"Waypoint_{i}",
                get_target_position_fn=_make_waypoint_position_fn(i),
                finger_status=finger_status,
            ))

    # Phase N+1: Open fingers.
    phases.append(
        Phase(
            name="OpenFingers",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_open_fingers_target,
        ))

    return PhaseSkill(name, types, params_space, config, phases).build()
