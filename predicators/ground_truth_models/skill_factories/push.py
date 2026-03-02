"""Push skill factory: creates a multi-phase push controller.

This module provides ``create_push_skill``, which builds a
``ParameterizedOption`` that pushes an object (e.g. a switch or domino) by:

  1. Closing the gripper.
  2. Moving through a caller-defined sequence of waypoints.
  3. Opening the gripper.

The caller supplies two callbacks:

- ``get_target_pose_fn`` extracts the push target ``(x, y, z, yaw)`` from
  the current state (e.g. the position of a switch or domino).
- ``waypoints_fn`` turns that target into a list of waypoints
  ``[(x, y, z, yaw, finger_status), ...]`` that define the approach,
  contact, and retreat trajectory.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_push_skill,
    )

    def _get_switch_pose(state, objects, params, config):
        _, switch = objects
        return (state.get(switch, "x"), state.get(switch, "y"),
                state.get(switch, "z"), state.get(switch, "yaw"))

    def _push_waypoints(tx, ty, tz, tyaw, config):
        # Approach from behind, push forward, then retreat upward.
        return [
            (tx - 0.05, ty, tz + 0.1, tyaw, "closed"),  # above & behind
            (tx - 0.05, ty, tz,       tyaw, "closed"),  # descend behind
            (tx + 0.05, ty, tz,       tyaw, "closed"),  # push through
            (tx,        ty, tz + 0.1, tyaw, "closed"),  # retreat up
        ]

    PushSwitch = create_push_skill(
        name="PushSwitch",
        types=[robot_type, switch_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_switch_pose,
        waypoints_fn=_push_waypoints,
    )
"""

from typing import Callable, List, Optional, Sequence, Tuple

from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_push_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    waypoints_fn: Callable[[float, float, float, float, SkillConfig],
                           List[Tuple[float, float, float, float, str]]],
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a multi-phase push skill from target and waypoint callbacks.

    Phases:
        0. **CloseFingers** -- Close the gripper before pushing.
        1..N. **Waypoint_0 .. Waypoint_N-1** -- Move through each waypoint
           returned by ``waypoints_fn``, in order.
        N+1. **OpenFingers** -- Open the gripper after pushing.

    The number of waypoint phases is determined at construction time by
    calling ``waypoints_fn`` with dummy arguments ``(0, 0, 0, 0, config)``.

    Args:
        name: Option name used for logging and matching (e.g. "Push",
            "SwitchOn").
        types: Ordered object types for the option signature.  The first
            element **must** be the robot type.
            Example: ``[robot_type, switch_type]``.
        params_space: Continuous parameter space.  Use ``Box(0, 1, (0,))``
            for zero-dimensional (no sampled parameters).
        config: Shared skill configuration.  See ``SkillConfig``.
        get_target_pose_fn: Callback that returns the push target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
            This is the pose of the object to be pushed (e.g. a switch
            or the first domino).
        waypoints_fn: Callback that computes a list of waypoints from
            ``(target_x, target_y, target_z, target_yaw, config)``.
            Each waypoint is ``(x, y, z, yaw, finger_status)`` where
            ``finger_status`` is ``"open"`` or ``"closed"``.  The robot
            moves through these waypoints sequentially.

    Returns:
        A ``ParameterizedOption`` implementing the push skill.

    Example::

        def _get_button_pose(state, objects, params, config):
            _, button = objects
            return (state.get(button, "x"), state.get(button, "y"),
                    state.get(button, "z"), 0.0)

        def _button_waypoints(tx, ty, tz, tyaw, config):
            return [
                (tx, ty - 0.05, tz + 0.05, tyaw, "closed"),
                (tx, ty - 0.05, tz,        tyaw, "closed"),
                (tx, ty + 0.05, tz,        tyaw, "closed"),
                (tx, ty,        tz + 0.10, tyaw, "closed"),
            ]

        push_button = create_push_skill(
            name="PushButton",
            types=[robot_type, button_type],
            params_space=Box(0, 1, (0,)),
            config=config,
            get_target_pose_fn=_get_button_pose,
            waypoints_fn=_button_waypoints,
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
        """Create a get_target_pose_fn for a specific waypoint."""

        def _get_target(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, yaw = get_target_pose_fn(state, objects, params, cfg)
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
            make_move_to_phase(
                name=f"Waypoint_{i}",
                get_target_pose_fn=_make_waypoint_position_fn(i),
                finger_status=finger_status,
            ))

    # Phase N+1: Open fingers.
    phases.append(
        Phase(
            name="OpenFingers",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_open_fingers_target,
        ))

    return PhaseSkill(name, types, params_space, config, phases,
                      params_description=params_description).build()
