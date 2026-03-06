"""Push skill factory: creates a multi-phase push controller.

This module provides ``create_push_skill``, which builds a
``ParameterizedOption`` that pushes an object (e.g. domino, switch, button)
using a standard 4-waypoint trajectory:

  1. Closing the gripper.
  2. Moving above & behind the target at ``transport_z``.
  3. Descending to contact height (target z + ``offset_z``).
  4. Pushing through the target (``push_through_frac * offset_x``
     past the target along its facing direction).
  5. Retreating to ``config.robot_home_pos``.
  6. Opening the gripper.

The "facing direction" is derived from the yaw returned by
``get_target_pose_fn`` as ``(sin(yaw), cos(yaw))``.  "Behind" means
opposite to the facing direction, i.e. the robot approaches from
behind and pushes forward through the target.

``config.robot_home_pos`` **must** be set — this is the ``(x, y, z)``
position the robot retreats to after the push.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_push_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
        robot_home_pos=(MyEnv.robot_init_x, MyEnv.robot_init_y,
                        MyEnv.robot_init_z),
    )

    def _get_domino_pose(state, objects, params, config):
        _, domino = objects
        return (state.get(domino, "x"), state.get(domino, "y"),
                state.get(domino, "z"), state.get(domino, "rot"))

    Push = create_push_skill(
        name="Push",
        types=[robot_type, domino_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_domino_pose,
        offset_x=0.05,
        offset_z=0.02,
        offset_rot=np.pi / 2,
        transport_z=0.65,
    )
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
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
    offset_x: float,
    offset_z: float,
    transport_z: float = 0.7,
    offset_rot: float = 0.0,
    push_through_frac: float = 0.25,
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a multi-phase push skill with a standard 4-waypoint trajectory.

    Phases:
        0. **CloseFingers** -- Close the gripper before approaching.
        1. **Waypoint_0** -- Move above & behind the target at
           ``transport_z``, offset by ``offset_x`` opposite the facing
           direction.
        2. **Waypoint_1** -- Descend to contact height
           (target z + ``offset_z``) at the same behind position.
        3. **Waypoint_2** -- Push forward through the target by
           ``offset_x * push_through_frac`` along the facing direction.
        4. **Waypoint_3** -- Retreat to ``config.robot_home_pos``.
        5. **OpenFingers** -- Open the gripper.

    The facing direction is ``(sin(yaw), cos(yaw))`` where ``yaw`` is
    the fourth element returned by ``get_target_pose_fn``.  The EE wrist
    rotation during the push is ``yaw + offset_rot``.

    Args:
        name: Option name used for logging and matching (e.g. "Push",
            "SwitchOn").
        types: Ordered object types for the option signature.  The first
            element **must** be the robot type.
            Example: ``[robot_type, domino_type]``.
        params_space: Continuous parameter space.  Use ``Box(0, 1, (0,))``
            for zero-dimensional (no sampled parameters).
        config: Shared skill configuration.  See ``SkillConfig``.
            ``config.robot_home_pos`` **must** be set to the
            ``(x, y, z)`` retreat position.
        get_target_pose_fn: Callback that returns the push target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
            ``(x, y)`` is the horizontal push target, ``z`` is the
            z position of the center of the object (before ``offset_z``
            is applied), and ``yaw``
            determines the push direction via
            ``facing = (sin(yaw), cos(yaw))``.
        offset_x: Approach distance behind the target along the facing
            direction.  The robot starts ``offset_x`` behind and pushes
            ``offset_x * push_through_frac`` past the target.
        offset_z: Height offset above the target z for contact.  The
            contact height is ``target_z + offset_z``.
        transport_z: Safe absolute Z height for the above-approach
            waypoint (default 0.7).
        offset_rot: EE yaw offset added to the target yaw.  The EE
            wrist rotation is ``yaw + offset_rot``.  Default ``0.0``.
        push_through_frac: Fraction of ``offset_x`` to push past the
            target center along the facing direction (default 0.25).
            Set to ``0.0`` to push exactly to the target without
            overshooting.

    Returns:
        A ``ParameterizedOption`` implementing the push skill.

    Example::

        def _get_switch_pose(state, objects, params, config):
            _, switch = objects
            rot = state.get(switch, "rot")
            # Map switch joint axis (cos, sin) to standard facing (sin, cos)
            return (state.get(switch, "x"), state.get(switch, "y"),
                    state.get(switch, "z"), np.pi / 2 - rot)

        switch_on = create_push_skill(
            name="SwitchOn",
            types=[robot_type, switch_type],
            params_space=Box(0, 1, (0,)),
            config=config,   # config.robot_home_pos must be set
            get_target_pose_fn=_get_switch_pose,
            offset_x=0.054,  # approach distance
            offset_z=0.104,  # contact height above switch z
            transport_z=0.7,
        )
    """
    if config.robot_home_pos is None:
        raise ValueError(
            "config.robot_home_pos must be set for create_push_skill.")

    # -- Standard 4-waypoint trajectory ----------------------------------

    def _waypoints(
        ox: float, oy: float, oz: float, oyaw: float, cfg: SkillConfig,
    ) -> List[Tuple[float, float, float, float, str]]:
        assert cfg.robot_home_pos is not None
        obj_xy = np.array([ox, oy])
        facing = np.array([np.sin(oyaw), np.cos(oyaw)])
        behind_xy = obj_xy - facing * offset_x
        push_xy = obj_xy + facing * offset_x * push_through_frac
        home_xy = np.array(cfg.robot_home_pos[:2])
        home_z = cfg.robot_home_pos[2]
        ee_yaw = oyaw + offset_rot
        return [
            (*behind_xy, transport_z, ee_yaw, "closed"),
            (*behind_xy, oz + offset_z, ee_yaw, "closed"),
            (*push_xy, oz + offset_z, ee_yaw, "closed"),
            (*home_xy, home_z, ee_yaw, "closed"),
        ]

    # -- Phase construction -----------------------------------------------

    def _close_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        del params
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
        del params
        robot_obj = objects[0]
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(robot_obj, "fingers"))
        target = cfg.open_fingers_joint - 0.01
        return current, target

    def _make_waypoint_position_fn(
        waypoint_idx: int,
    ) -> Callable[[State, Sequence[Object], Array, SkillConfig], Tuple[
            float, float, float, float]]:

        def _get_target(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            x, y, z, yaw = get_target_pose_fn(state, objects, params, cfg)
            wps = _waypoints(x, y, z, yaw, cfg)
            wx, wy, wz, wyaw, _ = wps[waypoint_idx]
            return wx, wy, wz, wyaw

        return _get_target

    dummy_waypoints = _waypoints(0.0, 0.0, 0.0, 0.0, config)

    phases: List[Phase] = []
    phases.append(
        Phase(name="CloseFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_close_fingers_target))

    for i, (_, _, _, _, finger_status) in enumerate(dummy_waypoints):
        phases.append(
            make_move_to_phase(
                name=f"Waypoint_{i}",
                get_target_pose_fn=_make_waypoint_position_fn(i),
                finger_status=finger_status))

    phases.append(
        Phase(name="OpenFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_open_fingers_target))

    return PhaseSkill(name, types, params_space, config, phases,
                      params_description=params_description).build()
