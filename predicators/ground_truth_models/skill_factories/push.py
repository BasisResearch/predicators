"""Push skill factory: creates a multi-phase push controller.

This module provides ``create_push_skill``, which builds a
``ParameterizedOption`` that pushes an object (e.g. domino, switch, button)
using a standard 4-waypoint trajectory:

  1. Closing the gripper.
  2. Moving above & behind the target at ``config.transport_z``.
  3. Descending to contact height (target z + ``contact_z_offset``).
  4. Pushing to the target position along its facing direction.
  5. Retreating to ``config.robot_home_pos``.
  6. Opening the gripper.

The "facing direction" is derived from the yaw returned by
``get_target_pose_fn`` as ``(sin(yaw), cos(yaw))``.  "Behind" means
opposite to the facing direction.

``config.robot_home_pos`` **must** be set.

Continuous parameters: ``(approach_distance, contact_z_offset)``

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
        config=config,
        get_target_pose_fn=_get_domino_pose,
    )
"""

from typing import Callable, List, Sequence, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Push.
_PUSH_PARAMS = [
    ("approach_distance (dist behind target along facing dir to start push)",
     0.00, 0.05),
    ("contact_z_offset (height above target z for contact)", 0.0, 0.1),
]


def create_push_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
) -> ParameterizedOption:
    """Create a multi-phase push skill with a standard 4-waypoint trajectory.

    Phases:
        0. **CloseFingers** -- Close the gripper before approaching.
        1. **Waypoint_0** -- Move above & behind the target at
           ``config.transport_z``, offset by ``approach_distance``
           opposite the facing direction.
        2. **Waypoint_1** -- Descend to contact height
           (target z + ``contact_z_offset``) at the same behind position.
        3. **Waypoint_2** -- Push forward to the target position.
        4. **Waypoint_3** -- Retreat to ``config.robot_home_pos``.
        5. **OpenFingers** -- Open the gripper.

    Continuous parameters:
        ``(approach_distance, contact_z_offset)``

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration.  ``config.robot_home_pos`` and
            ``config.transport_z`` must be set.
        get_target_pose_fn: Callback returning ``(x, y, z, yaw)`` from
            ``(state, objects, params, config)``.  ``params`` will be empty.

    Returns:
        A ``ParameterizedOption`` implementing the push skill.
    """
    if config.robot_home_pos is None:
        raise ValueError(
            "config.robot_home_pos must be set for create_push_skill.")

    params_space, params_description = build_params_space(_PUSH_PARAMS)
    _empty = np.array([], dtype=np.float32)

    # -- Standard 4-waypoint trajectory ----------------------------------

    def _waypoints(
        ox: float,
        oy: float,
        oz: float,
        oyaw: float,
        cfg: SkillConfig,
        s_offset_x: float,
        s_offset_z: float,
    ) -> List[Tuple[float, float, float, float, str]]:
        assert cfg.robot_home_pos is not None
        obj_xy = np.array([ox, oy])
        facing = np.array([np.sin(oyaw), np.cos(oyaw)])
        behind_xy = obj_xy - facing * s_offset_x
        push_xy = obj_xy
        home_xy = np.array(cfg.robot_home_pos[:2])
        home_z = cfg.robot_home_pos[2]
        ee_yaw = oyaw + np.pi / 2
        return [
            (*behind_xy, cfg.transport_z, ee_yaw, "closed"),
            (*behind_xy, oz + s_offset_z, ee_yaw, "closed"),
            (*push_xy, oz + s_offset_z, ee_yaw, "closed"),
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
            s_ox = float(params[0])
            s_oz = float(params[1])
            x, y, z, yaw = get_target_pose_fn(state, objects, _empty, cfg)
            wps = _waypoints(x, y, z, yaw, cfg, s_ox, s_oz)
            wx, wy, wz, wyaw, _ = wps[waypoint_idx]
            return wx, wy, wz, wyaw

        return _get_target

    phases: List[Phase] = []
    phases.append(
        Phase(name="CloseFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_close_fingers_target))

    for i in range(4):
        phases.append(
            make_move_to_phase(
                name=f"Waypoint_{i}",
                get_target_pose_fn=_make_waypoint_position_fn(i),
                finger_status="closed"))

    phases.append(
        Phase(name="OpenFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_open_fingers_target))

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description).build()
