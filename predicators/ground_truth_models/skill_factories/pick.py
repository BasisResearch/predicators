"""Pick skill factory: creates a multi-phase pick-and-lift controller.

This module provides ``create_pick_skill``, which builds a
``ParameterizedOption`` that picks up an object by:

  1. Moving above the object at ``config.transport_z`` (closed gripper).
  2. Descending to the grasp height (open gripper, collision-free via BiRRT).
  3. Closing the gripper.
  4. Lifting slightly above the grasp height.

The caller supplies a single callback ``get_target_pose_fn`` that extracts
the object's ``(x, y, z, yaw)`` from the current state.  All environment-
specific logic lives in this callback; the factory handles motion planning,
IK, and phase sequencing.

Continuous parameters: ``(grasp_z_offset,)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pick_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
        transport_z=0.8,
    )

    def _get_jug_pose(state, objects, params, config):
        _, jug = objects
        return (state.get(jug, "x"), state.get(jug, "y"),
                state.get(jug, "z"), state.get(jug, "rot"))

    PickJug = create_pick_skill(
        name="PickJug",
        types=[robot_type, jug_type],
        config=config,
        get_target_pose_fn=_get_jug_pose,
    )
"""

from typing import Sequence, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Pick.
_PICK_PARAMS = [
    ("grasp_z_offset (height above object origin to close gripper)", 0.0, 0.1),
]


def create_pick_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
) -> ParameterizedOption:
    """Create a multi-phase pick skill that grasps and lifts an object.

    Phases:
        0. **MoveAbove** -- Move above the object at ``config.transport_z``
           with closed gripper.
        1. **MoveToGrasp** -- Descend to object z + ``grasp_z_offset``
           with open gripper (collision-free via BiRRT).
        2. **Grasp** -- Close fingers.
        3. **LiftSlightly** -- Lift slightly above the grasp height.

    Continuous parameters:
        ``(grasp_z_offset,)`` -- offset added to z returned by
        ``get_target_pose_fn`` for the descend height.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).
        get_target_pose_fn: Callback returning ``(x, y, z, yaw)`` from
            ``(state, objects, params, config)``.  ``params`` will be empty.

    Returns:
        A ``ParameterizedOption`` implementing the pick skill.
    """
    params_space, params_description = build_params_space(_PICK_PARAMS)
    _empty = np.array([], dtype=np.float32)
    _shared: dict = {}

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

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params
        x, y, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        return x, y, cfg.transport_z, yaw

    def _descend_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        grasp_z_offset = float(params[0])
        x, y, z, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        grasp_z = z + grasp_z_offset
        _shared["grasp_z"] = grasp_z
        return x, y, grasp_z, yaw

    def _slight_lift_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params
        x, y, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        return x, y, _shared["grasp_z"] + 0.01, yaw

    phases = []
    phases.extend([
        make_move_to_phase("MoveAbove", _above_pose, "closed"),
        make_move_to_phase("MoveToGrasp", _descend_pose, "open"),
        Phase(
            name="Grasp",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_close_fingers_target,
            terminal_fn=None,
        ),
        make_move_to_phase("LiftSlightly", _slight_lift_pose, "closed")
    ])

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description).build()
