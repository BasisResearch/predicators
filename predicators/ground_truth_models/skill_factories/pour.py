"""Pour skill factory: creates a multi-phase pour controller.

This module provides ``create_pour_skill``, which builds a
``ParameterizedOption`` that pours from a held container (e.g. jug) into a
target (e.g. cup) by:

  1. Moving above the pour position at ``config.transport_z``.
  2. Descending to the pour height.
  3. Tilting the end-effector to ``pour_tilt`` (from params).

The tilt phase uses incremental IK (``use_motion_planning=False``) for fine
orientation control.

Continuous parameters: ``(pour_tilt,)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pour_skill,
    )

    def _get_pour_pose(state, objects, params, config):
        _, jug, cup = objects
        return (state.get(cup, "x") + 0.05, state.get(cup, "y"),
                state.get(cup, "z") + 0.1, state.get(jug, "yaw"))

    Pour = create_pour_skill(
        name="Pour",
        types=[robot_type, jug_type, cup_type],
        config=config,
        get_target_pose_fn=_get_pour_pose,
    )
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Pour.
_POUR_PARAMS = [
    ("pour_tilt (EE tilt angle for pouring, radians)", 0.5, 1.0),
]


def create_pour_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    tilt_terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None,
) -> ParameterizedOption:
    """Create a multi-phase pour skill that tilts to pour liquid.

    Phases:
        0. **MoveAbovePour** -- Move above the pour position at
           ``config.transport_z``, with the EE at ``config.robot_init_tilt``.
        1. **DescendToPour** -- Lower to the pour height at normal tilt.
        2. **Tilt** -- Tilt the EE to ``pour_tilt`` (from params). Uses
           incremental IK, not BiRRT, for fine orientation control.

    Continuous parameters:
        ``(pour_tilt,)`` -- target EE tilt angle for pouring.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).
        get_target_pose_fn: Callback returning ``(x, y, z, yaw)`` from
            ``(state, objects, params, config)``.  ``params`` will be empty.
        tilt_terminal_fn: Optional custom terminal for the Tilt phase.

    Returns:
        A ``ParameterizedOption`` implementing the pour skill.
    """
    params_space, params_description = build_params_space(_POUR_PARAMS)
    _empty = np.array([], dtype=np.float32)

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params  # unused
        x, y, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        return x, y, cfg.transport_z, yaw

    def _descend_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params  # unused
        return get_target_pose_fn(state, objects, _empty, cfg)

    def _tilt_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        pour_tilt = float(params[0])
        robot_obj = objects[0]
        current_position = (state.get(robot_obj,
                                      "x"), state.get(robot_obj, "y"),
                            state.get(robot_obj, "z"))
        current_orn = p.getQuaternionFromEuler(
            [0, state.get(robot_obj, "tilt"),
             state.get(robot_obj, "wrist")])
        current_pose = Pose(current_position, current_orn)
        tx, ty, tz, tyaw = get_target_pose_fn(state, objects, _empty, cfg)
        target_orn = p.getQuaternionFromEuler([0, pour_tilt, tyaw])
        target_pose = Pose((tx, ty, tz), target_orn)
        return current_pose, target_pose, "closed"

    phases = [
        # # Phase 0: Move above pour position at normal tilt
        # make_move_to_phase("MoveAbovePour", _above_pose, "closed"),
        # Phase 1: Descend to pour height at normal tilt
        make_move_to_phase("DescendToPour", _descend_pose, "closed"),
        # Phase 2: Tilt EE to pour angle (incremental IK for fine control)
        Phase(
            name="Tilt",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_tilt_target,
            terminal_fn=tilt_terminal_fn,
            use_motion_planning=False,
        ),
    ]

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description).build()
