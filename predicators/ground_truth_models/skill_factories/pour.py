"""Pour skill factory: creates a multi-phase pour controller.

This module provides ``create_pour_skill``, which builds a
``ParameterizedOption`` that pours from a held container (e.g. jug) into a
target (e.g. cup) by:

  1. Moving above the pour position at ``transport_z``.
  2. Descending to the pour height.
  3. Tilting the end-effector to ``pour_tilt`` angle.

The tilt phase uses incremental IK (``use_motion_planning=False``) for fine
orientation control.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pour_skill,
    )

    def _get_pour_pose(state, objects, params, config):
        _, jug, cup = objects
        # Position above the cup, offset for the jug spout.
        return (state.get(cup, "x") + 0.05, state.get(cup, "y"),
                state.get(cup, "z") + 0.1, state.get(jug, "yaw"))

    Pour = create_pour_skill(
        name="Pour",
        types=[robot_type, jug_type, cup_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_pour_pose,
        pour_tilt=1.2,
        transport_z=0.8,
    )
"""

from typing import Callable, Optional, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_pour_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    pour_tilt: float,
    transport_z: float,
    tilt_terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None,
) -> ParameterizedOption:
    """Create a multi-phase pour skill that tilts to pour liquid.

    Phases:
        0. **MoveAbovePour** -- Move above the pour position at
           ``transport_z``, with the EE at ``config.robot_init_tilt``.
        1. **DescendToPour** -- Lower to the pour height at normal tilt.
        2. **Tilt** -- Tilt the EE to ``pour_tilt`` (uses incremental IK,
           not BiRRT, for fine orientation control).

    Args:
        name: Option name used for logging and matching (e.g. "Pour").
        types: Ordered object types for the option signature.  The first
            element **must** be the robot type.
        params_space: Continuous parameter space.  Use ``Box(0, 1, (0,))``
            for zero-dimensional (no sampled parameters).
        config: Shared skill configuration.  See ``SkillConfig``.
        get_target_pose_fn: Callback that returns the pour target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
            ``(x, y, z)`` is the EE position for pouring (typically offset
            from the cup or target vessel), and ``yaw`` is the EE wrist
            rotation.
        pour_tilt: Target EE tilt (pitch) angle for the pouring position.
            This replaces ``config.robot_init_tilt`` in the Tilt phase.
        transport_z: Safe Z height for transit above obstacles.
        tilt_terminal_fn: Optional custom terminal for the Tilt phase.
            Signature: ``(state, objects, params, config) -> bool``.
            If ``None``, terminates when the EE reaches the target pose
            (default distance-based terminal).

    Returns:
        A ``ParameterizedOption`` implementing the pour skill.

    Example::

        def _get_pour_pose(state, objects, params, config):
            _, jug, cup = objects
            return (state.get(cup, "x"), state.get(cup, "y"),
                    state.get(cup, "z") + 0.1, config.robot_init_wrist)

        pour = create_pour_skill(
            name="Pour",
            types=[robot_type, jug_type, cup_type],
            params_space=Box(0, 1, (0,)),
            config=config,
            get_target_pose_fn=_get_pour_pose,
            pour_tilt=1.2,
            transport_z=0.8,
        )
    """

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, _, yaw = get_target_pose_fn(state, objects, params, cfg)
        return x, y, transport_z, yaw

    def _tilt_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        robot_obj = objects[0]
        current_position = (state.get(robot_obj, "x"),
                            state.get(robot_obj, "y"),
                            state.get(robot_obj, "z"))
        current_orn = p.getQuaternionFromEuler(
            [0, state.get(robot_obj, "tilt"),
             state.get(robot_obj, "wrist")])
        current_pose = Pose(current_position, current_orn)
        tx, ty, tz, tyaw = get_target_pose_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, pour_tilt, tyaw])
        target_pose = Pose((tx, ty, tz), target_orn)
        return current_pose, target_pose, "closed"

    phases = [
        # Phase 0: Move above pour position at normal tilt
        make_move_to_phase("MoveAbovePour", _above_pose, "closed"),
        # Phase 1: Descend to pour height at normal tilt
        make_move_to_phase("DescendToPour", get_target_pose_fn, "closed"),
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
