"""Place skill factory: creates a multi-phase place controller.

This module provides ``create_place_skill``, which builds a
``ParameterizedOption`` that places a held object by:

  1. Moving above the placement target at ``transport_z``.
  2. Descending to ``drop_z``.
  3. Opening the gripper to release.
  4. Retreating back up to ``transport_z``.

The caller supplies a single callback ``get_target_pose_fn`` that computes
the placement ``(x, y, z, yaw)`` from the current state.  The ``z`` return
value is unused (``transport_z`` and ``drop_z`` are used instead), but is
included for interface uniformity.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_place_skill,
    )

    def _get_placement_pose(state, objects, params, config):
        # Place at a fixed location with no rotation.
        return (1.2, 0.5, 0.0, 0.0)

    PlaceObj = create_place_skill(
        name="Place",
        types=[robot_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_placement_pose,
        transport_z=0.8,
        drop_z=0.45,
    )
"""

from typing import Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_place_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    transport_z: float,
    drop_z: float,
) -> ParameterizedOption:
    """Create a multi-phase place skill that releases a held object.

    Phases:
        0. **MoveAbove** -- Move end-effector above the placement at
           ``transport_z``, with fingers closed.
        1. **Descend** -- Lower to ``drop_z``, with fingers closed.
        2. **OpenFingers** -- Open the gripper to release the object.
        3. **Retreat** -- Rise back to ``transport_z``, with fingers open.

    Args:
        name: Option name used for logging and matching (e.g. "Place",
            "PlaceOnBurner").
        types: Ordered object types for the option signature.  The first
            element **must** be the robot type.
        params_space: Continuous parameter space.  Use ``Box(0, 1, (0,))``
            for zero-dimensional (no sampled parameters).
        config: Shared skill configuration.  See ``SkillConfig``.
        get_target_pose_fn: Callback that returns the placement target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
            The ``z`` value is ignored (``transport_z`` and ``drop_z`` are
            used instead), but the callback should still return a 4-tuple
            for interface uniformity.
        transport_z: Safe Z height for transit above obstacles.
        drop_z: Z height at which to release the object.

    Returns:
        A ``ParameterizedOption`` implementing the place skill.

    Example::

        def _get_target(state, objects, params, config):
            target_x, target_y = params[0], params[1]
            return (float(target_x), float(target_y), 0.0, 0.0)

        place = create_place_skill(
            name="Place",
            types=[robot_type],
            params_space=Box(low=np.array([0.0, 0.0]),
                             high=np.array([1.0, 1.0])),
            config=config,
            get_target_pose_fn=_get_target,
            transport_z=0.8,
            drop_z=0.45,
        )
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
        tx, ty, _, tyaw = get_target_pose_fn(state, objects, params, cfg)
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
        tx, ty, _, tyaw = get_target_pose_fn(state, objects, params, cfg)
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
        tx, ty, _, tyaw = get_target_pose_fn(state, objects, params, cfg)
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
