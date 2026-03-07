"""Place skill factory: creates a multi-phase place controller.

This module provides ``create_place_skill``, which builds a
``ParameterizedOption`` that places a held object by:

  1. Moving above the placement target at ``config.transport_z``.
  2. Descending to ``drop_z`` (from params).
  3. Opening the gripper to release.
  4. Retreating back up to ``config.transport_z``.

The placement target ``(x, y, yaw)`` and ``drop_z`` are all provided as
continuous parameters -- no callback is needed.

Continuous parameters: ``(x, y, yaw, drop_z)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_place_skill,
    )

    Place = create_place_skill(
        name="Place",
        types=[robot_type],
        config=config,
    )
"""

from typing import Sequence, Tuple

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

import numpy as np

# Canonical continuous parameters for Place.
_PLACE_PARAMS = [
    ("x", 0.4, 1.1),
    ("y", 1.1, 1.6),
    ("yaw", -np.pi, np.pi),
    ("drop_z", 0.4, 0.6),
]


def create_place_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
) -> ParameterizedOption:
    """Create a multi-phase place skill that releases a held object.

    Phases:
        0. **MoveAbove** -- Move end-effector above the placement at
           ``config.transport_z``, with fingers closed.
        1. **Descend** -- Lower to ``drop_z`` (from params), with fingers
           closed.
        2. **OpenFingers** -- Open the gripper to release the object.
        3. **Retreat** -- Rise back to ``config.transport_z``, with fingers
           open.

    Continuous parameters:
        ``(x, y, yaw, drop_z)`` -- placement position, orientation, and
        release height.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).

    Returns:
        A ``ParameterizedOption`` implementing the place skill.
    """
    params_space, params_description = build_params_space(_PLACE_PARAMS)

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

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, yaw = float(params[0]), float(params[1]), float(params[2])
        return x, y, cfg.transport_z, yaw

    def _drop_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, yaw = float(params[0]), float(params[1]), float(params[2])
        drop_z = float(params[3])
        return x, y, drop_z, yaw

    phases = [
        # Phase 0: Move above placement
        make_move_to_phase("MoveAbove", _above_pose, "closed"),
        # Phase 1: Descend to drop height
        make_move_to_phase("Descend", _drop_pose, "closed"),
        # Phase 2: Open fingers to release
        Phase(
            name="OpenFingers",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_open_fingers_target,
        ),
        # Phase 3: Retreat upward
        make_move_to_phase("Retreat", _above_pose, "open"),
    ]

    return PhaseSkill(name, types, params_space, config, phases,
                      params_description=params_description).build()
