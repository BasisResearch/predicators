"""Place skill factory: creates a multi-phase place controller.

This module provides ``create_place_skill``, which builds a
``ParameterizedOption`` that places a held object by:

  1. Moving directly to the release position (collision-free via BiRRT).
  2. Opening the gripper to release.
  3. Retreating back up to ``config.transport_z``.

When ``use_move_above=True``, an extra MoveAbove phase is inserted before
the descent, moving to ``config.transport_z`` first.

The placement target ``(target_x, target_y, target_yaw)`` and
``release_z`` are all provided as continuous parameters -- no callback
is needed.

Continuous parameters: ``(target_x, target_y, release_z, target_yaw)``

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

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Place.
_PLACE_PARAMS = [
    ("target_x (world x position for placement)", 0.4, 1.1),
    ("target_y (world y position for placement)", 1.1, 1.6),
    ("release_z (world z height to open gripper)", 0.5, 0.6),
    ("target_yaw (placement orientation in radians)", -np.pi, np.pi),
]


def create_place_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    use_move_above: bool = False,
) -> ParameterizedOption:
    """Create a multi-phase place skill that releases a held object.

    By default (``use_move_above=False``), the skill moves directly to the
    release position, relying on BiRRT for collision avoidance:

        0. **MoveToDrop** -- Move to ``(target_x, target_y, release_z)``.
        1. **OpenFingers** -- Release the object.
        2. **Retreat** -- Rise to ``config.transport_z``.

    With ``use_move_above=True``, an extra phase is prepended:

        0. **MoveAbove** -- Move to ``(target_x, target_y, transport_z)``.
        1. **Descend** -- Lower to ``release_z``.
        2. **OpenFingers** -- Release the object.
        3. **Retreat** -- Rise to ``config.transport_z``.

    Continuous parameters:
        ``(target_x, target_y, release_z, target_yaw)`` -- placement
        position, orientation, and release height.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).
        use_move_above: If True, add a MoveAbove phase before descending.

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
        del state, objects  # unused
        x, y, yaw = float(params[0]), float(params[1]), float(params[3])
        return x, y, cfg.transport_z, yaw

    def _drop_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del state, objects, cfg  # unused
        x, y = float(params[0]), float(params[1])
        drop_z, yaw = float(params[2]), float(params[3])
        return x, y, drop_z, yaw

    phases = []
    if use_move_above:
        phases.append(make_move_to_phase("MoveAbove", _above_pose, "closed"))
    phases.extend([
        make_move_to_phase("Descend" if use_move_above else "MoveToDrop",
                           _drop_pose, "closed"),
        Phase(
            name="OpenFingers",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_open_fingers_target,
        ),
        make_move_to_phase("Retreat", _above_pose, "open"),
    ])

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description).build()
