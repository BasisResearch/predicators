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

from typing import Optional, Sequence, Tuple

from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_place_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    transport_z: float,
    drop_z: float,
    params_description: Optional[Tuple[str, ...]] = None,
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

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, _, yaw = get_target_pose_fn(state, objects, params, cfg)
        return x, y, transport_z, yaw

    def _drop_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, _, yaw = get_target_pose_fn(state, objects, params, cfg)
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
