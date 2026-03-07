"""Pick skill factory: creates a multi-phase pick-and-lift controller.

This module provides ``create_pick_skill``, which builds a
``ParameterizedOption`` that picks up an object by:

  1. Moving above the object at ``transport_z``.
  2. Descending to the grasp height (object z + ``grasp_z_offset``).
  3. Closing the gripper.
  4. Lifting back to ``transport_z``.

The caller supplies a single callback ``get_target_pose_fn`` that extracts
the object's ``(x, y, z, yaw)`` from the current state.  All environment-
specific logic lives in this callback; the factory handles motion planning,
IK, and phase sequencing.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pick_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
    )

    def _get_jug_pose(state, objects, params, config):
        _, jug = objects
        return (state.get(jug, "x"), state.get(jug, "y"),
                state.get(jug, "z"), state.get(jug, "rot"))

    PickJug = create_pick_skill(
        name="PickJug",
        types=[robot_type, jug_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_jug_pose,
        transport_z=0.8,
    )
"""

from typing import Optional, Sequence, Tuple

from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_pick_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    transport_z: float = 0.7,
    grasp_z_offset: float = 0.0,
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a multi-phase pick skill that grasps and lifts an object.

    Phases:
        0. **MoveAbove** -- Move end-effector above the object at
           ``transport_z``, with fingers closed.
        1. **Descend** -- Lower to the object's z + ``grasp_z_offset``,
           with fingers open.
        2. **Grasp** -- Close fingers until contact (or custom terminal).
        3. **Lift** -- Lift back to ``transport_z``, with fingers closed.

    Args:
        name: Option name used for logging and matching (e.g. "Pick",
            "PickJug").
        types: Ordered object types for the option signature.  The first
            element **must** be the robot type.
            Example: ``[robot_type, obj_type]``.
        params_space: Continuous parameter space.  Use ``Box(0, 1, (0,))``
            for zero-dimensional (no sampled parameters).
        config: Shared skill configuration.  See ``SkillConfig``.
        get_target_pose_fn: Callback that returns the grasp target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
            ``(x, y)`` is the horizontal grasp position, ``z`` is the
            z position of the center of the object (before
            ``grasp_z_offset`` is applied), and
            ``yaw`` is the EE wrist rotation for approach.
        transport_z: Safe Z height for moving above obstacles before and
            after grasping.
        grasp_z_offset: Added to the z returned by ``get_target_pose_fn``
            to compute the actual descend height.  Default ``0.0``.

    Returns:
        A ``ParameterizedOption`` implementing the pick skill.

    Example::

        def _get_cup_pose(state, objects, params, config):
            _, cup = objects
            return (state.get(cup, "x"), state.get(cup, "y"),
                    state.get(cup, "z"), 0.0)

        pick_cup = create_pick_skill(
            name="PickCup",
            types=[robot_type, cup_type],
            params_space=Box(0, 1, (0,)),
            config=config,
            get_target_pose_fn=_get_cup_pose,
            transport_z=0.7,
            grasp_z_offset=0.02,
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

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, _, yaw = get_target_pose_fn(state, objects, params, cfg)
        return x, y, transport_z, yaw

    def _descend_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, z, yaw = get_target_pose_fn(state, objects, params, cfg)
        return x, y, z + grasp_z_offset, yaw

    phases = [
        # Phase 0: Move above object
        make_move_to_phase("MoveAbove", _above_pose, "closed"),
        # Phase 1: Descend to grasp height
        make_move_to_phase("Descend", _descend_pose, "open"),
        # Phase 2: Close fingers to grasp
        Phase(
            name="Grasp",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_close_fingers_target,
            terminal_fn=None,
        ),
        # Phase 3: Lift to transport height
        make_move_to_phase("Lift", _above_pose, "closed"),
    ]

    return PhaseSkill(name, types, params_space, config, phases,
                      params_description=params_description).build()
