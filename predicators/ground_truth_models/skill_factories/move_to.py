"""Move-to-pose skill factory and reusable phase builder.

This module provides:

- ``create_move_to_skill`` -- A single-phase skill that moves the EE to
  a target pose while preserving the current finger state.
- ``make_move_to_phase`` -- A lower-level helper that creates a single
  ``Phase`` object for use in custom ``PhaseSkill`` compositions (used
  internally by ``create_push_skill`` and available for building custom
  multi-phase skills).

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_move_to_skill,
    )

    def _get_home_pose(state, objects, params, config):
        return (1.35, 0.75, 0.75, 0.0)

    MoveHome = create_move_to_skill(
        name="MoveHome",
        types=[robot_type],
        params_space=Box(0, 1, (0,)),
        config=config,
        get_target_pose_fn=_get_home_pose,
    )
"""

from typing import Callable, Optional, Sequence, Tuple

import pybullet as p
from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn
from predicators.pybullet_helpers.geometry import Pose
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, State, Type


def create_move_to_skill(
    name: str,
    types: Sequence[Type],
    params_space: Box,
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    params_description: Optional[Tuple[str, ...]] = None,
    use_move_above: bool = False,
    retreat: bool = False,
    validate_ik: bool = False,
    base_mode: Optional[str] = None,
) -> ParameterizedOption:
    """Create a move-to-pose skill.

    Preserves the current finger status (open/closed) from state.

    Phases (with the default flags, only the middle one):
        0. **MoveAbove** (``use_move_above``) -- Move to the target xy
           at ``config.transport_z``, so the approach comes from above
           and a held object clears the scene.
        1. **Move** -- Move end-effector to the target pose, preserving
           the current finger state.
        2. **Retreat** (``retreat``) -- Return to the target xy at
           ``config.transport_z``.  Allows shallow held-object contacts
           at the start, so a target pose that ends grazing a surface
           by ~1 mm doesn't invalidate the retreat's motion-planning
           start config.

    Args:
        name: Option name.
        types: Ordered object types.  The first element **must** be the
            robot type.
        params_space: Continuous parameter space.
        config: Shared skill configuration.  See ``SkillConfig``.
        get_target_pose_fn: Callback that returns the target as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.
        use_move_above: Prepend the transport-height approach phase.
        retreat: Append the transport-height retreat phase.
        validate_ik: Gate the Move phase's target through validated IK.
        base_mode: Optional ``PhaseSkill`` base mode (e.g. ``"home"``).

    Returns:
        A ``ParameterizedOption`` implementing the move-to-pose skill.
    """

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, _, yaw = get_target_pose_fn(state, objects, params, cfg)
        return x, y, cfg.transport_z, yaw

    phases = []
    if use_move_above:
        phases.append(make_move_to_phase("MoveAbove", _above_pose))
    phases.append(
        make_move_to_phase("Move", get_target_pose_fn,
                           validate_ik=validate_ik))
    if retreat:
        phases.append(
            make_move_to_phase("Retreat",
                               _above_pose,
                               allow_shallow_held_object_contacts=True))
    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode=base_mode).build()


def _get_current_ee_pose(state: State, robot_obj: Object) -> Pose:
    """Extract current end-effector pose from state."""
    position = (state.get(robot_obj,
                          "x"), state.get(robot_obj,
                                          "y"), state.get(robot_obj, "z"))
    orientation = p.getQuaternionFromEuler(
        [0, state.get(robot_obj, "tilt"),
         state.get(robot_obj, "wrist")])
    return Pose(position, orientation)


def _get_finger_status(state: State, robot_obj: Object,
                       cfg: SkillConfig) -> str:
    """Infer 'open' or 'closed' from current finger state."""
    current_fingers = state.get(robot_obj, "fingers")
    finger_joint = cfg.fingers_state_to_joint(cfg.robot, current_fingers)
    if abs(finger_joint - cfg.open_fingers_joint) < \
            abs(finger_joint - cfg.closed_fingers_joint):
        return "open"
    return "closed"


def make_move_to_phase(
    name: str,
    get_target_pose_fn: TargetPoseFn,
    finger_status: Optional[str] = None,
    expect_contact: bool = False,
    allow_shallow_held_object_contacts: bool = False,
    validate_ik: bool = False,
    check_release_clearance: bool = False,
    use_motion_planning: Optional[bool] = None,
    terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None,
    max_step_norm: Optional[float] = None,
) -> Phase:
    """Create a MOVE_TO_POSE phase for use in a ``PhaseSkill``.

    This is a building block for composing custom multi-phase skills.
    For example, ``create_push_skill`` uses this internally to create
    each waypoint phase.

    Args:
        name: Phase name (for logging).
        get_target_pose_fn: Callback that returns ``(x, y, z, yaw)``
            from ``(state, objects, params, config)``.
        finger_status: ``"open"``, ``"closed"``, or ``"hold"`` (keep the
            current width, e.g. retreating from a partial-open release).
            If ``None``, preserves the current finger status from state.
        use_motion_planning: ``None`` (the default) defers to
            ``CFG.skill_phase_use_motion_planning``. Pass ``False`` for a
            contact stroke -- a phase whose goal pose is at or inside an
            object -- which must step IK straight at the target; a
            collision-free planner asked for such a goal either fails or
            reaches it by a detour that arrives from the wrong direction
            (see ``create_push_skill``).
        terminal_fn: Optional custom terminal override forwarded to the
            ``Phase`` (e.g. "held object made contact"); when ``None``
            the phase uses the default distance-based terminal.
        max_step_norm: Optional gentle-stroke step clamp forwarded to
            the ``Phase`` (see ``Phase.max_step_norm``): small EE steps
            plus a joint-jump guard and the stall abort, for
            incremental-IK phases that deliberately seek contact.

    Returns:
        A ``Phase`` that can be included in a ``PhaseSkill``.

    Example::

        from predicators.ground_truth_models.skill_factories import (
            Phase, PhaseAction, PhaseSkill, SkillConfig, make_move_to_phase,
        )

        def _above_target(state, objects, params, config):
            _, obj = objects
            return (state.get(obj, "x"), state.get(obj, "y"),
                    0.8, state.get(obj, "yaw"))

        def _at_target(state, objects, params, config):
            _, obj = objects
            return (state.get(obj, "x"), state.get(obj, "y"),
                    state.get(obj, "z"), state.get(obj, "yaw"))

        phases = [
            make_move_to_phase("MoveAbove", _above_target, "closed"),
            make_move_to_phase("Descend",   _at_target,    "open"),
        ]
        skill = PhaseSkill("Custom", types, params_space, config, phases)
        option = skill.build()
    """

    def _target_fn(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        robot_obj = objects[0]
        current_pose = _get_current_ee_pose(state, robot_obj)

        tx, ty, tz, tyaw = get_target_pose_fn(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, tyaw])
        target_pose = Pose((tx, ty, tz), target_orn)

        if finger_status is not None:
            status = finger_status
        else:
            status = _get_finger_status(state, robot_obj, cfg)
        return current_pose, target_pose, status

    # None means "whatever the config says", which is what Phase's own default
    # resolves to; both are read at construction time, so naming it here is the
    # same value the default would have picked.
    plan_motion = (CFG.skill_phase_use_motion_planning
                   if use_motion_planning is None else use_motion_planning)
    return Phase(
        name=name,
        action_type=PhaseAction.MOVE_TO_POSE,
        target_fn=_target_fn,
        terminal_fn=terminal_fn,
        expect_contact=expect_contact,
        allow_shallow_held_object_contacts=allow_shallow_held_object_contacts,
        validate_ik=validate_ik,
        check_release_clearance=check_release_clearance,
        use_motion_planning=plan_motion,
        max_step_norm=max_step_norm,
    )
