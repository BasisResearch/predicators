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

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Pick.
# The target object's half-width across the finger axis, in the State's
# finger units (see create_pick_skill's descend_finger_half_gap_fn).
TargetHalfGapFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                           float]

_PICK_PARAMS = [
    ("grasp_z_offset (height above object origin to close gripper; low "
     "values can put the gripper in contact with the object or its support "
     "at the grasp pose, making the grasp config infeasible)", 0.0, 0.1),
]


def create_pick_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    approach_open: bool = False,
    anchor_lift: bool = False,
    grasp_finger_tol: Optional[float] = None,
    lift_dz: float = 0.01,
    verify_lift: bool = False,
    param_defs: Optional[Sequence[Tuple[str, float, float]]] = None,
    descend_finger_half_gap_fn: Optional[TargetHalfGapFn] = None,
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
        approach_open: If True, the MoveAbove phase travels with OPEN
            fingers. The default closed-finger approach reopens the
            fingers only gradually during the descend, so the still-closed
            gripper can ram a light object and drag it a few cm before the
            grasp -- breaking tight downstream placement tolerances.
        anchor_lift: If True, the LiftSlightly phase lifts straight up
            from the xy cached at descend time instead of re-reading the
            (now held) object's xy each step. A held object hangs at a
            small offset from the EE, so a chasing lift target is
            unreachable and the lift can spin or fail IK near the reach
            limit.
        grasp_finger_tol: Optional override for the Grasp phase's finger
            terminal tolerance (squared). Needed when the grasped object
            is wide enough to block the fingers above the default
            terminal (target + sqrt(config.grasp_tol)).
        lift_dz: How far LiftSlightly rises above the grasp height.
            Raise it in cluttered scenes: with the default 1 cm, the
            just-closed gripper can end the pick still grazing (~1 mm)
            a neighboring object, which then invalidates the NEXT
            option's BiRRT start config -- unrecoverable by replanning
            since the arm physically stays put.
        verify_lift: If True, the option only succeeds when the target
            object actually rose with the gripper: at the end of
            LiftSlightly its pose-fn z must have gained at least half of
            ``lift_dz``, else the option raises
            ``OptionExecutionFailure`` instead of reporting a successful
            pick. This is the honest failure for grasps that contact-
            level held detection cannot reject: pads that close above a
            block cam over its top corners or pinch its top edge, the
            grasp constraint freezes the block dangling below the
            gripper, and the support drags it out of the constraint
            during the lift -- the block is left (near) its support
            while the state claims it is held, and the downstream place
            jams it into the support instead of failing here.
        param_defs: Optional override for the continuous parameter
            definitions (``(description, low, high)`` triples). The
            default box spans the whole plausible range for any hand,
            so on a short-fingered arm most of it is dead: below the
            collision edge the grasp pose is infeasible, and above the
            reach edge the fingers close on nothing. Narrow it when the
            env knows its object and its robot -- the dead ends are what
            a sampler spends its budget on.
        descend_finger_half_gap_fn: When given, the descent runs with
            the fingers pre-set to the object's half-width across the
            finger axis (this callback, in the State's finger units -
            the units of the env's ``open_fingers``/``closed_fingers``)
            plus a slack of half the executor's pose tolerance
            (``0.5 * sqrt(config.move_to_pose_tol)``), instead of fully
            open: a ``PreOpen`` finger phase after ``MoveAbove`` sets
            the width and ``MoveToGrasp`` then holds it. A fully open
            gripper on a narrow object carries each finger well past the
            object's face, and that overhang is what clips a close
            neighbor (measured 2026-09-02 on bridge: a finger 9.9 mm
            into the next staged block from every arm branch, so the
            grasp pose was refused outright). The slack keeps the target
            object's own top edges clear of the narrowed fingers under
            the lateral positioning error the pose tolerance permits;
            narrower would trade neighbor clearance for self-clipping.
            The BiRRT goal check scores the descent at the held width,
            so the narrower footprint is what gets validated.

    Returns:
        A ``ParameterizedOption`` implementing the pick skill.
    """
    if param_defs is None:
        param_defs = _PICK_PARAMS
    assert len(param_defs) == len(_PICK_PARAMS), \
        "param_defs must keep the canonical (grasp_z_offset,) order"
    params_space, params_description = build_params_space(param_defs)
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
        _shared["grasp_xy_yaw"] = (x, y, yaw)
        # The object's own (pose-fn) height while it still rests on its
        # support: the lift verification measures the object's rise
        # against this.
        _shared["rest_pose_z"] = z
        return x, y, grasp_z, yaw

    def _slight_lift_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params
        if anchor_lift:
            x, y, yaw = _shared["grasp_xy_yaw"]
        else:
            x, y, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        return x, y, _shared["grasp_z"] + lift_dz, yaw

    def _object_rose_with_gripper(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> bool:
        del params
        z_now = get_target_pose_fn(state, objects, _empty, cfg)[2]
        # Half of lift_dz separates cleanly: a properly-held object
        # tracks the gripper to within a few mm (constraint sag), while
        # a degenerate grasp's object is dragged out of the constraint
        # by its support and gains at most a third of the lift.
        return bool(z_now - _shared["rest_pose_z"] >= 0.5 * lift_dz)

    def _pre_open_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        assert descend_finger_half_gap_fn is not None
        robot_obj = objects[0]
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(robot_obj, "fingers"))
        half_gap = descend_finger_half_gap_fn(state, objects, params, cfg)
        slack = 0.5 * float(np.sqrt(cfg.move_to_pose_tol))
        target = cfg.fingers_state_to_joint(cfg.robot, half_gap + slack)
        lo, hi = sorted((cfg.closed_fingers_joint, cfg.open_fingers_joint))
        return current, float(np.clip(target, lo, hi))

    phases = []
    phases.append(
        make_move_to_phase("MoveAbove", _above_pose,
                           "open" if approach_open else "closed"))
    descend_finger_status = "open"
    if descend_finger_half_gap_fn is not None:
        # Pre-set the fingers to the object's width (+ slack) at transport
        # height, then descend holding that width: see the
        # descend_finger_half_gap_fn docstring.
        phases.append(
            Phase(
                name="PreOpen",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_pre_open_target,
                terminal_fn=None,
                finger_direction="close" if approach_open else "open",
            ))
        descend_finger_status = "hold"
    phases.extend([
        # Validate the grasp goal IK: the gripper descends to envelop the
        # target, and an imprecise (unvalidated) IK config can clip the target
        # object, making BiRRT reject a reachable grasp. See Phase.validate_ik.
        make_move_to_phase("MoveToGrasp",
                           _descend_pose,
                           descend_finger_status,
                           validate_ik=True),
        Phase(
            name="Grasp",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_close_fingers_target,
            terminal_fn=None,
            finger_direction="close",
            finger_tol=grasp_finger_tol,
        ),
        make_move_to_phase(
            "LiftSlightly",
            _slight_lift_pose,
            "closed",
            allow_shallow_held_object_contacts=True,
            verify_fn=_object_rose_with_gripper if verify_lift else None,
            verify_failure_msg=(
                "grasp verification failed: the object did not rise "
                "with the gripper (it was never actually wrapped by the "
                "fingers, or its support dragged it out of the grasp "
                "during the lift). Retry the pick with a grasp_z_offset "
                "that closes the fingers around the object's body, not "
                "above it.") if verify_lift else None)
    ])

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode="home").build()
