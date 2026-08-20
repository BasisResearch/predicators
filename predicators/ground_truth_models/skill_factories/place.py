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

from typing import Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.ground_truth_models.skill_factories.base import \
    _RELEASE_CLEAR_SLACK, _RELEASE_OPEN_STEP, Phase, PhaseAction, PhaseSkill, \
    SkillConfig, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Contact distances below this count as touching for the guarded
# settle (getContactPoints reports near-contacts up to the contact
# processing threshold with small positive separations).
_SETTLE_CONTACT_DIST = 1e-4


def _held_assembly_in_contact(state: State) -> bool:
    """True when the held object, or any body rigidly attached to it, touches a
    body outside the held assembly (the robot excluded).

    Attached partners matter: a carried multi-body assembly usually
    touches down through an OUTER member, not the grasped one. The
    assembly is discovered generically by BFS over the client's fixed
    constraints, skipping the grasp constraint (any fixed constraint
    involving the robot body).
    """
    sim_state = getattr(state, "simulator_state", None)
    if not isinstance(sim_state, dict):
        return False
    client = sim_state.get("physics_client_id")
    robot_id = sim_state.get("robot_id")
    if client is None:
        return False
    held_id: Optional[int] = None
    for obj in state:
        if "is_held" in obj.type.feature_names and \
                state.get(obj, "is_held") > 0.5:
            held_id = getattr(obj, "id", None)
            break
    if held_id is None:
        return False
    edges = []
    for i in range(p.getNumConstraints(physicsClientId=client)):
        cid = p.getConstraintUniqueId(i, physicsClientId=client)
        info = p.getConstraintInfo(cid, physicsClientId=client)
        parent, child, joint_type = info[0], info[2], info[4]
        if joint_type != p.JOINT_FIXED or robot_id in (parent, child):
            continue
        edges.append((parent, child))
    assembly: Set[int] = {held_id}
    frontier = [held_id]
    while frontier:
        cur = frontier.pop()
        for a, b in edges:
            for nxt in ((b, ) if a == cur else (a, ) if b == cur else ()):
                if nxt not in assembly:
                    assembly.add(nxt)
                    frontier.append(nxt)
    for body in assembly:
        for cp in p.getContactPoints(bodyA=body, physicsClientId=client):
            other = cp[2]
            if other == robot_id or other in assembly:
                continue
            if cp[8] < _SETTLE_CONTACT_DIST:
                return True
    return False


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
    param_defs: Optional[Sequence[Tuple[str, float, float]]] = None,
    compensate_held_offset: bool = False,
    compensate_held_z: bool = False,
    settle_to_contact_depth: Optional[float] = None,
    verify_xy_tol: Optional[float] = None,
    verify_max_retries: int = 2,
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

    When ``config.release_until_ungrasped`` is set, the release is
    grasp-relative: **OpenFingers** opens gradually just until the
    simulator drops the grasp constraint (observed as ``is_held``
    flipping in the state), **ClearFingers** opens a few millimetres
    more so the pads clear the released object, **Retreat** holds that
    width, and a final **FullyOpenFingers** phase opens fully at
    transport height - so a placement only needs side clearance for
    roughly the held object's thickness, not the full opening span.

    Continuous parameters:
        ``(target_x, target_y, release_z, target_yaw)`` -- placement
        position, orientation, and release height.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).
        use_move_above: If True, add a MoveAbove phase before descending.
        param_defs: Optional override for the continuous parameter
            definitions (``(description, low, high)`` triples). Must keep
            the canonical order ``(target_x, target_y, release_z,
            target_yaw)`` -- the phases index params positionally. Use
            this when an env needs wider bounds (e.g. releasing above a
            tall structure) than the ``_PLACE_PARAMS`` defaults.
        compensate_held_offset: If True, shift the EE target xy by the
            live (EE - held object) offset read from the state, so the
            HELD OBJECT (not the gripper) lands at ``(target_x,
            target_y)``. A grasp near the arm's reach limit can leave
            the object hanging ~2 cm off the EE (IK residual at the
            grasp pose); without compensation that error transfers
            verbatim to every placement.
        compensate_held_z: If True, additionally shift the EE target z
            by the live (EE - held object) z offset, so ``release_z``
            is the HELD OBJECT'S CENTER height at release rather than
            an EE height. Without it, samplers must budget the grasp
            depth AND the pick's IK z-residual into ``release_z`` --
            and a deep grasp (~2 cm residual) drives the held object
            into the support surface at the descend goal.
        settle_to_contact_depth: If set, insert a guarded
            **SettleToContact** phase between the descent and the
            release: an incremental-IK contact stroke (no BiRRT --
            its goal is intentionally at/inside the support) that
            lowers the held object up to this many meters below
            ``release_z`` and stops at the FIRST contact of the held
            assembly (the held object or anything rigidly attached
            to it) with a body outside the assembly. The release then
            happens at essentially zero gap, eliminating the
            free-fall bounce-and-slide scatter of an open-loop drop.
            ``release_z`` stays the (collision-checked) descend goal,
            so it must remain clear of the scene; choose the depth to
            exceed the largest drop clearance a sampler uses. If
            nothing is contacted within the depth, the phase ends at
            the depth and the place degrades to a normal (lower)
            drop. Note the release-clearance check still validates
            the finger-opening sweep at ``release_z``, an upper bound
            of the actual release pose.
        verify_xy_tol: If set (requires ``settle_to_contact_depth``),
            verify BEFORE releasing that the still-held object's xy is
            within this many meters of the commanded ``(target_x,
            target_y)``. On failure the skill rewinds to the descend
            phase (lifting the held object back to ``release_z``) and
            re-descends, up to ``verify_max_retries`` times. The settle
            stroke releases at FIRST contact, and plant sag (position
            control under gravity + payload) can walk that contact
            point ~15 mm from the commanded spot; the rewind learns an
            aim offset from the measured error, so the retried stroke
            aims upstream of the (repeatable) sag and lands on target
            with an unstrained servo.
        verify_max_retries: Retry budget for the verification.

    Returns:
        A ``ParameterizedOption`` implementing the place skill.
    """
    if param_defs is None:
        param_defs = _PLACE_PARAMS
    assert len(param_defs) == len(_PLACE_PARAMS), \
        "param_defs must keep the canonical (x, y, release_z, yaw) order"
    params_space, params_description = build_params_space(param_defs)

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
        target = cfg.open_fingers_joint
        return current, target

    def _current_fingers(state: State, robot_obj: Object,
                         cfg: SkillConfig) -> float:
        return cfg.fingers_state_to_joint(cfg.robot,
                                          state.get(robot_obj, "fingers"))

    def _release_open_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        # One anchored opening step from the measured grasp width - big
        # enough to exceed every env's _finger_action_tol so the
        # simulator drops the grasp constraint on the first action; the
        # phase terminates (via _nothing_held) as soon as the state
        # reflects the release, so the width stays bounded by
        # grasp + _RELEASE_OPEN_STEP. Anchored (see
        # Phase.anchor_finger_target) so the target does not ratchet
        # while waiting for is_held to flip.
        del params
        current = _current_fingers(state, objects[0], cfg)
        return current, min(cfg.open_fingers_joint,
                            current + _RELEASE_OPEN_STEP)

    def _clear_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        # Anchored at phase entry (see Phase.anchor_finger_target): the
        # width at which the release was observed, plus slack so the pads
        # clear the released object before the hold-width retreat.
        del params
        current = _current_fingers(state, objects[0], cfg)
        return current, min(cfg.open_fingers_joint,
                            current + _RELEASE_CLEAR_SLACK)

    def _nothing_held(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> bool:
        del objects, params, cfg
        for obj in state:
            if "is_held" in obj.type.feature_names and \
                    state.get(obj, "is_held") > 0.5:
                return False
        return True

    def _held_offset(state: State,
                     robot_obj: Object) -> Tuple[float, float, float]:
        """(EE - held object) offset, or zeros if nothing is held.

        The xy components apply under ``compensate_held_offset``; the z
        component under ``compensate_held_z``.
        """
        if not (compensate_held_offset or compensate_held_z):
            return 0.0, 0.0, 0.0
        for obj in state:
            if obj == robot_obj or \
                    "is_held" not in obj.type.feature_names:
                continue
            if state.get(obj, "is_held") > 0.5:
                dx = state.get(robot_obj, "x") - state.get(obj, "x")
                dy = state.get(robot_obj, "y") - state.get(obj, "y")
                dz = state.get(robot_obj, "z") - state.get(obj, "z")
                return ((dx, dy) if compensate_held_offset else (0.0, 0.0)) \
                    + ((dz, ) if compensate_held_z else (0.0, ))
        return 0.0, 0.0, 0.0

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, yaw = float(params[0]), float(params[1]), float(params[3])
        off_x, off_y, _ = _held_offset(state, objects[0])
        return x + off_x, y + off_y, cfg.transport_z, yaw

    def _drop_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del cfg  # unused
        x, y = float(params[0]), float(params[1])
        drop_z, yaw = float(params[2]), float(params[3])
        off_x, off_y, off_z = _held_offset(state, objects[0])
        return x + off_x, y + off_y, drop_z + off_z, yaw

    # With release_until_ungrasped, the drop-pose opening is
    # grasp-relative instead of full-span: open gradually until the
    # simulator drops the grasp constraint, open a few millimetres more
    # to clear the pads, retreat HOLDING that width (an "open" nudge
    # would keep widening next to the placed object's neighbors, and a
    # "closed" nudge would re-pinch it), and only open fully once back
    # at transport height, clear of the scene.
    partial_release = config.release_until_ungrasped

    def _settle_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        assert settle_to_contact_depth is not None
        x, y, z, yaw = _drop_pose(state, objects, params, cfg)
        return x, y, z - settle_to_contact_depth, yaw

    def _settled_or_at_depth(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> bool:
        # Contact ends the stroke; reaching the full depth (nothing
        # under the object within the budget) is the fallback so the
        # phase always terminates.
        if _held_assembly_in_contact(state):
            return True
        robot_obj = objects[0]
        current = (state.get(robot_obj,
                             "x"), state.get(robot_obj,
                                             "y"), state.get(robot_obj, "z"))
        tx, ty, tz, _ = _settle_pose(state, objects, params, cfg)
        squared_dist = float(
            np.sum(np.square(np.subtract(current, (tx, ty, tz)))))
        return squared_dist < cfg.move_to_pose_tol

    phases = []
    # A place's first move starts right after a pick, where a shallow
    # lift plus grasp-constraint droop can leave the held object modeled
    # grazing the surface it was picked from; allow those shallow start
    # contacts (the first motion is away from the surface) instead of
    # rejecting the whole plan at the start config.
    if use_move_above:
        phases.append(
            make_move_to_phase("MoveAbove",
                               _above_pose,
                               "closed",
                               allow_shallow_held_object_contacts=True))
    phases.append(
        make_move_to_phase(
            "Descend" if use_move_above else "MoveToDrop",
            _drop_pose,
            "closed",
            # Without a move-above, this is the post-pick first move
            # (see above). With a settle stroke, a failed verification
            # rewinds HERE while the held object rests on its support
            # (the stroke ended at contact); that start contact is
            # escapable -- the first motion is back up to release_z.
            allow_shallow_held_object_contacts=(not use_move_above
                                                or settle_to_contact_depth
                                                is not None),
            check_release_clearance=True))
    if settle_to_contact_depth is not None:

        def _held_xy_on_target(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> bool:
            # The held object itself (not the EE) must sit on the
            # commanded (x, y) before we let go. Nothing held (already
            # released, or the grasp broke) is unverifiable: pass.
            del cfg  # unused
            assert verify_xy_tol is not None
            tx, ty = float(params[0]), float(params[1])
            robot_obj = objects[0]
            for obj in state:
                if obj == robot_obj or \
                        "is_held" not in obj.type.feature_names:
                    continue
                if state.get(obj, "is_held") > 0.5:
                    err = float(
                        np.hypot(
                            state.get(obj, "x") - tx,
                            state.get(obj, "y") - ty))
                    return err <= verify_xy_tol
            return True

        # Gentle stroke: 3 mm steps bound the post-contact overshoot
        # (contact is only observed at the next policy step) and arm
        # the joint-jump guard -- single-shot IK once answered a plain
        # 2 cm descent with a wrist-flipped branch, and the flipped
        # retreat then batted the released block across the table.
        phases.append(
            make_move_to_phase(
                "SettleToContact",
                _settle_pose,
                "closed",
                expect_contact=True,
                use_motion_planning=False,
                terminal_fn=_settled_or_at_depth,
                max_step_norm=0.003,
                verify_fn=(_held_xy_on_target
                           if verify_xy_tol is not None else None),
                retry_to_phase=("Descend" if use_move_above else "MoveToDrop"),
                max_retries=(verify_max_retries
                             if verify_xy_tol is not None else 0)))
    if partial_release:
        phases.extend([
            Phase(
                name="OpenFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_release_open_target,
                finger_direction="open",
                terminal_fn=_nothing_held,
                anchor_finger_target=True,
            ),
            Phase(
                name="ClearFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_clear_fingers_target,
                finger_direction="open",
                anchor_finger_target=True,
                # The default grasp_tol accepts ~2 cm of finger error -
                # wider than the whole clear-slack travel, which would
                # terminate the phase before the fingers move.
                finger_tol=1e-6,
            ),
            make_move_to_phase("Retreat", _above_pose, "hold"),
            Phase(
                name="FullyOpenFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_open_fingers_target,
                finger_direction="open",
            ),
        ])
    else:
        phases.extend([
            Phase(
                name="OpenFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_open_fingers_target,
                finger_direction="open",
            ),
            make_move_to_phase("Retreat", _above_pose, "open"),
        ])

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode="home").build()
