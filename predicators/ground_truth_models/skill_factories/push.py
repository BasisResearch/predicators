"""Push skill factory: creates a multi-phase push controller.

This module provides ``create_push_skill``, which builds a
``ParameterizedOption`` that pushes an object (e.g. domino, switch, button)
using a standard 4-waypoint trajectory:

  1. Closing the gripper.
  2. Moving above & behind the target at ``config.transport_z``.
  3. Descending to contact height (target z + ``contact_z_offset``).
  4. Pushing to the target position along its facing direction.
  5. Retreating to ``config.robot_home_pos``.
  6. Opening the gripper.

The "facing direction" is derived from the yaw returned by
``get_target_pose_fn`` as ``(sin(yaw), cos(yaw))``.  "Behind" means
opposite to the facing direction.

``config.robot_home_pos`` **must** be set.

Continuous parameters: ``(approach_distance, contact_z_offset)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_push_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
        robot_home_pos=(MyEnv.robot_init_x, MyEnv.robot_init_y,
                        MyEnv.robot_init_z),
    )

    def _get_domino_pose(state, objects, params, config):
        _, domino = objects
        return (state.get(domino, "x"), state.get(domino, "y"),
                state.get(domino, "z"), state.get(domino, "rot"))

    Push = create_push_skill(
        name="Push",
        types=[robot_type, domino_type],
        config=config,
        get_target_pose_fn=_get_domino_pose,
    )
"""

import math
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Push. The approach upper bound
# leaves room for a side-oriented gripper (yaw offset 0),
# whose body extends along the approach axis: a descend waypoint closer
# than ~0.07 m can itself collide with the pushed object. That caveat is
# stated in the description because the tool-facing params text is the
# only place an agent learns the advertised range's usable interior.
_PUSH_PARAMS = [
    ("approach_distance (dist behind target along facing dir to start push; "
     "small values put the descend waypoint inside the gripper's own "
     "footprint along the approach axis, colliding with the target)", 0.00,
     0.10),
    ("contact_z_offset (height above target z for contact; near-zero values "
     "descend into the target/support and can stall, near-max values may "
     "pass over a short target)", 0.0, 0.11),
]

# A push target pose is computed from the pose of the body it strikes,
# so that body sits within a few millimetres of it; anything farther is
# not the push's target.
CONTACT_TARGET_RADIUS = 0.02


def object_at_pose(state: State,
                   pose: Tuple[float, float, float],
                   exclude: Set[Object],
                   radius: float = CONTACT_TARGET_RADIUS) -> Optional[Object]:
    """The posed object nearest ``pose`` within ``radius``, or None.

    Objects without x/y/z features and those in ``exclude`` (a
    grounding's own arguments, the robot) are skipped.
    """
    best: Optional[Object] = None
    best_dist = radius
    for obj in state:
        if obj in exclude:
            continue
        feats = obj.type.feature_names
        if not {"x", "y", "z"}.issubset(feats):
            continue
        dist = math.sqrt(
            sum((state.get(obj, f) - v)**2 for f, v in zip("xyz", pose)))
        if dist < best_dist:
            best, best_dist = obj, dist
    return best


def resolve_ee_yaw_offset(config: SkillConfig) -> float:
    """The EE yaw offset Push should use, in radians.

    Which face of the gripper leads into the object is a property of the
    hand, so it comes from the robot unless the config forces one: a
    ``push_ee_yaw_offset`` entry in ``config.extra`` (one skill's own
    choice, e.g. fingers-first at a narrow handle the hand's width would
    not clear) or ``CFG.skill_push_ee_yaw_offset`` (every push at once).
    """
    if "push_ee_yaw_offset" in config.extra:
        return float(config.extra["push_ee_yaw_offset"])
    if CFG.skill_push_ee_yaw_offset is None:
        return config.robot.push_ee_yaw_offset
    return float(CFG.skill_push_ee_yaw_offset)


def create_push_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    freeze_stroke: bool = False,
    extra_params: Sequence[Tuple[str, float, float]] = (),
    stroke_step_norm_fn: Optional[Callable[[Array], float]] = None,
    stroke_overshoot_fn: Optional[Callable[[Array], float]] = None,
    plan_transit: Optional[bool] = None,
    stroke_rise_fn: Optional[Callable[[State, Sequence[Object], Array],
                                      float]] = None,
    open_hand: bool = False,
    lift_before_retreat: bool = False,
) -> ParameterizedOption:
    """Create a multi-phase push skill with a standard 4-waypoint trajectory.

    Phases:
        0. **CloseFingers** -- Close the gripper before approaching.
        1. **Waypoint_0** -- Move above & behind the target at
           ``config.transport_z``, offset by ``approach_distance``
           opposite the facing direction.
        2. **Waypoint_1** -- Descend to contact height
           (target z + ``contact_z_offset``) at the same behind position.
        3. **Waypoint_2** -- Push forward to the target position.
        4. **Waypoint_3** -- Retreat to ``config.robot_home_pos``.
        5. **OpenFingers** -- Open the gripper.

    Continuous parameters:
        ``(approach_distance, contact_z_offset)``

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration.  ``config.robot_home_pos`` and
            ``config.transport_z`` must be set.
        get_target_pose_fn: Callback returning ``(x, y, z, yaw)`` from
            ``(state, objects, params, config)``.  ``params`` will be empty.
        freeze_stroke: Aim the push stroke (Waypoint_2) at the target's
            pose on the stroke's first step and hold it. Required when
            the pushed body slides away on contact (a tile on ice): a
            stroke re-aimed every step chases the body to the arm's
            reach limit. A body that stays put (a switch, a domino that
            topples in place) needs no freeze.
        extra_params: Further ``(description, low, high)`` continuous
            parameters appended after the two standard ones. The
            waypoints ignore them; they are for ``stroke_step_norm_fn``
            and for the caller's own samplers.
        stroke_step_norm_fn: Makes the push stroke (Waypoint_2) a
            gentle stroke whose metres-per-step clamp is this function
            of the option's params (see ``Phase.step_norm_fn``): a push
            whose speed is a parameter, e.g. ``params[2] * dt`` with an
            extra ``speed`` parameter in m/s.
        stroke_overshoot_fn: Carries the push stroke this far PAST the
            target pose along the facing direction, as a function of the
            option's params: a push-through, e.g. a plunger driven back
            by an extra ``depth`` parameter. The default stroke ends at
            the target pose itself.
        plan_transit: Whether the two transit waypoints (above and
            behind the target, the descend) use BiRRT. ``None`` defers
            to ``CFG.skill_phase_use_motion_planning``; ``False`` steps
            IK straight at them, for a probe that runs the push on a
            scratch simulator where speed matters more than clearance.
        stroke_rise_fn: Lifts the END of the push stroke this far above
            the contact height, as a function of ``(state, objects,
            params)``: a stroke along a chord rather than level, e.g.
            a hanging ball drawn back along its arc, whose centre rises
            as it is pulled. The default stroke stays level.
        open_hand: Push with the fingers OPEN rather than closed: no
            closing phase, the waypoints hold the hand open, no opening
            phase at the end. With the fingers leading (yaw offset 0)
            the open fingertips straddle a round object so it cannot
            slip sideways off the hand, e.g. a hanging ball drawn back
            along its arc.
        lift_before_retreat: Insert a waypoint straight above the
            stroke's end at ``config.transport_z`` before the retreat
            home, so the hand leaves a released body vertically rather
            than sweeping across its path, e.g. a drawn-back pendulum
            that swings forward the moment the hand lets go.

    Returns:
        A ``ParameterizedOption`` implementing the push skill.
    """
    if config.robot_home_pos is None:
        raise ValueError(
            "config.robot_home_pos must be set for create_push_skill.")

    params_space, params_description = build_params_space(
        list(_PUSH_PARAMS) + list(extra_params))
    _empty = np.array([], dtype=np.float32)

    def _contact_objects(state: State,
                         objects: Sequence[Object]) -> Set[Object]:
        # The body at the target pose is what the push is for (a
        # faucet's switch, a fan's switch): it is exempt from clearance
        # checks like an argument, see PhaseSkill.contact_objects.
        x, y, z, _ = get_target_pose_fn(state, objects, _empty, config)
        target = object_at_pose(state, (x, y, z), exclude=set(objects))
        return set() if target is None else {target}

    # -- Standard 4-waypoint trajectory ----------------------------------

    def _waypoints(
        ox: float,
        oy: float,
        oz: float,
        oyaw: float,
        cfg: SkillConfig,
        s_offset_x: float,
        s_offset_z: float,
        overshoot: float,
        robot_xy: Tuple[float, float],
    ) -> List[Tuple[float, float, float, float, str]]:
        assert cfg.robot_home_pos is not None
        obj_xy = np.array([ox, oy])
        facing = np.array([np.sin(oyaw), np.cos(oyaw)])
        behind_xy = obj_xy - facing * s_offset_x
        push_xy = obj_xy + facing * overshoot
        home_xy = np.array(cfg.robot_home_pos[:2])
        home_z = cfg.robot_home_pos[2]
        ee_yaw = oyaw + resolve_ee_yaw_offset(cfg)
        wps = [
            (*behind_xy, cfg.transport_z, ee_yaw, "closed"),
            (*behind_xy, oz + s_offset_z, ee_yaw, "closed"),
            (*push_xy, oz + s_offset_z, ee_yaw, "closed"),
        ]
        if lift_before_retreat:
            # Straight up from wherever the hand is now.
            wps.append((*robot_xy, cfg.transport_z, ee_yaw, "closed"))
        wps.append((*home_xy, home_z, ee_yaw, "closed"))
        return wps

    n_waypoints = 5 if lift_before_retreat else 4

    # -- Phase construction -----------------------------------------------

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

    def _make_waypoint_position_fn(
        waypoint_idx: int,
    ) -> Callable[[State, Sequence[Object], Array, SkillConfig], Tuple[
            float, float, float, float]]:

        def _get_target(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            s_ox = float(params[0])
            s_oz = float(params[1])
            overshoot = (0.0 if stroke_overshoot_fn is None else float(
                stroke_overshoot_fn(params)))
            x, y, z, yaw = get_target_pose_fn(state, objects, _empty, cfg)
            robot_xy = (float(state.get(objects[0], "x")),
                        float(state.get(objects[0], "y")))
            wps = _waypoints(x, y, z, yaw, cfg, s_ox, s_oz, overshoot,
                             robot_xy)
            wx, wy, wz, wyaw, _ = wps[waypoint_idx]
            if waypoint_idx == 2 and stroke_rise_fn is not None:
                wz += float(stroke_rise_fn(state, objects, params))
            return wx, wy, wz, wyaw

        return _get_target

    hand = "open" if open_hand else "closed"
    phases: List[Phase] = []
    if not open_hand:
        phases.append(
            Phase(name="CloseFingers",
                  action_type=PhaseAction.CHANGE_FINGERS,
                  target_fn=_close_fingers_target,
                  finger_direction="close"))

    for i in range(n_waypoints):
        # Waypoint_2 (push into target) and the waypoints after it (the
        # retreat from the target) expect robot-object contact, so
        # suppress collision diagnostics.
        #
        # They must also NOT be motion-planned: their goal poses sit at (or
        # inside) the pushed object, and BiRRT plans a COLLISION-FREE path.
        # What that does is a knife-edge of scene and hand geometry. If the
        # goal config registers as colliding (every sim scene so far),
        # planning fails and the ``expect_contact`` fallback quietly runs
        # incremental IK. If it squeaks past the ~1 mm
        # ``pybullet_birrt_contact_margin`` (the real captured scenes), BiRRT
        # SUCCEEDS by routing around -- measured: over the block's top, 59 mm
        # out to the side, past the block, then down onto the goal from the
        # far side, so the last hop struck the block AGAINST the push
        # direction and toppled it backwards. The direction of travel is the
        # payload of a stroke, and only stepping IK straight at the target
        # guarantees it -- identically in sim and on the real bench.
        #
        # Deliberately not fixed by dropping the pushed object from the
        # planner's collision set: the planner then remains free to detour
        # around a BYSTANDER near the stroke (the same wrong-direction strike
        # one object over), and the restricted Push variant grounds as
        # ``[robot]`` alone, where an index into ``objects`` silently misses.
        # A stroke that cannot go straight should fail and be resampled, not
        # rerouted.
        phases.append(
            make_move_to_phase(
                name=f"Waypoint_{i}",
                get_target_pose_fn=_make_waypoint_position_fn(i),
                finger_status=hand,
                expect_contact=(i >= 2),
                use_motion_planning=(False if i >= 2 else plan_transit),
                freeze_target=(freeze_stroke and i == 2),
                step_norm_fn=(stroke_step_norm_fn if i == 2 else None)))

    if not open_hand:
        phases.append(
            Phase(name="OpenFingers",
                  action_type=PhaseAction.CHANGE_FINGERS,
                  target_fn=_open_fingers_target,
                  finger_direction="open"))

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode="home",
                      contact_objects_fn=_contact_objects).build()
