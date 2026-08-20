"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Any, Collection, Dict, Iterator, List, Optional, Sequence, \
    Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box
from numpy.typing import NDArray

from predicators import utils
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG

# Escape allowance for ROBOT links already in modeled contact at the
# start configuration (see run_motion_planning): near the start, the
# path may keep such a contact, but never more than this much deeper
# than it began (meters).
_START_ESCAPE_DEPTH_SLACK = 0.003
# Deepest robot-link start contact that still earns the start-escape
# allowance. Deliberately its own bound rather than the shallow
# held-object margin: the rule's motivating reconstruction error is a
# finger or wrist link 5-15 mm inside the object it just grasped or
# settled onto (execution-side sag and settle are not in the feature
# model), well beyond the ~6 mm shallow margin. The allowance is
# escape-only (never deeper than the start, only near the start), so a
# generous depth here cannot be exploited elsewhere on the path.
_ROBOT_START_ESCAPE_MAX_DEPTH = -0.02
# ... and only while within this max-abs joint distance of the start
# configuration (radians for revolute joints). Beyond it -- and at any
# goal further away -- full margins apply, so the allowance cannot be
# exploited elsewhere on the path. The same radius bounds how far a
# body's start-earned contact-partner status carries; see
# run_motion_planning.
_START_LOCAL_JOINT_RADIUS = 0.5
# Margin applied to a MOVABLE body whose contact-partner status was
# earned only at the start configuration, once the path has left that
# start neighborhood: touching stays legal, penetration does not.
_DEMOTED_PARTNER_MARGIN = 0.0


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: Optional[int] = None,
    base_link_to_held_obj: Optional[NDArray] = None,
    held_attachments: Optional[Dict[int, Any]] = None,
    allow_shallow_held_object_contacts: bool = False,
    unbounded_shallow_bodies: Optional[Collection[int]] = None,
    goal_finger_joint: Optional[float] = None,
    held_bystander_clearance: Optional[float] = None,
    goal_candidates: Optional[Sequence[JointPositions]] = None,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Collision bodies near the robot or held object at the start or goal
    configuration are treated as intended contact partners and checked
    against ``CFG.pybullet_birrt_contact_margin``; all other bodies are
    bystanders from which the path must keep
    ``CFG.pybullet_birrt_bystander_clearance`` of separation.

    Partner status earned SOLELY at the start configuration is local to
    it: a movable body the robot merely happens to begin near is checked
    with the hard margin only while the path stays within a joint-space
    radius of the start, and with a no-penetration margin beyond it.
    Contact the start forces on us says nothing about what the path may
    do to that body half a metre later, and the hard margin tolerates
    enough penetration to shove a free-standing object (a retreat after
    a glue dab repeatedly nudged an assembled row it had grazed on the
    way out). Static bodies keep their partner margin throughout -- they
    cannot be shoved, so grazing them is a modeling artifact rather than
    a physical event -- and goal-earned partners keep it too, since the
    path is deliberately approaching them.

    ``held_attachments`` maps bodies rigidly attached to the held object
    (e.g. the welded members of a glued assembly) to their
    base-link-relative transforms, in the same ``(position, orientation)``
    convention as ``base_link_to_held_obj``. They move with the arm and
    are collision-checked exactly like the held object itself, so a
    transported assembly cannot sweep an unchecked member through a
    bystander. They must not appear in ``collision_bodies``.

    When ``goal_finger_joint`` is given, the goal configuration is
    additionally checked with both finger joints at that value (e.g. a
    place phase whose next phase opens the gripper: the opening sweep is
    not otherwise planned, so a goal whose open fingers would hit a
    neighbor must be rejected here).

    ``held_bystander_clearance`` overrides
    ``CFG.pybullet_birrt_held_bystander_clearance`` (the wider berth the
    held object keeps from bodies the path never intends to approach).

    ``goal_candidates`` (optional) supplies pose-equivalent IK branches
    for the goal; the first collision-free one replaces
    ``target_positions`` as the planning goal. Branches reach the same
    end-effector pose with different arm configurations, so goal-config
    collision is a property of the BRANCH, not the pose -- selecting
    among them here removes the per-IK-seed luck from goal acceptance.

    ``unbounded_shallow_bodies`` (used with
    ``allow_shallow_held_object_contacts``): bodies -- static supports
    like tables -- whose START-state contacts with the held assembly
    are allowed at ANY depth, not just down to the shallow margin. A
    lift-off phase legitimately begins with the held assembly resting
    on its support, and planning-time modeling artifacts can show that
    resting contact tens of mm deep (a welded row's outer span was
    once modeled 21 mm into the table it sat on); escaping away from a
    static support is always safe, whereas deep start penetration into
    a movable body still signals genuine trouble and keeps the margin.

    ROBOT links get an analogous (always-on) start-escape allowance: a
    robot-vs-body contact already present at the start configuration
    and no deeper than ``_ROBOT_START_ESCAPE_MAX_DEPTH`` does not
    reject the path near the start, as long as it never deepens beyond
    how it began (plus a
    small slack) and the configuration stays within a joint-space
    radius of the start. The planning scene is reconstructed from
    observable features, so a phase that begins right after a grasp or
    a settled place can model a finger or wrist link several mm inside
    the object it just touched; the start is a fact, and escaping from
    it is strictly better than the guaranteed option failure that
    rejecting it produces.

    Note that this function changes the state of the robot.
    """
    rng = np.random.default_rng(seed)
    # BiRRT plans in the arm-joint space. For mobile robots, action_space also
    # includes base-delta dims (appended last); strip them so sampled configs
    # match the arm joints that set_joints / forward_kinematics expect. For
    # fixed-base robots (base_action_dim == 0) this is a no-op.
    joint_space = robot.action_space
    base_dim = int(getattr(robot, "base_action_dim", 0))
    if base_dim > 0:
        joint_space = Box(low=np.asarray(joint_space.low[:-base_dim]),
                          high=np.asarray(joint_space.high[:-base_dim]),
                          dtype=np.float32)
    joint_space.seed(seed)
    num_interp = CFG.pybullet_birrt_extend_num_interp

    def _sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
        new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    # The held object and every body rigidly attached to it move with
    # the arm; collision-wise they form one assembly.
    held_assembly: List[Tuple[int, Any]] = []
    if held_object is not None:
        assert base_link_to_held_obj is not None
        held_assembly.append((held_object, base_link_to_held_obj))
        held_assembly.extend((held_attachments or {}).items())

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        if held_assembly:
            world_to_base_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=physics_client_id).com_pose
            for body, base_link_to_obj in held_assembly:
                world_to_obj = p.multiplyTransforms(world_to_base_link[0],
                                                    world_to_base_link[1],
                                                    base_link_to_obj[0],
                                                    base_link_to_obj[1])
                p.resetBasePositionAndOrientation(
                    body,
                    world_to_obj[0],
                    world_to_obj[1],
                    physicsClientId=physics_client_id)

    hard_margin = CFG.pybullet_birrt_contact_margin
    shallow_margin = CFG.pybullet_birrt_shallow_held_contact_margin
    bystander_clearance = CFG.pybullet_birrt_bystander_clearance

    # Body id -> the penetration depth the held assembly may keep
    # against it along the path (an escape allowance for contacts the
    # start config already has). Normal shallow-contact bodies get the
    # shallow margin; unbounded bodies (static supports) get whatever
    # depth the start shows, minus a little slack -- the start is
    # escapable at any modeled depth, but the path can never go DEEPER
    # than it began.
    allowed_shallow_held_margins: Dict[int, float] = {}
    if allow_shallow_held_object_contacts and held_assembly:
        _set_state(initial_positions)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_bodies:
            penetrating: List[float] = []
            for assembly_body, _ in held_assembly:
                contacts = p.getContactPoints(
                    assembly_body, body, physicsClientId=physics_client_id)
                penetrating.extend(c[8] for c in contacts
                                   if c[8] < hard_margin)
            if not penetrating:
                continue
            start_depth = min(penetrating)
            if unbounded_shallow_bodies is not None and \
                    body in unbounded_shallow_bodies:
                allowed_shallow_held_margins[body] = min(
                    shallow_margin, start_depth - 0.003)
            elif start_depth >= shallow_margin:
                allowed_shallow_held_margins[body] = shallow_margin

    # Robot links, like the held assembly, can begin a phase already in
    # modeled contact: the planning scene is reconstructed from
    # observable features, and right after a grasp or a settled place
    # that reconstruction can show a finger or wrist link 5-15 mm
    # inside the object it just touched (execution-side sag and settle
    # are not in the feature model). The start configuration is a fact,
    # not a choice -- rejecting it fails the whole option with
    # certainty -- so a start contact no deeper than
    # _ROBOT_START_ESCAPE_MAX_DEPTH gets a per-body escape allowance:
    # near the start the path may
    # keep that contact, never more than _START_ESCAPE_DEPTH_SLACK
    # deeper than it began, and only within
    # _START_LOCAL_JOINT_RADIUS of the start configuration. Deeper
    # start penetration still signals genuine scene corruption and
    # keeps the hard rejection.
    allowed_robot_escape_margins: Dict[int, float] = {}
    _set_state(initial_positions)
    p.performCollisionDetection(physicsClientId=physics_client_id)
    for body in collision_bodies:
        contacts = p.getContactPoints(robot.robot_id,
                                      body,
                                      physicsClientId=physics_client_id)
        depths = [c[8] for c in contacts if c[8] < hard_margin]
        if not depths:
            continue
        start_depth = min(depths)
        if start_depth >= _ROBOT_START_ESCAPE_MAX_DEPTH:
            allowed_robot_escape_margins[body] = \
                start_depth - _START_ESCAPE_DEPTH_SLACK

    # Bodies the robot or held object starts or deliberately ends within
    # the clearance of are intended contact partners (support surfaces,
    # grasp targets, placement neighbors) and keep the hard margin;
    # every other body is a bystander from which the whole path must
    # keep ``bystander_clearance`` of separation, so that a "collision-
    # free" path cannot physically graze it (the hard margin tolerates
    # ~1mm of penetration, enough to topple a knife-edge object).
    #
    # The held object additionally gets the larger
    # ``pybullet_birrt_held_bystander_clearance`` against bodies the path
    # has no business approaching at all: unlike the position-controlled
    # robot links, the held object hangs on a grasp constraint and lags
    # the end effector's mid-path orientation swings by ~0.05 rad
    # (centimetres at the tip of a long object), so a plan that clears a
    # bystander by only ``bystander_clearance`` still physically grazes
    # it. Bodies already within the held clearance at the start or goal
    # keep the plain ``bystander_clearance`` so deliberately tight
    # placements (butt joints, chain neighbors) stay plannable.
    held_clearance = held_bystander_clearance \
        if held_bystander_clearance is not None \
        else CFG.pybullet_birrt_held_bystander_clearance
    held_body_clearances: dict = {}
    contact_partners: set = set(collision_bodies)
    # Movable bodies that earned partner status only at the start (see
    # the docstring): their hard margin expires with the start
    # neighborhood.
    demoted_partners: set = set()
    if bystander_clearance > hard_margin:
        contact_partners = set()
        endpoint_partners: List[set] = [set(), set()]
        held_probe_radius = max(bystander_clearance, held_clearance)
        held_near_endpoint: set = set()
        for endpoint_idx, pt in enumerate(
            (initial_positions, target_positions)):
            _set_state(pt)
            for body in collision_bodies:
                if p.getClosestPoints(robot.robot_id,
                                      body,
                                      bystander_clearance,
                                      physicsClientId=physics_client_id):
                    contact_partners.add(body)
                    endpoint_partners[endpoint_idx].add(body)
                # Evaluate held proximity at EVERY endpoint for EVERY
                # body, even one already a partner: partner status
                # (within the clearance) at EITHER endpoint must win,
                # and this probe also feeds held_near_endpoint and the
                # per-endpoint demotion bookkeeping. Skipping bodies
                # already seen near the other endpoint once made a
                # butt-joint neighbor 3.1 mm away at the start but
                # 1.8 mm away at the (re-aimed) goal a permanent
                # bystander; skipping bodies the robot is near left a
                # robot-earned start partner that the held assembly
                # deliberately approaches at the goal out of
                # endpoint_partners[1] (wrongly demoting it to the
                # zero margin) and out of held_near_endpoint (wrongly
                # holding an intended contact to the wide held
                # clearance).
                if not held_assembly:
                    continue
                held_dists: List[float] = []
                for assembly_body, _ in held_assembly:
                    held_pts = p.getClosestPoints(
                        assembly_body,
                        body,
                        held_probe_radius,
                        physicsClientId=physics_client_id)
                    held_dists.extend(held_pt[8] for held_pt in held_pts)
                if held_dists:
                    if min(held_dists) < bystander_clearance:
                        contact_partners.add(body)
                        endpoint_partners[endpoint_idx].add(body)
                    held_near_endpoint.add(body)
        for body in endpoint_partners[0] - endpoint_partners[1]:
            # Base mass 0 marks a static body (table, wall): nothing the
            # path does can displace it, so its partner margin stands.
            if p.getDynamicsInfo(body, -1,
                                 physicsClientId=physics_client_id)[0] > 0:
                demoted_partners.add(body)
        if held_assembly and held_clearance > bystander_clearance:
            held_body_clearances = {
                body: held_clearance
                for body in collision_bodies if body not in held_near_endpoint
            }

    def _extend_fn(pt1: JointPositions,
                   pt2: JointPositions) -> Iterator[JointPositions]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointPositions) -> bool:
        _set_state(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        # Use a penetration margin to distinguish real collisions from
        # resting-on-surface contacts (e.g. a grasped object sitting on a
        # table, or a gripper touching a just-released object).
        # getContactPoints returns tuples where index 8 is contactDistance:
        # negative = penetration, ~0 = touching, positive = separation.
        # Contact partners fail only on penetration deeper than the hard
        # margin; bystanders additionally fail on separations below the
        # clearance (Bullet generates contact points out to its
        # contactBreakingThreshold, 0.02 by default, so millimetre-scale
        # positive distances are reported here).
        near_start = True
        if allowed_robot_escape_margins or demoted_partners:
            near_start = float(
                np.max(np.abs(np.subtract(
                    pt, initial_positions)))) < _START_LOCAL_JOINT_RADIUS
        robot_escape_active = bool(allowed_robot_escape_margins) and near_start
        for body in collision_bodies:
            margin = hard_margin if body in contact_partners \
                else bystander_clearance
            if not near_start and body in demoted_partners:
                margin = _DEMOTED_PARTNER_MARGIN
            robot_margin = margin
            if robot_escape_active and body in allowed_robot_escape_margins:
                robot_margin = allowed_robot_escape_margins[body]
            contacts = p.getContactPoints(robot.robot_id,
                                          body,
                                          physicsClientId=physics_client_id)
            if any(c[8] < robot_margin for c in contacts):
                return True
            for assembly_body, _ in held_assembly:
                # Clearances above Bullet's contactBreakingThreshold
                # (0.02 m) would be silently unenforced: getContactPoints
                # generates no points beyond it. Query closest points out
                # to the clearance instead when the held clearance
                # applies.
                held_margin = held_body_clearances.get(body, margin)
                contacts = p.getClosestPoints(
                    assembly_body,
                    body,
                    held_margin,
                    physicsClientId=physics_client_id) \
                    if held_margin > 0 else p.getContactPoints(
                        assembly_body, body,
                        physicsClientId=physics_client_id)
                contact_distances = [c[8] for c in contacts]
                escape_margin = allowed_shallow_held_margins.get(body)
                if escape_margin is not None:
                    if any(d < escape_margin for d in contact_distances):
                        return True
                    continue
                if any(d < held_margin for d in contact_distances):
                    return True
        return False

    # Collision-aware goal selection: the caller may supply several
    # pose-equivalent IK branches (see _solve_goal_ik_candidates). The
    # branches reach the same end-effector pose but differ in arm
    # configuration, and they are NOT collision-equivalent: one grasp
    # branch can sweep a link ~6 cm through a neighboring block while
    # another clears it. Which branch a single IK solve lands on is
    # seed-dependent, so goal-config rejection used to be a per-seed
    # coin flip (a validation repeat failed with the robot modeled
    # 59 mm inside a standing leg at a grasp goal that other repeats
    # planned fine). Take the first branch whose goal configuration
    # (and fingers-open variant, when a release follows) is collision-
    # free; when none passes, keep the primary target so the caller's
    # failure diagnostics report its contacts.
    if goal_candidates is not None:
        for cand in goal_candidates:
            cand_list = list(cand)
            if _collision_fn(cand_list):
                continue
            if goal_finger_joint is not None:
                cand_release = list(cand_list)
                cand_release[robot.left_finger_joint_idx] = goal_finger_joint
                cand_release[robot.right_finger_joint_idx] = goal_finger_joint
                if _collision_fn(cand_release):
                    continue
            target_positions = cand_list
            break

    if goal_finger_joint is not None:
        release_config = list(target_positions)
        release_config[robot.left_finger_joint_idx] = goal_finger_joint
        release_config[robot.right_finger_joint_idx] = goal_finger_joint
        if _collision_fn(release_config):
            return None

    def _distance_fn(from_pt: JointPositions, to_pt: JointPositions) -> float:
        # NOTE: only using positions to calculate distance. Should use
        # orientations as well in the near future.
        from_ee = robot.forward_kinematics(from_pt).position
        to_ee = robot.forward_kinematics(to_pt).position
        return sum(np.subtract(from_ee, to_ee)**2)

    birrt = utils.BiRRT(_sample_fn,
                        _extend_fn,
                        _collision_fn,
                        _distance_fn,
                        rng,
                        num_attempts=CFG.pybullet_birrt_num_attempts,
                        num_iters=CFG.pybullet_birrt_num_iters,
                        smooth_amt=CFG.pybullet_birrt_smooth_amt)

    path = birrt.query(initial_positions, target_positions)
    if path is not None and CFG.pybullet_birrt_path_subsample_ratio > 1:
        ratio = CFG.pybullet_birrt_path_subsample_ratio
        last = path[-1]
        path = [path[i] for i in range(0, len(path), ratio)]
        # Always include the final waypoint.
        if path[-1] is not last:
            path.append(last)
    return path
