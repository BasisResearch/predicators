"""Counterfactual clean-push probe for the domino cascade certificate.

The certificate's legitimacy question is "does the layout the robot
built actually work as a domino cascade?", and this module answers it
by physics instead of geometric forensics: from the recorded pre-push
state it re-runs the episode's own Push with a disembodied finger - a
dynamic fingertip-sized box driven along the Push skill's waypoint
path with the plan's recorded continuous parameters - and reports
whether the goal atoms hold after the cascade settles. The robot's arm
never moves, so collateral arm contact (the confound the old
swept-corridor attribution rules had to guess at) cannot contribute:
if the goal topples in the probe, the layout genuinely cascades; if it
never does, the real episode's topples are owed to the robot's body,
not the built chain.

Fidelity to the skill (``ground_truth_models/skill_factories/push.py``):
the finger materializes at the skill's behind-transport waypoint,
descends to contact height behind the green, and strokes to the
block's center xy - the same three pre-retreat waypoints, computed
with the same ``facing = (sin yaw, cos yaw)`` geometry and the
episode's actual ``(approach_distance, contact_z_offset)`` when the
step labels carry them. The finger is driven through a pose constraint
(a motor, not a teleport), so when the stroke ends at the block's
center it keeps PRESSING exactly like the arm's position-controlled
motors do - the follow-through that decides marginal topples. The two
deliberate departures are the collateral-only segments: there is no
arm body, and the retreat is a straight lift instead of the skill's
home sweep (the recorded arm-knock hacks live in the descent sweep of
the gripper BODY and the retreat - neither ever contributes to the
intended topple).

Corner layouts are contact-history knife-edge (see
``min_block_utils``), and a single replay can diverge from the real
rollout in either direction, so the probe retries the SAME push at a
few stroke speeds (execution-noise bracket, not different pushes) and
certifies on the first success. Everything is deterministic: no RNG,
fixed schedule, velocities zeroed before every attempt.

The probe runs in a dedicated probe env - a fresh instance of the
certifying env's class, physics mirrored via
``DominoComponent.physical_param_override`` - so it never contaminates
the certifying env's world (cross-episode residuals) and stays valid
for both the true env and an agent's belief env: each side probes with
its own physics, which is exactly the trust contract of belief-side
verdicts. States transfer between the worlds by the same mechanism the
option model relies on: same-class envs build the same body pool in
the same order, so the pybullet ids stamped on the state's objects
coincide.
"""

import logging
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.structs import Action, GroundAtom, Object, State

# Fallback Push parameters (approach_distance, contact_z_offset) for
# episodes whose step labels carry no continuous parameters (legacy
# 2-tuple labels, label-free certification). Matches the task
# generators' canonical probe push (``min_block_utils._PUSH_PARAMS``).
_CANONICAL_PUSH_PARAMS: Tuple[float, ...] = (0.04, 0.05)

# Stroke speed of the real end-effector during the push-to-object
# phase, measured on recorded episodes (m per env step); the descent is
# contact-free in a legal push, so its exact speed only costs steps.
_STROKE_SPEED = 0.008
_DESCEND_SPEED = 0.012

# Retries of the SAME push at bracketed stroke speeds: knife-edge
# cascades flip on residual-state noise between the real rollout and a
# replay, and execution speed is the one quantity the waypoint skill
# does not pin down (it comes from the arm controller). Any success is
# a success of the plan's own push.
_SPEED_FACTORS: Tuple[float, ...] = (1.0, 0.7, 1.4)

# The skill holds its final commanded pose until the phase terminates;
# a few pressed steps at the stroke's end reproduce that dwell.
_HOLD_STEPS = 8

# Retreat: lift the finger straight up after the stroke (the clean-push
# counterfactual has no sideways home sweep to knock anything).
_LIFT_STEPS = 5
_LIFT_DZ = 0.06

# No-op env steps after the stroke for the cascade to propagate and
# settle. A hop takes at most CASCADE_WINDOW_STEPS (24) env steps and
# recorded chains run 2-4 hops.
_SETTLE_STEPS = 100

# Finger half extents (m): a fingertip-sized box, comfortably narrower
# than a domino's width (0.07) so it strikes only the block it aims at.
_FINGER_HALF_EXTENTS = (0.01, 0.01, 0.02)
# The finger is dynamic and driven through a fixed constraint - a motor,
# not a teleport - so contact resolves with sustained force the way the
# arm's position-controlled joints press. The force cap is far above
# what toppling a 0.1 kg domino needs (the arm is effectively rigid at
# this scale) while staying finite for the solver.
_FINGER_MASS = 0.2
_FINGER_MAX_FORCE = 200.0


def _zero_all_velocities(physics_client_id: int) -> None:
    """Kill residual velocities on every body (see the velocity-residual
    lesson: pose resets do not clear velocities)."""
    for body in range(p.getNumBodies(physicsClientId=physics_client_id)):
        body_id = p.getBodyUniqueId(body, physicsClientId=physics_client_id)
        p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)


def _standing_goal_shortfall(env: Any, goal: Set[GroundAtom]) -> List[str]:
    """Names of goal atoms that do not hold in the env's current state."""
    # pylint: disable=protected-access
    final = env._get_state()
    return [
        str(atom) for atom in sorted(goal, key=str) if not atom.holds(final)
    ]


def _transport_z(env: Any) -> float:
    """The Push skill's transport height for this env.

    Mirrors ``PyBulletDominoGroundTruthOptionFactory._transport_z_push``
    (``table_height + domino_height * 1.5``); computed from the same
    env-class constants rather than imported to keep the env layer free
    of ground-truth-model imports.
    """
    return float(env.table_height + env.domino_height * 1.5)


def _drive_finger(env: Any, green: Object, state: State,
                  push_params: Tuple[float, ...], speed_factor: float) -> None:
    """Re-run the Push skill's stroke on ``green`` with a motorized finger.

    Waypoints replicate the skill's pre-retreat path for the given
    ``(approach_distance, contact_z_offset)``: appear behind the block
    at transport height, descend to contact height, stroke to the
    block's center xy, dwell pressed, then lift straight up. The finger
    is pulled toward each scheduled pose by a fixed constraint with a
    bounded force, so a blocked stroke presses instead of tunneling -
    the same sustained follow-through the arm's position-controlled
    motors deliver at the stroke's end.
    """
    # pylint: disable=protected-access
    approach, contact_z = push_params[0], push_params[1]
    cid = env._physics_client_id
    gx = state.get(green, "x")
    gy = state.get(green, "y")
    gz = state.get(green, "z")
    yaw = state.get(green, "yaw")
    facing = np.array([np.sin(yaw), np.cos(yaw)])
    behind_xy = np.array([gx, gy]) - facing * approach
    center_xy = np.array([gx, gy])
    contact_height = gz + contact_z

    start_pos = [*behind_xy, _transport_z(env)]
    collision = p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=_FINGER_HALF_EXTENTS,
                                       physicsClientId=cid)
    # A visual shape costs nothing in DIRECT mode and makes the probe
    # inspectable in renders/GUI debugging.
    visual = p.createVisualShape(p.GEOM_BOX,
                                 halfExtents=_FINGER_HALF_EXTENTS,
                                 rgbaColor=(0.9, 0.25, 0.15, 1.0),
                                 physicsClientId=cid)
    finger = p.createMultiBody(baseMass=_FINGER_MASS,
                               baseCollisionShapeIndex=collision,
                               baseVisualShapeIndex=visual,
                               basePosition=start_pos,
                               physicsClientId=cid)
    constraint = p.createConstraint(finger,
                                    -1,
                                    -1,
                                    -1,
                                    p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                    start_pos,
                                    physicsClientId=cid)
    noop = Action(np.array(env._pybullet_robot.initial_joint_positions))

    def _segment(target: np.ndarray, speed: float,
                 pose: np.ndarray) -> np.ndarray:
        """Drive the constraint from ``pose`` to ``target`` at ``speed``."""
        dist = float(np.linalg.norm(target - pose))
        n_steps = max(1, int(np.ceil(dist / speed)))
        for i in range(1, n_steps + 1):
            waypoint = pose + (target - pose) * (i / n_steps)
            p.changeConstraint(constraint,
                               jointChildPivot=waypoint.tolist(),
                               maxForce=_FINGER_MAX_FORCE,
                               physicsClientId=cid)
            env.step(noop)
        return target

    try:
        pose = np.array(start_pos)
        # Waypoint 1: descend behind the block to contact height.
        pose = _segment(np.array([*behind_xy, contact_height]),
                        _DESCEND_SPEED * speed_factor, pose)
        # Waypoint 2: the push stroke, to the block's center xy.
        pose = _segment(np.array([*center_xy, contact_height]),
                        _STROKE_SPEED * speed_factor, pose)
        # The skill holds its last commanded pose until the phase
        # terminates: dwell pressed at the stroke's end.
        for _ in range(_HOLD_STEPS):
            env.step(noop)
        # Retreat: straight up (deliberately not the skill's home sweep).
        _segment(
            np.array([*center_xy, contact_height + _LIFT_STEPS * _LIFT_DZ]),
            _LIFT_DZ, pose)
    finally:
        p.removeConstraint(constraint, physicsClientId=cid)
        p.removeBody(finger, physicsClientId=cid)


def run_counterfactual_push_probe(
        probe_env: Any,
        pre_push_state: State,
        greens: Sequence[Object],
        goal: Set[GroundAtom],
        push_params: Optional[Tuple[float, ...]] = None) -> Tuple[bool, str]:
    """Does the plan's own push, delivered by a clean finger, cascade to the
    goal?

    Re-runs the Push skill's stroke (waypoint path + the episode's
    recorded ``push_params``, falling back to the generators' canonical
    push when None) from ``pre_push_state`` in ``probe_env`` (a
    dedicated same-class env; see module docstring), retrying the same
    push at bracketed stroke speeds. Returns ``(ok, detail)``: ``ok``
    on the first attempt whose settled state satisfies every goal atom
    - or, after all attempts fail, the goal atoms the closest run left
    unsatisfied.
    """
    # pylint: disable=protected-access
    assert greens, "probe needs at least one pushed green block"
    params = tuple(push_params) if push_params else _CANONICAL_PUSH_PARAMS
    assert len(params) >= 2, f"malformed push params {params}"
    cid = probe_env._physics_client_id
    noop = Action(np.array(probe_env._pybullet_robot.initial_joint_positions))
    best_shortfall: Optional[List[str]] = None
    for speed_factor in _SPEED_FACTORS:
        probe_env._set_state(pre_push_state)
        # Park the arm out of the scene: the pre-push arm pose is
        # wherever the real arm hovered, and the counterfactual must not
        # let a parked arm body block (or brace) the cascade.
        probe_env._pybullet_robot.set_joints(
            probe_env._pybullet_robot.initial_joint_positions)
        _zero_all_velocities(cid)
        for green in greens:
            _drive_finger(probe_env, green, pre_push_state, params,
                          speed_factor)
        for _ in range(_SETTLE_STEPS):
            probe_env.step(noop)
        shortfall = _standing_goal_shortfall(probe_env, goal)
        if not shortfall:
            source = "the plan's" if push_params else "the canonical"
            detail = (f"{source} push (approach {params[0]:.2f} m, contact "
                      f"height +{params[1]:.2f} m) at stroke-speed factor "
                      f"{speed_factor:g} cascades to the goal")
            return True, detail
        if best_shortfall is None or len(shortfall) < len(best_shortfall):
            best_shortfall = shortfall
    assert best_shortfall is not None
    source = "the plan's own" if push_params else "the canonical"
    detail = (f"{source} push (approach {params[0]:.2f} m, contact height "
              f"+{params[1]:.2f} m) on {', '.join(g.name for g in greens)} "
              f"reaches the goal at none of {len(_SPEED_FACTORS)} stroke "
              f"speeds; closest run left {', '.join(best_shortfall)} "
              "unsatisfied")
    logging.debug("[cascade probe] %s", detail)
    return False, detail
