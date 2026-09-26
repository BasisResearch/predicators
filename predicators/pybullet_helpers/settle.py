"""Quasi-static settling of a PyBullet scene.

A scene written from a draw of a belief over object poses can put
bodies slightly inside one another, leave a resting body hovering above
its support or tilted on it, or stretch an attachment. The first engine
step resolves that with an impulse, which can knock a standing body
over, so a rollout would start from a different scene than the draw.
:func:`settle_bodies` resolves it quasi-statically instead: engine
substeps with every velocity zeroed after each, the anchors (the robot)
held, and no domain step, so no modeled mechanism acts. Only bodies the
static world holds up move. A body a mechanism holds up (a box lifted
by balloons, a ball in a wind stream) would sag under gravity alone, so
it keeps its pose.
"""
from typing import Collection, Dict, List, Sequence, Set, Tuple

import pybullet as p

# Substeps of the settle. With velocities zeroed after each, a body
# moves about g dt^2 per substep (0.17 mm at 240 Hz), so 240 substeps
# close a hover of a few centimetres.
SETTLE_SUBSTEPS = 240
# A body within this distance of a support below it rests on it
# (metres): about twice a centimetre of pose noise, and below the gap
# under a body that a mechanism holds up.
SUPPORT_EPS = 0.02
# A closest point supports the body above it when the normal on the
# support points at least this far upward.
_MIN_UP = 0.5

# A base pose: position and orientation quaternion.
Pose = Tuple[Sequence[float], Sequence[float]]


def held_bodies(client: int,
                bodies: Collection[int],
                anchors: Collection[int],
                eps: float = SUPPORT_EPS) -> Set[int]:
    """The bodies of ``bodies`` that the static world holds up.

    A body is held when it rests on a fixed or held body (a closest
    point within ``eps`` whose normal on the support points up) or when
    an engine constraint ties it to one or to the world. Fixed bodies
    are ``anchors`` and every body of zero mass.
    """
    candidates = set(bodies)
    everything = {
        p.getBodyUniqueId(i, physicsClientId=client)
        for i in range(p.getNumBodies(physicsClientId=client))
    }
    fixed = set(anchors) | {
        b
        for b in everything
        if p.getDynamicsInfo(b, -1, physicsClientId=client)[0] == 0.0
    }
    # The world, as the other end of a constraint with no child body.
    world = -1
    fixed.add(world)
    rests_on: Dict[int, Set[int]] = {b: set() for b in candidates}
    for body in candidates:
        for other in everything - {body}:
            points = p.getClosestPoints(body,
                                        other,
                                        eps,
                                        physicsClientId=client)
            # Index 7 is the normal on ``other``, pointing at ``body``.
            if any(point[7][2] >= _MIN_UP for point in points):
                rests_on[body].add(other)
    tied: Dict[int, Set[int]] = {b: set() for b in candidates}
    for i in range(p.getNumConstraints(physicsClientId=client)):
        uid = p.getConstraintUniqueId(i, physicsClientId=client)
        info = p.getConstraintInfo(uid, physicsClientId=client)
        parent, child = info[0], info[2]
        child = world if child < 0 else child
        if parent in tied:
            tied[parent].add(child)
        if child in tied:
            tied[child].add(parent)
    held: Set[int] = set()
    grew = True
    while grew:
        grew = False
        support = fixed | held
        for body in candidates - held:
            if rests_on[body] & support or tied[body] & support:
                held.add(body)
                grew = True
    return held | (candidates & fixed)


def _joint_positions(client: int, body: int) -> List[float]:
    return [
        p.getJointState(body, j, physicsClientId=client)[0]
        for j in range(p.getNumJoints(body, physicsClientId=client))
    ]


def _hold(client: int, body: int, pose: Pose, joints: List[float]) -> None:
    p.resetBasePositionAndOrientation(body,
                                      pose[0],
                                      pose[1],
                                      physicsClientId=client)
    p.resetBaseVelocity(body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                        physicsClientId=client)
    for j, q in enumerate(joints):
        p.resetJointState(body, j, q, 0.0, physicsClientId=client)


def _stop(client: int, body: int) -> None:
    p.resetBaseVelocity(body, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                        physicsClientId=client)
    for j, q in enumerate(_joint_positions(client, body)):
        p.resetJointState(body, j, q, 0.0, physicsClientId=client)


def settle_bodies(client: int,
                  bodies: Collection[int],
                  anchors: Collection[int],
                  substeps: int = SETTLE_SUBSTEPS,
                  eps: float = SUPPORT_EPS) -> Set[int]:
    """Settle the bodies of ``bodies`` the static world holds up.

    Runs ``substeps`` engine substeps, zeroing every body's velocity
    after each and holding ``anchors`` and the bodies that are not held
    (see :func:`held_bodies`) at their current poses and joint
    positions. Returns the bodies that were free to move.
    """
    moving = held_bodies(client, bodies, anchors, eps)
    pinned = [b for b in list(bodies) + list(anchors) if b not in moving]
    poses: Dict[int, Pose] = {}
    joints: Dict[int, List[float]] = {}
    for body in pinned:
        poses[body] = p.getBasePositionAndOrientation(body,
                                                      physicsClientId=client)
        joints[body] = _joint_positions(client, body)
    for _ in range(substeps):
        p.stepSimulation(physicsClientId=client)
        for body in pinned:
            _hold(client, body, poses[body], joints[body])
        for body in moving:
            _stop(client, body)
    return moving
