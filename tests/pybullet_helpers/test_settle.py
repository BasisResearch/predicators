"""Tests for the quasi-static settle of a drawn PyBullet scene."""
import numpy as np
import pybullet as p
import pytest

from predicators.pybullet_helpers.settle import held_bodies, settle_bodies

_HALF = 0.05  # half the side of each test cube
_TABLE_TOP = 0.05


def _box(client, position, mass=1.0, roll=0.0):
    shape = p.createCollisionShape(p.GEOM_BOX,
                                   halfExtents=[_HALF] * 3,
                                   physicsClientId=client)
    return p.createMultiBody(baseMass=mass,
                             baseCollisionShapeIndex=shape,
                             basePosition=position,
                             baseOrientation=p.getQuaternionFromEuler(
                                 [roll, 0.0, 0.0]),
                             physicsClientId=client)


def _scene(client):
    """A table with a cube sunk 8 mm into it and tilted, a cube sunk 5 mm into
    that one, a cube hovering 1 cm above the table, a cube 10 cm above it, a
    cube just under the high one, and a cube hanging 1 cm off the pivot of a
    constraint to the world."""
    p.setGravity(0, 0, -9.8, physicsClientId=client)
    table_shape = p.createCollisionShape(p.GEOM_BOX,
                                         halfExtents=[1.0, 1.0, 0.05],
                                         physicsClientId=client)
    table = p.createMultiBody(baseMass=0.0,
                              baseCollisionShapeIndex=table_shape,
                              basePosition=[0.0, 0.0, 0.0],
                              physicsClientId=client)
    rest = _TABLE_TOP + _HALF
    sunk = _box(client, [0.0, 0.0, rest - 0.008], roll=0.05)
    stacked = _box(client, [0.0, 0.0, rest + 2 * _HALF - 0.005])
    hovering = _box(client, [0.4, 0.0, rest + 0.01])
    high = _box(client, [-0.4, 0.0, rest + 0.30])
    under_high = _box(client, [-0.4, 0.0, rest + 0.30 - 2 * _HALF - 0.001])
    hanging = _box(client, [0.4, 0.4, 0.51])
    p.createConstraint(hanging,
                       -1,
                       -1,
                       -1,
                       p.JOINT_POINT2POINT, [0, 0, 0], [0, 0, 0],
                       [0.4, 0.4, 0.50],
                       physicsClientId=client)
    return table, sunk, stacked, hovering, high, under_high, hanging


def _position(client, body):
    return np.array(
        p.getBasePositionAndOrientation(body, physicsClientId=client)[0])


def test_held_bodies_follow_support_and_attachment(physics_client_id):
    """Resting (within the contact distance), stacked and tied bodies are held;
    a body in the air, and one touching it from below, are not."""
    client = physics_client_id
    table, sunk, stacked, hovering, high, under_high, hanging = \
        _scene(client)
    bodies = [sunk, stacked, hovering, high, under_high, hanging]
    held = held_bodies(client, bodies, anchors=[table])
    assert held == {sunk, stacked, hovering, hanging}


def test_settle_rests_held_bodies_and_keeps_the_others(physics_client_id):
    """Held bodies come to rest on their supports and attachments with no
    velocity left; bodies held up by nothing keep their poses exactly."""
    client = physics_client_id
    table, sunk, stacked, hovering, high, under_high, hanging = \
        _scene(client)
    bodies = [sunk, stacked, hovering, high, under_high, hanging]
    before = {
        b: p.getBasePositionAndOrientation(b, physicsClientId=client)
        for b in (high, under_high)
    }
    moved = settle_bodies(client, bodies, anchors=[table])
    assert moved == {sunk, stacked, hovering, hanging}
    rest = _TABLE_TOP + _HALF
    assert _position(client, sunk)[2] == pytest.approx(rest, abs=2e-3)
    roll = p.getEulerFromQuaternion(
        p.getBasePositionAndOrientation(sunk, physicsClientId=client)[1])[0]
    assert abs(roll) < 0.01
    assert _position(client, stacked)[2] == pytest.approx(rest + 2 * _HALF,
                                                          abs=3e-3)
    assert _position(client, hovering)[2] == pytest.approx(rest, abs=2e-3)
    np.testing.assert_allclose(_position(client, hanging), [0.4, 0.4, 0.50],
                               atol=2e-3)
    for body, pose in before.items():
        now = p.getBasePositionAndOrientation(body, physicsClientId=client)
        np.testing.assert_allclose(now[0], pose[0], atol=1e-9)
        np.testing.assert_allclose(now[1], pose[1], atol=1e-9)
    for body in bodies:
        linear, angular = p.getBaseVelocity(body, physicsClientId=client)
        assert np.allclose(linear, 0.0) and np.allclose(angular, 0.0)
    # The settled scene is at rest under the engine's own dynamics.
    for _ in range(60):
        p.stepSimulation(physicsClientId=client)
    assert _position(client, sunk)[2] == pytest.approx(rest, abs=2e-3)
    assert _position(client, stacked)[2] == pytest.approx(rest + 2 * _HALF,
                                                          abs=3e-3)
