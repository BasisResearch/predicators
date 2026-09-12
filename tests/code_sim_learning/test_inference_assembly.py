"""Physical support and motion checks for an explicit assembly prior."""
from dataclasses import replace

import numpy as np
import pybullet as p
import pytest
from scipy.spatial.transform import Rotation

from predicators.code_sim_learning.inference_assembly import AssemblyBody, \
    RigidAssemblyPrior


def _prior(moving=False):
    return RigidAssemblyPrior(
        (AssemblyBody("box", ((0., 0., 0.), (0., 0., 0., 1.)), .06),
         AssemblyBody("balloon", ((0., 0., .12), (0., 0., 0., 1.)), .02)),
        ((0., 1.), ) * 3,
        linear_half_width=.2 if moving else 0.,
        angular_half_width=.3 if moving else 0.)


def test_normalized_root_coordinates_and_haar_rotation():
    """Uniform rotation has no preferred direction; motion cases are
    explicit."""
    prior = _prior()
    assert len(prior.coordinates.names) == 6
    assert len(_prior(True).coordinates.names) == 12
    assert prior.digest != _prior(True).digest
    bounds = np.asarray(prior.coordinates.bounds)
    points = np.random.default_rng(12).uniform(*bounds.T, size=(4096, 6))
    axes = []
    for point in points:
        state = prior.lift(point)
        axes.append(
            Rotation.from_quat(state.poses["box"][1]).apply([0., 0., 1.]))
        assert all(
            np.array_equal(v, np.zeros((2, 3)))
            for v in state.velocities.values())
    axes = np.asarray(axes)
    np.testing.assert_allclose(axes.mean(axis=0), 0., atol=.025)
    np.testing.assert_allclose(axes.T @ axes / len(axes),
                               np.eye(3) / 3,
                               atol=.025)
    expected = np.array([.5, .5, .5])
    np.testing.assert_allclose(points[:, :3].mean(axis=0), expected, atol=.015)


def test_twist_matches_derivative_of_rigid_motion():
    """Offset bodies need omega-cross-offset motion, not copied root
    velocity."""
    prior = _prior(True)
    point = np.array([.4, .5, .6, .3, .7, .2, .1, -.2, .05, .2, -.1, .3])
    state = prior.lift(point)
    dt = 1e-7
    root = np.array(state.poses["box"][0])
    turn = Rotation.from_rotvec(point[9:] * dt)
    for name, (position, _) in state.poses.items():
        shifted = root + point[6:9] * dt + turn.apply(
            np.array(position) - root)
        np.testing.assert_allclose((shifted - position) / dt,
                                   state.velocities[name][0],
                                   atol=1e-8)
    assert state.velocities["box"][0] != state.velocities["balloon"][0]


def test_engine_geometry_and_weld_frames_are_consistent():
    """Check actual engine geometry, not just bounds in the coordinate map."""
    prior = _prior(True)
    bounds = np.asarray(prior.coordinates.bounds)
    pcid = p.connect(p.DIRECT)
    try:
        box_shape = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[.03] * 3,
                                           physicsClientId=pcid)
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE,
                                            radius=.02,
                                            physicsClientId=pcid)
        ids = {
            "box": p.createMultiBody(.1, box_shape, physicsClientId=pcid),
            "balloon": p.createMultiBody(.01, ball_shape, physicsClientId=pcid)
        }
        for point in np.random.default_rng(1).uniform(*bounds.T,
                                                      size=(32, 12)):
            state = prior.lift(point)
            for name, body_id in ids.items():
                p.resetBasePositionAndOrientation(body_id,
                                                  *state.poses[name],
                                                  physicsClientId=pcid)
                lo, hi = np.asarray(p.getAABB(body_id, physicsClientId=pcid))
                assert np.all(lo >= 0) and np.all(hi <= 1)
            assert not p.getClosestPoints(
                ids["box"], ids["balloon"], 0., physicsClientId=pcid)
            weld, = state.welds
            # Independent transform composition through Bullet's own API.
            parent_world = p.multiplyTransforms(*state.poses[weld.parent],
                                                *weld.parent_frame)
            child_world = p.multiplyTransforms(*state.poses[weld.child],
                                               *weld.child_frame)
            np.testing.assert_allclose(parent_world[0],
                                       child_world[0],
                                       atol=1e-7)
            np.testing.assert_allclose(parent_world[1],
                                       child_world[1],
                                       atol=1e-7)
    finally:
        p.disconnect(pcid)


def test_prior_rejects_unjustified_or_empty_support():
    """Bad enclosing geometry or a cell too small is not a sampled scene."""
    prior = _prior()
    with pytest.raises(ValueError, match="overlap"):
        replace(prior,
                bodies=(prior.bodies[0], replace(prior.bodies[1], radius=.1)))
    with pytest.raises(ValueError, match="positive widths"):
        replace(prior, free_cell=((0., .1), ) * 3)
    with pytest.raises(ValueError, match="Twist"):
        replace(prior, linear_half_width=.1)
    with pytest.raises(ValueError, match="outside"):
        prior.lift(np.zeros(6))
    with pytest.raises(ValueError, match="unit pose"):
        replace(prior.bodies[1], pose=((0., 0., .1), (0., 0., 0., 0.)))
    with pytest.raises(ValueError, match="identity root"):
        replace(prior, bodies=prior.bodies[::-1])


def test_supported_component_has_exact_plane_contact():
    """A face-supported component has three coordinates, not a tiny z band."""
    prior = replace(_prior(), support_depth=.03)
    assert len(prior.coordinates.names) == 3
    assert prior.digest != _prior().digest
    client = p.connect(p.DIRECT)
    try:
        plane = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
        floor = p.createMultiBody(0, plane, physicsClientId=client)
        shape = p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[.03] * 3,
                                       physicsClientId=client)
        box = p.createMultiBody(.1, shape, physicsClientId=client)
        for yaw in np.linspace(0., 1., 17):
            state = prior.lift(np.array([.4, .6, yaw]))
            p.resetBasePositionAndOrientation(box,
                                              *state.poses["box"],
                                              physicsClientId=client)
            contacts = p.getClosestPoints(box,
                                          floor,
                                          1e-8,
                                          physicsClientId=client)
            assert contacts
            # Engine distance roundoff, not an observation-noise likelihood.
            assert max(abs(c[8]) for c in contacts) < 1e-12
            assert state.poses["box"][0][2] == .03
            np.testing.assert_array_equal(state.velocities["box"],
                                          np.zeros((2, 3)))
        with pytest.raises(ValueError, match="face and rest"):
            replace(prior, linear_half_width=.1, angular_half_width=.1)
        with pytest.raises(ValueError, match="below support"):
            replace(prior,
                    bodies=(prior.bodies[0],
                            replace(prior.bodies[1],
                                    pose=((0., 0., -.12), (0., 0., 0., 1.)))))
    finally:
        p.disconnect(client)
