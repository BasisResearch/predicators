"""Original articulated-joint priors without synthetic Boolean readings."""
import numpy as np
import pytest

from predicators.code_sim_learning.inference_joints import \
    JointCoordinateBoundary, RestingJointPrior


@pytest.mark.parametrize("rest", [0., .7, 1.])
def test_original_joint_moments_and_atoms(rest: float) -> None:
    """Piecewise quadrature recovers the declared mixed prior exactly."""
    prior = RestingJointPrior("faucet.hinge", -1., 2., (-.3, .8), rest, .2)
    points = np.array([-1., 1.]) / np.sqrt(3.)
    weights = np.ones(2)
    moments = np.zeros(6)
    edges = (0., rest / 2, rest, 1.)
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == lower:
            continue
        for point, weight in zip(points, weights):
            u = lower + (point + 1) * (upper - lower) / 2
            for other, other_weight in zip(points, weights):
                q, v = prior.lift(np.array([u, (other + 1) / 2]))
                mass = weight * other_weight * (upper - lower) / 4
                moments += mass * np.array(
                    [1., q, q * q, v, v * v,
                     float(v == 0.)])
    # Uniform[-1,2] has first/second moments 1/2 and 1.
    expected = [
        1., rest * .25 + (1 - rest) * .5,
        rest * (.3**2 + .8**2) / 2 + (1 - rest), 0., (1 - rest) * .2**2 / 3,
        rest
    ]
    np.testing.assert_allclose(moments, expected, rtol=0, atol=1e-14)
    assert prior.coordinates.names == ("faucet.hinge.position_mixture",
                                       "faucet.hinge.velocity")
    assert prior.coordinates.bounds == ((0., 1.), (0., 1.))


def test_unobserved_sampling_preserves_existing_conditioning() -> None:
    """Both original atoms stay available and actual readings still
    condition."""
    prior = RestingJointPrior("slider", 0., 2., (0., 2.), .8, .1)
    identity = prior.digest
    off = prior.condition_above(1., False)
    on = prior.condition_above(1., True)
    before = [law.lift(np.array([.9, .75])) for law in (off, on)]
    assert prior.lift(np.array([.2, .25])) == (0., 0.)
    assert prior.lift(np.array([.6, .25])) == (2., 0.)
    q, v = prior.lift(np.array([.9, .75]))
    assert q == pytest.approx(1.) and v == pytest.approx(.05)
    assert [law.lift(np.array([.9, .75])) for law in (off, on)] == before
    assert prior.digest == identity
    assert off.log_observation_factor == pytest.approx(np.log(.5))
    assert on.log_observation_factor == pytest.approx(np.log(.5))


def test_original_joint_invalid_coordinates_and_seams() -> None:
    """Malformed coordinates and zero-measure seams stay explicit."""
    prior = RestingJointPrior("hinge", 0., 1., (0., 1.), .8, .1)
    for point in [np.array([.5]), np.array([np.nan, .5])]:
        with pytest.raises(ValueError):
            prior.lift(point)
    for point in [
            np.array([0., .5]),
            np.array([1., .5]),
            np.array([.5, 1.]),
            np.array([.8, .5])
    ]:
        with pytest.raises(JointCoordinateBoundary):
            prior.lift(point)
