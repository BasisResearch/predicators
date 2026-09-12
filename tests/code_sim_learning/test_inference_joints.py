"""Exact joint conditioning retains prior mass and unobserved motion."""
import math
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_joints import \
    ConditionedJointPrior, IncompatibleJointObservation, JointStatePrior


def test_conditioning_preserves_unobserved_positions_and_velocities():
    """Initial proprioception constrains positions, not unmeasured motion."""
    prior = JointStatePrior(("arm", "head", "fixed"),
                            ((-2., 2.), (-1., 1.), None), (.5, .25, 0.))
    observations = {"arm": .7, "fixed": 0.}
    conditional = prior.condition_positions(observations)
    observations["arm"] = -.2
    assert conditional.coordinates.names == ("head.position", "arm.velocity",
                                             "head.velocity")
    bounds = np.asarray(conditional.coordinates.bounds)
    rng = np.random.default_rng(0)
    values = np.array([
        conditional.lift(point)
        for point in rng.uniform(*bounds.T, size=(2048, 3))
    ])
    np.testing.assert_array_equal(values[:, 0, 0], np.full(2048, .7))
    np.testing.assert_array_equal(values[:, 2], np.zeros((2048, 2)))
    assert abs(values[:, 1, 0].mean()) < .04
    assert abs(values[:, 0, 1].mean()) < .03
    assert abs(values[:, 1, 1].mean()) < .02
    assert conditional.log_observation_factor == -math.log(4)
    assert conditional.prior.digest == prior.digest
    assert conditional.digest != prior.condition_positions({"arm": .8}).digest


def test_rest_and_fully_conditioned_cases_are_atoms():
    """A deterministic conditional still retains the original reading
    density."""
    prior = JointStatePrior(("arm", "fixed"), ((-2., 2.), None), (0., 0.))
    conditional = prior.condition_positions({"arm": 0., "fixed": 0.})
    assert conditional.coordinates is None
    assert conditional.lift(np.array([])) == ((0., 0.), (0., 0.))
    assert conditional.log_observation_factor == -math.log(4)
    # Uniform[-1,1] gives twice the observation density of Uniform[-2,2].
    narrower = replace(prior, position_bounds=((-1., 1.), None))
    ratio = math.exp(
        narrower.condition_positions({
            "arm": 0.
        }).log_observation_factor - conditional.log_observation_factor)
    assert ratio == pytest.approx(2.)


def test_exact_positions_are_not_wrapped_clipped_or_softened():
    """Winding and exact support violations stay explicit."""
    prior = JointStatePrior(("wheel", "fixed"), ((-8., 8.), None), (1., 0.))
    conditional = prior.condition_positions({"wheel": 7.})
    assert conditional.lift(np.array([.2]))[0] == (7., .2)
    for observations in ({"wheel": 8.00000001}, {"fixed": 1e-15}):
        with pytest.raises(IncompatibleJointObservation):
            prior.condition_positions(observations)
    with pytest.raises(ValueError, match="repeated"):
        ConditionedJointPrior(prior, (("wheel", 0.), ("wheel", 1.)))
    with pytest.raises(ValueError, match="finite"):
        prior.condition_positions({"wheel": math.nan})
    with pytest.raises(ValueError, match="outside"):
        conditional.lift(np.array([2.]))


def test_missing_joint_motion_cannot_be_invented():
    """The caller must supply complete mechanical and motion assumptions."""
    with pytest.raises(ValueError, match="matching"):
        JointStatePrior(("arm", "head"), ((-1., 1.), ), (0., ))
    with pytest.raises(ValueError, match="Fixed"):
        JointStatePrior(("fixed", ), (None, ), (.1, ))
    with pytest.raises(ValueError, match="finite position"):
        JointStatePrior(("wheel", ), ((-math.inf, math.inf), ), (1., ))
