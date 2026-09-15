"""Exact-observation references expose rejection and projection bias."""
import math
from dataclasses import replace

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    AffineConditioning, UnsupportedConditioning
from predicators.code_sim_learning.inference_data import content_digest
from predicators.code_sim_learning.inference_sampling import BoxPrior


def _product_chart() -> AffineConditioning:
    return AffineConditioning(
        BoxPrior(("theta", "start", "unused"),
                 ((1., 2.), (0., 1.), (-1., 1.))), ("start", ), (.5, ),
        content_digest(b"y = theta * start"))


def test_exact_output_rejection_and_correct_conditional_reference() -> None:
    """Conditioning y=theta*start informs theta through a 1/theta factor.

    Independent continuous draws miss the equality. Merely projecting
    start=y/theta keeps theta uniform and gives the wrong posterior.
    Midpoint quadrature over the same chart recovers the analytic
    answer.
    """
    rng = np.random.default_rng(51)
    prior_draws = rng.uniform((1, 0), (2, 1), size=(1000, 2))
    assert not np.any(prior_draws[:, 0] * prior_draws[:, 1] == .5)
    chart = _product_chart()
    theta = 1 + (np.arange(1000) + .5) / 1000
    points = [
        chart.lift(np.array([t, .25]), np.array([[t]]), np.array([0.]))
        for t in theta
    ]
    weights = np.exp([point.log_base_weight for point in points])
    assert np.mean(theta) == pytest.approx(1.5)
    assert np.average(theta, weights=weights) == pytest.approx(1 / math.log(2),
                                                               abs=1e-7)
    np.testing.assert_allclose([p.joint[1] for p in points], .5 / theta)
    assert all(p.joint[2] == .25 for p in points)
    assert all(p.max_constraint_residual <= p.numerical_residual_bound
               for p in points)


def test_initial_coordinate_conditioning_and_joint_order() -> None:
    """Exactly observed initial coordinates leave independent uniforms
    alone."""
    prior = BoxPrior(("x", "v", "z"), ((-2, 2), (-3, 3), (0, 10)))
    chart = AffineConditioning(prior, ("z", "x"), (6., .2),
                               content_digest(b"y = (z,x)"))
    assert chart.free_names == ("v", )
    assert chart.free_bounds == ((-3., 3.), )
    point = chart.lift(np.array([1.]), np.eye(2), np.zeros(2))
    assert point.joint == (.2, 1., 6.)
    assert point.log_base_weight == pytest.approx(-math.log(40))
    fully = AffineConditioning(prior, prior.names, (.2, 1., 6.),
                               content_digest(b"identity"))
    assert fully.lift(np.array([]), np.eye(3),
                      np.zeros(3)).joint == point.joint


def test_prior_support_and_singular_chart_are_distinct() -> None:
    """Point rejection does not establish global inconsistency."""
    chart = _product_chart()
    outside = chart.lift(np.array([1., 0.]), np.array([[.1]]), np.zeros(1))
    assert outside.joint[1] == 5
    assert outside.log_base_weight == -math.inf
    outside_free = chart.lift(np.array([3., 0.]), np.ones((1, 1)), np.zeros(1))
    assert outside_free.log_base_weight == -math.inf
    with pytest.raises(UnsupportedConditioning, match="Singular"):
        chart.lift(np.array([1., 0.]), np.zeros((1, 1)), np.zeros(1))
    with pytest.raises(ValueError, match="finite inputs"):
        chart.lift(np.array([1., 0.]), np.array([[np.nan]]), np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        chart.lift(np.array([1., 0.]), np.eye(2), np.zeros(1))


def test_change_of_output_units_preserves_normalized_density() -> None:
    """Scaling exact observations changes evidence, not normalized weights."""
    chart = _product_chart()
    scaled = replace(chart,
                     observed=(5., ),
                     equation_identity=content_digest(b"y = 10*theta*start"))
    assert scaled.digest != chart.digest
    for theta in (1., 1.25, 2.):
        point = chart.lift(np.array([theta, 0.]), np.array([[theta]]),
                           np.zeros(1))
        other = scaled.lift(np.array([theta, 0.]), np.array([[10 * theta]]),
                            np.zeros(1))
        np.testing.assert_allclose(point.joint, other.joint)
        assert other.log_base_weight - point.log_base_weight == pytest.approx(
            -math.log(10))


def test_affine_offset_and_validation() -> None:
    """Nontrivial matrices retain the full determinant, with no projection."""
    prior = BoxPrior(("a", "b", "c"), ((-10, 10), ) * 3)
    chart = AffineConditioning(prior, ("a", "b"), (4., 5.),
                               content_digest(b"coupled affine"))
    point = chart.lift(np.array([2.]), np.array([[2., 1.], [0., -3.]]),
                       np.array([1., 2.]))
    assert point.joint == (2., -1., 2.)
    assert point.log_base_weight == pytest.approx(-math.log(6 * 400))
    with pytest.raises(ValueError, match="distinct"):
        replace(chart, eliminated=("a", "a"))
    with pytest.raises(ValueError, match="finite observation"):
        replace(chart, observed=(math.inf, 0.))
    with pytest.raises(ValueError, match="SHA256"):
        replace(chart, equation_identity="description")
