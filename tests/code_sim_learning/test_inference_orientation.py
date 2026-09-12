"""Independent probability checks for the native Euler pushforward."""
import math

import numpy as np
import pybullet as p
import pytest
from scipy.special import roots_legendre

from predicators.code_sim_learning.inference_orientation import \
    QuaternionOutputError


@pytest.mark.parametrize("sigma", [.1, .5, 1.])
def test_centered_gaussian_has_analytic_mixed_density(sigma: float) -> None:
    """A spherical four-dimensional normal has a closed-form pushforward.

    Integrating the ordinary formula gives 1-exp(-c/(2*sigma^2)); each
    pole's four-pi yaw integral gives half the remaining mass.
    """
    process = QuaternionOutputError(sigma)
    mean = (0., 0., 0., 0.)
    for pitch in (0., .3, -1.1):
        expected = (math.log(math.cos(pitch)) -
                    math.log(16 * math.pi**2 * sigma**2) -
                    abs(math.sin(pitch)) / (2 * sigma**2))
        actual = process.log_density(mean, (.4, pitch, -.6))
        assert actual == pytest.approx(expected, abs=1e-7)
    pole = -process.pole_threshold / (2 * sigma**2) - math.log(8 * math.pi)
    for sign in (-1, 1):
        for yaw in (-4.7, 0., 4.7):
            assert process.log_density(mean, (0., sign * math.pi / 2, yaw)) \
                == pytest.approx(pole, abs=1e-7)
    ordinary_mass = -math.expm1(-process.pole_threshold / (2 * sigma**2))
    assert ordinary_mass + 8 * math.pi * math.exp(pole) == \
        pytest.approx(1., abs=1e-14)


def test_noncentral_pole_probability_matches_native_draws() -> None:
    """Integrate a yaw event and compare with independent raw Gaussian draws.

    Native API calls check the vectorized event classification on a
    subset, including non-unit quaternions and both antipodal signs.
    """
    process = QuaternionOutputError(.15)
    mean = np.array(p.getQuaternionFromEuler([.1, math.pi / 2, .4]))
    rng = np.random.default_rng(901)
    quaternions = mean + process.sigma * rng.normal(size=(200000, 4))
    quaternions *= rng.choice((-1, 1), size=(len(quaternions), 1))
    x, y, z, w = quaternions.T
    positive = 2 * (w * y - x * z) >= process.pole_threshold
    yaw = 2 * np.arctan2(-x, y)
    inside = positive & (yaw > -.5) & (yaw < .8)
    for q, is_pole, expected_yaw in zip(quaternions[:200], positive, yaw):
        native = p.getEulerFromQuaternion(q.tolist())
        assert (native[1] == math.pi / 2) == bool(is_pole)
        if is_pole:
            assert native[0] == 0
            assert native[2] == pytest.approx(expected_yaw, abs=1e-14)
    nodes, weights, _ = roots_legendre(24, mu=True)
    integral = sum(weight * math.exp(
        process.log_density(tuple(mean), (0., math.pi / 2, .15 + .65 * node)))
                   for node, weight in zip(nodes, weights)) * .65
    empirical = float(np.mean(inside))
    assert abs(integral - empirical) < .004
    assert .1 < empirical < .4  # The test exercises appreciable pole mass.


def test_noncentral_ordinary_probability_matches_native_draws() -> None:
    """A three-dimensional Euler box agrees with a Gaussian sample count."""
    process = QuaternionOutputError(.35)
    mean = (0., 0., 0., 1.)
    rng = np.random.default_rng(902)
    quaternions = np.array(mean) + process.sigma * rng.normal(size=(300000, 4))
    x, y, z, w = quaternions.T
    sine = 2 * (w * y - x * z)
    ordinary = abs(sine) < process.pole_threshold
    roll = np.arctan2(2 * (y * z + w * x), w * w - x * x - y * y + z * z)
    yaw = np.arctan2(2 * (x * y + w * z), w * w + x * x - y * y - z * z)
    inside = ordinary & (abs(roll) < .4) & (abs(sine) < math.sin(.3)) & \
        (abs(yaw) < .4)
    nodes, weights, _ = roots_legendre(6, mu=True)
    integral = 0.
    for i, roll_node in enumerate(nodes):
        for j, pitch_node in enumerate(nodes):
            for k, yaw_node in enumerate(nodes):
                reading = (.4 * roll_node, .3 * pitch_node, .4 * yaw_node)
                integral += weights[i] * weights[j] * weights[k] * \
                    math.exp(process.log_density(mean, reading)) * .4*.3*.4
    assert abs(integral - float(np.mean(inside))) < .002
    for q in quaternions[:100]:
        actual = p.getEulerFromQuaternion(q.tolist())
        if abs(actual[1]) < math.pi / 2:
            assert math.isfinite(process.log_density(mean, tuple(actual)))


def test_raw_branch_and_quadrature_sensitivity() -> None:
    """Sign marginalization preserves raw yaw measure and very small scores."""
    process = QuaternionOutputError(.02)
    mean = tuple(p.getQuaternionFromEuler([.2, 1.56, .7]))
    for reading in ((.2, 1.56, .7), (0., math.pi / 2, 4.7),
                    (0., -math.pi / 2, -.4), (1., -.4, -1.)):
        value = process.log_density(mean, reading)
        fine = process.log_density(mean, reading, 1e-9)
        assert value == pytest.approx(fine, abs=2e-7)
        assert value == pytest.approx(process.log_density(
            tuple(-x for x in mean), reading),
                                      abs=1e-10)
    assert process.log_density(mean, (1., -.4, -1.)) < -700
    # These are two distinct raw observations with equal antipodal mass.
    first = process.log_density(mean, (0., math.pi / 2, .2))
    second = process.log_density(mean, (0., math.pi / 2, .2 - 2 * math.pi))
    assert first == pytest.approx(second, abs=1e-9)


def test_invalid_support_is_distinct_from_invalid_input() -> None:
    """Do not wrap, renormalize or soften exact contradictions."""
    process = QuaternionOutputError(.1)
    mean = (0., 0., 0., 1.)
    for observed in ((.1, math.pi / 2, 0.), (0., math.pi / 2, 7.),
                     (0., 1.57, 0.), (0., 2., 0.), (4., 0., 0.)):
        assert process.log_density(mean, observed) == -math.inf
    with pytest.raises(ValueError):
        process.log_density(mean, (0., float("nan"), 0.))
    with pytest.raises(ValueError):
        process.log_density(mean, (0., 0., 0.), 0.)
    for sigma in (0., -.1, math.inf):
        with pytest.raises(ValueError):
            QuaternionOutputError(sigma)
    with pytest.raises(ValueError):
        QuaternionOutputError(.1, 1.)
    assert process.digest != QuaternionOutputError(.2).digest
    assert process.digest != QuaternionOutputError(.1, .99).digest


def test_recorded_near_pole_boundary_converges() -> None:
    """Retain the Balloons action-23 witness that failed tighter quadrature."""
    process = QuaternionOutputError(.0001)
    mean = tuple(
        p.getQuaternionFromEuler(
            [2.856054709212263, 1.564969365024118, -1.861413060789528]))
    observed = (1.1358128734980024, 1.566274408012441, 2.705544777791995)
    coarse = process.log_density(mean, observed, 1e-7)
    fine = process.log_density(mean, observed, 1e-9)
    assert coarse == pytest.approx(fine, abs=1e-7)
