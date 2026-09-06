"""Tests for predicators.code_sim_learning.active_experiment."""

# pylint: disable=unused-import

import numpy as np
import pytest

from predicators import utils  # noqa: F401  (settles import order)
from predicators.code_sim_learning.active_experiment import laplace_ensemble, \
    mean_bernoulli_entropy, perturbation_ensemble
from predicators.code_sim_learning.fit_space import ParamSpec


def _specs():
    return [
        ParamSpec("faucet_local_dy", -0.05, lo=-0.2, hi=0.1),
        ParamSpec("jug_at_faucet_dist", 0.11, lo=0.03, hi=0.25),
        ParamSpec("heat_rate", 1.0, lo=0.1, hi=5.0),
    ]


def test_perturbation_ensemble_anchor_is_member_zero():
    """Perturbation ensemble anchor is member zero."""
    point = {
        "faucet_local_dy": -0.05,
        "jug_at_faucet_dist": 0.11,
        "heat_rate": 1.0
    }
    rng = np.random.default_rng(0)
    members = perturbation_ensemble(point,
                                    _specs(),
                                    num_members=6,
                                    perturb_frac=0.15,
                                    rng=rng)
    assert len(members) == 6
    # Member 0 is the exact anchor.
    assert members[0] == point
    # It is a copy, not an alias.
    members[0]["heat_rate"] = 99.0
    assert point["heat_rate"] == 1.0


def test_perturbation_ensemble_respects_bounds():
    """Perturbation ensemble respects bounds."""
    point = {"faucet_local_dy": 0.09, "jug_at_faucet_dist": 0.04}
    rng = np.random.default_rng(1)
    members = perturbation_ensemble(point,
                                    _specs(),
                                    num_members=64,
                                    perturb_frac=1.0,
                                    rng=rng)
    for m in members:
        assert -0.2 <= m["faucet_local_dy"] <= 0.1
        assert 0.03 <= m["jug_at_faucet_dist"] <= 0.25


def test_perturbation_ensemble_leaves_discrete_params_alone():
    """A discrete parameter (an index) is never jittered.

    Its effect is a staircase, so jitter would either do nothing or
    rewire the model; neither is an uncertainty the point estimate
    carries.
    """
    specs = _specs() + [
        ParamSpec("driver_0", 1.0, lo=0.0, hi=4.0, discrete=True)
    ]
    point = {"heat_rate": 1.0, "driver_0": 1.0}
    members = perturbation_ensemble(point,
                                    specs,
                                    num_members=32,
                                    perturb_frac=1.0,
                                    rng=np.random.default_rng(3))
    assert all(m["driver_0"] == 1.0 for m in members)
    assert any(m["heat_rate"] != 1.0 for m in members[1:])


def test_perturbation_ensemble_size_one_is_point_estimate():
    """Perturbation ensemble size one is point estimate."""
    point = {"heat_rate": 1.0}
    rng = np.random.default_rng(2)
    members = perturbation_ensemble(point,
                                    _specs(),
                                    num_members=1,
                                    perturb_frac=0.5,
                                    rng=rng)
    assert members == [point]


def test_perturbation_ensemble_param_without_spec_carried_through():
    """Perturbation ensemble param without spec carried through."""
    point = {"unknown_param": 7.0}
    rng = np.random.default_rng(3)
    members = perturbation_ensemble(point,
                                    _specs(),
                                    num_members=4,
                                    perturb_frac=0.5,
                                    rng=rng)
    # No spec => value carried through unperturbed on every member.
    assert all(m["unknown_param"] == 7.0 for m in members)


def test_perturbation_ensemble_actually_spreads():
    """Perturbation ensemble actually spreads."""
    point = {"heat_rate": 1.0}
    rng = np.random.default_rng(4)
    members = perturbation_ensemble(point,
                                    _specs(),
                                    num_members=200,
                                    perturb_frac=0.2,
                                    rng=rng)
    vals = np.array([m["heat_rate"] for m in members])
    # The non-anchor members should have non-trivial spread.
    assert vals.std() > 0.05


def test_perturbation_ensemble_invalid_size():
    """Perturbation ensemble invalid size."""
    with pytest.raises(ValueError):
        perturbation_ensemble({},
                              _specs(),
                              num_members=0,
                              perturb_frac=0.1,
                              rng=np.random.default_rng(0))


def test_entropy_all_agree_is_zero():
    """Entropy all agree is zero."""
    # Every member agrees on every atom -> no information.
    mat = np.array([[True, False, True], [True, False, True]])
    assert mean_bernoulli_entropy(mat) == 0.0


def test_entropy_even_split_is_max():
    """Entropy even split is max."""
    # Two members, one atom, evenly split -> entropy 1.0 bit.
    mat = np.array([[True], [False]])
    assert mean_bernoulli_entropy(mat) == pytest.approx(1.0)


def test_entropy_partial_split():
    """Entropy partial split."""
    # 4 members on a single atom split 1/3 -> H(0.25) ~= 0.811.
    mat = np.array([[True], [False], [False], [False]])
    assert mean_bernoulli_entropy(mat) == pytest.approx(0.8112781, abs=1e-5)


def test_entropy_averages_over_atoms():
    """Entropy averages over atoms."""
    # Atom A even split (H=1), atom B unanimous (H=0) -> mean 0.5.
    mat = np.array([[True, True], [False, True]])
    assert mean_bernoulli_entropy(mat) == pytest.approx(0.5)


def test_entropy_empty_is_zero():
    """Entropy empty is zero."""
    assert mean_bernoulli_entropy(np.zeros((0, 0))) == 0.0


def test_entropy_rejects_non_2d():
    """Entropy rejects non 2d."""
    with pytest.raises(ValueError):
        mean_bernoulli_entropy(np.array([True, False]))


# ── laplace_ensemble ─────────────────────────────────────────────


def _laplace_specs():
    return [
        ParamSpec("a", 1.0, lo=-10.0, hi=10.0),
        ParamSpec("b", 1.0, lo=-10.0, hi=10.0)
    ]


def test_laplace_anchor_is_member_zero():
    """Laplace anchor is member zero."""
    point = {"a": 1.0, "b": 2.0}
    jac = np.eye(2)
    members = laplace_ensemble(point, ["a", "b"],
                               _laplace_specs(),
                               jac,
                               noise_sigma=0.1,
                               prior_sigma=[1.0, 1.0],
                               num_members=4,
                               rng=np.random.default_rng(0))
    assert len(members) == 4
    assert members[0] == point


def test_laplace_size_one_is_point_estimate():
    """Laplace size one is point estimate."""
    point = {"a": 1.0, "b": 2.0}
    members = laplace_ensemble(point, ["a", "b"],
                               _laplace_specs(),
                               np.eye(2),
                               noise_sigma=0.1,
                               prior_sigma=[1.0, 1.0],
                               num_members=1,
                               rng=np.random.default_rng(0))
    assert members == [point]


def test_laplace_stiff_direction_barely_moves():
    """Laplace stiff direction barely moves."""
    # Param 'a' is sharply constrained (large Jacobian column), 'b' is not
    # constrained by data at all (zero column) -> 'a' should spread far less
    # than 'b'. This is the whole point: calibrated, not uniform.
    point = {"a": 0.0, "b": 0.0}
    jac = np.array([[100.0, 0.0], [100.0, 0.0]])  # only 'a' is informed
    members = laplace_ensemble(point, ["a", "b"],
                               _laplace_specs(),
                               jac,
                               noise_sigma=1.0,
                               prior_sigma=[1.0, 1.0],
                               num_members=400,
                               rng=np.random.default_rng(7))
    a_vals = np.array([m["a"] for m in members[1:]])
    b_vals = np.array([m["b"] for m in members[1:]])
    assert a_vals.std() < 0.1 * b_vals.std()  # stiff << sloppy


def test_laplace_respects_box_bounds():
    """Laplace respects box bounds."""
    point = {"a": 0.0, "b": 0.0}
    specs = [
        ParamSpec("a", 0.0, lo=-0.01, hi=0.01),
        ParamSpec("b", 0.0, lo=-0.01, hi=0.01)
    ]
    jac = np.zeros((2, 2))  # no data -> wide prior-driven covariance
    members = laplace_ensemble(point, ["a", "b"],
                               specs,
                               jac,
                               noise_sigma=1.0,
                               prior_sigma=[100.0, 100.0],
                               num_members=200,
                               rng=np.random.default_rng(3))
    for m in members:
        assert -0.01 <= m["a"] <= 0.01
        assert -0.01 <= m["b"] <= 0.01


def test_laplace_degenerate_jacobian_returns_anchor_only():
    """Laplace degenerate jacobian returns anchor only."""
    point = {"a": 1.0}
    # Jacobian column count (1) mismatches names? Here names has 1, jac has
    # shape (0,) -> not 2D -> falls back to anchor only.
    members = laplace_ensemble(point, ["a"], [ParamSpec("a", 1.0)],
                               np.array([]),
                               noise_sigma=0.1,
                               prior_sigma=[1.0],
                               num_members=5,
                               rng=np.random.default_rng(0))
    assert members == [point]


def test_laplace_invalid_size():
    """Laplace invalid size."""
    with pytest.raises(ValueError):
        laplace_ensemble({}, [], [],
                         np.eye(1),
                         noise_sigma=0.1,
                         prior_sigma=[1.0],
                         num_members=0,
                         rng=np.random.default_rng(0))
