"""Tests for the parameter factor of the joint belief."""

import numpy as np
import pytest

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.parameter_belief import BeliefConfig, \
    LinePosterior, ParameterBelief, build_parameter_belief, stable_seed

_SIGMA_N = 0.05


def _gaussian_residuals(names, mean, precision_root, offset=()):
    """Residuals whose objective is a Gaussian negative log likelihood.

    With ``r = sigma_n * L (z - mean)`` the objective's negative log
    likelihood ``|r|^2 / (2 sigma_n^2)`` has precision ``L^T L``.
    ``offset`` appends constant residuals the program cannot explain.
    """
    root = np.asarray(precision_root, dtype=float)

    def residuals(params):
        z = np.array([params[n] for n in names]) - np.asarray(mean)
        return np.concatenate(
            [_SIGMA_N * root @ z,
             np.asarray(offset, dtype=float)])

    return residuals


def _build(specs, map_params, residuals, prior_sigma=1e6, config=None, seed=0):
    return build_parameter_belief(
        specs,
        map_params,
        residuals,
        noise_sigma=_SIGMA_N,
        prior_centers={s.name: s.init_value
                       for s in specs},
        prior_sigmas={s.name: prior_sigma
                      for s in specs},
        config=config or BeliefConfig(num_draws=64, line_max_evals=40),
        seed=seed)


def test_line_posterior_moments_and_sampling():
    """Exact moments of a piecewise-linear density match its samples."""
    line = LinePosterior.from_neg_log([0.0, 1.0, 2.0], [5.0, 0.0, np.log(2.0)])
    assert np.trapz(line.density, line.grid) == pytest.approx(1.0)
    samples = line.sample(np.random.default_rng(0), 200_000)
    assert samples.min() >= 0.0 and samples.max() <= 2.0
    assert samples.mean() == pytest.approx(line.mean(), abs=5e-3)
    assert samples.var() == pytest.approx(line.variance(), rel=2e-2)
    assert line.quantile(0.0) == pytest.approx(0.0)
    assert line.quantile(1.0) == pytest.approx(2.0)
    point = LinePosterior.from_neg_log([0.4], [3.0])
    assert point.is_point and point.variance() == 0.0
    assert np.all(point.sample(np.random.default_rng(0), 3) == 0.4)


def test_gaussian_lines_recover_conditional_variances():
    """Each line has the mean-field variance ``1 / Lambda_jj``."""
    specs = [
        ParamSpec("a", 0.3, lo=-10, hi=10),
        ParamSpec("b", 0.7, lo=-10, hi=10)
    ]
    root = np.array([[20.0, 0.0], [12.0, 16.0]])
    precision = root.T @ root
    belief = _build(specs, {
        "a": 0.3,
        "b": 0.7
    }, _gaussian_residuals(["a", "b"], [0.3, 0.7], root))
    assert belief.noise_scale == pytest.approx(1.0)
    for j, name in enumerate(["a", "b"]):
        line = belief.lines[name]
        assert line.mean() == pytest.approx([0.3, 0.7][j], abs=2e-3)
        assert line.variance() == pytest.approx(1.0 / precision[j, j],
                                                rel=0.06)


def test_flat_line_keeps_the_prior():
    """A parameter the data do not touch gets its (bounded) prior."""
    specs = [
        ParamSpec("a", 0.3, lo=-10, hi=10),
        ParamSpec("free", 1.0, lo=-10, hi=10)
    ]
    belief = _build(specs, {
        "a": 0.3,
        "free": 1.0
    },
                    _gaussian_residuals(["a"], [0.3], [[20.0]]),
                    prior_sigma=0.5)
    line = belief.lines["free"]
    assert line.mean() == pytest.approx(1.0, abs=5e-3)
    assert np.sqrt(line.variance()) == pytest.approx(0.5, rel=0.06)


def test_misfit_scales_the_noise_level():
    """Residuals beyond the noise model widen the posterior by sqrt(lambda)."""
    specs = [ParamSpec("a", 0.3, lo=-10, hi=10)]
    fits = _gaussian_residuals(["a"], [0.3], [[20.0]])
    misfit = _gaussian_residuals(["a"], [0.3], [[20.0]],
                                 offset=[3.0 * _SIGMA_N] * 3)
    narrow = _build(specs, {"a": 0.3}, fits)
    wide = _build(specs, {"a": 0.3}, misfit)
    # Four residual terms, 27 sigma_n^2 of unexplained squared error.
    assert wide.noise_scale == pytest.approx(27.0 / 4.0)
    ratio = np.sqrt(wide.lines["a"].variance() / narrow.lines["a"].variance())
    assert ratio == pytest.approx(np.sqrt(27.0 / 4.0), rel=0.06)


def test_draws_are_seeded_bounded_and_sample_discrete_parameters():
    """Draws respect the box, the scale, discreteness and the seed."""
    specs = [
        ParamSpec("friction", 0.5, lo=0.4, hi=0.6, scale="log"),
        ParamSpec("slot", 2.0, lo=0.0, hi=5.0, discrete=True),
        ParamSpec("gear", 1.0, discrete=True)
    ]
    residuals = _gaussian_residuals(["friction"], [0.5], [[2.0]])
    map_params = {"friction": 0.5, "slot": 2.0, "gear": 1.0}
    first = _build(specs, map_params, residuals, prior_sigma=0.5, seed=3)
    again = _build(specs, map_params, residuals, prior_sigma=0.5, seed=3)
    other = _build(specs, map_params, residuals, prior_sigma=0.5, seed=4)
    assert np.array_equal(first.draws, again.draws)
    assert not np.array_equal(first.draws, other.draws)
    friction = first.draws[:, 0]
    assert friction.min() >= 0.4 - 1e-12 and friction.max() <= 0.6 + 1e-12
    # The data say nothing about the slot, so its draws follow the prior
    # (centred on 2 with width 0.5) over the integers in its box.
    slots = first.draws[:, 1]
    assert np.all(slots == np.round(slots))
    assert slots.min() >= 0.0 and slots.max() <= 5.0
    assert np.mean(slots == 2.0) > 0.6
    # An unbounded discrete parameter cannot be enumerated and is held.
    assert first.held == ["gear"]
    assert np.all(first.draws[:, 2] == 1.0)
    lo, hi = first.interval("friction")
    assert 0.4 <= lo < 0.5 < hi <= 0.6
    assert first.interval("slot") == (2.0, 2.0)
    assert first.interval("gear") == (1.0, 1.0)
    assert len(first.draw_dicts()) == 64
    assert any("slot" in line and "discrete" in line
               for line in first.describe())
    restored = ParameterBelief.from_dict(first.to_dict())
    assert np.array_equal(restored.discrete["slot"].probs,
                          first.discrete["slot"].probs)


def test_discrete_draws_follow_the_target():
    """A discrete parameter is sampled in proportion to the target."""
    specs = [ParamSpec("slot", 3.0, lo=0.0, hi=6.0, discrete=True)]
    # SSE / (2 sigma_n^2) = 2 (slot - 3)^2, so p(slot) ~ exp(-2 (slot-3)^2).
    residuals = _gaussian_residuals(["slot"], [3.0], [[2.0]])
    belief = _build(specs, {"slot": 3.0},
                    residuals,
                    config=BeliefConfig(num_draws=20_000, line_max_evals=40))
    values = np.arange(0.0, 7.0)
    expected = np.exp(-2.0 * (values - 3.0)**2)
    expected /= expected.sum()
    dist = belief.discrete["slot"]
    assert np.allclose(dist.values, values)
    assert np.allclose(dist.probs, expected)
    freq = np.array([np.mean(belief.draws[:, 0] == v) for v in values])
    assert np.allclose(freq, expected, atol=0.01)
    assert belief.evaluations == 6  # every value but the MAP itself


def test_round_trip_and_stable_seed():
    """Checkpoints restore the belief exactly; seeds are reproducible."""
    specs = [ParamSpec("a", 0.3, lo=-10, hi=10)]
    belief = _build(specs, {"a": 0.3},
                    _gaussian_residuals(["a"], [0.3], [[20.0]]))
    restored = ParameterBelief.from_dict(belief.to_dict())
    assert np.array_equal(restored.draws, belief.draws)
    assert restored.interval("a") == belief.interval("a")
    assert restored.noise_scale == belief.noise_scale
    assert stable_seed(0, "v1", 0.5) == stable_seed(0, "v1", 0.5)
    assert stable_seed(0, "v1", 0.5) != stable_seed(1, "v1", 0.5)
    assert 0 <= stable_seed("x") < 2**63


def test_config_reads_the_flags():
    """The belief config mirrors the global flags."""
    # pylint: disable-next=import-outside-toplevel
    from predicators import utils
    utils.reset_config({
        "belief_joint_draws": 7,
        "belief_line_cutoff": 8.0,
        "belief_line_max_evals": 11,
    })
    try:
        assert BeliefConfig.from_cfg() == BeliefConfig(num_draws=7,
                                                       line_cutoff=8.0,
                                                       line_max_evals=11)
    finally:
        utils.reset_config({})


def test_prior_belief_is_the_bounded_prior():
    """Before any fit, the factor is the declared prior within the box."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.parameter_belief import prior_belief
    specs = [
        ParamSpec("mass", 1.0, lo=0.2, hi=5.0, scale="log"),
        ParamSpec("slot", 1.0, lo=0.0, hi=3.0, discrete=True)
    ]
    belief = prior_belief(specs, {
        "mass": 1.0,
        "slot": 1.0
    }, {
        "mass": 0.5,
        "slot": 1.0
    },
                          BeliefConfig(num_draws=4000),
                          seed=1)
    log_mass = np.log(belief.draws[:, 0])
    assert np.all((belief.draws[:, 0] >= 0.2) & (belief.draws[:, 0] <= 5.0))
    assert log_mass.mean() == pytest.approx(0.0, abs=0.03)
    assert log_mass.std() == pytest.approx(0.5, rel=0.05)
    # The discrete slot follows the prior over the integers in its box.
    values = np.arange(0.0, 4.0)
    expected = np.exp(-0.5 * (values - 1.0)**2)
    expected /= expected.sum()
    freq = np.array([np.mean(belief.draws[:, 1] == v) for v in values])
    assert np.allclose(freq, expected, atol=0.03)
    assert belief.noise_scale == 1.0 and belief.evaluations == 0
