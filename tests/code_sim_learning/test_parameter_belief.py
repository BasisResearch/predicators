"""Tests for the parameter factor of the joint belief."""

import numpy as np
import pytest

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.parameter_belief import BeliefConfig, \
    LinePosterior, ParameterBelief, build_parameter_belief, join_beliefs, \
    mix_beliefs, prior_belief, stable_seed

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


def _build(specs,
           map_params,
           residuals,
           prior_sigma=1e6,
           config=None,
           seed=0,
           misfit=0.0,
           noise_sse=0.0):
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
        seed=seed,
        misfit=misfit,
        noise_sse=noise_sse)


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


def test_highest_density_interval_holds_a_mode_at_a_bound():
    """A density that falls away from a bound has its mode at the bound: the
    central interval excludes it, the highest-density interval starts there and
    holds at least the same mass."""
    line = LinePosterior.from_neg_log([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    assert line.mode() == 0.0
    assert line.quantile(0.16) > 0.0
    lo, hi = line.highest_density_interval(0.68)
    assert lo == 0.0
    inside = (line.grid >= lo) & (line.grid <= hi)
    mass = np.trapz(line.density[inside], line.grid[inside])
    assert 0.68 <= mass <= 0.72
    # A narrower interval with the same mass does not exist.
    assert hi - lo <= line.quantile(0.84) - line.quantile(0.16)


def test_report_names_a_point_estimate_outside_the_belief():
    """The report gives the belief's own most likely value and 68% interval,
    and names the fit's point estimate when it lies outside that interval."""
    specs = [ParamSpec("a", 0.0, lo=-1.0, hi=1.0)]
    residuals = _gaussian_residuals(["a"], [0.3], [[20.0]])
    off = _build(specs, {"a": 0.0}, residuals)
    lo, hi = off.highest_density_interval("a")
    assert lo <= off.most_likely("a") <= hi
    assert off.most_likely("a") == pytest.approx(0.3, abs=0.02)
    assert not lo <= 0.0 <= hi
    text = "\n".join(off.describe())
    assert "the fit's point estimate 0 lies outside it" in text
    on = _build(specs, {"a": 0.3}, residuals)
    assert "point estimate" not in "\n".join(on.describe())


def _two_basin_beliefs():
    """Beliefs around two basins of (a, b): (0.2, 0.9) and (0.8, 0.1)."""
    specs = [
        ParamSpec("a", 0.5, lo=0.0, hi=1.0),
        ParamSpec("b", 0.5, lo=0.0, hi=1.0)
    ]
    config = BeliefConfig(num_draws=8, line_max_evals=40)
    first = _build(specs, {
        "a": 0.2,
        "b": 0.9
    },
                   _gaussian_residuals(["a", "b"], [0.2, 0.9],
                                       np.eye(2) * 40),
                   config=config)
    second = _build(specs, {
        "a": 0.8,
        "b": 0.1
    },
                    _gaussian_residuals(["a", "b"], [0.8, 0.1],
                                        np.eye(2) * 40),
                    config=config,
                    seed=1)
    return first, second


def test_mixed_beliefs_draw_each_basin_jointly():
    """A mixture over basins samples a basin, then its values: no draw pairs
    one basin's value of a with the other's value of b.

    Each factor covers both basins, the point estimate is the first
    basin's, and the report says the recordings fit several parameter
    sets.
    """
    first, second = _two_basin_beliefs()
    mixed = mix_beliefs([first, second], [1.0, 1.0])
    assert mixed.map_estimate == {"a": 0.2, "b": 0.9}
    assert mixed.num_draws == 16
    rows = mixed.sample(np.random.default_rng(0), 400)
    near_first = sum(r["a"] < 0.5 < r["b"] for r in rows)
    near_second = sum(r["b"] < 0.5 < r["a"] for r in rows)
    assert near_first + near_second == 400
    assert 140 < near_first < 260
    lo, hi = mixed.highest_density_interval("a")
    assert lo < 0.25 and hi > 0.75
    assert "2 distinct parameter sets" in "\n".join(mixed.describe())


def test_mixed_beliefs_round_trip_and_join():
    """Checkpoints keep the mixture, and joining a prior factor keeps each
    draw's basin."""
    first, second = _two_basin_beliefs()
    mixed = mix_beliefs([first, second], [3.0, 1.0])
    restored = ParameterBelief.from_dict(mixed.to_dict())
    assert restored.weights == pytest.approx([0.75, 0.25])
    assert len(restored.components) == 2
    assert np.array_equal(restored.draws, mixed.draws)
    extra = prior_belief([ParamSpec("c", 0.5, lo=0.0, hi=1.0)], {"c": 0.0},
                         {"c": 1.0},
                         BeliefConfig(num_draws=16),
                         seed=0)
    joined = join_beliefs(mixed, extra)
    assert len(joined.components) == 2
    rows = joined.sample(np.random.default_rng(1), 200)
    assert all((r["a"] < 0.5) == (r["b"] > 0.5) and "c" in r for r in rows)


def test_tempered_line_widens_within_its_traced_range():
    """A Gaussian line at four times its temperature is twice as wide; its
    cumulative mass inverts its quantiles, moving draws to the tempered line
    keeps their order, and past its traced range a tempered line is flat, never
    wider."""
    grid = np.linspace(-12.0, 12.0, 241)
    line = LinePosterior.from_neg_log(grid, 0.5 * grid**2)
    assert line.tempered(1.0) is line
    hot = line.tempered(4.0)
    assert np.trapz(hot.density, hot.grid) == pytest.approx(1.0)
    assert np.sqrt(hot.variance()) == pytest.approx(2.0, rel=0.02)
    assert hot.mode() == pytest.approx(line.mode())
    q = np.array([0.05, 0.3, 0.5, 0.9])
    assert line.cdf([line.quantile(p) for p in q]) == pytest.approx(q,
                                                                    abs=1e-6)
    z = line.sample(np.random.default_rng(0), 50)
    moved = line.transport(z, hot)
    assert np.array_equal(np.argsort(moved), np.argsort(z))
    assert moved == pytest.approx(2.0 * z, abs=0.05)
    short = np.linspace(-1.0, 1.0, 21)
    narrow = LinePosterior.from_neg_log(short, 0.5 * (short / 0.2)**2)
    flat = narrow.tempered(1e4)
    assert flat.grid[0] == -1.0 and flat.grid[-1] == 1.0
    assert np.ptp(flat.density) < 0.01 * np.max(flat.density)


def test_inflated_belief_tempers_lines_and_moves_draws():
    """At four times the temperature each line and the draws widen about twice
    about the same mode, each draw keeping its place in the order; the report
    states the temperature, and checkpoints keep it."""
    specs = [
        ParamSpec("a", 0.3, lo=-10, hi=10),
        ParamSpec("b", 2.0, lo=0.1, hi=20.0, scale="log")
    ]
    belief = _build(
        specs, {
            "a": 0.3,
            "b": 2.0
        }, _gaussian_residuals(["a", "b"], [0.3, 2.0],
                               np.eye(2) * 20))
    assert belief.inflated(1.0) is belief
    hot = belief.inflated(4.0)
    assert hot.inflation == 4.0 and hot.noise_scale == belief.noise_scale
    for j, name in enumerate(["a", "b"]):
        ratio = np.sqrt(hot.lines[name].variance() /
                        belief.lines[name].variance())
        assert 1.7 < ratio < 2.05
        assert hot.most_likely(name) == pytest.approx(belief.most_likely(name))
        before, after = belief.draws[:, j], hot.draws[:, j]
        if name == "b":
            before, after = np.log(before), np.log(after)
        assert np.array_equal(np.argsort(after), np.argsort(before))
        assert 1.5 < np.std(after) / np.std(before) < 2.2
    assert "4 times the fit's temperature" in "\n".join(hot.describe())
    assert "the fit's temperature" not in "\n".join(belief.describe())
    restored = ParameterBelief.from_dict(hot.to_dict())
    assert restored.inflation == 4.0
    assert np.array_equal(restored.draws, hot.draws)
    rows = hot.sample(np.random.default_rng(0), 4000)
    assert np.std([r["a"] for r in rows
                   ]) == pytest.approx(np.sqrt(hot.lines["a"].variance()),
                                       rel=0.05)


def test_inflated_belief_keeps_discrete_values_and_basins():
    """Tempering leaves discrete parameters' draws alone, widens each basin of
    a mixture about its own values, and survives a join."""
    belief = prior_belief([
        ParamSpec("a", 0.5, lo=0.0, hi=1.0),
        ParamSpec("n", 1.0, lo=0.0, hi=3.0, discrete=True)
    ], {
        "a": 0.5,
        "n": 1.0
    }, {
        "a": 0.05,
        "n": 1.0
    },
                          BeliefConfig(num_draws=16),
                          seed=0)
    hot = belief.inflated(4.0)
    assert np.array_equal(hot.draws[:, 1], belief.draws[:, 1])
    assert np.std(hot.draws[:, 0]) > 1.5 * np.std(belief.draws[:, 0])
    first, second = _two_basin_beliefs()
    mixed = mix_beliefs([first, second], [1.0, 1.0])
    hot_mix = mixed.inflated(4.0)
    assert [c.inflation for c in hot_mix.components] == [4.0, 4.0]
    assert np.array_equal(hot_mix.draws,
                          np.vstack([c.draws for c in hot_mix.components]))
    for rows in (slice(0, 8), slice(8, 16)):
        assert np.std(hot_mix.draws[rows,
                                    0]) > 1.5 * np.std(mixed.draws[rows, 0])
    sampled = hot_mix.sample(np.random.default_rng(0), 400)
    assert all((r["a"] < 0.5) == (r["b"] > 0.5) for r in sampled)
    lo, hi = hot_mix.highest_density_interval("a")
    assert lo < 0.25 and hi > 0.75
    extra = prior_belief([ParamSpec("c", 0.5, lo=0.0, hi=1.0)], {"c": 0.0},
                         {"c": 1.0},
                         BeliefConfig(num_draws=16),
                         seed=0)
    assert join_beliefs(hot_mix, extra).inflation == 4.0


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


def test_lines_traced_in_lockstep_are_the_same():
    """Tracing every line in lockstep through a batch scorer evaluates the same
    points and gives the same belief as tracing line by line, with one batch
    per round."""
    specs = [
        ParamSpec("a", 0.3, lo=-10, hi=10),
        ParamSpec("b", 0.7, lo=0.01, hi=10, scale="log"),
        ParamSpec("c", 0.5, lo=0.0, hi=1.0),
    ]
    residuals = _gaussian_residuals(["a", "b", "c"], [0.3, 0.7, 0.5],
                                    np.diag([20.0, 15.0, 0.001]),
                                    offset=[0.01])
    map_params = {"a": 0.3, "b": 0.7, "c": 0.5}
    batches = []

    def batch(points):
        batches.append(len(points))
        return [residuals(params) for params in points]

    alone = _build(specs, map_params, residuals, prior_sigma=2.0)
    together = build_parameter_belief(
        specs,
        map_params,
        residuals,
        noise_sigma=_SIGMA_N,
        prior_centers={s.name: s.init_value
                       for s in specs},
        prior_sigmas={s.name: 2.0
                      for s in specs},
        config=BeliefConfig(num_draws=64, line_max_evals=40),
        seed=0,
        batch_residuals=batch)
    for name in ("a", "b", "c"):
        assert np.array_equal(alone.lines[name].grid,
                              together.lines[name].grid)
        assert np.array_equal(alone.lines[name].density,
                              together.lines[name].density)
    assert np.array_equal(alone.draws, together.draws)
    assert alone.evaluations == together.evaluations
    # Rounds score up to one point per line, and fewer rounds than points.
    assert max(batches) <= len(specs)
    assert len(batches) < alone.evaluations


def test_a_deadline_cuts_the_lines_short_and_they_keep_their_priors(
        monkeypatch):
    """No round of replays starts that would end after the deadline, judged by
    the slowest round so far.

    A free parameter whose line was cut short keeps its prior; a
    declared one keeps what was traced when that brackets its point
    estimate, else its prior; the report names them.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import parameter_belief as module
    specs = [
        ParamSpec("a", 0.3, lo=-10, hi=10),
        ParamSpec("b", 0.7, lo=0.0, hi=2.0),
    ]
    residuals = _gaussian_residuals(["a", "b"], [0.3, 0.7],
                                    np.diag([20.0, 15.0]))
    clock = {"now": 0.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    def batch(points):
        clock["now"] += 10.0
        return [residuals(params) for params in points]

    def build(deadline, free_names=()):
        return build_parameter_belief(specs, {
            "a": 0.3,
            "b": 0.7
        },
                                      residuals,
                                      noise_sigma=_SIGMA_N,
                                      prior_centers={
                                          "a": 0.3,
                                          "b": 0.7
                                      },
                                      prior_sigmas={
                                          "a": 2.0,
                                          "b": 0.5
                                      },
                                      config=BeliefConfig(num_draws=16,
                                                          line_max_evals=40),
                                      seed=0,
                                      batch_residuals=batch,
                                      free_names=free_names,
                                      deadline=deadline)

    clock["now"] = 0.0
    full = build(None)
    assert full.cut_short == ()
    # A deadline already past: no round starts, and every line keeps its
    # prior.
    clock["now"] = 100.0
    late = build(50.0, free_names=("b", ))
    assert late.cut_short == ("a", "b")
    assert late.lines["a"].mean() == pytest.approx(0.3, abs=1e-3)
    assert np.sqrt(late.lines["a"].variance()) == pytest.approx(2.0, rel=0.02)
    assert "time ran out before it finished the lines of a, b" in \
        "\n".join(late.describe())
    # Four rounds fit before the deadline, the fifth would not. The
    # declared line has probed both sides of its estimate by then (three
    # probes on the left, one on the right) and keeps them; the free one
    # keeps its prior.
    clock["now"] = 0.0
    cut = build(45.0, free_names=("b", ))
    assert clock["now"] == pytest.approx(40.0)
    assert cut.cut_short == ("a", "b")
    grid = cut.lines["a"].grid
    assert grid[0] < 0.3 < grid[-1]
    assert np.sqrt(cut.lines["a"].variance()) < 2.0
    assert np.sqrt(cut.lines["b"].variance()) == pytest.approx(np.sqrt(
        _prior_line_variance(0.7, 0.5, 0.0, 2.0)),
                                                               rel=0.02)
    restored = ParameterBelief.from_dict(cut.to_dict())
    assert restored.cut_short == ("a", "b")


def _prior_line_variance(center, sigma, lo, hi):
    """The variance of a Gaussian prior truncated to [lo, hi]."""
    grid = np.linspace(max(lo, center - 6 * sigma),
                       min(hi, center + 6 * sigma), 20001)
    dens = np.exp(-0.5 * ((grid - center) / sigma)**2)
    dens /= np.trapz(dens, grid)
    mean = np.trapz(grid * dens, grid)
    return float(np.trapz((grid - mean)**2 * dens, grid))


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


def test_a_persistent_misfit_counts_as_one_error():
    """A misfit the fit measured counts once: the temperature is the misfit
    over sigma_n^2, where spreading the same 27 sigma_n^2 over four residual
    terms gives 27/4.

    Mixing, joining and checkpoints keep it, and the report says what it
    means.
    """
    specs = [ParamSpec("a", 0.3, lo=-10, hi=10)]
    residuals = _gaussian_residuals(["a"], [0.3], [[20.0]],
                                    offset=[3.0 * _SIGMA_N] * 3)
    spread = _build(specs, {"a": 0.3}, residuals)
    once = _build(specs, {"a": 0.3}, residuals, misfit=27.0 * _SIGMA_N**2)
    assert once.noise_scale == pytest.approx(27.0)
    assert once.misfit == pytest.approx(27.0 * _SIGMA_N**2)
    ratio = np.sqrt(once.lines["a"].variance() / spread.lines["a"].variance())
    assert ratio == pytest.approx(2.0, rel=0.06)
    assert "persists through each recording" in "\n".join(once.describe())
    assert "persists" not in "\n".join(spread.describe())
    # With the noise's own SSE known, the report states the misfit as a
    # multiple of it.
    assert "times the" not in "\n".join(once.describe())
    scaled = _build(specs, {"a": 0.3},
                    residuals,
                    misfit=27.0 * _SIGMA_N**2,
                    noise_sse=3.0 * _SIGMA_N**2)
    assert "(9 times the" in "\n".join(scaled.describe())
    assert "missing or wrong mechanism" in "\n".join(scaled.describe())
    mixed = mix_beliefs([scaled, scaled], [1.0, 1.0])
    restored = ParameterBelief.from_dict(mixed.to_dict())
    assert restored.misfit == once.misfit
    assert restored.noise_sse == pytest.approx(3.0 * _SIGMA_N**2)
    extra = prior_belief([ParamSpec("c", 0.5, lo=0.0, hi=1.0)], {"c": 0.0},
                         {"c": 1.0},
                         BeliefConfig(num_draws=128),
                         seed=0)
    assert join_beliefs(mixed, extra).misfit == once.misfit
    assert join_beliefs(mixed,
                        extra).noise_sse == pytest.approx(3.0 * _SIGMA_N**2)
    # A misfit within the noise leaves the ordinary posterior.
    fits = _gaussian_residuals(["a"], [0.3], [[20.0]])
    assert _build(specs, {
        "a": 0.3
    }, fits, misfit=_SIGMA_N**2).noise_scale == pytest.approx(1.0)


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


def test_joined_beliefs_pair_their_draws():
    """Joining a fitted belief with a prior-only one keeps each factor's draws
    and lines, pairs draw i with draw i, and refuses overlapping or short
    factors."""
    config = BeliefConfig(num_draws=8)
    fitted = _build([ParamSpec("a", 0.0, lo=-1.0, hi=1.0)], {"a": 0.0},
                    _gaussian_residuals(["a"], [0.0], [[1.0]]),
                    config=config)
    spin = ParamSpec("spin", 0.0, lo=0.0, hi=1.0)
    prior = prior_belief([spin], {"spin": 0.0}, {"spin": 1.0},
                         config=config,
                         seed=1)
    joined = join_beliefs(fitted, prior)
    assert joined.names == ["a", "spin"]
    assert joined.num_draws == 8
    assert np.allclose(joined.draws[:, 0], fitted.draws[:, 0])
    assert np.allclose(joined.draws[:, 1], prior.draws[:, 0])
    assert joined.interval("spin", 0.95) == prior.interval("spin", 0.95)
    assert all(set(d) == {"a", "spin"} for d in joined.draw_dicts())
    with pytest.raises(ValueError, match="share"):
        join_beliefs(joined, prior)
    short = prior_belief([spin], {"spin": 0.0}, {"spin": 1.0},
                         config=BeliefConfig(num_draws=4),
                         seed=1)
    with pytest.raises(ValueError, match="as many"):
        join_beliefs(fitted, short)
