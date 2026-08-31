"""Tests for the zero-gradient (threshold/gate) bracket search that backs up
Levenberg-Marquardt in ``code_sim_learning.lm``."""

from typing import List

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.lm import \
    bracket_search_zero_gradient_params, solve_lm, zero_jacobian_columns


def test_zero_jacobian_columns() -> None:
    """Only identically-zero columns count."""
    jac = np.array([[1.0, 0.0, 2.0], [0.5, 0.0, 0.0]])
    assert zero_jacobian_columns(jac) == [1]
    assert zero_jacobian_columns(np.zeros((0, 0))) == []


def test_solve_lm_bracket_searches_threshold_params() -> None:
    """A gate parameter with a zero LM gradient is still fit from data."""
    xs = np.linspace(0.0, 1.0, 41)
    true_gain, true_thresh = 2.0, 0.6
    observed = true_gain * (xs > true_thresh)

    def residuals(theta: np.ndarray) -> np.ndarray:
        gain, thresh = float(theta[0]), float(theta[1])
        return gain * (xs > thresh) - observed

    specs = [
        ParamSpec("gain", 1.0, lo=0.0, hi=5.0),
        ParamSpec("thresh", 0.2, lo=0.0, hi=1.0),
    ]
    notes: List[str] = []
    theta, jac = solve_lm(residuals, specs, 200, "test", notes_out=notes)
    assert jac is not None
    assert abs(theta[1] - true_thresh) < 0.03
    assert abs(theta[0] - true_gain) < 1e-3
    assert len(notes) == 1 and notes[0].startswith("thresh:")
    assert "moved it" in notes[0]


def test_solve_lm_reports_flat_gate_params() -> None:
    """A gate no data point crosses is reported as not fit, kept at init."""
    xs = np.linspace(0.0, 0.4, 11)  # every x below every threshold tried
    observed = 3.0 * xs

    def residuals(theta: np.ndarray) -> np.ndarray:
        slope, thresh = float(theta[0]), float(theta[1])
        return slope * xs + 10.0 * (xs > thresh) - observed

    specs = [
        ParamSpec("slope", 1.0, lo=0.0, hi=5.0),
        ParamSpec("thresh", 0.7, lo=0.5, hi=1.0),
    ]
    notes: List[str] = []
    theta, _ = solve_lm(residuals, specs, 200, "test", notes_out=notes)
    assert abs(theta[0] - 3.0) < 1e-3
    assert theta[1] == 0.7
    assert len(notes) == 1 and "NOT fit from data" in notes[0]


def test_zero_jacobian_columns_tolerance_and_prior_rows() -> None:
    """Junk-scale columns are flagged, and prior rows cannot mask them."""
    # Column 1 carries only finite-difference junk (~1e-10) next to an
    # O(1) column: the exact-zero test missed it, the tolerance flags it.
    jac = np.array([[1.0, 1e-10], [0.5, -1e-10]])
    assert zero_jacobian_columns(jac) == [1]
    # A MAP objective appends one prior row per parameter, and a prior
    # row's own-column entry is a nonzero constant - so on the full
    # Jacobian no column can ever read zero (the dead-bracket bug of
    # the 2026-08-30 bridge runs). Excluding the prior rows finds the
    # data-flat column again.
    data_rows = np.array([[1.0, 0.0], [0.5, 0.0]])
    prior_rows = np.array([[0.2, 0.0], [0.0, 0.2]])
    map_jac = np.vstack([data_rows, prior_rows])
    assert zero_jacobian_columns(map_jac) == []
    assert zero_jacobian_columns(map_jac, n_prior_rows=2) == [1]


def test_solve_lm_map_prior_rows_bracket_still_fires() -> None:
    """The bracket search fires on a MAP objective (prior rows folded).

    Regression for the 2026-08-30 bridge runs: with the Gaussian prior
    folded in as residual rows, the all-rows exact-zero column test
    could never fire and the bracket search was dead code on the rollout
    MAP path (seed0's cure_steps improvement went unfound).
    """
    xs = np.linspace(0.0, 1.0, 41)
    true_gain, true_thresh = 2.0, 0.6
    observed = true_gain * (xs > true_thresh)
    specs = [
        ParamSpec("gain", 1.0, lo=0.0, hi=5.0),
        ParamSpec("thresh", 0.2, lo=0.0, hi=1.0),
    ]
    centers = np.array([1.0, 0.2])
    sigmas = np.array([5.0, 5.0])  # wide prior: the data should win

    def residuals(theta: np.ndarray) -> np.ndarray:
        gain, thresh = float(theta[0]), float(theta[1])
        data = gain * (xs > thresh) - observed
        prior = 0.05 * (np.asarray(theta) - centers) / sigmas
        return np.concatenate([data, prior])

    notes: List[str] = []
    flat: List[str] = []
    theta, _ = solve_lm(residuals,
                        specs,
                        200,
                        "test",
                        notes_out=notes,
                        n_prior_rows=2,
                        flat_params_out=flat)
    assert abs(theta[1] - true_thresh) < 0.03
    assert abs(theta[0] - true_gain) < 1e-2
    assert any(n.startswith("thresh:") and "moved it" in n for n in notes)
    assert not flat


def test_bracket_search_flat_param_settled_from_edge_evals() -> None:
    """A box-flat parameter is settled from the two edge evaluations, is
    reported in ``flat_out``, and never pays the full 9-point grid."""
    calls: List[np.ndarray] = []

    def residuals(z: np.ndarray) -> np.ndarray:
        calls.append(np.array(z))
        return np.array([3.0 - z[0]])  # depends only on param 0

    specs = [
        ParamSpec("slope", 3.0, lo=0.0, hi=5.0),
        ParamSpec("gate", 0.7, lo=0.5, hi=1.0),
    ]
    lo = np.array([0.0, 0.5])
    hi = np.array([5.0, 1.0])
    z = np.array([3.0, 0.7])
    flat: List[str] = []
    z_new, _, notes = bracket_search_zero_gradient_params(residuals,
                                                          z,
                                                          lo,
                                                          hi, [1],
                                                          specs,
                                                          "test",
                                                          flat_out=flat)
    assert z_new[1] == 0.7
    assert flat == ["gate"]
    assert len(notes) == 1 and "NOT fit from data" in notes[0]
    # One baseline evaluation plus the two box edges.
    assert len(calls) == 3


def test_bracket_search_flat_test_ignores_prior_rows() -> None:
    """Prior rows must not make a data-flat parameter look responsive.

    At a box edge the prior rows alone raise the TOTAL SSE well above
    the flat tolerance; the flat verdict must therefore be judged on the
    data rows, or every data-flat parameter pays the full search and
    loses its honest 'NOT fit from data' note.
    """
    center = 0.75

    def residuals(z: np.ndarray) -> np.ndarray:
        data = np.array([3.0 - z[0]])
        prior = np.array([0.0, (z[1] - center) / 0.5])
        return np.concatenate([data, prior])

    specs = [
        ParamSpec("slope", 3.0, lo=0.0, hi=5.0),
        ParamSpec("gate", center, lo=0.5, hi=1.0),
    ]
    lo = np.array([0.0, 0.5])
    hi = np.array([5.0, 1.0])
    z = np.array([3.0, center])
    flat: List[str] = []
    _, _, notes = bracket_search_zero_gradient_params(residuals,
                                                      z,
                                                      lo,
                                                      hi, [1],
                                                      specs,
                                                      "test",
                                                      n_prior_rows=2,
                                                      flat_out=flat)
    assert flat == ["gate"]
    assert len(notes) == 1 and "NOT fit from data" in notes[0]
