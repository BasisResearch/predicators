"""Tests for the zero-gradient (threshold/gate) bracket search that backs up
Levenberg-Marquardt in ``code_sim_learning.lm``."""

from typing import List

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.lm import solve_lm, zero_jacobian_columns


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
