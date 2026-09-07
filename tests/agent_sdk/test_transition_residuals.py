"""Tests for the per-transition ``sim.residuals()`` report extensions.

Two behaviors added for the start-of-session divergence report: (1) the
report always sweeps features OUTSIDE the scored scope and lists the
ones that differ from the data (a feature that changes but has no rule
is a candidate missing mechanism - previously invisible whenever a
scope was declared); (2) worst examples are located as (traj, step,
executing option) instead of a flat concatenated index.
"""

from pathlib import Path
from typing import cast

import numpy as np

from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools.synthesis import SynthesisToolkit, \
    create_synthesis_tools
from predicators.structs import Action, Object, State, Type

_BALL_TYPE = Type("ball", ["x", "y"])

_EMPTY_SCOPE_SIMULATOR = """
def _noop_rule(state, updates, params):
    _ = params["dummy_k"]
    return updates

RESIDUAL_RULES = [_noop_rule]
PARAM_SPECS = [ParamSpec("dummy_k", 0.1, lo=0.0, hi=1.0)]
RESIDUAL_FEATURES = {"ball": []}
"""

_X_SCOPE_SIMULATOR = """
def _noop_rule(state, updates, params):
    _ = params["dummy_k"]
    return updates

RESIDUAL_RULES = [_noop_rule]
PARAM_SPECS = [ParamSpec("dummy_k", 0.1, lo=0.0, hi=1.0)]
RESIDUAL_FEATURES = {"ball": ["x"]}
"""


class _NoApproach:
    """Minimal stand-in: no trajectory grouping, no rollout surface."""


def _drifting_triples(num_steps: int = 6):
    """Base predicts x frozen at 0 and y frozen at 0; observed x drifts by 1
    per step while y stays 0, so ball.x differs at every transition (worst at
    the last) and ball.y differs nowhere."""
    ball = Object("ball0", _BALL_TYPE)
    triples = []
    for t in range(num_steps):
        s_base = State({ball: np.array([0.0, 0.0])})
        s_obs = State({ball: np.array([float(t + 1), 0.0])})
        triples.append((s_base, Action(np.zeros(1, dtype=np.float32)), s_obs))
    return triples


def _make_toolkit(tmp_path: Path, simulator_src: str) -> SynthesisToolkit:
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text(simulator_src, encoding="utf-8")
    return create_synthesis_tools(exec_ns={},
                                  base_pred_triples=_drifting_triples(),
                                  inferred_residual_features={},
                                  simulator_file=str(sim_file),
                                  versions_dir=str(tmp_path / "versions"),
                                  approach=cast(SynthesisBackend,
                                                _NoApproach()))


def test_out_of_scope_diffs_reported_with_empty_scope(tmp_path) -> None:
    """A declared-empty scope no longer hides the differing feature."""
    toolkit = _make_toolkit(tmp_path, _EMPTY_SCOPE_SIMULATOR)
    report = toolkit.residuals_runner()
    assert "OUTSIDE the declared scope" in report
    assert "ball.x" in report
    # All six transitions differ; the worst is the last one, located by
    # trajectory and step rather than a flat index.
    assert "6/6 differ" in report
    assert "traj 0 step 5" in report
    # The quiet feature must not be flagged.
    assert "ball.y" not in report
    # The scoped table is empty but the report still renders.
    assert "no in-scope features" in report


def test_in_scope_feature_not_duplicated_out_of_scope(tmp_path) -> None:
    """A feature under the declared scope stays out of the outside section, and
    its worst examples carry the (traj, step) location."""
    toolkit = _make_toolkit(tmp_path, _X_SCOPE_SIMULATOR)
    report = toolkit.residuals_runner()
    assert "ball.x" in report
    assert "Worst" in report
    assert "traj 0 step 5" in report
    outside = report.split("OUTSIDE")[-1] if "OUTSIDE" in report else ""
    assert "ball.x" not in outside


def test_missing_simulator_scores_base_divergence(tmp_path) -> None:
    """With no simulator.py at all (cycle 0), the report scores the base
    simulator alone: everything is out of scope, so the drifting feature shows
    up as a candidate missing mechanism (fit_params is a no-op)."""
    toolkit = create_synthesis_tools(exec_ns={},
                                     base_pred_triples=_drifting_triples(),
                                     inferred_residual_features={},
                                     simulator_file=str(tmp_path /
                                                        "simulator.py"),
                                     versions_dir=str(tmp_path / "versions"),
                                     approach=cast(SynthesisBackend,
                                                   _NoApproach()))
    report = toolkit.residuals_runner(fit_params=True)
    assert "no_simulator_yet" in report
    assert "ball.x" in report
    assert "6/6 differ" in report
    assert "traj 0 step 5" in report
    assert "ball.y" not in report
    # The open-loop / physics modes need a real file.
    assert "write the file first" in toolkit.residuals_runner(rollout=True)
