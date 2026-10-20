"""Tests for the shared rollout-sysID orchestrator.

One flow behind both the approach's final-commit fit and the agent's
``sim.fit`` tool: end-to-end fit + report + trust selection, whole-fit
memoization per (artifact, data) signature, and caller-local report
adjustment on cache hits.
"""
# pylint: disable=protected-access
import dataclasses
import time

import numpy as np
import pybullet as p
import pytest

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators import utils
from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
from predicators.code_sim_learning.identifiability import Verdict
from predicators.code_sim_learning.orchestrator import _build_belief, \
    _draw_shares, run_rollout_sysid
from predicators.code_sim_learning.parameter_belief import BeliefConfig
from predicators.code_sim_learning.rollout_env import num_rollouts_run
from predicators.structs import Action, Object, State, Type

_DOMINO_TYPE = Type("domino", ["x"])
_RESIDUAL_FEATURES = {"domino": ["x"]}
_DOMINO = Object("d0", _DOMINO_TYPE)


class _GainEnv:
    """DIRECT-client rollout env: ``x`` advances by ``0.01 * gain``/step."""

    def __init__(self):
        self._physics_client_id = p.connect(p.DIRECT)
        self._params = {"gain": 1.0}
        self._x = 0.0

    def get_physical_param_info(self):
        """Registry: one wide log-scale parameter, baseline 1.0."""
        return {
            "gain": {
                "default": 1.0,
                "lo": 0.1,
                "hi": 10.0,
                "scale": "log",
                "description": "",
            },
        }

    def apply_physical_param_overrides(self, params):
        """Sticky per-param merge, like the real env API."""
        self._params.update(params)

    def _set_state(self, state):
        self._x = float(state.get(_DOMINO, "x"))

    def step(self, action):
        """Advance ``x`` by the current gain's per-step motion."""
        del action
        self._x += 0.01 * self._params["gain"]
        return State({_DOMINO: np.array([self._x], dtype=float)})


def _trajectory(num_steps=10, gain=2.0):
    """Observed data generated at the true ``gain``."""
    states = [
        State({_DOMINO: np.array([0.01 * gain * t], dtype=float)})
        for t in range(num_steps + 1)
    ]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(num_steps)]
    return states, actions


def test_run_rollout_sysid_fit_cache_and_report_isolation():
    """The fit core is memoized per (artifact, data) key; adjusters and trust
    selection stay caller-local on deep-copied reports."""
    # The trust selection runs only without the joint belief, and an
    # earlier test may leave belief_joint_draws set.
    utils.reset_config({})
    env = _GainEnv()
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    traj = _trajectory()
    anchors = {"gain": 1.0}
    fit_cache = {}

    outcome = run_rollout_sysid(env, [traj], [spec],
                                _RESIDUAL_FEATURES,
                                anchors=anchors,
                                rms_cache={},
                                fit_cache=fit_cache,
                                fit_cache_key="vers_001")
    assert not outcome.from_cache
    assert outcome.num_survivors == 1
    assert outcome.report["gain"]["verdict"].applies_fitted
    assert abs(outcome.applied["gain"] - 2.0) < 0.2
    assert outcome.post_sse < outcome.pre_sse
    assert len(fit_cache) == 1

    # Reading the versioned adapter must not run inference or consume the
    # caller's RNG; mutating its diagnostics cannot corrupt the cache.
    count = num_rollouts_run()
    inference = outcome.inference
    assert num_rollouts_run() == count
    assert inference.schema_version == 1
    assert inference.estimator == "legacy_rollout_sysid"
    assert inference.uncertainty_kind == "legacy_widths"
    assert inference.point_estimate == outcome.fitted
    assert inference.selected_parameters == outcome.applied
    inference.selected_parameters["gain"] = 99.0
    inference.parameter_diagnostics["gain"]["verdict"] = Verdict.INCONSISTENT
    assert outcome.applied["gain"] != 99.0
    assert outcome.report["gain"]["verdict"].applies_fitted

    # Identical call: zero new rollouts, same applied values.
    n_before = num_rollouts_run()
    outcome2 = run_rollout_sysid(env, [traj], [spec],
                                 _RESIDUAL_FEATURES,
                                 anchors=anchors,
                                 rms_cache={},
                                 fit_cache=fit_cache,
                                 fit_cache_key="vers_001")
    assert outcome2.from_cache
    assert num_rollouts_run() == n_before
    assert outcome2.applied == outcome.applied

    # A cached call with a caller-local adjuster (the cross-cycle
    # INCONSISTENT demotion) changes ITS selection only - the cached
    # report and earlier outcomes are untouched (deep copy).
    def demote(_result, report, sse_fn):
        # The fit's own SSE probe rides along even on cache hits (the
        # cross-cycle arbitration needs it); sanity-check it is live.
        assert sse_fn is not None and np.isfinite(sse_fn({"gain": 1.0}))
        report["gain"]["verdict"] = Verdict.INCONSISTENT
        report["gain"]["note"] = "test demotion"

    outcome3 = run_rollout_sysid(env, [traj], [spec],
                                 _RESIDUAL_FEATURES,
                                 anchors=anchors,
                                 rms_cache={},
                                 fit_cache=fit_cache,
                                 fit_cache_key="vers_001",
                                 report_adjuster=demote,
                                 held={"gain": 1.5})
    assert outcome3.from_cache
    assert outcome3.applied == {"gain": 1.5}
    assert outcome3.inference.selected_parameters == {"gain": 1.5}
    assert outcome3.inference.parameter_diagnostics["gain"]["verdict"] == \
        Verdict.INCONSISTENT
    assert outcome.report["gain"]["verdict"].applies_fitted

    # A different artifact key recomputes.
    outcome4 = run_rollout_sysid(env, [traj], [spec],
                                 _RESIDUAL_FEATURES,
                                 anchors=anchors,
                                 rms_cache={},
                                 fit_cache=fit_cache,
                                 fit_cache_key="vers_002")
    assert not outcome4.from_cache
    assert len(fit_cache) == 2


def test_fit_reports_the_laplace_evidence_when_asked():
    """With code_sim_learning_fit_evidence on, a fit with a Jacobian at the MAP
    carries a finite evidence; off, none is computed."""
    env = _GainEnv()
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    traj = _trajectory()
    utils.reset_config({"code_sim_learning_fit_evidence": True})
    try:
        outcome = run_rollout_sysid(env, [traj], [spec],
                                    _RESIDUAL_FEATURES,
                                    anchors={"gain": 1.0},
                                    rms_cache={})
        assert outcome.evidence is not None
        assert np.isfinite(outcome.evidence.log_evidence)
        assert outcome.evidence.num_params == 1
        assert outcome.evidence.num_residuals > 0
        utils.reset_config({"code_sim_learning_fit_evidence": False})
        plain = run_rollout_sysid(env, [traj], [spec],
                                  _RESIDUAL_FEATURES,
                                  anchors={"gain": 1.0},
                                  rms_cache={})
        assert plain.evidence is None
    finally:
        utils.reset_config({})


def test_fit_builds_the_parameter_belief_when_asked():
    """With belief draws requested, the fit carries q(theta) around its MAP,
    applies the MAP itself, and a cached fit reuses the same draws."""
    env = _GainEnv()
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    traj = _trajectory()
    fit_cache = {}
    config = BeliefConfig(num_draws=16)
    outcome = run_rollout_sysid(env, [traj], [spec],
                                _RESIDUAL_FEATURES,
                                anchors={"gain": 1.0},
                                rms_cache={},
                                fit_cache=fit_cache,
                                fit_cache_key="vers_001",
                                belief_config=config)
    belief = outcome.belief
    assert belief is not None
    assert belief.names == ["gain"]
    assert belief.draws.shape == (16, 1)
    assert np.all((belief.draws >= 0.1) & (belief.draws <= 10.0))
    assert outcome.applied == {"gain": outcome.fitted["gain"]}
    lo, hi = belief.interval("gain")
    assert lo <= outcome.fitted["gain"] <= hi
    assert abs(outcome.fitted["gain"] - 2.0) < 0.2
    count = num_rollouts_run()
    again = run_rollout_sysid(env, [traj], [spec],
                              _RESIDUAL_FEATURES,
                              anchors={"gain": 1.0},
                              rms_cache={},
                              fit_cache=fit_cache,
                              fit_cache_key="vers_001",
                              belief_config=config)
    assert again.from_cache and num_rollouts_run() == count
    assert again.belief is not None
    assert np.array_equal(again.belief.draws, belief.draws)
    plain = run_rollout_sysid(env, [traj], [spec],
                              _RESIDUAL_FEATURES,
                              anchors={"gain": 1.0},
                              rms_cache={},
                              belief_config=BeliefConfig(num_draws=0))
    assert plain.belief is None


def test_draw_shares_follow_the_weights():
    """Draws split in proportion to the basin weights, at least one each when
    there are enough."""
    assert _draw_shares(16, [1.0, np.exp(-1.0 / 3.0)]) == [9, 7]
    assert _draw_shares(16, [1.0]) == [16]
    assert _draw_shares(2, [1.0, 1.0, 1.0]) == [1, 1, 0]
    shares = _draw_shares(16, [1.0, 0.01, 0.01])
    assert sum(shares) == 16 and min(shares) == 1


def test_belief_mixes_the_fit_basins():
    """A fit with two basins gets a belief over both, its draws split by
    weight, and the point estimate stays the best basin's; every basin's factor
    takes the temperature of the misfit that weighed them."""
    spec = ParamSpec("k", 0.5, lo=0.0, hi=1.0)

    def residuals(params):
        k = params["k"]
        return np.array([0.05 * 40.0 * (k - 0.2) * (k - 0.8) / 0.6])

    result = FitResult(names=["k"],
                       samples=np.array([[0.8]]),
                       log_probs=np.zeros(1),
                       noise_sigma=0.05,
                       prior_sigma=np.array([1e6]),
                       scales=["linear"],
                       basins=[({
                           "k": 0.8
                       }, 10.3, 1.0),
                               ({
                                   "k": 0.2
                               }, 10.5, float(np.exp(-1.0 / 3.0)))],
                       misfit=0.3)
    belief = _build_belief(result, [spec], {"k": 0.5}, residuals,
                           BeliefConfig(num_draws=16, line_max_evals=40), 1)
    assert len(belief.components) == 2
    assert belief.num_draws == 16
    assert belief.map_estimate == {"k": 0.8}
    values = np.sort(belief.draws[:, 0])
    assert values[0] < 0.5 < values[-1]
    assert [c.noise_scale for c in belief.components] == \
        pytest.approx([0.3 / 0.05**2] * 2)
    assert belief.misfit == pytest.approx(0.3)


def test_a_basin_the_deadline_leaves_untraced_is_left_out(monkeypatch):
    """No basin's lines start that, by the longest basin so far, would end
    after the deadline: the belief mixes the basins traced, their draws make up
    the whole count, and the report says how many were left out."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import orchestrator
    spec = ParamSpec("k", 0.5, lo=0.0, hi=1.0)
    clock = {"now": 0.0}
    monkeypatch.setattr(orchestrator.time, "monotonic", lambda: clock["now"])

    def residuals(params):
        clock["now"] += 1.0
        k = params["k"]
        return np.array([0.05 * 40.0 * (k - 0.2) * (k - 0.8) / 0.6])

    result = FitResult(names=["k"],
                       samples=np.array([[0.8]]),
                       log_probs=np.zeros(1),
                       noise_sigma=0.05,
                       prior_sigma=np.array([1e6]),
                       scales=["linear"],
                       basins=[({
                           "k": 0.8
                       }, 10.3, 1.0),
                               ({
                                   "k": 0.2
                               }, 10.5, float(np.exp(-1.0 / 3.0)))],
                       misfit=0.3)
    config = BeliefConfig(num_draws=16, line_max_evals=40)
    both = _build_belief(result, [spec], {"k": 0.5}, residuals, config, 1)
    assert both.untraced_basins == 0
    # The basins mirror each other, so each takes half of the time both
    # took; after the first, the time left is less than that.
    total = clock["now"]
    clock["now"] = 0.0
    alone = _build_belief(result, [spec], {"k": 0.5},
                          residuals,
                          config,
                          1,
                          deadline=0.75 * total)
    assert alone.untraced_basins == 1
    assert len(alone.components) == 1
    assert alone.num_draws == 16
    assert alone.components[0].map_estimate == {"k": 0.8}
    assert alone.components[0].num_draws == 16
    assert "1 more of the parameter sets" in "\n".join(alone.describe())


def test_belief_traces_lines_for_the_free_parameters():
    """A free parameter sits at its own value in every point the belief scores
    and gets a line like a fitted one, under the prior an unfitted parameter
    has: one the recordings bind keeps its draws near its own value, and one
    they ignore keeps its prior's spread."""
    spec = ParamSpec("k", 0.5, lo=0.0, hi=1.0)
    bound = ParamSpec("roll", 0.001, lo=0.0, hi=0.02)
    loose = ParamSpec("tint", 0.5, lo=0.0, hi=1.0)
    scored = []

    def residuals(params):
        scored.append(dict(params))
        # One noise sigma per 0.01 of k and per 0.0005 of roll.
        return 0.05 * np.array([(params["k"] - 0.5) / 0.01,
                                (params["roll"] - 0.001) / 0.0005])

    result = FitResult(names=["k"],
                       samples=np.array([[0.5]]),
                       log_probs=np.zeros(1),
                       noise_sigma=0.05,
                       prior_sigma=np.array([1e6]),
                       scales=["linear"])
    belief = _build_belief(result, [spec], {"k": 0.5},
                           residuals,
                           BeliefConfig(num_draws=64),
                           1,
                           free_specs=[bound, loose])
    assert belief.names == ["k", "roll", "tint"]
    assert belief.map_estimate["roll"] == pytest.approx(0.001)
    assert all(point.keys() == {"k", "roll", "tint"} for point in scored)
    assert {point["roll"] for point in scored} != {0.001}
    roll = belief.draws[:, 1]
    assert np.all(np.abs(roll - 0.001) < 0.003)
    assert np.std(belief.draws[:, 2]) > 0.1
    # The recordings leave tint's line flat, so it stops after its outward
    # probes; roll's is traced in full.
    tint_points = [point for point in scored if point["tint"] != 0.5]
    roll_points = [point for point in scored if point["roll"] != 0.001]
    assert len(tint_points) <= 8 < len(roll_points)


def test_free_lines_wait_for_time_before_the_deadline():
    """A fit traces its free parameters' lines only while enough of its time is
    left before its deadline; otherwise they keep their priors.

    Past its deadline it still returns a belief: the lines it could not
    trace keep their priors and the belief names them.
    """
    utils.reset_config({})
    env = _GainEnv()
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    free = ParamSpec("drag", 0.5, lo=0.0, hi=1.0)
    belief_config = BeliefConfig(num_draws=8)
    roomy = run_rollout_sysid(env, [_trajectory()], [spec],
                              _RESIDUAL_FEATURES,
                              anchors={"gain": 1.0},
                              rms_cache={},
                              belief_config=belief_config,
                              belief_free_specs=[free])
    assert roomy.belief is not None
    assert roomy.belief.names == ["gain", "drag"]
    assert roomy.belief.cut_short == ()
    late = run_rollout_sysid(env, [_trajectory()], [spec],
                             _RESIDUAL_FEATURES,
                             anchors={"gain": 1.0},
                             rms_cache={},
                             config=dataclasses.replace(
                                 SysIdConfig.from_cfg(),
                                 deadline=time.monotonic()),
                             belief_config=belief_config,
                             belief_free_specs=[free])
    assert late.belief is not None and late.belief.names == ["gain"]
    assert late.belief.cut_short == ("gain", )
