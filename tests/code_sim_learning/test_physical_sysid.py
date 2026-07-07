"""Tests for the rollout system-ID pure-Python pieces.

The rollout engine itself (PyBullet resets, velocity zeroing) is
exercised end-to-end by the domino experiments; here we unit-test the
noise-aware curvature probe (given an ``sse_fn``) and the settled-tail
trajectory truncation.
"""

import numpy as np

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators.code_sim_learning import physical_sysid
from predicators.code_sim_learning.physical_sysid import \
    fit_params_rollout_trimmed, identifiability_report, \
    select_trustworthy_params, truncate_settled_tail
from predicators.code_sim_learning.training import FitResult, ParamSpec
from predicators.structs import Action, Object, State, Type

_DOMINO_TYPE = Type("domino", ["x"])
_ROBOT_TYPE = Type("robot", ["x"])
_PROCESS_FEATURES = {"domino": ["x"]}


def _trajectory(domino_xs, robot_xs=None):
    """Build a (states, actions) trajectory from per-step feature values."""
    domino = Object("d0", _DOMINO_TYPE)
    robot = Object("r0", _ROBOT_TYPE)
    if robot_xs is None:
        robot_xs = [0.0] * len(domino_xs)
    states = [
        State({
            domino: np.array([dx], dtype=float),
            robot: np.array([rx], dtype=float),
        }) for dx, rx in zip(domino_xs, robot_xs)
    ]
    actions = [
        Action(np.zeros(1, dtype=np.float32)) for _ in range(len(states) - 1)
    ]
    return states, actions


def test_truncate_cuts_static_tail():
    """Motion for 10 steps, static for 90 -> cut at last motion + margin."""
    xs = [0.01 * i for i in range(11)] + [0.1] * 90
    states, actions = truncate_settled_tail(_trajectory(xs),
                                            _PROCESS_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    # Last moving step is index 9 (states[9] -> states[10]); keep 9+1+5.
    assert len(actions) == 15
    assert len(states) == 16


def test_truncate_keeps_intermediate_pause():
    """Push -> pause -> second push: the cut anchors to the LAST motion."""
    xs = ([0.01 * i for i in range(11)] + [0.1] * 40 +
          [0.1 + 0.01 * i for i in range(10)] + [0.2] * 60)
    states, actions = truncate_settled_tail(_trajectory(xs),
                                            _PROCESS_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    # Last motion is the step onto the final plateau (index 60); both
    # pushes and the pause between them are retained.
    assert len(actions) == 66
    assert states[-1].get(Object("d0", _DOMINO_TYPE), "x") == 0.2


def test_truncate_noop_when_motion_never_stops():
    """Continuous motion to the last step -> nothing is cut."""
    xs = [0.01 * i for i in range(100)]
    traj = _trajectory(xs)
    states, actions = truncate_settled_tail(traj,
                                            _PROCESS_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    assert len(actions) == len(traj[1])
    assert len(states) == len(traj[0])


def test_truncate_static_trajectory_keeps_margin_prefix():
    """No scored feature ever moves -> keep only the margin prefix."""
    xs = [0.5] * 100
    states, actions = truncate_settled_tail(_trajectory(xs),
                                            _PROCESS_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    assert len(actions) == 5
    assert len(states) == 6


def test_truncate_ignores_unscored_features():
    """Robot motion is not scored, so it must not defer the cut."""
    domino_xs = [0.01 * i for i in range(11)] + [0.1] * 90
    robot_xs = [0.02 * i for i in range(101)]  # robot moves the whole time
    states, actions = truncate_settled_tail(_trajectory(domino_xs, robot_xs),
                                            _PROCESS_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    assert len(actions) == 15
    assert len(states) == 16


def _single_sample_result(names, values, prior_sigma):
    return FitResult(names=list(names),
                     samples=np.array([values], dtype=float),
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=0.05,
                     prior_sigma=np.asarray(prior_sigma, dtype=float))


def test_probe_identifies_curved_param_with_deterministic_sse():
    """A strongly curved direction is identified; a flat one is not."""
    result = _single_sample_result(["curved", "flat"], [0.5, 0.5],
                                   [0.25, 0.25])
    specs = [
        ParamSpec("curved", 0.5, lo=0.0, hi=1.0),
        ParamSpec("flat", 0.5, lo=0.0, hi=1.0),
    ]

    def sse_fn(params):
        return 1e4 * (params["curved"] - 0.5)**2

    report = identifiability_report(result, sse_fn, specs)
    assert report["curved"]["verdict"] == "identified"
    assert "NOT identified" in report["flat"]["verdict"]


def test_probe_discounts_curvature_below_noise_floor():
    """Same-theta SSE jitter must not read as curvature.

    Regression for run_20260705_203314: a chaotic rollout objective
    jittered by thousands of SSE units at the SAME parameters, and the
    probe declared every parameter identified because the jitter looked
    like a large SSE increase under perturbation. With the noise-aware
    probe, a perturbation response inside the same-theta spread yields
    no curvature -> NOT identified.
    """
    result = _single_sample_result(["friction"], [0.5], [0.375])
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0)]

    # Deterministic pseudo-noise: SSE cycles through a large spread at
    # every call, independent of the parameters (pure jitter, no signal).
    values = [24748.0, 27359.0, 9671.0, 25406.0, 20000.0, 30000.0]
    calls = {"n": 0}

    def sse_fn(_params):
        v = values[calls["n"] % len(values)]
        calls["n"] += 1
        return v

    report = identifiability_report(result, sse_fn, specs)
    assert "NOT identified" in report["friction"]["verdict"]


def test_probe_survives_noise_floor_on_top_of_signal():
    """Real curvature well above the noise floor is still identified."""
    result = _single_sample_result(["friction"], [0.5], [0.375])
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0)]
    rng = np.random.default_rng(0)

    def sse_fn(params):
        # Strong quadratic signal + jitter two orders below the signal
        # at prior scale.
        signal = 1e6 * (params["friction"] - 0.1)**2
        return signal + float(rng.uniform(0.0, 100.0))

    report = identifiability_report(result, sse_fn, specs)
    assert report["friction"]["verdict"] == "identified"


def test_select_trustworthy_params_keeps_init_for_unidentified():
    """Un-contracted params keep the declared init; identified ones apply.

    On uninformative data the fitted value of a NOT-identified parameter
    is arbitrary (grid seed lands on a noise minimum), so applying it
    would move the planner's belief randomly.
    """
    fitted = {"friction": 1.337, "restitution": 0.44}
    inits = {"friction": 0.5, "restitution": 0.02}
    report = {
        "friction": {
            "verdict": "NOT identified (posterior ~= prior; MAP arbitrary)"
        },
        "restitution": {
            "verdict": "identified"
        },
    }
    applied = select_trustworthy_params(fitted, inits,
                                        ["friction", "restitution"], report)
    assert applied == {"friction": 0.5, "restitution": 0.44}


def test_select_trustworthy_params_weakly_identified_applies():
    """Weak contraction still counts as data-constrained -> apply."""
    fitted = {"friction": 0.12}
    inits = {"friction": 0.5}
    report = {"friction": {"verdict": "weakly identified"}}
    applied = select_trustworthy_params(fitted, inits, ["friction"], report)
    assert applied == {"friction": 0.12}


def _patch_fit_and_rms(monkeypatch, fit_thetas, rms_by_count):
    """Stub the heavy PyBullet fit/residual calls for trimming tests.

    ``fit_thetas`` maps the number of trajectories passed to the fit to
    the friction value it "fits"; ``rms_by_count`` maps it to the per-
    trajectory RMS list returned at those params.
    """

    def fake_fit(_env, trajectories, *_args, **_kwargs):
        theta = fit_thetas[len(trajectories)]
        return FitResult(names=["friction"],
                         samples=np.array([[theta]]),
                         log_probs=np.zeros(1),
                         jacobian=None,
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.375]))

    def fake_rms(_env, trajectories, *_args, **_kwargs):
        return rms_by_count[len(trajectories)]

    monkeypatch.setattr(physical_sysid, "fit_params_rollout", fake_fit)
    monkeypatch.setattr(physical_sysid, "per_trajectory_rms", fake_rms)


def test_trimming_drops_unexplainable_and_refits(monkeypatch):
    """A high-RMS trajectory is dropped and the refit's theta is returned."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = ["chaotic", "clean"]  # stand-ins; the stubs only count them
    # Pooled fit (2 trajs) lands at 0.34; refit on the 1 survivor at 0.1.
    _patch_fit_and_rms(monkeypatch, {
        2: 0.34,
        1: 0.1
    }, {
        2: [0.7, 0.001],
        1: [0.001]
    })
    result, survivors, rms = fit_params_rollout_trimmed(
        None, trajs, specs, {"domino": ["x"]})
    assert survivors == ["clean"]
    assert rms == [0.7, 0.001]
    assert result.point_estimate["friction"] == 0.1


def test_trimming_keeps_all_when_explainable(monkeypatch):
    """No trajectory above threshold -> single fit, nothing dropped."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = ["a", "b"]
    _patch_fit_and_rms(monkeypatch, {2: 0.1}, {2: [0.001, 0.002]})
    result, survivors, rms = fit_params_rollout_trimmed(
        None, trajs, specs, {"domino": ["x"]})
    assert survivors == ["a", "b"]
    assert result.point_estimate["friction"] == 0.1
    assert rms == [0.001, 0.002]


def test_consistency_loop_drops_disagreeing_survivor(monkeypatch):
    """Explainable-but-wrong data loses to cleaner data on disagreement.

    Regression for the measured failure: a quiet chaotic shove was
    explainable at friction ~1.34 (best RMS 0.034) while the clean
    topple's best is at ~0.1 (best RMS 0.01); the joint fit compromised
    where the clean trajectory fits far worse than its own best. The
    consistency loop must drop the higher-best-RMS survivor (the shove)
    and refit on the clean one.
    """
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    shove, clean = "shove", "clean"
    state = {"fit_called": False}

    def fake_fit(_env, trajectories, *_args, **_kwargs):
        state["fit_called"] = True
        theta = {
            ("shove", "clean"): 1.34,
            ("clean", ): 0.1
        }[tuple(trajectories)]
        return FitResult(names=["friction"],
                         samples=np.array([[theta]]),
                         log_probs=np.zeros(1),
                         jacobian=None,
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.375]))

    def fake_rms(_env, trajectories, *_args, **_kwargs):
        if not state["fit_called"]:
            # min_explainable_rms sweep: both look explainable.
            return [0.034, 0.010]
        # Consistency check at the joint fit (friction 1.34): the shove
        # fits at its best, the clean trajectory fits 6x worse.
        assert tuple(trajectories) == ("shove", "clean")
        return [0.034, 0.060]

    monkeypatch.setattr(physical_sysid, "fit_params_rollout", fake_fit)
    monkeypatch.setattr(physical_sysid, "per_trajectory_rms", fake_rms)
    result, survivors, _rms = fit_params_rollout_trimmed(
        None, [shove, clean], specs, {"domino": ["x"]})
    assert survivors == ["clean"]
    assert result.point_estimate["friction"] == 0.1


def test_trimming_all_dropped_pins_result_at_inits(monkeypatch):
    """Nothing explainable -> empty survivors, result PINNED at inits.

    The fit must not run at all (nothing to learn from), and the point
    estimate must be the declared init so that applying it is a no-op
    even if a downstream probe on chaotic data falsely says
    "identified".
    """
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = ["chaos1", "chaos2"]
    # No fit_thetas entries: any call to the (stubbed) fit would KeyError.
    _patch_fit_and_rms(monkeypatch, {}, {2: [0.8, 0.6]})
    result, survivors, _rms = fit_params_rollout_trimmed(
        None, trajs, specs, {"domino": ["x"]})
    assert survivors == []
    assert result.point_estimate["friction"] == 0.5


def test_mcmc_samples_bypass_probe():
    """With a real chain, widths come from the samples, not the probe."""
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.normal(0.1, 0.01, size=200),  # contracted -> identified
        rng.normal(0.5, 0.375, size=200),  # prior-wide -> NOT identified
    ])
    result = FitResult(names=["friction", "restitution"],
                       samples=samples,
                       log_probs=np.zeros(200),
                       jacobian=None,
                       noise_sigma=0.05,
                       prior_sigma=np.array([0.375, 0.375]))
    report = identifiability_report(result)
    assert report["friction"]["verdict"] == "identified"
    assert "NOT identified" in report["restitution"]["verdict"]
