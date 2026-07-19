"""Tests for the rollout system-ID pure-Python pieces.

The rollout engine's physics (PyBullet resets, velocity zeroing) is
exercised end-to-end by the domino experiments; here we unit-test the
noise-aware curvature probe (given an ``sse_fn``), the settled-tail
trajectory truncation, and the fresh-env-per-rollout plumbing.
"""
# pylint: disable=protected-access,import-outside-toplevel

import numpy as np
import pybullet as p

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators.code_sim_learning import physical_sysid
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
from predicators.code_sim_learning.physical_sysid import \
    fit_params_rollout_trimmed, identifiability_report, \
    select_trustworthy_params, truncate_settled_tail
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
    # The stub trajectories are plain strings; skip the data-derived
    # residual scaling (exercised by its own tests).
    monkeypatch.setattr(physical_sysid, "compute_residual_scaling",
                        lambda *_a, **_k: None)


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
    monkeypatch.setattr(physical_sysid, "compute_residual_scaling",
                        lambda *_a, **_k: None)
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


# ── Stale-override hygiene ────────────────────────────────────────


class _FakeStickyEnv:
    """Env stub with the real API's sticky per-param merge semantics."""

    def __init__(self, info):
        self._info = info
        self.overrides = {}

    def get_physical_param_info(self):
        """Return the per-param registry this stub was built with."""
        return self._info

    def apply_physical_param_overrides(self, params):
        """Merge overrides in, rejecting any undeclared param name."""
        unknown = set(params) - set(self._info)
        assert not unknown, unknown
        self.overrides.update(params)


_REGISTRY = {
    "lateral_friction": {
        "default": 0.5,
        "lo": 0.01,
        "hi": 2.0,
        "scale": "log",
        "description": "",
    },
    "rolling_friction": {
        "default": 0.006,
        "lo": 0.0,
        "hi": 0.1,
        "description": "",
    },
}


def test_pin_all_physical_params_reverts_undeclared():
    """A fit declaring a subset must not inherit an earlier fit's values."""
    env = _FakeStickyEnv(_REGISTRY)
    # Earlier fit's eval left lateral_friction at 2.0 on the shared env.
    physical_sysid._pin_all_physical_params(env, {"lateral_friction": 2.0})
    assert env.overrides["lateral_friction"] == 2.0
    # A later fit declares only rolling_friction: lateral_friction must be
    # pinned back to the env default, not stay at 2.0.
    physical_sysid._pin_all_physical_params(env, {"rolling_friction": 0.02})
    assert env.overrides["lateral_friction"] == 0.5
    assert env.overrides["rolling_friction"] == 0.02


def test_pin_all_physical_params_env_without_registry():
    """Envs without a registry just get the params applied."""

    class _Bare:
        applied = None

        def apply_physical_param_overrides(self, params):
            """Record whatever params get applied."""
            self.applied = dict(params)

    env = _Bare()
    physical_sysid._pin_all_physical_params(env, {"k": 1.0})
    assert env.applied == {"k": 1.0}


def test_apply_identified_reverts_params_dropped_from_declaration():
    """Regression for run_20260707_112310: an intermediate artifact applied
    lateral_friction 1.9993, the final artifact declared only rolling_friction,
    and the planner silently kept 1.9993."""
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    approach = AgentSimLearningApproach.__new__(AgentSimLearningApproach)
    env = _FakeStickyEnv(_REGISTRY)
    approach._base_env = env
    approach._identified_physical_params = {}
    apply = approach._apply_identified_physical_params

    apply({"lateral_friction": 1.9993, "rolling_friction": 0.007})
    assert env.overrides["lateral_friction"] == 1.9993

    apply({"rolling_friction": 0.006})
    assert env.overrides["lateral_friction"] == 0.5  # reverted to default
    assert env.overrides["rolling_friction"] == 0.006
    # The tracked dict mirrors the current declaration exactly, so env
    # recreation re-applies only what the final artifact declared.
    assert approach._identified_physical_params == {"rolling_friction": 0.006}


# ── Fresh-env-per-rollout plumbing ────────────────────────────────


class _FreshRolloutEnv:
    """Env stub owning a real DIRECT PyBullet client (empty world), so velocity
    zeroing and client disposal exercise the real API."""

    def __init__(self, log):
        self._physics_client_id = p.connect(p.DIRECT)
        log.append(self)
        self.reset_to = None
        self.steps = 0

    def apply_physical_param_overrides(self, params):
        """No-op: this stub carries no physical params."""
        del params

    def _set_state(self, state):
        self.reset_to = state

    def step(self, action):
        """Advance the step counter and return a marker of progress."""
        del action
        self.steps += 1
        return f"post_{self.steps}"

    @property
    def connected(self):
        """Whether this stub's PyBullet client is still connected."""
        info = p.getConnectionInfo(self._physics_client_id)
        return bool(info["isConnected"])


def test_rollout_states_factory_builds_and_disposes_fresh_env():
    """A factory ``base_env`` gets one fresh env per rollout, disposed after.

    Regression for run_20260708_213258: rollouts on a reused env are
    nondeterministic (same-theta SSE alternated 0.15/78).
    """
    states, actions = _trajectory([0.0, 0.1, 0.2])
    built = []

    def factory():
        return _FreshRolloutEnv(built)

    out1 = physical_sysid.rollout_states(factory, states[0], actions,
                                         {"k": 1.0})
    out2 = physical_sysid.rollout_states(factory, states[0], actions,
                                         {"k": 1.0})
    assert out1 == ["post_1", "post_2"] == out2
    assert len(built) == 2  # one fresh env per rollout
    assert all(not env.connected for env in built)  # both disposed
    assert all(env.reset_to is states[0] for env in built)


def test_rollout_states_factory_disposes_on_rollout_error():
    """The fresh env is disconnected even when a step raises."""
    states, actions = _trajectory([0.0, 0.1, 0.2])
    built = []

    class _ExplodingEnv(_FreshRolloutEnv):

        def step(self, action):
            raise RuntimeError("mid-rollout failure")

    def factory():
        return _ExplodingEnv(built)

    try:
        physical_sysid.rollout_states(factory, states[0], actions, {})
        assert False, "expected the step error to propagate"
    except RuntimeError:
        pass
    assert len(built) == 1
    assert not built[0].connected


def test_rollout_states_env_instance_is_not_disposed():
    """Passing an env instance keeps the caller-owned-env behavior."""
    states, actions = _trajectory([0.0, 0.1, 0.2])
    built = []
    env = _FreshRolloutEnv(built)
    out = physical_sysid.rollout_states(env, states[0], actions, {"k": 1.0})
    assert out == ["post_1", "post_2"]
    assert env.connected  # caller-owned env stays alive
    p.disconnect(env._physics_client_id)  # pylint: disable=protected-access


# ── Residual scaling (angle wrap + per-feature normalization) ──────

_ANGULAR_TYPE = Type("spinner", ["x", "yaw"], angular_features=["yaw"])


def _angular_trajectory(xs, yaws):
    obj = Object("s0", _ANGULAR_TYPE)
    states = [
        State({obj: np.array([x, y], dtype=float)}) for x, y in zip(xs, yaws)
    ]
    actions = [
        Action(np.zeros(1, dtype=np.float32)) for _ in range(len(states) - 1)
    ]
    return states, actions


def test_residual_scaling_wraps_angular_features():
    """-pi vs +pi is the same orientation, so the residual must be ~0."""
    scaling = physical_sysid.ResidualScaling(angular=frozenset({("spinner",
                                                                 "yaw")}),
                                             scales={
                                                 ("spinner", "yaw"):
                                                 float(np.pi)
                                             })
    near_zero = scaling.residual("spinner", "yaw", -np.pi + 1e-6, np.pi)
    assert abs(near_zero) < 1e-5
    # A genuine quarter-turn error survives the wrap: pi/2 / pi = 0.5.
    quarter = scaling.residual("spinner", "yaw", np.pi / 2, 0.0)
    assert abs(quarter - 0.5) < 1e-9


def test_residual_scaling_normalizes_linear_features():
    """Linear residuals are divided by the feature's scale entry."""
    scaling = physical_sysid.ResidualScaling(angular=frozenset(),
                                             scales={("spinner", "x"): 0.5})
    assert abs(scaling.residual("spinner", "x", 0.6, 0.5) - 0.2) < 1e-12
    # Unknown features fall back to scale 1 (raw difference).
    assert abs(scaling.residual("spinner", "q", 0.6, 0.5) - 0.1) < 1e-12


def test_compute_residual_scaling_reads_type_metadata():
    """Angular features come from the Type; linear scales from the span."""
    xs = [0.0, 0.1, 0.4]  # span 0.4 > floor
    yaws = [0.0, 1.0, 3.0]
    traj = _angular_trajectory(xs, yaws)
    scaling = physical_sysid.compute_residual_scaling(
        [traj], {"spinner": ["x", "yaw"]})
    assert scaling is not None
    assert ("spinner", "yaw") in scaling.angular
    assert abs(scaling.scales[("spinner", "yaw")] - np.pi) < 1e-12
    assert abs(scaling.scales[("spinner", "x")] - 0.4) < 1e-12


def test_compute_residual_scaling_floors_static_features(monkeypatch):
    """A feature that never moves gets the floor, not a ~0 divisor."""
    from predicators.settings import CFG
    monkeypatch.setattr(CFG, "code_sim_learning_rollout_feature_scale_floor",
                        0.05)
    traj = _angular_trajectory([0.2, 0.2, 0.2], [0.0, 0.0, 0.0])
    scaling = physical_sysid.compute_residual_scaling([traj],
                                                      {"spinner": ["x"]})
    assert scaling is not None
    assert abs(scaling.scales[("spinner", "x")] - 0.05) < 1e-12


def test_compute_residual_scaling_disabled(monkeypatch):
    """The CFG kill switch restores the raw (legacy) objective."""
    from predicators.settings import CFG
    monkeypatch.setattr(CFG, "code_sim_learning_rollout_scale_residuals",
                        False)
    traj = _angular_trajectory([0.0, 0.1], [0.0, 0.1])
    assert physical_sysid.compute_residual_scaling(
        [traj], {"spinner": ["x", "yaw"]}) is None


# ── Rest-point segmentation (multiple shooting) ────────────────────


def test_split_at_rest_points_separates_phases():
    """Two motion phases with a long rest between -> two segments."""
    xs = ([0.01 * i for i in range(11)] + [0.1] * 40 +
          [0.1 + 0.01 * i for i in range(11)] + [0.2] * 40)
    segments = physical_sysid.split_at_rest_points(_trajectory(xs),
                                                   _PROCESS_FEATURES,
                                                   motion_tol=1e-3,
                                                   min_rest_steps=10,
                                                   margin=5)
    assert len(segments) == 2
    for states, actions in segments:
        assert len(states) == len(actions) + 1
    # Each segment starts at an at-rest observed state and keeps a
    # settle margin after its last motion.
    first_states, first_actions = segments[0]
    # Last active step index 9 -> keep 9 + 1 + margin 5 = 15 steps,
    # matching the truncate_settled_tail convention.
    assert len(first_actions) == 15
    assert float(first_states[0].get(Object("d0", _DOMINO_TYPE), "x")) == 0.0
    second_states, _ = segments[1]
    assert abs(
        float(second_states[0].get(Object("d0", _DOMINO_TYPE), "x")) -
        0.1) < 1e-12


def test_split_at_rest_points_keeps_short_pauses_together():
    """A pause shorter than min_rest_steps must not split the segment."""
    xs = ([0.01 * i for i in range(11)] + [0.1] * 4 +
          [0.1 + 0.01 * i for i in range(11)] + [0.2] * 30)
    segments = physical_sysid.split_at_rest_points(_trajectory(xs),
                                                   _PROCESS_FEATURES,
                                                   motion_tol=1e-3,
                                                   min_rest_steps=10,
                                                   margin=5)
    assert len(segments) == 1


def test_split_at_rest_points_static_trajectory_yields_nothing():
    """No scored motion -> no segments (no parameter signal at all)."""
    segments = physical_sysid.split_at_rest_points(_trajectory([0.5] * 50),
                                                   _PROCESS_FEATURES,
                                                   motion_tol=1e-3,
                                                   min_rest_steps=10,
                                                   margin=5)
    assert not segments


# ── Verdict hardening ──────────────────────────────────────────────


def test_identified_downgraded_on_single_segment():
    """A sharp posterior from one explainable segment is only weak."""

    def sse_fn(params):
        return 1e4 * (params["friction"] - 0.1)**2

    specs = [ParamSpec("friction", 0.1, lo=0.01, hi=2.0)]
    result = _single_sample_result(["friction"], [0.1], [0.75])
    report = identifiability_report(result, sse_fn, specs, num_explainable=1)
    assert report["friction"]["verdict"].startswith("weakly identified")
    # With two segments the same posterior keeps the full verdict.
    report2 = identifiability_report(result, sse_fn, specs, num_explainable=2)
    assert report2["friction"]["verdict"] == "identified"


def test_insensitive_param_overrides_probe_and_is_not_applied():
    """The sensitivity screen wins over apparent probe curvature."""

    def sse_fn(params):
        return 1e4 * (params["mass"] - 0.05)**2  # chaos-fake curvature

    specs = [ParamSpec("mass", 0.05, lo=0.005, hi=1.0)]
    result = _single_sample_result(["mass"], [0.03], [0.75])
    result.sensitivity = {
        "mass": {
            "sse_span": 0.001,
            "noise_floor": 0.01,
            "sensitive": False
        }
    }
    report = identifiability_report(result, sse_fn, specs, num_explainable=3)
    assert report["mass"]["verdict"].startswith("insensitive")
    applied = select_trustworthy_params({"mass": 0.03}, {"mass": 0.05},
                                        ["mass"],
                                        report,
                                        anchors={"mass": 0.05})
    assert applied == {"mass": 0.05}


def test_untrusted_param_falls_back_to_anchor_not_declared_init():
    """Unsupported agent hypotheses must not reach the planner."""
    report = {"restitution": {"verdict": "unknown"}}
    applied = select_trustworthy_params({"restitution": 0.13},
                                        {"restitution": 0.15}, ["restitution"],
                                        report,
                                        anchors={"restitution": 0.02})
    assert applied == {"restitution": 0.02}
    # Without an anchor the declared init remains the fallback.
    applied2 = select_trustworthy_params({"restitution": 0.13},
                                         {"restitution": 0.15},
                                         ["restitution"], report)
    assert applied2 == {"restitution": 0.15}


def test_annotated_weak_verdicts_still_apply():
    """Downgraded-but-supported verdicts keep applying the fitted value."""
    report = {
        "friction": {
            "verdict":
            "weakly identified (sharp posterior, but only 1 "
            "explainable segment(s) back it)"
        }
    }
    applied = select_trustworthy_params({"friction": 0.09}, {"friction": 0.2},
                                        ["friction"],
                                        report,
                                        anchors={"friction": 0.5})
    assert applied == {"friction": 0.09}


def test_trimming_uses_rms_cache(monkeypatch):
    """A second call with the same signature reuses the cached sweep."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = [_trajectory([0.0, 0.1, 0.2]), _trajectory([0.0, 0.05, 0.1])]
    calls = {"sweep": 0}

    def fake_min_rms(_env, trajectories, *_args, **_kwargs):
        calls["sweep"] += 1
        return [0.001] * len(trajectories)

    def fake_fit(_env, _trajectories, *_args, **_kwargs):
        return FitResult(names=["friction"],
                         samples=np.array([[0.1]]),
                         log_probs=np.zeros(1),
                         jacobian=None,
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.375]))

    monkeypatch.setattr(physical_sysid, "min_explainable_rms", fake_min_rms)
    monkeypatch.setattr(physical_sysid, "fit_params_rollout", fake_fit)
    monkeypatch.setattr(physical_sysid, "per_trajectory_rms",
                        lambda *_a, **_k: [0.001, 0.001])
    cache = {}
    for _ in range(2):
        _result, survivors, _rms = fit_params_rollout_trimmed(
            None,
            trajs,
            specs, {"domino": ["x"]},
            anchors={"friction": 0.5},
            rms_cache=cache)
        assert len(survivors) == 2
    assert calls["sweep"] == 1


def test_map_at_box_bound_is_not_trusted():
    """A MAP pinned at its box edge must not be applied, however sharp.

    Regression for the replay of run_20260711_141026: mass fit to its
    1.0 hi bound with posterior_std 5e-11 ("identified" per the
    one-sided curvature probe, which reads the box wall as sharpness)
    while the true value was 0.1.
    """

    def sse_fn(params):
        return 1e6 * (1.0 - params["mass"])**2  # optimum on the box wall

    specs = [ParamSpec("mass", 0.1, lo=0.005, hi=1.0)]
    result = _single_sample_result(["mass"], [1.0], [0.75])
    report = identifiability_report(result, sse_fn, specs, num_explainable=5)
    assert report["mass"]["verdict"].startswith("at box hi bound")
    applied = select_trustworthy_params({"mass": 1.0}, {"mass": 0.05},
                                        ["mass"],
                                        report,
                                        anchors={"mass": 0.1})
    assert applied == {"mass": 0.1}
    # An interior MAP with the same curvature stays identified.
    result_interior = _single_sample_result(["mass"], [0.5], [0.75])
    report2 = identifiability_report(result_interior,
                                     lambda p: 1e6 * (p["mass"] - 0.5)**2,
                                     specs,
                                     num_explainable=5)
    assert report2["mass"]["verdict"] == "identified"


# ── Grid sweep: flat set, multi-pass, flat-edge refinement ─────────


def _run_sweep(monkeypatch, sse_fn, specs, anchors, **flags):
    """Run the grid seeding against a synthetic SSE landscape."""
    from predicators.settings import CFG
    for flag, value in flags.items():
        monkeypatch.setattr(CFG, flag, value)

    def fake_sse(_env, _trajs, params, *_args, **_kwargs):
        return sse_fn(params)

    monkeypatch.setattr(physical_sysid, "compute_rollout_sse", fake_sse)
    seeded, info = physical_sysid._grid_seed_physical_specs(None, ["traj"],
                                                            specs,
                                                            {"domino": ["x"]},
                                                            [], [],
                                                            None,
                                                            anchors=anchors)
    return {s.name: s.init_value for s in seeded}, info


def _saturating_sse(params):
    """The measured run_20260711_224624 shape: lateral friction saturates
    above ~0.45 (topple reach), and low spinning friction buys a token
    1.6% SSE gain by compensating lateral's quantization error."""
    lat = params["lateral_friction"]
    spin = params["spinning_friction"]
    base = 60.0 if lat >= 0.45 else 60.0 + 300.0 * (0.45 - lat) / 0.45
    return base - (1.0 if spin < 0.05 else 0.0)


_HIGH_FRICTION_SPECS = [
    ParamSpec("lateral_friction", 0.1, lo=0.01, hi=2.0, scale="log"),
    ParamSpec("spinning_friction", 0.5, lo=0.01, hi=2.0, scale="log"),
]
_HIGH_FRICTION_ANCHORS = {"lateral_friction": 0.1, "spinning_friction": 0.5}


def test_grid_sweep_keeps_anchor_for_insignificant_gain(monkeypatch):
    """A compensating param must not move off its anchor for a ~1.6% gain.

    Regression for run_20260711_224624: with lateral_friction parked
    high, the argmin sweep dragged spinning_friction 0.5 -> 0.024 (true
    0.5) for a 1.6% SSE improvement - well inside the flat tolerance.
    """
    seeds, _info = _run_sweep(monkeypatch, _saturating_sse,
                              _HIGH_FRICTION_SPECS, _HIGH_FRICTION_ANCHORS)
    assert seeds["spinning_friction"] == 0.5


def test_grid_sweep_refines_flat_edge_toward_anchor(monkeypatch):
    """A true value mid-grid-gap becomes expressible via edge bisection.

    On run_20260711_224624 the 7-point log grid had no candidate between
    0.342 and 0.827, so a saturated landscape (flat above ~0.45) was
    always reported as the 0.827 grid point against a true 0.5. The
    flat-edge bisection must land near the saturation onset instead,
    and the flat interval must report the full data-equivalent range.
    """
    seeds, info = _run_sweep(monkeypatch, _saturating_sse,
                             _HIGH_FRICTION_SPECS, _HIGH_FRICTION_ANCHORS)
    assert 0.44 <= seeds["lateral_friction"] <= 0.54
    lo, hi = info["lateral_friction"]["flat_interval"]
    assert lo <= 0.46
    assert hi == 2.0
    # Without refinement the seed is quantized to the grid point.
    seeds_coarse, _info2 = _run_sweep(
        monkeypatch,
        _saturating_sse,
        _HIGH_FRICTION_SPECS,
        _HIGH_FRICTION_ANCHORS,
        code_sim_learning_rollout_grid_refine_evals=0)
    assert abs(seeds_coarse["lateral_friction"] - 0.827) < 1e-2


def test_grid_sweep_second_pass_relaxes_early_param(monkeypatch):
    """A param swept before its neighbor moved is re-examined next pass."""

    def coupled_sse(params):
        a, b = params["a"], params["b"]
        return 50.0 * (a * b - 6.0)**2 + 50.0 * (b - 3.0)**2

    specs = [
        ParamSpec("a", 1.0, lo=0.0, hi=4.0),
        ParamSpec("b", 1.0, lo=0.0, hi=4.0),
    ]
    anchors = {"a": 1.0, "b": 1.0}
    # Pass 1 greedily sends a to the 4.0 grid edge (with b still at 1);
    # after b moves to 2, pass 2 must relax a back to 3.
    seeds, _info = _run_sweep(monkeypatch,
                              coupled_sse,
                              specs,
                              anchors,
                              code_sim_learning_rollout_grid_seed_points=5,
                              code_sim_learning_rollout_grid_refine_evals=0)
    assert seeds == {"a": 3.0, "b": 2.0}
    seeds_one_pass, _info2 = _run_sweep(
        monkeypatch,
        coupled_sse,
        specs,
        anchors,
        code_sim_learning_rollout_grid_seed_points=5,
        code_sim_learning_rollout_grid_refine_evals=0,
        code_sim_learning_rollout_grid_sweep_passes=1)
    assert seeds_one_pass == {"a": 4.0, "b": 2.0}


def test_grid_sweep_sharp_basin_not_dragged_to_anchor(monkeypatch):
    """Refinement must respect a sharp basin's own extent.

    The relative flat tolerance shrinks with the best SSE, so on clean
    data (deep narrow basin, e.g. the low-friction-arm landscape) the
    anchor-ward bisection stops at the basin edge instead of dragging
    the seed toward the anchor.
    """

    def basin_sse(params):
        v = params["lateral_friction"]
        return 0.05 if 0.12 <= v <= 0.16 else 80.0

    specs = [ParamSpec("lateral_friction", 0.5, lo=0.01, hi=2.0, scale="log")]
    seeds, info = _run_sweep(monkeypatch, basin_sse, specs,
                             {"lateral_friction": 0.5})
    assert 0.12 <= seeds["lateral_friction"] <= 0.165
    lo, hi = info["lateral_friction"]["flat_interval"]
    assert 0.12 <= lo and hi <= 0.165


def test_flat_interval_annotates_identified_report():
    """A wide data-equivalent interval is surfaced next to "identified"."""
    specs = [ParamSpec("friction", 0.6, lo=0.01, hi=2.0, scale="log")]
    result = _single_sample_result(["friction"], [0.6], [0.75])
    result.sensitivity = {
        "friction": {
            "sse_span": 100.0,
            "noise_floor": 0.0,
            "sensitive": True,
            "flat_interval": (0.5, 2.0),
        }
    }

    def sharp_sse(params):
        return 1e3 * (np.log(params["friction"]) - np.log(0.6))**2

    report = identifiability_report(result,
                                    sharp_sse,
                                    specs,
                                    num_explainable=3)
    entry = report["friction"]
    assert entry["verdict"] == "identified"
    assert entry["flat_interval"] == (0.5, 2.0)
    assert entry["flat_wide"]
    text = physical_sysid.format_identifiability(report)
    assert "data-equivalent over [0.5, 2]" in text
    assert "not a unique optimum" in text
    # A degenerate interval is neither rendered nor flagged wide.
    result.sensitivity["friction"]["flat_interval"] = (0.6, 0.6)
    report2 = identifiability_report(result,
                                     sharp_sse,
                                     specs,
                                     num_explainable=3)
    assert not report2["friction"]["flat_wide"]
    assert "data-equivalent" not in physical_sysid.format_identifiability(
        report2)
