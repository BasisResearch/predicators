"""Trajectory preparation for the rollout system-ID objective.

Settled-tail truncation, rest-point segmentation (multiple shooting),
and the per-(type, feature) residual scaling shared by every SSE/RMS
evaluation of one fit. See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for the identification problem these serve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from predicators.code_sim_learning.config import DEFAULT_NOISE_SIGMA, \
    SysIdConfig
from predicators.code_sim_learning.rollout_env import RolloutTrajectory
from predicators.observation_noise import ObservationNoise
from predicators.structs import Action, State, Type

logger = logging.getLogger(__name__)


def _filter_params(
        config: SysIdConfig) -> Tuple[Optional[ObservationNoise], int, float]:
    """The fit-side filter's ``(noise, window, sigmas)``: the declared channel
    with its detection window and threshold when the filter is on and a channel
    is declared, else ``(None, 1, 0.0)`` (the per-step detector, the observed
    first frame)."""
    noise = config.observation_noise
    if not config.noise_filter or noise is None:
        return None, 1, 0.0
    return noise, max(int(config.noise_window), 1), float(config.settle_sigmas)


def _wrap_angle(delta: float) -> float:
    """``delta`` wrapped to [-pi, pi]."""
    return float((delta + np.pi) % (2.0 * np.pi) - np.pi)


def _circular_mean(values: np.ndarray) -> float:
    """The mean direction of angles in radians (robust to the +-pi seam)."""
    return float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


def _mean_delta(after: np.ndarray, before: np.ndarray, angular: bool) -> float:
    """The displacement between two frame windows' means, wrapped for an
    angular feature; nan when either window is empty."""
    if after.size == 0 or before.size == 0:
        return float("nan")
    if angular:
        return _wrap_angle(_circular_mean(after) - _circular_mean(before))
    return float(after.mean() - before.mean())


def _windowed_active_steps(states: List[State], actions: List[Action],
                           residual_features: Dict[str, List[str]],
                           motion_tol: float, noise: ObservationNoise,
                           window: int, sigmas: float) -> List[int]:
    """The sigma-relative detector behind :func:`_active_step_indices`.

    Under a declared channel the per-step delta of a noisy feature is
    mostly noise (its standard deviation is ``sqrt(2) * sigma_f``), so
    the detector compares the mean of the ``window`` frames after step
    ``i`` with the mean of the ``window`` frames up to and including it:
    the difference has standard error ``sigma_f * sqrt(2 / window)`` at
    rest, and a step is active when it exceeds ``sigmas`` of those
    (floored at ``motion_tol``). Exact features keep the per-step test.
    Windows are truncated at the trajectory's ends. Objects are matched
    by name across frames; an object missing from a frame contributes
    nothing there.
    """
    num_steps = len(actions)
    if num_steps == 0:
        return []
    by_name = [{o.name: o for o in s} for s in states]
    tracked: List[Tuple[np.ndarray, bool, float, bool]] = []
    for obj in states[0]:
        feats = residual_features.get(obj.type.name, [])
        if not feats:
            continue
        angular = set(getattr(obj.type, "angular_features", ()))
        for feat in feats:
            series = np.array([
                float(s.get(m[obj.name], feat)) if obj.name in m else np.nan
                for s, m in zip(states, by_name)
            ])
            sigma = noise.feature_sigma(obj.type, feat)
            noisy = sigma > 0.0
            tol = (max(motion_tol, sigmas * sigma *
                       np.sqrt(2.0 / window)) if noisy else motion_tol)
            tracked.append((series, feat in angular, tol, noisy))
    active: List[int] = []
    for i in range(num_steps):
        for series, is_angular, tol, noisy in tracked:
            if noisy:
                before = series[max(0, i - window + 1):i + 1]
                after = series[i + 1:i + 1 + window]
                delta = _mean_delta(after[np.isfinite(after)],
                                    before[np.isfinite(before)], is_angular)
            else:
                delta = series[i + 1] - series[i]
                if is_angular:
                    delta = _wrap_angle(delta)
            if np.isfinite(delta) and abs(delta) > tol:
                active.append(i)
                break
    return active


def _rest_mean_state(states: List[State], index: int, window: int,
                     noise: ObservationNoise) -> State:
    """A copy of ``states[index]`` whose noisy features are the mean over the
    ``window`` frames ending at ``index`` (circular for angular features).

    The rest pose the noise hides: a rest-anchored segment starts here,
    so its rollout's initial condition carries ``sigma / sqrt(window)``
    of noise instead of one frame's ``sigma``. Exact features and
    objects absent from the earlier frames keep the frame's values.
    """
    start = states[index].copy()
    lo = max(0, index - window + 1)
    frames = states[lo:index + 1]
    if len(frames) <= 1:
        return start
    by_name = [{o.name: o for o in s} for s in frames]
    for obj in start:
        angular = set(getattr(obj.type, "angular_features", ()))
        for feat in obj.type.feature_names:
            if noise.feature_sigma(obj.type, feat) <= 0.0:
                continue
            values = np.array([
                float(s.get(m[obj.name], feat))
                for s, m in zip(frames, by_name) if obj.name in m
            ])
            if values.size <= 1:
                continue
            start.set(
                obj, feat,
                _circular_mean(values)
                if feat in angular else float(values.mean()))
    return start


def _active_step_indices(states: List[State],
                         actions: List[Action],
                         residual_features: Dict[str, List[str]],
                         motion_tol: float,
                         noise: Optional[ObservationNoise] = None,
                         window: int = 1,
                         sigmas: float = 0.0) -> List[int]:
    """Indices of steps where any scored feature moved more than
    ``motion_tol``.

    Shared observed-delta scan behind :func:`truncate_settled_tail` and
    :func:`split_at_rest_points`: step ``i`` compares ``states[i]`` to
    ``states[i + 1]`` (objects matched by name) and counts as active as
    soon as one in-scope feature's per-step delta exceeds the tolerance.
    With ``noise`` and a ``window`` above 1 (the fit-side filter, see
    :func:`_filter_params`) noisy features are judged by
    :func:`_windowed_active_steps` instead.
    """
    if noise is not None and window > 1:
        return _windowed_active_steps(states, actions, residual_features,
                                      motion_tol, noise, window, sigmas)
    active: List[int] = []
    for i in range(len(actions)):
        s_prev, s_next = states[i], states[i + 1]
        prev_by_name = {o.name: o for o in s_prev}
        for obj in s_next:
            feats = residual_features.get(obj.type.name, [])
            prev_obj = prev_by_name.get(obj.name)
            if not feats or prev_obj is None:
                continue
            if any(
                    abs(
                        float(s_next.get(obj, f)) -
                        float(s_prev.get(prev_obj, f))) > motion_tol
                    for f in feats):
                active.append(i)
                break
    return active


def truncate_settled_tail(
    trajectory: RolloutTrajectory,
    residual_features: Dict[str, List[str]],
    motion_tol: Optional[float] = None,
    margin: Optional[int] = None,
    config: Optional[SysIdConfig] = None,
) -> RolloutTrajectory:
    """Cut a recorded trajectory once its scored features have settled.

    Scans the OBSERVED per-step deltas of the ``residual_features`` and
    keeps everything up to the last step where any of them moved by more
    than ``motion_tol``, plus a ``margin`` of settle steps (so the
    rollout is still scored on coming to rest at the right pose). The
    static remainder is dropped: it contains no physics signal — the
    scored bodies no longer move — but re-scores whatever pose
    divergence the free-running rollout has accumulated on every
    remaining step, which is exactly the chaos-amplification term that
    drowned the friction signal in run_20260705_203314. Intermediate
    still phases are safe: the cut is anchored to the LAST motion, so a
    push -> settle -> second push trajectory keeps both pushes.

    A trajectory whose scored features never move carries no signal at
    all; it is truncated to the first ``margin`` steps (kept non-empty
    so callers' trajectory counts stay meaningful) and logged.
    """
    config = config or SysIdConfig.from_cfg()
    if motion_tol is None:
        motion_tol = config.settle_tol
    if margin is None:
        margin = config.settle_margin
    states, actions = trajectory
    noise, window, sigmas = _filter_params(config)
    active = _active_step_indices(states, actions, residual_features,
                                  motion_tol, noise, window, sigmas)
    last_active = active[-1] if active else -1
    if last_active < 0:
        logger.warning(
            "truncate_settled_tail: no scored feature ever moved more than "
            "%g in a %d-step trajectory; keeping only the first %d steps "
            "(the trajectory carries no physical-parameter signal).",
            motion_tol, len(actions), margin)
    keep = min(len(actions), last_active + 1 + margin)
    if keep >= len(actions):
        return trajectory
    return states[:keep + 1], actions[:keep]


@dataclass
class ResidualScaling:
    """Per-(type, feature) residual semantics for the rollout objective.

    ``angular`` features (declared on their :class:`~predicators.structs
    .Type` via ``angular_features``) have their prediction errors
    wrapped to [-pi, pi] before scoring, so equivalent orientations
    (a settled domino at roll -pi vs +pi) do not read as a (2*pi)^2
    error per step. Every residual is then divided by its feature's
    ``scales`` entry, making residuals dimensionless fractions of
    typical motion: without this, radians and meters share one implicit
    unit and rotation errors drown position information (measured on
    run_20260711_141026: roll+yaw carried 83% of the post-fit SSE, with
    max errors of exactly 2*pi and pi - pure representation artifacts).
    """

    angular: FrozenSet[Tuple[str, str]]
    scales: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def residual(self, type_name: str, feat: str, pred: float,
                 obs: float) -> float:
        """Scaled (and, for angular features, wrapped) ``pred - obs``."""
        key = (type_name, feat)
        diff = pred - obs
        if key in self.angular:
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        return diff / self.scales.get(key, 1.0)

    def signature(self) -> Tuple:
        """Hashable identity for caching explainability verdicts."""
        return (tuple(sorted(self.angular)),
                tuple(sorted(
                    (k, round(v, 12)) for k, v in self.scales.items())))


def compute_residual_scaling(
    trajectories: List[RolloutTrajectory],
    residual_features: Dict[str, List[str]],
    config: Optional[SysIdConfig] = None,
    noise_sigma: Optional[float] = None,
) -> Optional[ResidualScaling]:
    """Data-derived :class:`ResidualScaling` for a fit's trajectory set.

    Angular features come from each Type's declared ``angular_features``
    metadata (the states carry their types, so no env handle is
    needed). Scales: angular features get a constant ``pi`` (the
    largest possible wrapped error, so a full topple-direction mistake
    scores ~0.5); linear features get their observed span (max - min)
    across ALL observed states of the fit data, floored at
    ``CFG.code_sim_learning_rollout_feature_scale_floor`` so static
    features do not amplify sensor noise. Computed from observations
    only, so it is deterministic per dataset and MUST be shared across
    every SSE/RMS evaluation of one fit - per-trajectory scales would
    make trimming verdicts incomparable.

    Under a declared observation-noise channel (``config.observation_noise``)
    each scale also folds its feature's noise sigma in, so a residual the
    noise explains never reads as model error; see
    :meth:`~predicators.observation_noise.ObservationNoise.residual_scale`.
    ``noise_sigma`` is the fit's Gaussian width on scaled residuals
    (:data:`DEFAULT_NOISE_SIGMA` when None) and must be the one the fit
    scores with, since the fold is relative to it.

    Returns ``None`` when ``code_sim_learning_rollout_scale_residuals``
    is off (raw, unwrapped residuals - the legacy objective).
    """
    config = config or SysIdConfig.from_cfg()
    if not config.scale_residuals:
        return None
    floor = config.feature_scale_floor
    angular: Set[Tuple[str, str]] = set()
    lo: Dict[Tuple[str, str], float] = {}
    hi: Dict[Tuple[str, str], float] = {}
    types_by_name: Dict[str, Type] = {}
    for states, _actions in trajectories:
        for state in states:
            for obj in state:
                feats = residual_features.get(obj.type.name, [])
                if not feats:
                    continue
                types_by_name.setdefault(obj.type.name, obj.type)
                type_angular = set(getattr(obj.type, "angular_features", ()))
                for feat in feats:
                    key = (obj.type.name, feat)
                    if feat in type_angular:
                        angular.add(key)
                        continue
                    val = float(state.get(obj, feat))
                    lo[key] = min(lo.get(key, val), val)
                    hi[key] = max(hi.get(key, val), val)
    scales = {key: max(hi[key] - lo_val, floor) for key, lo_val in lo.items()}
    for key in angular:
        scales[key] = float(np.pi)
    noise = config.observation_noise
    if noise is not None:
        sigma_n = DEFAULT_NOISE_SIGMA if noise_sigma is None else noise_sigma
        scales = {(type_name, feat):
                  noise.residual_scale(types_by_name[type_name], feat, scale,
                                       sigma_n)
                  for (type_name, feat), scale in scales.items()}
    return ResidualScaling(angular=frozenset(angular), scales=scales)


def expected_noise_sse(
    trajectories: List[RolloutTrajectory],
    residual_features: Dict[str, List[str]],
    scaling: Optional[ResidualScaling],
    config: Optional[SysIdConfig] = None,
) -> float:
    """The SSE the declared observation noise alone leaves in the fit's
    objective, in expectation.

    Every scored per-step residual ``(pred - obs) / scale`` carries the
    observation's own noise: variance ``(sigma_f / scale_f)^2`` for a
    feature with declared sigma ``sigma_f``. Summed over the residuals
    the objective scores (one per object, in-scope feature and
    rolled-out step, plus the settled-endpoint summary residuals at
    their weight) this is the SSE a perfect model is still left with.
    Under the interval belief the grid sweep's data-equivalence
    tolerance measures its relative fraction against the SSE in EXCESS
    of this floor (:func:`grid_seed.flat_tolerance`), so a noisy
    dataset's floor no longer widens the flat set. The rollout's own
    error from starting at a noisy initial observation is not counted
    (it is the errors-in-variables term the fit-side filter removes),
    which keeps the estimate conservative. 0 without a declared channel
    or without residual scaling (the raw objective has no per-feature
    scale to express the noise in).
    """
    config = config or SysIdConfig.from_cfg()
    noise = config.observation_noise
    if noise is None or scaling is None:
        return 0.0
    weight = max(config.summary_weight, 0.0)
    total = 0.0
    for states, _actions in trajectories:
        n_steps = len(states) - 1
        if n_steps <= 0:
            continue
        for obj in states[0]:
            for feat in residual_features.get(obj.type.name, []):
                sigma = noise.feature_sigma(obj.type, feat)
                if sigma <= 0.0:
                    continue
                scale = scaling.scales.get((obj.type.name, feat), 1.0)
                total += (sigma / scale)**2 * (n_steps + weight)
    return float(total)


def split_at_rest_points(
    trajectory: RolloutTrajectory,
    residual_features: Dict[str, List[str]],
    motion_tol: Optional[float] = None,
    min_rest_steps: Optional[int] = None,
    margin: Optional[int] = None,
    config: Optional[SysIdConfig] = None,
) -> List[RolloutTrajectory]:
    """Split a recording into independently-scored rest-anchored segments.

    Multiple shooting for chaotic contact dynamics: free-running an
    entire manipulation trajectory lets small early divergence compound
    across phases, which shifts the SSE minimum away from the true
    parameters (replay-divergence bias - observed pulling the fitted
    friction both above and below truth on different runs). Cutting at
    rest points (every scored feature quiescent for at least
    ``min_rest_steps`` consecutive steps) bounds the compounding
    horizon while keeping each segment's zero-velocity re-anchor exact:
    the observed anchor state genuinely is at rest, so no momentum is
    discarded (the failure mode that rules out per-step teacher
    forcing, see the :mod:`.physical_sysid` module docstring).

    Each segment spans from the at-rest state just before its first
    motion to ``margin`` steps after its last motion (so it is still
    scored on settling at the right pose). Fully-static stretches
    between segments are dropped: they carry no parameter signal but
    would re-score accumulated divergence every step. A trajectory with
    no scored motion at all yields ``[]``.

    Trimming/consistency then operate per segment, so one chaotic phase
    (e.g. a scraping robot push) no longer discards the clean cascade
    recorded seconds later in the same episode.

    Under the fit-side filter (:func:`_filter_params`) motion is
    detected sigma-relatively and each segment's first frame is the
    mean of its preceding rest window (:func:`_rest_mean_state`), the
    errors-in-variables correction for the rollout's initial condition.
    """
    config = config or SysIdConfig.from_cfg()
    if motion_tol is None:
        motion_tol = config.settle_tol
    if min_rest_steps is None:
        min_rest_steps = config.segment_min_rest_steps
    if margin is None:
        margin = config.settle_margin
    states, actions = trajectory
    num_steps = len(actions)
    noise, window, sigmas = _filter_params(config)
    active = _active_step_indices(states, actions, residual_features,
                                  motion_tol, noise, window, sigmas)
    if not active:
        return []
    runs: List[Tuple[int, int]] = []
    run_start = active[0]
    prev = active[0]
    for i in active[1:]:
        if i - prev > min_rest_steps:
            runs.append((run_start, prev))
            run_start = i
        prev = i
    runs.append((run_start, prev))
    segments: List[RolloutTrajectory] = []
    for a, b in runs:
        end = min(num_steps, b + 1 + margin)
        seg_states = states[a:end + 1]
        if noise is not None and window > 1:
            seg_states = [_rest_mean_state(states, a, window, noise)
                          ] + seg_states[1:]
        segments.append((seg_states, actions[a:end]))
    return segments
