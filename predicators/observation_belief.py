"""The execution-time belief over the observed frame.

See docs/uncertainty/design.md, sections 3.3 and 3.6.

Under the observation-noise channel every frame is one noisy draw. An
object that has not moved for a while has been observed several times,
and the mean of those readings is a better estimate of where it is than
the latest frame: its standard error is the declared sigma over the
square root of the frames averaged. This module is that smoothed frame,
the spread that comes with it, draws from it (where the objects may
really be, given the observation) and the fraction of draws on which an
atom holds, which is what the observation text, the invocation monitor
and the belief probe read.

Two ways of judging how long an object has rested:

* :func:`change_point_belief`, the state factor ``q(x_base,t | H_t)`` of
  the joint belief (paper Section 3.2 and Appendix B.3): each object's
  noisy features are modelled as constant since the object last moved,
  under a prior uniform over each feature's recorded range and the
  declared Gaussian noise. The unknown run length ``r`` gets posterior
  weights by Bayesian online change-point detection (Adams and MacKay,
  2007) with a constant per-step motion hazard, over at most ``window``
  frames. Given ``r``, a feature is Gaussian around the mean of the last
  ``r`` frames with spread ``sigma / sqrt(r)``, up to truncation at the
  range; the belief is the mixture over ``r``, and draws pick ``r``
  first.
* :func:`smooth_frames`, the legacy rest test: an object is at rest over
  a window of frames when the means of the window's two halves differ by
  less than a few standard errors of that difference on every noisy
  feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
# SciPy exports ndtr and ndtri through compiled ufuncs.
# pylint: disable=no-name-in-module
from scipy.special import ndtr, ndtri
from scipy.stats import chi2

# pylint: enable=no-name-in-module
from predicators import utils
from predicators.observation_noise import EXACT_TYPES, ObservationNoise
from predicators.structs import GroundAtom, Object, Predicate, State

# A feature's recorded (lo, hi) range.
Range = Tuple[float, float]


@dataclass(frozen=True)
class BeliefFrame:
    """The smoothed frame with its spread."""

    frame: State
    # (object name, feature) -> standard error of the smoothed value,
    # for every noisy feature; exact features are absent.
    spread: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # object name -> frames averaged (1 = the latest frame only); under
    # the change-point belief, the most likely run length.
    frames_used: Dict[str, int] = field(default_factory=dict)
    # The change-point belief's mixture (empty under the legacy test):
    # object name -> posterior weights of run lengths 1..R, and
    # (object name, feature) -> the mean of the last r frames for each r.
    run_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    run_means: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    # (object name, feature) -> declared noise sigma, the (lo, hi) range
    # draws are truncated to, and which features are angles.
    sigmas: Dict[Tuple[str, str], float] = field(default_factory=dict)
    ranges: Dict[Tuple[str, str], Range] = field(default_factory=dict)
    angular: Set[Tuple[str, str]] = field(default_factory=set)
    # object name -> the (mean, standard error) of its linear velocity
    # along x, y and z, for the bodies seen moving at the frame
    # (:func:`with_motion`); a body absent here is at rest.
    velocities: Dict[str, Tuple[np.ndarray,
                                np.ndarray]] = field(default_factory=dict)

    def max_spread(self) -> float:
        """The largest standard error in the frame, 0 when exact."""
        return max(self.spread.values(), default=0.0)

    def object_line(self, obj: Object) -> str:
        """One object's belief in words: each noisy feature as ``value +-
        spread`` and the frames averaged."""
        parts = []
        for feat in obj.type.feature_names:
            key = (obj.name, feat)
            if key in self.spread:
                parts.append(f"{feat} {self.frame.get(obj, feat):.4f}+-"
                             f"{self.spread[key]:.4f}")
        n = self.frames_used.get(obj.name, 1)
        return (f"{obj.name}: " + ", ".join(parts) +
                f" ({n} frame{'s' if n != 1 else ''})")


def _wrap(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _mean(values: np.ndarray, angular: bool) -> float:
    if angular:
        return float(
            np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))
    return float(values.mean())


def _prefix_sums(values: np.ndarray, angular: bool) -> np.ndarray:
    """Prefix sums of ``values`` (of their sines and cosines for an angle),
    with a leading zero row: ``sums[i]`` totals the first ``i`` values."""
    parts = ([np.sin(values), np.cos(values)] if angular else [values])
    return np.vstack(
        [np.concatenate([[0.0], np.cumsum(part)]) for part in parts]).T


def _run_means(sums: np.ndarray, end: int, runs: np.ndarray) -> np.ndarray:
    """For each run length ``r`` in ``runs``, :func:`_mean` of the ``r`` values
    before index ``end``, from :func:`_prefix_sums` output."""
    totals = sums[end] - sums[end - runs]
    if sums.shape[1] == 2:
        return np.arctan2(totals[:, 0], totals[:, 1])
    return totals[:, 0] / runs


def _rest_window(series: Dict[str, np.ndarray], sigmas_by_feat: Dict[str,
                                                                     float],
                 angular: Set[str], window: int, threshold: float) -> int:
    """The longest suffix (up to ``window`` frames) over which every noisy
    feature in ``series`` reads at rest: the means of the suffix's two halves
    differ by less than ``threshold`` standard errors of that difference."""
    total = len(next(iter(series.values())))
    best = 1
    for n in range(2, min(window, total) + 1):
        half = n // 2
        at_rest = True
        for feat, values in series.items():
            recent = values[-n:]
            first, second = recent[:n - half], recent[n - half:]
            delta = _mean(second, feat in angular) - _mean(
                first, feat in angular)
            if feat in angular:
                delta = float(_wrap(np.array([delta]))[0])
            err = sigmas_by_feat[feat] * np.sqrt(1.0 / len(first) +
                                                 1.0 / len(second))
            if abs(delta) > threshold * err:
                at_rest = False
                break
        if not at_rest:
            break
        best = n
    return best


def smooth_frames(frames: Sequence[State], noise: ObservationNoise,
                  window: int, sigmas: float) -> BeliefFrame:
    """The belief over the latest of ``frames`` (the observed views of one
    episode, oldest first).

    Per object, the belief is the mean of its noisy features over its
    rest window (:func:`_rest_window`), circular for angular features,
    with spread ``sigma / sqrt(n)``; objects of exact types and exact
    features keep the latest frame's values. ``window`` caps the frames
    averaged and ``sigmas`` is the rest test's threshold.
    """
    assert frames, "a belief needs at least one frame"
    latest = frames[-1]
    belief = latest.copy()
    spread: Dict[Tuple[str, str], float] = {}
    frames_used: Dict[str, int] = {}
    recent = list(frames[-max(int(window), 1):])
    by_name = [{o.name: o for o in s} for s in recent]
    for obj in latest:
        if obj.type.name in EXACT_TYPES:
            continue
        noisy = {
            feat: noise.feature_sigma(obj.type, feat)
            for feat in obj.type.feature_names
            if noise.feature_sigma(obj.type, feat) > 0.0
        }
        if not noisy:
            continue
        # Only the frames that carry this object, contiguous at the end.
        present = [m for m in by_name if obj.name in m]
        held = [s for s, m in zip(recent, by_name)
                if obj.name in m][-len(present):]
        series = {
            feat: np.array([
                s.get(m[obj.name], feat)
                for s, m in zip(held, [{o.name: o
                                        for o in s} for s in held])
            ])
            for feat in noisy
        }
        angular = set(getattr(obj.type, "angular_features", ()))
        n = _rest_window(series, noisy, angular, window, sigmas)
        frames_used[obj.name] = n
        for feat, sigma in noisy.items():
            values = series[feat][-n:]
            belief.set(obj, feat, _mean(values, feat in angular))
            spread[(obj.name, feat)] = float(sigma / np.sqrt(n))
    return BeliefFrame(frame=belief, spread=spread, frames_used=frames_used)


# The smallest range a feature's uniform prior spans, in noise sigmas: a
# scene whose objects all share one value (every yaw zero, say) must not
# make a jump of a few sigmas look like a move to anywhere.
_MIN_RANGE_SIGMAS = 10.0


def feature_ranges(frames: Sequence[State],
                   noise: ObservationNoise,
                   margin_sigmas: float = 2.0) -> Dict[str, Range]:
    """Each noisy feature's recorded range over ``frames`` and all objects.

    The prior of a feature after a move is uniform over this range: the
    span of the values the feature took in the recorded frames, widened
    by ``margin_sigmas`` noise sigmas on each side and to at least
    ``_MIN_RANGE_SIGMAS`` sigmas. Angles span the full circle.
    """
    values: Dict[str, List[float]] = {}
    sigma_of: Dict[str, float] = {}
    angles: Set[str] = set()
    for state in frames:
        for obj in state:
            if obj.type.name in EXACT_TYPES:
                continue
            angular = set(getattr(obj.type, "angular_features", ()))
            for feat in obj.type.feature_names:
                sigma = noise.feature_sigma(obj.type, feat)
                if sigma <= 0.0:
                    continue
                sigma_of[feat] = max(sigma_of.get(feat, 0.0), sigma)
                if feat in angular or feat in ("rot", "roll", "pitch", "yaw",
                                               "tilt", "wrist"):
                    angles.add(feat)
                values.setdefault(feat, []).append(float(state.get(obj, feat)))
    ranges: Dict[str, Range] = {}
    for feat, vals in values.items():
        if feat in angles:
            ranges[feat] = (-np.pi, np.pi)
            continue
        sigma = sigma_of[feat]
        lo = min(vals) - margin_sigmas * sigma
        hi = max(vals) + margin_sigmas * sigma
        short = _MIN_RANGE_SIGMAS * sigma - (hi - lo)
        if short > 0.0:
            lo, hi = lo - short / 2.0, hi + short / 2.0
        ranges[feat] = (lo, hi)
    return ranges


def _run_length_weights(series: Dict[str, np.ndarray],
                        sigmas_by_feat: Dict[str, float], angular: Set[str],
                        log_range: float, hazard: float) -> np.ndarray:
    """Posterior weights of run lengths 1..T after the last of T frames.

    Bayesian online change-point detection with a still-object model: a
    run's features are constant, observed under Gaussian noise, so the
    next frame's predictive under a run of ``r`` frames is Gaussian
    around their mean with variance ``sigma^2 (1 + 1/r)``; the first
    frame of a new run has density ``exp(-log_range)``, the uniform
    prior over the features' ranges. The first frame starts a run.
    """
    total = len(next(iter(series.values())))
    log_w = np.array([0.0])  # run length 1 after the first frame
    log_h, log_stay = np.log(hazard), np.log1p(-hazard)
    # Prefix sums make each run's mean one subtraction, so the pass is
    # quadratic in the frames rather than cubic.
    sums = {
        feat: _prefix_sums(values, feat in angular)
        for feat, values in series.items()
    }
    for t in range(1, total):
        # Predictive log density of frame t under each current run length.
        runs = np.arange(1, log_w.size + 1)
        pred = np.zeros(runs.size)
        for feat, values in series.items():
            sigma = sigmas_by_feat[feat]
            delta = values[t] - _run_means(sums[feat], t, runs)
            if feat in angular:
                delta = _wrap(delta)
            var = sigma**2 * (1.0 + 1.0 / runs)
            pred += -0.5 * (delta**2 / var + np.log(2.0 * np.pi * var))
        grow = log_w + log_stay + pred
        change = np.logaddexp.reduce(log_w + log_h) - log_range
        log_w = np.concatenate([[change], grow])
        log_w -= np.logaddexp.reduce(log_w)
    weights = np.exp(log_w)
    return weights / weights.sum()


def change_point_belief(
        frames: Sequence[State],
        noise: ObservationNoise,
        window: int,
        hazard: float,
        ranges: Optional[Dict[str, Range]] = None) -> BeliefFrame:
    """The state factor over the latest of ``frames`` (one episode's observed
    views, oldest first).

    Per object, jointly over its noisy features, the run-length weights
    of :func:`_run_length_weights` over at most ``window`` frames; each
    noisy feature's value is the mixture mean over run lengths and its
    spread the mixture's standard deviation. Objects of exact types and
    exact features keep the latest frame's values. ``ranges`` maps a
    feature to its recorded range (default: :func:`feature_ranges` over
    ``frames``).
    """
    assert frames, "a belief needs at least one frame"
    assert 0.0 < hazard < 1.0, hazard
    ranges = ranges if ranges is not None else feature_ranges(frames, noise)
    latest = frames[-1]
    belief = latest.copy()
    out = BeliefFrame(frame=belief)
    recent = list(frames[-max(int(window), 1):])
    by_name = [{o.name: o for o in s} for s in recent]
    for obj in latest:
        if obj.type.name in EXACT_TYPES:
            continue
        noisy = {
            feat: noise.feature_sigma(obj.type, feat)
            for feat in obj.type.feature_names
            if noise.feature_sigma(obj.type, feat) > 0.0
        }
        if not noisy:
            continue
        held = [(s, m[obj.name]) for s, m in zip(recent, by_name)
                if obj.name in m]
        # Only the frames that carry this object, contiguous at the end.
        series = {
            feat: np.array([float(s.get(o, feat)) for s, o in held])
            for feat in noisy
        }
        angular = set(getattr(obj.type, "angular_features", ())) | {
            f
            for f in noisy if ranges.get(f, (0.0, 0.0)) == (-np.pi, np.pi)
        }
        log_range = 0.0
        for feat in noisy:
            lo, hi = ranges.get(feat, (-np.inf, np.inf))
            width = hi - lo
            if not np.isfinite(width) or width <= 0.0:
                width = _MIN_RANGE_SIGMAS * noisy[feat]
            log_range += np.log(width)
        weights = _run_length_weights(series, noisy, angular, log_range,
                                      hazard)
        total = weights.size
        out.run_weights[obj.name] = weights
        out.frames_used[obj.name] = int(np.argmax(weights)) + 1
        runs = np.arange(1, total + 1)
        for feat, sigma in noisy.items():
            values = series[feat]
            # The mean of each run ending at the latest frame.
            means = _run_means(_prefix_sums(values, feat in angular), total,
                               runs)
            key = (obj.name, feat)
            out.run_means[key] = means
            out.sigmas[key] = float(sigma)
            out.ranges[key] = ranges.get(feat, (-np.inf, np.inf))
            if feat in angular:
                out.angular.add(key)
                mean = float(
                    np.arctan2(np.dot(weights, np.sin(means)),
                               np.dot(weights, np.cos(means))))
                offsets = _wrap(means - mean)
            else:
                mean = float(np.dot(weights, means))
                offsets = means - mean
            var = float(np.dot(weights, sigma**2 / runs + offsets**2))
            belief.set(obj, feat, mean)
            out.spread[key] = float(np.sqrt(max(var, 0.0)))
    return out


# A body moves at the frame when a line through its recent readings has a
# slope that a body at rest would show by chance in at most _MOTION_P of
# its frames, over the _SPANS spans with_motion tries, and its readings
# stray from that line no more than the declared noise allows at
# _LINE_FIT_P (see with_motion). Bodies rest far more often than they move,
# and a velocity staged on a resting body moves it through every rollout,
# so the level is strict.
_MOTION_P = 1e-4
_SPANS = 2
_LINE_FIT_P = 0.01
_POSITION = ("x", "y", "z")


def _line_velocity(
        values: np.ndarray,
        sigmas: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per column of ``values`` (one reading per step, oldest first), the slope
    of the least-squares line and its standard error under the declared
    ``sigmas``, in units per step; None when the slopes, taken together, are as
    far from zero as a body at rest puts them more often than ``_MOTION_P /
    _SPANS`` (a chi-square test over the columns), or the readings stray from
    the line by more than the noise explains (a chi-square test at
    ``_LINE_FIT_P``): the body is at rest, or moves in a way no constant
    velocity describes, such as bobbing or stopping."""
    count = len(values)
    if count < 3:
        return None
    centred = np.arange(count) - (count - 1) / 2.0
    spread = float(np.sum(centred**2))
    slope = centred @ (values - values.mean(axis=0)) / spread
    error = sigmas / np.sqrt(spread)
    distance = float(np.sum((slope / error)**2))
    if chi2.sf(distance, values.shape[1]) > _MOTION_P / _SPANS:
        return None
    residuals = values - values.mean(axis=0) - np.outer(centred, slope)
    misfit = float(np.sum((residuals / sigmas)**2))
    if chi2.sf(misfit, values.shape[1] * (count - 2)) <= _LINE_FIT_P:
        return None
    return slope, error


def with_motion(belief: BeliefFrame, frames: Sequence[State],
                seconds_per_step: float) -> BeliefFrame:
    """``belief`` with the linear velocity of each body it sees moving.

    The frames record positions only, so a rollout staged from them
    starts every body at rest, and a body rolling or sliding at the
    frame stops dead in the rehearsal. A body moves when a line through
    its readings of x, y and z explains them within the declared noise
    and has a slope that a body at rest would show in fewer than
    ``_MOTION_P`` of its frames (:func:`_line_velocity`): first over all
    of ``frames`` (oldest first, one per step), else over their latest
    half, for a body whose motion changed earlier. Its velocity is that
    slope, with the slope's standard error. The frame carries the
    estimate in ``simulator_state["body_velocities"]`` (no angular
    velocity, since no reading shows it, so a rolling body starts
    sliding), and :func:`belief_draw` draws around it. Every other body
    stays at rest, including one that bobs or has just stopped, whose
    velocity at the frame no line through its readings gives.
    """
    if not frames or seconds_per_step <= 0.0 or not isinstance(
            getattr(belief.frame, "simulator_state", None), dict):
        return belief
    velocities: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for obj in belief.frame:
        keys = [(obj.name, f) for f in _POSITION]
        if not all(k in belief.sigmas for k in keys):
            continue
        series = []
        for frame in frames:
            match = [o for o in frame if o.name == obj.name]
            if match:
                series.append(
                    [float(frame.get(match[0], f)) for f in _POSITION])
        if len(series) < 3:
            continue
        values = np.asarray(series)
        sigmas = np.array([belief.sigmas[k] for k in keys])
        # The _SPANS spans: every frame, then the latest half.
        for span in (len(values), len(values) // 2):
            fit = _line_velocity(values[-span:], sigmas)
            if fit is not None:
                velocity, error = fit
                velocities[obj.name] = (velocity / seconds_per_step,
                                        error / seconds_per_step)
                break
    if not velocities:
        return belief
    frame = belief.frame.copy()
    sim_state = frame.simulator_state
    assert isinstance(sim_state, dict)
    recorded = dict(sim_state.get("body_velocities") or {})
    for name, (mean, _) in velocities.items():
        recorded[name] = (tuple(float(v) for v in mean), (0.0, 0.0, 0.0))
    sim_state["body_velocities"] = recorded
    return replace(belief, frame=frame, velocities=velocities)


def _truncated_normal(mean: float, sd: float, lo: float, hi: float,
                      rng: np.random.Generator) -> float:
    """One draw of ``N(mean, sd^2)`` restricted to ``[lo, hi]``, by inverse
    CDF: the posterior of a feature whose prior is uniform on its range."""
    a, b = ndtr((lo - mean) / sd), ndtr((hi - mean) / sd)
    if b - a < 1e-12:
        # No mass the float CDF resolves: the nearer end of the range.
        return float(np.clip(mean, lo, hi))
    return float(np.clip(mean + sd * ndtri(rng.uniform(a, b)), lo, hi))


def _draw_velocities(belief: BeliefFrame, draw: State,
                     rng: np.random.Generator) -> None:
    """Set each moving body's velocity in ``draw`` to a draw around its
    estimate (see :func:`with_motion`)."""
    sim_state = getattr(draw, "simulator_state", None)
    if not belief.velocities or not isinstance(sim_state, dict):
        return
    recorded = dict(sim_state.get("body_velocities") or {})
    for name, (mean, sd) in belief.velocities.items():
        value = mean + sd * rng.standard_normal(mean.shape)
        recorded[name] = (tuple(float(v) for v in value), (0.0, 0.0, 0.0))
    sim_state["body_velocities"] = recorded


def belief_draw(belief: BeliefFrame, rng: np.random.Generator) -> State:
    """One draw of where the objects may really be.

    Under the change-point belief, each object's run length is drawn
    from its weights, then each noisy feature from a Gaussian around
    that run's mean with spread ``sigma / sqrt(r)``, truncated to the
    feature's range (angles wrapped). Under the legacy belief, the
    smoothed frame with each noisy feature jittered by its spread.
    """
    draw = belief.frame.copy()
    if belief.run_weights:
        for obj in draw:
            weights = belief.run_weights.get(obj.name)
            if weights is None:
                continue
            r = int(rng.choice(weights.size, p=weights)) + 1
            for feat in obj.type.feature_names:
                key = (obj.name, feat)
                means = belief.run_means.get(key)
                if means is None:
                    continue
                mean = float(means[r - 1])
                sd = belief.sigmas[key] / np.sqrt(r)
                if key in belief.angular:
                    value = float(_wrap(np.array([rng.normal(mean, sd)]))[0])
                else:
                    lo, hi = belief.ranges.get(key, (-np.inf, np.inf))
                    value = _truncated_normal(mean, sd, lo, hi, rng)
                draw.set(obj, feat, value)
        _draw_velocities(belief, draw, rng)
        return draw
    for obj in draw:
        for feat in obj.type.feature_names:
            sd = belief.spread.get((obj.name, feat), 0.0)
            if sd > 0.0:
                draw.set(obj, feat, draw.get(obj, feat) + rng.normal(0.0, sd))
    _draw_velocities(belief, draw, rng)
    return draw


def atom_fractions(belief: BeliefFrame, predicates: Set[Predicate],
                   num_draws: int,
                   rng: np.random.Generator) -> Dict[GroundAtom, float]:
    """The fraction of ``num_draws`` belief draws on which each atom of
    ``predicates`` holds; atoms that hold on no draw are absent."""
    counts: Dict[GroundAtom, int] = {}
    draws = max(int(num_draws), 1)
    for _ in range(draws):
        for atom in utils.abstract(belief_draw(belief, rng), predicates):
            counts[atom] = counts.get(atom, 0) + 1
    return {atom: n / draws for atom, n in counts.items()}


def likely_atoms(fractions: Dict[GroundAtom, float]) -> Set[GroundAtom]:
    """The atoms more likely to hold than not under the belief."""
    return {atom for atom, f in fractions.items() if f >= 0.5}


def describe_fractions(fractions: Dict[GroundAtom, float],
                       atoms: Sequence[GroundAtom]) -> str:
    """``atoms`` with their belief fractions, in words."""
    return ", ".join(f"{atom} {fractions.get(atom, 0.0):.2f}"
                     for atom in sorted(atoms, key=str))


def uncertain_atoms(fractions: Dict[GroundAtom, float]) -> List[GroundAtom]:
    """The atoms the belief is unsure about (fraction strictly between 0 and
    1), most uncertain first."""
    return sorted((a for a, f in fractions.items() if 0.0 < f < 1.0),
                  key=lambda a: (abs(fractions[a] - 0.5), str(a)))
