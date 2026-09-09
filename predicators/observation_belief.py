"""The execution-time belief over the observed frame (docs/continual-
uncertainty.md, sections 3.3 and 3.6).

Under the observation-noise channel every frame is one noisy draw. An
object that has not moved for a while has been observed several times,
and the mean of those readings is a better estimate of where it is than
the latest frame: its standard error is the declared sigma over the
square root of the frames averaged. This module is that smoothed frame,
the spread that comes with it, draws from it (where the objects may
really be, given the observation) and the fraction of draws on which an
atom holds, which is what the observation text, the invocation monitor
and the belief probe read.

Motion is judged the way the fit-side filter judges it: an object is at
rest over a window of frames when the means of the window's two halves
differ by less than a few standard errors of that difference on every
noisy feature. The window grows one frame per step while the object
rests and collapses to the latest frame when it moves, so a moving
object is never smoothed across its motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.observation_noise import EXACT_TYPES, ObservationNoise
from predicators.structs import GroundAtom, Object, Predicate, State


@dataclass(frozen=True)
class BeliefFrame:
    """The smoothed frame with its spread."""

    frame: State
    # (object name, feature) -> standard error of the smoothed value,
    # for every noisy feature; exact features are absent.
    spread: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # object name -> frames averaged (1 = the latest frame only).
    frames_used: Dict[str, int] = field(default_factory=dict)

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


def belief_draw(belief: BeliefFrame, rng: np.random.Generator) -> State:
    """One draw of where the objects may really be: the smoothed frame with
    each noisy feature jittered by its spread."""
    draw = belief.frame.copy()
    for obj in draw:
        for feat in obj.type.feature_names:
            sd = belief.spread.get((obj.name, feat), 0.0)
            if sd > 0.0:
                draw.set(obj, feat, draw.get(obj, feat) + rng.normal(0.0, sd))
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
