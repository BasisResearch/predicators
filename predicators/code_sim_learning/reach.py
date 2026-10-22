"""How far each scored body goes in the recordings, against the fitted model's
replay of them.

The fit scores where a body is at each step, and it can trade how far a
body goes for a closer match elsewhere. Many failures turn on how far a
body goes: a balloon that reaches the ceiling bursts, a ball that rolls
past an edge falls. Round fixes_r18's Balloons seed 0 fitted a box drag
that quieted its balloons' sideways swing, and its replays stopped the
box 0.04 to 0.07 m short of four of the five recorded peaks; its
rehearsal of the next release showed no burst, and the release burst a
balloon. The check replays each segment the fit kept at the fitted
values and compares, for each scored feature that moves, the highest and
the lowest value the replay reaches with the recording's.
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.rollout_env import RolloutTrajectory
from predicators.code_sim_learning.rollout_objective import step_residuals
from predicators.code_sim_learning.trajectory_prep import ResidualScaling

# A replay's extreme is short of a noisy recording's when the gap exceeds
# this many sigmas of the feature's declared noise. The replay's extreme is
# read through the same noise first, so for a replay that matches the
# recording the gap is noise with a root mean square of 0.5 to 0.7 sigma
# (measured on 400 recordings each of a body at rest, a ramp and a peak).
_GAP_SIGMAS = 2.0
# One such gap in one segment is reported only past this many sigmas; two
# or more segments short past _GAP_SIGMAS are reported at any size. A fit
# compares hundreds of extremes, so a lone gap must be far past what the
# noise reaches by chance.
_LONE_GAP_SIGMAS = 4.0
# A noisy feature moves in a segment when its recorded range exceeds this
# many sigmas, so a body at rest is never read as moving.
_MOVE_SIGMAS = 6.0
# An exact feature (no declared noise) is short when the gap exceeds this
# fraction of its range in the segment.
_EXACT_FRACTION = 0.05
# Noise draws a replay's extremes are averaged over.
_DRAWS = 16
# How many features the rehearsal reminder names.
_NAMED = 3


@dataclasses.dataclass(frozen=True)
class ReachGap:
    """A scored feature whose replayed extreme stops short of the recorded one.

    ``side`` is ``"highest"`` or ``"lowest"``. ``short`` counts the
    segments where the replay stops short by more than the noise allows,
    of the ``moving`` segments in which the feature moves. The largest
    gap is in ``segment`` (an index into the segments checked), where
    the replay reaches ``model`` and the recording ``recorded``.
    """
    type_name: str
    feature: str
    side: str
    short: int
    moving: int
    segment: int
    model: float
    recorded: float

    @property
    def gap(self) -> float:
        """How far the replay stops short, in the feature's units."""
        return abs(self.recorded - self.model)


def _replay_extremes(model: np.ndarray, sigma: float, recorded: np.ndarray,
                     key: str) -> Tuple[float, float]:
    """The replay's highest and lowest value, read through ``sigma`` of noise
    as the recording was.

    A noisy recording's extremes lie outside the true ones (by about 2
    sigma over 50 frames at rest), so a replay that matches it would
    look short without the same noise. The draws are fixed by the
    recording, so one fit's report always reads the same.
    """
    if sigma <= 0.0:
        return float(model.max()), float(model.min())
    seed = zlib.crc32(
        np.ascontiguousarray(recorded, dtype=np.float64).tobytes() +
        key.encode("utf-8"))
    rng = np.random.default_rng(seed)
    noisy = model[None, :] + sigma * rng.standard_normal((_DRAWS, model.size))
    return float(noisy.max(axis=1).mean()), float(noisy.min(axis=1).mean())


def reach_gaps(base_env: Any,
               segments: Sequence[RolloutTrajectory],
               params: Dict[str, float],
               residual_features: Dict[str, List[str]],
               physical_names: Sequence[str],
               rules: Sequence[Any] = (),
               latent_init: Any = None,
               scaling: Optional[ResidualScaling] = None,
               config: Optional[SysIdConfig] = None) -> List[ReachGap]:
    """The scored features whose replays at ``params`` stop short of the
    highest or lowest value the recordings reach, largest gap (in sigmas or in
    ranges) first.

    Angular features are skipped: where their wrap falls would decide
    their extremes. So are segments scored by intervals, which have no
    per-step replay to compare.
    """
    config = config or SysIdConfig.from_cfg()
    noise = config.observation_noise
    worst: Dict[Tuple[str, str, str], Tuple[float, int, float, float]] = {}
    short: Dict[Tuple[str, str, str], int] = {}
    moving: Dict[Tuple[str, str], int] = {}
    for index, segment in enumerate(segments):
        values = step_residuals(base_env,
                                segment,
                                params,
                                residual_features,
                                physical_names,
                                rules,
                                latent_init,
                                scaling,
                                config,
                                episode_count=len(segments))
        if not values:
            continue
        types = {obj.name: obj.type for obj in segment[0][0]}
        series: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
        for value in values:
            model_list, recorded_list = series.setdefault(
                (value.obj_name, value.feature), ([], []))
            model_list.append(value.model)
            recorded_list.append(value.recorded)
        for (obj_name, feat), (model_list, recorded_list) in series.items():
            obj_type = types.get(obj_name)
            if obj_type is None or feat in getattr(obj_type,
                                                   "angular_features", ()):
                continue
            model = np.asarray(model_list, dtype=float)
            recorded = np.asarray(recorded_list, dtype=float)
            span = float(recorded.max() - recorded.min())
            sigma = (noise.feature_sigma(obj_type, feat)
                     if noise is not None else 0.0)
            if sigma > 0.0:
                if span <= _MOVE_SIGMAS * sigma:
                    continue
                unit = sigma
                tol = _GAP_SIGMAS * sigma
            else:
                if span <= 0.0:
                    continue
                unit = span
                tol = _EXACT_FRACTION * span
            moving[(obj_type.name, feat)] = moving.get(
                (obj_type.name, feat), 0) + 1
            high, low = _replay_extremes(model, sigma, recorded,
                                         f"{obj_name}.{feat}")
            for side, model_end, recorded_end in (
                ("highest", high, float(recorded.max())),
                ("lowest", low, float(recorded.min())),
            ):
                gap = (recorded_end -
                       model_end if side == "highest" else model_end -
                       recorded_end)
                if gap <= tol:
                    continue
                key = (obj_type.name, feat, side)
                short[key] = short.get(key, 0) + 1
                if gap / unit > worst.get(key, (0.0, 0, 0.0, 0.0))[0]:
                    worst[key] = (gap / unit, index, model_end, recorded_end)
    gaps = []
    for key, (size, index, model_end, recorded_end) in worst.items():
        noisy = noise is not None and noise.feature_sigma(
            next(o.type for s in segments
                 for o in s[0][0] if o.type.name == key[0]), key[1]) > 0.0
        if noisy and short[key] < 2 and size <= _LONE_GAP_SIGMAS:
            continue
        gaps.append((size,
                     ReachGap(key[0], key[1], key[2], short[key],
                              moving[(key[0], key[1])], index, model_end,
                              recorded_end)))
    return [gap for _, gap in sorted(gaps, key=lambda item: -item[0])]


def describe_reach(
    gaps: Sequence[ReachGap], where: Sequence[str] = ()) -> List[str]:
    """The fit report's lines on ``gaps``; ``where`` names each checked
    segment's recording and steps, when known."""
    if not gaps:
        return []
    lines = [
        "Reach check: each kept segment replayed at the fitted values, with "
        "the replay's extremes read through the declared noise, against the "
        "recording's:"
    ]
    for gap in gaps:
        place = (f" ({where[gap.segment]})"
                 if gap.segment < len(where) and where[gap.segment] else "")
        lines.append(
            f"  {gap.type_name} {gap.feature}: the {gap.side} value stops "
            f"short of the recording's in {gap.short} of {gap.moving} "
            f"segments that move it, by up to {gap.gap:.4g} (segment "
            f"{gap.segment}{place}: {gap.model:.4g} against "
            f"{gap.recorded:.4g}).")
    lines.append(
        "A failure that turns on how far a body goes (a height, an edge, a "
        "level) can happen in the world when a rehearsal of this model "
        "clears it by less than these gaps. If a parameter that sets how "
        "far it goes is also fitted to other motion, give each motion its "
        "own parameter.")
    return lines


def reach_reminder(gaps: Sequence[ReachGap]) -> Optional[str]:
    """One line for a rehearsal on the model these gaps were found on; None
    without gaps."""
    if not gaps:
        return None
    parts = [
        f"{gap.type_name} {gap.feature} {gap.side} by up to {gap.gap:.4g} "
        f"({gap.short} of {gap.moving} segments)" for gap in gaps[:_NAMED]
    ]
    more = (f" and {len(gaps) - _NAMED} more feature(s)"
            if len(gaps) > _NAMED else "")
    return ("The fitted model's replays stop short of how far bodies went "
            "in the recordings: " + "; ".join(parts) + more +
            " (fit report, reach check). A rehearsal that clears a failure "
            "by less than these gaps can fail in the world.")
