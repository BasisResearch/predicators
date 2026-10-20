"""Where a model's replay leaves the recorded motion the fit drops.

The fit drops a recorded motion segment that no parameter values it
tries replay within the noise (``fit_params_rollout_trimmed`` in
:mod:`~predicators.code_sim_learning.physical_sysid`), and the
segment's best RMS says only how far off the replay stays. To
repair the model, the agent needs to know where: which feature carries
the error, from which step, and what the model reads against what was
recorded. Each dropped segment is replayed at its own best values from
the trimming sweep, the closest the model came to it, and the feature
with the largest share of its squared error is located.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.rollout_env import RolloutTrajectory
from predicators.code_sim_learning.rollout_objective import StepResidual, \
    step_residuals
from predicators.code_sim_learning.trajectory_prep import ResidualScaling

logger = logging.getLogger(__name__)

# How many segments each feature's line names; the rest are counted.
_EXAMPLES_PER_FEATURE = 3


@dataclasses.dataclass(frozen=True)
class Departure:
    """The feature a dropped segment's replay misses most, and where.

    ``segment`` indexes the fit's motion segments and ``num_steps`` is
    the segment's length in actions. ``share`` is the part of the
    segment's squared error this feature carries. Steps count the
    segment's actions from 1: ``first_step`` is the first step at which
    the feature is at least half as far off as at ``worst_step``, its
    farthest, and each step gives the model's value and the recording's.
    """
    segment: int
    num_steps: int
    obj_name: str
    feature: str
    share: float
    first_step: int
    first_model: float
    first_recorded: float
    worst_step: int
    worst_model: float
    worst_recorded: float


def find_departure(segment: int,
                   residuals: Sequence[StepResidual]) -> Optional[Departure]:
    """The departure in one segment's step residuals, or None when the replay
    matches its recording exactly or scores nothing."""
    by_feature: Dict[Tuple[str, str], List[StepResidual]] = {}
    for value in residuals:
        by_feature.setdefault((value.obj_name, value.feature),
                              []).append(value)
    totals = {
        key: sum(v.residual**2 for v in values)
        for key, values in by_feature.items()
    }
    grand = sum(totals.values())
    if grand <= 0.0:
        return None
    key = max(totals, key=lambda k: totals[k])
    series = by_feature[key]
    worst = max(series, key=lambda v: abs(v.residual))
    first = next(v for v in series
                 if abs(v.residual) >= 0.5 * abs(worst.residual))
    return Departure(segment=segment,
                     num_steps=max(v.step for v in residuals) + 1,
                     obj_name=key[0],
                     feature=key[1],
                     share=totals[key] / grand,
                     first_step=first.step + 1,
                     first_model=first.model,
                     first_recorded=first.recorded,
                     worst_step=worst.step + 1,
                     worst_model=worst.model,
                     worst_recorded=worst.recorded)


def segment_departures(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    segments: Sequence[int],
    params: Sequence[Dict[str, float]],
    residual_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
    config: Optional[SysIdConfig] = None,
) -> List[Departure]:
    """The departure of each of ``segments`` (indices into ``trajectories``),
    replayed at its own ``params``.

    Empty under interval scoring, which compares a replay with its
    recording only at events. A segment whose replay raises is left out:
    the note is a diagnosis, and the trimming sweep has scored the
    segment already.
    """
    found: List[Departure] = []
    for index, point in zip(segments, params):
        try:
            residuals = step_residuals(base_env,
                                       trajectories[index],
                                       point,
                                       residual_features,
                                       physical_names,
                                       rules,
                                       latent_init,
                                       scaling,
                                       config,
                                       episode_count=len(trajectories))
        except Exception:  # pylint: disable=broad-except
            logger.warning("Departure note: replaying segment %d failed.",
                           index,
                           exc_info=True)
            continue
        if residuals is None:
            return []
        departure = find_departure(index, residuals)
        if departure is not None:
            found.append(departure)
    return found


def describe_departures(
    departures: Sequence[Departure],
    num_dropped: int,
    origins: Optional[Sequence[Optional[Tuple[int,
                                              int]]]] = None) -> List[str]:
    """Lines naming, per feature, the dropped segments whose error it carries
    most, with where their replays leave the recordings.

    ``origins`` gives, per fit segment, the recorded episode it was cut
    from (0-based) and the episode step it starts at; with it, steps are
    the episode's.
    """
    if not departures:
        return []
    groups: Dict[Tuple[str, str], List[Departure]] = {}
    for departure in departures:
        groups.setdefault((departure.obj_name, departure.feature),
                          []).append(departure)
    lines = [
        "Where your model leaves each dropped segment, replayed at the "
        "segment's own best values from that sweep: the feature that "
        "carries most of the segment's error, your model's value against "
        "the recording's from the first step it is half as far off as at "
        "its farthest, and at its farthest."
    ]
    for (obj_name, feature), members in sorted(groups.items(),
                                               key=lambda kv: -len(kv[1])):
        parts = [
            _describe_one(d, None if origins is None else origins[d.segment])
            for d in members[:_EXAMPLES_PER_FEATURE]
        ]
        more = len(members) - _EXAMPLES_PER_FEATURE
        tail = f"; and {more} more" if more > 0 else ""
        lines.append(f"  {obj_name}.{feature}, in {len(members)} of the "
                     f"{num_dropped}: " + "; ".join(parts) + tail + ".")
    return lines


def _describe_one(departure: Departure, origin: Optional[Tuple[int,
                                                               int]]) -> str:
    """One segment's departure in words."""
    if origin is None:
        where = f"segment {departure.segment} ({departure.num_steps} steps)"
        first, worst = departure.first_step, departure.worst_step
        step_word = "its step"
    else:
        episode, start = origin
        first = start + departure.first_step
        worst = start + departure.worst_step
        where = (f"segment {departure.segment} (episode {episode + 1}, "
                 f"steps {start + 1}-{start + departure.num_steps})")
        step_word = "step"
    text = (f"{where}: {departure.first_model:.4g} against "
            f"{departure.first_recorded:.4g} from {step_word} {first}")
    if departure.worst_step != departure.first_step:
        text += (f", {departure.worst_model:.4g} against "
                 f"{departure.worst_recorded:.4g} at {step_word} {worst}")
    if departure.share < 0.5:
        text += (f" ({departure.share:.0%} of its error; other features "
                 "carry the rest)")
    return text
