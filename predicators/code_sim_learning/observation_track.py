"""An external per-frame pose track, and the topple onsets read off it.

Step 3 of the open-loop plan. The markerless pipeline post-processes an
episode's recording into a dense pose track; this module turns that track into
the quantity the friction fit is scored on -- when each domino started to fall,
relative to the first.

**Why intervals rather than poses.** Two independent measurements say so. The
extrinsics put absolute position 25-38 mm off in the base frame while measured
*displacements* are accurate to ~1 mm, so scoring absolute poses would score the
calibration. And an interval between two onsets is invariant to however the
track's clock is offset from the robot's, which is what lets alignment be an
event rather than a clock reading.

**Why onsets are confirmed and then backdated.** Two measured effects produce
spurious falls, and a naive threshold fires on both:

* Occlusion. When the gripper covers the domino it is about to push, the fit is
  a confident fit to the visible sliver -- one take reported a 29 deg topple 15
  frames before the real one, at a residual well inside the gate.
* Drift. A domino that was never touched had its measured fall angle wander from
  4.4 deg to 12.9 deg, across the 10 deg the twin calls toppled.

Neither reaches an unambiguous fall. So a fall is only *believed* once the angle
passes ``confirm_deg`` (far above anything either effect produced) and stays
there; the onset is then found by walking backwards from that confirmation. The
shape is deliberately the same as ``cascade_certificate._topple_onset``, which
requires a topple to persist rather than merely occur.

A rate or jump gate is NOT used, and that is a measured decision rather than an
oversight: during a real cascade the other dominoes translate 22-36 mm/frame,
overlapping the 48-67 mm of the artifacts, so speed cannot separate them. The
pipeline gates on visibility upstream instead, which shows up here as a domino
simply missing from some frames.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from predicators.structs import State

# How often to look for a track the pipeline is still writing. Seconds, and
# deliberately coarse: the thing being waited on takes minutes.
_POLL_S = 2.0

# Frames whose fit the tracker itself doubted are dropped before anything is
# read off them; ``suspect`` is how a masklet touching the crop edge is
# reported, and it has no other symptom (the fit stays small because the box
# explains the points it was given).
_SUSPECT_KEY = "suspect"


@dataclass(frozen=True)
class ObservationTrack:
    """Per-frame domino angles from an external pose track.

    ``angles_deg`` is keyed by the track's own domino id, each a list of
    ``(seconds, fall_deg)`` in frame order, with frames the pipeline
    dropped simply absent. Times are seconds from the track's first
    frame -- an origin of its own, deliberately: nothing here is
    comparable to the robot's clock, and nothing needs to be, because
    every quantity scored is a difference between two of these.
    """
    angles_deg: Dict[int, List[Tuple[float, float]]]
    n_frames: int
    source: str
    # Each domino's first observed (x, y) in the base frame, which is what
    # the ids are matched to the twin's objects by.
    first_xy: Dict[int, Tuple[float, float]]

    @property
    def duration_s(self) -> float:
        """Seconds from the first sample to the last, over all dominoes."""
        times = [t for series in self.angles_deg.values() for t, _ in series]
        return (max(times) - min(times)) if times else 0.0


def load_track(path: str, fallback_fps: float = 60.0) -> ObservationTrack:
    """Read the markerless pipeline's trajectory JSON.

    ``timestamp_ns`` is per frame and may be absent (the recorder only
    attaches it when the take's ``meta.json`` carried one). When any is
    missing the whole track falls back to ``fallback_fps`` and says so:
    a track that silently mixed real timestamps with assumed ones would
    put a fabricated interval into the fit.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    frames = raw.get("frames") or []
    stamps = [fr.get("timestamp_ns") for fr in frames]
    have_stamps = bool(stamps) and all(s is not None for s in stamps)
    if not have_stamps:
        logging.warning(
            "observation track %s has no per-frame timestamps; assuming "
            "%.1f fps. Intervals are only as good as that assumption.", path,
            fallback_fps)
    t0 = float(stamps[0]) if have_stamps else 0.0
    angles: Dict[int, List[Tuple[float, float]]] = {}
    first_xy: Dict[int, Tuple[float, float]] = {}
    for i, frame in enumerate(frames):
        if have_stamps:
            seconds = (float(frame["timestamp_ns"]) - t0) / 1e9
        else:
            seconds = float(frame.get("index", i)) / fallback_fps
        for record in frame.get("dominoes") or []:
            if record.get(_SUSPECT_KEY):
                continue
            if "fall_deg" not in record:
                continue
            obj_id = int(record["id"])
            angles.setdefault(obj_id, []).append(
                (seconds, float(record["fall_deg"])))
            centre = record.get("center_base_m")
            if obj_id not in first_xy and centre:
                first_xy[obj_id] = (float(centre[0]), float(centre[1]))
    return ObservationTrack(angles_deg=angles,
                            n_frames=len(frames),
                            source=path,
                            first_xy=first_xy)


def load_tracks(path: str,
                fallback_fps: float = 60.0,
                wait_s: float = 0.0) -> List[ObservationTrack]:
    """Every track a fit should score against, in episode order.

    ``path`` is either one track or a run manifest. The manifest is the
    normal case once episodes are recorded automatically: it names each
    episode's track and says whether the take it came from was usable.

    **Waiting is correct here, and only here.** The manifest is written
    as each take closes, so it is always present -- but the pipeline
    that fills in its tracks runs about 3x the length of a take, and the
    online loop fits as soon as an episode ends. Returning early would
    mean falling back to per-step scoring, which under open-loop scores
    the twin against itself: the very defect this path exists to avoid,
    reached silently. Open-loop execution exists to stop the ROBOT
    waiting on perception; the learner has no such excuse, because the
    track is its data. So a fit blocks up to ``wait_s`` for the tracks
    the manifest promised.

    Entries are skipped, with a reason logged, when the take lost a
    camera -- an unusable take yields a well-formed track of the wrong
    thing -- or when the wait expires.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if "frames" in raw:  # a single track, named directly
        return [load_track(path, fallback_fps)]
    episodes = raw.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path} is neither a track nor a run manifest")
    wanted = [e for e in episodes if e.get("usable", True)]
    for entry in episodes:
        if not entry.get("usable", True):
            logging.warning(
                "episode %s is skipped: its take reported a camera error, so "
                "its track would describe the wrong thing",
                entry.get("episode", "?"))
    _await_tracks(wanted, wait_s)
    tracks: List[ObservationTrack] = []
    for entry in wanted:
        track_path = entry.get("track")
        if not track_path or not os.path.exists(track_path):
            logging.warning(
                "episode %s still has no track at %s after waiting %.0fs; it "
                "is skipped, so this fit sees less evidence than the run "
                "recorded. Raise code_sim_learning_track_wait_s if "
                "post-processing is simply slow.", entry.get("episode", "?"),
                track_path, wait_s)
            continue
        tracks.append(load_track(track_path, fallback_fps))
    return tracks


def _await_tracks(entries: List[Dict[str, Any]], wait_s: float) -> None:
    """Block until every promised track exists, or ``wait_s`` expires.

    Polls rather than joining the pipeline processes: the fit runs in a
    different process from the recorder that launched them, so the file
    appearing is the only signal available to it.
    """
    if wait_s <= 0:
        return
    missing = [
        e for e in entries
        if e.get("track") and not os.path.exists(str(e["track"]))
    ]
    if not missing:
        return
    logging.info(
        "waiting up to %.0fs for %d episode track(s) to finish "
        "post-processing; each run takes about 3x the length of its take",
        wait_s, len(missing))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if all(os.path.exists(str(e["track"])) for e in missing):
            logging.info("all episode tracks are ready")
            return
        time.sleep(_POLL_S)


def topple_onsets(series_by_id: Mapping[int, Sequence[Tuple[float, float]]],
                  confirm_deg: float = 45.0,
                  onset_deg: float = 5.0,
                  min_persist: int = 3) -> Dict[int, float]:
    """Time each domino started falling, by confirm-then-backdate.

    A fall is believed only once the angle rises ``confirm_deg`` above
    that domino's OWN first reading and holds there for ``min_persist``
    consecutive samples. Measuring against its own baseline rather than
    an absolute matters because a domino can be placed slightly
    off-vertical, and the per-camera calibration offset is not shared.

    The onset is then the first sample of the unbroken run that reaches
    the confirmation -- walking back to where the angle was last within
    ``onset_deg`` of baseline. Dominoes that never confirm are absent
    from the result rather than given a sentinel: "did not fall" is a
    different statement from "fell late", and the caller decides what
    to do with it.
    """
    onsets: Dict[int, float] = {}
    for obj_id, series in series_by_id.items():
        samples = list(series)
        if len(samples) < min_persist:
            continue
        baseline = samples[0][1]
        confirmed_at: Optional[int] = None
        run = 0
        for i, (_t, angle) in enumerate(samples):
            if angle - baseline >= confirm_deg:
                run += 1
                if run >= min_persist:
                    confirmed_at = i - run + 1
                    break
            else:
                run = 0
        if confirmed_at is None:
            continue
        # Backdate: the onset is where this fall left the upright band, not
        # where it became unambiguous.
        onset_i = confirmed_at
        while onset_i > 0 and samples[onset_i - 1][1] - baseline > onset_deg:
            onset_i -= 1
        onsets[obj_id] = samples[onset_i][0]
    return onsets


def propagation_intervals(onsets: Dict[int, float]) -> Dict[int, float]:
    """Each onset relative to the earliest one, in seconds.

    The first domino to fall defines the origin and contributes a zero
    that carries no information, so it is dropped: what friction sets is
    how fast the cascade travels *down the row*, not when the push
    happened.
    """
    if len(onsets) < 2:
        return {}
    first = min(onsets.values())
    return {obj_id: t - first for obj_id, t in onsets.items() if t > first}


def sim_topple_series(
        states: Sequence[State], step_s: float,
        name_to_id: Dict[str, int]) -> Dict[int, List[Tuple[float, float]]]:
    """The same ``(seconds, fall_deg)`` series, read off simulated states.

    The twin carries ``roll`` in radians and the track reports a fall
    angle in degrees, so the two are put in the same units here rather
    than at the comparison, where a unit slip would look like a physics
    disagreement.
    """
    series: Dict[int, List[Tuple[float, float]]] = {}
    for t, state in enumerate(states):
        seconds = t * step_s
        for obj in state:
            obj_id = name_to_id.get(obj.name)
            if obj_id is None:
                continue
            if "roll" not in obj.type.feature_names:
                continue
            fall_deg = abs(math.degrees(float(state.get(obj, "roll"))))
            series.setdefault(obj_id, []).append((seconds, fall_deg))
    return series


def interval_residuals(sim_intervals: Dict[int, float],
                       obs_intervals: Dict[int, float],
                       missing_penalty_s: float) -> List[float]:
    """``sim - obs`` per domino, in seconds, over the union of both.

    A domino present in one side only is the strongest evidence the
    track carries -- the cascade completed under one friction and
    stalled under the other -- so it is scored at
    ``missing_penalty_s`` rather than skipped. Skipping it would make a
    friction that stops the cascade early look *better* than one that
    reproduces it, because it would simply have fewer terms.
    """
    residuals: List[float] = []
    for obj_id in sorted(set(sim_intervals) | set(obs_intervals)):
        sim_t = sim_intervals.get(obj_id)
        obs_t = obs_intervals.get(obj_id)
        if sim_t is None or obs_t is None:
            residuals.append(missing_penalty_s)
            continue
        residuals.append(sim_t - obs_t)
    return residuals


def track_name_to_id(state: State, prefix: str) -> Dict[str, int]:
    """Map ``<prefix><n>`` object names onto the track's integer ids by name.

    The fallback, used only when the track carries no positions. The
    track numbers dominoes by the order their initialization boxes were
    drawn, which is not the env's numbering unless something forced it
    to be -- so this is a guess, and
    :func:`match_ids_by_position` is the real answer.
    """
    mapping: Dict[str, int] = {}
    for obj in state:
        name = obj.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.isdigit():
            mapping[name] = int(suffix)
    return mapping


def match_ids_by_position(state: State,
                          first_xy: Dict[int, Tuple[float, float]],
                          prefix: str,
                          tol_m: float = 0.04) -> Dict[str, int]:
    """Match track ids to object names by where each domino actually is.

    The track's ids are the order its initialization boxes were drawn,
    which nothing guarantees matches the env's numbering. Rather than
    assume, or force the box order and hope it survives a re-draw, the
    two sets of positions are matched: at the track's first frame every
    domino is standing where the twin says it is.

    **The calibration offset is found by vote, not by centroid.**
    Absolute base-frame position is 25-38 mm off, but that error is a
    constant per camera while dominoes sit ~100 mm apart, so removing it
    leaves the ~1 mm regime measured displacements live in. Cancelling
    each set's centroid would do it -- except that a single bad
    detection, or one domino that genuinely moved, drags the centroid
    and then NO pair matches. So every twin-point/track-point pairing is
    treated as a candidate offset, and the one that brings the most
    dominoes within ``tol_m`` wins. With a handful of dominoes that is a
    few hundred comparisons, and it tolerates outliers by construction.

    A pair further apart than ``tol_m`` is refused rather than forced: a
    wrong assignment would silently attribute one domino's topple to
    another, which is worse than scoring nothing.
    """
    twin: Dict[str, Tuple[float, float]] = {}
    for obj in state:
        if not obj.name.startswith(prefix):
            continue
        if "x" not in obj.type.feature_names:
            continue
        twin[obj.name] = (float(state.get(obj,
                                          "x")), float(state.get(obj, "y")))
    if not twin or not first_xy:
        return {}
    best: Dict[str, int] = {}
    best_score = (0, 0.0)
    for tx, ty in twin.values():
        for ox, oy in first_xy.values():
            mapping = _match_at_offset(twin, first_xy, (tx - ox, ty - oy),
                                       tol_m)
            if not mapping:
                continue
            total = _total_distance(twin, first_xy, mapping,
                                    (tx - ox, ty - oy))
            # Most dominoes matched wins; ties go to the tighter fit, so the
            # result does not depend on dict ordering.
            score = (len(mapping), -total)
            if score > best_score:
                best_score, best = score, mapping
    mapping = best
    unmatched = sorted(set(twin) - set(mapping))
    if unmatched:
        logging.warning(
            "could not match %d domino(s) %s to a track id within %.0f mm of "
            "where the twin says they are; they contribute no interval. "
            "Either the scene moved between the capture and the episode, or "
            "the track is of a different arrangement.", len(unmatched),
            unmatched, tol_m * 1000)
    return mapping


def _match_at_offset(twin: Dict[str, Tuple[float, float]],
                     track: Dict[int, Tuple[float, float]],
                     offset: Tuple[float,
                                   float], tol_m: float) -> Dict[str, int]:
    """Greedy nearest-neighbour once the offset is assumed, closest first."""
    pairs = []
    for name, (tx, ty) in twin.items():
        for obj_id, (ox, oy) in track.items():
            dist = math.hypot(tx - (ox + offset[0]), ty - (oy + offset[1]))
            if dist <= tol_m:
                pairs.append((dist, name, obj_id))
    pairs.sort()
    mapping: Dict[str, int] = {}
    used: set = set()
    for _dist, name, obj_id in pairs:
        if name in mapping or obj_id in used:
            continue
        mapping[name] = obj_id
        used.add(obj_id)
    return mapping


def _total_distance(twin: Dict[str, Tuple[float, float]],
                    track: Dict[int, Tuple[float, float]],
                    mapping: Dict[str, int], offset: Tuple[float,
                                                           float]) -> float:
    """How tightly a candidate offset explains the pairs it matched."""
    total = 0.0
    for name, obj_id in mapping.items():
        tx, ty = twin[name]
        ox, oy = track[obj_id]
        total += math.hypot(tx - (ox + offset[0]), ty - (oy + offset[1]))
    return total
