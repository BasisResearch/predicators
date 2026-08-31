"""Tests for scoring against an external markerless pose track.

The onset detector is the part that matters: it is what the friction fit
is ultimately measuring, and two spurious-fall mechanisms have been
measured on real takes that a naive threshold fires on. Both appear here
as traces.
"""
import io
import json
import math
import os

import pytest

from predicators import utils
from predicators.code_sim_learning.observation_track import \
    interval_residuals, load_track, propagation_intervals, sim_topple_series, \
    topple_onsets, track_name_to_id
from predicators.structs import Object, State, Type

_DOMINO = Type("domino", ["x", "y", "z", "yaw", "roll", "r", "g", "b"])


@pytest.fixture(autouse=True)
def _restore_config():
    """Put CFG back after every test in this module.

    Several tests here turn ``score_observed_only`` on, which changes
    what the rollout objective scores. A test that does not reset the
    config itself would inherit that and fail for a reason that has
    nothing to do with it -- which is exactly what happened to
    test_orchestrator when this file was first added.
    """
    yield
    utils.reset_config({})
    # Tracks are cached per path for the life of the process, so a test that
    # reuses a path would otherwise see the previous one's tracks.
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        reset_track_cache
    reset_track_cache()


def _series(*angles, dt=1 / 60.0):
    """A (seconds, fall_deg) series at a fixed frame rate."""
    return [(i * dt, float(a)) for i, a in enumerate(angles)]


def _fall(start=0.2, steps=20):
    """A real topple: rises monotonically well past any artifact."""
    return [start + (90.0 - start) * i / (steps - 1) for i in range(steps)]


# -- onset detection ---------------------------------------------------------
def test_a_real_topple_is_detected_and_backdated():
    """The onset is where the fall left the upright band, not where it became
    unambiguous -- otherwise every interval inherits the confirmation lag."""
    upright = [0.2] * 10
    onsets = topple_onsets({0: _series(*(upright + _fall()))})

    assert 0 in onsets
    # Backdated into the rise, not sitting at the 45 deg confirmation.
    assert 10 / 60.0 <= onsets[0] <= 13 / 60.0


def test_orientation_drift_does_not_manufacture_a_topple():
    """A domino that was never touched drifted 4.4 -> 12.9 deg on a real take,
    crossing the 10 deg the twin calls toppled.

    It must not register.
    """
    drift = [4.4 + (12.9 - 4.4) * i / 99 for i in range(100)]

    assert not topple_onsets({0: _series(*drift)})


def test_an_occlusion_spike_does_not_manufacture_a_topple():
    """The measured case: 29 deg for a frame or two while the gripper covers
    the domino it is about to push, 15 frames before the real fall, at a fit
    residual well inside the gate."""
    trace = [0.2] * 5 + [6.8, 29.2, 6.0] + [0.2] * 5

    assert not topple_onsets({0: _series(*trace)})


def test_a_spike_before_a_real_fall_does_not_move_the_onset():
    """Both together, which is the real shape of the pushed domino's trace:

    the artifact must not backdate the true onset 15 frames early.
    """
    trace = [0.2] * 5 + [6.8, 29.2, 6.0] + [0.2] * 5 + _fall()
    onsets = topple_onsets({0: _series(*trace)})

    assert 0 in onsets
    assert onsets[0] >= 12 / 60.0, "the onset was dragged back to the spike"


def test_one_sample_past_the_threshold_is_not_a_fall():
    """Persistence, mirroring cascade_certificate._TOPPLE_MIN_STEPS: a single
    reading carries no information."""
    trace = [0.2] * 5 + [80.0] + [0.2] * 5

    assert not topple_onsets({0: _series(*trace)})


def test_a_domino_that_never_falls_is_absent_not_late():
    """"Did not fall" is a different statement from "fell late", and the caller
    decides what to do with it."""
    assert not topple_onsets({0: _series(*([0.3] * 50))})


def test_each_domino_is_measured_against_its_own_baseline():
    """A domino can be placed slightly off-vertical, and the per-camera
    calibration offset is not shared, so an absolute threshold would treat a
    crooked placement as a head start."""
    crooked = [8.0] * 5 + [8.0 + a for a in _fall()]
    upright = [0.0] * 5 + _fall()

    onsets = topple_onsets({0: _series(*crooked), 1: _series(*upright)})

    assert set(onsets) == {0, 1}
    assert abs(onsets[0] - onsets[1]) < 1e-9


def test_missing_frames_do_not_break_the_detector():
    """The visibility gate drops a record rather than flagging it, so a domino
    is simply absent from some frames."""
    samples = [(i / 60.0, a) for i, a in enumerate([0.2] * 5 + _fall())
               if i % 3 != 0]

    assert 0 in topple_onsets({0: samples})


# -- intervals ---------------------------------------------------------------
def test_intervals_are_relative_to_the_first_onset():
    """What friction sets is how fast the cascade travels down the row, so
    every onset is measured from the earliest.

    The origin is KEPT, at zero. It carries no information while both
    streams agree on which domino fell first, and everything when they
    do not -- see
    test_a_tie_at_the_earliest_onset_keeps_both_dominoes.
    """
    intervals = propagation_intervals({0: 10.0, 1: 10.2, 2: 10.5})

    assert intervals[0] == pytest_approx(0.0)
    assert intervals[1] == pytest_approx(0.2)
    assert intervals[2] == pytest_approx(0.5)


def test_a_tie_at_the_earliest_onset_keeps_both_dominoes():
    """The sim samples one state per action, 83.3 ms, while the track runs at
    60 fps -- so the pushed domino and the one beside it land on the SAME sim
    step and are 5x resolvable on camera.

    Dropping every entry at the earliest time would delete BOTH, and the
    track keeps the second, which then has no counterpart and draws the
    full missing-cascade penalty -- reporting that the twin's chain never
    reached a domino it had in fact laid flat. Measured on
    run_20260819_104757: domino_3 and domino_4 both at 0.0833 s in the
    twin, 350 ms apart on camera.
    """
    tied = propagation_intervals({0: 0.0833, 1: 0.0833, 2: 0.3333})

    assert set(tied) == {0, 1, 2}, "a tie must not delete both dominoes"
    assert tied[0] == pytest_approx(0.0)
    assert tied[1] == pytest_approx(0.0)
    assert tied[2] == pytest_approx(0.25)


def test_streams_that_disagree_on_the_origin_still_compare():
    """Neither stream can see the other from inside propagation_intervals, so
    dropping exactly one would need an agreement they cannot reach.

    Keeping every entry needs no agreement: where the two disagree about
    which domino fell first, both sides still carry both dominoes and
    the comparison yields real differences instead of two penalties.
    """
    # The sim has 0 first; the track resolves 1 as first instead.
    sim = propagation_intervals({0: 1.00, 1: 1.00, 2: 1.30})
    obs = propagation_intervals({0: 1.35, 1: 1.00, 2: 1.30})

    assert set(sim) == set(obs), \
        "every domino must have a counterpart, whoever the origin is"
    residuals = interval_residuals(sim, obs, missing_penalty_s=999.0)
    assert 999.0 not in residuals, "no term may fall back to the penalty"


def test_intervals_are_invariant_to_a_clock_offset():
    """This is why alignment can be an event rather than a clock reading: a
    constant offset between the track's clock and the robot's cancels."""
    base = {0: 10.0, 1: 10.2, 2: 10.5}
    shifted = {k: v + 1234.5 for k, v in base.items()}

    got = propagation_intervals(shifted)
    want = propagation_intervals(base)
    assert set(got) == set(want)
    # To float precision, not bit-exactly: the offset is subtracted off at a
    # different magnitude, which is the whole point of the test.
    for key, value in want.items():
        assert got[key] == pytest_approx(value, abs=1e-6)


def test_one_onset_yields_its_origin_not_nothing():
    """A cascade of one still says WHICH domino fell, and the other stream
    needs a counterpart for it.

    Collapsing to {} while the origin is kept for longer cascades is an
    inconsistency that costs a residual: one stream returns nothing
    while the other keeps every entry including its origin, so a domino
    BOTH streams watched fall has no counterpart and draws the missing-
    cascade penalty.
    """
    assert propagation_intervals({0: 1.0}) == {0: 0.0}
    assert propagation_intervals({}) == {}


def test_a_cascade_that_stalls_on_one_side_is_penalised_not_skipped():
    """Skipping it would make a friction that stops the cascade early look
    BETTER than one that reproduces it, by having fewer terms."""
    residuals = interval_residuals({1: 0.2}, {1: 0.2, 2: 0.5}, 3.0)

    assert sorted(residuals) == [0.0, 3.0]


# -- sim side ----------------------------------------------------------------
def test_sim_series_converts_roll_to_degrees():
    """The twin carries roll in radians and the track reports degrees; the two
    are put in the same units once, here."""
    obj = Object("domino_0", _DOMINO)
    states = [
        State({obj: [0, 0, 0, 0, 0.0, 0, 0, 0]}),
        State({obj: [0, 0, 0, 0, 1.5707963, 0, 0, 0]}),
    ]

    series = sim_topple_series(states, 0.0833, {"domino_0": 0})

    assert series[0][0][1] == pytest_approx(0.0)
    assert series[0][1][1] == pytest_approx(90.0, abs=1e-3)
    assert series[0][1][0] == pytest_approx(0.0833)


def _domino_state(positions, roll=0.0):
    """A state with dominoes at the given (x, y), optionally toppling."""
    return State({
        Object(name, _DOMINO): [x, y, 0, 0, roll, 0, 0, 0]
        for name, (x, y) in positions.items()
    })


def test_ids_are_matched_by_where_the_dominoes_actually_are():
    """The track's ids are box-drawing order, which nothing guarantees matches
    the env's numbering -- so they are matched, not assumed.

    Here the boxes were drawn in reverse.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        match_ids_by_position
    state = _domino_state({
        "domino_0": (0.60, 1.30),
        "domino_1": (0.70, 1.30),
        "domino_2": (0.80, 1.30),
    })
    drawn_backwards = {0: (0.80, 1.30), 1: (0.70, 1.30), 2: (0.60, 1.30)}

    mapping = match_ids_by_position(state, drawn_backwards, "domino_")

    assert mapping == {"domino_0": 2, "domino_1": 1, "domino_2": 0}


def test_matching_survives_the_calibration_offset():
    """Absolute base-frame position is 25-38 mm off, but that error is a
    constant per camera and dominoes sit ~100 mm apart, so cancelling each
    set's centroid leaves the ~1 mm regime displacements live in."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        match_ids_by_position
    state = _domino_state({
        "domino_0": (0.60, 1.30),
        "domino_1": (0.70, 1.30),
        "domino_2": (0.80, 1.30),
    })
    # Every reading shifted by 38 mm, the worst measured offset.
    offset = {i: (0.60 + 0.10 * i + 0.038, 1.30 + 0.038) for i in range(3)}

    mapping = match_ids_by_position(state, offset, "domino_")

    assert mapping == {"domino_0": 0, "domino_1": 1, "domino_2": 2}


def test_one_bad_detection_does_not_break_the_other_matches(caplog):
    """Why the offset is voted for rather than taken from the centroid: one
    outlier drags a centroid far enough that NO pair matches.

    The dominoes that are where they should be must still match, and the
    one that is not must be refused -- a wrong assignment would
    attribute one domino's topple to another.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        match_ids_by_position
    state = _domino_state({
        "domino_0": (0.60, 1.30),
        "domino_1": (0.70, 1.30),
        "domino_2": (0.80, 1.30),
    })
    # Two are where they should be; the third is half a metre away.
    with_outlier = {0: (0.60, 1.30), 1: (0.70, 1.30), 2: (1.40, 1.30)}

    with caplog.at_level("WARNING"):
        mapping = match_ids_by_position(state, with_outlier, "domino_")

    assert mapping == {"domino_0": 0, "domino_1": 1}
    assert "could not match" in caplog.text


def test_a_track_in_the_robot_base_frame_is_rotated_into_the_env_frame():
    """The pipeline emits ROBOT BASE poses; a twin state is in the env's world
    frame, and for the domino env the two differ by a quarter turn.

    Matching votes over candidate translations, so it absorbs the camera
    calibration offset -- but a rotation is not a translation, and
    without this transform every pair lands hundreds of mm apart.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        ObservationTrack, match_ids_by_position
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _track_in_world_frame
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs.pybullet_domino.real_geometry import \
        DOMINO_WORLD_ROBOT_XY, DOMINO_WORLD_ROBOT_YAW
    utils.reset_config({
        "code_sim_learning_track_object_prefix":
        "domino_",
        "code_sim_learning_track_frame_yaw":
        DOMINO_WORLD_ROBOT_YAW,
        "code_sim_learning_track_frame_xy":
        DOMINO_WORLD_ROBOT_XY,
    })
    # A row along the base frame's +y, as the pipeline reports it.
    base_xy = {0: (0.55, -0.15), 1: (0.55, 0.0), 2: (0.55, 0.15)}
    # The same row after the quarter turn: what the env's state carries.
    world = _domino_state({
        "domino_0": (0.75 + 0.15, 0.72 + 0.55),
        "domino_1": (0.75 - 0.00, 0.72 + 0.55),
        "domino_2": (0.75 - 0.15, 0.72 + 0.55),
    })
    track = ObservationTrack(
        angles_deg={i: _series(*_fall())
                    for i in base_xy},
        n_frames=20,
        source="test",
        first_xy=base_xy)
    config = SysIdConfig.from_cfg()

    # Not "matches nothing": the offset is voted for from the candidate
    # pairings, so whichever pair supplies the winning offset always matches
    # itself. What a rotated frame costs is every OTHER domino.
    raw = match_ids_by_position(world, track.first_xy, "domino_")
    assert len(raw) < 3, \
        "raw base-frame positions cannot match a world-frame state"

    moved = _track_in_world_frame(track, config)
    mapping = match_ids_by_position(world, moved.first_xy, "domino_")

    assert mapping == {"domino_0": 0, "domino_1": 1, "domino_2": 2}
    assert moved.angles_deg == track.angles_deg, \
        "a yaw of the frame leaves every per-domino fall angle alone"


def test_the_frame_transform_is_identity_by_default():
    """An env whose track already shares the twin's frame, and every test that
    builds both sides in one frame, must be untouched."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        ObservationTrack
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _track_in_world_frame
    utils.reset_config({})
    track = ObservationTrack(angles_deg={0: _series(*_fall())},
                             n_frames=20,
                             source="test",
                             first_xy={0: (0.55, -0.15)})

    assert _track_in_world_frame(track, SysIdConfig.from_cfg()) is track


def test_ids_are_matched_once_per_episode_not_per_segment(caplog):
    """Segmentation splits ONE episode into several scored trajectories, and
    only the first of them starts where the track's first frame does.

    This episode picks and places a domino before the push, so by the
    second segment the twin has it 200 mm from where frame 0 saw it --
    five times the matching tolerance. Matching per segment drops that
    domino, and the interval it carries, from every segment after the
    first; matching once per episode keeps it.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        ObservationTrack
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _episode_id_maps
    utils.reset_config({"code_sim_learning_track_object_prefix": "domino_"})
    start = {
        "domino_0": (0.60, 1.30),
        "domino_1": (0.70, 1.30),
        "domino_2": (0.40, 1.00),
    }
    # domino_2 is the one the plan relocates: 300 mm in x, 300 mm in y.
    after_place = dict(start, domino_2=(0.80, 1.30))
    track = ObservationTrack(
        angles_deg={i: _series(*_fall())
                    for i in range(3)},
        n_frames=20,
        source="test",
        first_xy={
            0: after_place["domino_0"],
            1: after_place["domino_1"],
            2: after_place["domino_2"]
        })
    # Two segments of one episode: the second begins after the place and
    # carries the cascade, which is where the anchor comes from. A track of a
    # push-only take shows the PLACED row, not where the episode began.
    toppling = [
        _domino_state(after_place, roll=math.radians(a))
        for a in _fall(steps=20)
    ]
    trajectories = [([_domino_state(start)], []),
                    ([_domino_state(after_place)] + toppling, [])]

    with caplog.at_level("WARNING"):
        maps = _episode_id_maps([track],
                                trajectories,
                                SysIdConfig.from_cfg(),
                                paired=False)

    expected = {"domino_0": 0, "domino_1": 1, "domino_2": 2}
    assert maps == [expected, expected], \
        "every segment of an episode shares ONE mapping, taken from the " \
        "arrangement the cascade actually ran along"
    assert "could not match" not in caplog.text


def test_the_anchor_survives_a_take_that_starts_at_the_push(tmp_path):
    """A recording that starts just before the Push, to save post-processing
    time, has its first frame AFTER the plan rearranged the scene.

    The episode-start anchor cannot work for such a take, and a segment-
    start anchor cannot work for a whole-episode one. The settled
    arrangement immediately before the cascade is identifiable in both
    streams whichever window was recorded, so it is what both sides
    anchor on.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _episode_id_maps
    utils.reset_config({"code_sim_learning_track_object_prefix": "domino_"})
    row = {"domino_0": (0.60, 1.30), "domino_1": (0.60, 1.45)}
    # domino_2 starts well off the row and the plan places it at the end.
    start = dict(row, domino_2=(0.95, 1.00))
    placed = dict(row, domino_2=(0.60, 1.60))

    def _state(positions, roll):
        """One twin state: the given layout at the given fall angle."""
        return State({
            Object(n, _DOMINO): [x, y, 0, 0, roll, 0, 0, 0]
            for n, (x, y) in positions.items()
        })

    # Before the place, after it, then the cascade.
    states = ([_state(start, 0.0)] * 5 + [_state(placed, 0.0)] * 5 +
              [_state(placed, math.radians(a)) for a in _fall(steps=20)])
    # The take begins two frames before the first domino moves: everything
    # it ever sees is the PLACED layout.
    frames = [{
        "index":
        i,
        "timestamp_ns":
        i * 16_666_667,
        "dominoes": [{
            "id": {
                "domino_0": 2,
                "domino_1": 0,
                "domino_2": 1
            }[name],
            "fall_deg": angle,
            "center_base_m": [x, y, 0.0],
        } for name, (x, y) in placed.items()],
    } for i, angle in enumerate([0.2] * 2 + _fall(steps=20))]
    track = load_track(_write_track(tmp_path, frames))
    # One episode, split into two scored segments by the place.
    segments = [(states[:10], []), (states[10:], [])]

    maps = _episode_id_maps([track],
                            segments,
                            SysIdConfig.from_cfg(),
                            paired=False)

    expected = {"domino_0": 2, "domino_1": 0, "domino_2": 1}
    assert maps == [expected, expected], \
        "the placed domino must match, though it is 600 mm from where the " \
        "episode began"


def test_no_cascade_means_no_anchor_rather_than_the_episode_start():
    """The fallback this replaces was not arbitrary, it was reliably WRONG.

    settled_xy_before_cascade used to return states[0] when nothing
    toppled. For a take that starts at the push that is the pre-prologue
    layout, while the track shows the post-prologue one -- so the dominoes
    the arm PLACES get compared against positions they have not occupied
    since before the episode began, and only the ones it never touches
    still match.

    Measured on run_20260819_152448: 2 of 5 matched, 63 times, with the
    unmatched set exactly the three placed dominoes.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        settled_xy_before_cascade
    placed = {
        "domino_0": (0.60, 1.30),
        "domino_1": (0.70, 1.30),
        "domino_2": (0.80, 1.30),
    }
    still = [_domino_state(placed)] * 8
    toppling = still + [
        _domino_state(placed, roll=math.radians(a)) for a in _fall(steps=20)
    ]

    assert settled_xy_before_cascade(still, "domino_") == {}, \
        "no cascade means there is no moment the two streams are known " \
        "to share, so no anchor"
    assert set(settled_xy_before_cascade(toppling, "domino_")) == set(placed)


def test_names_are_not_a_fallback_for_a_track_that_has_positions(
        tmp_path, monkeypatch):
    """track_name_to_id is for a track carrying NO positions.

    Using it when the twin merely could not be anchored is worse than
    scoring nothing: the ids are box-drawing order, and on
    run_20260819_152448 the true mapping was a permutation (domino_3 ->
    id 4, domino_4 -> id 3), so a name match would have attributed each
    domino's topple to another one -- silently, with a full set of
    confident-looking residuals.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        ObservationTrack
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _episode_id_maps
    del monkeypatch, tmp_path
    utils.reset_config({"code_sim_learning_track_object_prefix": "domino_"})
    placed = {"domino_0": (0.60, 1.30), "domino_1": (0.70, 1.30)}
    # Nothing topples, so there is no anchor -- but the track HAS positions.
    trajectories = [([_domino_state(placed)] * 4, [])]
    track = ObservationTrack(angles_deg={
        0: _series(*_fall()),
        1: _series(*_fall())
    },
                             n_frames=20,
                             source="test",
                             first_xy={
                                 0: (0.60, 1.30),
                                 1: (0.70, 1.30)
                             })

    maps = _episode_id_maps([track],
                            trajectories,
                            SysIdConfig.from_cfg(),
                            paired=True)

    assert maps == [{}], \
        "a positioned track must not be matched by name as a consolation"


def test_paired_tracks_still_anchor_on_their_own_episode():
    """One track per trajectory means no segmentation happened, so each
    trajectory is its own episode and anchors on its own initial state -- it
    must NOT be forced onto the first trajectory's layout."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        ObservationTrack
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _episode_id_maps
    utils.reset_config({"code_sim_learning_track_object_prefix": "domino_"})

    def _track(first_xy):
        """A track carrying only the geometry the matching reads."""
        return ObservationTrack(
            angles_deg={i: _series(*_fall())
                        for i in first_xy},
            n_frames=20,
            source="test",
            first_xy=first_xy)

    layout_a = {"domino_0": (0.60, 1.30), "domino_1": (0.70, 1.30)}
    # A second episode, re-laid a long way from the first, with its boxes
    # drawn in the other order.
    layout_b = {"domino_0": (1.60, 2.30), "domino_1": (1.70, 2.30)}
    tracks = [
        _track({
            0: layout_a["domino_0"],
            1: layout_a["domino_1"]
        }),
        _track({
            0: layout_b["domino_1"],
            1: layout_b["domino_0"]
        }),
    ]

    def _episode(layout):
        """One episode: the layout, then its cascade.

        A cascade is required now: with nothing toppling there is no
        moment the two streams are known to share, and
        settled_xy_before_cascade refuses rather than anchoring on a
        state that may be the wrong one.
        """
        return [_domino_state(layout)] + [
            _domino_state(layout, roll=math.radians(a))
            for a in _fall(steps=20)
        ]

    trajectories = [(_episode(layout_a), []), (_episode(layout_b), [])]

    maps = _episode_id_maps(tracks,
                            trajectories,
                            SysIdConfig.from_cfg(),
                            paired=True)

    assert maps == [{
        "domino_0": 0,
        "domino_1": 1
    }, {
        "domino_0": 1,
        "domino_1": 0
    }]


def test_object_names_map_onto_track_ids():
    """The one place the numbering assumption lives."""
    state = State({
        Object("domino_0", _DOMINO): [0] * 8,
        Object("domino_11", _DOMINO): [0] * 8,
        Object("robot", _DOMINO): [0] * 8,
    })

    assert track_name_to_id(state, "domino_") == {
        "domino_0": 0,
        "domino_11": 11,
    }


# -- loading -----------------------------------------------------------------
def _write_track(tmp_path, frames):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({
        "frame": "robot_base",
        "n_frames": len(frames),
        "frames": frames,
    }),
                    encoding="utf-8")
    return str(path)


def _truncate(path, fraction=0.45):
    """Leave the file real, non-empty, and mid-document.

    What a reader sees while the pipeline is still writing: the path
    exists and the bytes so far are genuine, they just stop partway
    through.
    """
    text = io.open(path, encoding="utf-8").read()
    io.open(path, "w",
            encoding="utf-8").write(text[:int(len(text) * fraction)])
    return path


def test_a_half_written_track_is_not_complete(tmp_path):
    """Existence is not completion.

    On run_20260818_092302 the fit logged "all episode tracks are ready"
    and then failed to parse 28 MB of track at char 11,997,567, because
    the path had appeared while the pipeline was still writing.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        track_is_complete
    frames = [{
        "index":
        i,
        "timestamp_ns":
        i,
        "dominoes": [{
            "id": 0,
            "fall_deg": float(i),
            "center_base_m": [0.5, 0.1, 0.0]
        }]
    } for i in range(200)]
    path = _write_track(tmp_path, frames)

    assert track_is_complete(path), "a finished track is complete"

    _truncate(path)

    assert not track_is_complete(path), \
        "a track still being written must not read as ready"
    assert os.path.exists(path), \
        "and it is not absent either -- which is why existence cannot decide"


def test_the_wait_does_not_end_on_a_half_written_track(tmp_path, caplog):
    """The wait must run to its deadline rather than declaring victory on a
    file that is merely present."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import _await_tracks
    frames = [{
        "index":
        0,
        "timestamp_ns":
        0,
        "dominoes": [{
            "id": 0,
            "fall_deg": 1.0,
            "center_base_m": [0.5, 0.1, 0.0]
        }]
    }]
    path = _truncate(_write_track(tmp_path, frames))

    with caplog.at_level("INFO"):
        _await_tracks([{"episode": 1, "track": path}], wait_s=0.1)

    assert "waiting up to" in caplog.text
    assert "all episode tracks are ready" not in caplog.text


def test_one_half_written_track_does_not_discard_the_finished_ones(
        tmp_path, caplog):
    """The caller catches at the granularity of the whole manifest, so a single
    truncated file used to take every other episode's evidence with it."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks

    def _episode(name):
        """One finished track, written under its own name."""
        sub = tmp_path / name
        sub.mkdir()
        frames = [{
            "index":
            i,
            "timestamp_ns":
            i * 1000,
            "dominoes": [{
                "id": 0,
                "fall_deg": float(i),
                "center_base_m": [0.5, 0.1, 0.0]
            }]
        } for i in range(60)]
        return _write_track(sub, frames)

    good, half = _episode("good"), _truncate(_episode("half"))
    manifest = tmp_path / "tracks.json"
    manifest.write_text(json.dumps({
        "episodes": [
            {
                "episode": 1,
                "track": good
            },
            {
                "episode": 2,
                "track": half
            },
        ]
    }),
                        encoding="utf-8")

    with caplog.at_level("WARNING"):
        tracks = load_tracks(str(manifest), wait_s=0.0)

    assert len(tracks) == 1, "the finished episode survives its neighbour"
    assert tracks[0].source == good
    assert "still being written" in caplog.text


def test_loading_reads_timestamps_and_angles(tmp_path):
    """The schema the pipeline emits, unchanged."""
    frames = [{
        "index": i,
        "timestamp_ns": 1_000_000_000 + i * 16_666_667,
        "dominoes": [{
            "id": 0,
            "fall_deg": float(i)
        }],
    } for i in range(3)]

    track = load_track(_write_track(tmp_path, frames))

    assert track.n_frames == 3
    assert [a for _t, a in track.angles_deg[0]] == [0.0, 1.0, 2.0]
    assert track.angles_deg[0][0][0] == pytest_approx(0.0)
    assert track.angles_deg[0][2][0] == pytest_approx(0.0333, abs=1e-3)


def test_suspect_records_are_dropped(tmp_path):
    """A masklet the tracker doubted has no other symptom: the fit stays small
    because the box explains the points it was given."""
    frames = [{
        "index":
        0,
        "timestamp_ns":
        0,
        "dominoes": [{
            "id": 0,
            "fall_deg": 1.0
        }, {
            "id": 1,
            "fall_deg": 80.0,
            "suspect": True
        }],
    }]

    track = load_track(_write_track(tmp_path, frames))

    assert set(track.angles_deg) == {0}


def test_a_track_without_timestamps_falls_back_to_a_frame_rate(tmp_path):
    """Better than mixing real stamps with assumed ones, which would put a
    fabricated interval into the fit."""
    frames = [{
        "index": i,
        "timestamp_ns": None,
        "dominoes": [{
            "id": 0,
            "fall_deg": float(i)
        }]
    } for i in range(3)]

    track = load_track(_write_track(tmp_path, frames), fallback_fps=60.0)

    assert track.angles_deg[0][2][0] == pytest_approx(2 / 60.0)


# -- the flag ----------------------------------------------------------------
def test_scoring_falls_back_loudly_without_a_track(caplog):
    """Flag on with no track: one WARNING and per-step scoring.

    Scoring zero residuals would make every theta equally good and
    return the prior centre with a confident-looking identifiability
    report.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _load_scored_track
    utils.reset_config({
        "code_sim_learning_rollout_score_observed_only": True,
        "code_sim_learning_rollout_track_path": "",
    })

    with caplog.at_level("WARNING"):
        assert _load_scored_track(SysIdConfig.from_cfg()) is None
    assert "no track path" in caplog.text


def test_scoring_falls_back_when_the_track_is_unreadable(tmp_path, caplog):
    """A missing file is a fallback, not a crash mid-sweep."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _load_scored_track
    utils.reset_config({
        "code_sim_learning_rollout_score_observed_only":
        True,
        "code_sim_learning_rollout_track_path":
        str(tmp_path / "absent.json"),
    })

    with caplog.at_level("WARNING"):
        assert _load_scored_track(SysIdConfig.from_cfg()) is None
    assert "could not read" in caplog.text


# -- the scored scope (3.1) --------------------------------------------------
def test_scope_keeps_everything_that_moves_by_default():
    """The fidelity report's own semantics, unchanged: an empty scope_types
    must leave this exactly as it was."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.synthesis import moving_feature_scope
    utils.reset_config({"code_sim_learning_rollout_scope_types": []})
    obj = Object("domino_0", _DOMINO)
    robot = Object("robby", Type("robot", ["x"]))
    states = [
        State({
            obj: [0, 0, 0, 0, 0, 0.0, 0, 0],
            robot: [0.0]
        }),
        State({
            obj: [1, 0, 0, 0, 0, 1.0, 0, 0],
            robot: [1.0]
        }),
    ]

    scope = moving_feature_scope([(states, [])])

    assert "robot" in scope, "the arm is in scope by default"
    assert "r" in scope["domino"], "so is a colour channel"


def test_scope_types_drops_the_arm_and_the_nonkinematic_features():
    """What the friction experiment sets.

    The arm is commanded, so it reproduces at every candidate friction
    and can only dilute; a colour channel does not move at all.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.synthesis import moving_feature_scope
    utils.reset_config({"code_sim_learning_rollout_scope_types": ["domino"]})
    obj = Object("domino_0", _DOMINO)
    robot = Object("robby", Type("robot", ["x"]))
    states = [
        State({
            obj: [0, 0, 0, 0, 0, 0.0, 0, 0],
            robot: [0.0]
        }),
        State({
            obj: [1, 0, 0, 0, 0, 1.0, 0, 0],
            robot: [1.0]
        }),
    ]

    scope = moving_feature_scope([(states, [])])

    assert set(scope) == {"domino"}
    assert "r" not in scope["domino"]
    assert "x" in scope["domino"]


# -- the run manifest --------------------------------------------------------
def _one_frame_track(tmp_path, name):
    """A minimal but well-formed track on disk."""
    path = tmp_path / name
    path.write_text(json.dumps({
        "n_frames":
        1,
        "frames": [{
            "index": 0,
            "timestamp_ns": 0,
            "dominoes": [{
                "id": 0,
                "fall_deg": 1.0
            }]
        }],
    }),
                    encoding="utf-8")
    return str(path)


def _manifest(tmp_path, episodes):
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    return str(path)


def test_a_manifest_yields_one_track_per_episode(tmp_path):
    """The normal case once episodes are recorded automatically."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks
    episodes = [{
        "episode": i,
        "usable": True,
        "track": _one_frame_track(tmp_path, f"t{i}.json")
    } for i in (1, 2)]

    assert len(load_tracks(_manifest(tmp_path, episodes))) == 2


def test_a_single_track_path_still_works(tmp_path):
    """Naming one track directly stays valid -- that is how a track produced by
    hand is scored."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks

    assert len(load_tracks(_one_frame_track(tmp_path, "solo.json"))) == 1


def test_an_unusable_episode_is_skipped(tmp_path, caplog):
    """A take that lost a camera yields a well-formed track of the wrong thing,
    which is worse than no track at all."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks
    episodes = [
        {
            "episode": 1,
            "usable": False,
            "track": _one_frame_track(tmp_path, "bad.json")
        },
        {
            "episode": 2,
            "usable": True,
            "track": _one_frame_track(tmp_path, "good.json")
        },
    ]

    with caplog.at_level("WARNING"):
        tracks = load_tracks(_manifest(tmp_path, episodes))

    assert len(tracks) == 1
    assert "camera error" in caplog.text


def test_a_fit_waits_for_a_track_the_pipeline_is_still_writing(tmp_path):
    """The online loop fits as soon as an episode ends, while post-processing
    is still running.

    Not waiting would fall back to per-step scoring, which under open-
    loop scores the twin against itself -- the defect this path exists
    to avoid, reached silently.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import observation_track
    late = tmp_path / "late.json"
    episodes = [{"episode": 1, "usable": True, "track": str(late)}]
    manifest = _manifest(tmp_path, episodes)
    calls = []

    def _fake_sleep(_seconds):
        """Third look, the pipeline has finished."""
        calls.append(1)
        if len(calls) >= 3:
            _one_frame_track(tmp_path, "late.json")

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(observation_track.time, "sleep", _fake_sleep)
    try:
        tracks = observation_track.load_tracks(manifest, wait_s=60.0)
    finally:
        monkeypatched.undo()

    assert len(tracks) == 1
    assert calls, "it returned without ever waiting"


def test_the_wait_gives_up_rather_than_hanging(tmp_path, caplog):
    """A pipeline that died must not stall the run forever; the episode is
    skipped and the fit says it saw less than the run recorded."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks
    episodes = [{
        "episode": 1,
        "usable": True,
        "track": str(tmp_path / "never.json")
    }]

    with caplog.at_level("WARNING"):
        assert not load_tracks(_manifest(tmp_path, episodes), wait_s=0.01)
    assert "has no usable track" in caplog.text


def test_no_wait_returns_immediately(tmp_path):
    """0 disables the wait, which is what an offline re-fit over finished
    tracks wants."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks
    episodes = [{
        "episode": 1,
        "usable": True,
        "track": str(tmp_path / "absent.json")
    }]

    assert not load_tracks(_manifest(tmp_path, episodes), wait_s=0.0)


def _cascade_states(onset_steps):
    """Simulated states in which each domino topples at its given step."""
    objs = [Object(f"domino_{i}", _DOMINO) for i in range(len(onset_steps))]
    n_steps = max(onset_steps) + 25
    states = []
    for t in range(n_steps):
        data = {}
        for obj, onset in zip(objs, onset_steps):
            # 0 until the onset, then a quick monotone rise well past the
            # confirmation angle -- the shape a real topple has.
            frac = min(max((t - onset) / 8.0, 0.0), 1.0)
            data[obj] = [0, 0, 0, 0, frac * 1.5707963, 0, 0, 0]
        states.append(State(data))
    return states


def test_the_objective_prefers_the_cascade_that_matches_the_track(
        tmp_path, monkeypatch):
    """The property the whole step exists for.

    Two candidate frictions produce two cascades; the track carries one
    of them. The objective must score the matching one lower -- and it
    must do so from the intervals alone, because under open-loop the
    recorded states are the twin's own simulation and carry no
    information about which is right.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning import rollout_objective
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_sse

    # The track: onsets 6 and 4 frames apart at 60 fps, the spacing measured
    # on a real four-domino cascade.
    track_onsets = [0, 12, 20, 24]
    frames = []
    for t in range(200):
        records = []
        for i, onset in enumerate(track_onsets):
            frac = min(max((t - onset) / 8.0, 0.0), 1.0)
            records.append({"id": i, "fall_deg": frac * 90.0})
        frames.append({
            "index": t,
            "timestamp_ns": int(t * (1e9 / 60.0)),
            "dominoes": records,
        })
    track_path = _write_track(tmp_path, frames)

    matching = [0, 12, 20, 24]
    slower = [0, 30, 55, 75]

    def _fake_rollout_states(_env, _init, _actions, physical, **_kwargs):
        """Stand in for the physics: friction picks the cascade."""
        onsets = matching if physical.get("friction") == 0.5 else slower
        # The track is at 60 fps; the sim steps at 1/step_s. Convert so both
        # sides describe the same seconds.
        step_s = 0.0833
        scaled = [int(round(o / 60.0 / step_s)) for o in onsets]
        return _cascade_states(scaled)

    monkeypatch.setattr(rollout_objective, "rollout_states",
                        _fake_rollout_states)
    utils.reset_config({
        "code_sim_learning_rollout_score_observed_only": True,
        "code_sim_learning_rollout_track_path": track_path,
    })

    init = _cascade_states([0, 12, 20, 24])[0]
    trajectories = [(([init] * 2), [None])]
    sse_true = compute_rollout_sse(None, trajectories, {"friction": 0.5}, {},
                                   ["friction"])
    sse_wrong = compute_rollout_sse(None, trajectories, {"friction": 0.1}, {},
                                    ["friction"])

    assert sse_true < sse_wrong, \
        "the interval objective did not prefer the matching cascade"
    assert sse_wrong > 10 * max(sse_true, 1e-9)


def test_the_track_cache_does_not_fix_the_frame_for_the_whole_process(
        tmp_path):
    """What is cached must be the FILE, never a config-dependent derivative.

    The frame transform used to be applied before storing, with the path
    alone as the key, so whichever caller loaded first fixed the frame
    for every later one. A single load with the transform unset then
    left every subsequent evaluation matching base-frame track positions
    against world-frame twin states -- which does not raise, it silently
    degrades the id matching. On run_20260819_133802 that held matching
    at 2 of 5 dominoes for 99 evaluations.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _load_scored_track, reset_track_cache

    # Its own fixture, with CENTRES: the frame transform moves positions, and
    # the shared _cascade_track carries angles only.
    frames = [{
        "index":
        t,
        "timestamp_ns":
        int(t * (1e9 / 60.0)),
        "dominoes": [{
            "id": i,
            "fall_deg": 0.0,
            "center_base_m": [0.55, -0.15 + 0.1 * i, 0.0]
        } for i in range(4)],
    } for t in range(5)]
    track_path = _write_track(tmp_path, frames)

    def _loaded(yaw, xy):
        """The track as a caller with this frame config would see it."""
        utils.reset_config({
            "code_sim_learning_rollout_score_observed_only": True,
            "code_sim_learning_rollout_track_path": track_path,
            "code_sim_learning_track_frame_yaw": yaw,
            "code_sim_learning_track_frame_xy": xy,
        })
        return _load_scored_track(SysIdConfig.from_cfg())

    reset_track_cache()
    # First loader has NO transform -- the case that used to poison the cache.
    untransformed = _loaded(0.0, (0.0, 0.0))[0].first_xy
    # A later caller asks for the quarter turn the domino env actually uses.
    transformed = _loaded(1.5707963267948966, (0.75, 0.72))[0].first_xy

    assert untransformed != transformed, \
        "the second caller inherited the first caller's frame"
    for obj_id, (x, y) in untransformed.items():
        want = (0.75 - y, 0.72 + x)
        got = transformed[obj_id]
        assert got[0] == pytest_approx(want[0], abs=1e-6)
        assert got[1] == pytest_approx(want[1], abs=1e-6)


def test_the_cache_key_carries_the_fps_the_timings_depend_on(tmp_path):
    """A track without per-frame timestamps has every sample time computed as
    index/fallback_fps, baked straight into angles_deg.

    Keying the cache on the path alone therefore let the first loader
    fix the TIMEBASE for the whole process too -- the same first-loader-
    poisons-everyone bug as the frame transform, one field over.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        _load_scored_track, reset_track_cache

    # No timestamp_ns anywhere, so the fps is what sets the times.
    frames = [{
        "index":
        t,
        "dominoes": [{
            "id": 0,
            "fall_deg": min(max((t - 4) / 8.0, 0.0), 1.0) * 90.0
        }],
    } for t in range(40)]
    track_path = _write_track(tmp_path, frames)

    def _at(fps):
        """The track as a caller asking for this fps would see it."""
        utils.reset_config({
            "code_sim_learning_rollout_score_observed_only": True,
            "code_sim_learning_rollout_track_path": track_path,
            "code_sim_learning_track_fallback_fps": fps,
        })
        return _load_scored_track(SysIdConfig.from_cfg())[0].angles_deg[0]

    reset_track_cache()
    slow = _at(30.0)
    fast = _at(120.0)

    assert slow[-1][0] == pytest_approx(fast[-1][0] * 4.0, abs=1e-6), \
        "the second caller inherited the first caller's timebase"


def test_a_partial_mapping_does_not_score_zero_in_silence(
        tmp_path, monkeypatch, caplog):
    """A skip means "no cascade here", which is only readable when every domino
    could be named.

    With a partial mapping the same emptiness can mean "the dominoes
    that fell are the ones I could not identify". Returning nothing then
    reports a flat objective built on a measurement that never happened
    -- which the fit read as "insensitive to friction".
    """
    track_path = _cascade_track(tmp_path, [0, 12, 20, 24])
    still = _cascade_states([10_000] * 4)[:40]

    with caplog.at_level("WARNING"):
        sse = _segment_sse(still, still, track_path, monkeypatch)

    assert sse == 0.0
    # The warning says "could be matched", so the obvious negative assertion
    # -- "could not be matched" not in text -- is VACUOUSLY true and would
    # pass with the guard deleted. Match the string the code actually logs.
    assert "could be matched" not in caplog.text, \
        "a WHOLE mapping with no cascade is a legitimate silent skip"


def _cascade_track(tmp_path, onsets):
    """A track whose dominoes fall at the given frame indices."""
    frames = []
    for t in range(200):
        recs = [{
            "id": i,
            "fall_deg": min(max((t - o) / 8.0, 0.0), 1.0) * 90.0
        } for i, o in enumerate(onsets)]
        frames.append({
            "index": t,
            "timestamp_ns": int(t * (1e9 / 60.0)),
            "dominoes": recs
        })
    return _write_track(tmp_path, frames)


def _segment_sse(sim_states, recorded, track_path, monkeypatch):
    """What one scored segment contributes, through the real objective."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import rollout_objective
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_sse, reset_track_cache
    monkeypatch.setattr(rollout_objective, "rollout_states",
                        lambda *_a, **_k: sim_states)
    utils.reset_config({
        "code_sim_learning_rollout_score_observed_only": True,
        "code_sim_learning_rollout_track_path": track_path,
    })
    reset_track_cache()
    return compute_rollout_sse(None, [(recorded, [None])], {"friction": 0.5},
                               {}, ["friction"])


def test_one_episode_yields_one_set_of_residuals_not_one_per_segment(
        tmp_path, monkeypatch):
    """The track covers the WHOLE episode, so comparing against it is an
    episode-level operation.

    Scoring per segment compared the same observed intervals once per
    segment -- a cascade watched by one camera counted as many times as
    the episode happened to be cut. Segmentation is a rollout device
    (multiple shooting, to bound divergence), not a statement about how
    the evidence divides.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import rollout_objective
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_residuals, reset_track_cache
    track_path = _cascade_track(tmp_path, [0, 12, 20, 24])
    cascade = _cascade_states([0, 2, 4, 6])
    monkeypatch.setattr(rollout_objective, "rollout_states",
                        lambda *_a, **_k: cascade)
    utils.reset_config({
        "code_sim_learning_rollout_score_observed_only": True,
        "code_sim_learning_rollout_track_path": track_path,
    })

    def _terms(n_segments):
        """Residual count for one episode cut into n pieces."""
        reset_track_cache()
        return len(
            compute_rollout_residuals(None, [(cascade, [None])] * n_segments,
                                      {"friction": 0.5}, {}, ["friction"]))

    one = _terms(1)
    three = _terms(3)

    assert one > 0, "the episode must score something"
    assert three == one, \
        "cutting the episode into more segments must not multiply the " \
        "evidence -- the same observed intervals were being counted once " \
        "per segment"


def test_a_segment_with_no_cascade_scores_nothing_instead_of_penalties(
        tmp_path, monkeypatch):
    """Segmentation splits an episode; the track covers all of it.

    A pick-and-place segment has no onsets to offer, so every observed
    interval used to read as a cascade the sim had failed to reproduce
    and drew the full missing-cascade penalty. On run_20260819_104757
    that put 21-24 WHOLE penalties into every evaluation -- the SSEs
    came back as integer multiples of one -- which made every physical
    parameter read as flat and is what the agent then declined on.
    """
    track_path = _cascade_track(tmp_path, [0, 12, 20, 24])
    still = _cascade_states([10_000] * 4)[:40]

    assert _segment_sse(still, still, track_path, monkeypatch) == 0.0, \
        "a segment with no cascade on either side must contribute nothing"


def test_a_theta_that_stalls_the_cascade_is_still_penalised(
        tmp_path, monkeypatch):
    """The trap in that skip, and why the recorded states are consulted too.

    A candidate that BREAKS the chain also produces no onsets. Skipping
    on the rollout alone would score it zero -- making a friction that
    stops the cascade look better than one that reproduces it, the
    inversion interval_residuals penalises a one-sided domino to
    prevent.
    """
    track_path = _cascade_track(tmp_path, [0, 12, 20, 24])
    stalled = _cascade_states([10_000] * 4)[:40]
    # The recorded states DO cascade, so this is a real cascade segment.
    recorded = _cascade_states([0, 2, 4, 6])

    assert _segment_sse(stalled, recorded, track_path, monkeypatch) > 0.0, \
        "a stalled cascade in a real cascade segment must still be penalised"


def test_the_objective_scores_a_cascade_the_same_however_slow_it_was(
        tmp_path, monkeypatch):
    """The scaling has to reach the objective, not just exist beside it.

    Two episodes disagreeing by the SAME FRACTION of their own cascade
    must score identically: one cascade takes twice as long as the other
    and the twin is wrong by twice as much, so the twin is equally wrong
    in both. In raw seconds the slow one scores 4x the fast one purely
    for having taken longer, which is what let a slow cascade sit above
    the trim bar while an identically-wrong fast one passed.
    """
    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    fast_dir.mkdir()
    slow_dir.mkdir()
    fast = _segment_sse(_cascade_states([0, 3, 6, 9]),
                        _cascade_states([0, 3, 6, 9]),
                        _cascade_track(fast_dir, [0, 10, 20, 30]), monkeypatch)
    slow = _segment_sse(_cascade_states([0, 6, 12, 18]),
                        _cascade_states([0, 6, 12, 18]),
                        _cascade_track(slow_dir, [0, 20, 40, 60]), monkeypatch)

    assert fast > 0.0, "the twin disagrees with the track, so this must score"
    assert slow == pytest.approx(fast, rel=1e-6), \
        "the same proportional disagreement must cost the same; scoring in " \
        "raw seconds charges the slower cascade 4x for its duration alone"


def test_the_episode_path_scales_its_residuals_too(tmp_path, monkeypatch):
    """The two interval paths are wired separately, so both need proving.

    One track against one trajectory scores through
    ``_interval_residual_terms``; one track against the several segments
    an episode was cut into scores through ``_episode_interval_terms``.
    A fix applied to one and not the other is invisible in a test that
    only drives the first.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import rollout_objective
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_sse, reset_track_cache

    def _episode_sse(sim_states, track_path):
        """One episode cut into two segments, so the counts cannot pair."""
        monkeypatch.setattr(rollout_objective, "rollout_states",
                            lambda *_a, **_k: sim_states)
        utils.reset_config({
            "code_sim_learning_rollout_score_observed_only": True,
            "code_sim_learning_rollout_track_path": track_path,
        })
        reset_track_cache()
        return compute_rollout_sse(None, [(sim_states, [None])] * 2,
                                   {"friction": 0.5}, {}, ["friction"])

    fast_dir = tmp_path / "fast"
    slow_dir = tmp_path / "slow"
    fast_dir.mkdir()
    slow_dir.mkdir()
    fast = _episode_sse(_cascade_states([0, 3, 6, 9]),
                        _cascade_track(fast_dir, [0, 10, 20, 30]))
    slow = _episode_sse(_cascade_states([0, 6, 12, 18]),
                        _cascade_track(slow_dir, [0, 20, 40, 60]))

    assert fast > 0.0, "the twin disagrees with the track, so this must score"
    assert slow == pytest.approx(fast, rel=1e-6), \
        "the episode path must scale by the observed span as well"


def test_interval_residuals_are_a_fraction_of_the_observed_span():
    """Seconds are the wrong units for every consumer of a residual.

    The trim threshold is ``trim_rms_factor * noise_sigma`` = 0.1,
    meaning "10% of typical motion" because per-step residuals go
    through compute_residual_scaling. Interval residuals never did, so
    the same bar demanded the twin reproduce a cascade to ~45 ms before
    the fit would look at it, and every cascade-bearing segment was
    dropped as unexplainable.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.rollout_objective import _interval_scale

    # A real cascade: the span is the last domino's interval.
    obs = {3: 0.0, 2: 0.7669, 1: 0.9001, 0: 1.0668}
    sim = {3: 0.0, 2: 0.0833, 1: 0.4167, 0: 0.5833}
    scale = _interval_scale(obs, penalty=25.62)

    assert scale == pytest.approx(1.0668), \
        "the scale is the observed propagation span, read off the track"
    scaled = interval_residuals(sim, obs, 25.62, scale)
    raw = interval_residuals(sim, obs, 25.62)

    assert scaled == pytest.approx([r / 1.0668 for r in raw])
    assert max(abs(r) for r in scaled) < 1.0, \
        "a disagreement smaller than the whole cascade must score under 1.0"
    # The divisor must not come from the rollout: a theta that stalls the
    # chain would stretch the sim's span and shrink its own residuals.
    assert _interval_scale({}, penalty=25.62) == pytest.approx(25.62), \
        "with no observed span the penalty is the scale, so a one-sided " \
        "domino costs 1.0 rather than an unbounded number of seconds"


def test_a_trajectory_with_no_residuals_scores_infinite_not_zero(monkeypatch):
    """Nothing measured must never outrank something measured.

    ``per_trajectory_rms`` used to turn an empty residual vector into an
    RMS of 0.0 -- the best score obtainable. Under interval scoring a
    segment holding no cascade yields exactly that empty vector, so on
    run_20260820_123606 the trimmer saw best RMS ['0', '1.497'] against
    a 0.1 bar, kept the segment that measured nothing and dropped the
    one carrying the only cascade in the run.
    """
    # pylint: disable-next=import-outside-toplevel
    import numpy as np

    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import rollout_objective

    def _fake(_env, trajectories, *_a, **_k):
        """Empty for the first trajectory, two real residuals for the
        second."""
        return (np.asarray([], dtype=float) if trajectories[0][0] == "empty"
                else np.asarray([0.3, 0.4], dtype=float))

    monkeypatch.setattr(rollout_objective, "compute_rollout_residuals", _fake)
    rms = rollout_objective.per_trajectory_rms(None, [("empty", []),
                                                      ("real", [])], {}, {},
                                               [])

    assert math.isinf(rms[0]), \
        "a trajectory nothing could be measured on must not score 0.0, " \
        "which is the best RMS there is and beats every real measurement"
    assert rms[1] == pytest.approx(math.sqrt((0.3**2 + 0.4**2) / 2))
    assert rms[0] > rms[1], \
        "the unmeasured trajectory must rank WORSE than the measured one"


def test_per_trajectory_rms_reports_the_callers_own_episode_count(monkeypatch):
    """Pairing is a property of the whole set, not of one call's sublist.

    ``per_trajectory_rms`` scores segments one at a time, so every call
    reaches the objective with a list of length 1. With a single track
    loaded, ``len(tracks) == len(trajectories)`` is then 1 == 1 and each
    fragment was treated as a whole episode and anchored on itself --
    which is why segments before the cascade reported all five dominoes
    unmatched. The full count has to travel with the call.
    """
    # pylint: disable-next=import-outside-toplevel
    import numpy as np

    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning import rollout_objective

    seen = []

    def _fake(_env, trajectories, *_a, **kwargs):
        """Record what the objective was told about the caller's list."""
        seen.append((len(trajectories), kwargs.get("episode_count")))
        return np.asarray([1.0], dtype=float)

    monkeypatch.setattr(rollout_objective, "compute_rollout_residuals", _fake)
    rollout_objective.per_trajectory_rms(None, [("a", []), ("b", []),
                                                ("c", [])], {}, {}, [])

    assert seen == [(1, 3), (1, 3), (1, 3)], \
        "each call scores one trajectory but must report that the caller " \
        "holds 3, so one track cannot be mistaken for a per-episode pairing"


def pytest_approx(value, abs=1e-9):  # pylint: disable=redefined-builtin
    """Local approx so the comparisons above read as equations."""
    return pytest.approx(value, abs=abs)


def test_sysid_config_track_path_anchored_at_launch_cwd(monkeypatch, tmp_path):
    """A relative track path resolves against the launch directory even when
    ``from_cfg`` runs under a different working directory.

    Regression: run_python's exec window chdirs into the agent sandbox,
    so a sim.fit issued from agent code would open the fan domain's
    relative track path against the sandbox, miss it, and silently fall
    back to per-step scoring.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.config import SysIdConfig
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import LAUNCH_CWD
    utils.reset_config({
        "code_sim_learning_rollout_track_path":
        "logs/zed_tracks/tracks.json",
    })
    monkeypatch.chdir(tmp_path)  # simulate the sandbox exec window
    cfg = SysIdConfig.from_cfg()
    assert cfg.track_path == os.path.join(LAUNCH_CWD,
                                          "logs/zed_tracks/tracks.json")
    # Absolute and empty paths pass through untouched.
    abs_path = str(tmp_path / "tracks.json")
    utils.reset_config({"code_sim_learning_rollout_track_path": abs_path})
    assert SysIdConfig.from_cfg().track_path == abs_path
    utils.reset_config({"code_sim_learning_rollout_track_path": ""})
    assert SysIdConfig.from_cfg().track_path == ""
