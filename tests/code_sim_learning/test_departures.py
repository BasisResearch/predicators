"""Tests for the notes on where a model's replay leaves a dropped segment."""
# pylint: disable=protected-access

import numpy as np

from predicators.code_sim_learning import departures as departures_module
from predicators.code_sim_learning.departures import Departure, \
    describe_departures, find_departure, segment_departures
from predicators.code_sim_learning.rollout_objective import StepResidual
from predicators.code_sim_learning.trajectory_prep import segment_origins
from predicators.structs import Object, State
from tests.code_sim_learning.test_physical_sysid import _DOMINO_TYPE, \
    _RESIDUAL_FEATURES, _LinearParamEnv, _trajectory


def _series(obj_name, feature, residuals, recorded=0.4):
    """Step residuals of one feature whose recording stays at ``recorded``."""
    return [
        StepResidual(step, obj_name, feature, recorded + r, recorded, r)
        for step, r in enumerate(residuals)
    ]


def test_the_departure_is_where_the_largest_error_starts():
    """The feature with the most squared error is named, from the first step at
    least half as far off as its farthest."""
    residuals = (_series("ball", "x", [0.01, 0.02, 0.01, 0.0]) +
                 _series("ball", "z", [0.01, -0.3, -0.5, -0.4]))
    found = find_departure(7, residuals)
    assert found is not None
    assert (found.segment, found.num_steps) == (7, 4)
    assert (found.obj_name, found.feature) == ("ball", "z")
    assert np.isclose(found.share, 0.5001 / (0.5001 + 0.0006))
    assert found.first_step == 2
    assert np.isclose(found.first_model, 0.1)
    assert found.first_recorded == 0.4
    assert found.worst_step == 3
    assert np.isclose(found.worst_model, -0.1)


def test_a_replay_that_matches_its_recording_has_no_departure():
    """No error, or nothing scored, is no departure."""
    assert find_departure(0, _series("ball", "x", [0.0, 0.0])) is None
    assert find_departure(0, []) is None


def test_each_dropped_segment_is_replayed_at_its_own_values():
    """A real replay: the domino the model moves by k = 0.05 per step falls
    behind a recording that moves 0.1 per step."""
    built = []
    trajectories = [
        _trajectory([0.0, 0.0, 0.0, 0.0]),
        _trajectory([0.0, 0.1, 0.2, 0.3]),
    ]
    found = segment_departures(lambda: _LinearParamEnv(built), trajectories,
                               [1], [{
                                   "k": 0.05
                               }], _RESIDUAL_FEATURES, ["k"])
    assert len(found) == 1
    departure = found[0]
    assert (departure.segment, departure.num_steps) == (1, 3)
    assert (departure.obj_name, departure.feature) == ("d0", "x")
    assert departure.share == 1.0
    assert departure.first_step == 2
    assert np.isclose(departure.first_model, 0.1)
    assert np.isclose(departure.first_recorded, 0.2)
    assert departure.worst_step == 3
    assert np.isclose(departure.worst_model, 0.15)
    assert np.isclose(departure.worst_recorded, 0.3)
    assert len(built) == 1
    assert not built[0].connected


def test_a_segment_whose_replay_fails_is_left_out(monkeypatch):
    """A replay that raises leaves its segment out; interval scoring, which has
    no per-step comparison, leaves them all out."""
    calls = []

    def fake_step_residuals(_env, trajectory, *_args, **_kwargs):
        calls.append(trajectory)
        if trajectory == "broken":
            raise RuntimeError("model raised")
        return _series("ball", "z", [0.0, 0.2])

    monkeypatch.setattr(departures_module, "step_residuals",
                        fake_step_residuals)
    found = segment_departures(None, ["broken", "fine"], [0, 1], [{}, {}], {},
                               [])
    assert [d.segment for d in found] == [1]
    assert calls == ["broken", "fine"]
    monkeypatch.setattr(departures_module, "step_residuals",
                        lambda *_a, **_k: None)
    assert not segment_departures(None, ["a"], [0], [{}], {}, [])


def _departure(segment, feature="z", num_steps=22, share=0.97):
    return Departure(segment=segment,
                     num_steps=num_steps,
                     obj_name="ball",
                     feature=feature,
                     share=share,
                     first_step=2,
                     first_model=0.342,
                     first_recorded=0.433,
                     worst_step=num_steps,
                     worst_model=0.04,
                     worst_recorded=0.445)


def test_the_note_groups_departures_by_feature():
    """One line per feature, its most common first, naming up to three segments
    by their episode steps and counting the rest."""
    found = [_departure(s) for s in (1, 3, 4, 6)]
    found.append(_departure(9, feature="y", num_steps=50, share=0.4))
    origins = {s: (0, 1130 + 100 * s) for s in (1, 3, 4, 6, 9)}
    lines = describe_departures(found, 6, [origins.get(s) for s in range(10)])
    assert lines[0].startswith("Where your model leaves each dropped segment")
    assert lines[1] == (
        "  ball.z, in 4 of the 6: segment 1 (episode 1, steps 1231-1252): "
        "0.342 against 0.433 from step 1232, 0.04 against 0.445 at step "
        "1252; segment 3 (episode 1, steps 1431-1452): 0.342 against 0.433 "
        "from step 1432, 0.04 against 0.445 at step 1452; segment 4 "
        "(episode 1, steps 1531-1552): 0.342 against 0.433 from step 1532, "
        "0.04 against 0.445 at step 1552; and 1 more.")
    assert lines[2] == (
        "  ball.y, in 1 of the 6: segment 9 (episode 1, steps 2031-2080): "
        "0.342 against 0.433 from step 2032, 0.04 against 0.445 at step "
        "2080 (40% of its error; other features carry the rest).")
    # Without provenance the steps are the segment's own.
    assert describe_departures([_departure(5)], 1)[1] == (
        "  ball.z, in 1 of the 1: segment 5 (22 steps): 0.342 against "
        "0.433 from its step 2, 0.04 against 0.445 at its step 22.")
    assert not describe_departures([], 3)


def test_segments_are_found_in_the_recordings_they_were_cut_from():
    """A segment is located by its later states, whose identity slicing keeps;
    its first state may be a new one."""
    first, _ = _trajectory([0.0, 0.1, 0.2])
    second, actions = _trajectory([0.0, 0.0, 0.1, 0.2, 0.3, 0.3])
    rest_mean = State({Object("d0", _DOMINO_TYPE): np.array([0.0])})
    cut = ([rest_mean] + second[2:5], actions[1:4])
    foreign, foreign_actions = _trajectory([0.0, 0.1])
    assert segment_origins([first, second],
                           [(first, actions[:2]), cut,
                            (foreign, foreign_actions)]) == [(0, 0), (1, 1),
                                                             None]
