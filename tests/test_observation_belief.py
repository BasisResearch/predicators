"""Tests for the execution-time belief (``predicators/observation_belief.py``,
docs/uncertainty/design.md 3.3 and 3.6)."""

import numpy as np
import pytest

from predicators.observation_belief import _truncated_normal, atom_fractions, \
    belief_draw, change_point_belief, describe_fractions, likely_atoms, \
    smooth_frames, uncertain_atoms
from predicators.observation_noise import ObservationNoise
from predicators.structs import GroundAtom, Object, Predicate, State, Type

_BLOCK = Type("block", ["x", "yaw", "lit"], angular_features=["yaw"])
_ROBOT = Type("robot", ["x"])
_b0, _b1, _r = Object("b0", _BLOCK), Object("b1", _BLOCK), Object("r", _ROBOT)


def _frames(b0_xs, b1_xs, yaw=0.0, robot_x=0.3):
    return [
        State({
            _b0: np.array([x0, yaw, 1.0], dtype=float),
            _b1: np.array([x1, yaw, 0.0], dtype=float),
            _r: np.array([robot_x], dtype=float),
        }) for x0, x1 in zip(b0_xs, b1_xs)
    ]


def test_rest_windows_grow_and_moving_objects_are_not_smoothed():
    """A resting object averages its frames (spread sigma / sqrt(n)); a moving
    one keeps its latest frame; exact types and features are untouched."""
    rng = np.random.default_rng(1)
    noise = ObservationNoise(position=0.01, orientation=0.0)
    rest = [0.5 + rng.normal(0.0, 0.01) for _ in range(6)]
    moving = [0.2 + 0.05 * k for k in range(6)]
    frames = _frames(rest, moving)
    belief = smooth_frames(frames, noise, window=8, sigmas=3.0)
    assert belief.frames_used["b0"] == 6
    assert belief.frames_used["b1"] == 1
    assert belief.frame.get(_b0, "x") == pytest.approx(np.mean(rest))
    assert belief.frame.get(_b1, "x") == moving[-1]
    assert belief.spread[("b0", "x")] == pytest.approx(0.01 / np.sqrt(6))
    assert belief.spread[("b1", "x")] == pytest.approx(0.01)
    assert ("b0", "lit") not in belief.spread
    assert ("r", "x") not in belief.spread
    assert belief.frame.get(_r, "x") == 0.3
    assert belief.frame.get(_b0, "lit") == 1.0
    assert belief.max_spread() == pytest.approx(0.01)
    # The window caps the frames averaged.
    capped = smooth_frames(frames, noise, window=4, sigmas=3.0)
    assert capped.frames_used["b0"] == 4
    assert capped.frame.get(_b0, "x") == pytest.approx(np.mean(rest[-4:]))
    line = belief.object_line(_b0)
    assert line.startswith("b0: x ") and line.endswith("(6 frames)")
    assert "yaw" not in line  # orientation is exact here
    single = smooth_frames(frames[:1], noise, window=8, sigmas=3.0)
    assert single.frames_used["b0"] == 1
    assert single.frame.get(_b0, "x") == rest[0]


def test_a_move_resets_the_window_and_angles_average_circularly():
    """Rest, a jump, rest again: the window restarts at the jump.

    Angles at the +-pi seam average to the seam.
    """
    rng = np.random.default_rng(2)
    noise = ObservationNoise(position=0.01, orientation=0.05)
    xs = [0.5 + rng.normal(0.0, 0.01) for _ in range(5)] + \
        [0.9 + rng.normal(0.0, 0.01) for _ in range(3)]
    frames = _frames(xs, [0.0] * 8, yaw=0.0)
    belief = smooth_frames(frames, noise, window=8, sigmas=3.0)
    assert belief.frames_used["b0"] == 3
    assert belief.frame.get(_b0, "x") == pytest.approx(np.mean(xs[-3:]))
    yaws = [np.pi - 0.02 + rng.normal(0.0, 0.05) for _ in range(6)]
    yaws = [float((y + np.pi) % (2 * np.pi) - np.pi) for y in yaws]
    assert min(yaws) < 0 < max(yaws)
    seam = [
        State({
            _b0: np.array([0.5, y, 1.0], dtype=float),
            _b1: np.array([0.0, 0.0, 0.0], dtype=float),
            _r: np.array([0.3], dtype=float),
        }) for y in yaws
    ]
    belief_seam = smooth_frames(seam, noise, window=8, sigmas=3.0)
    assert belief_seam.frames_used["b0"] == 6
    mean_yaw = belief_seam.frame.get(_b0, "yaw")
    assert abs(((mean_yaw - np.pi) + np.pi) % (2 * np.pi) - np.pi) < 0.06


def test_draws_and_atom_fractions_follow_the_spread():
    """Draws jitter noisy features by the spread only; an atom near its
    boundary holds on a fraction of draws, an atom far from it always."""
    noise = ObservationNoise(position=0.01, orientation=0.0)
    frames = _frames([0.5] * 4, [0.9] * 4)
    belief = smooth_frames(frames, noise, window=8, sigmas=3.0)
    rng = np.random.default_rng(3)
    draws = [belief_draw(belief, rng) for _ in range(200)]
    xs = np.array([d.get(_b0, "x") for d in draws])
    assert abs(xs.mean() - 0.5) < 0.002
    assert abs(xs.std() - 0.005) < 0.0015
    assert all(d.get(_r, "x") == 0.3 for d in draws)
    assert all(d.get(_b0, "lit") == 1.0 for d in draws)
    near = Predicate("NearHalf", [_BLOCK], lambda s, o: s.get(o[0], "x") < 0.5)
    far = Predicate("BelowOne", [_BLOCK], lambda s, o: s.get(o[0], "x") < 1.0)
    fractions = atom_fractions(belief, {near, far}, 400,
                               np.random.default_rng(4))
    assert fractions[GroundAtom(far, [_b0])] == 1.0
    assert fractions[GroundAtom(far, [_b1])] == 1.0
    assert 0.35 < fractions[GroundAtom(near, [_b0])] < 0.65
    assert GroundAtom(near, [_b1]) not in fractions
    likely = likely_atoms(fractions)
    assert GroundAtom(far, [_b0]) in likely
    unsure = uncertain_atoms(fractions)
    assert unsure == [GroundAtom(near, [_b0])]
    text = describe_fractions(fractions, unsure + [GroundAtom(near, [_b1])])
    assert text.startswith("NearHalf(b0:block) 0.") and text.endswith(
        "NearHalf(b1:block) 0.00")


def _scene_frames(b0_xs, scale=1.0, yaw=0.0):
    """Frames of b0 moving through ``b0_xs`` with b1 parked across the
    table, in units of ``scale`` (1 = metres)."""
    return [
        State({
            _b0: np.array([x * scale, yaw, 1.0], dtype=float),
            _b1: np.array([1.0 * scale, yaw, 0.0], dtype=float),
            _r: np.array([0.3], dtype=float),
        }) for x in b0_xs
    ]


def test_change_point_weights_concentrate_on_the_whole_rest():
    """A resting object puts its run-length mass on the longest run, and its
    mixture mean and spread approach the plain mean and sigma / sqrt(n)."""
    rng = np.random.default_rng(3)
    noise = ObservationNoise(position=0.01, orientation=0.0)
    rest = [0.5 + rng.normal(0.0, 0.01) for _ in range(8)]
    belief = change_point_belief(_scene_frames(rest),
                                 noise,
                                 window=8,
                                 hazard=0.02)
    weights = belief.run_weights["b0"]
    assert weights.shape == (8, )
    assert weights.sum() == pytest.approx(1.0)
    assert weights[-1] > 0.8
    assert belief.frames_used["b0"] == 8
    assert belief.frame.get(_b0, "x") == pytest.approx(np.mean(rest), abs=1e-3)
    assert belief.spread[("b0", "x")] == pytest.approx(0.01 / np.sqrt(8),
                                                       rel=0.35)
    assert ("r", "x") not in belief.spread
    assert belief.frame.get(_b0, "lit") == 1.0


def test_change_point_jump_moves_the_mass_to_the_latest_frame():
    """A jump after a rest leaves only the latest frame in the run."""
    rng = np.random.default_rng(4)
    noise = ObservationNoise(position=0.01, orientation=0.0)
    xs = [0.5 + rng.normal(0.0, 0.01) for _ in range(6)] + [0.65]
    belief = change_point_belief(_scene_frames(xs),
                                 noise,
                                 window=8,
                                 hazard=0.02)
    weights = belief.run_weights["b0"]
    assert weights[0] > 0.95
    assert belief.frames_used["b0"] == 1
    assert belief.frame.get(_b0, "x") == pytest.approx(0.65, abs=1e-3)
    assert belief.spread[("b0", "x")] == pytest.approx(0.01, rel=0.1)


def test_change_point_weights_do_not_depend_on_units():
    """Rescaling a feature, its noise and hence its range leaves the run-
    length weights unchanged."""
    rng = np.random.default_rng(5)
    xs = [0.5 + rng.normal(0.0, 0.01) for _ in range(5)] + \
        [0.53 + rng.normal(0.0, 0.01) for _ in range(3)]
    metres = change_point_belief(_scene_frames(xs),
                                 ObservationNoise(position=0.01,
                                                  orientation=0.0),
                                 window=8,
                                 hazard=0.05)
    millimetres = change_point_belief(_scene_frames(xs, scale=1000.0),
                                      ObservationNoise(position=10.0,
                                                       orientation=0.0),
                                      window=8,
                                      hazard=0.05)
    assert np.allclose(metres.run_weights["b0"], millimetres.run_weights["b0"])


def test_change_point_draws_follow_the_mixture_and_stay_in_range():
    """Draws pick a run length, then a value around that run's mean; they stay
    inside the recorded range and angles stay wrapped."""
    noise = ObservationNoise(position=0.01, orientation=0.1)
    frames = _scene_frames([0.5] * 6, yaw=3.1)
    belief = change_point_belief(frames, noise, window=8, hazard=0.02)
    rng = np.random.default_rng(6)
    xs, yaws = [], []
    for _ in range(4000):
        draw = belief_draw(belief, rng)
        xs.append(draw.get(_b0, "x"))
        yaws.append(draw.get(_b0, "yaw"))
        assert draw.get(_r, "x") == 0.3
    lo, hi = belief.ranges[("b0", "x")]
    assert min(xs) >= lo and max(xs) <= hi
    assert np.mean(xs) == pytest.approx(0.5, abs=1e-3)
    assert np.std(xs) == pytest.approx(belief.spread[("b0", "x")], rel=0.1)
    assert all(-np.pi <= y <= np.pi for y in yaws)
    # The yaw draws straddle the wrap at pi without averaging to zero.
    assert np.mean(np.cos(yaws)) < -0.9


def test_draws_at_a_range_end_follow_the_truncated_posterior():
    """With the mean at the range's upper end, draws fill the half below it
    with the half-normal's density; none pile up on the end itself."""
    rng = np.random.default_rng(0)
    values = np.array(
        [_truncated_normal(1.0, 0.1, 0.0, 1.0, rng) for _ in range(4000)])
    assert values.max() < 1.0 and values.min() >= 0.0
    assert values.mean() == pytest.approx(1.0 - 0.1 * np.sqrt(2 / np.pi),
                                          abs=0.005)
    assert np.mean(values > 0.999) < 0.02
    # A range the Gaussian puts no resolvable mass on: the nearer end.
    assert _truncated_normal(5.0, 0.1, 0.0, 1.0, rng) == 1.0
