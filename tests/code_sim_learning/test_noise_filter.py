"""Tests for the fit-side filter (``code_sim_learning_rollout_noise_filter``)
and the carried posterior (``code_sim_learning_carry_posterior``),
docs/continual-uncertainty.md 3.3.

Under a centimetre of observation noise the per-step motion detector
flags every step (1 cm of noise against a 1 mm tolerance), so the
settled-tail truncation never cuts and the rest-point segmentation never
finds a rest point. The filter judges motion by window means, in units
of the declared sigma, and starts each rest-anchored segment from the
mean of its preceding rest window, the errors-in-variables correction
for the rollout's initial condition.
"""

# pylint: disable=protected-access

import dataclasses

import numpy as np
import pytest

from predicators import utils
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning import trajectory_prep
from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.identifiability import Verdict
from predicators.observation_noise import ObservationNoise
from predicators.structs import Action, Object, State, Type

_DOMINO = Type("domino", ["x", "yaw"], angular_features=["yaw"])
_ROBOT = Type("robot", ["x"])
_FEATURES = {"domino": ["x", "yaw"]}


def _trajectory(xs, yaws=None):
    domino = Object("d0", _DOMINO)
    robot = Object("r0", _ROBOT)
    if yaws is None:
        yaws = [0.0] * len(xs)
    states = [
        State({
            domino: np.array([x, yaw], dtype=float),
            robot: np.array([0.0], dtype=float),
        }) for x, yaw in zip(xs, yaws)
    ]
    actions = [
        Action(np.zeros(1, dtype=np.float32)) for _ in range(len(states) - 1)
    ]
    return states, actions


def _config(noise_filter, window=8, sigmas=4.0, sigma=0.01):
    utils.reset_config({})
    return dataclasses.replace(SysIdConfig.from_cfg(),
                               noise_filter=noise_filter,
                               noise_window=window,
                               settle_sigmas=sigmas,
                               observation_noise=ObservationNoise(
                                   position=sigma, orientation=0.0),
                               settle_tol=1e-3,
                               settle_margin=20,
                               segment_min_rest_steps=10)


def _noisy_push():
    """Rest for 40 frames at 0.5, 20 steps of 1 cm motion, rest for 60 frames
    at 0.7, all read through 1 cm of position noise."""
    truth = [0.5] * 40 + [0.5 + 0.01 * k for k in range(1, 21)] + [0.7] * 60
    rng = np.random.default_rng(7)
    observed = [x + rng.normal(0.0, 0.01) for x in truth]
    return truth, observed


def test_windowed_detector_finds_the_rest_phases():
    """The per-step detector reads noise as motion everywhere; the windowed one
    isolates the push."""
    _truth, observed = _noisy_push()
    states, actions = _trajectory(observed)
    legacy = trajectory_prep._active_step_indices(states, actions, _FEATURES,
                                                  1e-3)
    assert len(legacy) > 100  # nearly every one of the 119 steps
    noise = ObservationNoise(position=0.01)
    active = trajectory_prep._active_step_indices(states,
                                                  actions,
                                                  _FEATURES,
                                                  1e-3,
                                                  noise=noise,
                                                  window=8,
                                                  sigmas=4.0)
    assert active, "the push must be detected"
    # The push spans steps 39..58; the window means see it a few steps
    # early and late, never in the rest phases.
    assert 30 <= min(active) <= 40
    assert 55 <= max(active) <= 68


def test_filter_truncates_and_segments_under_noise():
    """With the filter on, the settled tail is cut and the segment starts from
    the denoised rest pose; off, nothing is cut and nothing is found."""
    _truth, observed = _noisy_push()
    traj = _trajectory(observed)
    on = _config(True)
    off = _config(False)
    cut_states, cut_actions = trajectory_prep.truncate_settled_tail(traj,
                                                                    _FEATURES,
                                                                    config=on)
    assert len(cut_actions) < len(traj[1])
    assert len(cut_states) == len(cut_actions) + 1
    # The last motion (step 58) plus the margin, give or take the window.
    assert 75 <= len(cut_actions) <= 90
    kept_states, kept_actions = trajectory_prep.truncate_settled_tail(
        traj, _FEATURES, config=off)
    assert len(kept_actions) == len(traj[1])
    assert kept_states is traj[0]
    segments = trajectory_prep.split_at_rest_points(traj, _FEATURES, config=on)
    assert len(segments) == 1
    seg_states, seg_actions = segments[0]
    domino = next(o for o in seg_states[0] if o.name == "d0")
    # The segment's first frame is the window mean of the rest phase:
    # within a couple of millimetres of the true 0.5, unlike the raw
    # frame it replaces.
    denoised = seg_states[0].get(domino, "x")
    assert abs(denoised - 0.5) < 0.006
    raw_frames = [s.get(domino, "x") for s in traj[0]]
    assert denoised not in raw_frames
    # The rest of the segment is the observed frames, untouched.
    assert seg_states[1].get(domino, "x") in raw_frames
    assert len(seg_actions) >= 20
    # Off: no rest point is ever found, so the whole trajectory is one
    # segment starting at its first frame.
    segments_off = trajectory_prep.split_at_rest_points(traj,
                                                        _FEATURES,
                                                        config=off)
    assert len(segments_off) == 1
    assert segments_off[0][0][0] is traj[0][0]
    assert len(segments_off[0][1]) == len(traj[1])


def test_exact_channel_leaves_the_legacy_path_untouched():
    """Without a declared channel the filter flag changes nothing."""
    xs = [0.5] * 30 + [0.5 + 0.01 * k for k in range(1, 11)] + [0.6] * 60
    traj = _trajectory(xs)
    on = dataclasses.replace(_config(True), observation_noise=None)
    off = dataclasses.replace(_config(False), observation_noise=None)
    seg_on = trajectory_prep.split_at_rest_points(traj, _FEATURES, config=on)
    seg_off = trajectory_prep.split_at_rest_points(traj, _FEATURES, config=off)
    assert len(seg_on) == len(seg_off) == 1
    # The same frames, by identity: no denoised copy was made.
    assert all(a is b for a, b in zip(seg_on[0][0], seg_off[0][0]))
    assert len(seg_on[0][1]) == len(seg_off[0][1])
    cut_on = trajectory_prep.truncate_settled_tail(traj, _FEATURES, config=on)
    cut_off = trajectory_prep.truncate_settled_tail(traj,
                                                    _FEATURES,
                                                    config=off)
    assert len(cut_on[1]) == len(cut_off[1]) < len(traj[1])


def test_rest_mean_is_circular_for_angles():
    """A rest pose at the +-pi seam averages to the seam, not to zero."""
    rng = np.random.default_rng(3)
    yaws = [np.pi - 0.01 + rng.normal(0.0, 0.05) for _ in range(10)]
    yaws = [float((y + np.pi) % (2 * np.pi) - np.pi) for y in yaws]
    assert min(yaws) < 0 < max(yaws)  # the readings straddle the seam
    states, _actions = _trajectory([0.5] * 10, yaws)
    noise = ObservationNoise(position=0.0, orientation=0.05)
    start = trajectory_prep._rest_mean_state(states, 9, 8, noise)
    domino = next(o for o in start if o.name == "d0")
    mean_yaw = start.get(domino, "yaw")
    assert abs(trajectory_prep._wrap_angle(mean_yaw - np.pi)) < 0.05
    # Exact features keep the frame's value.
    assert start.get(domino, "x") == 0.5
    # A window of one frame is the frame itself.
    single = trajectory_prep._rest_mean_state(states, 0, 8, noise)
    assert single.get(domino, "yaw") == yaws[0]


class _RegistryEnv:
    """Stands in for a base env with one registered physical param."""

    def get_physical_param_info(self):
        """The registry: one param with its default."""
        return {"friction": {"default": 0.1}}


def _bare_approach():
    approach = object.__new__(AgentSimLearningApproach)
    approach._base_env = _RegistryEnv()
    approach._carried_physical_prior = {}
    return approach


def test_carried_posterior_becomes_the_next_prior_centre():
    """A deployed fit's value replaces the registry anchor for the next fit; an
    anchor fallback is never carried; off, the registry anchor stands."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log")]
    approach = _bare_approach()
    utils.reset_config({"code_sim_learning_carry_posterior": True})
    assert approach.fit_prior_anchors(specs) == {"friction": 0.1}
    approach.note_carried_posterior(
        {"friction": 0.1}, {"friction": {
            "verdict": Verdict.NOT_IDENTIFIED
        }})
    assert approach.fit_prior_anchors(specs) == {"friction": 0.1}
    approach.note_carried_posterior({"friction": 0.4},
                                    {"friction": {
                                        "verdict": Verdict.WIDE
                                    }})
    assert approach.fit_prior_anchors(specs) == {"friction": 0.4}
    assert approach._carried_physical_prior == {"friction": 0.4}
    # A parameter no longer declared is not carried into the anchors.
    other = [ParamSpec("restitution", 0.02, lo=0.0, hi=0.9)]
    assert approach.fit_prior_anchors(other) == {}
    utils.reset_config({"code_sim_learning_carry_posterior": False})
    assert approach.fit_prior_anchors(specs) == {"friction": 0.1}
    approach.note_carried_posterior(
        {"friction": 0.6}, {"friction": {
            "verdict": Verdict.IDENTIFIED
        }})
    assert approach._carried_physical_prior == {"friction": 0.4}
    utils.reset_config({})


def test_filter_params_follow_the_flag_and_the_channel():
    """The filter is on only with the flag and a declared channel."""
    on = _config(True, window=5, sigmas=2.5)
    assert trajectory_prep._filter_params(on)[1:] == (5, 2.5)
    assert trajectory_prep._filter_params(_config(False)) == (None, 1, 0.0)
    exact = dataclasses.replace(on, observation_noise=None)
    assert trajectory_prep._filter_params(exact) == (None, 1, 0.0)
    assert trajectory_prep._mean_delta(np.array([]), np.array([1.0]),
                                       False) != pytest.approx(0.0)
