"""The fit's reach check: how far each scored body goes in the recordings,
against the fitted model's replay of them."""
# pylint: disable=protected-access

import dataclasses
from types import SimpleNamespace

import numpy as np

from predicators import utils
from predicators.agent_sdk.tools import ToolContext
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.code_sim_learning import rollout_objective
from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.reach import describe_reach, reach_gaps, \
    reach_reminder
from predicators.observation_noise import ObservationNoise
from predicators.structs import Action, Object, State, Type

_BOX_TYPE = Type("box", ["z"])
_BOX = Object("box", _BOX_TYPE)
_FEATURES = {"box": ["z"]}
_SIGMA = 0.01


def _states(zs):
    return [State({_BOX: np.array([z], dtype=float)}) for z in zs]


def _rise(start, steps=60, over=0.5):
    """A box released at ``start`` that settles 0.1 higher, overshooting by
    ``over`` of the rise."""
    t = np.arange(1, steps + 1) * 0.05
    zeta = -np.log(over) / np.sqrt(np.pi**2 + np.log(over)**2) if over else 1.0
    if zeta >= 1.0:
        return start + 0.1 * (1.0 - np.exp(-4.0 * t) * (1.0 + 4.0 * t))
    root = np.sqrt(1.0 - zeta**2)
    return start + 0.1 - 0.1 * np.exp(-zeta * 4.0 * t) * (
        np.cos(4.0 * root * t) + zeta / root * np.sin(4.0 * root * t))


def _segments(starts, truths, seed, sigma=_SIGMA):
    """Recordings of ``truths`` read with ``sigma`` of noise, each starting at
    rest at its own height."""
    rng = np.random.default_rng(seed)
    segments = []
    for start, truth in zip(starts, truths):
        recorded = truth + sigma * rng.standard_normal(truth.size)
        segments.append(
            (_states([start] + list(recorded)),
             [Action(np.zeros(1, dtype=np.float32)) for _ in recorded]))
    return segments


def _replays(monkeypatch, starts, models):
    """Replays that put the box at ``models[i]`` in the segment that starts at
    ``starts[i]``."""
    by_start = {round(start, 6): model for start, model in zip(starts, models)}

    def _rollout(_env, init_state, _actions, _params, post_step=None):
        del post_step
        return _states(by_start[round(float(init_state.get(_BOX, "z")), 6)])

    monkeypatch.setattr(rollout_objective, "rollout_states", _rollout)


def _config(noise):
    utils.reset_config({"env": "cover"})
    return dataclasses.replace(SysIdConfig.from_cfg(), observation_noise=noise)


def test_a_replay_that_stops_short_of_the_recorded_peaks_is_named(monkeypatch):
    """Three recorded releases overshoot by half the rise; the fitted model
    settles at the same height without overshooting.

    The check names the box's highest value, short in all 3 segments by
    about the overshoot.
    """
    starts = (0.40, 0.45, 0.50)
    segments = _segments(starts, [_rise(s) for s in starts], seed=0)
    _replays(monkeypatch, starts, [_rise(s, over=0.0) for s in starts])
    gaps = reach_gaps(None,
                      segments, {"k": 0.0},
                      _FEATURES, ["k"],
                      config=_config(ObservationNoise(position=_SIGMA)))
    assert [(g.type_name, g.feature, g.side)
            for g in gaps] == [("box", "z", "highest")]
    gap, = gaps
    assert (gap.short, gap.moving) == (3, 3)
    # The overshoot is 0.05; read through the noise, the replay's long
    # plateau reaches about 2 sigma above its level.
    assert 0.025 < gap.gap < 0.06
    lines = describe_reach(gaps, ["recording 0, steps 1-60"] * 3)
    assert "box z: the highest value stops short of the recording's in 3 " \
           "of 3 segments" in lines[1]
    assert "recording 0, steps 1-60" in lines[1]
    reminder = reach_reminder(gaps)
    assert reminder is not None and "box z highest by up to" in reminder


def test_a_matching_replay_is_not_named_under_noise(monkeypatch):
    """Twenty noisy recordings of what the model replays: the noise alone
    leaves the recorded extremes outside the replayed ones, and the check,
    which reads the replay through the same noise, names nothing."""
    starts = tuple(0.40 + 0.01 * i for i in range(20))
    truths = [_rise(s) for s in starts]
    segments = _segments(starts, truths, seed=1)
    _replays(monkeypatch, starts, truths)
    assert not reach_gaps(None,
                          segments, {"k": 0.0},
                          _FEATURES, ["k"],
                          config=_config(ObservationNoise(position=_SIGMA)))
    assert reach_reminder([]) is None
    assert not describe_reach([])


def test_a_body_at_rest_is_not_checked(monkeypatch):
    """A box that never moves past its noise is not checked, even where the
    model holds it 2 cm off."""
    segments = _segments([0.40], [np.full(60, 0.40)], seed=2)
    _replays(monkeypatch, [0.40], [np.full(60, 0.38)])
    assert not reach_gaps(None,
                          segments, {"k": 0.0},
                          _FEATURES, ["k"],
                          config=_config(ObservationNoise(position=_SIGMA)))


def test_an_exact_feature_is_short_past_a_twentieth_of_its_range(monkeypatch):
    """Without noise any gap past 5 % of the recorded range counts, in one
    segment as well."""
    segments = _segments([0.40], [_rise(0.40)], seed=3, sigma=0.0)
    _replays(monkeypatch, [0.40], [_rise(0.40, over=0.3)])
    gaps = reach_gaps(None,
                      segments, {"k": 0.0},
                      _FEATURES, ["k"],
                      config=_config(None))
    gap, = gaps
    assert (gap.side, gap.short, gap.moving) == ("highest", 1, 1)


def test_a_rehearsal_reads_the_reach_of_the_fit_of_the_file_as_it_is(tmp_path):
    """The reminder a canonical fit publishes is served while ``simulator.py``
    is the file that fit ran on, and dropped after an edit."""
    utils.reset_config({})
    approach = object.__new__(AgentContinualApproach)
    approach._fitted_params = {}
    approach._tool_context = ToolContext()
    simulator_file = str(tmp_path / "simulator.py")
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write("RESIDUAL_FEATURES = {'box': ['z']}\n")
    setattr(approach, "_resolve_synthesis_paths",
            lambda: SimpleNamespace(simulator_file=simulator_file))
    approach._publish_probe_fit({"k": 1.0},
                                "cycle_000_vers_001",
                                simulator_file,
                                reach="REACH")
    assert approach._reach_note() == "REACH"
    with open(simulator_file, "a", encoding="utf-8") as f:
        f.write("# edited\n")
    assert approach._reach_note() is None
    approach._publish_probe_fit({"k": 1.0}, "cycle_000_vers_002",
                                simulator_file)
    assert approach._reach_note() is None
