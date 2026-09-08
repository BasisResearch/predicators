"""Tests for the interval-first parameter belief
(``code_sim_learning_interval_belief``, docs/continual-uncertainty.md 3.7).

The verdict switch discarded a moved-but-wide posterior for the anchor
(domino at 1 cm noise: friction 0.40 with interval [0.22, 0.72] reverted
to the 0.1 anchor, outside the interval, and never swept). Under the
interval belief such a parameter reads ``Verdict.WIDE``: deployed, swept
end to end, rendered in words with the anchor's position, and a mixed
certification sweep is summarised as the interval straddling the plan's
success boundary.
"""

import dataclasses

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.active_experiment import \
    mean_bernoulli_entropy, noisy_read_information
from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
from predicators.code_sim_learning.grid_seed import flat_tolerance
from predicators.code_sim_learning.identifiability import Verdict, \
    format_identifiability, identifiability_report, physics_sigma_points, \
    select_trustworthy_params, straddle_summary
from predicators.code_sim_learning.trajectory_prep import ResidualScaling, \
    expected_noise_sse
from predicators.observation_noise import ObservationNoise
from predicators.structs import Action, Object, State, Type


def _wide_result(flat_interval=(0.5, 2.0)):
    """A single-row friction fit at 0.6 whose landscape interval leaves the
    posterior at ~0.92 of the prior (NOT identified under the switch)."""
    result = FitResult(names=["friction"],
                       samples=np.array([[0.6]], dtype=float),
                       log_probs=np.zeros(1),
                       jacobian=None,
                       noise_sigma=0.05,
                       prior_sigma=np.array([0.75]))
    result.sensitivity = {
        "friction": {
            "sse_span": 100.0,
            "noise_floor": 0.0,
            "sensitive": True,
            "flat_interval": flat_interval,
        }
    }
    return result


_SPECS = [ParamSpec("friction", 0.6, lo=0.01, hi=2.0, scale="log")]


def _no_probe(_params):
    raise AssertionError("a swept parameter must not be probed")


def test_moved_wide_posterior_is_deployed_and_swept():
    """A NOT-identified-by-contraction parameter the data moved off its anchor
    reads WIDE under the interval belief: applied, and the physics sweep spans
    its whole interval."""
    report = identifiability_report(_wide_result(),
                                    _no_probe,
                                    _SPECS,
                                    num_explainable=3,
                                    belief_interval=True,
                                    anchors={"friction": 0.1})
    entry = report["friction"]
    assert entry["verdict"] is Verdict.WIDE
    assert 0.9 < entry["contraction"] < 1.0
    lo, hi = entry["belief_interval"]
    assert lo == pytest.approx(0.6 * np.exp(-entry["posterior_std"]))
    assert hi == pytest.approx(0.6 * np.exp(entry["posterior_std"]))
    assert entry["map"] == 0.6
    assert entry["anchor"] == 0.1
    applied = select_trustworthy_params({"friction": 0.6}, {"friction": 0.6},
                                        ["friction"],
                                        report,
                                        anchors={"friction": 0.1})
    assert applied == {"friction": 0.6}
    points = physics_sigma_points(applied, report, _SPECS, num_points=3)
    values = sorted(p["friction"] for p in points)
    assert values[0] == pytest.approx(lo)
    assert values[-1] == pytest.approx(hi)


def test_wide_needs_a_move_and_information():
    """The anchor stands when the MAP never left it, when the width exceeds the
    prior (the data carried nothing), and under the legacy switch."""
    unmoved = identifiability_report(_wide_result(),
                                     _no_probe,
                                     _SPECS,
                                     num_explainable=3,
                                     belief_interval=True,
                                     anchors={"friction": 0.6})
    assert unmoved["friction"]["verdict"] is Verdict.NOT_IDENTIFIED
    # The belief fields are still reported for the agent's benefit.
    assert "belief_interval" in unmoved["friction"]
    # A box-wide landscape: the reported width is wider than the prior.
    nothing = identifiability_report(_wide_result((0.01, 2.0)),
                                     _no_probe,
                                     _SPECS,
                                     num_explainable=3,
                                     belief_interval=True,
                                     anchors={"friction": 0.1})
    assert nothing["friction"]["verdict"] is Verdict.NOT_IDENTIFIED
    assert nothing["friction"]["contraction"] > 1.0
    legacy = identifiability_report(_wide_result(),
                                    _no_probe,
                                    _SPECS,
                                    num_explainable=3,
                                    anchors={"friction": 0.1})
    assert legacy["friction"]["verdict"] is Verdict.NOT_IDENTIFIED
    assert "belief_interval" not in legacy["friction"]
    assert not physics_sigma_points({"friction": 0.1}, legacy, _SPECS)


def test_belief_line_names_the_anchor_position():
    """The report says the interval in words and where the anchor sits."""
    report = identifiability_report(_wide_result(),
                                    _no_probe,
                                    _SPECS,
                                    num_explainable=3,
                                    belief_interval=True,
                                    anchors={"friction": 0.1})
    text = format_identifiability(report)
    assert "wide posterior" in text
    assert "belief: most likely 0.6, interval [0.3, 1.2]" in text
    assert "the anchor 0.1 lies BELOW the interval" in text
    assert "deployed as the planner's value" in text
    inside = identifiability_report(_wide_result(),
                                    _no_probe,
                                    _SPECS,
                                    num_explainable=3,
                                    belief_interval=True,
                                    anchors={"friction": 0.5})
    text_inside = format_identifiability(inside)
    assert "the anchor 0.5 lies inside the interval" in text_inside
    unmoved = identifiability_report(_wide_result(),
                                     _no_probe,
                                     _SPECS,
                                     num_explainable=3,
                                     belief_interval=True,
                                     anchors={"friction": 0.6})
    assert "NOT deployed (NOT identified)" in format_identifiability(unmoved)


def test_straddle_summary_names_passing_and_failing_ranges():
    """A mixed sweep is described per parameter as contiguous runs."""
    points = [{
        "friction": v,
        "restitution": 0.02
    } for v in (0.2, 0.3, 0.4, 0.5, 0.6)]
    passed = [False, False, True, True, False]
    text = straddle_summary(points, passed)
    # restitution does not vary across the sweep, so it is omitted.
    assert text == ("friction: fails on [0.2, 0.3], passes on [0.4, 0.5], "
                    "fails at 0.6")
    assert straddle_summary([], []) == ""


def test_flat_tolerance_measures_the_excess_over_noise():
    """The relative tolerance applies to the SSE above the expected noise
    floor, with the likelihood-ratio term as the floor."""
    # Legacy: 5% of the best SSE.
    assert flat_tolerance(2.0, 0.0, 0.05) == pytest.approx(0.1)
    # Interval belief: the same fit whose SSE is mostly noise.
    assert flat_tolerance(2.0, 0.0, 0.05, noise_sse=1.9,
                          sigma_tol=0.0025) == pytest.approx(0.005)
    # A model that explains everything but the noise: the floor stands.
    assert flat_tolerance(1.9, 0.0, 0.05, noise_sse=2.5,
                          sigma_tol=0.0025) == pytest.approx(0.0025)
    # The same-theta noise floor still wins when it is larger.
    assert flat_tolerance(2.0, 0.3, 0.05, noise_sse=1.9,
                          sigma_tol=0.0025) == pytest.approx(0.3)


def test_expected_noise_sse_counts_scored_residuals():
    """One noise variance per object, in-scope noisy feature and rolled-out
    step (plus the endpoint summary at its weight)."""
    utils.reset_config({})
    block = Type("block", ["x", "y", "z", "lit"])
    robot = Type("robot", ["x"])
    b0, b1, r = Object("b0", block), Object("b1", block), Object("r", robot)

    def _state():
        return State({
            b0: np.zeros(4, dtype=np.float32),
            b1: np.zeros(4, dtype=np.float32),
            r: np.zeros(1, dtype=np.float32),
        })

    states = [_state() for _ in range(4)]  # 3 rolled-out steps
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(3)]
    features = {"block": ["x", "z", "lit"], "robot": ["x"]}
    scaling = ResidualScaling(angular=frozenset(),
                              scales={
                                  ("block", "x"): 0.1,
                                  ("block", "z"): 0.2,
                                  ("block", "lit"): 1.0,
                                  ("robot", "x"): 0.1,
                              })
    base = SysIdConfig.from_cfg()
    config = dataclasses.replace(base,
                                 observation_noise=ObservationNoise(
                                     position=0.01, orientation=0.0),
                                 summary_weight=0.0)
    total = expected_noise_sse([(states, actions)], features, scaling, config)
    # 2 blocks x 3 steps x ((0.01/0.1)^2 + (0.01/0.2)^2); "lit" is exact
    # and the robot reads itself exactly.
    assert total == pytest.approx(2 * 3 * (0.01 + 0.0025))
    weighted = dataclasses.replace(config, summary_weight=2.0)
    assert expected_noise_sse([(states, actions)], features, scaling,
                              weighted) == pytest.approx(2 * 5 *
                                                         (0.01 + 0.0025))
    # No channel, or no residual scaling: nothing to subtract.
    assert expected_noise_sse([(states, actions)], features, scaling,
                              base) == 0.0
    assert expected_noise_sse([(states, actions)], features, None,
                              config) == 0.0


def test_noisy_read_information_reduces_to_entropy_when_reads_are_sure():
    """Sure reads recover the Bernoulli entropy; coin-flip reads score 0."""
    sure = np.array([[1.0], [0.0], [1.0]])
    assert noisy_read_information(sure) == pytest.approx(
        mean_bernoulli_entropy(sure.astype(bool)))
    # Every member reads the atom as a coin flip: one observation cannot
    # tell them apart.
    assert noisy_read_information(np.array([[0.5], [0.5], [0.5]])) == 0.0
    # Two atoms: a resolvable one and an unresolvable one average.
    both = np.array([[1.0, 0.5], [0.0, 0.5]])
    assert noisy_read_information(both) == pytest.approx(0.5)
    assert noisy_read_information(np.zeros((0, 0))) == 0.0
    with pytest.raises(ValueError):
        noisy_read_information(np.zeros(3))
