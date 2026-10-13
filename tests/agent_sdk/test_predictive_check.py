"""Tests for the predictive check of a request's outcome against its
rehearsal's draws, and for the temperature it keeps per fitted program."""

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest
from scipy.stats import t as student_t

from predicators.agent_sdk.predictive_check import MAX_INFLATION, \
    check_prediction
from predicators.agent_sdk.tools.context import ToolContext
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.parameter_belief import BeliefConfig, \
    ParameterBelief, prior_belief
from predicators.observation_noise import ObservationNoise
from predicators.structs import Object, State, Type

_BOX = Type("box", ["x", "yaw", "colour"], angular_features=("yaw", ))
_ROBOT = Type("robot", ["x"])
_BOX0 = Object("box0", _BOX)
_ROBOT0 = Object("robot", _ROBOT)
_FEATURES = {"box": ["x", "yaw"], "robot": ["x"]}
_QUIET = ObservationNoise(position=0.0, orientation=0.0, scalar=0.0)


def _state(x: float, yaw: float = 0.0, colour: float = 0.0) -> State:
    return State({
        _BOX0: np.array([x, yaw, colour]),
        _ROBOT0: np.array([0.0]),
    })


def _draws(xs: Sequence[float],
           yaws: Optional[Sequence[float]] = None) -> List[State]:
    yaws = [0.0] * len(xs) if yaws is None else yaws
    return [_state(x, yaw) for x, yaw in zip(xs, yaws)]


def test_an_outcome_within_the_draws_is_not_surprising() -> None:
    """An outcome inside the draws' spread passes, whatever the unscored
    features and the robot do."""
    predicted = _draws([0.50, 0.52, 0.48, 0.51])
    observed = _state(0.505, colour=9.0)
    observed.data[_ROBOT0] = np.array([5.0])
    check = check_prediction(observed, predicted, _FEATURES, _QUIET)
    assert check is not None
    assert check.num_features == 2  # x and yaw; the robot is left out
    assert check.num_draws == 4
    assert not check.surprising
    assert check.inflation == 1.0


def test_an_outcome_beyond_the_draws_names_the_feature() -> None:
    """An outcome many predictive spreads out is surprising, names the worst
    feature, and calls for the temperature that puts it two spreads out."""
    predicted = _draws([0.50, 0.52, 0.48, 0.51])
    check = check_prediction(_state(0.70), predicted, _FEATURES, _QUIET)
    assert check is not None and check.surprising
    assert check.worst == ("box0", "x")
    # A new draw spreads by (1 + 1/n) of the draws' sample variance.
    spread = np.std([0.50, 0.52, 0.48, 0.51], ddof=1) * np.sqrt(1.25)
    assert check.spread == pytest.approx(spread, rel=1e-6)
    assert check.mean == pytest.approx(0.5025)
    assert check.z_max == pytest.approx((0.70 - 0.5025) / spread, rel=1e-6)
    assert check.inflation == pytest.approx((check.z_max / 2.0)**2)
    # Four draws give the spread three degrees of freedom, and the
    # critical distance grows with the number of features scored.
    assert check.critical == pytest.approx(student_t.isf(0.01 / 4, 3))
    one = check_prediction(_state(0.70), predicted, {"box": ["x"]}, _QUIET)
    assert one is not None and one.critical < check.critical


def test_few_draws_widen_the_tails() -> None:
    """With a handful of draws their spread is itself uncertain, so an outcome
    a known-variance test would flag stays within the Student-t critical
    distance; with known observation noise dominating, the distance falls back
    toward the normal one."""
    xs = [0.50, 0.52, 0.48, 0.51]
    spread = np.std(xs, ddof=1) * np.sqrt(1.25)
    observed = _state(0.5025 + 4.0 * spread)
    check = check_prediction(observed, _draws(xs), {"box": ["x"]}, _QUIET)
    assert check is not None and not check.surprising
    assert check.z_max == pytest.approx(4.0)
    # A normal test of one feature would have flagged it at 2.576.
    assert check.critical > 4.0
    noisy = ObservationNoise(position=0.2, orientation=0.0, scalar=0.0)
    loud = check_prediction(observed, _draws(xs), {"box": ["x"]}, noisy)
    assert loud is not None
    assert loud.critical == pytest.approx(2.576, abs=0.01)


def test_observation_noise_widens_the_prediction() -> None:
    """The observation noise joins the draws' spread, so identical draws do not
    call a noisy reading surprising."""
    predicted = _draws([0.5, 0.5, 0.5])
    noisy = ObservationNoise(position=0.01, orientation=0.0, scalar=0.0)
    check = check_prediction(_state(0.51), predicted, {"box": ["x"]}, noisy)
    assert check is not None and not check.surprising
    assert check.spread == pytest.approx(0.01, rel=1e-6)
    quiet = check_prediction(_state(0.51), predicted, {"box": ["x"]}, _QUIET)
    assert quiet is not None and quiet.surprising
    # Draws that do not vary cannot be tempered toward the outcome.
    assert quiet.out_of_reach == (("box0", "x"), )
    assert quiet.inflation == 1.0


def test_a_miss_out_of_reach_leaves_the_temperature_to_the_others() -> None:
    """A feature the draws barely vary is named out of reach and does not set
    the temperature; a reachable miss beside it still does."""
    xs = [0.50, 0.52, 0.48, 0.51]
    yaws = [0.10, 0.10, 0.10, 0.10]
    observed = _state(0.70, yaw=0.40)
    check = check_prediction(observed, _draws(xs, yaws), _FEATURES, _QUIET)
    assert check is not None and check.surprising
    assert check.out_of_reach == (("box0", "yaw"), )
    draw_var = 1.25 * float(np.var(xs, ddof=1))
    assert check.inflation == pytest.approx(
        (0.70 - 0.5025)**2 / 4.0 / draw_var)
    assert check.inflation < MAX_INFLATION


def test_the_temperature_widens_only_the_draws() -> None:
    """Tempering widens the draws' spread and leaves the observation noise as
    it is, so the factor puts the outcome exactly two predictive spreads out
    once the draws' variance is multiplied by it."""
    noisy = ObservationNoise(position=0.01, orientation=0.0, scalar=0.0)
    xs = [0.5 + d for d in (-0.012, -0.004, 0.004, 0.012)]
    predicted = _draws(xs)
    check = check_prediction(_state(0.6), predicted, {"box": ["x"]}, noisy)
    assert check is not None and check.surprising
    draw_var = 1.25 * float(np.var(xs, ddof=1))
    spread = np.sqrt(check.inflation * draw_var + 0.01**2)
    assert abs(0.6 - check.mean) / spread == pytest.approx(2.0)
    # Treating the whole spread as the draws' would call for less.
    assert check.inflation > (check.z_max / 2.0)**2


def test_angles_compare_on_the_circle() -> None:
    """Draws around +-pi predict an outcome at pi, not one at 0."""
    yaws = [np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02, -np.pi + 0.02]
    predicted = _draws([0.5] * 4, yaws)
    near = check_prediction(_state(0.5, yaw=-np.pi), predicted,
                            {"box": ["yaw"]}, _QUIET)
    assert near is not None and not near.surprising
    far = check_prediction(_state(0.5, yaw=0.0), predicted, {"box": ["yaw"]},
                           _QUIET)
    assert far is not None and far.surprising


def test_nothing_to_score() -> None:
    """With fewer than two draws or no scored feature there is no check."""
    assert check_prediction(_state(0.5), _draws([0.5]), _FEATURES,
                            _QUIET) is None
    assert check_prediction(_state(0.5), _draws([0.5, 0.6]), {"robot": ["x"]},
                            _QUIET) is None


class _Arm(AgentContinualApproach):
    """The continual arm's predictive temperature, on a given belief and the
    fitted program it came from."""

    # pylint: disable-next=super-init-not-called
    def __init__(self, belief: ParameterBelief, fitted: Optional[str]):
        self._tool_context = ToolContext()
        self.sourced: Tuple[Optional[ParameterBelief],
                            Optional[str]] = (belief, fitted)

    def _sourced_parameter_belief(
            self) -> Tuple[Optional[ParameterBelief], Optional[str]]:
        return self.sourced

    def inflation(self) -> float:
        """The factor the belief it hands out was tempered by."""
        belief = self.parameter_belief()
        assert belief is not None
        return belief.inflation

    def basis(self) -> Optional[Tuple[str, float]]:
        """The program and temperature a rehearsal would record."""
        return self._prediction_basis()

    def set_floor(self, program: str, temperature: float) -> None:
        """What the predictive check records after a surprise."""
        self._tool_context.predictive_program = program
        self._tool_context.predictive_temperature = temperature


def test_the_temperature_floor_belongs_to_the_fitted_program() -> None:
    """The fitted program's belief runs at no less than the temperature the
    check found it needs; a refit that runs hotter keeps its own, the prior
    never changes, and an edit of the program starts over."""
    belief = prior_belief([ParamSpec("a", 0.5, lo=0.0, hi=1.0)], {"a": 0.5},
                          {"a": 0.05},
                          BeliefConfig(num_draws=16),
                          seed=0)
    arm = _Arm(belief, "prog")
    assert arm.parameter_belief() is belief
    assert arm.basis() == ("prog", 1.0)
    arm.set_floor("prog", 4.0)
    assert arm.parameter_belief() is not belief
    assert arm.inflation() == pytest.approx(4.0)
    assert arm.basis() == ("prog", 4.0)
    # A refit that runs cooler than the floor is brought up to it; one
    # that runs hotter keeps its own temperature.
    arm.sourced = (dataclasses.replace(belief, noise_scale=2.0), "prog")
    assert arm.inflation() == pytest.approx(2.0)
    assert arm.basis() == ("prog", 4.0)
    hot = dataclasses.replace(belief, noise_scale=5.0)
    arm.sourced = (hot, "prog")
    assert arm.parameter_belief() is hot
    assert arm.basis() == ("prog", 5.0)
    # The prior, and an edited program, run as they are.
    arm.sourced = (belief, None)
    assert arm.parameter_belief() is belief
    assert arm.basis() is None
    arm.sourced = (belief, "edited")
    assert arm.parameter_belief() is belief
    assert arm.basis() == ("edited", 1.0)
