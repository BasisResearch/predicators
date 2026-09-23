"""Evaluator gating tests for ``sim.refine(require_solved=True)``.

Drives the probe with a fake option model (no PyBullet). When the
task has an evaluator, refinement success is gated on its scoring: a
parameterization that reaches the goal atoms but scores as a non-solve
is discarded and the search keeps sampling; if no candidate is ever
certified the result is a FAILURE. The report speaks only in verdict
terms - the certificate's reason strings never reach the agent.
"""

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools import ToolContext
from predicators.envs.pybullet_domino.cascade_probe import _apply_process_step
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, TaskEvaluator, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)

_ReachedHi = Predicate("ReachedHi", [_block_type],
                       lambda s, o: s.get(o[0], "x") >= 0.9)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)

_SKETCH_TEXT = "Move(block0:block)[] -> {ReachedHi(block0:block)}"


class _Model:
    """Fake option model: Move sets block.x to its parameter value."""

    last_execution_failure = None

    def __init__(self):
        self.num_calls = 0
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


class _BandEvaluator(TaskEvaluator):
    """Certifies only rollouts whose final x lands in [lo, hi].

    Emulates a legitimacy band: goal-reaching parameterizations outside
    the band are reward hacks (terminated without the success bonus).
    """

    def __init__(self, goal, lo, hi):
        super().__init__(goal)
        self._lo = lo
        self._hi = hi

    def _certify(self, states, step_options, sim_env=None):
        x = states[-1].get(_block, "x")
        if self._lo <= x <= self._hi:
            return True, ""
        return False, "band: outside the certified interval"


def _make_probe(evaluator, model=None):
    utils.reset_config({
        # 200 uniform draws in [0, 1] make "no goal-reaching draw at all"
        # (p = 0.1 each) a 1e-9 event, so the certified/rejected paths
        # below are exercised at every probe seed, not just lucky ones.
        "agent_bilevel_max_samples_per_step": 200,
        "agent_bilevel_use_llm_initial_params": False,
    })
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    task = Task(init, goal, evaluator=evaluator)
    if model is None:
        model = _Model()
    ctx = ToolContext(
        types={_block_type},
        predicates={_ReachedHi},
        processes=set(),
        options={_Move},
        train_tasks=[task],
        example_state=init,
        option_model=model,
        current_task=task,
    )
    return BeliefProbe(ctx).reset(task_idx=0)


def _run_refine(evaluator, model=None):
    return _make_probe(evaluator, model).refine(_SKETCH_TEXT,
                                                timeout=10,
                                                require_goal=True,
                                                require_solved=True)


def test_certified_refinement_reports_success():
    """A wide certification band accepts the first goal-reaching params."""
    goal = {GroundAtom(_ReachedHi, [_block])}
    res = _run_refine(_BandEvaluator(goal, 0.9, 1.0))
    text = str(res)
    assert res.success
    assert "SUCCESS" in text
    assert "evaluator-solved" in res.verdict
    assert 0.9 <= float(res.plan_lines[0].split("[")[1].split("]")[0])
    assert "legitimate" not in text


def test_all_non_solve_attempts_demote_to_failure():
    """An empty certification band makes every goal-reaching rollout a
    non-solve: the search fails, no params are certified, and no reason
    string leaks."""
    goal = {GroundAtom(_ReachedHi, [_block])}
    res = _run_refine(_BandEvaluator(goal, 2.0, 3.0))
    text = str(res)
    assert not res.success
    assert "FAILURE" in text
    assert not res.verdict
    assert "band: outside the certified interval" not in text
    assert "legitimate" not in text


class _RejectFirstParamEvaluator(TaskEvaluator):
    """Rejects every rollout that ends at the first final-x it ever saw,
    certifies rollouts ending anywhere else.

    Keyed on rollout content, not call count: ``reward()`` re-invokes
    ``_certify`` within one scoring pass, so a call counter would give
    inconsistent verdicts inside a single evaluation. Deterministically
    exercises the resample path: the first candidate reaches the goal
    atoms but scores as a non-solve, a later draw (a different accepted
    parameter) is certified.
    """

    def __init__(self, goal):
        super().__init__(goal)
        self._rejected_x = None

    def _certify(self, states, step_options, sim_env=None):
        x = round(float(states[-1].get(_block, "x")), 6)
        if self._rejected_x is None:
            self._rejected_x = x
        if x == self._rejected_x:
            return False, "stub: first parameterization rejected"
        return True, ""


def test_non_solve_attempt_recovered_by_resampling():
    """A discarded first candidate is resampled past; the result is a certified
    SUCCESS that still withholds the reason string."""
    goal = {GroundAtom(_ReachedHi, [_block])}
    res = _run_refine(_RejectFirstParamEvaluator(goal))
    text = str(res)
    assert res.success
    assert "evaluator-solved" in res.verdict
    assert res.total_samples >= 2
    assert "stub: first parameterization rejected" not in text
    assert "legitimate" not in text


class _BrokenEvaluator(TaskEvaluator):
    """An infrastructure failure is not a successful task certificate."""

    def _certify(self, states, step_options, sim_env=None):
        raise NotImplementedError("internal evaluator detail")


class _CoarseModel(_Model):
    """An option-boundary trace cannot certify a causal cascade."""

    def get_next_state_and_num_actions(self, state, option):
        result = super().get_next_state_and_num_actions(state, option)
        self.last_trajectory = None
        return result


@pytest.mark.parametrize("coarse", [False, True])
def test_unavailable_certificate_never_reports_solved(coarse):
    """Exercise require_solved via the public API, including its report."""
    goal = {GroundAtom(_ReachedHi, [_block])}
    evaluator = (_BandEvaluator(goal, 0.9, 1.0)
                 if coarse else _BrokenEvaluator(goal))
    result = _run_refine(evaluator, _CoarseModel() if coarse else _Model())
    assert not result.success
    assert "evaluator-solved" not in result.verdict
    assert result.near_miss is not None
    assert "unavailable" in result.near_miss["reason"].lower()
    assert "internal evaluator detail" not in str(result)


@pytest.mark.parametrize(
    "status",
    ["accepted", "rejected", "error", "unavailable", "not_requested"])
def test_trial_evaluation_status_is_explicit(status):
    """Goal reach, task rejection and evaluator failure stay
    distinguishable."""
    goal = {GroundAtom(_ReachedHi, [_block])}
    evaluator = (_BrokenEvaluator(goal) if status == "error" else
                 _BandEvaluator(goal, 2.0 if status == "rejected" else 0.9,
                                3.0 if status == "rejected" else 1.0))
    model = _CoarseModel() if status == "unavailable" else _Model()
    probe = _make_probe(evaluator, model)
    result = probe.run("Move(block0:block)[1.0]",
                       trials=2,
                       solved=status != "not_requested")
    assert result.successes == 2
    assert all(t["evaluation_status"] == status for t in result.trials)
    expected = {"accepted": True, "rejected": False}.get(status)
    assert all(t["solved"] is expected for t in result.trials)
    if status in {"error", "unavailable"}:
        assert "NOT certified" in str(result)
        assert "internal evaluator detail" not in str(result)


def test_counterfactual_process_error_is_not_base_physics_success():
    """A broken learned rule cannot be silently removed from a certificate."""

    def broken_rule(_state, _action):
        raise RuntimeError("broken learned dynamics")

    state = State({_block: np.array([0.0], dtype=np.float32)})
    with pytest.raises(RuntimeError, match="broken learned dynamics"):
        _apply_process_step(None, broken_rule, state,
                            Action(np.zeros(1, dtype=np.float32)))
