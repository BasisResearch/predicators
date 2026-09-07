"""Tests for ``refine_sketch(pin_proposed_params=True)``: a step's proposed
continuous parameters are a decision, not a seed."""
import numpy as np
from gym.spaces import Box

from predicators import utils  # noqa: F401  pylint: disable=unused-import
from predicators.agent_sdk.sketch_refinement import refine_sketch
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


def _true(_s, _m, _o, _p):
    return True


def _false(_s, _m, _o, _p):
    return False


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)
_Reached = Predicate("Reached", [_block_type],
                     lambda s, o: s.get(o[0], "x") > 0.5)
_PREDICATES = {_Reached}


class _CountingModel:
    """Move sets block.x to its parameter; records every proposed param."""
    last_execution_failure = None

    def __init__(self):
        self.params_seen = []

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step."""
        self.params_seen.append(float(option.params[0]))
        nxt = state.copy()
        nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def _task():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(_Reached, [_block])})


def _refine(model, proposal, pin, retries=3, info_scorer=None):
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Reached, [_block])},
                   initial_params=(None if proposal is None else np.array(
                       [proposal], dtype=np.float32)))
    ]
    return refine_sketch(_task(),
                         sketch,
                         model,
                         predicates=_PREDICATES,
                         timeout=10.0,
                         rng=np.random.default_rng(0),
                         max_samples_per_step=20,
                         check_subgoals=True,
                         check_final_goal=False,
                         info_scorer=info_scorer,
                         info_n_feasible_target=4 if info_scorer else 1,
                         pin_proposed_params=pin,
                         pinned_step_retries=retries)


def test_pinned_proposal_is_retried_never_resampled():
    """A failing pinned proposal is re-proposed ``pinned_step_retries`` times
    and the search then gives up without sampling a replacement."""
    model = _CountingModel()
    outcome = _refine(model, proposal=0.2, pin=True, retries=3)
    assert not outcome.success
    np.testing.assert_allclose(model.params_seen, [0.2, 0.2, 0.2], rtol=1e-6)


def test_unpinned_proposal_is_a_seed_then_searched():
    """Without pinning the proposal is tried once and the step is sampled."""
    model = _CountingModel()
    outcome = _refine(model, proposal=0.2, pin=False)
    assert outcome.success
    assert abs(model.params_seen[0] - 0.2) < 1e-6
    assert len(model.params_seen) > 1 and outcome.plan[0].params[0] > 0.5


def test_pinned_proposal_that_works_executes_verbatim():
    """A feasible pinned proposal is the plan, with exactly one rollout."""
    model = _CountingModel()
    outcome = _refine(model, proposal=0.7, pin=True)
    assert outcome.success
    np.testing.assert_allclose(model.params_seen, [0.7], rtol=1e-6)
    assert abs(float(outcome.plan[0].params[0]) - 0.7) < 1e-6


def test_pinned_steps_are_not_info_seeking_probes():
    """With an info scorer wired, a pinned step is never pooled: the proposal
    runs once, no candidate draws."""
    model = _CountingModel()
    outcome = _refine(model,
                      proposal=0.7,
                      pin=True,
                      info_scorer=lambda _s, _atoms: 1.0)
    assert outcome.success
    np.testing.assert_allclose(model.params_seen, [0.7], rtol=1e-6)
    assert outcome.total_pool_rollouts == 0


def test_unspecified_step_is_still_searched_under_pinning():
    """Pinning only touches steps that carry a proposal."""
    model = _CountingModel()
    outcome = _refine(model, proposal=None, pin=True)
    assert outcome.success
    assert outcome.plan[0].params[0] > 0.5
