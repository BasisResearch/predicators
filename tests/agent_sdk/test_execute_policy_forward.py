"""Tests for the closed-loop policy execution core (policy_execution).

Covers the failure-surfacing contract shared by belief validation and
the real executor: option failures surface via memory['last_failure']
and the policy is asked again; policy-code bugs and DONE end the
episode; the option cap bounds retry loops.
"""
import numpy as np
import pytest
from gym.spaces import Box

from predicators.agent_sdk.policy_execution import PolicyError, \
    build_policy_option_fn, execute_policy_forward
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

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


class _Model:
    """Move sets block.x to its parameter; optional per-call failures."""

    last_execution_failure = None

    def __init__(self, fail_calls=()):
        self.num_calls = 0
        self._fail_calls = set(fail_calls)

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        if self.num_calls in self._fail_calls:
            self.last_execution_failure = "simulated option failure"
            return state.copy(), 0
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


def _make_task():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    return Task(init, goal)


def _fn(source):
    task = _make_task()
    fn, err = build_policy_option_fn(source,
                                     task,
                                     predicates={_ReachedHi},
                                     options={_Move},
                                     types={_block_type})
    assert err is None, err
    return fn, task


_GOAL_POLICY = '''
def get_option(state, memory):
    for obj in state:
        if state.get(obj, "x") >= 0.9:
            return None
    return "Move(block0:block)[0.95]"
'''


def test_reaches_goal_then_done():
    """A goal-reaching policy stops via DONE with a clean result."""
    fn, task = _fn(_GOAL_POLICY)
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert r.goal_reached and r.policy_error is None
    assert len(r.steps) == 1 and r.goal_step_idx == 0


def test_done_before_goal_is_honest():
    """DONE before the goal ends the episode with no policy error."""
    fn, task = _fn("def get_option(state, memory):\n    return None\n")
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert not r.goal_reached and r.policy_error is None
    assert not r.steps


def test_option_cap_is_attributable():
    """The option cap converts a non-terminating policy into a failure."""
    fn, task = _fn("def get_option(state, memory):\n"
                   "    return 'Move(block0:block)[0.1]'\n")
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=3)
    assert len(r.steps) == 3
    assert r.policy_error is not None and "option budget" in r.policy_error


def test_policy_exception_is_fatal():
    """An exception inside get_option ends the episode as a policy error."""
    fn, task = _fn("def get_option(state, memory):\n"
                   "    raise RuntimeError('bug')\n")
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert r.policy_error is not None and "bug" in r.policy_error
    assert not r.steps


def test_unparsable_line_is_fatal():
    """A line that fails to parse ends the episode as a policy error."""
    fn, task = _fn("def get_option(state, memory):\n"
                   "    return 'Bogus(block0:block)[]'\n")
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert r.policy_error is not None and "unparsable" in r.policy_error


def test_stuck_loop_identical_failures_is_fatal():
    """K consecutive failures of one identical command end the episode as a
    policy bug instead of burning the whole option budget."""
    fn, task = _fn("def get_option(state, memory):\n"
                   "    return 'Move(block0:block)[0.95]'\n")
    model = _Model(fail_calls=set(range(1, 50)))
    r = execute_policy_forward(task,
                               fn,
                               model,
                               predicates={_ReachedHi},
                               max_policy_options=50)
    assert r.policy_error is not None
    assert "re-issued the same failing option" in r.policy_error
    assert len(r.steps) == 3  # the CFG default guard, not the 50 cap


def test_stuck_loop_resets_on_changed_params():
    """A policy that adapts its parameters after each failure never trips the
    stuck-loop guard, even across many consecutive failures."""
    fn, task = _fn('''
def get_option(state, memory):
    for obj in state:
        if state.get(obj, "x") >= 0.9:
            return None
    n = memory.get("n", 0)
    memory["n"] = n + 1
    return "Move(block0:block)[0.9" + str(n) + "]"
''')
    model = _Model(fail_calls={1, 2, 3, 4})
    r = execute_policy_forward(task,
                               fn,
                               model,
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert r.goal_reached and r.policy_error is None
    assert len(r.steps) == 5


def test_stuck_loop_resets_on_clean_step():
    """A success between identical failures resets the guard counter."""
    fn, task = _fn('''
def get_option(state, memory):
    n = memory.get("n", 0)
    memory["n"] = n + 1
    if n >= 6:
        return "Move(block0:block)[0.95]"
    return "Move(block0:block)[0.1]"
''')
    # Calls 1, 3, 5 fail with the identical [0.1] command, but calls 2
    # and 4 succeed in between: the consecutive count never reaches 3.
    model = _Model(fail_calls={1, 3, 5})
    r = execute_policy_forward(task,
                               fn,
                               model,
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert r.policy_error is None
    assert r.goal_reached


def test_noop_livelock_identical_completions_is_fatal():
    """K consecutive clean completions of one identical command that
    change nothing observable end the episode as a livelock, before the
    option budget is burned."""
    fn, task = _fn("def get_option(state, memory):\n"
                   "    return 'Move(block0:block)[0.5]'\n")
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=50)
    assert r.policy_error is not None
    assert "no observable state change" in r.policy_error
    # Step 1 moves x to 0.5 (real progress, counter resets); steps 2-4
    # re-complete the identical command as no-ops and the third trips
    # the guard.
    assert len(r.steps) == 4


def test_noop_livelock_resets_on_state_change():
    """Commands that keep changing the state never trip the livelock
    guard: the episode runs to the option budget instead."""
    fn, task = _fn('''
def get_option(state, memory):
    n = memory.get("n", 0)
    memory["n"] = n + 1
    return "Move(block0:block)[0." + str(2 + (n % 2)) + "]"
''')
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=6)
    assert r.policy_error is not None
    assert "option budget" in r.policy_error
    assert len(r.steps) == 6


def test_option_failure_surfaces_and_policy_recovers():
    """A failed option does not end the episode: the failure arrives in
    memory['last_failure'] and the policy retries to the goal."""
    fn, task = _fn('''
def get_option(state, memory):
    for obj in state:
        if state.get(obj, "x") >= 0.9:
            return None
    if memory.get("last_failure"):
        memory["saw_failure"] = memory["last_failure"]
    return "Move(block0:block)[0.95]"
''')
    model = _Model(fail_calls={1})
    r = execute_policy_forward(task,
                               fn,
                               model,
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert r.goal_reached and r.policy_error is None
    assert len(r.steps) == 2
    assert r.steps[0].failure_reason == "simulated option failure"
    assert r.first_failure_idx == 0


def test_memory_persists_within_episode_and_resets_across():
    """memory carries within an episode; a fresh instance starts empty."""
    src = '''
def get_option(state, memory):
    memory["n"] = memory.get("n", 0) + 1
    if memory["n"] >= 3:
        return None
    return "Move(block0:block)[0.1]"
'''
    fn, task = _fn(src)
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=10)
    assert len(r.steps) == 2  # n=1, n=2 issue options; n=3 -> DONE
    # A FRESH composed instance starts with empty memory.
    fn2, _ = _fn(src)
    r2 = execute_policy_forward(task,
                                fn2,
                                _Model(),
                                predicates={_ReachedHi},
                                max_policy_options=10)
    assert len(r2.steps) == 2


def test_state_copy_shields_executor_state():
    """A policy mutating its state argument cannot corrupt execution."""
    fn, task = _fn('''
def get_option(state, memory):
    for obj in state:
        state.set(obj, "x", 0.99)  # buggy in-place mutation
    if memory.get("n"):
        return None
    memory["n"] = 1
    return "Move(block0:block)[0.1]"
''')
    r = execute_policy_forward(task,
                               fn,
                               _Model(),
                               predicates={_ReachedHi},
                               max_policy_options=10)
    # The executor's state was never mutated: goal not reached via 0.1.
    assert not r.goal_reached


def test_load_error_reported_not_raised():
    """Broken policy source returns an agent-facing error message."""
    task = _make_task()
    fn, err = build_policy_option_fn("this is not python",
                                     task,
                                     predicates={_ReachedHi},
                                     options={_Move},
                                     types={_block_type})
    assert fn is None and err is not None and "failed to load" in err
    fn, err = build_policy_option_fn("get_option = 3",
                                     task,
                                     predicates={_ReachedHi},
                                     options={_Move},
                                     types={_block_type})
    assert fn is None and err is not None and "not callable" in err


def test_non_string_return_is_fatal():
    """A non-string, non-None return is a PolicyError."""
    fn, task = _fn("def get_option(state, memory):\n    return 42\n")
    with pytest.raises(PolicyError, match="plan-line string"):
        fn(task.init, None)
