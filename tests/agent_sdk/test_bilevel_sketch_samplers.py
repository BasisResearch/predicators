"""Tests for per-skill synthesized samplers in bilevel_sketch refinement.

Verifies that a sampler registered under an option name in
``option_samplers`` is consulted (with the step's subgoal + objects +
the option's params box) to draw that option's continuous params during
refinement — on both the plain and info-seeking paths — and that a
missing / misbehaving sampler falls back to uniform sampling so
refinement is byte-for-byte unchanged when no usable sampler is
supplied.
"""

# pylint: disable=unused-import

import numpy as np
from gym.spaces import Box

from predicators import utils  # noqa: F401  (settles import order)
from predicators.agent_sdk import bilevel_sketch
from predicators.agent_sdk.bilevel_sketch import SketchStep, sample_params
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


# A 1-D option whose parameter becomes the post-state x of the block.
_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=_true,
    terminal=_false,
)


class _FakeOptionModel:
    """Deterministic model: Move sets block.x to its parameter value."""

    last_execution_failure = None

    def __init__(self):
        self.num_calls = 0

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        return nxt, 1


# Subgoal uniform sampling hits only ~10% of the time (x >= 0.9), but a
# targeted sampler lands on the first draw.
_ReachedHi = Predicate("ReachedHi", [_block_type],
                       lambda s, o: s.get(o[0], "x") >= 0.9)
# Always-true subgoal so the first draw (uniform or sampled) is accepted.
_Reached = Predicate("Reached", [_block_type], lambda s, o: True)


def _task_hi():
    init = State({_block: np.array([0.0], dtype=np.float32)})
    return Task(init, {GroundAtom(_ReachedHi, [_block])})


def _sketch_hi():
    return [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_ReachedHi, [_block])})
    ]


def _easy_task_and_sketch():
    sketch = [
        SketchStep(option=_Move,
                   objects=[_block],
                   subgoal_atoms={GroundAtom(_Reached, [_block])})
    ]
    task = Task(State({_block: np.array([0.0], dtype=np.float32)}),
                {GroundAtom(_Reached, [_block])})
    return task, sketch


def test_registered_sampler_is_used():
    """A targeted sampler lands the hard subgoal on the first sample."""
    calls = []

    def sampler(state, subgoal_atoms, rng, objects):
        del state, rng
        calls.append((objects, subgoal_atoms))
        return np.array([0.95], dtype=np.float32)

    model = _FakeOptionModel()
    plan, success, total = bilevel_sketch.refine_sketch(
        _task_hi(),
        _sketch_hi(),
        model,
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        option_samplers={"Move": sampler})
    assert success
    assert np.isclose(float(plan[0].params[0]), 0.95)
    # Feasible on the very first attempt — none of the uniform churn.
    assert total == 1
    assert model.num_calls == 1
    # The sampler saw the right subgoal and objects.
    objs, subgoal = calls[0]
    assert [o.name for o in objs] == ["block0"]
    assert GroundAtom(_ReachedHi, [_block]) in subgoal


def test_missing_entry_falls_back_to_uniform():
    """A sampler keyed by another option leaves Move on the uniform path."""
    seed = 7
    first = float(sample_params(_Move, np.random.default_rng(seed))[0])
    task, sketch = _easy_task_and_sketch()

    def other(*_args):
        raise AssertionError("sampler for a different option was called")

    plan, success, _ = bilevel_sketch.refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={_Reached},
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        option_samplers={"OtherOption": other})
    assert success
    # Identical to the no-sampler uniform draw.
    assert float(plan[0].params[0]) == first


def test_bad_shape_falls_back_to_uniform():
    """A wrong-shaped return is rejected; uniform sampling still succeeds."""
    task, sketch = _easy_task_and_sketch()

    def bad(*_args):
        return np.array([0.5, 0.5], dtype=np.float32)  # shape (2,) != (1,)

    plan, success, _ = bilevel_sketch.refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={_Reached},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        option_samplers={"Move": bad})
    assert success
    assert 0.0 <= float(plan[0].params[0]) <= 1.0


def test_raising_sampler_falls_back_to_uniform():
    """A sampler that raises is caught and uniform sampling proceeds."""
    task, sketch = _easy_task_and_sketch()

    def boom(*_args):
        raise ValueError("nope")

    _, success, _ = bilevel_sketch.refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={_Reached},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        option_samplers={"Move": boom})
    assert success


def test_none_samplers_unchanged():
    """option_samplers=None reproduces the plain first-uniform-draw param."""
    seed = 7
    first = float(sample_params(_Move, np.random.default_rng(seed))[0])
    task, sketch = _easy_task_and_sketch()
    plan, success, _ = bilevel_sketch.refine_sketch(
        task,
        sketch,
        _FakeOptionModel(),
        predicates={_Reached},
        timeout=10.0,
        rng=np.random.default_rng(seed),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        option_samplers=None)
    assert success
    assert float(plan[0].params[0]) == first


def test_sampler_used_on_info_seeking_path():
    """The info-seeking draw loop also routes through the sampler."""

    def sampler(_s, _a, rng, _o):
        # Jitter so candidates differ but all clear the x>=0.9 subgoal.
        return np.array([0.9 + 0.05 * rng.random()], dtype=np.float32)

    model = _FakeOptionModel()
    plan, success, _ = bilevel_sketch.refine_sketch(
        _task_hi(),
        _sketch_hi(),
        model,
        predicates={_ReachedHi},
        timeout=10.0,
        rng=np.random.default_rng(0),
        max_samples_per_step=50,
        check_subgoals=True,
        check_final_goal=False,
        info_scorer=lambda s, _a: s.get(_block, "x"),
        info_n_feasible_target=4,
        option_samplers={"Move": sampler})
    assert success
    # Every pooled candidate came from the sampler => satisfies x >= 0.9.
    assert float(plan[0].params[0]) >= 0.9
