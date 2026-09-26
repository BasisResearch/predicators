"""Tests for loading the agent's ``GROUND_SAMPLERS``, the named samplers a
sketch step references with ``~ name``."""

import numpy as np
from gym.spaces import Box

from predicators.agent_sdk.proposal_exec import build_exec_context, \
    load_ground_samplers
from predicators.structs import Action, Object, ParameterizedOption, \
    Predicate, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)

_Reached = Predicate("Reached", [_block_type], lambda s, o: True)

_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=lambda _s, _m, _o, _p: Action(np.zeros(1, dtype=np.float32)),
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)


def test_load_ground_samplers_happy_and_bad_entries():
    """GROUND_SAMPLERS loads callables; bad keys/values warn and drop."""
    ctx = build_exec_context(types={_block_type},
                             predicates={_Reached},
                             options={_Move})
    code = """\
def _fn(state, subgoal_atoms, rng, objects):
    del state, subgoal_atoms, objects
    return np.array([0.5], dtype=np.float32)

GROUND_SAMPLERS = {"hi_band": _fn, "not-an-identifier": _fn, "seven": 7}
"""
    fns, warnings, err = load_ground_samplers(code, ctx)
    assert err is None
    assert set(fns) == {"hi_band"}
    assert len(warnings) == 2
    assert any("identifiers" in w for w in warnings)
    assert any("not callable" in w for w in warnings)


def test_load_ground_samplers_errors():
    """Exec failures and non-dict bindings load nothing, with an error."""
    ctx = build_exec_context(types={_block_type},
                             predicates={_Reached},
                             options={_Move})
    fns, _, err = load_ground_samplers("raise RuntimeError('boom')", ctx)
    assert not fns
    assert err is not None and "boom" in err
    ctx = build_exec_context(types={_block_type},
                             predicates={_Reached},
                             options={_Move})
    fns, _, err = load_ground_samplers("GROUND_SAMPLERS = [1]", ctx)
    assert not fns
    assert err is not None and "must be a dict" in err
