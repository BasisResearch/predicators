"""Hidden object types (CFG.excluded_objects_in_state_str) stay out of every
listing the agent reads: dict_str, pretty_str and the task digest."""
from typing import Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk.tools.digests import render_task_digest
from predicators.structs import Object, State, Task, Type, \
    excluded_object_type_names


def _scene() -> Tuple[State, Object, Object]:
    fan = Type("fan", ["x", "is_on"])
    switch = Type("switch", ["x", "is_on"])
    fan_1, switch_1 = Object("fan_1", fan), Object("switch_1", switch)
    state = State({
        fan_1: np.array([0.5, 0.0]),
        switch_1: np.array([0.6, 0.0]),
    })
    return state, fan_1, switch_1


def test_listings_hide_excluded_types() -> None:
    """With ``switch`` excluded, no listing names switch_1; the object is still
    in the State for skills and simulators."""
    utils.reset_config({"excluded_objects_in_state_str": "switch"})
    assert excluded_object_type_names() == {"switch"}
    state, fan_1, switch_1 = _scene()
    assert switch_1 in state
    assert "switch_1" not in state.dict_str()
    assert "switch" not in state.pretty_str()
    assert "fan_1" in state.pretty_str()
    digest = render_task_digest(Task(state, set()), 0, [])
    assert "Objects: [fan_1:fan]" in digest
    assert "switch_1" not in digest
    del fan_1


def test_listings_show_everything_by_default() -> None:
    """The default (empty) exclusion lists every object."""
    utils.reset_config({})
    assert excluded_object_type_names() == set()
    state, _, _ = _scene()
    assert "switch_1" in state.dict_str()
    assert "switch_1" in state.pretty_str()
    assert "switch_1:switch" in render_task_digest(Task(state, set()), 0, [])
