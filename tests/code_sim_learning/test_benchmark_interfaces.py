"""Agent-facing benchmark boundaries, exercised through model loading."""
# Dynamic injected model classes cannot be resolved by pylint.
# pylint: disable=protected-access,no-member
from typing import Any

import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.envs import create_new_env
from predicators.run.recording import sanitize_state
from tests.code_sim_learning.test_fan_gt_simulator import _make_noop


@pytest.mark.parametrize("domain,hidden", [
    ("boil", "_simulate_heating"),
    ("fan", "_simulate_fans_dynamic"),
    ("bridge", "_domain_specific_step"),
    ("domino", "_make_domino_component"),
    ("balloons", "hover_height"),
])
@pytest.mark.parametrize("partial", [False, True])
def test_model_base_has_only_visible_core(domain: str, hidden: str,
                                          partial: bool) -> None:
    """The injected base has bodies/readouts, not hidden laws or task
    makers."""
    utils.reset_config({
        "env": "pybullet_" + domain,
        "num_train_tasks": 1,
        "partially_observable": partial
    })
    real: Any = create_new_env("pybullet_" + domain, do_cache=False)
    base: Any = base_simulator_class(real.get_name())
    assert type(real) not in base.__mro__
    if hidden != "_domain_specific_step":
        assert not hasattr(base, hidden)
    model = base(use_gui=False)
    try:
        state = sanitize_state(real.get_train_tasks()[0].init)
        model._set_state(state)
        observed = model._get_state()
        assert {o.name for o in observed} == {o.name for o in state}
        action = _make_noop(observed, model)  # type: ignore[no-untyped-call]
        predicted = model.simulate(state, action)
        assert {o.name for o in predicted} == {o.name for o in state}
        assert model.get_train_tasks() == []
        assert model.get_test_tasks() == []
        assert model.predicates == set()
    finally:
        p.disconnect(model._physics_client_id)
        p.disconnect(real._physics_client_id)
