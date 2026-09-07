"""Tests for the observation-view contract: residual rules never see the env-
only ``privileged`` / ``simulator_state`` channels of a ``State``.

The strip happens at the rule entry points (first argument) and at every
``History`` append site; the caller's own state must keep both fields,
because the env's ``_set_state`` restore path reads ``privileged`` back.
"""
import numpy as np

from predicators.code_sim_learning.utils import apply_rules, \
    apply_rules_with_latent, observation_view, rollout_predictions
from predicators.structs import Action, Object, State, Type

_THING_TYPE = Type("thing", ["x"])


def _make_state(x: float) -> State:
    obj = Object("thing0", _THING_TYPE)
    return State({obj: np.array([x], dtype=np.float64)},
                 simulator_state="opaque-engine-blob",
                 privileged={"thing0": {
                     "hidden": 1.0
                 }})


def test_rules_receive_observation_view():
    """Both entry points strip privileged/simulator_state from the first
    argument; the caller's state keeps both fields."""
    state = _make_state(0.0)
    seen = []

    def legacy_rule(observation, updates, params):
        del params
        seen.append(observation)
        return updates

    def recurrent_rule(observation, latent, history, updates, params):
        del latent, params
        seen.append(observation)
        seen.extend(s for s, _ in history)
        return updates

    apply_rules(state, [legacy_rule], {})
    apply_rules_with_latent(state, {}, [], [recurrent_rule], {})
    assert len(seen) == 2
    for obs in seen:
        assert obs.privileged is None
        assert obs.simulator_state is None
        assert obs.data is state.data  # a view, not a copy
    # The caller's state is untouched (the env restore path needs it).
    assert state.privileged == {"thing0": {"hidden": 1.0}}
    assert state.simulator_state == "opaque-engine-blob"


def test_history_entries_are_observation_views():
    """The canonical rollout loop appends stripped observations, so a rule
    cannot recover the hidden truth through ``history`` either."""
    histories = []

    def recurrent_rule(observation, latent, history, updates, params):
        del observation, latent, params
        histories.append(list(history))
        return updates

    group = [(_make_state(float(i)), Action(np.zeros(1)), _make_state(0.0))
             for i in range(3)]
    rollout_predictions([recurrent_rule], {}, [group])
    assert len(histories) == 3
    assert len(histories[-1]) == 3
    for hist in histories:
        for obs, _act in hist:
            assert obs.privileged is None
            assert obs.simulator_state is None
    # The recorded trajectory itself keeps its hidden blocks.
    for base_state, _act, _next_obs in group:
        assert base_state.privileged is not None


def test_observation_view_passthrough_when_nothing_to_strip():
    """With no env-only fields set, the view is the state itself (object
    identity preserved on fully-observable paths), and the agent's own
    ``latent`` block always rides along."""
    obj = Object("thing0", _THING_TYPE)
    plain = State({obj: np.array([0.0], dtype=np.float64)},
                  latent={"count": 3})
    assert observation_view(plain) is plain
    hidden = _make_state(0.0)
    hidden.latent = {"count": 3}
    view = observation_view(hidden)
    assert view is not hidden
    assert view.latent is hidden.latent
