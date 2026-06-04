"""Tests for code sim-learning training utilities."""

import numpy as np

from predicators import utils
from predicators.code_sim_learning.training import ParamSpec, \
    compute_sse_recurrent, fit_params
from predicators.code_sim_learning.utils import has_latent_rules, \
    rollout_predictions
from predicators.structs import Action, State, Type


def _mk_jug_trajectory():
    """A 2-step single-object trajectory: (base, action, next_obs) triples.

    ``bubbling`` rises 0 -> 0.5 -> 1.0 across the trajectory, which a
    recurrent rule can only predict by accumulating a hidden quantity.
    """
    jug = Type("jug", ["bubbling"])
    j = jug("jug0")
    act = Action(np.zeros(1, dtype=np.float32))

    def s(v):
        return State({j: np.array([v], dtype=np.float32)})

    group = [(s(0.0), act, s(0.5)), (s(0.5), act, s(1.0))]
    return j, group


def test_rollout_predictions_threads_latent_for_recurrent_rules():
    """A correct 5-arg rule is rolled out with the latent threaded.

    This is the path the synthesis tools now take for recurrent rules.
    The old per-transition path called such a rule with 3 args and
    raised ``TypeError``, which misled the agent into writing a broken
    3-arg rule.
    """
    j, group = _mk_jug_trajectory()

    def bubbling_rule(state, latent, history, updates, params):
        del state, history
        latent["heat"] = latent.get("heat", 0.0) + params["rate"]
        updates.setdefault(j, {})["bubbling"] = min(1.0, latent["heat"])
        return updates

    rules = [bubbling_rule]
    assert has_latent_rules(rules)

    preds = rollout_predictions(rules, {"rate": 0.5}, [group],
                                latent_init={"heat": 0.0})
    # Latent accumulates across steps: 0.5 then 1.0 (not reset each step).
    assert [round(float(sp.get(j, "bubbling")), 3) for sp, _ in preds] == \
        [0.5, 1.0]
    # And the recurrent SSE agrees with the observations exactly.
    sse = compute_sse_recurrent(rules, [group], {"rate": 0.5}, {"heat": 0.0},
                                {"jug": ["bubbling"]})
    assert sse == 0.0


def test_rollout_predictions_legacy_rules_are_independent():
    """3-arg rules apply per-transition; latent_init is ignored."""
    j, group = _mk_jug_trajectory()

    def legacy_rule(state, updates, params):
        del state
        updates.setdefault(j, {})["bubbling"] = params["const"]
        return updates

    rules = [legacy_rule]
    assert not has_latent_rules(rules)
    preds = rollout_predictions(rules, {"const": 0.3}, [group],
                                latent_init={"heat": 0.0})
    # Each step predicts the constant independently — no accumulation.
    assert [round(float(sp.get(j, "bubbling")), 3) for sp, _ in preds] == \
        [0.3, 0.3]


def test_fit_params_can_skip_training_with_cfg():
    """Test that CFG can disable parameter fitting."""
    utils.reset_config({"code_sim_learning_num_mcmc_steps": 0})
    param_specs = [ParamSpec("rate", 2.5), ParamSpec("threshold", 0.7)]

    result = fit_params(
        simulator_fn=lambda _s, _a, _p: {},
        transitions=[],
        param_specs=param_specs,
        process_features={},
    )

    assert result.point_estimate == {"rate": 2.5, "threshold": 0.7}
    np.testing.assert_allclose(result.samples, np.array([[2.5, 0.7]]))
    np.testing.assert_allclose(result.log_probs, np.array([0.0]))
