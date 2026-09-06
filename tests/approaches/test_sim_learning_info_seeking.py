"""Tests for AgentSimLearningApproach.score_atom_disagreement.

Validates the param-swap mechanism that turns a parameter ensemble into a
boundary-straddling-detector: a learned predicate whose classifier reads
the approach's live ``_fitted_params`` is evaluated under each ensemble
member, and the across-member disagreement is the info score.

Also covers ensemble selection (Laplace when the LM Jacobian is present,
else uniform jitter) and the LM-seed short-circuit of the recurrent fit.
"""

# pylint: disable=protected-access,import-outside-toplevel,unused-import

import numpy as np

from predicators import utils  # noqa: F401  (settles import order)
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.structs import Action, GroundAtom, Object, Predicate, State, \
    Type

_t = Type("block", ["x"])
_block = Object("b", _t)


def _bare_approach(ensemble, fitted):
    """An approach instance with only the fields the scorer touches."""
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = dict(fitted)
    approach._param_ensemble = [dict(m) for m in ensemble]
    return approach


def _at_target_atom(approach):
    """AtTarget(block) holds iff x < the live fitted threshold."""

    def _classifier(s, o):
        return s.get(o[0], "x") < approach._fitted_params["thresh"]

    return GroundAtom(Predicate("AtTarget", [_t], _classifier), [_block])


def _state(x):
    return State({_block: np.array([x], dtype=np.float32)})


def test_disagreement_high_at_boundary():
    """Disagreement high at boundary."""
    ens = [{"thresh": t} for t in (0.5, 0.3, 0.4, 0.6, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    atom = _at_target_atom(approach)
    # x=0.5 splits the ensemble (3 say False, 2 say True) -> nonzero entropy.
    assert approach.score_atom_disagreement(_state(0.5), {atom}) > 0.0


def test_disagreement_zero_far_from_boundary():
    """Disagreement zero far from boundary."""
    ens = [{"thresh": t} for t in (0.5, 0.3, 0.4, 0.6, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    atom = _at_target_atom(approach)
    # x=0.05 < every threshold -> all members agree True -> no disagreement.
    assert approach.score_atom_disagreement(_state(0.05), {atom}) == 0.0
    # x=0.95 > every threshold -> all agree False -> no disagreement.
    assert approach.score_atom_disagreement(_state(0.95), {atom}) == 0.0


def test_fitted_params_restored_after_scoring():
    """Fitted params restored after scoring."""
    ens = [{"thresh": t} for t in (0.3, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    atom = _at_target_atom(approach)
    approach.score_atom_disagreement(_state(0.5), {atom})
    # The scorer must leave the MAP params exactly as it found them.
    assert approach._fitted_params == {"thresh": 0.5}


def test_singleton_ensemble_scores_zero():
    """Singleton ensemble scores zero."""
    approach = _bare_approach([{"thresh": 0.5}], {"thresh": 0.5})
    atom = _at_target_atom(approach)
    assert approach.score_atom_disagreement(_state(0.5), {atom}) == 0.0


def test_empty_atoms_scores_zero():
    """Empty atoms scores zero."""
    ens = [{"thresh": t} for t in (0.3, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    assert approach.score_atom_disagreement(_state(0.5), set()) == 0.0


def test_rebuild_param_ensemble_respects_flag():
    """Rebuild param ensemble respects flag."""
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {"a": 1.0}
    approach._param_specs = []
    approach._param_ensemble = [{"a": 1.0}, {"a": 2.0}]
    approach._last_fit_result = None  # no calibrated fit -> uniform fallback
    approach._rng = np.random.default_rng(0)
    utils.reset_config({"agent_explorer_info_seeking": False})
    approach._rebuild_param_ensemble()
    assert approach._param_ensemble == []  # cleared when off

    from predicators.code_sim_learning.fit_space import ParamSpec
    approach._param_specs = [ParamSpec("a", 1.0, lo=0.0, hi=2.0)]
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_ensemble_size": 5,
        "agent_explorer_info_perturb_frac": 0.2,
    })
    approach._rebuild_param_ensemble()
    assert len(approach._param_ensemble) == 5
    assert approach._param_ensemble[0] == {"a": 1.0}  # member 0 is anchor


def test_rebuild_param_ensemble_empty_under_oracle_params():
    """Oracle params carry no uncertainty, so no ensemble is built.

    Without this the uniform-jitter fallback would hand the capture gate
    members that no plan can satisfy (a zero rate, a rewired lamp), and
    the gate would refuse every plan on a model that is exactly right.
    """
    from predicators.code_sim_learning.fit_space import ParamSpec
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {"a": 1.0}
    approach._param_specs = [ParamSpec("a", 1.0, lo=0.0, hi=2.0)]
    approach._param_ensemble = [{"a": 1.0}, {"a": 2.0}]
    approach._last_fit_result = None
    approach._rng = np.random.default_rng(0)
    utils.reset_config({
        "agent_plan_validation_rule_param_margin": True,
        "agent_explorer_info_ensemble_size": 5,
        "agent_sim_learn_oracle_sim_params": True,
    })
    approach._rebuild_param_ensemble()
    assert approach._param_ensemble == []

    utils.reset_config({
        "agent_plan_validation_rule_param_margin": True,
        "agent_explorer_info_ensemble_size": 5,
        "agent_sim_learn_oracle_sim_params": False,
    })
    approach._rebuild_param_ensemble()
    assert len(approach._param_ensemble) == 5


def _selector_approach(fit_result):
    from predicators.code_sim_learning.fit_space import ParamSpec
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {"a": 1.0, "b": 2.0}
    approach._param_specs = [
        ParamSpec("a", 1.0, lo=-10.0, hi=10.0),
        ParamSpec("b", 2.0, lo=-10.0, hi=10.0),
    ]
    approach._last_fit_result = fit_result
    approach._rng = np.random.default_rng(0)
    return approach


def test_select_ensemble_uses_laplace_when_only_jacobian():
    """Select ensemble uses laplace when the LM Jacobian is present."""
    from predicators.code_sim_learning.fit_space import FitResult

    # The Laplace bundle (Jacobian + sigmas) is present.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.0, 2.0]]),
                    log_probs=np.zeros(1),
                    jacobian=np.eye(2),
                    noise_sigma=0.1,
                    prior_sigma=np.array([1.0, 1.0]))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 4,
    })
    members, method = approach._select_param_ensemble(4)
    assert method == "laplace"
    assert len(members) == 4
    assert members[0] == {"a": 1.0, "b": 2.0}


def test_select_ensemble_falls_back_to_uniform_without_calibration():
    """Select ensemble falls back to uniform without calibration."""
    from predicators.code_sim_learning.fit_space import FitResult

    # Single-row samples and no Jacobian (LM skipped/failed) -> uniform.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.0, 2.0]]),
                    log_probs=np.zeros(1))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 4,
        "agent_explorer_info_perturb_frac": 0.2,
    })
    _, method = approach._select_param_ensemble(4)
    assert method == "uniform-perturb"


def test_select_ensemble_uniform_when_calibration_disabled():
    """Select ensemble uniform when calibration disabled."""
    from predicators.code_sim_learning.fit_space import FitResult

    # Posterior samples exist, but the calibration flag is off -> uniform.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.1, 2.1], [0.9, 1.9]]),
                    log_probs=np.zeros(2))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": False,
        "agent_explorer_info_ensemble_size": 4,
        "agent_explorer_info_perturb_frac": 0.2,
    })
    _, method = approach._select_param_ensemble(4)
    assert method == "uniform-perturb"


def test_underdetermined_guard_widens_ensemble():
    """With the guard on, a fit with dof<min gets a wide ensemble.

    An exactly-determined fit (as many residuals as params) fits its
    data perfectly, so its tight Laplace width is false certainty; the
    guard replaces it with a box-relative spread so the margin gate can
    refuse the overfit plan.
    """
    from predicators.code_sim_learning.fit_space import FitResult

    # 2 residuals, 2 params -> dof = 0 (< default min 1): under-determined.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.0, 2.0]]),
                    log_probs=np.zeros(1),
                    jacobian=np.ones((2, 2)),
                    noise_sigma=0.1,
                    prior_sigma=np.array([1.0, 1.0]))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 6,
        "agent_sim_learn_underdetermined_guard": True,
    })
    members, method = approach._select_param_ensemble(6)
    assert method == "underdetermined"
    assert len(members) == 6 and members[0] == {"a": 1.0, "b": 2.0}
    # The guard must not fire when it is off (original Laplace behaviour)...
    utils.reset_config({
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 6,
        "agent_sim_learn_underdetermined_guard": False,
    })
    _, method_off = approach._select_param_ensemble(6)
    assert method_off == "laplace"


def test_underdetermined_guard_leaves_well_determined_fit_on_laplace():
    """A fit with enough residuals keeps its calibrated Laplace ensemble."""
    from predicators.code_sim_learning.fit_space import FitResult

    # 8 residuals, 2 params -> dof = 6 (>= min 1): well-determined.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.0, 2.0]]),
                    log_probs=np.zeros(1),
                    jacobian=np.ones((8, 2)),
                    noise_sigma=0.1,
                    prior_sigma=np.array([1.0, 1.0]))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 4,
        "agent_sim_learn_underdetermined_guard": True,
    })
    _, method = approach._select_param_ensemble(4)
    assert method == "laplace"


def test_fit_params_no_data_seeds_declared_inits(monkeypatch):
    """With no transitions, params seed from inits and no fit runs.

    This is the oracle-sim-program no-demos path: every demo failed, so
    ``_learn_simulator`` reaches the fit with empty
    ``base_pred_triples`` and must fall back to the declared init values
    instead of fitting.
    """
    from predicators.code_sim_learning.fit_space import ParamSpec

    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {}
    approach._param_specs = []
    approach._physical_param_specs = []
    approach._param_ensemble = []
    approach._last_fit_result = None
    approach._fit_sse = 0.0
    approach._rng = np.random.default_rng(0)

    def _fail_fit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("fit must not run with no data")

    monkeypatch.setattr(
        "predicators.approaches.agent_sim_learning_approach"
        ".fit_rule_parameters", _fail_fit)
    utils.reset_config({
        "agent_sim_learn_oracle_sim_params": False,
        "agent_explorer_info_seeking": False,
    })
    specs = [ParamSpec("a", 1.5, lo=0.0, hi=5.0)]
    approach._fit_params_after_synthesis([], specs, [], {})
    assert approach._fitted_params == {"a": 1.5}
    assert approach._last_fit_result is None
    assert approach._fit_sse == float("inf")


def test_lm_seed_skips_lm_refit(monkeypatch):
    """A precomputed (theta_map, jac) short-circuits the LM prefit."""
    import predicators.code_sim_learning.fitting as fitting_mod
    from predicators.code_sim_learning.fit_space import ParamSpec
    utils.reset_config({
        "code_sim_learning_warm_start_with_lm": True,
        "agent_explorer_info_seeking": True,
    })
    calls = {"n": 0}

    def _counting_lm(*_a, **_k):
        calls["n"] += 1
        return np.array([1.2]), np.array([[0.5]])

    monkeypatch.setattr(fitting_mod, "fit_map_lm_recurrent", _counting_lm)
    monkeypatch.setattr(fitting_mod, "compute_sse_recurrent",
                        lambda *a, **k: 0.0)
    specs = [ParamSpec("a", 1.0, lo=0.0, hi=2.0)]
    # Without a seed the LM prefit runs.
    result = fitting_mod.fit_params_recurrent(rules=[],
                                              trajectories=[[]],
                                              param_specs=specs,
                                              latent_init=None,
                                              residual_features={})
    assert calls["n"] == 1
    # With a seed it does not, and the seed's jacobian is carried.
    jac = np.array([[0.7]])
    result = fitting_mod.fit_params_recurrent(rules=[],
                                              trajectories=[[]],
                                              param_specs=specs,
                                              latent_init=None,
                                              residual_features={},
                                              lm_seed=(np.array([1.3]), jac))
    assert calls["n"] == 1
    assert result.jacobian is not None
    assert float(result.jacobian[0, 0]) == 0.7
    assert result.point_estimate == {"a": 1.3}
