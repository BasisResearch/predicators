"""Tests for the sim-learning ablation flags.

* ``agent_sim_learn_param_uncertainty`` (A6+A7): nothing consumes a
  parameter posterior - no physics-margin points, no rule-parameter
  ensemble.
* ``agent_sim_learn_declared_params_only`` (A4): no estimation runs;
  the declaration is the estimate and its box the plausible interval.
* ``agent_sim_learn_zero_shot`` (A2): the synthesis session runs with
  no recorded transitions.
"""
# pylint: disable=protected-access
from typing import Any, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.tools import create_synthesis_tools
from predicators.approaches import agent_sim_learning_approach as asla
from predicators.code_sim_learning.fit_space import ParamSpec, \
    declared_interval_fit_result, declared_interval_report
from predicators.code_sim_learning.identifiability import physics_sigma_points
from predicators.settings import CFG


class _RegistryEnv:
    """Env double recording the physical-param overrides it receives."""

    def __init__(self) -> None:
        self.applied: List[Dict[str, float]] = []

    def get_physical_param_info(self) -> Dict[str, Any]:
        """No registry defaults to revert to."""
        return {}

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Record each application in order."""
        self.applied.append(dict(params))


def _bare_approach() -> Any:
    approach: Any = asla.AgentSimLearningApproach.__new__(
        asla.AgentSimLearningApproach)
    approach._base_env = _RegistryEnv()
    approach._identified_physical_params = {}
    approach._identified_physical_sigma_points = []
    approach._cycle_applied_physical = {}
    approach._fitted_params = {}
    approach._param_ensemble = []
    approach._param_specs = []
    approach._last_fit_result = None
    approach._fit_sse = float("nan")
    approach._rng = np.random.default_rng(0)
    approach._oracle_param_sse = lambda *a, **k: 1.5
    return approach


_RULE_SPECS = [ParamSpec("k", 2.0, 1.0, 3.0)]
_PHYS_SPECS = [ParamSpec("lateral_friction", 0.5, 0.2, 0.8, "log")]


def test_declared_interval_fit_result_spans_the_boxes() -> None:
    """Sample 0 is the declared init; the rest fill each declared box."""
    specs = [
        ParamSpec("a", 1.0, 0.5, 2.0, "log"),
        ParamSpec("b", 0.0, -1.0, 1.0),
        ParamSpec("c", 3.0),
    ]
    result = declared_interval_fit_result(specs, 32, np.random.default_rng(0))
    assert result.names == ["a", "b", "c"]
    assert result.samples.shape == (32, 3)
    # The declared inits are the MAP.
    assert result.point_estimate == {"a": 1.0, "b": 0.0, "c": 3.0}
    assert np.all(result.samples[:, 0] >= 0.5)
    assert np.all(result.samples[:, 0] <= 2.0)
    assert np.all(np.abs(result.samples[:, 1]) <= 1.0)
    # Not all at the init: the box was actually sampled.
    assert len(set(np.round(result.samples[:, 0], 6))) > 1
    # An unboxed param never moves.
    assert np.all(result.samples[:, 2] == 3.0)
    assert result.scales == ["log", "linear", "linear"]


def test_declared_interval_report_feeds_the_margin_hull() -> None:
    """The declared bounds become the physics-margin hull."""
    report = declared_interval_report(_PHYS_SPECS + [ParamSpec("open", 1.0)])
    assert report["lateral_friction"]["candidate_values"] == [0.2, 0.8]
    assert report["open"]["candidate_values"] == []
    points = physics_sigma_points({"lateral_friction": 0.5},
                                  report,
                                  _PHYS_SPECS,
                                  num_points=5)
    values = [p["lateral_friction"] for p in points]
    # The sweep spans the declared box (geometrically, log scale) and
    # the exact declared point is dropped like a fitted point would be.
    assert values[0] == pytest.approx(0.2)
    assert values[-1] == pytest.approx(0.8)
    assert 0.5 not in values
    # A param with no declared box is not perturbable.
    assert not physics_sigma_points({"open": 1.0}, report,
                                    [ParamSpec("open", 1.0)])


def test_deploy_declared_params_uses_the_declaration_as_the_estimate() -> None:
    """A4: inits deploy as the estimate, boxes as margins and ensemble."""
    utils.reset_config({
        "agent_sim_learn_declared_params_only": True,
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_ensemble_size": 8,
        "agent_plan_validation_physics_margin_points": 4,
    })
    approach = _bare_approach()
    approach._physical_param_specs = list(_PHYS_SPECS)
    approach._fit_params_after_synthesis([], list(_RULE_SPECS),
                                         [("s", "a", "s2")], {})
    # Rule params at their declared init; the declared physical init went
    # to the planning env.
    assert approach._fitted_params == {"k": 2.0}
    assert approach._base_env.applied == [{"lateral_friction": 0.5}]
    assert approach._cycle_applied_physical == {"lateral_friction": 0.5}
    # Physics margin spans the declared box.
    frictions = [
        p["lateral_friction"]
        for p in approach._identified_physical_sigma_points
    ]
    assert frictions[0] == pytest.approx(0.2)
    assert frictions[-1] == pytest.approx(0.8)
    # The ensemble samples the declared boxes: anchor at the declaration,
    # every other member inside both boxes.
    assert len(approach._param_ensemble) == 8
    assert approach._param_ensemble[0] == {
        "lateral_friction": 0.5,
        "k": 2.0,
    }
    for member in approach._param_ensemble[1:]:
        assert 0.2 <= member["lateral_friction"] <= 0.8
        assert 1.0 <= member["k"] <= 3.0
    assert any(m["k"] != 2.0 for m in approach._param_ensemble[1:])
    assert approach._fit_sse == 1.5


def test_rule_param_margin_alone_builds_the_ensemble() -> None:
    """A6 (info-seeking off, gate on) keeps the validation ensemble; with both
    consumers off (A6+A7) none is built."""
    utils.reset_config({
        "agent_sim_learn_declared_params_only": True,
        "agent_explorer_info_seeking": False,
        "agent_plan_validation_rule_param_margin": True,
        "agent_explorer_info_ensemble_size": 5,
    })
    approach = _bare_approach()
    approach._physical_param_specs = list(_PHYS_SPECS)
    approach._fit_params_after_synthesis([], list(_RULE_SPECS),
                                         [("s", "a", "s2")], {})
    assert len(approach._param_ensemble) == 5
    utils.reset_config({
        "agent_sim_learn_declared_params_only": True,
        "agent_explorer_info_seeking": False,
        "agent_plan_validation_rule_param_margin": False,
    })
    approach = _bare_approach()
    approach._physical_param_specs = list(_PHYS_SPECS)
    approach._fit_params_after_synthesis([], list(_RULE_SPECS),
                                         [("s", "a", "s2")], {})
    assert not approach._param_ensemble


def test_deploy_declared_params_without_data_has_no_sse() -> None:
    """A4 with nothing recorded: inits deploy, no SSE, no ensemble."""
    utils.reset_config({
        "agent_sim_learn_declared_params_only": True,
        "agent_explorer_info_seeking": False,
    })
    approach = _bare_approach()
    approach._physical_param_specs = []
    approach._fit_params_after_synthesis([], list(_RULE_SPECS), [], {})
    assert approach._fitted_params == {"k": 2.0}
    assert approach._fit_sse == float("inf")
    assert not approach._param_ensemble
    assert not approach._base_env.applied


def test_no_uncertainty_flag_removes_margins_and_ensemble() -> None:
    """A6+A7: point estimates deploy; no margin points, no ensemble."""
    utils.reset_config({
        "agent_sim_learn_declared_params_only": True,
        "agent_sim_learn_param_uncertainty": False,
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_ensemble_size": 8,
    })
    approach = _bare_approach()
    approach._physical_param_specs = list(_PHYS_SPECS)
    approach._fit_params_after_synthesis([], list(_RULE_SPECS), [], {})
    # The point estimates deploy exactly as before ...
    assert approach._fitted_params == {"k": 2.0}
    assert approach._base_env.applied == [{"lateral_friction": 0.5}]
    # ... but nothing that consumes a posterior exists.
    assert not approach._identified_physical_sigma_points
    assert not approach._param_ensemble
    assert not approach._physics_margin_points(
        {"lateral_friction": 0.5}, declared_interval_report(_PHYS_SPECS),
        list(_PHYS_SPECS))


def test_no_data_seeding_applies_declared_physical_inits() -> None:
    """The fit-less no-transitions branch deploys the declared physical inits
    to the planning env (the agent's stated belief, not the registry default),
    which the zero-shot arm relies on."""
    utils.reset_config({
        "agent_sim_learn_declared_params_only": False,
        "agent_sim_learn_oracle_sim_params": False,
        "agent_explorer_info_seeking": False,
    })
    approach = _bare_approach()
    approach._physical_param_specs = list(_PHYS_SPECS)
    approach._fit_params_after_synthesis([], list(_RULE_SPECS), [], {})
    assert approach._fitted_params == {"k": 2.0}
    assert approach._base_env.applied == [{"lateral_friction": 0.5}]
    assert approach._last_fit_result is None


def test_zero_shot_flag_gates_data_free_synthesis() -> None:
    """With no transitions, _learn_simulator returns early unless the zero-shot
    flag is set, in which case synthesis runs on empty data."""
    approach: Any = asla.AgentSimLearningApproach.__new__(
        asla.AgentSimLearningApproach)
    approach._explainability_cache = {}
    approach._sysid_fit_cache = {}
    approach._persist_fit_trajectories = lambda *a, **k: None
    approach._maybe_install_oracle_samplers = lambda: None
    approach._extract_obs_triples = lambda trajs: []
    approach._residual_rules = None
    approach._learned_simulator = None
    approach._fitted_params = {}
    calls: List[Any] = []

    def _synth(trajectories, obs_triples, base_pred_triples, inferred_hint):
        calls.append(
            (trajectories, obs_triples, base_pred_triples, inferred_hint))

    approach._synthesize_with_agent = _synth
    utils.reset_config({
        "agent_sim_learn_zero_shot": False,
        "agent_sim_learn_oracle_sim_program": False,
    })
    approach._learn_simulator([])
    assert not calls
    utils.reset_config({
        "agent_sim_learn_zero_shot": True,
        "agent_sim_learn_oracle_sim_program": False,
    })
    approach._learn_simulator([])
    assert calls == [([], [], [], {})]


def test_declared_params_prompt_section_is_flag_gated() -> None:
    """The no-estimation section renders only under the A3 flag."""
    kwargs: Dict[str, Any] = dict(
        partially_observable=False,
        residual_rule_signature="def rule(state, updates, params):",
        scene_viz_hint="look",
    )
    plain = learn_prompts.build_learn_system_prompt(**kwargs)
    declared = learn_prompts.build_learn_system_prompt(
        declared_params_only=True, **kwargs)
    marker = "Parameter estimation is DISABLED"
    assert marker not in plain
    assert marker in declared
    assert "__" not in declared.replace("__init__", "")
    zero_shot = learn_prompts.render_zero_shot_message()
    assert "No trajectory has been recorded" in zero_shot


def test_estimation_surfaces_refuse_under_declared_params(tmp_path) -> None:
    """A4: sim.fit, fit_params and sweep_params refuse; the plain report
    runs."""
    utils.reset_config({"agent_sim_learn_declared_params_only": True})
    toolkit = create_synthesis_tools(exec_ns={},
                                     base_pred_triples=[],
                                     inferred_residual_features={},
                                     simulator_file=str(tmp_path /
                                                        "simulator.py"),
                                     versions_dir=str(tmp_path / "v"))
    out = toolkit.fit_runner()
    assert "sim.fit is unavailable" in out
    assert "declared" in out
    out = toolkit.residuals_runner(fit_params=True)
    assert "sim.residuals(fit_params=True) is unavailable" in out
    out = toolkit.residuals_runner(rollout=True, sweep_params="all")
    assert "sweep_params" in out and "unavailable" in out
    # The plain report is still allowed (it scores the declared values).
    out = toolkit.residuals_runner()
    assert "unavailable" not in out
    utils.reset_config({"agent_sim_learn_declared_params_only": False})
    assert CFG.agent_sim_learn_declared_params_only is False
