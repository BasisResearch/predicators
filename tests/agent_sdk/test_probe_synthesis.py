"""Tests for the synthesis-phase behavior of the explore_python probe.

Covers the ``ctx.probe_option_model_provider`` hook: model resolution
(candidate provider vs. the solve-phase ``ctx.option_model`` fallback),
the explicit-``task_idx`` guard on ``reset`` during synthesis, the
provider's load/cache glue, and the phase-appropriate tool description.
"""
# pylint: disable=protected-access
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.probe_api import ProbeSim
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.option_model import _OptionModelBase
from predicators.structs import Object, State, Task, Type


def _tiny_task() -> Task:
    obj_type = Type("thing", ["x"])
    obj = Object("thing0", obj_type)
    return Task(State({obj: np.array([0.0], dtype=np.float32)}), set())


def _fake_model() -> _OptionModelBase:
    """An opaque sentinel standing in for an option model."""
    return cast(_OptionModelBase, object())


def test_probe_model_resolution_prefers_provider() -> None:
    """With no provider the probe runs ctx.option_model (solve phase); with a
    provider installed it runs the provider's model and never touches
    ctx.option_model (synthesis phase)."""
    stale = _fake_model()
    candidate = _fake_model()
    ctx = ToolContext()
    ctx.option_model = stale
    sim = ProbeSim(ctx)
    assert sim._option_model() is stale

    ctx.probe_option_model_provider = lambda: candidate
    assert sim._option_model() is candidate

    # Provider errors (e.g. no loadable simulator.py yet) surface to the
    # caller instead of falling back to the stale model.
    def _no_candidate() -> _OptionModelBase:
        raise RuntimeError("no candidate simulator yet")

    ctx.probe_option_model_provider = _no_candidate
    with pytest.raises(RuntimeError, match="no candidate simulator"):
        sim._option_model()


def test_probe_reset_requires_task_idx_during_synthesis() -> None:
    """During synthesis, reset() must not silently fall back to the (stale,
    solve-time) ctx.current_task: task_idx is required."""
    task = _tiny_task()
    ctx = ToolContext()
    ctx.train_tasks = [task]
    ctx.current_task = task

    # Solve phase: current-task fallback works.
    sim = ProbeSim(ctx)
    sim.reset()
    assert sim._state is not None

    # Synthesis phase: explicit task_idx required...
    ctx.probe_option_model_provider = _fake_model
    sim = ProbeSim(ctx)
    with pytest.raises(ValueError, match="task_idx explicitly"):
        sim.reset()
    # ...and accepted.
    sim.reset(task_idx=0)
    assert sim._state is not None


def test_candidate_probe_model_provider_glue(tmp_path) -> None:
    """The provider gates on a loadable simulator.py, caches by content hash
    (no refit for an unchanged file), and rebuilds on change.

    Exercises the real ``_make_candidate_probe_model_provider`` and the
    real file loader; only the fit/build layer below
    ``build_candidate_option_model`` is stubbed (its body is the shared
    ``evaluate_plan_refinement`` path).
    """
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    from predicators.code_sim_learning.fit_space import FitResult

    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {}
    approach._latent_init = None
    fit_calls = {"n": 0}

    def _fake_fit(rules, specs, triples, features):
        del rules, triples, features
        fit_calls["n"] += 1
        names = [s.name for s in specs]
        return FitResult(names=names,
                         samples=np.array([[s.init_value for s in specs]]),
                         log_probs=np.array([0.0])), 0.0

    setattr(approach, "_fit_parameters", _fake_fit)
    setattr(approach, "_build_combined_simulator", lambda learned: learned)
    setattr(approach, "_build_option_model", lambda sim: ("model", sim))

    simulator_file = str(tmp_path / "simulator.py")
    provider = approach._make_candidate_probe_model_provider(
        simulator_file,
        trajectories=[],
        base_pred_triples=[],
        inferred_hint={"thing": ["x"]})

    # No file yet: hard error, never a fallback model.
    with pytest.raises(RuntimeError, match="no candidate simulator yet"):
        provider()

    # Broken file: hard error too.
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write("PROCESS_RULES = None\n")
    with pytest.raises(RuntimeError, match="failed to load"):
        provider()

    valid = ("def _rule(state, updates, params):\n"
             "    return updates\n"
             "PROCESS_RULES = [_rule]\n"
             "PARAM_SPECS = [ParamSpec('k', 1.0, lo=0.0, hi=2.0)]\n"
             "PROCESS_FEATURES = {'thing': ['x']}\n")
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write(valid)
    model = provider()
    assert fit_calls["n"] == 1
    assert approach._fitted_params == {"k": 1.0}

    # Unchanged content: cached, no refit.
    assert provider() is model
    assert fit_calls["n"] == 1

    # Changed content: rebuilt.
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write(valid.replace("1.0, lo", "1.5, lo"))
    assert provider() is not model
    assert fit_calls["n"] == 2
    assert approach._fitted_params == {"k": 1.5}


def test_explore_python_description_follows_phase() -> None:
    """The tool description flips with the installed provider: candidate
    simulator + evaluate_plan_refinement wording in synthesis sessions, belief
    simulator + evaluate_option_plan wording in solve sessions."""
    utils.reset_config({"agent_planner_use_explore_python": True})

    def _desc(ctx: ToolContext) -> str:
        (tool, ) = (t for t in create_mcp_tools(ctx, ["explore_python"])
                    if getattr(t, "name", "") == "explore_python")
        description = getattr(tool, "description", "")
        assert description
        return description

    solve_desc = _desc(ToolContext())
    assert "belief simulator" in solve_desc
    assert "evaluate_option_plan" in solve_desc

    ctx = ToolContext()
    ctx.probe_option_model_provider = _fake_model
    synth_desc = _desc(ctx)
    assert "CANDIDATE simulator" in synth_desc
    assert "task_idx is required" in synth_desc
    assert "validate the simulator itself via evaluate_plan_refinement" \
        in synth_desc
