"""Adaptive info-seeking: the proactive apparatus stays dormant until the
capture gate refuses a plan as parameter-sensitive.

Covers the run-context gate (``ToolContext.info_seeking_active``), which
every consumer (``sim.suggest_probes``, the explorer guidance) reads,
and the model-only play-prompt guidance that teaches the submit-first
protocol only when the flag is on.
"""
# pylint: disable=protected-access
from __future__ import annotations

from predicators import utils
from predicators.agent_sdk.play_prompts import build_play_system_prompt
from predicators.agent_sdk.tools import ToolContext

_GUIDANCE_MARKER = "Only if the capture gate refuses"


def _ctx() -> ToolContext:
    return ToolContext(types=set(),
                       predicates=set(),
                       processes=set(),
                       options=set(),
                       train_tasks=[])


def test_gate_off_when_info_seeking_disabled():
    """No info-seeking at all -> never active, regardless of the signal."""
    utils.reset_config({
        "agent_explorer_info_seeking": False,
        "agent_explorer_info_seeking_adaptive": True,
    })
    ctx = _ctx()
    assert not ctx.info_seeking_active()
    ctx.param_sensitive_refusal_pending = True
    assert not ctx.info_seeking_active()


def test_gate_always_on_when_not_adaptive():
    """Info-seeking on, adaptive off -> always active (original behaviour)."""
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_seeking_adaptive": False,
    })
    ctx = _ctx()
    assert ctx.info_seeking_active()
    # The refusal signal is irrelevant in the non-adaptive mode.
    ctx.param_sensitive_refusal_pending = False
    assert ctx.info_seeking_active()


def test_gate_tracks_refusal_signal_when_adaptive():
    """Adaptive -> dormant until a refusal is pending, then active."""
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_seeking_adaptive": True,
    })
    ctx = _ctx()
    assert not ctx.info_seeking_active()
    ctx.param_sensitive_refusal_pending = True
    assert ctx.info_seeking_active()
    # A captured plan clears it and the apparatus goes dormant again.
    ctx.param_sensitive_refusal_pending = False
    assert not ctx.info_seeking_active()


def test_prompt_guidance_only_under_adaptive_model_arm():
    """The submit-first guidance appears only for the model arm with the
    adaptive flag on; the always-on and model-free arms never see it."""
    model_tools = ["env_observe", "run_python", "submit_plan"]
    free_tools = ["env_observe", "submit_plan"]

    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_seeking_adaptive": True,
    })
    adaptive_prompt = build_play_system_prompt(model_tools,
                                               base_sim_refs=["envs/x.py"])
    assert _GUIDANCE_MARKER in adaptive_prompt
    # Model-free arm: the guidance lives in the model-only section.
    assert _GUIDANCE_MARKER not in build_play_system_prompt(free_tools)

    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_seeking_adaptive": False,
    })
    assert _GUIDANCE_MARKER not in build_play_system_prompt(
        model_tools, base_sim_refs=["envs/x.py"])


_GUARD_MARKER = "Separate what you can READ"


def test_underdetermined_prompt_only_under_guard_model_arm():
    """The observable-vs-hidden rule appears only for the model arm with the
    under-determination guard on; model-free never sees it."""
    model_tools = ["env_observe", "run_python", "submit_plan"]
    free_tools = ["env_observe", "submit_plan"]

    utils.reset_config({"agent_sim_learn_underdetermined_guard": True})
    assert _GUARD_MARKER in build_play_system_prompt(
        model_tools, base_sim_refs=["envs/x.py"])
    assert _GUARD_MARKER not in build_play_system_prompt(free_tools)

    utils.reset_config({"agent_sim_learn_underdetermined_guard": False})
    assert _GUARD_MARKER not in build_play_system_prompt(
        model_tools, base_sim_refs=["envs/x.py"])
