"""The experimental repair workflow belongs only to the MB arm."""
# pylint: disable=import-outside-toplevel
from predicators import utils
from predicators.agent_sdk.play_prompts import build_play_system_prompt


def test_repair_is_opt_in_and_model_free_is_unchanged():
    """Changing the repair flag must leave the MF comparison untouched."""
    utils.reset_config({"agent_model_repair": False})
    mb_before = build_play_system_prompt(["run_python", "env_step"])
    mf_before = build_play_system_prompt(["env_step"])
    assert "## Model repair" not in mb_before
    utils.reset_config({"agent_model_repair": True})
    mb_after = build_play_system_prompt(["run_python", "env_step"])
    assert "sim.validate()" in mb_after
    assert "alternative dynamics structures" in mb_after
    assert "not a hard action gate" in mb_after
    assert "simulate candidate real probes" in mb_after
    assert build_play_system_prompt(["env_step"]) == mf_before


def test_repair_advice_does_not_diagnose_chaos_from_error_alone():
    """A large residual calls for model comparison before real recollection."""
    from predicators.agent_sdk.tools.synthesis import _trim_cause_note
    utils.reset_config({"agent_model_repair": True})
    note = " ".join(_trim_cause_note([0.15, 0.4], 0.1))
    assert "does not identify the cause" in note
    assert "sim.validate()" in note
    assert "before spending more real steps" in note
    utils.reset_config({})
