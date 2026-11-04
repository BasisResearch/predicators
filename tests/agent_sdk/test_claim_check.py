"""Tests for the claim check of a play session."""
from typing import Any, Dict, List

from predicators import utils
from predicators.agent_sdk.tools.claim_check import AFTER_RESET, LOW_P_HAT, \
    READY_TO_WIN, ClaimCheck


def _args(**changes: Any) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "claim":
        "the vertical face can never be reached",
        "rules_out":
        "gluing the end faces",
        "direct_tests": [{
            "recording": 0,
            "what": "one dab at mid height left no bond"
        }],
        "what_varied":
        "nothing",
        "rival":
        "the glue only works near the top edge",
        "next_experiment":
        "dab at three heights",
        "predictions": {
            "claim": "no bond at any height",
            "rival": "a bond near the top"
        },
    }
    args.update(changes)
    return args


def _record(check: ClaimCheck,
            *,
            level: int = 0,
            resets: int = 2,
            **changes: Any) -> Any:
    return check.record(_args(**changes),
                        recordings=2,
                        level=level,
                        resets=resets,
                        run_steps=100,
                        rehearsals=3)


def test_a_record_is_due_at_each_reset_from_the_set_one_on() -> None:
    """A training level asks at reset 2 and again at reset 3, once each."""
    utils.reset_config({"continual_claim_check_resets": 2})
    check = ClaimCheck()
    assert check.due("train", 0, 1, None) is None
    assert check.due("train", 0, 2, None) == AFTER_RESET
    check.refusal(AFTER_RESET, 2, None)
    assert _record(check)[0]
    assert check.due("train", 0, 2, None) is None
    assert check.due("train", 0, 3, None) == AFTER_RESET


def test_p_hat_triggers_fire_once_a_level() -> None:
    """A rehearsal near a win asks once a training level, and acting low asks
    once a test level; a request with no rehearsal asks neither."""
    utils.reset_config({
        "continual_claim_check_win_p_hat": 0.5,
        "continual_claim_check_low_p_hat": 0.75,
    })
    check = ClaimCheck()
    assert check.due("train", 0, 0, 0.4) is None
    assert check.due("train", 0, 0, None) is None
    assert check.due("train", 0, 0, 0.5) == READY_TO_WIN
    check.refusal(READY_TO_WIN, 0, 0.5)
    assert _record(check, resets=0)[0]
    assert check.due("train", 0, 0, 0.9) is None
    assert check.due("train", 1, 0, 0.9) == READY_TO_WIN
    assert check.due("test", 2, 0, 0.9) is None
    assert check.due("test", 2, 0, None) is None
    assert check.due("test", 2, 0, 0.5) == LOW_P_HAT
    check.resolutions[0] = {}
    check.refusal(LOW_P_HAT, 0, 0.5)
    assert _record(check, level=2, resets=0)[0]
    assert check.due("test", 2, 0, 0.5) is None


def test_a_record_needs_its_form() -> None:
    """Each field is checked, and a test must cite a recording that exists."""
    check = ClaimCheck()
    assert not _record(check)[0]  # None is due.
    check.refusal(AFTER_RESET, 2, None)
    bad_changes: List[Dict[str, Any]] = [
        {
            "claim": " "
        },
        {
            "rival": ""
        },
        {
            "next_experiment": ""
        },
        {
            "direct_tests": []
        },
        {
            "direct_tests": [{
                "recording": 2,
                "what": "x"
            }]
        },
        {
            "direct_tests": [{
                "recording": 0
            }]
        },
        {
            "direct_tests": ["recording 0"]
        },
        {
            "predictions": {
                "claim": "x"
            }
        },
    ]
    for bad in bad_changes:
        assert not _record(check, **bad)[0], bad
    assert check.pending == AFTER_RESET
    assert _record(check)[0]
    assert check.pending is None


def test_a_verdict_needs_a_run_since_the_record() -> None:
    """A verdict from an experiment needs a real step or a rehearsal after the
    record; not_run needs neither, and every other input is refused."""
    check = ClaimCheck()
    check.refusal(AFTER_RESET, 2, None)
    assert _record(check)[0]
    args = {"id": 0, "verdict": "refuted", "result": "bond at the top"}
    assert not check.resolve(args, run_steps=100, rehearsals=3)[0]
    assert not check.resolve(
        {
            "id": 7,
            **{k: v
               for k, v in args.items() if k != "id"}
        },
        run_steps=120,
        rehearsals=3)[0]
    assert not check.resolve({
        **args, "id": "x"
    }, run_steps=120, rehearsals=3)[0]
    assert not check.resolve(
        {
            **args, "verdict": "maybe"
        }, run_steps=120, rehearsals=3)[0]
    assert not check.resolve(
        {
            **args, "result": ""
        }, run_steps=120, rehearsals=3)[0]
    assert check.open_ids() == [0]
    assert check.resolve(args, run_steps=100, rehearsals=4)[0]
    assert check.open_ids() == []
    check.refusal(AFTER_RESET, 3, None)
    assert _record(check, resets=3)[0]
    assert check.resolve(
        {
            "id": 1,
            "verdict": "not_run",
            "result": "no steps left"
        },
        run_steps=100,
        rehearsals=3)[0]


def test_open_records_come_before_a_new_one() -> None:
    """A second record waits for the first to close, and the refusal says
    so."""
    check = ClaimCheck()
    check.refusal(AFTER_RESET, 2, None)
    assert _record(check)[0]
    assert "claims_resolve" in check.refusal(AFTER_RESET, 3, None)
    accepted, reply = _record(check, resets=3)
    assert not accepted and "claims_resolve" in reply


def test_a_resumed_run_reads_its_records_back(tmp_path: Any) -> None:
    """A restart keeps what was answered and what is open."""
    path = str(tmp_path / "claims.jsonl")
    check = ClaimCheck(path)
    check.refusal(AFTER_RESET, 2, None)
    assert _record(check)[0]
    check.refusal(AFTER_RESET, 3, None)
    assert check.resolve({
        "id": 0,
        "verdict": "supported",
        "result": "x"
    },
                         run_steps=120,
                         rehearsals=3)[0]
    assert _record(check, resets=3)[0]
    again = ClaimCheck(path)
    assert again.open_ids() == [1]
    assert again.due("train", 0, 3, None) is None
    assert again.due("train", 0, 4, None) == AFTER_RESET
