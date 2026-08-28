"""Tests for the info.log parser of scripts/log_viewer.py."""

from pathlib import Path

from scripts.log_viewer import _parse_info_log

_SAVE = ("INFO: Saved local sandbox query/response to logs/x/sandbox/"
         "session_logs/{name}.md")
_RESUME = ("INFO: --auto_resume: checkpoint(s) found at saved_approaches/x"
           ".saved_* (last completed cycle: {last}); resuming with "
           "load_approach + restart_learning, skip_until_cycle={skip}.")
_CATCH_UP = ("INFO: Resumed past cycle {c} whose test never ran; testing it "
             "now before continuing.")


def _write_log(tmp_path: Path, lines: list) -> str:
    path = tmp_path / "info.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_resumed_catch_up_test_belongs_to_its_cycle(tmp_path: Path) -> None:
    """A resumed run re-tests its last completed cycle before its first banner;
    those sessions are that cycle's, not the offline phase's."""
    path = _write_log(tmp_path, [
        _RESUME.format(last=0, skip=1),
        _CATCH_UP.format(c=0),
        _SAVE.format(name="001_test_task0_20260827_171624"),
        "INFO: Tasks solved: 1 / 1",
        "ONLINE LEARNING CYCLE 1",
        _SAVE.format(name="002_explore_20260827_181820"),
        "INFO: Interaction episode on train task 0: reward=1.00, "
        "terminated=True, accepted=True",
        _SAVE.format(name="003_learn_20260827_205835"),
        _SAVE.format(name="004_test_task0_20260827_232442"),
        "INFO: Tasks solved: 0 / 1",
    ])
    parsed = _parse_info_log(path)
    assert parsed["resume_cycle"] == 1
    assert parsed["session_cycles"] == {1: 0, 2: 1, 3: 1, 4: 1}
    assert parsed["round_cycles"] == [0, 1]
    assert parsed["explore"][2]["cycle"] == 1
    assert parsed["last_cycle"] == 1


def test_fresh_run_pre_banner_sessions_stay_offline(tmp_path: Path) -> None:
    """Without the catch-up line, sessions saved before the first banner are
    the offline learn / pre-loop test (cycle None)."""
    path = _write_log(tmp_path, [
        _SAVE.format(name="001_learn_20260827_121109"),
        _SAVE.format(name="002_test_task0_20260827_130000"),
        "INFO: Tasks solved: 0 / 1",
        "ONLINE LEARNING CYCLE 0",
        _SAVE.format(name="003_explore_20260827_140000"),
    ])
    parsed = _parse_info_log(path)
    assert parsed["resume_cycle"] is None
    assert parsed["session_cycles"] == {1: None, 2: None, 3: 0}
    assert parsed["round_cycles"] == [None]
    assert parsed["last_cycle"] == 0


def test_catch_up_line_never_overrides_a_real_banner(tmp_path: Path) -> None:
    """The catch-up line only stands in for a banner while none has been seen;
    a later occurrence cannot rewind the cycle."""
    path = _write_log(tmp_path, [
        "ONLINE LEARNING CYCLE 2",
        _CATCH_UP.format(c=0),
        _SAVE.format(name="001_test_task0_20260827_171624"),
        "INFO: Tasks solved: 1 / 1",
    ])
    parsed = _parse_info_log(path)
    assert parsed["session_cycles"] == {1: 2}
    assert parsed["round_cycles"] == [2]
