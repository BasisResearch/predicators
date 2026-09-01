"""Tests for the sim.fit trimming-advice helper.

Regression for the 2026-08-31 bridge runs: every recorded segment of
three independent runs scored 0.1001-0.134 against the 0.100 trimming
threshold, and the report's only advice ("the recordings are chaotic,
collect different experiments") sent the agents into pointless
re-collection loops that burned their solve budgets. A segment a few
percent over the cutoff is a model-fidelity floor - no re-collection can
move it - and the advice must say so; only far-over segments earn the
chaotic-recording advice.
"""

from predicators.agent_sdk.tools.synthesis import _TRIM_BORDERLINE_FACTOR, \
    _trim_cause_note

_THRESHOLD = 0.1


def test_borderline_segments_get_model_fidelity_advice() -> None:
    """The 2026-08-31 pattern: everything a hair over the cutoff."""
    rms = [0.1001, 0.1025, 0.1126, 0.134]
    notes = _trim_cause_note(rms, _THRESHOLD)
    assert len(notes) == 1
    assert "model-fidelity limit" in notes[0]
    assert "re-collecting the same experiments will score the same" \
        in notes[0]
    assert "0.1001" in notes[0]  # the closest miss is named
    assert "0.1" in notes[0]  # so is the threshold
    assert "chaotic" not in notes[0].replace("not chaotic data", "")


def test_far_segments_get_chaos_advice() -> None:
    """Only far-over segments keep the chaotic-recording advice."""
    rms = [0.3037, 0.4216]
    notes = _trim_cause_note(rms, _THRESHOLD)
    assert len(notes) == 1
    assert "not repeatable under replay" in notes[0]
    assert "0.4216" in notes[0]  # the worst offender is named


def test_mixed_segments_get_both_notes_with_correct_counts() -> None:
    """Borderline and far segments each get their own counted note."""
    rms = [0.1001, 0.1126, 0.3037, 0.4216, 0.3037]
    notes = _trim_cause_note(rms, _THRESHOLD)
    assert len(notes) == 2
    assert notes[0].startswith("2 dropped segment(s)")
    assert notes[1].startswith("3 dropped segment(s)")


def test_survivors_are_ignored() -> None:
    """Values at or under the threshold were kept, not dropped: they contribute
    to neither note."""
    rms = [0.05, 0.1, 0.1126]
    notes = _trim_cause_note(rms, _THRESHOLD)
    assert len(notes) == 1
    assert notes[0].startswith("1 dropped segment(s)")


def test_no_dropped_segments_means_no_notes() -> None:
    """With nothing over the threshold there is nothing to advise on."""
    assert not _trim_cause_note([0.01, 0.02], _THRESHOLD)
    assert not _trim_cause_note([], _THRESHOLD)


def test_boundary_lands_on_the_borderline_side() -> None:
    """A segment exactly at factor x threshold is still borderline; just past
    it is chaos."""
    at_edge = _TRIM_BORDERLINE_FACTOR * _THRESHOLD
    notes = _trim_cause_note([at_edge], _THRESHOLD)
    assert len(notes) == 1
    assert "model-fidelity limit" in notes[0]
    notes = _trim_cause_note([at_edge * 1.01], _THRESHOLD)
    assert len(notes) == 1
    assert "not repeatable under replay" in notes[0]
