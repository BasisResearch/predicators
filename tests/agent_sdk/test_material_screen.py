"""Tests for the screen of the materials a model does not declare."""
import json

from predicators.agent_sdk import material_screen
from predicators.code_sim_learning.fit_space import ParamSpec

_SPECS = [
    # The scene's value sits on the low end: one run, at the high end.
    ParamSpec("ball_rolling_friction", 0.0, lo=0.0, hi=0.02),
    ParamSpec("ball_mass", 0.3, lo=0.1, hi=0.9, scale="log"),
    ParamSpec("ramp_restitution", 0.0, lo=0.0, hi=0.5),
]
_OPEN = {"lost": False, "success": False, "failure": None}
_LOST = {"lost": True, "success": False, "failure": None, "lost_line": 2}
_WON = {"lost": False, "success": True, "failure": None}
_STALLED = {"lost": False, "success": False, "failure": "step 1 (Push): x"}


def test_screen_points_cover_each_end_the_scene_value_is_not_on():
    """The scene's own values first, then one point per end of each range,
    without the ends the scene's value sits on."""
    assert material_screen.screen_points(_SPECS) == [
        {},
        {
            "ball_rolling_friction": 0.02
        },
        {
            "ball_mass": 0.1
        },
        {
            "ball_mass": 0.9
        },
        {
            "ramp_restitution": 0.5
        },
    ]
    assert material_screen.screen_points([]) == [{}]


def test_outcome_ranks_a_loss_over_a_win_over_a_stall():
    """A run that ended in GAME_OVER is lost whatever else it did."""
    assert material_screen.outcome(_OPEN) == "open"
    assert material_screen.outcome(_WON) == "won"
    assert material_screen.outcome(_STALLED) == "stalled"
    assert material_screen.outcome({**_STALLED, "lost": True}) == "lost"
    assert material_screen.outcome({**_WON, "lost": True}) == "lost"
    # An unscored run (no evaluator verdict) is judged by its success.
    assert material_screen.outcome({"lost": None, "success": True}) == "won"


def test_note_names_the_materials_the_outcome_changes_with():
    """A material is named when the plan's outcome at an end of its range
    differs from the outcome at the scene's values; the others are not."""
    points = material_screen.screen_points(_SPECS)
    runs = [_LOST, _OPEN, _LOST, _WON, _LOST]
    changed = material_screen.changed_materials(_SPECS, points, runs)
    assert [(spec.name, [value for value, _ in ends])
            for spec, ends in changed] == [("ball_rolling_friction", [0.02]),
                                           ("ball_mass", [0.9])]
    note = material_screen.screen_note(_SPECS, points, runs)
    assert note is not None
    assert "hold the 3 materials that simulator.py does not declare" in note
    assert "for 3 of them" in note
    assert "At the scene's values the plan ends the episode in GAME_OVER " \
        "during line 3." in note
    assert "The outcome changes with 2 of them:" in note
    assert "- `ball_rolling_friction` (scene value 0): at 0.02 the plan " \
        "runs to its end without the goal and without a GAME_OVER." in note
    assert "- `ball_mass` (scene value 0.3): at 0.9 the plan reaches the " \
        "goal." in note
    assert "ramp_restitution" not in note
    assert "Declare each of these in simulator.py." in note
    # Both ends of one material, and a stalled skill.
    runs = [_OPEN, _OPEN, _STALLED, _LOST, _OPEN]
    note = material_screen.screen_note(_SPECS, points, runs)
    assert note is not None and "The outcome changes with 1 of them:" in note
    assert "at 0.1 the plan stops on a failed skill (step 1 (Push): x); " \
        "at 0.9 the plan ends the episode in GAME_OVER during line 3." in note


def test_no_note_when_no_outcome_changes_or_the_runs_stop_short():
    """The same outcome at every point finds nothing; a point the time budget
    left without a run counts for nothing."""
    points = material_screen.screen_points(_SPECS)
    assert material_screen.screen_note(_SPECS, points, [_OPEN] * 5) is None
    assert material_screen.screen_note(_SPECS, points, []) is None
    assert material_screen.screen_note(_SPECS, points, [_OPEN, _OPEN]) is None
    note = material_screen.screen_note(_SPECS, points, [_OPEN, _LOST])
    assert note is not None and "for 1 of them" in note


def test_note_lists_the_names_past_the_described_materials():
    """Past the materials a note describes one by one, it lists names."""
    specs = [
        ParamSpec(f"m{i}_restitution", 0.0, lo=0.0, hi=0.5) for i in range(10)
    ]
    points = material_screen.screen_points(specs)
    note = material_screen.screen_note(specs, points, [_OPEN] + [_LOST] * 10)
    assert note is not None and note.count("(scene value 0)") == 8
    assert "- the same holds for `m8_restitution`, `m9_restitution`." in note


def test_log_screen_appends_one_row(tmp_path):
    """Each screen leaves its points, their outcomes, its time and its note in
    material_screens.jsonl; no log folder, no row."""
    points = material_screen.screen_points(_SPECS)
    runs = [_LOST, _OPEN]
    material_screen.log_screen(None, "Plan", points, runs, 1.26, None)
    material_screen.log_screen(str(tmp_path), "Plan", points, runs, 1.26,
                               "a note")
    rows = [
        json.loads(line)
        for line in (tmp_path /
                     "material_screens.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["plan"] == "Plan" and rows[0]["seconds"] == 1.3
    assert rows[0]["points"] == 5 and rows[0]["note"] == "a note"
    assert rows[0]["runs"] == [{
        "params": {},
        "outcome": "lost",
        "lost_line": 2,
        "failure": None
    }, {
        "params": {
            "ball_rolling_friction": 0.02
        },
        "outcome": "open",
        "lost_line": None,
        "failure": None
    }]
