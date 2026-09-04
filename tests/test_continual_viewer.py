"""Tests for scripts/continual_viewer.py against a real cover run."""
import os
from typing import Any

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.controllers import create_controller
from scripts import continual_viewer as viewer


def _run(tmp_path: Any, approach_name: str, **overrides: Any) -> ContinualRun:
    utils.reset_config({
        "env":
        "cover",
        "approach":
        approach_name,
        "seed":
        3,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "horizon":
        30,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        40,
        "continual_render":
        True,
        "continual_scorecards_dir":
        os.path.join(str(tmp_path), "cards"),
        "continual_recordings_dir":
        os.path.join(str(tmp_path), "recs"),
        "experiment_id":
        "viewer",
        **overrides,
    })
    env = create_new_env("cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach(approach_name, env.predicates, options,
                               env.types, env.action_space,
                               [t.task for t in env.get_train_tasks()])
    run = ContinualRun(env, approach, create_controller(env, approach))
    run.run()
    return run


def test_pages_render_for_a_finished_run(tmp_path: Any) -> None:
    """Index, run and level pages describe an oracle run and its renders."""
    run = _run(tmp_path, "oracle")
    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"))
    run_id = run.card.run_id
    index = viewer.index_page()
    assert run_id in index
    assert "all_levels_won" in index
    assert "cover" in index and "oracle" in index

    run_html = viewer.run_page(run_id)
    assert run_html is not None
    assert "Cumulative steps vs levels won" in run_html
    assert "<svg" in run_html
    assert "/L1" in run_html and "/L2" in run_html
    points = viewer.win_points(run.card.to_dict())
    assert points[0] == (0, 0)
    assert points[-1][1] == 2

    level = viewer.level_page(run_id, 0)
    assert level is not None
    assert "level_start" in level and "invoke" in level and "win" in level
    assert level.count("class='thumb'") == \
        2 + run.card.levels[0].skill_invocations
    assert "/file/" in level
    # Atoms diffs show what each skill changed.
    assert "class='add'" in level or "class='del'" in level
    assert viewer.level_page(run_id, 5) is None
    assert viewer.run_page("no-such-run") is None
    assert viewer.level_page("no-such-run", 0) is None


def test_pages_render_for_a_capped_run(tmp_path: Any) -> None:
    """A run that hit the cap shows game overs, resets and the reason."""
    run = _run(tmp_path, "random_options", continual_render=False)
    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"))
    run_id = run.card.run_id
    assert run.card.end_reason == "step_cap"
    index = viewer.index_page()
    assert "step_cap" in index
    run_html = viewer.run_page(run_id)
    assert run_html is not None and "step_cap" in run_html
    level = viewer.level_page(run_id, 0)
    assert level is not None
    if run.card.levels[0].game_overs:
        assert "game_over" in level and "horizon" in level
        assert "reset" in level


def test_helpers() -> None:
    """Formatting helpers and the path guard."""
    assert viewer.fmt_duration(5) == "5s"
    assert viewer.fmt_duration(125) == "2m 05s"
    assert viewer.fmt_duration(3725) == "1h 02m"
    assert viewer.fmt_ts(None) == ""
    assert viewer.fmt_age(None) == ""
    assert viewer.esc("<a>") == "&lt;a&gt;"
    root = os.path.realpath("/tmp")
    assert viewer.safe_join(root, "x/y") == os.path.join(root, "x", "y")
    assert viewer.safe_join(root, "../etc/passwd") is None
    assert viewer.atoms_diff(["A()", "B()"],
                             ["B()", "C()"]).count("<span") == 2
    assert "no change" in viewer.atoms_diff(["A()"], ["A()"])
    live, cls = viewer.liveness({"end_reason": None, "updated_at": 0})
    assert live.startswith("stalled") and cls == "warn"
    assert viewer.liveness({"end_reason": "all_levels_won"}) == \
        ("all_levels_won", "ok")


def test_aggregate_scorecards(tmp_path: Any) -> None:
    """The aggregator writes the run and level CSVs and the summary."""
    # pylint: disable-next=import-outside-toplevel
    from scripts import aggregate_scorecards as agg
    run = _run(tmp_path, "oracle", continual_render=False)
    out = os.path.join(str(tmp_path), "analysis")
    paths = agg.aggregate(os.path.join(str(tmp_path), "cards"), out)
    with open(paths["runs"], "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("run_id,env,arm,seed")
    assert len(lines) == 2 and run.card.run_id in lines[1]
    with open(paths["levels"], "r", encoding="utf-8") as f:
        level_lines = f.read().splitlines()
    assert len(level_lines) == 1 + run.card.levels_total
    with open(paths["summary"], "r", encoding="utf-8") as f:
        summary = f.read()
    assert "## cover" in summary and "| oracle | 1 | 1 |" in summary


def test_agent_sessions_render(tmp_path: Any) -> None:
    """A run with an agent directory lists its transcripts, journal and
    attempts, and serves one transcript with image references linked."""
    run = _run(tmp_path, "oracle", continual_render=False)
    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"))
    run_id = run.card.run_id
    adir = os.path.join(str(tmp_path), "recs", run_id, "agent")
    os.makedirs(os.path.join(adir, "sandbox", "test_images"))
    with open(os.path.join(adir, "001_play_20260904_120000.md"),
              "w",
              encoding="utf-8") as f:
        f.write("# Session\n\nLooked at ./test_images/session_001.png\n"
                "<script>alert(1)</script>\n")
    with open(os.path.join(adir, "sandbox", "journal.md"),
              "w",
              encoding="utf-8") as f:
        f.write("### notes\nthe jug fills slowly\n")
    with open(os.path.join(adir, "sandbox", "attempts.md"),
              "w",
              encoding="utf-8") as f:
        f.write("### Session 1\n- stepped\n")
    run_html = viewer.run_page(run_id)
    assert run_html is not None
    assert "Agent sessions (1)" in run_html
    assert "001_play_20260904_120000.md" in run_html
    assert "the jug fills slowly" in run_html and "- stepped" in run_html
    session = viewer.session_page(run_id, "001_play_20260904_120000.md")
    assert session is not None
    assert "&lt;script&gt;" in session and "<script>alert" not in session
    assert "/agent/sandbox/test_images/session_001.png" in session
    assert viewer.session_page(run_id, "../../etc/passwd") is None
    assert viewer.session_page(run_id, "999_play_20260904_120000.md") is None
    assert viewer.list_session_logs(run_id)[0]["kind"] == "play"
