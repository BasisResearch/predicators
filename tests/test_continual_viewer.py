"""Tests for scripts/continual_viewer.py against a real cover run."""
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

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


def _no_owners(monkeypatch: Any) -> None:
    """Neither Slurm nor a local process runs anything."""
    monkeypatch.setattr(viewer, "_squeue_rows", lambda: None)
    monkeypatch.setattr(viewer, "_ps_lines", lambda: [])


def test_pages_render_for_a_finished_run(tmp_path: Any,
                                         monkeypatch: Any) -> None:
    """Index, run and level pages describe an oracle run and its renders."""
    run = _run(tmp_path, "oracle")
    _no_owners(monkeypatch)
    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"))
    run_id = run.card.run_id
    index = viewer.index_page()
    assert run_id in index
    assert "all_levels_won" in index
    assert "cover" in index and "oracle" in index
    # The runs table nests under agent headers (rendered) and env headers
    # (empty hosts the page script fills), with the group toggle and
    # filter in the top bar.
    assert "data-kind='agent' data-name='oracle'" in index
    assert "data-kind='env' data-name='cover'" in index
    assert "data-agent='oracle' data-env='cover' data-config='viewer'" in \
        index
    assert index.count("class='grp exp'") == 1
    # The leaf header names the experiment id, then the inner level.
    assert "<summary>viewer <span class='muted'><span class='lbl env'>" in \
        index
    # A fixed-layout grid with one shared column layout and a per-level
    # inner grid, so columns and levels line up across every group.
    assert "class='grid runs'" in index and "<colgroup>" in index
    assert index.count("class='lvgrid'") == 1
    assert index.count(f"<col style='width:{viewer.LEVEL_COL_W}px'>") == \
        run.card.levels_total
    assert "id='groupbtn'" in index and "id='runfilter'" in index
    assert f"data-text='{run_id} " in index and "seed3" in index
    # The run column names the run by start stamp and seed and carries
    # the copy, pause and delete buttons; nothing runs it, so no pause.
    stamp = viewer.run_stamp(run.card.to_dict())
    assert len(stamp) == 15 and stamp[8] == "_"
    assert f">{stamp}/seed3</a>" in index and "viewer/seed3" not in index
    assert viewer.run_stamp({"run_id": "x"}) == "x"
    assert "<th class='num'>seed</th>" not in index
    rec = os.path.relpath(os.path.join(str(tmp_path), "recs", run_id))
    assert f"class='rowbtn copy' data-copy='{rec}'" in index
    assert f'deleteRun("{run_id}", false)' in index
    assert f'pauseRun("{run_id}")' not in index

    # The run page is a sidebar plus a pane the page script fills from
    # the hash route; the replay, opened at the newest attempted level,
    # is the default.
    run_html = viewer.run_page(run_id)
    assert run_html is not None
    assert "id='content'" in run_html and "data-default='replay/L2'" in \
        run_html
    assert "href='#replay'" in run_html and "href='#L2/events'" in run_html
    assert "href='#replay/L1' data-lv='1'" in run_html
    assert "function initReplay" in run_html and "loadHash" in run_html
    assert "L01/" in run_html and "index.jsonl" in run_html
    overview = viewer.fragment(run_id, "overview")
    assert overview is not None
    assert "Cumulative steps vs levels won" in overview
    assert "<svg" in overview
    assert "href='#L1/events'" in overview and "href='#replay/L2'" in overview

    # One replay for the whole run: the frames of both levels in order,
    # numbered run-wide, and a summary per level with its frame range.
    data = viewer.build_run_replay(run_id)
    n1 = len(viewer.read_index(run_id, 0))
    n2 = len(viewer.read_index(run_id, 1))
    assert [lv["k"] for lv in data["levels"]] == [1, 2]
    assert data["levels"][0] == {
        "k": 1,
        "start": 0,
        "end": n1 - 1,
        "won": True,
        "lost": False,
        "split": "train",
        "task_idx": run.card.levels[0].task_idx
    }
    assert data["levels"][1]["start"] == n1
    assert data["levels"][1]["end"] == n1 + n2 - 1
    frames = data["frames"]
    assert [f["i"] for f in frames] == list(range(n1 + n2))
    assert [f["level"] for f in frames] == [1] * n1 + [2] * n2
    assert frames[0]["event"] == "level_start" and frames[0]["marker"]
    assert frames[n1 - 1]["event"] == "win" and frames[n1]["level"] == 2
    assert all(f["render"] for f in frames)
    replay = viewer.fragment(run_id, "replay")
    assert replay is not None and "id='replay-data'" in replay
    assert "id='bands'" in replay and "id='marks'" in replay
    assert "id='lvsel'" in replay and f"max='{n1 + n2 - 1}'" in replay
    assert "2 / 2 won" in replay
    assert f"href='/run/{run_id}/replay.json'" in replay
    exported = viewer.replay_json(run_id)
    assert exported is not None and b'"levels"' in exported
    assert viewer.replay_json("no-such-run") is None
    assert viewer.build_run_replay("no-such-run") == {
        "levels": [],
        "frames": []
    }
    points = viewer.win_points(run.card.to_dict())
    assert points[0] == (0, 0)
    assert points[-1][1] == 2

    level = viewer.fragment(run_id, "L1/events")
    assert level is not None
    assert "level_start" in level and "invoke" in level and "win" in level
    assert level.count("class='thumb'") == \
        2 + run.card.levels[0].skill_invocations
    assert "/file/" in level and "href='#replay/L1'" in level
    # Atoms diffs show what each skill changed.
    assert "class='add'" in level or "class='del'" in level
    assert viewer.fragment(run_id, "L6/events") is None
    assert viewer.fragment(run_id, "L1") is None
    assert viewer.fragment(run_id, "bogus") is None
    assert viewer.run_page("no-such-run") is None
    assert viewer.fragment("no-such-run", "L1/events") is None
    # Files of the recording: a directory listing with its renders as a
    # gallery, a text file, a binary, and no escape from the run's dir.
    listing = viewer.fragment(run_id, "f=L01")
    assert listing is not None and "index.jsonl" in listing
    renders = viewer.fragment(run_id, "f=L01/renders")
    assert renders is not None and "class='gallery'" in renders
    text = viewer.fragment(run_id, "f=L01/index.jsonl")
    assert text is not None and "level_start" in text and ">raw<" in text
    binary = viewer.fragment(run_id, "f=L01/episodes.pkl")
    assert binary is not None and "Binary file" in binary
    root = viewer.fragment(run_id, "f=")
    assert root is not None and "L01/" in root
    assert viewer.fragment(run_id, "f=../../etc/passwd") is None
    assert viewer.fragment(run_id, "f=L01/missing.txt") is None


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
    level = viewer.fragment(run_id, "L1/events")
    assert level is not None
    if run.card.levels[0].game_overs:
        assert "game_over" in level and "horizon" in level
        assert "reset" in level
    # Every reset and game over is a mark of the run replay, in order.
    frames = viewer.build_run_replay(run_id)["frames"]
    marks = [f for f in frames if f["marker"]]
    assert len([f for f in marks if f["event"] == "reset"]) == \
        run.card.levels[0].resets
    assert len([f for f in marks if f["event"] == "game_over"]) == \
        len(run.card.levels[0].game_overs)
    assert [f["i"] for f in marks] == sorted(f["i"] for f in marks)


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
    assert viewer.liveness({
        "end_reason": None,
        "updated_at": 0
    }, [viewer.Owner("job", "1_0", "PD")]) == ("queued", "warn")
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
    assert "href='#session/001_play_20260904_120000.md'" in run_html
    # The sandbox files sit in the sidebar tree and open in the pane.
    assert "journal.md" in run_html and "attempts.md" in run_html
    journal = viewer.fragment(run_id, "f=agent/sandbox/journal.md")
    assert journal is not None and "the jug fills slowly" in journal
    sandbox = viewer.fragment(run_id, "f=agent/sandbox")
    assert sandbox is not None and "href='#f=agent/sandbox/attempts.md'" in \
        sandbox
    agent = viewer.fragment(run_id, "f=agent")
    assert agent is not None and \
        "href='#session/001_play_20260904_120000.md'" in agent
    session = viewer.fragment(run_id, "session/001_play_20260904_120000.md")
    assert session is not None
    assert "&lt;script&gt;" in session and "<script>alert" not in session
    assert "/agent/sandbox/test_images/session_001.png" in session
    assert viewer.fragment(run_id, "session/../../etc/passwd") is None
    # An old card with fresh agent files is a run in a learning session.
    label, cls = viewer.liveness({
        "end_reason": None,
        "updated_at": 0,
        "run_id": run_id
    })
    assert label == "live (agent session)" and cls == "live"
    assert viewer.liveness({
        "end_reason": None,
        "updated_at": 0,
        "run_id": "no-such"
    })[0].startswith("stalled")
    assert viewer.fragment(run_id,
                           "session/999_play_20260904_120000.md") is None
    assert viewer.list_session_logs(run_id)[0]["kind"] == "play"


def test_owners_pause_and_delete(tmp_path: Any, monkeypatch: Any) -> None:
    """Slurm jobs and local processes are attributed to a run by experiment id,
    seed, env and approach; pause cancels them; delete removes the scorecard,
    recording and approach checkpoints, a live run only with kill."""
    run = _run(tmp_path, "oracle", continual_render=False)
    run_id = run.card.run_id
    approaches = os.path.join(str(tmp_path), "saved")
    os.makedirs(approaches)
    ckpt = os.path.join(approaches, run_id + ".saved_0.Oracle")
    kept = os.path.join(approaches, run_id + "2.saved_0.Oracle")
    for path in (ckpt, kept):
        with open(path, "w", encoding="utf-8") as f:
            f.write("x")
    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"), approaches)
    # squeue rows: id, array task, plain id, state, name, stdout pattern.
    pattern = "/x/logs/cover__oracle__viewer__%a__%j.log"
    rows = [
        ["7_3", "3", "70", "R", "viewer", pattern],  # this run
        ["7_4", "4", "71", "R", "viewer", pattern],  # another seed
        ["8_3", "3", "80", "PD", "other", pattern],  # another experiment
        [
            "9", "N/A", "9", "R", "viewer",
            "/x/logs/cover__oracle__viewer__3__9.log"
        ],  # seed from stdout
        [
            "10_3", "3", "100", "R", "viewer",
            "/x/logs/cover__random_options__viewer__%a__%j.log"
        ],  # other arm
    ]
    procs = [
        "4242 python predicators/main.py --env cover --approach oracle "
        "--seed 3 --experiment_id viewer --experiment_protocol continual",
        "4243 python predicators/main.py --env cover --approach oracle "
        "--seed 5 --experiment_id viewer",
        "4244 /bin/bash -c grep main.py --env cover --approach oracle "
        "--seed 3 --experiment_id viewer",
    ]
    monkeypatch.setattr(viewer, "_squeue_rows", lambda: rows)
    monkeypatch.setattr(viewer, "_ps_lines", lambda: procs)
    monkeypatch.setattr(viewer, "_descendant_pids", lambda pids: [])
    assert viewer.owners_for_run(run_id) == [
        viewer.Owner("job", "7_3", "R"),
        viewer.Owner("job", "9", "R"),
        viewer.Owner("proc", "4242", "R"),
    ]
    assert not viewer.owners_for_run("no-such")
    index = viewer.index_page()
    assert f'pauseRun("{run_id}")' in index
    assert f'deleteRun("{run_id}", true)' in index
    assert "Slurm job 7_3 (R), Slurm job 9 (R), pid 4242" in index
    cancelled: List[List[str]] = []
    signalled: List[Tuple[List[str], int]] = []

    def fake_scancel(jobs: List[str]) -> Tuple[bool, str]:
        cancelled.append(list(jobs))
        return True, "cancelled Slurm job(s) " + ", ".join(jobs)

    monkeypatch.setattr(viewer, "_scancel", fake_scancel)
    monkeypatch.setattr(
        viewer, "_signal_pids", lambda pids, sig: signalled.append(
            (list(pids), int(sig))))
    ok, msg = viewer.pause_run(run_id)
    assert ok and "auto_resume" in msg and "7_3, 9" in msg
    assert cancelled == [["7_3", "9"]] and signalled == [(["4242"], 15)]
    ok, msg = viewer.delete_run(run_id, kill=False)
    assert not ok and "live" in msg
    assert viewer.load_card(run_id) is not None
    # With kill, the owners are stopped, waited out, and the files go;
    # the other run's checkpoint stays.
    calls = {"n": 0}

    def leaving() -> Optional[List[List[str]]]:
        calls["n"] += 1
        return rows if calls["n"] == 1 else []

    monkeypatch.setattr(viewer, "_squeue_rows", leaving)
    monkeypatch.setattr(viewer, "_ps_lines", lambda: [])
    ok, msg = viewer.delete_run(run_id, kill=True)
    assert ok, msg
    assert "scorecard" in msg and "recording" in msg and "1 checkpoint" in msg
    assert cancelled[-1] == ["7_3", "9"]
    assert viewer.load_card(run_id) is None
    assert not os.path.isdir(os.path.join(str(tmp_path), "recs", run_id))
    assert not os.path.exists(ckpt) and os.path.exists(kept)
    assert viewer.delete_run(run_id, kill=False) == (False, "no such run")
    assert viewer.pause_run(run_id) == (False, "no such run")
    assert viewer.delete_run("../cards", kill=False) == (False, "not a run id")
    assert viewer.delete_run("", kill=False) == (False, "not a run id")


def test_http_endpoints(tmp_path: Any, monkeypatch: Any) -> None:
    """The server answers GETs, refuses cross-origin POSTs, and pauses or
    deletes over POST only."""
    run = _run(tmp_path, "oracle", continual_render=False)
    run_id = run.card.run_id
    _no_owners(monkeypatch)
    viewer.configure(os.path.join(str(tmp_path), "cards"),
                     os.path.join(str(tmp_path), "recs"),
                     os.path.join(str(tmp_path), "saved"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"127.0.0.1:{server.server_address[1]}"

    def call(method: str,
             path: str,
             headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        req = urllib.request.Request(f"http://{host}{path}",
                                     method=method,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    try:
        status, body = call("GET", "/")
        assert status == 200 and run_id in body
        assert call("GET", f"/run/{run_id}")[0] == 200
        assert call("GET", f"/run/{run_id}/frag/replay")[0] == 200
        assert call("GET", f"/run/{run_id}/replay.json")[0] == 200
        assert call("GET", f"/run/{run_id}/L1/replay")[0] == 200  # redirect
        assert call("GET", "/card/nope")[0] == 404
        assert call("POST", f"/delete?r={run_id}",
                    {"Origin": "http://evil.example"})[0] == 403
        assert call("GET", f"/delete?r={run_id}")[0] == 404
        assert call("POST", "/nope")[0] == 404
        status, body = call("POST", f"/pause?r={run_id}")
        assert status == 409 and "no queued" in body
        status, body = call("POST", f"/delete?r={run_id}",
                            {"Origin": f"http://{host}"})
        assert status == 200 and "scorecard" in body
        assert call("GET", f"/run/{run_id}")[0] == 404
        assert viewer.load_card(run_id) is None
    finally:
        server.shutdown()
        server.server_close()


def test_level_status_marks_lost_levels() -> None:
    """A level that ended in GAME_OVER with no reset available reads as lost
    everywhere the viewer names a level's state."""
    lost = {"index": 1, "split": "test", "attempted": True, "lost": True}
    assert viewer._level_status(lost) == ("lost", "bad")  # pylint: disable=protected-access
    mark = viewer._level_mark(lost)  # pylint: disable=protected-access
    assert "✗" in mark and "bad" in mark and "lost" in mark
    assert viewer._level_status({"index": 0, "won": True}) == ("won", "ok")  # pylint: disable=protected-access
    assert viewer._level_status({"index": 0, "attempted": True}) == \
        ("in progress", "warn")  # pylint: disable=protected-access
    assert viewer._level_status({"index": 0}) == ("not attempted", "")  # pylint: disable=protected-access
    assert viewer.liveness({"end_reason": "level_lost"}) == \
        ("level_lost", "bad")
