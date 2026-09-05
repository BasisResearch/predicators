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


def _runs_root(tmp_path: Any) -> str:
    return os.path.join(str(tmp_path), "runs")


def _run(tmp_path: Any, approach_name: str, **overrides: Any) -> ContinualRun:
    utils.reset_config({
        "env": "cover",
        "approach": approach_name,
        "seed": 3,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "horizon": 30,
        "experiment_protocol": "continual",
        "continual_steps_per_level": 40,
        "continual_render": True,
        "continual_runs_dir": _runs_root(tmp_path),
        "experiment_id": "viewer",
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


def _serve(tmp_path: Any, approaches: str = "saved") -> str:
    """Point the viewer at the test's runs root; returns the run's key."""
    viewer.configure(_runs_root(tmp_path),
                     os.path.join(str(tmp_path), approaches))
    return viewer.run_key(os.path.join(_runs_root(tmp_path)))


def _no_owners(monkeypatch: Any) -> None:
    """Neither Slurm nor a local process runs anything."""
    monkeypatch.setattr(viewer, "_squeue_rows", lambda: None)
    monkeypatch.setattr(viewer, "_ps_lines", lambda: [])


def test_pages_render_for_a_finished_run(tmp_path: Any,
                                         monkeypatch: Any) -> None:
    """Index, run and level pages describe an oracle run and its renders."""
    run = _run(tmp_path, "oracle")
    _no_owners(monkeypatch)
    _serve(tmp_path)
    key = viewer.run_key(run.run_dir)
    run_id = run.card.run_id
    # The key is the run directory under the root: approach, experiment
    # id, seed and launch stamp.
    parts = key.split("/")
    assert parts[:3] == ["oracle", "viewer", "seed3"]
    assert viewer.RUN_KEY_RE.match(key)
    assert viewer.run_dir(key) == os.path.realpath(run.run_dir)
    assert viewer.run_dir("oracle/viewer/seed3") is None
    assert viewer.run_dir("../x/seed3/run_20260101_000000") is None
    index = viewer.index_page()
    assert key in index and run_id in index
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
    assert f"data-text='{key} {run_id} " in index and "seed3" in index
    # The run column names the run by its launch stamp and seed, off its
    # directory, and carries the copy, pause and delete buttons; nothing
    # runs it, so no pause.
    name = viewer.run_name(key)
    assert name == f"{parts[3][len('run_'):]}/seed3"
    assert f">{name}</a>" in index
    assert viewer.run_name("x") == "x"
    assert "<th class='num'>seed</th>" not in index
    assert f"class='rowbtn copy' data-copy='{os.path.relpath(run.run_dir)}'" \
        in index
    assert f'deleteRun("{key}", false)' in index
    assert f'pauseRun("{key}")' not in index

    # The run page is a sidebar plus a pane the page script fills from
    # the hash route; the replay, opened at the newest attempted level,
    # is the default.
    run_html = viewer.run_page(key)
    assert run_html is not None
    assert "id='content'" in run_html and "data-default='replay/L2'" in \
        run_html
    assert f"data-run='{key}'" in run_html
    assert "href='#replay'" in run_html and "href='#L2/events'" in run_html
    assert "href='#replay/L1' data-lv='1'" in run_html
    assert "function initReplay" in run_html and "loadHash" in run_html
    # The file tree is the run directory: the scorecard beside the
    # level recordings.
    assert "L01/" in run_html and "index.jsonl" in run_html
    assert "scorecard.json" in run_html
    overview = viewer.fragment(key, "overview")
    assert overview is not None
    assert "Cumulative steps vs levels won" in overview
    assert "<svg" in overview
    assert f"href='/card/{viewer.q(key)}'" in overview
    assert "href='#L1/events'" in overview and "href='#replay/L2'" in overview

    # One replay for the whole run: the frames of both levels in order,
    # numbered run-wide, and a summary per level with its frame range.
    data = viewer.build_run_replay(key)
    n1 = len(viewer.read_index(key, 0))
    n2 = len(viewer.read_index(key, 1))
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
    # Renders are served from under the runs root, by their path in the
    # run directory.
    assert frames[0]["render"].startswith("/file/" + "/".join(
        viewer.q(p) for p in key.split("/")) + "/L01/renders/")
    replay = viewer.fragment(key, "replay")
    assert replay is not None and "id='replay-data'" in replay
    assert "id='bands'" in replay and "id='marks'" in replay
    assert "id='lvsel'" in replay and f"max='{n1 + n2 - 1}'" in replay
    assert "2 / 2 won" in replay
    assert f"href='/run/{viewer.q(key)}/replay.json'" in replay
    exported = viewer.replay_json(key)
    assert exported is not None and b'"levels"' in exported
    assert viewer.replay_json("no-such-run") is None
    assert viewer.build_run_replay("no-such-run") == {
        "levels": [],
        "frames": []
    }
    points = viewer.win_points(run.card.to_dict())
    assert points[0] == (0, 0)
    assert points[-1][1] == 2

    level = viewer.fragment(key, "L1/events")
    assert level is not None
    assert "level_start" in level and "invoke" in level and "win" in level
    assert level.count("class='thumb'") == \
        2 + run.card.levels[0].skill_invocations
    assert "/file/" in level and "href='#replay/L1'" in level
    # Atoms diffs show what each skill changed.
    assert "class='add'" in level or "class='del'" in level
    assert viewer.fragment(key, "L6/events") is None
    assert viewer.fragment(key, "L1") is None
    assert viewer.fragment(key, "bogus") is None
    assert viewer.run_page("no-such-run") is None
    assert viewer.fragment("no-such-run", "L1/events") is None
    # Files of the run: a directory listing with its renders as a
    # gallery, a text file, a binary, and no escape from the run's dir.
    listing = viewer.fragment(key, "f=L01")
    assert listing is not None and "index.jsonl" in listing
    renders = viewer.fragment(key, "f=L01/renders")
    assert renders is not None and "class='gallery'" in renders
    text = viewer.fragment(key, "f=L01/index.jsonl")
    assert text is not None and "level_start" in text and ">raw<" in text
    binary = viewer.fragment(key, "f=L01/episodes.pkl")
    assert binary is not None and "Binary file" in binary
    root = viewer.fragment(key, "f=")
    assert root is not None and "L01/" in root and "scorecard.json" in root
    assert viewer.fragment(key, "f=../../etc/passwd") is None
    assert viewer.fragment(key, "f=L01/missing.txt") is None


def test_pages_render_for_a_capped_run(tmp_path: Any) -> None:
    """A run that hit the cap shows game overs, resets and the reason."""
    run = _run(tmp_path, "random_options", continual_render=False)
    _serve(tmp_path)
    key = viewer.run_key(run.run_dir)
    assert run.card.end_reason == "step_cap"
    index = viewer.index_page()
    assert "step_cap" in index
    run_html = viewer.run_page(key)
    assert run_html is not None and "step_cap" in run_html
    level = viewer.fragment(key, "L1/events")
    assert level is not None
    if run.card.levels[0].game_overs:
        assert "game_over" in level and "horizon" in level
        assert "reset" in level
    # Every reset and game over is a mark of the run replay, in order.
    frames = viewer.build_run_replay(key)["frames"]
    marks = [f for f in frames if f["marker"]]
    assert len([f for f in marks if f["event"] == "reset"]) == \
        run.card.levels[0].resets
    assert len([f for f in marks if f["event"] == "game_over"]) == \
        len(run.card.levels[0].game_overs)
    assert [f["i"] for f in marks] == sorted(f["i"] for f in marks)


def test_every_run_of_an_experiment_is_listed(tmp_path: Any,
                                              monkeypatch: Any) -> None:
    """Two launches of one config are two run directories, and the index lists
    both under the one experiment table."""
    first = _run(tmp_path, "oracle", continual_render=False)
    second = _run(tmp_path, "oracle", continual_render=False)
    assert first.run_dir != second.run_dir
    assert os.path.dirname(first.run_dir) == os.path.dirname(second.run_dir)
    _no_owners(monkeypatch)
    _serve(tmp_path)
    cards = viewer.list_cards()
    assert sorted(c["key"] for c in cards) == sorted(
        viewer.run_key(r.run_dir) for r in (first, second))
    index = viewer.index_page()
    assert index.count("class='grp exp'") == 1
    assert index.count("class='runrow'") == 2


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
    assert viewer.run_name("a/b/seed7/run_20260904_065251") == \
        "20260904_065251/seed7"


def test_aggregate_scorecards(tmp_path: Any) -> None:
    """The aggregator writes the run and level CSVs and the summary from every
    run directory under the root."""
    # pylint: disable-next=import-outside-toplevel
    from scripts import aggregate_scorecards as agg
    run = _run(tmp_path, "oracle", continual_render=False)
    out = os.path.join(str(tmp_path), "analysis")
    paths = agg.aggregate(_runs_root(tmp_path), out)
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
    _serve(tmp_path)
    key = viewer.run_key(run.run_dir)
    adir = os.path.join(run.run_dir, "agent")
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
    run_html = viewer.run_page(key)
    assert run_html is not None
    assert "Agent rounds (1)" in run_html
    assert "href='#session/001_play_20260904_120000.md'" in run_html
    # The sandbox files sit in the sidebar tree and open in the pane.
    assert "journal.md" in run_html and "attempts.md" in run_html
    journal = viewer.fragment(key, "f=agent/sandbox/journal.md")
    assert journal is not None and "the jug fills slowly" in journal
    sandbox = viewer.fragment(key, "f=agent/sandbox")
    assert sandbox is not None and "href='#f=agent/sandbox/attempts.md'" in \
        sandbox
    agent = viewer.fragment(key, "f=agent")
    assert agent is not None and \
        "href='#session/001_play_20260904_120000.md'" in agent
    session = viewer.fragment(key, "session/001_play_20260904_120000.md")
    assert session is not None
    assert "&lt;script&gt;" in session and "<script>alert" not in session
    assert "/agent/sandbox/test_images/session_001.png" in session
    assert session.count("/file/" + "/".join(
        viewer.q(p) for p in key.split("/")) + "/agent/sandbox/") >= 1
    assert viewer.fragment(key, "session/../../etc/passwd") is None
    # An old card with fresh agent files is a run in a learning session.
    label, cls = viewer.liveness({
        "end_reason": None,
        "updated_at": 0,
        "key": key
    })
    assert label == "live (agent session)" and cls == "live"
    assert viewer.liveness({
        "end_reason": None,
        "updated_at": 0,
        "key": "no-such"
    })[0].startswith("stalled")
    assert viewer.fragment(key, "session/999_play_20260904_120000.md") is None
    assert viewer.list_session_logs(key)[0]["kind"] == "play"


def test_owners_pause_and_delete(tmp_path: Any, monkeypatch: Any) -> None:
    """Slurm jobs and local processes are attributed to a run by experiment id,
    seed, env and approach; pause cancels them; delete removes the run
    directory and the approach checkpoints, a live run only with kill."""
    run = _run(tmp_path, "oracle", continual_render=False)
    run_id = run.card.run_id
    approaches = os.path.join(str(tmp_path), "saved")
    os.makedirs(approaches)
    ckpt = os.path.join(approaches, run_id + ".saved_0.Oracle")
    kept = os.path.join(approaches, run_id + "2.saved_0.Oracle")
    for path in (ckpt, kept):
        with open(path, "w", encoding="utf-8") as f:
            f.write("x")
    _serve(tmp_path)
    key = viewer.run_key(run.run_dir)
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
    assert viewer.owners_for_run(key) == [
        viewer.Owner("job", "7_3", "R"),
        viewer.Owner("job", "9", "R"),
        viewer.Owner("proc", "4242", "R"),
    ]
    assert not viewer.owners_for_run("no-such")
    index = viewer.index_page()
    assert f'pauseRun("{key}")' in index
    assert f'deleteRun("{key}", true)' in index
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
    ok, msg = viewer.pause_run(key)
    assert ok and "auto_resume" in msg and "7_3, 9" in msg
    assert cancelled == [["7_3", "9"]] and signalled == [(["4242"], 15)]
    ok, msg = viewer.delete_run(key, kill=False)
    assert not ok and "live" in msg
    assert viewer.load_card(key) is not None
    # With kill, the owners are stopped, waited out, and the run directory
    # goes; the other run's checkpoint stays.
    calls = {"n": 0}

    def leaving() -> Optional[List[List[str]]]:
        calls["n"] += 1
        return rows if calls["n"] == 1 else []

    monkeypatch.setattr(viewer, "_squeue_rows", leaving)
    monkeypatch.setattr(viewer, "_ps_lines", lambda: [])
    ok, msg = viewer.delete_run(key, kill=True)
    assert ok, msg
    assert "run directory" in msg and "1 checkpoint" in msg
    assert cancelled[-1] == ["7_3", "9"]
    assert viewer.load_card(key) is None
    assert not os.path.exists(run.run_dir)
    assert not os.path.exists(ckpt) and os.path.exists(kept)
    assert viewer.delete_run(key, kill=False) == (False, "no such run")
    assert viewer.pause_run(key) == (False, "no such run")
    assert viewer.delete_run("../runs", kill=False) == \
        (False, "not a run key")
    assert viewer.delete_run("oracle/viewer/seed3", kill=False) == \
        (False, "not a run key")
    assert viewer.delete_run("", kill=False) == (False, "not a run key")


def _call(host: str,
          method: str,
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


def test_http_endpoints(tmp_path: Any, monkeypatch: Any) -> None:
    """The server answers GETs with the run key as one URL segment, refuses
    cross-origin POSTs, and pauses or deletes over POST only."""
    run = _run(tmp_path, "oracle", continual_render=False)
    _no_owners(monkeypatch)
    _serve(tmp_path)
    key = viewer.run_key(run.run_dir)
    qkey = viewer.q(key)
    assert "%2F" in qkey
    server = ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"127.0.0.1:{server.server_address[1]}"
    try:
        status, body = _call(host, "GET", "/")
        assert status == 200 and key in body
        assert _call(host, "GET", f"/run/{qkey}")[0] == 200
        assert _call(host, "GET", f"/run/{qkey}/frag/replay")[0] == 200
        assert _call(host, "GET", f"/run/{qkey}/replay.json")[0] == 200
        assert _call(host, "GET", f"/run/{qkey}/L1/replay")[0] == 200
        status, body = _call(host, "GET", f"/card/{qkey}")
        assert status == 200 and '"run_id"' in body
        assert _call(host, "GET", "/card/nope")[0] == 404
        # The run's files are served by their path under the root; the
        # unencoded key is not a run page.
        status, body = _call(host, "GET", f"/file/{key}/L01/index.jsonl")
        assert status == 200 and "level_start" in body
        assert _call(host, "GET", f"/run/{key}")[0] == 404
        assert _call(host, "POST", f"/delete?r={qkey}",
                     {"Origin": "http://evil.example"})[0] == 403
        assert _call(host, "GET", f"/delete?r={qkey}")[0] == 404
        assert _call(host, "POST", "/nope")[0] == 404
        status, body = _call(host, "POST", f"/pause?r={qkey}")
        assert status == 409 and "no queued" in body
        status, body = _call(host, "POST", f"/delete?r={qkey}",
                             {"Origin": f"http://{host}"})
        assert status == 200 and "run directory" in body
        assert _call(host, "GET", f"/run/{qkey}")[0] == 404
        assert viewer.load_card(key) is None
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


def test_video_route_and_range_requests(tmp_path: Any,
                                        monkeypatch: Any) -> None:
    """A run whose directory holds run.mp4 gets a Video link, a player
    fragment, and the file streams over /video/<key> with HTTP Range
    support."""
    run = _run(tmp_path, "oracle", continual_render=False)
    _no_owners(monkeypatch)
    _serve(tmp_path)
    key = viewer.run_key(run.run_dir)
    assert viewer.video_path(key) is None
    fragment = viewer.video_fragment(key) or ""
    assert "No video yet" in fragment
    assert f"--run_dir {os.path.relpath(run.run_dir)}" in fragment
    assert "#video" not in (viewer.run_page(key) or "")
    payload = bytes(range(256)) * 4
    video = os.path.join(run.run_dir, "run.mp4")
    with open(video, "wb") as f:
        f.write(payload)
    assert viewer.video_path(key) == os.path.realpath(video)
    assert "<video" in (viewer.video_fragment(key) or "")
    assert "<video" in (viewer.file_fragment(key, "run.mp4") or "")
    assert "#video" in (viewer.run_page(key) or "")
    assert "labelled replay" in (viewer.overview_fragment(key) or "")
    assert viewer.fragment(key, "video") == viewer.video_fragment(key)
    assert viewer.video_fragment("nope") is None

    server = ThreadingHTTPServer(("127.0.0.1", 0), viewer.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = f"127.0.0.1:{server.server_address[1]}"
    url = f"/video/{viewer.q(key)}"

    def get(path: str, headers: Optional[Dict[str, str]] = None) -> Any:
        req = urllib.request.Request(f"http://{host}{path}",
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    try:
        status, headers, body = get(url)
        assert status == 200 and body == payload
        assert headers["Content-Type"] == "video/mp4"
        assert headers["Accept-Ranges"] == "bytes"
        status, headers, body = get(url, {"Range": "bytes=10-19"})
        assert status == 206 and body == payload[10:20]
        assert headers["Content-Range"] == f"bytes 10-19/{len(payload)}"
        status, _, body = get(url, {"Range": "bytes=1000-"})
        assert status == 206 and body == payload[1000:]
        status, _, body = get(url, {"Range": "bytes=-16"})
        assert status == 206 and body == payload[-16:]
        status, _, _ = get(url, {"Range": "bytes=5000-6000"})
        assert status == 416
        assert get("/video/nope")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
