#!/usr/bin/env python3
"""Local web viewer for continual-protocol runs (docs/continual-protocol.md).

Stdlib-only browser over the two on-disk products of a run:

* ``scorecards/<run_id>.json``, the ``RunCard`` written by
  ``predicators/run/scorecard.py`` after every skill invocation, reset
  and level event;
* ``recordings/<run_id>/L<k>/``, the per-level recording written by
  ``predicators/run/recording.py``: ``index.jsonl`` (one line per skill
  invocation, reset, resume, win, game over and level start),
  ``actions.jsonl`` (one line per primitive step), ``renders/*.png``.

Pages:

* ``/``: every run, one table per (agent, env) pair nested under
  agent-name or env-name headers (a toggle, as in the phased log
  viewer) with a filter box: levels won, steps, resets, invocations,
  active and queue time, LLM cost, liveness. A row names its run by
  experiment id and seed and carries three buttons: copy the recording
  path, pause (cancel the run's Slurm job or local process; the run
  keeps its scorecard, recording and checkpoints, so relaunching its
  config with ``--auto_resume`` continues it) and delete (scorecard,
  recording and approach checkpoints).
* ``/run/<run_id>``: the run as a sidebar plus a content pane that the
  hash route fills. ``#overview``: metadata, the cumulative steps-
  versus-levels-won curve, one row per level with the section 4.4
  metrics and its episodes. ``#L<k>``: the level as a replay, after the
  ARC-AGI-3 replay viewer: one frame per recorded event with the
  render, the action, the agent's thinking and text that led to it, the
  tool call and its result, and the atoms that changed; keyboard
  playback (frame, ±10, marker, home/end, play/pause, jump) and a JSON
  export. ``#L<k>/events``: the level's timeline, one row per index
  event. ``#session/<name>``: one transcript as a structured
  conversation: thinking, assistant text, tool calls with their
  results, renders inline. ``#f=<path>``: any file of the recording
  (the system prompt, the sandbox's journal, attempts and data, a
  level's index and actions) or a directory as a listing with an image
  gallery.
* ``/run/<run_id>/L<k>/replay.json``: the replay frames.

Usage:
    python scripts/continual_viewer.py [--scorecards scorecards] \\
        [--recordings recordings] [--approaches saved_approaches] \\
        [--port 25152] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import html
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

# Run as a script or imported as scripts.continual_viewer: either way the
# helper module lives next to this file, under the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# pylint: disable-next=wrong-import-position
from scripts import continual_transcripts as tr  # noqa: E402

SCORECARDS_ROOT = ""  # absolute, set in main()
RECORDINGS_ROOT = ""  # absolute, set in main()
APPROACHES_ROOT = ""  # absolute, set in main(); checkpoints to delete
LIVE_WINDOW_S = 15 * 60  # a card updated within this window is "live"

# ── Small helpers ──────────────────────────────────────────────────


def esc(text: Any) -> str:
    """HTML-escape."""
    return html.escape(str(text), quote=True)


def q(text: str) -> str:
    """URL-quote one path segment."""
    return urllib.parse.quote(text, safe="")


def fmt_duration(seconds: float) -> str:
    """``1h 05m`` style."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m {int(seconds % 60):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def fmt_ts(ts: Optional[float]) -> str:
    """Local wall-clock time of a unix timestamp."""
    if not ts:
        return ""
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_age(ts: Optional[float]) -> str:
    """``3m ago``."""
    if not ts:
        return ""
    return fmt_duration(time.time() - ts) + " ago"


def safe_join(root: str, rel: str) -> Optional[str]:
    """``root/rel`` if it stays under ``root``, else ``None``."""
    path = os.path.realpath(os.path.join(root, rel))
    if path != root and not path.startswith(root + os.sep):
        return None
    return path


# ── Data access ────────────────────────────────────────────────────


def list_cards() -> List[Dict[str, Any]]:
    """Every scorecard on disk, newest update first."""
    cards: List[Dict[str, Any]] = []
    if not os.path.isdir(SCORECARDS_ROOT):
        return cards
    for name in os.listdir(SCORECARDS_ROOT):
        if not name.endswith(".json"):
            continue
        card = load_card(name[:-len(".json")])
        if card is not None:
            cards.append(card)
    cards.sort(key=lambda c: float(c.get("updated_at") or 0), reverse=True)
    return cards


def load_card(run_id: str) -> Optional[Dict[str, Any]]:
    """One scorecard as a dict, or ``None``."""
    path = safe_join(SCORECARDS_ROOT, f"{run_id}.json")
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            card: Dict[str, Any] = json.load(f)
    except (OSError, ValueError):
        return None
    card.setdefault("run_id", run_id)
    return card


def level_dir(run_id: str, level_index: int) -> Optional[str]:
    """The level's recording directory, if it exists."""
    rel = os.path.join(run_id, f"L{level_index + 1:02d}")
    path = safe_join(RECORDINGS_ROOT, rel)
    if path is None or not os.path.isdir(path):
        return None
    return path


def read_index(run_id: str, level_index: int) -> List[Dict[str, Any]]:
    """The level's index entries in order."""
    ldir = level_dir(run_id, level_index)
    if ldir is None:
        return []
    path = os.path.join(ldir, "index.jsonl")
    entries: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
    return entries


SESSION_LOG_RE = re.compile(
    r"^(\d{3})_([a-z_]+?)(?:_task\d+)?_(\d{8}_\d{6})\.md$")
IMAGE_REF_RE = re.compile(r"\./test_images/[\w./-]+\.png")
MAX_TEXT_CHARS = 60000


def agent_dir(run_id: str) -> Optional[str]:
    """The agent arm's stable directory for a run, if it exists."""
    path = safe_join(RECORDINGS_ROOT, os.path.join(run_id, "agent"))
    if path is None or not os.path.isdir(path):
        return None
    return path


def list_session_logs(run_id: str) -> List[Dict[str, Any]]:
    """The agent's session transcripts, oldest first."""
    adir = agent_dir(run_id)
    if adir is None:
        return []
    logs = []
    for name in sorted(os.listdir(adir)):
        m = SESSION_LOG_RE.match(name)
        if m is None:
            continue
        path = os.path.join(adir, name)
        logs.append({
            "name": name,
            "number": int(m.group(1)),
            "kind": m.group(2),
            "stamp": m.group(3),
            "size": os.path.getsize(path),
            "mtime": os.path.getmtime(path),
        })
    return logs


def read_agent_file(run_id: str, rel: str) -> Optional[str]:
    """A text file under the agent dir, whole."""
    adir = agent_dir(run_id)
    if adir is None:
        return None
    path = safe_join(adir, rel)
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def liveness(
    card: Dict[str, Any], owners: Sequence["Owner"] = ()) -> Tuple[str, str]:
    """(label, css class) for a card's state; ``owners`` are the Slurm jobs and
    local processes running it (see ``live_owners``)."""
    if card.get("end_reason"):
        reason = str(card["end_reason"])
        cls = "ok" if reason == "all_levels_won" else "bad"
        return reason, cls
    if owners and all(o.state == "PD" for o in owners):
        return "queued", "warn"
    age = time.time() - float(card.get("updated_at") or 0)
    if age < LIVE_WINDOW_S:
        return "live", "live"
    # Between env events the card rests, but an agent in a learning
    # session keeps writing its transcript and sandbox notes.
    agent_age = agent_activity_age(str(card.get("run_id") or ""))
    if agent_age is not None and agent_age < LIVE_WINDOW_S:
        return "live (agent session)", "live"
    return f"stalled ({fmt_age(card.get('updated_at'))})", "warn"


def agent_activity_age(run_id: str) -> Optional[float]:
    """Seconds since the agent last wrote a transcript, its session id, its
    journal or attempts record, or a session log; ``None`` without any."""
    adir = agent_dir(run_id) if run_id else None
    if adir is None:
        return None
    newest = 0.0
    candidates = [os.path.join(adir, "session_info.json")]
    for sub in ("", "sandbox", os.path.join("sandbox", "session_logs")):
        d = os.path.join(adir, sub)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        candidates += [
            os.path.join(d, n) for n in names if n.endswith((".md", ".json"))
        ]
    for path in candidates:
        try:
            newest = max(newest, os.path.getmtime(path))
        except OSError:
            pass
    return None if newest == 0.0 else time.time() - newest


def render_url(render_path: Optional[str]) -> Optional[str]:
    """The viewer URL for a render path stored in an index entry."""
    if not render_path:
        return None
    path = os.path.realpath(str(render_path))
    if not path.startswith(RECORDINGS_ROOT + os.sep):
        # A run that was moved: fall back to the basename under the
        # level's render dir, resolved by the caller.
        return None
    rel = os.path.relpath(path, RECORDINGS_ROOT)
    return "/file/" + "/".join(q(part) for part in rel.split(os.sep))


def level_render_url(run_id: str, level_index: int,
                     render_path: Optional[str]) -> Optional[str]:
    """Render URL, tolerant of a run that was moved after recording."""
    url = render_url(render_path)
    if url is not None or not render_path:
        return url
    name = os.path.basename(str(render_path))
    rel = os.path.join(run_id, f"L{level_index + 1:02d}", "renders", name)
    path = safe_join(RECORDINGS_ROOT, rel)
    if path is None or not os.path.isfile(path):
        return None
    return "/file/" + "/".join(q(part) for part in rel.split(os.sep))


# ── Page scaffold ──────────────────────────────────────────────────

CSS = """
:root {
  --bg: #ffffff; --fg: #24292f; --muted: #57606a; --border: #d0d7de;
  --panel: #f6f8fa; --accent: #0969da; --ok: #1a7f37; --bad: #cf222e;
  --warn: #9a6700; --live: #0969da; --code-bg: #f6f8fa;
  --add: #dafbe1; --del: #ffebe9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e; --border: #30363d;
    --panel: #161b22; --accent: #58a6ff; --ok: #3fb950; --bad: #f85149;
    --warn: #d29922; --live: #58a6ff; --code-bg: #161b22;
    --add: #12261e; --del: #35181a;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
  Arial, sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.topbar { position: sticky; top: 0; z-index: 10; display: flex; gap: 12px;
  align-items: center; padding: 8px 16px; background: var(--panel);
  border-bottom: 1px solid var(--border); }
.topbar h1 { font-size: 15px; margin: 0; white-space: nowrap; }
.topbar .crumb { color: var(--muted); font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.content { padding: 16px 24px; max-width: 100%; }
h2 { font-size: 16px; margin: 18px 0 8px; }
h3 { font-size: 14px; margin: 14px 0 6px; color: var(--muted); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid
  var(--border); vertical-align: top; }
th { color: var(--muted); font-weight: 600; white-space: nowrap;
  position: sticky; top: 45px; background: var(--bg); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums;
  white-space: nowrap; }
tr.event-win td { background: color-mix(in srgb, var(--ok) 10%, transparent); }
tr.event-game_over td { background: color-mix(in srgb, var(--bad) 8%,
  transparent); }
tr.event-reset td, tr.event-resume td, tr.event-level_start td {
  background: var(--panel); }
.chip { display: inline-block; padding: 0 7px; border-radius: 10px;
  border: 1px solid var(--border); font-size: 12px; white-space: nowrap;
  margin-right: 4px; }
.chip.ok { color: var(--ok); border-color: var(--ok); }
.chip.bad { color: var(--bad); border-color: var(--bad); }
.chip.warn { color: var(--warn); border-color: var(--warn); }
.chip.live { color: var(--live); border-color: var(--live); }
.muted { color: var(--muted); }
.thumb { max-height: 96px; max-width: 160px; border: 1px solid var(--border);
  border-radius: 4px; cursor: zoom-in; }
.thumb.big { max-height: none; max-width: min(90vw, 900px); cursor: zoom-out; }
.atoms { font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
  max-width: 420px; white-space: normal; }
.atoms .add { background: var(--add); padding: 0 3px; border-radius: 3px; }
.atoms .del { background: var(--del); padding: 0 3px; border-radius: 3px;
  text-decoration: line-through; }
.meta { display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px;
  font-size: 13px; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; }
.filters label { margin-right: 12px; font-size: 13px; }
svg.curve { background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; }
details summary { cursor: pointer; color: var(--muted); }
.note { max-width: 320px; white-space: pre-wrap; }
pre.doc { white-space: pre-wrap; word-break: break-word; background:
  var(--code-bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px 12px; max-height: 70vh; overflow: auto; }
/* replay */
.replay { display: grid; grid-template-columns: minmax(420px, 1fr)
  minmax(360px, 520px); gap: 18px; align-items: start; }
.stage { position: sticky; top: 56px; }
.stagehead { display: flex; gap: 10px; align-items: center; margin: 0 0 6px;
  font-variant-numeric: tabular-nums; }
.stagehead .big { font-size: 15px; font-weight: 600; }
img.frame { width: 100%; max-height: 62vh; object-fit: contain;
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; }
.noframe { width: 100%; height: 240px; display: flex; align-items: center;
  justify-content: center; color: var(--muted); background: var(--panel);
  border: 1px dashed var(--border); border-radius: 6px; }
.controls { display: flex; gap: 6px; align-items: center; margin: 8px 0;
  flex-wrap: wrap; }
.controls input[type=range] { flex: 1; min-width: 160px; }
.filmstrip { display: flex; gap: 4px; overflow-x: auto; padding: 4px 0;
  scroll-behavior: smooth; }
.filmstrip .cell { flex: 0 0 auto; width: 64px; height: 64px; border: 2px
  solid var(--border); border-radius: 4px; overflow: hidden; cursor: pointer;
  display: flex; align-items: center; justify-content: center; font-size:
  11px; color: var(--muted); background: var(--panel); }
.filmstrip .cell img { width: 100%; height: 100%; object-fit: cover; }
.filmstrip .cell.cur { border-color: var(--accent); }
.filmstrip .cell.win { border-color: var(--ok); }
.filmstrip .cell.game_over { border-color: var(--bad); }
.filmstrip .cell.reset, .filmstrip .cell.resume, .filmstrip .cell.level_start
  { border-color: var(--warn); }
.keys { font-size: 12px; }
.panel { border: 1px solid var(--border); border-radius: 6px; padding: 8px
  12px; margin-bottom: 12px; background: var(--bg); }
.panel h3 { margin: 0 0 6px; }
.panel pre { white-space: pre-wrap; word-break: break-word; margin: 4px 0;
  background: var(--code-bg); border-radius: 4px; padding: 6px 8px;
  max-height: 40vh; overflow: auto; }
blockquote.think { margin: 6px 0; padding: 6px 10px; border-left: 3px solid
  var(--muted); color: var(--muted); white-space: pre-wrap;
  word-break: break-word; font-size: 13px; }
.say { white-space: pre-wrap; word-break: break-word; margin: 6px 0; }
.turn { border-top: 1px solid var(--border); padding: 10px 0; }
.turn h3 { margin: 0 0 6px; }
.call { border: 1px solid var(--border); border-radius: 6px; padding: 6px
  10px; margin: 8px 0; background: var(--panel); }
.call.envtool { border-color: var(--accent); }
.call .name { font-weight: 600; }
.call pre { white-space: pre-wrap; word-break: break-word; margin: 4px 0;
  background: var(--code-bg); border-radius: 4px; padding: 6px 8px;
  max-height: 50vh; overflow: auto; font-size: 12px; }
.result.err { border-left: 3px solid var(--bad); }
/* index: filter, group toggle, collapsible groups */
.topbar input { flex: 1; max-width: 360px; padding: 3px 10px; border: 1px
  solid var(--border); border-radius: 6px; background: var(--bg);
  color: var(--fg); font: inherit; }
.topbar button { padding: 3px 10px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--bg); color: var(--fg); font: inherit;
  cursor: pointer; white-space: nowrap; }
details.grp { border: 1px solid var(--border); border-radius: 6px;
  margin: 10px 0; }
details.grp > summary { cursor: pointer; padding: 6px 12px; color: var(--fg);
  font-weight: 600; }
details.grp[open] > summary { border-bottom: 1px solid var(--border); }
details.grp.family > summary { font-size: 15px; }
details.grp > *:not(summary) { margin: 8px 12px; }
details.grp > table { width: calc(100% - 24px); }
details.grp.hidden, tr.hidden { display: none; }
html:not([data-group='env']) .lbl.agent { display: none; }
html[data-group='env'] .lbl.env { display: none; }
/* run page: sidebar + content pane */
.run { display: flex; height: calc(100vh - 45px); }
.sidebar { flex: 0 0 300px; overflow: auto; border-right: 1px solid
  var(--border); padding: 10px 12px; font-size: 13px;
  background: var(--panel); }
.sidebar h4 { margin: 14px 0 4px; font-size: 11px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--muted); }
.sidebar .runhead { font-weight: 600; word-break: break-all; }
.sidebar .nav a { display: block; padding: 2px 6px; border-radius: 4px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sidebar .nav a.active { background: var(--bg); font-weight: 600; }
.sidebar .nav a.sub { display: inline; padding: 0 4px; font-size: 12px; }
.sidebar .lvl { display: flex; align-items: center; }
.sidebar .lvl a:first-child { flex: 1; }
.sidebar details { margin-left: 10px; }
.sidebar details > summary { color: var(--fg); padding: 2px 0;
  white-space: nowrap; }
.sidebar details > a { margin-left: 14px; }
#content { flex: 1; min-width: 0; overflow: auto; padding: 16px 24px; }
#content th { top: 0; }
#content .stage { top: 0; }
.gallery { display: flex; flex-wrap: wrap; gap: 6px; }
/* index: fixed-layout runs grid, one shared column layout page-wide */
.tablewrap { overflow-x: auto; margin: 8px 12px; }
table.grid { border-collapse: collapse; margin: 0; }
table.grid th, table.grid td { border: 1px solid var(--border);
  padding: 4px 8px; font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; vertical-align: middle; }
table.grid th { background: var(--panel); position: static; }
table.grid.runs { table-layout: fixed; }
.runcell { display: flex; align-items: center; }
.runcell a { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.runcell .btns { flex: none; }
button.rowbtn { padding: 0 4px; margin-left: 4px; border: 1px solid
  transparent; border-radius: 4px; background: none; color: var(--muted);
  font-size: 12px; line-height: 16px; cursor: pointer; visibility: hidden; }
tr.runrow:hover button.rowbtn { visibility: visible; }
button.rowbtn:hover { color: var(--accent); border-color: var(--accent); }
button.rowbtn.del:hover { color: var(--bad); border-color: var(--bad); }
button.rowbtn.copied { color: var(--ok); visibility: visible; }
table.lvgrid { border-collapse: collapse; table-layout: fixed; margin: 0;
  width: 100%; }
table.lvgrid td { border: none; padding: 0 6px 0 0; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
"""

JS = r"""
function zoom(img) { img.classList.toggle('big'); }
function applyFilters() {
  var boxes = document.querySelectorAll('.filters input[type=checkbox]');
  var on = {};
  boxes.forEach(function (b) { on[b.value] = b.checked; });
  document.querySelectorAll('tr[data-event]').forEach(function (tr) {
    tr.style.display = on[tr.dataset.event] === false ? 'none' : '';
  });
}
function $all(s, r) { return Array.from((r || document).querySelectorAll(s)); }
// Index page: the (agent, env) leaf tables nest under agent headers
// (agent > env) or env headers (env > agent). The server renders each
// leaf once, inside the agent view, plus an empty header per env; this
// moves the leaves into the headers of the chosen view, sorted by the
// inner level's name. The mode and each group's open state are the
// user's, kept in localStorage so the auto-refresh reloads keep them.
function groupKey(d) { return 'cv-grp:' + d.dataset.key; }
function restoreGroups() {
  $all('details.grp').forEach(function (d) {
    var v = localStorage.getItem(groupKey(d));
    if (v !== null) d.open = v === '1';
  });
}
document.addEventListener('toggle', function (e) {
  var d = e.target;
  if (d.classList && d.classList.contains('grp'))
    localStorage.setItem(groupKey(d), d.open ? '1' : '0');
}, true);
function setAllGroups(open) {
  $all('details.grp').forEach(function (d) {
    d.open = open;
    localStorage.setItem(groupKey(d), open ? '1' : '0');
  });
}
function groupMode() {
  return localStorage.getItem('cv-group') === 'env' ? 'env' : 'agent';
}
function toggleGroupMode() {
  localStorage.setItem('cv-group', groupMode() === 'env' ? 'agent' : 'env');
  applyGroupMode();
}
function applyGroupMode() {
  var btn = document.getElementById('groupbtn');
  if (!btn) return;
  var mode = groupMode();
  var inner = mode === 'env' ? 'agent' : 'env';
  document.documentElement.dataset.group = mode;
  btn.textContent = mode === 'env' ? 'group: env › agent'
                                   : 'group: agent › env';
  var hosts = {};
  $all('details.grp.family').forEach(function (d) {
    hosts[d.dataset.kind + ':' + d.dataset.name] = d;
  });
  var leaves = $all('details.grp.exp');
  leaves.sort(function (a, b) {
    return a.dataset[inner] < b.dataset[inner] ? -1
         : a.dataset[inner] > b.dataset[inner] ? 1 : 0;
  });
  leaves.forEach(function (l) {
    hosts[mode + ':' + l.dataset[mode]].appendChild(l);
  });
  document.getElementById('view-agent').style.display =
    mode === 'env' ? 'none' : '';
  document.getElementById('view-env').style.display =
    mode === 'env' ? '' : 'none';
}
// The filter matches every word against a row's run id, config, arm,
// env, seed and state; groups with no visible row hide too. Per tab.
function filterRuns(text) {
  sessionStorage.setItem('cv-filter', text);
  applyRunFilter();
}
function applyRunFilter() {
  var box = document.getElementById('runfilter');
  if (!box) return;
  var text = sessionStorage.getItem('cv-filter') || '';
  box.value = text;
  var words = text.toLowerCase().split(/\s+/).filter(Boolean);
  $all('tr[data-text]').forEach(function (tr) {
    var hay = tr.dataset.text.toLowerCase();
    tr.classList.toggle('hidden', !words.every(function (w) {
      return hay.indexOf(w) >= 0;
    }));
  });
  $all('details.grp.exp').forEach(function (d) {
    d.classList.toggle('hidden', !$all('tr[data-text]', d).some(
      function (tr) { return !tr.classList.contains('hidden'); }));
  });
  $all('details.grp.family').forEach(function (d) {
    d.classList.toggle('hidden', !$all('details.grp.exp', d).some(
      function (x) { return !x.classList.contains('hidden'); }));
  });
}
// Copy-path buttons next to run names.
document.addEventListener('click', function (e) {
  var b = e.target.closest('button.rowbtn.copy');
  if (!b) return;
  var done = function () {
    b.textContent = '\u2713';
    b.classList.add('copied');
    setTimeout(function () {
      b.textContent = '\u29c9';
      b.classList.remove('copied');
    }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(b.dataset.copy).then(done);
  } else {  // non-localhost http has no clipboard API
    var ta = document.createElement('textarea');
    ta.value = b.dataset.copy;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
    done();
  }
});
// Pause / delete buttons on index run rows. POST only, so the auto-
// refresh GETs can never trip these; reload shortly after success so
// the state chip reflects the job actually leaving the queue.
function postRun(url, msg) {
  if (!confirm(msg)) return;
  fetch(url, {method: 'POST'}).then(function (r) {
    r.text().then(function (t) {
      if (!r.ok) { alert(t); return; }
      setTimeout(function () { location.reload(); }, 600);
    });
  }).catch(function (e) { alert('request failed: ' + e); });
}
function pauseRun(id) {
  postRun('/pause?r=' + encodeURIComponent(id),
          'Pause ' + id + ' ?\nIts Slurm job (or local process) is ' +
          'cancelled. The run keeps its scorecard, recording and ' +
          'checkpoints and continues when its config is relaunched ' +
          'with --auto_resume.');
}
function deleteRun(id, live) {
  var msg = live
    ? 'This run appears LIVE:\n' + id + '\nCancel its job AND delete ' +
      'its scorecard, recording and approach checkpoints?'
    : 'Delete the scorecard, recording and approach checkpoints of\n' +
      id + ' ?';
  postRun('/delete?r=' + encodeURIComponent(id) + (live ? '&kill=1' : ''),
          msg);
}
document.addEventListener('DOMContentLoaded', function () {
  restoreGroups();
  applyGroupMode();
  applyRunFilter();
});

// Run page: hash routing into the content pane. The route is the hash
// without its trailing frame or turn locator: overview, L<k> (replay),
// L<k>/events, session/<name>, f=<path>. A same-route hash change only
// moves within the loaded content; the replay updates the frame
// locator with replaceState, which fires no hashchange.
var LOADED = null;
function currentHash() { return decodeURIComponent(location.hash.slice(1)); }
function routeOf(h) {
  return h.replace(/\/frame=\d+$/, '').replace(/\/turn-\d+$/, '');
}
function loadHash() {
  var content = document.getElementById('content');
  if (!content) return;
  var h = currentHash();
  if (!h) {
    h = content.dataset.default || 'overview';
    history.replaceState(null, '', '#' + h);
  }
  var route = routeOf(h);
  if (route === LOADED) { jumpWithin(h); markNav(route); return; }
  var url = '/run/' + encodeURIComponent(content.dataset.run) + '/frag/' +
    route.split('/').map(encodeURIComponent).join('/');
  content.innerHTML = '<p class="muted">Loading…</p>';
  fetch(url).then(function (r) { return r.text(); }).then(function (html) {
    if (routeOf(currentHash()) !== route) return;  // superseded
    LOADED = route;
    stopReplay();
    content.innerHTML = html;
    var root = content.querySelector('.replay-root');
    if (root) initReplay(root, route + '/');
    markNav(route);
    restoreScroll(content);
    jumpWithin(h);
  });
}
function jumpWithin(h) {
  var t = /turn-(\d+)$/.exec(h);
  if (t) {
    var el = document.getElementById('turn-' + t[1]);
    if (el) el.scrollIntoView();
  }
  var m = /frame=(\d+)$/.exec(h);
  if (m && REPLAY) REPLAY.go(Number(m[1]));
}
function markNav(route) {
  $all('.sidebar a[href]').forEach(function (a) {
    a.classList.toggle('active', a.getAttribute('href') === '#' + route);
  });
}
// The pane's scroll position survives the auto-refresh reloads (the
// pane is an inner div, so native scroll restoration does not cover it).
function scrollKey() {
  return 'cv-scroll:' + location.pathname + '#' + routeOf(currentHash());
}
function saveScroll() {
  var c = document.getElementById('content');
  if (!c) return;
  try { sessionStorage.setItem(scrollKey(), String(c.scrollTop)); }
  catch (e) {}
}
function restoreScroll(c) {
  var v = null;
  try { v = sessionStorage.getItem(scrollKey()); } catch (e) {}
  c.scrollTop = v ? Number(v) : 0;
}
window.addEventListener('hashchange', loadHash);
window.addEventListener('beforeunload', saveScroll);
document.addEventListener('DOMContentLoaded', loadHash);

// Replay player: one per pane at a time, built from the fragment's
// frames JSON; the keyboard handler below drives whichever is active.
var REPLAY = null;
function stopReplay() { if (REPLAY) REPLAY.stop(); REPLAY = null; }
function initReplay(root, prefix) {
  var F = JSON.parse(root.querySelector('#replay-data').textContent);
  var N = F.length, cur = 0, timer = null, speed = 1;
  function el(id) { return root.querySelector('#' + id); }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g,
    function (c) { return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',
    "'":'&#39;'}[c]; }); }
  function chip(t, cls) { return "<span class='chip " + (cls||'') + "'>" +
    esc(t) + "</span>"; }
  function stateCls(st) { return st === 'WIN' ? 'ok' : st === 'GAME_OVER' ?
    'bad' : ''; }
  function statusCls(st) { return st === 'succeeded' ? 'ok' : st === 'failed'
    ? 'bad' : st === 'interrupted' ? 'warn' : ''; }
  function atomsDiff(prev, curA) {
    if (!curA) return "<span class='muted'>no atoms recorded</span>";
    var p = new Set(prev || []), c = new Set(curA), out = [];
    curA.forEach(function (a) { if (!p.has(a)) out.push(
      "<span class='add'>+" + esc(a) + "</span>"); });
    (prev || []).forEach(function (a) { if (!c.has(a)) out.push(
      "<span class='del'>" + esc(a) + "</span>"); });
    var same = curA.filter(function (a) { return p.has(a); });
    return (out.length ? out.join(' ') + '<br>' : '') +
      "<span class='muted'>" + esc(same.join(', ')) + "</span>";
  }
  function render() {
    var f = F[cur];
    el('counter').textContent = (cur + 1) + ' / ' + N;
    el('evt').innerHTML = chip(f.event,
      f.event === 'win' ? 'ok' : f.event === 'game_over' ? 'bad' : '');
    el('state').innerHTML = f.state ? chip(f.state, stateCls(f.state)) : '';
    el('steps').textContent = 'episode ' +
      (f.episode == null ? '?' : f.episode) + ' · level steps ' +
      (f.level_steps == null ? '?' : f.level_steps) + ' · run steps ' +
      (f.run_steps == null ? '?' : f.run_steps);
    var img = el('frame'), nof = el('noframe');
    if (f.render) { img.src = f.render; img.style.display = ''; nof.style.display
      = 'none'; } else { img.style.display = 'none'; nof.style.display = ''; }
    el('scrub').value = cur;
    // action panel
    var a = [];
    if (f.skill) a.push('<div><code>' + esc(f.skill) + '[' +
      (f.params || []).map(function (p) { return Number(p).toPrecision(4); })
      .join(', ') + ']</code> ' + (f.status ? chip(f.status,
      statusCls(f.status)) : '') + (f.steps != null ? ' <span class="muted">'
      + f.steps + ' steps</span>' : '') + '</div>');
    if (f.reason) a.push('<div class="muted">' + esc(f.reason) + '</div>');
    if (f.event === 'reset') a.push('<div>reset by ' + esc(f.by) + '</div>');
    if (f.event === 'resume') a.push('<div>resume: replayed ' +
      esc(f.replayed_steps) + ' steps, verified ' + esc(f.verified) + '</div>');
    if (f.event === 'level_start') a.push('<div>goal: ' +
      esc((f.goal || []).join(', ')) + '</div><div class="muted">' +
      esc(f.goal_nl) + '</div>');
    if (f.note) a.push('<div><b>note:</b> ' + esc(f.note) + '</div>');
    if ((f.expected || []).length || (f.expected_absent || []).length) {
      a.push('<div><b>expected:</b> ' + esc((f.expected || []).join(', ')) +
        ((f.expected_absent || []).length ? ' NOT ' +
        esc(f.expected_absent.join(', NOT ')) : '') + '</div>');
      if ((f.missing || []).length || (f.present || []).length)
        a.push('<div class="del">DIVERGED: missing ' +
          esc((f.missing || []).join(', ')) + (f.present && f.present.length ?
          '; present ' + esc(f.present.join(', ')) : '') + '</div>');
      else a.push('<div class="add">expected outcome held</div>');
    }
    if (f.session != null) a.push('<div class="muted">session ' + f.session +
      ', turn ' + f.turn + ' · <a href="#session/' +
      encodeURIComponent(f.session_name) + '/turn-' + f.turn +
      '">transcript</a></div>');
    el('action').innerHTML = a.join('') ||
      '<span class="muted">harness event</span>';
    // reasoning panel
    var r = [];
    (f.thinking || []).forEach(function (t) { r.push(
      '<blockquote class="think">' + esc(t) + '</blockquote>'); });
    (f.texts || []).forEach(function (t) { r.push('<div class="say">' +
      esc(t) + '</div>'); });
    el('reasoning').innerHTML = r.join('') ||
      '<span class="muted">no agent text paired with this frame</span>';
    // call panel
    var c = '';
    if (f.call) {
      c += '<div><span class="name">' + esc(f.call.name) + '</span></div>';
      c += '<pre>' + esc(JSON.stringify(f.call.args, null, 2)) + '</pre>';
      if (f.call.result) c += '<details' + (f.call.result.length < 900 ?
        ' open' : '') + '><summary>result' + (f.call.is_error ? ' (error)' :
        '') + '</summary><pre>' + esc(f.call.result) + '</pre></details>';
    }
    el('call').innerHTML = c || '<span class="muted">none</span>';
    // atoms panel
    var prev = null;
    for (var j = cur - 1; j >= 0; j--) { if (F[j].atoms) { prev = F[j].atoms;
      break; } }
    var at = '<div class="atoms">' + atomsDiff(prev, f.atoms) + '</div>';
    if (f.env_atoms) at += '<details><summary>env atoms (' +
      f.env_atoms.length + ')</summary><div class="atoms muted">' +
      esc(f.env_atoms.join(', ')) + '</div></details>';
    el('atoms').innerHTML = at;
    // filmstrip
    var cells = root.querySelectorAll('.filmstrip .cell');
    cells.forEach(function (x, i) { x.classList.toggle('cur', i === cur); });
    if (cells[cur]) cells[cur].scrollIntoView({inline: 'center', block:
      'nearest'});
    history.replaceState(null, '', '#' + prefix + 'frame=' + cur);
  }
  function go(i) { cur = Math.max(0, Math.min(N - 1, i)); render(); }
  function step(d) { go(cur + d); }
  function marker(d) {
    var i = cur + d;
    while (i >= 0 && i < N && !F[i].marker) i += d;
    if (i >= 0 && i < N) go(i);
  }
  function stop() {
    if (!timer) return;
    clearInterval(timer); timer = null;
    el('play').textContent = '▶ play';
  }
  function toggle() {
    if (timer) { stop(); return; }
    el('play').textContent = '⏸ pause';
    timer = setInterval(function () { if (cur >= N - 1) { stop(); return; }
      step(1); }, 900 / speed);
  }
  el('speed').addEventListener('change', function (e) {
    speed = Number(e.target.value); if (timer) { stop(); toggle(); } });
  el('scrub').addEventListener('input', function (e) {
    go(Number(e.target.value)); });
  REPLAY = {go: go, step: step, toggle: toggle, marker: marker, stop: stop,
            N: N};
  var m = /frame=(\d+)/.exec(location.hash);
  go(m ? Number(m[1]) : 0);
}
document.addEventListener('keydown', function (e) {
  var R = REPLAY;
  if (!R) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') { R.step(e.shiftKey ? 10 : 1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { R.step(e.shiftKey ? -10 : -1);
    e.preventDefault(); }
  else if (e.key === ']') R.marker(1);
  else if (e.key === '[') R.marker(-1);
  else if (e.key === 'Home') R.go(0);
  else if (e.key === 'End') R.go(R.N - 1);
  else if (e.key === ' ') { R.toggle(); e.preventDefault(); }
  else if (e.key === 'g' || e.key === 'G') { var v = prompt('Jump to frame ' +
    '(1-' + R.N + ')'); if (v) R.go(Number(v) - 1); }
});
"""


def page(title: str,
         crumb: str,
         body: str,
         refresh: int = 0,
         controls: str = "",
         wrap: bool = True) -> str:
    """The full HTML page; ``controls`` sits in the top bar after the crumb;
    ``wrap`` False puts the body straight under the top bar (the run page
    brings its own two-pane layout)."""
    meta = (f"<meta http-equiv='refresh' content='{refresh}'>"
            if refresh else "")
    content = f"<div class='content'>{body}</div>" if wrap else body
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{esc(title)}</title>{meta}<style>{CSS}</style>"
            f"<script>{JS}</script></head><body>"
            "<div class='topbar'><h1><a href='/'>continual viewer</a></h1>"
            f"<span class='crumb'>{crumb}</span>{controls}</div>"
            f"{content}</body></html>")


def chip(label: Any, cls: str = "", title: str = "") -> str:
    """Small labelled span."""
    return (f"<span class='chip {cls}' title='{esc(title)}'>"
            f"{esc(label)}</span>")


# ── Live jobs and processes ────────────────────────────────────────
# A run is owned by this user's Slurm job named after its experiment id
# with its seed as the array task, the launcher's convention (see
# scripts/engaging/submit_engaging_job.py; the job's stdout path spells
# <env>__<approach>__<experiment_id>__<seed>__<job>.log), or by a local
# main.py process whose flags name the same env, approach, seed and
# experiment id. Only these are ever cancelled or signalled, never a
# name-based sweep: parallel sessions share the machine.


class Owner(NamedTuple):
    """A Slurm job or a local process running a run."""
    kind: str  # "job" (ident is what scancel takes) or "proc" (a pid)
    ident: str
    state: str  # the Slurm compact state; "R" for a process


_SQUEUE_FIELDS = (("JobArrayID", 32), ("ArrayTaskID", 24), ("JobID", 24),
                  ("StateCompact", 8), ("Name", 512), ("stdout", 1024))
_PS_ARG_RE = re.compile(r"--(env|approach|seed|experiment_id)[= ]+(\S+)")
_PS_MAIN_RE = re.compile(r"^\S*python[\d.]*\s+\S*main\.py\s")
_CANCEL_WAIT_S = 5.0
_CANCEL_POLL_S = 0.5


def _squeue_rows() -> Optional[List[List[str]]]:
    """This user's queued and running Slurm jobs, one row of stripped
    ``_SQUEUE_FIELDS`` per job; ``None`` without Slurm (no squeue, or one that
    rejects the format), as opposed to ``[]``: nothing queued."""
    if shutil.which("squeue") is None:
        return None
    fmt = ",".join(f"{name}:{width}" for name, width in _SQUEUE_FIELDS)
    try:
        proc = subprocess.run(
            ["squeue", "-u",
             getpass.getuser(), "-h", "-O", fmt],
            capture_output=True,
            text=True,
            check=False,
            timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    rows: List[List[str]] = []
    for line in proc.stdout.splitlines():
        cells: List[str] = []
        pos = 0
        for _, width in _SQUEUE_FIELDS:
            cells.append(line[pos:pos + width].strip())
            pos += width
        rows.append(cells)
    return rows


def _expand_stdout(stdout: str, job_id: str, task_id: str, plain_id: str,
                   name: str) -> str:
    """A squeue stdout path with its %-specifiers substituted, as squeue can
    hand back the unexpanded sbatch pattern; "" when one is left unresolved."""
    for spec, value in (("%%", "%"), ("%A", job_id.partition("_")[0]),
                        ("%a", task_id), ("%j", plain_id or job_id),
                        ("%x", name), ("%u", getpass.getuser())):
        stdout = stdout.replace(spec, value)
    return "" if "%" in stdout else stdout


def _ps_lines() -> List[str]:
    """``pid args`` lines of this user's processes."""
    try:
        proc = subprocess.run(
            ["ps", "-u", getpass.getuser(), "-o", "pid=,args="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return proc.stdout.splitlines() if proc.returncode == 0 else []


def _card_key(card: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """(env, approach, seed, experiment id) naming a card's run; the run id
    spells ``--approach``, which the arm's name need not equal."""
    parts = str(card.get("run_id") or "").split("__")
    approach = parts[1] if len(parts) > 2 else str(card.get("arm"))
    return (str(card.get("env")), approach, str(card.get("seed")),
            str(card.get("config")))


def live_owners(cards: Sequence[Dict[str, Any]]) -> Dict[str, List[Owner]]:
    """run id -> the Slurm jobs and local processes running each card."""
    keys = {str(c["run_id"]): _card_key(c) for c in cards}
    owners: Dict[str, List[Owner]] = {}
    for job_id, task_id, plain_id, state, name, stdout in _squeue_rows() or []:
        base = os.path.basename(
            _expand_stdout(stdout, job_id, task_id, plain_id, name))
        parts = base.split("__")
        # An array job carries the seed as its task id; a single job
        # spells it in its stdout path, after the experiment id.
        seed = task_id if task_id.isdigit() else (
            parts[3] if len(parts) > 3 and parts[3].isdigit() else "")
        for run_id, (env, approach, run_seed, config) in keys.items():
            if (name, seed) != (config, run_seed):
                continue
            if len(parts) > 2 and parts[:2] != [env, approach]:
                continue
            owners.setdefault(run_id, []).append(Owner("job", job_id, state))
    for line in _ps_lines():
        pid, _, args = line.strip().partition(" ")
        args = args.strip()
        if not pid.isdigit() or not _PS_MAIN_RE.match(args):
            continue
        flags = dict(_PS_ARG_RE.findall(args))
        key = tuple(
            flags.get(k, "")
            for k in ("env", "approach", "seed", "experiment_id"))
        for run_id, run_key in keys.items():
            if key == run_key:
                owners.setdefault(run_id, []).append(Owner("proc", pid, "R"))
    return owners


def owners_for_run(run_id: str) -> List[Owner]:
    """The Slurm jobs and local processes running one run."""
    card = load_card(run_id)
    return live_owners([card]).get(run_id, []) if card else []


def _descendant_pids(pids: List[str]) -> List[str]:
    """Transitive child pids of pids among this user's processes."""
    try:
        out = subprocess.run(
            ["ps", "-u", getpass.getuser(), "-o", "pid=,ppid="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: Dict[str, List[str]] = {}
    for line in out.splitlines():
        pid, _, ppid = line.strip().partition(" ")
        children.setdefault(ppid.strip(), []).append(pid)
    seen = set(pids)
    stack = list(pids)
    found: List[str] = []
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in seen:
                seen.add(child)
                found.append(child)
                stack.append(child)
    return found


def _signal_pids(pids: Sequence[str], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(int(pid), sig)
        except (OSError, ValueError):
            pass  # already exited, or a stale ps line


def _scancel(jobs: Sequence[str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(["scancel"] + list(jobs),
                              capture_output=True,
                              text=True,
                              check=False,
                              timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"scancel failed: {e}"
    if proc.returncode != 0:
        return False, f"scancel failed: {proc.stderr.strip()}"
    return True, f"cancelled Slurm job(s) {', '.join(jobs)}"


def _stop_owners(owners: Sequence[Owner],
                 sig: int = signal.SIGTERM) -> Tuple[bool, str]:
    """Cancel the jobs and signal the process trees among ``owners``.

    Workers and helper subprocesses would survive a signal to main.py
    alone, so its whole descendant tree is signalled. scancel escalates
    to KILL on the compute node itself, so ``sig`` is for processes.
    """
    notes = []
    pids = [o.ident for o in owners if o.kind == "proc"]
    if pids:
        targets = pids + _descendant_pids(pids)
        _signal_pids(targets, sig)
        notes.append(f"sent signal {int(sig)} to {len(targets)} "
                     f"process(es) {', '.join(targets)}")
    jobs = [o.ident for o in owners if o.kind == "job"]
    if jobs:
        ok, msg = _scancel(jobs)
        if not ok:
            return False, msg
        notes.append(msg)
    return True, "; ".join(notes)


def pause_run(run_id: str) -> Tuple[bool, str]:
    """Cancel the Slurm job or signal the local process running a run.

    The run keeps its scorecard, recording and approach checkpoints, so
    relaunching its config with ``--auto_resume`` continues it from the
    last checkpoint: a pause, not a kill.
    """
    if load_card(run_id) is None:
        return False, "no such run"
    owners = owners_for_run(run_id)
    if not owners:
        return False, ("no queued or running Slurm job or local process "
                       "found for this run")
    ok, msg = _stop_owners(owners)
    if not ok:
        return False, msg
    return True, (f"{msg}; the run keeps its checkpoints and continues "
                  "when relaunched with --auto_resume")


def _remove_checkpoints(run_id: str) -> int:
    """Delete the approach checkpoints of a run; their count."""
    if not os.path.isdir(APPROACHES_ROOT):
        return 0
    removed = 0
    for name in os.listdir(APPROACHES_ROOT):
        if name.startswith(f"{run_id}.saved"):
            os.remove(os.path.join(APPROACHES_ROOT, name))
            removed += 1
    return removed


def delete_run(run_id: str, kill: bool) -> Tuple[bool, str]:
    """Delete a run's scorecard, recording and approach checkpoints.

    A run with a live job or process is refused unless ``kill`` is set:
    then it is stopped as ``pause_run`` does and waited on until it
    leaves the queue (a lingering local process gets SIGKILL) before the
    files go, since a job still running would recreate them under a
    half-deleted dir. The checkpoints go too: ``--auto_resume`` keys off
    them, and a relaunch after a delete must start fresh rather than
    load the deleted run's learned state over a new scorecard.
    """
    card_path = safe_join(SCORECARDS_ROOT, f"{run_id}.json")
    rec_dir = safe_join(RECORDINGS_ROOT, run_id)
    if (not run_id or os.path.basename(run_id) != run_id or card_path is None
            or rec_dir is None):
        return False, "not a run id"
    if not os.path.isfile(card_path) and not os.path.isdir(rec_dir):
        return False, "no such run"
    owners = owners_for_run(run_id)
    if owners:
        if not kill:
            return False, "run has a live job or process; pause it first"
        ok, msg = _stop_owners(owners)
        if not ok:
            return False, msg
        deadline = time.monotonic() + _CANCEL_WAIT_S
        while time.monotonic() < deadline and owners_for_run(run_id):
            time.sleep(_CANCEL_POLL_S)
        owners = owners_for_run(run_id)
        pids = [o.ident for o in owners if o.kind == "proc"]
        if pids:
            _signal_pids(pids + _descendant_pids(pids), signal.SIGKILL)
            time.sleep(_CANCEL_POLL_S)
        if any(o.kind == "job" for o in owners):
            return False, "Slurm job is still cancelling; retry shortly"
    removed = []
    try:
        if os.path.isfile(card_path):
            os.remove(card_path)
            removed.append("scorecard")
        if os.path.isdir(rec_dir):
            shutil.rmtree(rec_dir)
            removed.append("recording")
        n_ckpt = _remove_checkpoints(run_id)
    except OSError as e:
        return False, f"delete failed: {e}"
    if n_ckpt:
        removed.append(f"{n_ckpt} checkpoint file(s)")
    return True, "deleted " + ", ".join(removed)


# ── Index page ─────────────────────────────────────────────────────


def _totals(card: Dict[str, Any]) -> Dict[str, Any]:
    totals = card.get("totals")
    if isinstance(totals, dict):
        return totals
    levels = card.get("levels", [])
    return {
        "levels_total":
        len(levels),
        "levels_completed":
        sum(1 for lv in levels if lv.get("won")),
        "total_steps":
        sum(int(lv.get("steps", 0)) for lv in levels),
        "total_resets":
        sum(int(lv.get("resets", 0)) for lv in levels),
        "total_skill_invocations":
        sum(int(lv.get("skill_invocations", 0)) for lv in levels),
        "total_wall_clock":
        sum(float(lv.get("wall_clock", 0.0)) for lv in levels),
        "total_downtime":
        sum(float(lv.get("downtime", 0.0)) for lv in levels),
        "total_llm_cost":
        sum(
            float((lv.get("sandbox") or {}).get("llm_cost_usd", 0.0))
            for lv in levels),
    }


def index_page() -> str:
    """The runs overview: one table per (agent, env) pair, nested under agent-
    name headers or env-name headers (client-side toggle, as in the phased log
    viewer)."""
    cards = list_cards()
    if not cards:
        body = (f"<p class='muted'>No scorecards under "
                f"<code>{esc(SCORECARDS_ROOT)}</code> yet.</p>")
        return page("continual viewer", "runs", body, refresh=30)
    leaves: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for card in cards:
        key = (str(card.get("arm")), str(card.get("env")))
        leaves.setdefault(key, []).append(card)
    n_runs = {key: len(runs) for key, runs in leaves.items()}
    # One column layout for the whole page: the levels column is as wide
    # as the largest level count, so L1, L2, ... line up across groups.
    n_levels = max(len(card.get("levels", [])) for card in cards)
    owners = live_owners(cards)
    parts = [
        f"<p class='muted'>{len(cards)} run(s) under "
        f"<code>{esc(SCORECARDS_ROOT)}</code>; recordings under "
        f"<code>{esc(RECORDINGS_ROOT)}</code>. Auto-refreshes every 30 s."
        "</p>"
    ]
    # Agent view: every leaf rendered once under its agent header.
    parts.append("<div id='view-agent'>")
    for agent in sorted({a for a, _ in leaves}):
        keys = sorted(k for k in leaves if k[0] == agent)
        parts.append(
            _group_header("agent", agent, len(keys),
                          sum(n_runs[k] for k in keys)))
        for _, env in keys:
            parts.append(
                _leaf_table(agent, env, leaves[(agent, env)], n_levels,
                            owners))
        parts.append("</details>")
    # Env view: empty headers that applyGroupMode fills with the leaves.
    parts.append("</div><div id='view-env' style='display:none'>")
    for env in sorted({e for _, e in leaves}):
        keys = [k for k in leaves if k[1] == env]
        parts.append(
            _group_header("env", env, len(keys), sum(n_runs[k]
                                                     for k in keys)) +
            "</details>")
    parts.append("</div>")
    controls = (
        "<input id='runfilter' placeholder='filter runs…' "
        "oninput='filterRuns(this.value)'>"
        "<button id='groupbtn' onclick='toggleGroupMode()' title='Nest the "
        "run tables under agent names (one table per env) or under env "
        "names (one table per agent)'>group</button>"
        "<button onclick='setAllGroups(true)'>expand all</button>"
        "<button onclick='setAllGroups(false)'>collapse all</button>")
    return page("continual viewer",
                "runs",
                "".join(parts),
                refresh=30,
                controls=controls)


def _group_header(kind: str, name: str, n_inner: int, n_runs: int) -> str:
    """Opening tag and summary of an agent-name or env-name group."""
    inner = "envs" if kind == "agent" else "agents"
    return (f"<details class='grp family' data-key='{kind}:{esc(name)}' "
            f"data-kind='{kind}' data-name='{esc(name)}' open>"
            f"<summary>{esc(name)} <span class='muted'>"
            f"({n_inner} {inner}, {n_runs} runs)</span></summary>")


def _leaf_table(agent: str, env: str, cards: Sequence[Dict[str, Any]],
                n_levels: int, owners: Dict[str, List[Owner]]) -> str:
    """One (agent, env) pair's runs table, wrapped in its leaf group.

    The summary carries both names; the CSS shows the one naming the
    inner level of the current nesting (see applyGroupMode in the JS).
    """
    return (f"<details class='grp exp' data-key='exp:{esc(agent)}/{esc(env)}'"
            f" data-agent='{esc(agent)}' data-env='{esc(env)}' open>"
            f"<summary><span class='lbl env'>{esc(env)}</span>"
            f"<span class='lbl agent'>{esc(agent)}</span> "
            f"<span class='muted'>({len(cards)} runs)</span></summary>"
            f"<div class='tablewrap'>"
            f"{_runs_table(cards, n_levels, owners)}</div>"
            "</details>")


# Fixed column widths of the runs grid, in the order of _runs_table's
# header: run (experiment id/seed plus the buttons), state, levels (None:
# LEVEL_COL_W per level of the page's largest level count), steps,
# resets, invocations, active, queue, LLM cost, updated, git. Every leaf
# table uses them, so columns line up across agents and envs, as in the
# phased log viewer.
RUN_COL_W = (400, 150, None, 64, 56, 84, 72, 64, 64, 96, 80)
LEVEL_COL_W = 104


def _runs_table(cards: Sequence[Dict[str, Any]], n_levels: int,
                owners: Dict[str, List[Owner]]) -> str:
    widths = [w or LEVEL_COL_W * max(1, n_levels) for w in RUN_COL_W]
    cols = "<colgroup>" + "".join(f"<col style='width:{w}px'>"
                                  for w in widths) + "</colgroup>"
    rows = []
    for card in cards:
        totals = _totals(card)
        run_id = str(card["run_id"])
        run_owners = owners.get(run_id, [])
        label, cls = liveness(card, run_owners)
        levels = card.get("levels", [])
        search = " ".join(
            str(x)
            for x in (run_id, card.get("config") or "", card.get("arm"),
                      card.get("env"), f"seed{card.get('seed')}", label))
        rows.append(
            f"<tr class='runrow' data-text='{esc(search)}'>"
            f"<td>{_run_cell(card, run_owners)}</td>"
            f"<td>{chip(label, cls, _owners_title(run_owners))}</td>"
            f"<td>{_level_grid(levels, n_levels)}</td>"
            f"<td class='num'>{totals['total_steps']}</td>"
            f"<td class='num'>{totals['total_resets']}</td>"
            f"<td class='num'>{totals['total_skill_invocations']}</td>"
            f"<td class='num'>{fmt_duration(totals['total_wall_clock'])}</td>"
            f"<td class='num'>{fmt_duration(totals['total_downtime'])}</td>"
            f"<td class='num'>${float(totals['total_llm_cost']):.2f}</td>"
            f"<td class='muted'>{esc(fmt_age(card.get('updated_at')))}</td>"
            f"<td class='muted'><code>{esc(card.get('git_sha', ''))}</code>"
            "</td></tr>")
    head = ("<thead><tr><th>run</th>"
            "<th>state</th><th title='per level: won / in progress / not "
            "attempted, steps, resets'>levels</th>"
            "<th class='num'>steps</th><th class='num'>resets</th>"
            "<th class='num'>invoc.</th><th class='num'>active</th>"
            "<th class='num'>queue</th><th class='num'>LLM $</th>"
            "<th>updated</th><th>git</th></tr></thead>")
    return (f"<table class='grid runs' style='width:{sum(widths)}px'>"
            f"{cols}{head}<tbody>{''.join(rows)}</tbody></table>")


def _owners_title(owners: Sequence[Owner]) -> str:
    return ", ".join(f"Slurm job {o.ident} ({o.state})" if o.kind ==
                     "job" else f"pid {o.ident}" for o in owners)


def _run_cell(card: Dict[str, Any], owners: Sequence[Owner]) -> str:
    """The run's link, named by experiment id and seed, and its copy, pause
    (only while a job or process runs it) and delete buttons; the buttons show
    on hover, as in the phased log viewer."""
    run_id = str(card["run_id"])
    name = f"{card.get('config') or run_id}/seed{card.get('seed')}"
    esc_id = esc(run_id)
    live = bool(owners)
    pause = ""
    if live:
        pause = (f"<button class='rowbtn pause' title='Pause: cancel "
                 f"{esc(_owners_title(owners))}; the run keeps its "
                 "checkpoints and continues when relaunched with "
                 f"--auto_resume' onclick='pauseRun(\"{esc_id}\")'>"
                 "⏸</button>")
    del_title = ("Cancel the live job, then delete this run" if live else
                 "Delete this run: scorecard, recording and checkpoints")
    delete = (f"<button class='rowbtn del' title='{del_title}' "
              f"onclick='deleteRun(\"{esc_id}\", "
              f"{'true' if live else 'false'})'>✕</button>")
    return (f"<div class='runcell'><a href='/run/{q(run_id)}' "
            f"title='{esc_id}'>{esc(name)}</a><span class='btns'>"
            f"<button class='rowbtn copy' data-copy='{esc(copy_path(run_id))}'"
            f" title='Copy recording path'>⧉</button>{pause}{delete}"
            "</span></div>")


def copy_path(run_id: str) -> str:
    """What the copy button puts on the clipboard: the recording dir, relative
    to the viewer's working dir (the repo root when started from there), or the
    scorecard for a run without one."""
    rec = os.path.join(RECORDINGS_ROOT, run_id)
    if not os.path.isdir(rec):
        rec = os.path.join(SCORECARDS_ROOT, f"{run_id}.json")
    return os.path.relpath(rec)


def _level_grid(levels: Sequence[Dict[str, Any]], n_levels: int) -> str:
    """The per-level cells of one run in a fixed inner grid, so L1 sits under
    L1 in every row of the page."""
    cells = []
    for i in range(max(1, n_levels)):
        if i >= len(levels):
            cells.append("<td></td>")
            continue
        lv = levels[i]
        detail = (f"{lv.get('steps', 0)}s {lv.get('resets', 0)}r"
                  if lv.get("attempted") or lv.get("won") else "")
        cells.append(f"<td>{_level_mark(lv)}"
                     f"<span class='muted'>{detail}</span></td>")
    cols = "".join(f"<col style='width:{LEVEL_COL_W}px'>"
                   for _ in range(max(1, n_levels)))
    return (f"<table class='lvgrid'><colgroup>{cols}</colgroup>"
            f"<tr>{''.join(cells)}</tr></table>")


def _level_mark(lv: Dict[str, Any]) -> str:
    if lv.get("won"):
        return chip(
            "✓", "ok", f"L{lv['index'] + 1} won at step "
            f"{lv.get('won_at_step')}")
    if lv.get("attempted"):
        return chip("…", "warn", f"L{lv['index'] + 1} in progress")
    return chip("·", "", f"L{lv['index'] + 1} not attempted")


# ── Run page ───────────────────────────────────────────────────────


def run_page(run_id: str) -> Optional[str]:
    """One run: a sidebar (overview, each level's replay and events, the
    agent's sessions, the recording's files) and a content pane the page script
    fills from the hash route (see loadHash in the JS)."""
    card = load_card(run_id)
    if card is None:
        return None
    label, cls = liveness(card)
    levels = card.get("levels", [])
    nav = [
        f"<div class='runhead'>{esc(card.get('config') or run_id)} "
        f"{chip(label, cls)}</div>",
        "<div class='nav'><a href='#overview'>Overview</a>",
        "<h4>Levels</h4>",
    ]
    for lv in levels:
        k = int(lv["index"]) + 1
        mark = "✓" if lv.get("won") else ("…" if lv.get("attempted") else "·")
        nav.append(f"<div class='lvl'><a href='#L{k}' title='replay L{k}'>"
                   f"▶ L{k} <span class='muted'>{esc(lv.get('split'))}"
                   f"[{esc(lv.get('task_idx'))}]</span> {mark}</a>"
                   f"<a class='sub' href='#L{k}/events'>events</a></div>")
    if agent_dir(run_id) is not None:
        logs = list_session_logs(run_id)
        nav.append(f"<h4>Agent sessions ({len(logs)})</h4>")
        for log in logs:
            nav.append(
                f"<a href='#session/{esc(log['name'])}' "
                f"title='{esc(log['name'])}, {esc(fmt_ts(log['mtime']))}'>"
                f"{int(log['number']):03d} {esc(log['kind'])} "
                f"<span class='muted'>{log['size'] // 1024} KB, "
                f"{esc(fmt_age(log['mtime']))}</span></a>")
    nav.append("<h4>Files</h4>" + _file_tree(run_id) + "</div>")
    attempted = [int(lv["index"]) + 1 for lv in levels if lv.get("attempted")]
    default = f"L{attempted[-1]}" if attempted else "overview"
    body = (f"<div class='run'><div class='sidebar'>{''.join(nav)}</div>"
            f"<div id='content' data-run='{esc(run_id)}' "
            f"data-default='{default}'><p class='muted'>Loading…</p>"
            "</div></div>")
    crumb = f"<a href='/run/{q(run_id)}'>{esc(run_id)}</a>"
    refresh = 0 if card.get("end_reason") else 30
    return page(run_id, crumb, body, refresh=refresh, wrap=False)


TREE_SKIP = {".git"}
TREE_MAX_FILES = 30
TREE_MAX_DEPTH = 4


def _file_tree(run_id: str) -> str:
    """The recording directory as nested collapsible lists: files link to their
    view, transcripts to their session view, and a directory's "list" link to
    its listing."""
    root = safe_join(RECORDINGS_ROOT, run_id)
    if root is None or not os.path.isdir(root):
        return "<p class='muted'>No recordings.</p>"
    return "<div class='tree'>" + _tree_dir(root, "", 0) + "</div>"


def _tree_dir(path: str, rel: str, depth: int) -> str:
    try:
        names = sorted(n for n in os.listdir(path) if n not in TREE_SKIP)
    except OSError:
        return ""
    dirs = [n for n in names if os.path.isdir(os.path.join(path, n))]
    files = [n for n in names if n not in dirs]
    out = []
    list_link = ("<a class='sub' href='#f=%s' onclick='event.preventDefault();"
                 " location.hash = this.getAttribute(\"href\")'>list</a>")
    for name in dirs:
        sub = f"{rel}/{name}" if rel else name
        open_attr = " open" if sub in ("agent", "agent/sandbox") else ""
        inner = (_tree_dir(os.path.join(path, name), sub, depth +
                           1) if depth + 1 < TREE_MAX_DEPTH else "")
        out.append(f"<details{open_attr}><summary>{esc(name)}/ "
                   f"{list_link % esc(sub)}</summary>{inner}</details>")
    for name in files[:TREE_MAX_FILES]:
        sub = f"{rel}/{name}" if rel else name
        out.append(f"<a href='{_file_href(rel, name)}' title='{esc(sub)}'>"
                   f"{esc(name)}</a>")
    if len(files) > TREE_MAX_FILES:
        out.append(f"<a class='muted' href='#f={esc(rel)}'>… "
                   f"{len(files) - TREE_MAX_FILES} more</a>")
    return "".join(out)


def _file_href(rel: str, name: str) -> str:
    """The hash route viewing a file: transcripts get the session view."""
    if rel == "agent" and SESSION_LOG_RE.match(name):
        return f"#session/{esc(name)}"
    return f"#f={esc(f'{rel}/{name}' if rel else name)}"


# ── Pane fragments ─────────────────────────────────────────────────


def fragment(run_id: str, route: str) -> Optional[str]:
    """The content pane for one hash route of the run page: ``overview``,
    ``L<k>`` (the replay), ``L<k>/events``, ``session/<name>`` or ``f=<path>``
    (a file or directory of the recording)."""
    if load_card(run_id) is None:
        return None
    m = re.fullmatch(r"L(\d+)(/events)?", route)
    if m:
        k = int(m.group(1)) - 1
        if m.group(2):
            return events_fragment(run_id, k)
        return replay_fragment(run_id, k)
    if route == "overview":
        return overview_fragment(run_id)
    if route.startswith("session/"):
        return session_fragment(run_id, route[len("session/"):])
    if route.startswith("f="):
        return file_fragment(run_id, route[len("f="):])
    return None


def overview_fragment(run_id: str) -> Optional[str]:
    """Run metadata, the cost curve and the level table."""
    card = load_card(run_id)
    if card is None:
        return None
    totals = _totals(card)
    label, cls = liveness(card)
    meta = [
        ("env", esc(card.get("env"))),
        ("arm", esc(card.get("arm"))),
        ("seed", esc(card.get("seed"))),
        ("config", esc(card.get("config") or "")),
        ("state", chip(label, cls) + esc(card.get("end_note") or "")),
        ("levels", f"{totals['levels_completed']} / {totals['levels_total']}"
         " won"),
        ("steps", f"{totals['total_steps']} of cap {esc(card.get('step_cap'))}"
         f" ({totals['total_resets']} resets, "
         f"{totals['total_skill_invocations']} skill invocations)"),
        ("active", fmt_duration(totals["total_wall_clock"]) + " of cap " +
         fmt_duration(float(card.get("wall_clock_cap") or 0))),
        ("queue time", fmt_duration(totals["total_downtime"])),
        ("LLM cost", f"${float(totals['total_llm_cost']):.2f}"),
        ("started", esc(fmt_ts(card.get("started_at")))),
        ("updated", esc(fmt_ts(card.get("updated_at"))) + " (" +
         esc(fmt_age(card.get("updated_at"))) + ")"),
        ("finished", esc(fmt_ts(card.get("finished_at")))),
        ("git", f"<code>{esc(card.get('git_sha', ''))}</code>"),
        ("scorecard", f"<a href='/card/{q(run_id)}'>json</a>"),
    ]
    meta_html = "<dl class='meta'>" + "".join(f"<dt>{k}</dt><dd>{v}</dd>"
                                              for k, v in meta) + "</dl>"
    return (f"<h2>{esc(card.get('config') or run_id)}</h2>{meta_html}"
            "<h2>Cumulative steps vs levels won</h2>" + curve_svg(card) +
            "<h2>Levels</h2>" + _levels_table(card))


TEXT_EXTS = {
    "", ".md", ".txt", ".log", ".json", ".jsonl", ".py", ".yaml", ".yml",
    ".csv", ".sh", ".toml", ".cfg", ".html"
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
GALLERY_MAX = 400


def file_fragment(run_id: str, rel: str) -> Optional[str]:
    """A file of the recording (text, image, or a binary's size) or a directory
    listing with a thumbnail gallery of its images."""
    root = safe_join(RECORDINGS_ROOT, run_id)
    if root is None:
        return None
    rel = "" if rel in ("", ".") else rel
    path = root if not rel else safe_join(root, rel)
    if path is None or not os.path.exists(path):
        return None
    head = f"<h2>{esc(rel or run_id)}</h2>"
    if os.path.isdir(path):
        return head + _dir_listing(run_id, path, rel)
    raw = _raw_url(run_id, rel)
    info = (f"<p class='muted'>{os.path.getsize(path)} bytes, "
            f"{esc(fmt_ts(os.path.getmtime(path)))} · "
            f"<a href='{raw}'>raw</a></p>")
    ext = os.path.splitext(rel)[1].lower()
    if ext in IMAGE_EXTS:
        return (head + info +
                f"<img class='thumb big' src='{raw}' onclick='zoom(this)'>")
    if ext in TEXT_EXTS:
        text, note = _read_text(path, tail=ext in (".log", ".jsonl"))
        return (head + info + note +
                f"<pre class='doc'>{_image_links(run_id, esc(text))}</pre>")
    return head + info + "<p class='muted'>Binary file; not shown.</p>"


def _raw_url(run_id: str, rel: str) -> str:
    return "/file/" + "/".join(q(p) for p in [run_id, *rel.split("/")] if p)


def _read_text(path: str, tail: bool) -> Tuple[str, str]:
    """A text file's content cut to MAX_TEXT_CHARS from the head (the tail for
    logs), and a note when it was cut."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) <= MAX_TEXT_CHARS:
        return text, ""
    which = "last" if tail else "first"
    note = (f"<p class='muted'>Showing the {which} {MAX_TEXT_CHARS} of "
            f"{len(text)} characters.</p>")
    return (text[-MAX_TEXT_CHARS:] if tail else text[:MAX_TEXT_CHARS]), note


def _dir_listing(run_id: str, path: str, rel: str) -> str:
    try:
        names = sorted(n for n in os.listdir(path) if n not in TREE_SKIP)
    except OSError:
        return "<p class='muted'>Unreadable directory.</p>"
    rows, thumbs = [], []
    for name in names:
        sub = f"{rel}/{name}" if rel else name
        full = os.path.join(path, name)
        is_dir = os.path.isdir(full)
        size = "" if is_dir else str(os.path.getsize(full))
        rows.append(f"<tr><td><a href='{_file_href(rel, name)}'>{esc(name)}"
                    f"{'/' if is_dir else ''}</a></td>"
                    f"<td class='num'>{size}</td><td class='muted'>"
                    f"{esc(fmt_ts(os.path.getmtime(full)))}</td></tr>")
        if not is_dir and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            thumbs.append(f"<a href='#f={esc(sub)}'><img class='thumb' "
                          f"src='{_raw_url(run_id, sub)}' loading='lazy' "
                          f"title='{esc(name)}' onclick='zoom(this); "
                          "event.preventDefault()'></a>")
    table = ("<table><thead><tr><th>name</th><th class='num'>bytes</th>"
             "<th>modified</th></tr></thead><tbody>" + "".join(rows) +
             "</tbody></table>")
    if not thumbs:
        return table
    more = (f"<p class='muted'>… {len(thumbs) - GALLERY_MAX} more images</p>"
            if len(thumbs) > GALLERY_MAX else "")
    return (f"<h3>{len(thumbs)} images</h3><div class='gallery'>"
            f"{''.join(thumbs[:GALLERY_MAX])}</div>{more}" + table)


def win_points(card: Dict[str, Any]) -> List[Tuple[int, int]]:
    """(cumulative steps, levels won) at each win, plus the current end."""
    points: List[Tuple[int, int]] = [(0, 0)]
    cumulative = 0
    won = 0
    for lv in card.get("levels", []):
        if lv.get("won") and lv.get("won_at_step") is not None:
            won += 1
            points.append((cumulative + int(lv["won_at_step"]), won))
        cumulative += int(lv.get("steps", 0))
    if not points or points[-1][0] != cumulative:
        points.append((cumulative, won))
    return points


def curve_svg(card: Dict[str, Any]) -> str:
    """An inline SVG step curve of levels won against steps spent."""
    points = win_points(card)
    width, height, pad = 640, 200, 36
    max_x = max(max(p[0] for p in points), 1)
    max_y = max(len(card.get("levels", [])), 1)

    def sx(x: float) -> float:
        return pad + (width - 2 * pad) * x / max_x

    def sy(y: float) -> float:
        return height - pad - (height - 2 * pad) * y / max_y

    path = []
    prev_y = 0
    for i, (x, y) in enumerate(points):
        if i == 0:
            path.append(f"M{sx(x):.1f},{sy(y):.1f}")
        else:
            path.append(f"L{sx(x):.1f},{sy(prev_y):.1f}")
            path.append(f"L{sx(x):.1f},{sy(y):.1f}")
        prev_y = y
    dots = "".join(
        f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.5' fill='var(--ok)'>"
        f"<title>level {y} won after {x} cumulative steps</title></circle>"
        for x, y in points[1:] if y > 0)
    cap = card.get("step_cap")
    cap_line = ""
    if cap and int(cap) <= max_x * 1.5:
        cap_line = (f"<line x1='{sx(int(cap)):.1f}' y1='{pad}' "
                    f"x2='{sx(int(cap)):.1f}' y2='{height - pad}' "
                    "stroke='var(--bad)' stroke-dasharray='4 3'/>")
    return (f"<svg class='curve' width='{width}' height='{height}' "
            f"viewBox='0 0 {width} {height}'>"
            f"<line x1='{pad}' y1='{height - pad}' x2='{width - pad}' "
            f"y2='{height - pad}' stroke='var(--border)'/>"
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height - pad}' "
            "stroke='var(--border)'/>"
            f"<path d='{' '.join(path)}' fill='none' stroke='var(--accent)' "
            "stroke-width='2'/>" + dots + cap_line +
            f"<text x='{width - pad}' y='{height - 10}' text-anchor='end' "
            f"font-size='11' fill='var(--muted)'>steps ({max_x})</text>"
            f"<text x='{pad + 4}' y='{pad - 8}' font-size='11' "
            f"fill='var(--muted)'>levels won ({max_y} total)</text></svg>")


def _levels_table(card: Dict[str, Any]) -> str:
    rows = []
    for lv in card.get("levels", []):
        k = int(lv["index"])
        status = ("won" if lv.get("won") else
                  ("in progress" if lv.get("attempted") else "not attempted"))
        cls = "ok" if lv.get("won") else (
            "warn" if lv.get("attempted") else "")
        game_overs = lv.get("game_overs", [])
        go_text = ", ".join(sorted(set(str(g) for g in game_overs)))
        sandbox = lv.get("sandbox") or {}
        episodes = " ".join(
            chip(
                f"ep{ep['index']} {ep['steps']} {ep['end']}",
                "ok" if ep["end"] == "win" else (
                    "bad" if str(ep["end"]).startswith("game_over") else ""))
            for ep in lv.get("episodes", []))
        first = ("" if lv.get("steps_before_first_win") is None else
                 f"{lv['steps_before_first_win']} / "
                 f"{lv['resets_before_first_win']}")
        recovery = (f"{lv.get('preemptions', 0)}/{lv.get('resumes', 0)}/"
                    f"{lv.get('harness_resets', 0)}/"
                    f"{fmt_duration(float(lv.get('downtime', 0.0)))}")
        rows.append(
            "<tr>"
            f"<td><a href='#L{k + 1}/events'>L{k + 1}</a> "
            f"<a href='#L{k + 1}' title='replay'>▶</a> "
            f"<span class='muted'>{esc(lv.get('split'))}"
            f"[{esc(lv.get('task_idx'))}]</span></td>"
            f"<td>{chip(status, cls)}</td>"
            f"<td class='num'>{lv.get('steps', 0)}</td>"
            f"<td class='num'>{lv.get('resets', 0)}</td>"
            f"<td class='num'>{lv.get('skill_invocations', 0)} "
            f"<span class='muted'>({lv.get('failed_skill_invocations', 0)}"
            " failed)</span></td>"
            f"<td class='num'>{len(game_overs)} "
            f"<span class='muted'>{esc(go_text)}</span></td>"
            f"<td class='num'>{lv.get('divergences', 0)}</td>"
            f"<td class='num'>{first}</td>"
            f"<td class='num'>{fmt_duration(float(lv.get('wall_clock', 0)))}"
            f" <span class='muted'>env "
            f"{fmt_duration(float(lv.get('wall_clock_env', 0)))}</span></td>"
            f"<td class='num'>{esc(recovery)}</td>"
            f"<td class='num'>${float(sandbox.get('llm_cost_usd', 0)):.2f}"
            f" <span class='muted'>{int(sandbox.get('sessions', 0))} sess, "
            f"{int(sandbox.get('sim_rollouts', 0))} roll, "
            f"{int(sandbox.get('fits', 0))} fits</span></td>"
            f"<td>{episodes}</td>"
            f"<td class='atoms'>{esc(', '.join(lv.get('goal', [])))}"
            f"<br><span class='muted'>{esc(lv.get('goal_nl', ''))}</span>"
            "</td></tr>")
    return ("<table><thead><tr><th>level</th><th>status</th>"
            "<th class='num'>steps</th><th class='num'>resets</th>"
            "<th class='num'>invocations</th><th class='num'>game overs</th>"
            "<th class='num'>diverg.</th>"
            "<th class='num' title='steps / resets before the first win'>"
            "first win</th><th class='num'>active</th>"
            "<th class='num' title='preemptions/resumes/harness resets/"
            "downtime'>recovery</th><th class='num'>sandbox</th>"
            "<th>episodes</th><th>goal</th></tr></thead><tbody>" +
            "".join(rows) + "</tbody></table>")


def load_transcripts(run_id: str) -> List[tr.Transcript]:
    """Every parsed session transcript of a run, oldest first."""
    out = []
    for log in list_session_logs(run_id):
        text = read_agent_file(run_id, log["name"])
        if text is None:
            continue
        out.append(tr.parse_transcript(text, log["name"]))
    return out


def build_replay(run_id: str, level_index: int) -> List[Dict[str, Any]]:
    """The level's replay frames: index entries paired with the agent's tool
    calls and reasoning, each with a render URL."""
    entries = read_index(run_id, level_index)
    paired = tr.pair_entries(entries, load_transcripts(run_id))
    return tr.frames_from_entries(
        paired, lambda p: level_render_url(run_id, level_index, p))


def replay_fragment(run_id: str, level_index: int) -> Optional[str]:
    """The level replay: one frame per recorded event, with the render, the
    action, the agent's reasoning and the tool call, keyboard driven (the
    player is initReplay in the page script)."""
    card = load_card(run_id)
    if card is None or not 0 <= level_index < len(card.get("levels", [])):
        return None
    frames = build_replay(run_id, level_index)
    lv = card["levels"][level_index]
    k = level_index + 1
    if not frames:
        return (f"<h2>L{k} replay</h2>"
                "<p class='muted'>No recorded events yet.</p>")
    cells = []
    for f in frames:
        cls = f"cell {esc(f['event'])}"
        inner = (f"<img src='{f['render']}' loading='lazy'>"
                 if f["render"] else esc(f["event"][:6]))
        cells.append(f"<div class='{cls}' title='{esc(f['event'])} "
                     f"{esc(f['skill'])}' onclick='REPLAY.go({f['i']})'>"
                     f"{inner}</div>")
    # JSON inside a script element: '<' must not open a tag.
    data = json.dumps(frames, default=str).replace("<", "\\u003c")
    won_chip = chip("won" if lv.get("won") else "not won",
                    "ok" if lv.get("won") else "warn")
    json_url = f"/run/{q(run_id)}/L{k}/replay.json"
    return (
        f"<div class='replay-root'><h2>L{k} replay {won_chip}"
        f" <span class='muted'>{len(frames)} frames · "
        f"<a href='#L{k}/events'>events</a> · "
        f"<a href='{json_url}'>Download JSON</a></span></h2>"
        "<div class='replay'><div class='stage'>"
        "<div class='stagehead'><span class='big' id='counter'></span>"
        "<span id='evt'></span><span id='state'></span>"
        "<span class='muted' id='steps'></span></div>"
        "<img id='frame' class='frame' alt='render'>"
        "<div id='noframe' class='noframe'>no render for this event</div>"
        "<div class='controls'>"
        "<button onclick='REPLAY.go(0)' title='Home'>⏮</button>"
        "<button onclick='REPLAY.step(-1)' title='←'>◀</button>"
        "<button id='play' onclick='REPLAY.toggle()' title='Space'>▶ play"
        "</button>"
        "<button onclick='REPLAY.step(1)' title='→'>▶</button>"
        f"<button onclick='REPLAY.go({len(frames) - 1})' title='End'>⏭"
        "</button>"
        "<select id='speed'><option value='1'>1×</option>"
        "<option value='2'>2×</option><option value='4'>4×</option></select>"
        f"<input type='range' id='scrub' min='0' max='{len(frames) - 1}' "
        "value='0'></div>"
        f"<div class='filmstrip'>{''.join(cells)}</div>"
        "<p class='muted keys'>← → frame · Shift+← → ±10 · [ ] marker · "
        "Home End · Space play/pause · G jump</p></div>"
        "<div class='panels'>"
        "<div class='panel'><h3>Action</h3><div id='action'></div></div>"
        "<div class='panel'><h3>Agent reasoning</h3><div id='reasoning'>"
        "</div></div>"
        "<div class='panel'><h3>Tool call</h3><div id='call'></div></div>"
        "<div class='panel'><h3>Atoms (change since the previous frame)</h3>"
        "<div id='atoms'></div></div></div></div>"
        f"<script type='application/json' id='replay-data'>{data}</script>"
        "</div>")


def replay_json(run_id: str, level_index: int) -> Optional[bytes]:
    """The replay frames as JSON."""
    card = load_card(run_id)
    if card is None or not 0 <= level_index < len(card.get("levels", [])):
        return None
    return json.dumps(build_replay(run_id, level_index), default=str,
                      indent=1).encode("utf-8")


def _image_links(run_id: str, escaped: str) -> str:
    """Make ``./test_images/x.png`` references in escaped text viewable."""

    def _img(m: "re.Match[str]") -> str:
        rel = m.group(0)[2:]
        url = "/file/" + "/".join(
            q(p) for p in (run_id, "agent", "sandbox", *rel.split("/")))
        return (f"{m.group(0)} <a href='{url}'><img class='thumb' "
                f"src='{url}' loading='lazy' onclick='zoom(this);"
                "event.preventDefault()'></a>")

    return IMAGE_REF_RE.sub(_img, escaped)


def session_fragment(run_id: str, name: str) -> Optional[str]:
    """One session transcript as a structured conversation: thinking, assistant
    text, tool calls with their results, images inline."""
    if SESSION_LOG_RE.match(name) is None:
        return None
    text = read_agent_file(run_id, name)
    if text is None:
        return None
    tx = tr.parse_transcript(text, name)
    if not tx.turns:
        if len(text) > MAX_TEXT_CHARS:
            text = ("[... earlier content truncated ...]\n" +
                    text[-MAX_TEXT_CHARS:])
        # Not a formatter transcript (or an empty one): show it raw.
        return (f"<h2>{esc(name)}</h2><pre class='doc'>"
                f"{_image_links(run_id, esc(text))}</pre>")
    names = [log["name"] for log in list_session_logs(run_id)]
    nav = []
    if name in names:
        i = names.index(name)
        if i > 0:
            nav.append(f"<a href='#session/{esc(names[i - 1])}'>"
                       f"← {esc(names[i - 1])}</a>")
        if i + 1 < len(names):
            nav.append(f"<a href='#session/{esc(names[i + 1])}'>"
                       f"{esc(names[i + 1])} →</a>")
    parts = [
        f"<h2>{esc(name)} {chip(tx.kind)}</h2>",
        f"<p>{' · '.join(nav)}"
        f"{' · ' if nav else ''}<span class='muted'>{len(tx.turns)} turns"
        f"{'; ' + esc(tx.result_line) if tx.result_line else ''}</span></p>",
        "<details><summary>Prompt</summary><pre class='doc'>" +
        _image_links(run_id, esc(tx.prompt)) + "</pre></details>",
    ]
    for turn in tx.turns:
        parts.append(f"<section class='turn' id='turn-{turn.number}'>"
                     f"<h3>Turn {turn.number}</h3>")
        for th in turn.thinking:
            parts.append(f"<blockquote class='think'>{esc(th)}</blockquote>")
        for say in turn.texts:
            parts.append(f"<div class='say'>{_image_links(run_id, esc(say))}"
                         "</div>")
        for call in turn.calls:
            short = call.short_name
            cls = "call envtool" if short in tr.ENV_TOOLS or short in (
                "env_observe", "learn_run", "session_end", "env_end_run",
                "run_python", "skills_list") else "call"
            parts.append(f"<div class='{cls}'><span class='name'>"
                         f"{esc(short)}</span>")
            for key, val in call.args.items():
                if isinstance(val, str) and "\n" in val:
                    parts.append(f"<div class='muted'>{esc(key)}:</div>"
                                 f"<pre>{esc(val)}</pre>")
            scalars = {
                k: v
                for k, v in call.args.items()
                if not (isinstance(v, str) and "\n" in v)
            }
            if scalars:
                parts.append("<pre>" +
                             esc(json.dumps(scalars, indent=2, default=str)) +
                             "</pre>")
            if call.result:
                open_attr = " open" if len(call.result) < 1200 else ""
                err_cls = " err" if call.is_error else ""
                label = "error" if call.is_error else "result"
                parts.append(
                    f"<details class='result{err_cls}'{open_attr}>"
                    f"<summary>{label} ({len(call.result)} chars)</summary>"
                    f"<pre>{_image_links(run_id, esc(call.result))}</pre>"
                    "</details>")
            parts.append("</div>")
        parts.append("</section>")
    for err in tx.errors:
        parts.append(f"<p class='del'>{esc(err)}</p>")
    return "".join(parts)


# ── Level page ─────────────────────────────────────────────────────

EVENT_KINDS = ("level_start", "invoke", "reset", "resume", "win", "game_over")


def events_fragment(run_id: str, level_index: int) -> Optional[str]:
    """One level's timeline."""
    card = load_card(run_id)
    if card is None:
        return None
    levels = card.get("levels", [])
    if not 0 <= level_index < len(levels):
        return None
    lv = levels[level_index]
    entries = read_index(run_id, level_index)
    status = ("won" if lv.get("won") else
              ("in progress" if lv.get("attempted") else "not attempted"))
    head = (f"<h2>L{level_index + 1} {esc(lv.get('split'))}"
            f"[{esc(lv.get('task_idx'))}] {chip(status)}</h2>"
            f"<p class='atoms'>goal: {esc(', '.join(lv.get('goal', [])))}"
            f"<br><span class='muted'>{esc(lv.get('goal_nl', ''))}</span></p>"
            f"<p>steps {lv.get('steps', 0)}, resets {lv.get('resets', 0)}, "
            f"invocations {lv.get('skill_invocations', 0)} "
            f"({lv.get('failed_skill_invocations', 0)} failed), "
            f"game overs {len(lv.get('game_overs', []))}, "
            f"divergences {lv.get('divergences', 0)}, "
            f"active {fmt_duration(float(lv.get('wall_clock', 0)))}, "
            f"resumes {lv.get('resumes', 0)}, harness resets "
            f"{lv.get('harness_resets', 0)}.</p>")
    filters = "<div class='filters'>" + "".join(
        f"<label><input type='checkbox' value='{kind}' checked "
        f"onchange='applyFilters()'> {kind}</label>"
        for kind in EVENT_KINDS) + "</div>"
    nav = _level_nav(level_index, len(levels))
    return (head + nav + filters +
            _timeline_table(run_id, level_index, entries))


def _level_nav(k: int, n: int) -> str:
    links = [f"<a href='#L{k + 1}'>▶ replay</a>"]
    if k > 0:
        links.append(f"<a href='#L{k}/events'>← L{k}</a>")
    if k + 1 < n:
        links.append(f"<a href='#L{k + 2}/events'>L{k + 2} →</a>")
    return "<p>" + " · ".join(links) + "</p>"


def atoms_diff(prev: Sequence[str], cur: Sequence[str]) -> str:
    """Added (green) and removed (red) atoms between two events."""
    prev_set, cur_set = set(prev), set(cur)
    added = sorted(cur_set - prev_set)
    removed = sorted(prev_set - cur_set)
    parts = [f"<span class='add'>+{esc(a)}</span>" for a in added]
    parts += [f"<span class='del'>{esc(a)}</span>" for a in removed]
    if not parts:
        return "<span class='muted'>no change</span>"
    return " ".join(parts)


def _timeline_table(run_id: str, level_index: int,
                    entries: Sequence[Dict[str, Any]]) -> str:
    if not entries:
        return "<p class='muted'>No index entries recorded yet.</p>"
    rows = []
    prev_atoms: List[str] = []
    for i, e in enumerate(entries):
        event = str(e.get("event", ""))
        skill = e.get("skill", "")
        params = e.get("params")
        if params:
            skill += "[" + ", ".join(f"{float(p):.3g}" for p in params) + "]"
        status = str(e.get("status", ""))
        status_cls = {
            "succeeded": "ok",
            "failed": "bad",
            "interrupted": "warn"
        }.get(status, "")
        state = str(e.get("state", ""))
        state_cls = {"WIN": "ok", "GAME_OVER": "bad"}.get(state, "")
        detail = ""
        if event == "game_over":
            detail = esc(e.get("reason", ""))
        elif event == "resume":
            detail = (f"replayed {e.get('replayed_steps')} steps, "
                      f"verified={e.get('verified')}")
        elif event == "reset":
            detail = f"by {esc(e.get('by', ''))}"
        elif event == "level_start":
            detail = esc(", ".join(e.get("goal", [])))
        expected = e.get("expected") or []
        missing = e.get("missing") or []
        expectation = ""
        if expected:
            expectation = "expected: " + esc(", ".join(expected))
            if missing:
                expectation += ("<br><span class='del'>missing: " +
                                esc(", ".join(missing)) + "</span>")
        atoms = e.get("atoms")
        diff = ""
        if isinstance(atoms, list):
            diff = atoms_diff(prev_atoms, atoms)
            prev_atoms = [str(a) for a in atoms]
        url = level_render_url(run_id, level_index, e.get("render"))
        img = (f"<img class='thumb' src='{url}' onclick='zoom(this)' "
               f"loading='lazy'>" if url else "")
        rows.append(
            f"<tr class='event-{esc(event)}' data-event='{esc(event)}'>"
            f"<td class='num muted'>{i}</td>"
            f"<td class='muted'>{esc(fmt_ts(e.get('t')))}</td>"
            f"<td>{chip(event)}</td>"
            f"<td class='num'>{esc(e.get('episode', ''))}</td>"
            f"<td><code>{esc(skill)}</code> {detail}</td>"
            f"<td>{chip(status, status_cls) if status else ''}"
            f"<span class='muted'>{esc(e.get('reason', ''))}</span></td>"
            f"<td class='num'>{esc(e.get('steps', ''))}</td>"
            f"<td class='num'>{esc(e.get('level_steps', ''))}</td>"
            f"<td>{chip(state, state_cls) if state else ''}</td>"
            f"<td class='atoms'>{expectation}</td>"
            f"<td class='atoms'>{diff}</td>"
            f"<td class='note'>{esc(e.get('note', ''))}</td>"
            f"<td>{img}</td></tr>")
    return ("<table><thead><tr><th class='num'>#</th><th>time</th>"
            "<th>event</th><th class='num'>ep</th><th>skill / detail</th>"
            "<th>status</th><th class='num'>steps</th>"
            "<th class='num'>level steps</th><th>state</th>"
            "<th>expected outcome</th><th>atoms changed</th><th>note</th>"
            "<th>render</th></tr></thead><tbody>" + "".join(rows) +
            "</tbody></table>")


# ── Server ─────────────────────────────────────────────────────────


def _is_level(part: str) -> bool:
    return part.startswith("L") and part[1:].isdigit()


class Handler(BaseHTTPRequestHandler):
    """Routes: /, /run/<id> (the run page), /run/<id>/frag/<route> (its
    pane content), /run/<id>/L<k>/replay.json, /card/<id>, /file/<rel>;
    POST /pause?r=<id> and POST /delete?r=<id>[&kill=1]. The former
    /run/<id>/L<k>[/replay] and /run/<id>/session/<name> pages redirect
    to the run page's hash routes."""

    # pylint: disable-next=redefined-builtin
    def log_message(self, format: str, *args: Any) -> None:
        del format, args  # quiet

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Serve one page, fragment or file."""
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        try:
            if not parts:
                self._html(index_page())
            elif parts[0] == "run" and len(parts) == 2:
                self._html_or_404(run_page(parts[1]))
            elif parts[0] == "run" and len(parts) >= 4 and \
                    parts[2] == "frag":
                self._html_or_404(fragment(parts[1], "/".join(parts[3:])))
            elif parts[0] == "run" and len(parts) == 3 and \
                    _is_level(parts[2]):
                self._redirect(f"/run/{q(parts[1])}#{parts[2]}/events")
            elif parts[0] == "run" and len(parts) == 4 and \
                    parts[2] == "session":
                self._redirect(f"/run/{q(parts[1])}#session/{parts[3]}")
            elif parts[0] == "run" and len(parts) == 4 and \
                    _is_level(parts[2]) and parts[3] == "replay":
                self._redirect(f"/run/{q(parts[1])}#{parts[2]}")
            elif parts[0] == "run" and len(parts) == 4 and \
                    _is_level(parts[2]) and parts[3] == "replay.json":
                self._bytes(replay_json(parts[1],
                                        int(parts[2][1:]) - 1),
                            "application/json")
            elif parts[0] == "card" and len(parts) == 2:
                self._file(safe_join(SCORECARDS_ROOT, parts[1] + ".json"))
            elif parts[0] == "file" and len(parts) >= 2:
                self._file(safe_join(RECORDINGS_ROOT,
                                     os.path.join(*parts[1:])))
            else:
                self.send_error(404)
        except Exception as e:  # pylint: disable=broad-except
            self.send_error(500, str(e))

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Pause or delete a run, with a plain-text reply.

        POST only, so the index page's auto-refresh GETs can never trip
        these, and same-origin only: a page anywhere can fire a cross-
        origin POST at localhost.
        """
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host', '')}":
            self._text("cross-origin POST rejected", 403)
            return
        parsed = urllib.parse.urlparse(self.path)
        params = {
            k: v[0]
            for k, v in urllib.parse.parse_qs(parsed.query).items()
        }
        try:
            if parsed.path == "/pause":
                ok, msg = pause_run(params.get("r", ""))
            elif parsed.path == "/delete":
                ok, msg = delete_run(params.get("r", ""),
                                     kill=params.get("kill") == "1")
            else:
                self._text("not found", 404)
                return
            self._text(msg, 200 if ok else 409)
        except Exception as e:  # pylint: disable=broad-except
            self._text(str(e), 500)

    def _text(self, msg: str, status: int) -> None:
        data = msg.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _bytes(self, data: Optional[bytes], ctype: str) -> None:
        if data is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _html_or_404(self, body: Optional[str]) -> None:
        if body is None:
            self.send_error(404)
        else:
            self._html(body)

    def _html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Optional[str]) -> None:
        if path is None or not os.path.isfile(path):
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def configure(scorecards: str,
              recordings: str,
              approaches: str = "saved_approaches") -> None:
    """Point the module at the roots (also used by tests)."""
    global SCORECARDS_ROOT, RECORDINGS_ROOT, APPROACHES_ROOT  # pylint: disable=global-statement
    SCORECARDS_ROOT = os.path.realpath(scorecards)
    RECORDINGS_ROOT = os.path.realpath(recordings)
    APPROACHES_ROOT = os.path.realpath(approaches)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--scorecards", default="scorecards")
    parser.add_argument("--recordings", default="recordings")
    parser.add_argument("--approaches",
                        default="saved_approaches",
                        help="approach checkpoints (CFG.approach_dir); "
                        "a run's delete removes its files here too")
    parser.add_argument("--port", type=int, default=25152)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    configure(args.scorecards, args.recordings, args.approaches)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"continual viewer: http://{args.host}:{args.port}/  "
          f"(scorecards={SCORECARDS_ROOT}, recordings={RECORDINGS_ROOT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
