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

* ``/``: every run, grouped by env then arm: levels won, steps, resets,
  invocations, active and queue time, LLM cost, liveness.
* ``/run/<run_id>``: the run's metadata, the cumulative steps-versus-
  levels-won curve, and one row per level with the section 4.4 metrics
  and its episodes.
* ``/run/<run_id>/L<k>``: the level's timeline: one row per index event
  with the skill, its parameters, status, steps, episode state, the
  agent's expected outcome and what was missing, the note, the atoms
  that changed since the previous event, and the render.
* ``/run/<run_id>/L<k>/replay``: the level as a replay, after the
  ARC-AGI-3 replay viewer: one frame per recorded event with the render,
  the action, the agent's thinking and text that led to it, the tool
  call and its result, and the atoms that changed; keyboard playback
  (frame, ±10, marker, home/end, play/pause, jump) and a JSON export.
* ``/run/<run_id>/session/<name>``: one session transcript as a
  structured conversation: thinking, assistant text, tool calls with
  their results, renders inline.

Usage:
    python scripts/continual_viewer.py [--scorecards scorecards] \\
        [--recordings recordings] [--port 25152] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import mimetypes
import os
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Run as a script or imported as scripts.continual_viewer: either way the
# helper module lives next to this file, under the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# pylint: disable-next=wrong-import-position
from scripts import continual_transcripts as tr  # noqa: E402

SCORECARDS_ROOT = ""  # absolute, set in main()
RECORDINGS_ROOT = ""  # absolute, set in main()
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


def read_agent_text(run_id: str, rel: str) -> Optional[str]:
    """A text file under the agent dir (journal, attempts, a session log),
    tail-truncated to ``MAX_TEXT_CHARS`` for display."""
    text = read_agent_file(run_id, rel)
    if text is not None and len(text) > MAX_TEXT_CHARS:
        text = "[... earlier content truncated ...]\n" + text[-MAX_TEXT_CHARS:]
    return text


def liveness(card: Dict[str, Any]) -> Tuple[str, str]:
    """(label, css class) for a card's state."""
    if card.get("end_reason"):
        reason = str(card["end_reason"])
        cls = "ok" if reason == "all_levels_won" else "bad"
        return reason, cls
    age = time.time() - float(card.get("updated_at") or 0)
    if age < LIVE_WINDOW_S:
        return "live", "live"
    return f"stalled ({fmt_age(card.get('updated_at'))})", "warn"


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
"""

JS = """
function zoom(img) { img.classList.toggle('big'); }
function applyFilters() {
  var boxes = document.querySelectorAll('.filters input[type=checkbox]');
  var on = {};
  boxes.forEach(function (b) { on[b.value] = b.checked; });
  document.querySelectorAll('tr[data-event]').forEach(function (tr) {
    tr.style.display = on[tr.dataset.event] === false ? 'none' : '';
  });
}
"""


def page(title: str, crumb: str, body: str, refresh: int = 0) -> str:
    """The full HTML page."""
    meta = (f"<meta http-equiv='refresh' content='{refresh}'>"
            if refresh else "")
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{esc(title)}</title>{meta}<style>{CSS}</style>"
            f"<script>{JS}</script></head><body>"
            "<div class='topbar'><h1><a href='/'>continual viewer</a></h1>"
            f"<span class='crumb'>{crumb}</span></div>"
            f"<div class='content'>{body}</div></body></html>")


def chip(label: Any, cls: str = "", title: str = "") -> str:
    """Small labelled span."""
    return (f"<span class='chip {cls}' title='{esc(title)}'>"
            f"{esc(label)}</span>")


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
    """The runs overview."""
    cards = list_cards()
    if not cards:
        body = (f"<p class='muted'>No scorecards under "
                f"<code>{esc(SCORECARDS_ROOT)}</code> yet.</p>")
        return page("continual viewer", "runs", body, refresh=30)
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for card in cards:
        groups.setdefault(str(card.get("env")),
                          {}).setdefault(str(card.get("arm")), []).append(card)
    parts = [
        f"<p class='muted'>{len(cards)} run(s) under "
        f"<code>{esc(SCORECARDS_ROOT)}</code>; recordings under "
        f"<code>{esc(RECORDINGS_ROOT)}</code>. Auto-refreshes every 30 s."
        "</p>"
    ]
    for env in sorted(groups):
        parts.append(f"<h2>{esc(env)}</h2>")
        for arm in sorted(groups[env]):
            parts.append(f"<h3>{esc(arm)}</h3>")
            parts.append(_runs_table(groups[env][arm]))
    return page("continual viewer", "runs", "".join(parts), refresh=30)


def _runs_table(cards: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for card in cards:
        totals = _totals(card)
        label, cls = liveness(card)
        run_id = str(card["run_id"])
        levels = card.get("levels", [])
        marks = "".join(_level_mark(lv) for lv in levels)
        rows.append(
            "<tr>"
            f"<td><a href='/run/{q(run_id)}'>"
            f"{esc(card.get('config') or run_id)}</a>"
            f"<br><span class='muted'>{esc(run_id)}</span></td>"
            f"<td class='num'>{esc(card.get('seed'))}</td>"
            f"<td>{chip(label, cls)}</td>"
            f"<td class='num'>{totals['levels_completed']}/"
            f"{totals['levels_total']} {marks}</td>"
            f"<td class='num'>{totals['total_steps']}</td>"
            f"<td class='num'>{totals['total_resets']}</td>"
            f"<td class='num'>{totals['total_skill_invocations']}</td>"
            f"<td class='num'>{fmt_duration(totals['total_wall_clock'])}</td>"
            f"<td class='num'>{fmt_duration(totals['total_downtime'])}</td>"
            f"<td class='num'>${float(totals['total_llm_cost']):.2f}</td>"
            f"<td class='muted'>{esc(fmt_age(card.get('updated_at')))}</td>"
            f"<td class='muted'><code>{esc(card.get('git_sha', ''))}</code>"
            "</td></tr>")
    return ("<table><thead><tr><th>run</th><th class='num'>seed</th>"
            "<th>state</th><th class='num'>levels</th>"
            "<th class='num'>steps</th><th class='num'>resets</th>"
            "<th class='num'>invocations</th><th class='num'>active</th>"
            "<th class='num'>queue</th><th class='num'>LLM $</th>"
            "<th>updated</th><th>git</th></tr></thead><tbody>" +
            "".join(rows) + "</tbody></table>")


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
    """One run: metadata, the cost curve, the level table."""
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
    body = (f"<h2>{esc(card.get('config') or run_id)}</h2>{meta_html}"
            "<h2>Cumulative steps vs levels won</h2>" + curve_svg(card) +
            "<h2>Levels</h2>" + _levels_table(card) + _agent_section(run_id))
    crumb = f"<a href='/run/{q(run_id)}'>{esc(run_id)}</a>"
    refresh = 0 if card.get("end_reason") else 30
    return page(run_id, crumb, body, refresh=refresh)


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
    run_id = str(card["run_id"])
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
            f"<td><a href='/run/{q(run_id)}/L{k + 1}'>L{k + 1}</a> "
            f"<a href='/run/{q(run_id)}/L{k + 1}/replay' title='replay'>▶</a> "
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


def _agent_section(run_id: str) -> str:
    """Session transcripts, journal and attempts of the agent arm."""
    if agent_dir(run_id) is None:
        return ""
    logs = list_session_logs(run_id)
    rows = []
    for log in logs:
        rows.append("<tr>"
                    f"<td class='num'>{log['number']}</td>"
                    f"<td>{chip(log['kind'])}</td>"
                    f"<td><a href='/run/{q(run_id)}/session/{q(log['name'])}'>"
                    f"{esc(log['name'])}</a></td>"
                    f"<td class='num'>{log['size'] // 1024} KB</td>"
                    f"<td class='muted'>{esc(fmt_ts(log['mtime']))}</td></tr>")
    if rows:
        table = ("<table><thead><tr><th class='num'>#</th><th>kind</th>"
                 "<th>transcript</th><th class='num'>size</th>"
                 "<th>written</th></tr></thead><tbody>" + "".join(rows) +
                 "</tbody></table>")
    else:
        table = "<p class='muted'>No session transcripts yet.</p>"
    parts = [f"<h2>Agent sessions ({len(logs)})</h2>", table]
    for title, rel in (("Journal (journal.md, the agent's)",
                        "sandbox/journal.md"),
                       ("Attempts record (attempts.md, the harness's)",
                        "sandbox/attempts.md")):
        text = read_agent_text(run_id, rel)
        if text is None:
            continue
        parts.append(f"<details><summary>{esc(title)} "
                     f"({len(text)} chars)</summary>"
                     f"<pre class='doc'>{esc(text)}</pre></details>")
    return "".join(parts)


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
    """The level's replay frames: index entries paired with the agent's
    tool calls and reasoning, each with a render URL."""
    entries = read_index(run_id, level_index)
    paired = tr.pair_entries(entries, load_transcripts(run_id))
    return tr.frames_from_entries(
        paired, lambda p: level_render_url(run_id, level_index, p))


REPLAY_JS = r"""
(function () {
  var F = window.FRAMES, N = F.length, cur = 0, timer = null, speed = 1;
  var runId = window.RUN_ID, levelNum = window.LEVEL_NUM;
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
    document.getElementById('counter').textContent = (cur + 1) + ' / ' + N;
    document.getElementById('evt').innerHTML = chip(f.event,
      f.event === 'win' ? 'ok' : f.event === 'game_over' ? 'bad' : '');
    document.getElementById('state').innerHTML = f.state ?
      chip(f.state, stateCls(f.state)) : '';
    document.getElementById('steps').textContent = 'episode ' +
      (f.episode == null ? '?' : f.episode) + ' · level steps ' +
      (f.level_steps == null ? '?' : f.level_steps) + ' · run steps ' +
      (f.run_steps == null ? '?' : f.run_steps);
    var img = document.getElementById('frame'), nof =
      document.getElementById('noframe');
    if (f.render) { img.src = f.render; img.style.display = ''; nof.style.display
      = 'none'; } else { img.style.display = 'none'; nof.style.display = ''; }
    document.getElementById('scrub').value = cur;
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
      ', turn ' + f.turn + ' · <a href="/run/' + encodeURIComponent(runId) +
      '/session/' + encodeURIComponent(f.session_name) + '#turn-' + f.turn +
      '">transcript</a></div>');
    document.getElementById('action').innerHTML = a.join('') ||
      '<span class="muted">harness event</span>';
    // reasoning panel
    var r = [];
    (f.thinking || []).forEach(function (t) { r.push(
      '<blockquote class="think">' + esc(t) + '</blockquote>'); });
    (f.texts || []).forEach(function (t) { r.push('<div class="say">' +
      esc(t) + '</div>'); });
    document.getElementById('reasoning').innerHTML = r.join('') ||
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
    document.getElementById('call').innerHTML = c ||
      '<span class="muted">none</span>';
    // atoms panel
    var prev = null;
    for (var j = cur - 1; j >= 0; j--) { if (F[j].atoms) { prev = F[j].atoms;
      break; } }
    var at = '<div class="atoms">' + atomsDiff(prev, f.atoms) + '</div>';
    if (f.env_atoms) at += '<details><summary>env atoms (' +
      f.env_atoms.length + ')</summary><div class="atoms muted">' +
      esc(f.env_atoms.join(', ')) + '</div></details>';
    document.getElementById('atoms').innerHTML = at;
    // filmstrip
    var cells = document.querySelectorAll('.filmstrip .cell');
    cells.forEach(function (el, i) { el.classList.toggle('cur', i === cur); });
    if (cells[cur]) cells[cur].scrollIntoView({inline: 'center', block:
      'nearest'});
    location.hash = 'frame=' + cur;
  }
  function go(i) { cur = Math.max(0, Math.min(N - 1, i)); render(); }
  function step(d) { go(cur + d); }
  function marker(d) {
    var i = cur + d;
    while (i >= 0 && i < N && !F[i].marker) i += d;
    if (i >= 0 && i < N) go(i);
  }
  function toggle() {
    if (timer) { clearInterval(timer); timer = null;
      document.getElementById('play').textContent = '▶ play'; return; }
    document.getElementById('play').textContent = '⏸ pause';
    timer = setInterval(function () { if (cur >= N - 1) { toggle(); return; }
      step(1); }, 900 / speed);
  }
  window.go = go; window.step = step; window.toggle = toggle;
  window.marker = marker;
  document.getElementById('speed').addEventListener('change', function (e) {
    speed = Number(e.target.value); if (timer) { toggle(); toggle(); } });
  document.getElementById('scrub').addEventListener('input', function (e) {
    go(Number(e.target.value)); });
  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowRight') { step(e.shiftKey ? 10 : 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { step(e.shiftKey ? -10 : -1);
      e.preventDefault(); }
    else if (e.key === ']') marker(1);
    else if (e.key === '[') marker(-1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(N - 1);
    else if (e.key === ' ') { toggle(); e.preventDefault(); }
    else if (e.key === 'g' || e.key === 'G') { var v = prompt('Jump to frame ' +
      '(1-' + N + ')'); if (v) go(Number(v) - 1); }
  });
  var m = /frame=(\d+)/.exec(location.hash);
  go(m ? Number(m[1]) : 0);
})();
"""


def replay_page(run_id: str, level_index: int) -> Optional[str]:
    """The level replay: one frame per recorded event, with the render, the
    action, the agent's reasoning and the tool call, keyboard driven."""
    card = load_card(run_id)
    if card is None or not 0 <= level_index < len(card.get("levels", [])):
        return None
    frames = build_replay(run_id, level_index)
    lv = card["levels"][level_index]
    crumb = (
        f"<a href='/run/{q(run_id)}'>{esc(run_id)}</a> / "
        f"<a href='/run/{q(run_id)}/L{level_index + 1}'>L{level_index + 1}"
        "</a> / replay")
    if not frames:
        return page(f"{run_id} L{level_index + 1} replay", crumb,
                    "<p class='muted'>No recorded events yet.</p>")
    cells = []
    for f in frames:
        cls = f"cell {esc(f['event'])}"
        inner = (f"<img src='{f['render']}' loading='lazy'>"
                 if f["render"] else esc(f["event"][:6]))
        cells.append(f"<div class='{cls}' title='{esc(f['event'])} "
                     f"{esc(f['skill'])}' onclick='go({f['i']})'>{inner}"
                     "</div>")
    data = json.dumps(frames, default=str)
    won_chip = chip("won" if lv.get("won") else "not won",
                    "ok" if lv.get("won") else "warn")
    json_url = f"/run/{q(run_id)}/L{level_index + 1}/replay.json"
    body = (
        f"<h2>L{level_index + 1} replay {won_chip}"
        f" <span class='muted'>{len(frames)} frames · "
        f"<a href='{json_url}'>Download JSON</a></span></h2>"
        "<div class='replay'><div class='stage'>"
        "<div class='stagehead'><span class='big' id='counter'></span>"
        "<span id='evt'></span><span id='state'></span>"
        "<span class='muted' id='steps'></span></div>"
        "<img id='frame' class='frame' alt='render'>"
        "<div id='noframe' class='noframe'>no render for this event</div>"
        "<div class='controls'>"
        "<button onclick='go(0)' title='Home'>⏮</button>"
        "<button onclick='step(-1)' title='←'>◀</button>"
        "<button id='play' onclick='toggle()' title='Space'>▶ play</button>"
        "<button onclick='step(1)' title='→'>▶</button>"
        f"<button onclick='go({len(frames) - 1})' title='End'>⏭</button>"
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
        f"<script>window.FRAMES = {data}; window.RUN_ID = {json.dumps(run_id)};"
        f" window.LEVEL_NUM = {level_index + 1};</script>"
        f"<script>{REPLAY_JS}</script>")
    return page(f"{run_id} L{level_index + 1} replay", crumb, body)


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


def session_page(run_id: str, name: str) -> Optional[str]:
    """One session transcript as a structured conversation: thinking,
    assistant text, tool calls with their results, images inline."""
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
        return page(
            f"{run_id} {name}",
            f"<a href='/run/{q(run_id)}'>{esc(run_id)}</a> / {esc(name)}",
            f"<h2>{esc(name)}</h2><pre class='doc'>"
            f"{_image_links(run_id, esc(text))}</pre>")
    names = [log["name"] for log in list_session_logs(run_id)]
    nav = []
    if name in names:
        i = names.index(name)
        if i > 0:
            nav.append(f"<a href='/run/{q(run_id)}/session/{q(names[i - 1])}'>"
                       f"← {esc(names[i - 1])}</a>")
        if i + 1 < len(names):
            nav.append(f"<a href='/run/{q(run_id)}/session/{q(names[i + 1])}'>"
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
    crumb = (f"<a href='/run/{q(run_id)}'>{esc(run_id)}</a> / "
             f"{esc(name)}")
    return page(f"{run_id} {name}", crumb, "".join(parts))


# ── Level page ─────────────────────────────────────────────────────

EVENT_KINDS = ("level_start", "invoke", "reset", "resume", "win", "game_over")


def level_page(run_id: str, level_index: int) -> Optional[str]:
    """One level's timeline."""
    card = load_card(run_id)
    if card is None:
        return None
    levels = card.get("levels", [])
    if not 0 <= level_index < len(levels):
        return None
    lv = levels[level_index]
    entries = read_index(run_id, level_index)
    crumb = (f"<a href='/run/{q(run_id)}'>{esc(run_id)}</a> / "
             f"L{level_index + 1}")
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
    nav = _level_nav(run_id, level_index, len(levels))
    refresh = 0 if (lv.get("won") or card.get("end_reason")) else 30
    return page(f"{run_id} L{level_index + 1}",
                crumb,
                head + nav + filters +
                _timeline_table(run_id, level_index, entries),
                refresh=refresh)


def _level_nav(run_id: str, k: int, n: int) -> str:
    links = [f"<a href='/run/{q(run_id)}/L{k + 1}/replay'>▶ replay</a>"]
    if k > 0:
        links.append(f"<a href='/run/{q(run_id)}/L{k}'>← L{k}</a>")
    if k + 1 < n:
        links.append(f"<a href='/run/{q(run_id)}/L{k + 2}'>L{k + 2} →</a>")
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


class Handler(BaseHTTPRequestHandler):
    """Routes: /, /run/<id>, /run/<id>/L<k>, /run/<id>/L<k>/replay[.json],
    /run/<id>/session/<name>, /card/<id>, /file/<rel>."""

    # pylint: disable-next=redefined-builtin
    def log_message(self, format: str, *args: Any) -> None:
        del format, args  # quiet

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Serve one page or file."""
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        try:
            if not parts:
                self._html(index_page())
            elif parts[0] == "run" and len(parts) == 2:
                self._html_or_404(run_page(parts[1]))
            elif parts[0] == "run" and len(parts) == 3 and \
                    parts[2].startswith("L") and parts[2][1:].isdigit():
                self._html_or_404(level_page(parts[1], int(parts[2][1:]) - 1))
            elif parts[0] == "run" and len(parts) == 4 and \
                    parts[2] == "session":
                self._html_or_404(session_page(parts[1], parts[3]))
            elif parts[0] == "run" and len(parts) == 4 and \
                    parts[2].startswith("L") and parts[2][1:].isdigit() and \
                    parts[3] in ("replay", "replay.json"):
                level = int(parts[2][1:]) - 1
                if parts[3] == "replay":
                    self._html_or_404(replay_page(parts[1], level))
                else:
                    self._bytes(replay_json(parts[1], level),
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


def configure(scorecards: str, recordings: str) -> None:
    """Point the module at the roots (also used by tests)."""
    global SCORECARDS_ROOT, RECORDINGS_ROOT  # pylint: disable=global-statement
    SCORECARDS_ROOT = os.path.realpath(scorecards)
    RECORDINGS_ROOT = os.path.realpath(recordings)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--scorecards", default="scorecards")
    parser.add_argument("--recordings", default="recordings")
    parser.add_argument("--port", type=int, default=25152)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    configure(args.scorecards, args.recordings)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"continual viewer: http://{args.host}:{args.port}/  "
          f"(scorecards={SCORECARDS_ROOT}, recordings={RECORDINGS_ROOT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
