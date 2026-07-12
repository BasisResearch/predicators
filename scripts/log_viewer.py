#!/usr/bin/env python3
"""Local web viewer for agent experiment logs under logs/.

Stdlib-only TensorBoard-style browser for run directories of the form
logs/<family>/<env>-<approach>/seed<N>/run_<timestamp>/. Features:

  * runs overview with per-episode pass/fail chips, costs, and run comparison
  * episode markdown transcripts with collapsible turns and inline images
  * unified diffs between simulator_versions / predicates_versions files
  * image galleries, ANSI-colored info.log / debug.log, and a full file tree
    so every logged file is inspectable

Format contracts this viewer relies on:
  * episode filenames {query:03d}_{kind}[_task{K}]_{YYYYMMDD_HHMMSS}.md from
    session_log_filename() in predicators/agent_sdk/tools.py
  * markdown layout (## sections, ### Turn N, trailing "**Result:** ..." line)
    from format_conversation_markdown() in predicators/agent_sdk/log_formatter.py
  * "Goal achieved: True|False" lines inside tool-result blocks (tools.py)
  * "Test results: defaultdict(..., {...})" lines in info.log

Usage:
    python scripts/log_viewer.py [--logs logs] [--port 8765]
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import mimetypes
import os
import re
import sys
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

LOGS_ROOT = ""  # absolute, set in main()

EPISODE_RE = re.compile(r"^(\d{3})_([a-z]+)(?:_task(\d+))?_(\d{8}_\d{6})\.md$")
RESULT_RE = re.compile(
    r"\*\*Result:\*\* (\d+) turns, \$([\d.]+) this solve, \$([\d.]+) total")
GOAL_RE = re.compile(r"Goal achieved: (True|False)")
NUM_SOLVED_RE = re.compile(r"'num_solved': ([\d.]+)")
NUM_TOTAL_RE = re.compile(r"'num_total': ([\d.]+)")
# Env-side verdict lines from predicators/main.py _run_testing, e.g.
# "INFO: Task 2 / 5: Policy failed to reach goal" or
# "INFO: [main.py] Task 1 / 5: approach failed with error: ..."
TASK_VERDICT_RE = re.compile(
    r"^INFO: (?:\[main\.py\] )?Task (\d+) / (\d+): (.+)$")
TASKS_SOLVED_RE = re.compile(r"Tasks solved: (\d+) / (\d+)")
ANSI_RE = re.compile(r"\x1b\[([\d;]*)m")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi"}
TEXT_EXTS = {
    ".txt", ".log", ".json", ".yaml", ".yml", ".csv", ".pddl", ".cfg", ".ini",
    ".sh", ".tex"
}
CODE_EXTS = {".py"}
# Bare image paths inside code blocks / inline code, e.g. "Saved images:".
PATH_IN_TEXT_RE = re.compile(r"(?:/|\./)?[\w][\w./\-]*\.(?:png|jpe?g|gif)")

# abs path -> ((mtime, size), parsed value)
_parse_cache: Dict[str, Tuple[Tuple[float, int], Any]] = {}


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;").replace('"', "&quot;"))


def q(text: str) -> str:
    return urllib.parse.quote(text, safe="")


def safe_join(rel: str) -> Optional[str]:
    """Resolve rel against LOGS_ROOT; return abs path or None if escaping."""
    if not rel or rel.startswith(("/", "\\")) or ".." in rel.split("/"):
        return None
    path = os.path.realpath(os.path.join(LOGS_ROOT, rel))
    root = os.path.realpath(LOGS_ROOT)
    if path != root and not path.startswith(root + os.sep):
        return None
    return path


def _cached(path: str, fn: Callable[[str], Any]) -> Any:
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    hit = _parse_cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    value = fn(path)
    _parse_cache[path] = (key, value)
    return value


def read_text(path: str, max_bytes: Optional[int] = None) -> Tuple[str, bool]:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if max_bytes and size > max_bytes:
                f.seek(size - max_bytes)
                data = f.read()
                return data.decode("utf-8", "replace"), True
            return f.read().decode("utf-8", "replace"), False
    except OSError as e:
        return "(error reading file: %s)" % e, False


# ---------------------------------------------------------------- scanning

_RUN_NAME_TS_RE = re.compile(r"^run_(\d{8})_(\d{6})$")


def _run_start_ts(name: str, mtime: float) -> float:
    """Start time from the run dir name; dir mtime as a fallback.

    The dir mtime moves whenever a file inside is added or touched, so
    only the name timestamp gives a stable newest-first ordering.
    """
    m = _RUN_NAME_TS_RE.match(name)
    if not m:
        return mtime
    try:
        return datetime.datetime.strptime(
            m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return mtime


def find_runs() -> List[Dict[str, Any]]:
    """Return run dicts sorted by experiment, then newest first."""
    runs: List[Dict[str, Any]] = []
    for dirpath, dirnames, _ in os.walk(LOGS_ROOT):
        rel = os.path.relpath(dirpath, LOGS_ROOT)
        if rel.count(os.sep) > 4:
            dirnames[:] = []
            continue
        for d in list(dirnames):
            if d.startswith("run_"):
                run_rel = os.path.normpath(os.path.join(rel, d)).replace(
                    os.sep, "/")
                parts = run_rel.split("/")
                seed = next((p for p in parts if p.startswith("seed")), "")
                exp = "/".join(p for p in parts[:-1]
                               if not p.startswith("seed")) or "(root)"
                runs.append({
                    "rel":
                    run_rel,
                    "exp":
                    exp,
                    "seed":
                    seed,
                    "name":
                    d,
                    "mtime":
                    os.path.getmtime(os.path.join(dirpath, d)),
                })
                dirnames.remove(d)  # do not descend into run dirs
    runs.sort(key=lambda r:
              (r["exp"], r["seed"], -_run_start_ts(r["name"], r["mtime"])))
    return runs


def _parse_episode(path: str) -> Dict[str, Any]:
    text, _ = read_text(path)
    info: Dict[str, Any] = {}
    goals = GOAL_RE.findall(text)
    info["goal"] = (goals[-1] == "True") if goals else None
    m: Optional[re.Match[str]] = None
    for m in RESULT_RE.finditer(text):
        pass
    if m:
        info["turns"] = int(m.group(1))
        info["solve_cost"] = float(m.group(2))
        info["total_cost"] = float(m.group(3))
    return info


def list_episodes(run_abs: str) -> List[Dict[str, Any]]:
    """Episode metadata for md files at the run dir root."""
    episodes: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(run_abs))
    except OSError:
        return episodes
    for name in names:
        m = EPISODE_RE.match(name)
        if not m:
            continue
        ep: Dict[str, Any] = {
            "file": name,
            "num": int(m.group(1)),
            "kind": m.group(2),
            "task": int(m.group(3)) if m.group(3) else None,
            "ts": m.group(4),
        }
        ep.update(_cached(os.path.join(run_abs, name), _parse_episode) or {})
        episodes.append(ep)
    return episodes


def _parse_info_log(path: str) -> Dict[str, Any]:
    """Extract authoritative test outcomes from main.py's info.log.

    Returns {"totals": [(num_solved, num_total), ...] one per cycle,
    "rounds": [{task_idx0: {"solved": bool, "msg": str}, ...}, ...]} where
    each round is one test phase, closed by a "Tasks solved: X / Y" line.
    """
    text, _ = read_text(path, max_bytes=8 * 1024 * 1024)
    text = ANSI_RE.sub("", text)
    totals: List[Tuple[int, int]] = []
    rounds: List[Dict[int, Dict[str, Any]]] = []
    current: Dict[int, Dict[str, Any]] = {}
    for line in text.splitlines():
        if "Test results:" in line:
            ms, mt = NUM_SOLVED_RE.search(line), NUM_TOTAL_RE.search(line)
            if ms and mt:
                totals.append(
                    (int(float(ms.group(1))), int(float(mt.group(1)))))
            continue
        m = TASK_VERDICT_RE.match(line)
        if m:
            msg = m.group(3).strip()
            # A later verdict for the same task within a round (e.g. the
            # impossible-goal SOLVED after an approach failure) overwrites.
            current[int(m.group(1)) - 1] = {
                "solved": msg.startswith("SOLVED"),
                "msg": msg,
            }
            continue
        if TASKS_SOLVED_RE.search(line):
            rounds.append(current)
            current = {}
    if current:  # run still in progress or crashed mid-round
        rounds.append(current)
    return {"totals": totals, "rounds": rounds}


def run_summary(run_rel: str) -> Optional[Dict[str, Any]]:
    run_abs = safe_join(run_rel)
    if not run_abs or not os.path.isdir(run_abs):
        return None
    episodes = list_episodes(run_abs)
    parsed = _cached(os.path.join(run_abs, "info.log"), _parse_info_log) or {}
    rounds = parsed.get("rounds", [])
    # A test round r happens after r learn episodes, so each test episode
    # maps to the round given by the number of learn episodes before it.
    learn_seen = 0
    for ep in episodes:
        ep["round"] = learn_seen
        if ep["kind"] == "learn":
            learn_seen += 1
        if (ep["kind"] == "test" and ep["task"] is not None
                and ep["round"] < len(rounds)):
            verdict = rounds[ep["round"]].get(ep["task"])
            if verdict is not None:
                ep["env_solved"] = verdict["solved"]
                ep["env_msg"] = verdict["msg"]
    total_cost = max((e.get("total_cost", 0.0) for e in episodes), default=0.0)
    return {
        "episodes": episodes,
        "test_results": parsed.get("totals", []),
        "rounds": rounds,
        "total_cost": total_cost
    }


def run_stamp(run_rel: str) -> str:
    """Cheap change fingerprint of a run dir (file count, mtime, bytes)."""
    run_abs = safe_join(run_rel)
    if not run_abs or not os.path.isdir(run_abs):
        return "gone"
    count, max_mtime, total = 0, 0.0, 0
    for dirpath, _, files in os.walk(run_abs):
        for f in files:
            try:
                st = os.stat(os.path.join(dirpath, f))
            except OSError:
                continue
            count += 1
            total += st.st_size
            max_mtime = max(max_mtime, st.st_mtime)
    return "%d-%d-%d" % (count, int(max_mtime), total)


def index_stamp() -> str:
    """Fingerprint of the runs overview: run set + run-root activity."""
    parts = []
    for r in find_runs():
        run_abs = safe_join(r["rel"])
        if not run_abs:
            continue
        try:
            st = os.stat(run_abs)
            info = os.path.join(run_abs, "info.log")
            info_size = os.path.getsize(info) if os.path.exists(info) else 0
        except OSError:
            continue
        parts.append("%s:%d:%d" % (r["rel"], int(st.st_mtime), info_size))
    return "%d:%d" % (len(parts), hash(tuple(parts)) & 0xffffffff)


# ------------------------------------------------------------ asset lookup


def resolve_asset(ref: str, run_rel: str) -> Optional[str]:
    """Map an image/file reference inside a transcript to a /raw URL."""
    ref = ref.strip().strip("`'\"")
    run_abs = safe_join(run_rel)
    if not run_abs:
        return None
    candidates = []
    if ref.startswith("/"):
        candidates.append(ref)
        # Absolute path from another machine: re-root at the run dir name.
        marker = "/" + os.path.basename(run_abs) + "/"
        if marker in ref:
            candidates.append(os.path.join(run_abs, ref.split(marker, 1)[1]))
    else:
        rel = ref[2:] if ref.startswith("./") else ref
        candidates += [
            os.path.join(run_abs, rel),
            os.path.join(run_abs, "sandbox", rel),
            os.path.join(run_abs, "test_images", os.path.basename(rel)),
            os.path.join(run_abs, "sandbox", "test_images",
                         os.path.basename(rel)),
        ]
    root = os.path.realpath(LOGS_ROOT)
    for cand in candidates:
        real = os.path.realpath(cand)
        if real.startswith(root + os.sep) and os.path.isfile(real):
            return "/raw?p=" + q(
                os.path.relpath(real, root).replace(os.sep, "/"))
    return None


def thumbs_for_paths(text: str, run_rel: str, cap: int = 24) -> str:
    seen, out = set(), []
    for m in PATH_IN_TEXT_RE.finditer(text):
        url = resolve_asset(m.group(0), run_rel)
        if url and url not in seen:
            seen.add(url)
            out.append('<a class="thumb" href="%s" target="_blank">'
                       '<img loading="lazy" src="%s"></a>' % (url, url))
        if len(out) >= cap:
            break
    if not out:
        return ""
    return '<div class="thumbs">%s</div>' % "".join(out)


# --------------------------------------------------------------- markdown

INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
LINK_MD_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def render_inline(line: str, run_rel: str) -> str:
    """Inline markdown on an already-raw line; escapes HTML itself."""
    tokens: List[str] = []

    def stash(html_frag: str) -> str:
        tokens.append(html_frag)
        return "\x00%d\x00" % (len(tokens) - 1)

    def sub_code(m: re.Match[str]) -> str:
        content = m.group(1)
        frag = "<code>%s</code>" % esc(content)
        url = resolve_asset(content, run_rel)
        if url and os.path.splitext(content)[1].lower() in IMG_EXTS:
            frag += (' <a class="thumb inline" href="%s" target="_blank">'
                     '<img loading="lazy" src="%s"></a>' % (url, url))
        return stash(frag)

    def sub_img(m: re.Match[str]) -> str:
        alt, src = m.group(1), m.group(2)
        url = resolve_asset(src, run_rel) or esc(src)
        return stash('<img class="mdimg" loading="lazy" alt="%s" src="%s">' %
                     (esc(alt), url))

    def sub_link(m: re.Match[str]) -> str:
        label, href = m.group(1), m.group(2)
        url = resolve_asset(href, run_rel) if not href.startswith(
            ("http://", "https://")) else esc(href)
        return stash('<a href="%s" target="_blank">%s</a>' %
                     (url or esc(href), esc(label)))

    line = INLINE_CODE_RE.sub(sub_code, line)
    line = IMG_MD_RE.sub(sub_img, line)
    line = LINK_MD_RE.sub(sub_link, line)
    line = esc(line)
    line = BOLD_RE.sub(r"<strong>\1</strong>", line)
    for i, frag in enumerate(tokens):
        line = line.replace("\x00%d\x00" % i, frag)
    return line


def render_md_chunk(lines: List[str], run_rel: str) -> str:
    """Render markdown lines (no section splitting) to HTML."""
    out = []
    i, n = 0, len(lines)
    para: List[str] = []
    list_tag: Optional[str] = None

    def flush_para() -> None:
        if para:
            out.append("<p>%s</p>" %
                       "<br>".join(render_inline(l, run_rel) for l in para))
            del para[:]

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append("</%s>" % list_tag)
            list_tag = None

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_para()
            close_list()
            lang = stripped[3:].strip()
            block = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = "\n".join(block)
            out.append('<pre class="code" data-lang="%s"><code>%s</code>'
                       "</pre>" % (esc(lang), esc(code)))
            out.append(thumbs_for_paths(code, run_rel))
            continue
        if not stripped:
            flush_para()
            close_list()
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_para()
            close_list()
            level = min(len(m.group(1)) + 1, 6)  # demote: page owns h1
            out.append("<h%d>%s</h%d>" %
                       (level, render_inline(m.group(2), run_rel), level))
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,})$", stripped):
            flush_para()
            close_list()
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            flush_para()
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append("<li>%s</li>" % render_inline(m.group(1), run_rel))
            i += 1
            continue
        m = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m:
            flush_para()
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append("<li>%s</li>" % render_inline(m.group(1), run_rel))
            i += 1
            continue
        if stripped.startswith(">"):
            flush_para()
            close_list()
            out.append("<blockquote>%s</blockquote>" %
                       render_inline(stripped.lstrip("> "), run_rel))
            i += 1
            continue
        # Indented pseudo-code blocks (state dumps) keep their spacing.
        if line.startswith(("    ", "\t")) and not para:
            close_list()
            block = []
            while i < n and (lines[i].startswith(
                ("    ", "\t")) or not lines[i].strip()):
                if not lines[i].strip() and (i + 1 >= n
                                             or not lines[i + 1].startswith(
                                                 ("    ", "\t"))):
                    break
                block.append(lines[i])
                i += 1
            code = "\n".join(block)
            out.append('<pre class="code"><code>%s</code></pre>' % esc(code))
            out.append(thumbs_for_paths(code, run_rel))
            continue
        para.append(line)
        i += 1
    flush_para()
    close_list()
    return "\n".join(out)


def split_fence_aware(
    lines: List[str], pattern: re.Pattern[str]
) -> Tuple[List[str], List[Tuple[re.Match[str], List[str]]]]:
    """Split lines at non-fenced lines matching pattern.

    Returns (head_lines, [(match, body_lines), ...]).
    """
    head: List[str] = []
    sections: List[Tuple[re.Match[str], List[str]]] = []
    current: Optional[Tuple[re.Match[str], List[str]]] = None
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else pattern.match(line)
        if m:
            current = (m, [])
            sections.append(current)
        elif current is not None:
            current[1].append(line)
        else:
            head.append(line)
    return head, sections


H2_RE = re.compile(r"^##\s+(.*)$")
TURN_RE = re.compile(r"^###\s+(Turn\s+\d+.*)$")


def turn_hint(body_lines: List[str], run_rel: str) -> str:
    for line in body_lines:
        s = line.strip()
        if s.startswith("```"):
            continue
        s = re.sub(r"[*`#\[\]]", "", s).strip()
        if not s or re.fullmatch(r"[-=_.|:]+", s):
            continue
        if len(s) > 90:
            s = s[:90] + "…"
        return esc(s)
    return ""


def render_md_document(text: str, run_rel: str) -> str:
    """Full markdown doc: collapsible ## sections, nested Turn details."""
    lines = text.splitlines()
    head, sections = split_fence_aware(lines, H2_RE)
    out = [render_md_chunk(head, run_rel)]
    for m, body in sections:
        title = m.group(1)
        sub_head, turns = split_fence_aware(body, TURN_RE)
        inner = [render_md_chunk(sub_head, run_rel)]
        for j, (tm, tbody) in enumerate(turns):
            is_last = j == len(turns) - 1
            inner.append(
                '<details class="turn"%s><summary>%s '
                '<span class="hint">%s</span></summary>%s</details>' %
                (" open" if is_last else "", esc(tm.group(1)),
                 turn_hint(tbody, run_rel), render_md_chunk(tbody, run_rel)))
        default_open = (turns or title.lower().startswith(
            ("goal", "conversation", "prompt")))
        out.append(
            '<details class="section"%s><summary>%s</summary>%s</details>' %
            (" open" if default_open else "", esc(title), "\n".join(inner)))
    return "\n".join(out)


# ------------------------------------------------------------- ANSI logs

ANSI_COLORS = {
    "30": "#6e7681",
    "31": "#e5534b",
    "32": "#57ab5a",
    "33": "#c69026",
    "34": "#539bf5",
    "35": "#b083f0",
    "36": "#39c5cf",
    "37": "#adbac7",
    "90": "#6e7681",
    "91": "#ff938a",
    "92": "#6bc46d",
    "93": "#daaa3f",
    "94": "#6cb6ff",
    "95": "#dcbdfb",
    "96": "#56d4dd",
    "97": "#cdd9e5",
}


def ansi_to_html(text: str) -> str:
    out, pos, open_span = [], 0, False
    for m in ANSI_RE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        pos = m.end()
        codes = (m.group(1) or "0").split(";")
        if open_span:
            out.append("</span>")
            open_span = False
        styles = []
        for c in codes:
            if c in ANSI_COLORS:
                styles.append("color:%s" % ANSI_COLORS[c])
            elif c == "1":
                styles.append("font-weight:bold")
        if styles:
            out.append('<span style="%s">' % ";".join(styles))
            open_span = True
    out.append(esc(text[pos:]))
    if open_span:
        out.append("</span>")
    return "".join(out)


# ------------------------------------------------------------ HTML shell

CSS = """
:root {
  --bg: #ffffff; --fg: #24292f; --muted: #57606a; --border: #d0d7de;
  --panel: #f6f8fa; --accent: #0969da; --ok: #1a7f37; --bad: #cf222e;
  --code-bg: #f6f8fa; --add: #dafbe1; --del: #ffebe9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e; --border: #30363d;
    --panel: #161b22; --accent: #58a6ff; --ok: #3fb950; --bad: #f85149;
    --code-bg: #161b22; --add: #12261e; --del: #35181a;
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
.topbar input { flex: 1; max-width: 420px; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 6px;
  background: var(--bg); color: var(--fg); }
.topbar .crumb { color: var(--muted); font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
button { padding: 4px 10px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--bg); color: var(--fg);
  cursor: pointer; font-size: 12px; }
button:hover { border-color: var(--accent); }
.layout { display: flex; height: calc(100vh - 45px); }
.sidebar { width: 340px; min-width: 240px; overflow-y: auto; padding: 10px;
  border-right: 1px solid var(--border); background: var(--panel);
  resize: horizontal; }
.content { flex: 1; overflow-y: auto; padding: 16px 24px; }
.content { max-width: 100%; }
.content pre.code { background: var(--code-bg); padding: 10px 12px;
  border: 1px solid var(--border); border-radius: 6px; overflow-x: auto;
  white-space: pre; }
.content h2, .content h3 { border-bottom: 1px solid var(--border);
  padding-bottom: 4px; }
details.section, details.turn { border: 1px solid var(--border);
  border-radius: 6px; margin: 8px 0; background: var(--bg); }
details.section > summary, details.turn > summary { cursor: pointer;
  padding: 6px 12px; font-weight: 600; background: var(--panel);
  border-radius: 6px; }
details[open].section > summary, details[open].turn > summary {
  border-bottom: 1px solid var(--border);
  border-radius: 6px 6px 0 0; }
details.section > *:not(summary), details.turn > *:not(summary) {
  margin-left: 12px; margin-right: 12px; }
details.turn { margin-left: 8px; }
summary .hint { color: var(--muted); font-weight: 400; font-size: 12px; }
.chip { display: inline-block; padding: 0 7px; border-radius: 10px;
  font-size: 11px; line-height: 18px; border: 1px solid var(--border);
  color: var(--muted); margin-right: 3px; white-space: nowrap; }
.chip.ok { color: var(--ok); border-color: var(--ok); }
.chip.bad { color: var(--bad); border-color: var(--bad); }
.chip.kind-explore { color: #b083f0; border-color: #b083f0; }
.chip.kind-learn { color: #daaa3f; border-color: #daaa3f; }
.banner { padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;
  border: 1px solid var(--border); background: var(--panel);
  display: flex; gap: 18px; flex-wrap: wrap; }
.banner.warn { border-color: var(--bad); color: var(--bad);
  font-weight: 600; }
.ok { color: var(--ok); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }
table.grid { border-collapse: collapse; margin: 10px 0; }
table.grid th, table.grid td { border: 1px solid var(--border);
  padding: 4px 10px; text-align: left; font-size: 13px; }
table.grid th { background: var(--panel); }
.sidebar .nav a { display: block; padding: 3px 6px; border-radius: 4px;
  color: var(--fg); font-size: 13px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.sidebar .nav a:hover { background: var(--bg); text-decoration: none; }
.sidebar .nav a.active { background: var(--accent); color: #fff; }
.sidebar .nav a.active .chip, .sidebar .nav a.active .muted {
  color: #fff; border-color: #fff; }
.sidebar h3 { font-size: 12px; text-transform: uppercase;
  letter-spacing: .5px; color: var(--muted); margin: 14px 0 4px; }
.sidebar details { margin-left: 8px; }
.sidebar details summary { cursor: pointer; font-size: 13px;
  color: var(--muted); }
.thumbs { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0; }
.thumb img { height: 110px; border: 1px solid var(--border);
  border-radius: 4px; }
.thumb.inline img { height: 60px; vertical-align: middle; }
img.mdimg { max-width: 480px; max-height: 400px;
  border: 1px solid var(--border); border-radius: 6px; display: block;
  margin: 6px 0; }
.gallery { display: grid; gap: 8px;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
.gallery a { text-align: center; font-size: 11px; color: var(--muted);
  overflow-wrap: anywhere; }
.gallery img { width: 100%; border: 1px solid var(--border);
  border-radius: 4px; }
pre.logview { white-space: pre-wrap; overflow-wrap: anywhere; }
.diff .add { background: var(--add); display: block; }
.diff .del { background: var(--del); display: block; }
.diff .hunk { color: var(--accent); display: block; }
#lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.85);
  display: none; align-items: center; justify-content: center;
  z-index: 100; cursor: zoom-out; }
#lightbox img { max-width: 96vw; max-height: 96vh; }
.runrow.hidden { display: none; }
#refreshpill { position: fixed; right: 18px; bottom: 18px; z-index: 50;
  background: var(--accent); color: #fff; border: none;
  padding: 8px 14px; border-radius: 18px; font-weight: 600;
  box-shadow: 0 2px 8px rgba(0,0,0,.35); }
details.grp { border: 1px solid var(--border); border-radius: 6px;
  margin: 8px 0; }
details.grp > summary { cursor: pointer; padding: 6px 12px;
  font-weight: 600; background: var(--panel); border-radius: 6px; }
details.grp[open] > summary { border-bottom: 1px solid var(--border);
  border-radius: 6px 6px 0 0; }
details.grp.family > summary { font-size: 15px; }
details.grp > *:not(summary) { margin: 8px 12px; }
details.grp.hidden { display: none; }
.muted { color: var(--muted); }
"""

JS = """
function $(s, r) { return (r || document).querySelector(s); }
function $all(s, r) { return Array.from((r || document).querySelectorAll(s)); }

// Lightbox for images.
document.addEventListener('click', function(e) {
  var a = e.target.closest('a.thumb, a.shot');
  var img = e.target.closest('img.mdimg, .gallery img');
  var src = null;
  if (a) { src = a.href; }
  else if (img) { src = img.src; }
  if (src && /raw\\?p=/.test(src)) {
    e.preventDefault();
    var lb = $('#lightbox');
    lb.firstElementChild.src = src;
    lb.style.display = 'flex';
  }
});
document.addEventListener('DOMContentLoaded', function() {
  var lb = document.createElement('div');
  lb.id = 'lightbox';
  lb.innerHTML = '<img>';
  lb.onclick = function() { lb.style.display = 'none'; };
  document.body.appendChild(lb);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') lb.style.display = 'none';
  });
});

function setAllDetails(open) {
  $all('.content details').forEach(function(d) { d.open = open; });
}

// Index page: filter + compare + collapsible groups.
function groupKey(d) { return 'lv-grp:' + d.dataset.key; }
function restoreGroups() {
  $all('details.grp').forEach(function(d) {
    var v = localStorage.getItem(groupKey(d));
    if (v !== null) d.open = v === '1';
  });
}
document.addEventListener('DOMContentLoaded', restoreGroups);
document.addEventListener('toggle', function(e) {
  var d = e.target;
  if (d.classList && d.classList.contains('grp') && !window._filtering)
    localStorage.setItem(groupKey(d), d.open ? '1' : '0');
}, true);
function setAllGroups(open) {
  window._filtering = true;
  $all('details.grp').forEach(function(d) {
    d.open = open;
    localStorage.setItem(groupKey(d), open ? '1' : '0');
  });
  window._filtering = false;
}
function filterRuns(text) {
  text = text.toLowerCase();
  sessionStorage.setItem('lv-filter', text);
  window._filtering = true;
  $all('.runrow').forEach(function(row) {
    row.classList.toggle('hidden',
      !!text && row.dataset.key.indexOf(text) === -1);
  });
  $all('details.grp.exp').forEach(function(d) {
    var any = $all('.runrow', d).some(function(r) {
      return !r.classList.contains('hidden');
    });
    d.classList.toggle('hidden', !any);
    if (text) d.open = any;
  });
  $all('details.grp.family').forEach(function(d) {
    var any = $all('details.grp.exp', d).some(function(x) {
      return !x.classList.contains('hidden');
    });
    d.classList.toggle('hidden', !any);
    if (text) d.open = any;
  });
  if (!text) restoreGroups();
  window._filtering = false;
}
function compareSelected() {
  var sel = $all('input.cmp:checked').map(function(c) { return c.value; });
  if (sel.length < 2) { alert('Select at least two runs to compare.'); return; }
  location.href = '/compare?runs=' + encodeURIComponent(sel.join(';'));
}

// Run page: hash routing into the content pane.
function loadHash() {
  var content = $('#content');
  if (!content) return;
  var h = decodeURIComponent(location.hash.slice(1));
  var run = content.dataset.run;
  var url = null;
  if (h.indexOf('f=') === 0) {
    url = '/view?d=' + encodeURIComponent(run) +
          '&f=' + encodeURIComponent(h.slice(2));
  } else if (h.indexOf('gallery=') === 0) {
    url = '/gallery?d=' + encodeURIComponent(run) +
          '&p=' + encodeURIComponent(h.slice(8));
  } else if (h.indexOf('diff=') === 0) {
    url = '/diff?d=' + encodeURIComponent(run) +
          '&f=' + encodeURIComponent(h.slice(5));
  } else if (content.dataset.default) {
    location.hash = '#f=' + content.dataset.default;
    return;
  }
  if (!url) return;
  content.innerHTML = '<p class="muted">Loading…</p>';
  fetch(url).then(function(r) { return r.text(); }).then(function(html) {
    content.innerHTML = html;
    var r = window._restore;
    window._restore = null;
    if (r && r.open) {
      var ds = $all('details', content);
      r.open.forEach(function(i) { if (ds[i]) ds[i].open = true; });
    }
    var want = (r && typeof r.content === 'number') ? r.content : 0;
    content.scrollTop = want;
    if (want) {
      // Lazy images may not have expanded the pane yet; re-apply once.
      setTimeout(function() {
        if (content.scrollTop < want) content.scrollTop = want;
      }, 400);
    }
    $all('.sidebar .nav a').forEach(function(a) {
      a.classList.toggle('active', a.getAttribute('href') === '#' + h);
    });
  });
}
window.addEventListener('hashchange', loadHash);
document.addEventListener('DOMContentLoaded', loadHash);

// Preserve scroll positions (and open sections) across reloads. The
// scrollable panes are inner divs, so native scroll restoration does not
// cover them.
function scrollKey() {
  return 'lv-scroll:' + location.pathname + location.search + location.hash;
}
function saveScroll() {
  var state = {};
  var c = $('#content');
  var p = $('.content');
  if (c) {
    state.content = c.scrollTop;
    state.open = $all('details', c)
      .map(function(d, i) { return d.open ? i : -1; })
      .filter(function(i) { return i >= 0; });
  } else if (p) {
    state.page = p.scrollTop;
  }
  var sb = $('.sidebar');
  if (sb) state.sidebar = sb.scrollTop;
  try { sessionStorage.setItem(scrollKey(), JSON.stringify(state)); }
  catch (e) {}
}
window.addEventListener('beforeunload', saveScroll);
window.addEventListener('pagehide', saveScroll);
window._restore = (function() {
  var raw = null;
  try { raw = sessionStorage.getItem(scrollKey()); } catch (e) {}
  if (!raw) return null;
  sessionStorage.removeItem(scrollKey());
  try { return JSON.parse(raw); } catch (e) { return null; }
})();

// Auto-refresh: poll /stamp every 10s; reload only when logs changed.
var REFRESH_MS = 10000;
function autoOn() { return localStorage.getItem('lv-auto') !== '0'; }
function toggleAuto() {
  localStorage.setItem('lv-auto', autoOn() ? '0' : '1');
  paintAutoBtn();
}
function paintAutoBtn() {
  var b = $('#arbtn');
  if (b) b.textContent = 'auto-refresh: ' + (autoOn() ? 'on' : 'off');
}
function showRefreshPill() {
  if ($('#refreshpill')) return;
  var p = document.createElement('button');
  p.id = 'refreshpill';
  p.textContent = 'Logs updated — refresh';
  p.onclick = function() { location.reload(); };
  document.body.appendChild(p);
}
function pollStamp() {
  if (document.visibilityState !== 'visible') return;
  var content = $('#content');
  var url = '/stamp' + (content && content.dataset.run
    ? '?d=' + encodeURIComponent(content.dataset.run) : '');
  fetch(url).then(function(r) { return r.text(); }).then(function(s) {
    if (window._stamp === undefined) { window._stamp = s; return; }
    if (s !== window._stamp) {
      window._stamp = s;
      if (autoOn()) { location.reload(); } else { showRefreshPill(); }
    }
  }).catch(function() {});
}
document.addEventListener('DOMContentLoaded', function() {
  paintAutoBtn();
  pollStamp();
  setInterval(pollStamp, REFRESH_MS);
  var f = $('#runfilter');
  if (f) {
    f.value = sessionStorage.getItem('lv-filter') || '';
    if (f.value) filterRuns(f.value);
  }
  var r = window._restore;
  if (r) {
    var sb = $('.sidebar');
    if (sb && typeof r.sidebar === 'number') sb.scrollTop = r.sidebar;
    var p = $('.content');
    if (!$('#content') && p && typeof r.page === 'number') {
      p.scrollTop = r.page;
      window._restore = null;
    }
  }
});

function filterLog() {
  var needle = $('#logfilter').value.toLowerCase();
  $all('#logview > span.logline').forEach(function(l) {
    l.style.display =
      !needle || l.textContent.toLowerCase().indexOf(needle) !== -1
      ? '' : 'none';
  });
}
"""


def page(title: str, topbar_extra: str, body: str) -> str:
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>%s</title><style>%s</style><script>%s</script></head>"
            "<body><div class='topbar'><h1><a href='/'>log viewer</a></h1>"
            "%s<button id='arbtn' onclick='toggleAuto()'></button>"
            "</div>%s</body></html>" %
            (esc(title), CSS, JS, topbar_extra, body))


def chip(label: Any, cls: str = "", title: str = "") -> str:
    return '<span class="chip %s" title="%s">%s</span>' % (cls, esc(title),
                                                           esc(str(label)))


def test_mark(ep: Dict[str, Any]) -> Tuple[str, str, str]:
    """(mark, css class, tooltip) for a test episode.

    The env-side verdict from info.log is authoritative; the
    transcript's own "Goal achieved" claim is shown parenthesized as a
    fallback only.
    """
    if "env_solved" in ep:
        if ep["env_solved"]:
            return "✓", "ok", "env eval: SOLVED"
        return "✗", "bad", "env eval: " + ep.get("env_msg", "failed")
    if ep.get("goal") is True:
        return "(✓)", "ok", "agent-reported only; no env verdict in info.log"
    if ep.get("goal") is False:
        return "(✗)", "bad", "agent-reported only; no env verdict in info.log"
    return "", "", ""


def episode_chips(episodes: List[Dict[str, Any]]) -> str:
    out = []
    for ep in episodes:
        label = "%03d" % ep["num"]
        cls, title = "", ""
        if ep["kind"] == "test":
            mark, cls, title = test_mark(ep)
            label += " t%s" % ep["task"]
            if mark:
                label += " " + mark
        else:
            label += " " + ep["kind"]
            cls = "kind-" + ep["kind"]
        out.append(chip(label, cls, title))
    return "".join(out)


# ----------------------------------------------------------------- pages


def run_row(r: Dict[str, Any]) -> str:
    summary = run_summary(r["rel"]) or {}
    eps = summary.get("episodes", [])
    tr = summary.get("test_results", [])
    tr_str = " → ".join("%d/%d" % t for t in tr) or "-"
    cost = summary.get("total_cost", 0.0)
    mstr = datetime.datetime.fromtimestamp(
        r["mtime"]).strftime("%Y-%m-%d %H:%M")
    key = ("%s %s %s" % (r["exp"], r["seed"], r["name"])).lower()
    return ("<tr class='runrow' data-key='%s'>"
            "<td><input type='checkbox' class='cmp' value='%s'></td>"
            "<td><a href='/run?d=%s'>%s</a></td><td>%s</td>"
            "<td>%s</td><td>%s</td><td>%s</td>"
            "<td class='muted'>%s</td></tr>" %
            (esc(key), esc(r["rel"]), q(
                r["rel"]), esc(r["name"]), esc(r["seed"]), episode_chips(eps),
             esc(tr_str), "$%.2f" % cost if cost else "-", mstr))


def index_page() -> str:
    runs = find_runs()
    families: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in runs:
        fam, _, rest = r["exp"].partition("/")
        families.setdefault(fam, {}).setdefault(rest or fam, []).append(r)
    body = [
        "<div class='content' style='height:calc(100vh - 45px);"
        "overflow-y:auto'>"
    ]
    if not runs:
        body.append("<p>No run_* directories found under %s.</p>" %
                    esc(LOGS_ROOT))
    table_head = ("<tr><th></th><th>run</th><th>seed</th><th>episodes"
                  "</th><th>test results (info.log)</th><th>cost</th>"
                  "<th>modified</th></tr>")
    for fam in sorted(families):
        exps = families[fam]
        n_runs = sum(len(v) for v in exps.values())
        body.append("<details class='grp family' data-key='fam:%s' open>"
                    "<summary>%s <span class='muted'>(%d experiments, %d runs)"
                    "</span></summary>" %
                    (esc(fam), esc(fam), len(exps), n_runs))
        for expname in sorted(exps):
            rows = "".join(run_row(r) for r in exps[expname])
            body.append(
                "<details class='grp exp' data-key='exp:%s/%s'>"
                "<summary>%s <span class='muted'>(%d runs)</span>"
                "</summary><table class='grid'>%s%s</table></details>" %
                (esc(fam), esc(expname), esc(expname), len(
                    exps[expname]), table_head, rows))
        body.append("</details>")
    body.append("</div>")
    topbar = ("<input id='runfilter' placeholder='filter runs…' "
              "oninput='filterRuns(this.value)'>"
              "<button onclick='setAllGroups(true)'>expand all</button>"
              "<button onclick='setAllGroups(false)'>collapse all</button>"
              "<button onclick='compareSelected()'>Compare selected"
              "</button><span class='crumb'>%s</span>" % esc(LOGS_ROOT))
    return page("runs - log viewer", topbar, "".join(body))


def file_tree_html(run_abs: str, run_rel: str, rel: str = "") -> str:
    """Nested details tree; big image dirs collapse to a gallery link."""
    abs_dir = os.path.join(run_abs, rel) if rel else run_abs
    try:
        entries = sorted(os.listdir(abs_dir))
    except OSError:
        return ""
    dirs = [e for e in entries if os.path.isdir(os.path.join(abs_dir, e))]
    files = [e for e in entries if not os.path.isdir(os.path.join(abs_dir, e))]
    out = []
    for d in dirs:
        child_rel = (rel + "/" + d) if rel else d
        child_abs = os.path.join(abs_dir, d)
        try:
            child_entries = os.listdir(child_abs)
        except OSError:
            child_entries = []
        n_imgs = sum(1 for c in child_entries
                     if os.path.splitext(c)[1].lower() in IMG_EXTS)
        if n_imgs > 20 and n_imgs > len(child_entries) * 0.8:
            out.append("<div class='nav'><a href='#gallery=%s'>%s/ "
                       "<span class='muted'>(%d images, gallery)</span>"
                       "</a></div>" % (q(child_rel), esc(d), n_imgs))
            continue
        out.append("<details><summary>%s/ <span class='muted'>(%d)</span>"
                   "</summary>%s</details>" %
                   (esc(d), len(child_entries),
                    file_tree_html(run_abs, run_rel, child_rel)))
    shown = files[:200]
    out.append("<div class='nav'>")
    for f in shown:
        child_rel = (rel + "/" + f) if rel else f
        out.append("<a href='#f=%s'>%s</a>" % (q(child_rel), esc(f)))
    if len(files) > 200:
        out.append("<span class='muted'>… %d more files</span>" %
                   (len(files) - 200))
    out.append("</div>")
    return "".join(out)


def run_page(run_rel: str) -> Optional[str]:
    run_abs = safe_join(run_rel)
    if not run_abs or not os.path.isdir(run_abs):
        return None
    summary = run_summary(run_rel)
    assert summary is not None
    eps = summary["episodes"]
    side = ["<div class='sidebar'>"]
    if eps:
        side.append("<h3>Episodes</h3><div class='nav'>")
        for ep in eps:
            mark, cls, title = "", "", ""
            if ep["kind"] == "test":
                mark, cls, title = test_mark(ep)
            cost = ("  $%.2f" % ep["solve_cost"] if "solve_cost" in ep else "")
            label = "%03d %s%s" % (
                ep["num"], ep["kind"] +
                (" task%d" % ep["task"] if ep["task"] is not None else ""),
                " " + mark if mark else "")
            side.append(
                "<a href='#f=%s'><span class='chip %s' title='%s'>"
                "%s</span><span class='muted'>%s</span></a>" %
                (q(ep["file"]), cls, esc(title), esc(label), esc(cost)))
        side.append("</div>")
    other_md = sorted(f for f in os.listdir(run_abs)
                      if f.endswith(".md") and not EPISODE_RE.match(f))
    logs = sorted(f for f in os.listdir(run_abs) if f.endswith(".log"))
    if other_md:
        side.append("<h3>Prompts</h3><div class='nav'>")
        side += ["<a href='#f=%s'>%s</a>" % (q(f), esc(f)) for f in other_md]
        side.append("</div>")
    if logs:
        side.append("<h3>Logs</h3><div class='nav'>")
        side += ["<a href='#f=%s'>%s</a>" % (q(f), esc(f)) for f in logs]
        side.append("</div>")
    side.append("<h3>All files</h3>")
    side.append(file_tree_html(run_abs, run_rel))
    side.append("</div>")
    default = q(eps[0]["file"]) if eps else ""
    tr = summary["test_results"]
    tr_str = " → ".join("%d/%d" % t for t in tr)
    banner = ("<span>%s</span><span>%s</span>" %
              (("test results: " + esc(tr_str)) if tr_str else "",
               ("total cost: $%.2f" %
                summary["total_cost"]) if summary["total_cost"] else ""))
    body = ("<div class='layout'>%s"
            "<div style='flex:1;display:flex;flex-direction:column;"
            "overflow:hidden'>"
            "<div class='banner' style='margin:8px 16px 0'>%s"
            "<span style='margin-left:auto'>"
            "<button onclick='setAllDetails(true)'>expand all</button> "
            "<button onclick='setAllDetails(false)'>collapse all</button>"
            "</span></div>"
            "<div class='content' id='content' data-run='%s' "
            "data-default='%s'><p class='muted'>Select a file.</p></div>"
            "</div></div>" % ("".join(side), banner, esc(run_rel), default))
    topbar = "<span class='crumb'>%s</span>" % esc(run_rel)
    return page(run_rel + " - log viewer", topbar, body)


# ------------------------------------------------------------- fragments


def episode_banner(ep: Dict[str, Any], n_rounds: int = 0) -> str:
    parts = []
    if ep["kind"] == "test":
        if "env_solved" in ep:
            if ep["env_solved"]:
                parts.append("<span class='ok'>SOLVED ✓ (env eval)</span>")
            else:
                parts.append("<span class='bad'>FAILED ✗ (env eval: %s)"
                             "</span>" % esc(ep.get("env_msg", "")))
        else:
            parts.append("<span class='muted'>no env verdict in info.log"
                         "</span>")
    if ep.get("goal") is True:
        parts.append("<span class='muted'>agent-reported: goal achieved"
                     "</span>")
    elif ep.get("goal") is False:
        parts.append("<span class='muted'>agent-reported: goal NOT achieved"
                     "</span>")
    if ep.get("task") is not None:
        parts.append("task %d" % ep["task"])
    if n_rounds > 1 and "round" in ep:
        parts.append("round %d" % ep["round"])
    parts.append("kind: %s" % ep["kind"])
    if "turns" in ep:
        parts.append("%d turns" % ep["turns"])
    if "solve_cost" in ep:
        parts.append("$%.2f this solve / $%.2f total" %
                     (ep["solve_cost"], ep["total_cost"]))
    banner = "<div class='banner'>%s</div>" % "".join("<span>%s</span>" % p
                                                      for p in parts)
    if ep.get("env_solved") is False and ep.get("goal") is True:
        banner += ("<div class='banner warn'>⚠ The transcript claims the "
                   "goal was achieved, but the env-side evaluation says "
                   "this task FAILED. Trust the env verdict.</div>")
    elif ep.get("env_solved") is True and ep.get("goal") is False:
        banner += ("<div class='banner warn'>⚠ The env-side evaluation "
                   "says SOLVED although the transcript reports the goal "
                   "as not achieved.</div>")
    return banner


def versions_nav(run_abs: str, run_rel: str, file_rel: str) -> str:
    """Prev/diff links when the file sits in a *_versions directory."""
    d = os.path.dirname(file_rel)
    if not d.endswith("_versions"):
        return ""
    dir_abs = os.path.join(run_abs, d)
    try:
        siblings = sorted(os.listdir(dir_abs))
    except OSError:
        return ""
    name = os.path.basename(file_rel)
    if name not in siblings:
        return ""
    idx = siblings.index(name)
    links = []
    if idx > 0:
        links.append("<a href='#diff=%s'>diff vs %s</a>" %
                     (q(file_rel), esc(siblings[idx - 1])))
        links.append("<a href='#f=%s'>← %s</a>" %
                     (q(d + "/" + siblings[idx - 1]), esc(siblings[idx - 1])))
    if idx < len(siblings) - 1:
        links.append("<a href='#f=%s'>%s →</a>" %
                     (q(d + "/" + siblings[idx + 1]), esc(siblings[idx + 1])))
    return ("<div class='banner'>version %d/%d &nbsp; %s</div>" %
            (idx + 1, len(siblings), " &nbsp;|&nbsp; ".join(links)))


def view_fragment(run_rel: str, file_rel: str) -> str:
    run_abs = safe_join(run_rel)
    file_abs = safe_join(run_rel + "/" + file_rel)
    if not run_abs or not file_abs:
        return "<p>Not found.</p>"
    if os.path.isdir(file_abs):
        return ("<h2>%s/</h2>%s" %
                (esc(file_rel), file_tree_html(run_abs, run_rel, file_rel)))
    if not os.path.isfile(file_abs):
        return "<p>Not found.</p>"
    name = os.path.basename(file_rel)
    ext = os.path.splitext(name)[1].lower()
    raw_rel = os.path.relpath(file_abs, os.path.realpath(LOGS_ROOT))
    raw_url = "/raw?p=" + q(raw_rel.replace(os.sep, "/"))
    header = ("<h2>%s <a class='muted' style='font-size:12px' href='%s' "
              "target='_blank'>raw</a></h2>" % (esc(file_rel), raw_url))
    if ext in IMG_EXTS:
        return header + "<img class='mdimg' style='max-width:96%%' src='%s'>" \
            % raw_url
    if ext in VIDEO_EXTS:
        return header + ("<video controls style='max-width:96%%' src='%s'>"
                         "</video>" % raw_url)
    if ext == ".md":
        banner = ""
        if EPISODE_RE.match(name):
            summary = run_summary(run_rel) or {}
            matches = [
                e for e in summary.get("episodes", []) if e["file"] == name
            ]
            if matches:
                banner = episode_banner(matches[0],
                                        len(summary.get("rounds", [])))
        text, truncated = read_text(file_abs, max_bytes=4 * 1024 * 1024)
        note = ("<p class='muted'>(truncated to last 4MB)</p>"
                if truncated else "")
        return header + banner + note + render_md_document(text, run_rel)
    if ext == ".log" or name in ("info.log", "debug.log"):
        text, truncated = read_text(file_abs, max_bytes=4 * 1024 * 1024)
        note = ("<p class='muted'>(showing last 4MB)</p>" if truncated else "")
        lines = "".join("<span class='logline'>%s\n</span>" % ansi_to_html(l)
                        for l in text.splitlines())
        return (header + note +
                "<input id='logfilter' placeholder='filter lines…' "
                "oninput='filterLog()' style='width:300px;padding:4px 10px;"
                "border:1px solid var(--border);border-radius:6px;"
                "background:var(--bg);color:var(--fg)'>" +
                "<pre class='code logview' id='logview'>%s</pre>" % lines)
    if ext in CODE_EXTS or ext in TEXT_EXTS or ext == "":
        text, truncated = read_text(file_abs, max_bytes=4 * 1024 * 1024)
        note = ("<p class='muted'>(truncated to last 4MB)</p>"
                if truncated else "")
        extra = versions_nav(run_abs, run_rel, file_rel)
        body = "<pre class='code'><code>%s</code></pre>" % esc(text)
        if ext == ".txt":
            body += thumbs_for_paths(text, run_rel)
        return header + extra + note + body
    size = os.path.getsize(file_abs)
    return header + ("<p><a href='%s'>download</a> (%.1f KB)</p>" %
                     (raw_url, size / 1024.0))


def diff_fragment(run_rel: str, file_rel: str) -> str:
    run_abs = safe_join(run_rel)
    file_abs = safe_join(run_rel + "/" + file_rel)
    if not run_abs or not file_abs or not os.path.isfile(file_abs):
        return "<p>Not found.</p>"
    d = os.path.dirname(file_rel)
    dir_abs = os.path.join(run_abs, d)
    siblings = sorted(os.listdir(dir_abs))
    name = os.path.basename(file_rel)
    idx = siblings.index(name) if name in siblings else 0
    if idx == 0:
        return "<p>No previous version to diff against.</p>"
    prev = siblings[idx - 1]
    old, _ = read_text(os.path.join(dir_abs, prev))
    new, _ = read_text(file_abs)
    diff = difflib.unified_diff(old.splitlines(),
                                new.splitlines(),
                                fromfile=prev,
                                tofile=name,
                                lineterm="")
    out = []
    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            out.append("<span class='add'>%s</span>" % esc(line))
        elif line.startswith("-") and not line.startswith("---"):
            out.append("<span class='del'>%s</span>" % esc(line))
        elif line.startswith("@@"):
            out.append("<span class='hunk'>%s</span>" % esc(line))
        else:
            out.append(esc(line) + "\n")
    header = ("<h2>diff: %s → %s</h2>"
              "<p><a href='#f=%s'>view %s</a> &nbsp; "
              "<a href='#f=%s'>view %s</a></p>" %
              (esc(prev), esc(name), q(d + "/" + prev), esc(prev), q(file_rel),
               esc(name)))
    if len(out) == 0:
        return header + "<p class='muted'>Files are identical.</p>"
    return header + "<pre class='code diff'>%s</pre>" % "".join(out)


def gallery_fragment(run_rel: str, dir_rel: str) -> str:
    dir_abs = safe_join(run_rel + "/" + dir_rel)
    if not dir_abs or not os.path.isdir(dir_abs):
        return "<p>Not found.</p>"
    root = os.path.realpath(LOGS_ROOT)
    images = sorted(f for f in os.listdir(dir_abs)
                    if os.path.splitext(f)[1].lower() in IMG_EXTS)
    groups: Dict[str, List[str]] = {}
    for f in images:
        m = re.match(r"^(iter\d+_test\d+|task\d+)", f)
        groups.setdefault(m.group(1) if m else "(other)", []).append(f)
    out = [
        "<h2>%s/ <span class='muted'>(%d images)</span></h2>" %
        (esc(dir_rel), len(images))
    ]
    for gname in sorted(groups):
        imgs = groups[gname]
        cells = []
        for f in imgs:
            rel = os.path.relpath(os.path.join(dir_abs, f), root)
            url = "/raw?p=" + q(rel.replace(os.sep, "/"))
            cells.append("<a class='shot' href='%s' target='_blank'>"
                         "<img loading='lazy' src='%s'><br>%s</a>" %
                         (url, url, esc(f)))
        out.append("<details%s><summary>%s <span class='muted'>(%d)"
                   "</span></summary><div class='gallery'>%s</div>"
                   "</details>" % (" open" if len(groups) == 1 else "",
                                   esc(gname), len(imgs), "".join(cells)))
    return "".join(out)


def compare_page(run_rels: List[str]) -> str:
    summaries: List[Tuple[str, Dict[str, Any]]] = []
    for rel in run_rels:
        s = run_summary(rel)
        if s:
            summaries.append((rel, s))
    if len(summaries) < 2:
        return page(
            "compare", "", "<div class='content'>"
            "<p>Need at least two valid runs.</p></div>")
    all_tasks = sorted({
        ep["task"]
        for _, s in summaries for ep in s["episodes"]
        if ep["kind"] == "test" and ep["task"] is not None
    })
    head = "".join(
        "<th><a href='/run?d=%s'>%s</a><br>"
        "<span class='muted'>%s</span></th>" %
        (q(rel), esc(rel.split("/")[-1]), esc("/".join(rel.split("/")[:-1])))
        for rel, _ in summaries)
    rows = []
    for task in all_tasks:
        cells = []
        for _, s in summaries:
            verdicts = [r[task] for r in s.get("rounds", []) if task in r]
            if verdicts:
                marks = " → ".join(
                    "<span class='%s' title='%s'>%s</span>" %
                    (("ok", esc(v["msg"]),
                      "✓") if v["solved"] else ("bad", esc(v["msg"]), "✗"))
                    for v in verdicts)
            else:
                attempts = [
                    ep for ep in s["episodes"]
                    if ep["kind"] == "test" and ep["task"] == task
                ]
                marks = "".join(
                    "<span class='%s'>%s</span>" %
                    (("ok", "(✓)") if ep.get("goal") else (
                        ("bad", "(✗)") if ep.get("goal") is False else
                        ("muted", "?"))) for ep in attempts)
                if marks:
                    marks += " <span class='muted'>agent-reported</span>"
            cells.append("<td>%s</td>" % (marks or "-"))
        rows.append("<tr><th>task %d</th>%s</tr>" % (task, "".join(cells)))

    def stat_row(label: str, fn: Callable[[Dict[str, Any]], str]) -> str:
        return ("<tr><th>%s</th>%s</tr>" %
                (label, "".join("<td>%s</td>" % fn(s) for _, s in summaries)))

    rows.append(
        stat_row(
            "test results",
            lambda s: esc(" → ".join("%d/%d" % t
                                     for t in s["test_results"]) or "-")))
    rows.append(stat_row("episodes", lambda s: str(len(s["episodes"]))))
    rows.append(
        stat_row(
            "explore / learn", lambda s: "%d / %d" %
            (sum(1 for e in s["episodes"] if e["kind"] == "explore"),
             sum(1 for e in s["episodes"] if e["kind"] == "learn"))))
    rows.append(
        stat_row(
            "total cost", lambda s: ("$%.2f" % s["total_cost"])
            if s["total_cost"] else "-"))
    body = ("<div class='content' style='height:calc(100vh - 45px);"
            "overflow-y:auto'><h2>Run comparison</h2>"
            "<table class='grid'><tr><th></th>%s</tr>%s</table></div>" %
            (head, "".join(rows)))
    return page("compare - log viewer", "", body)


# ------------------------------------------------------------ HTTP layer


class Handler(BaseHTTPRequestHandler):

    def log_message(
            self,
            format: str,  # pylint: disable=redefined-builtin
            *args: Any) -> None:
        pass

    def send_html(self, html_text: str, status: int = 200) -> None:
        data = html_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_raw(self, path: str) -> None:
        ctype = mimetypes.guess_type(path)[0]
        if not ctype or os.path.splitext(path)[1].lower() in (
            {".md", ".log", ".py"} | TEXT_EXTS):
            ctype = "text/plain; charset=utf-8"
        size = os.path.getsize(path)
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", range_header or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:
                start = max(size - int(m.group(2)), 0)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range",
                             "bytes %d-%d/%d" % (start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            try:
                self.send_html("<pre>%s</pre>" % esc(str(e)), status=500)
            except OSError:
                pass

    def route(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = {
            k: v[0]
            for k, v in urllib.parse.parse_qs(parsed.query).items()
        }
        route = parsed.path
        if route == "/":
            self.send_html(index_page())
        elif route == "/run":
            html_text = run_page(params.get("d", ""))
            if html_text is None:
                self.send_html("<p>Run not found.</p>", status=404)
            else:
                self.send_html(html_text)
        elif route == "/view":
            self.send_html(
                view_fragment(params.get("d", ""), params.get("f", "")))
        elif route == "/diff":
            self.send_html(
                diff_fragment(params.get("d", ""), params.get("f", "")))
        elif route == "/gallery":
            self.send_html(
                gallery_fragment(params.get("d", ""), params.get("p", "")))
        elif route == "/compare":
            rels = [r for r in params.get("runs", "").split(";") if r]
            self.send_html(compare_page(rels))
        elif route == "/stamp":
            d = params.get("d", "")
            stamp = run_stamp(d) if d else index_stamp()
            data = stamp.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif route == "/raw":
            path = safe_join(params.get("p", ""))
            if not path or not os.path.isfile(path):
                self.send_html("<p>Not found.</p>", status=404)
            else:
                self.send_raw(path)
        else:
            self.send_html("<p>Not found.</p>", status=404)


def main() -> None:
    global LOGS_ROOT
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--logs",
                        default="logs",
                        help="logs root directory (default: logs)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    LOGS_ROOT = os.path.abspath(args.logs)
    if not os.path.isdir(LOGS_ROOT):
        sys.exit("logs directory not found: %s" % LOGS_ROOT)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Serving %s at http://%s:%d/" % (LOGS_ROOT, args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
