"""Session transcripts and replay frames for the continual viewer.

Two jobs:

* parse the markdown transcripts ``agent_sdk.log_formatter.
  format_conversation_markdown`` writes for every play and learn session
  (``NNN_<kind>_<stamp>.md``) into turns of thinking, assistant text,
  tool calls and tool results;
* pair each recorded level event (``index.jsonl``) with the tool call
  that caused it and the agent's text that led to it, so a replay can
  show, per frame, the render, the action and the reasoning, the way the
  ARC-AGI-3 replay pairs a frame with its ``reasoning`` log.

Pairing walks a session's env and skill tool calls in order and assigns
index entries in the same order: one ``skills_invoke`` is one ``invoke``
entry, one ``skills_execute_plan`` is as many ``invoke`` entries as its
result reports, one ``env_reset`` is one agent reset. Entries are first
bucketed by session from their timestamps.
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

TURN_RE = re.compile(r"^### Turn (\d+)\s*$")
TOOL_CALL_RE = re.compile(
    r"^\*\*Tool Call:\*\* `([^`]+)` \(id: `([^`]*)`\)\s*$")
TOOL_RESULT_RE = re.compile(
    r"^\*\*Tool (Result|Error)\*\* \(tool_use_id: `([^`]*)`\):\s*$")
ASSISTANT_RE = re.compile(r"^\*\*Assistant:\*\* ?(.*)$")
USER_RE = re.compile(r"^\*\*User:\*\* ?(.*)$")
RESULT_LINE_RE = re.compile(r"^\*\*Result:\*\* (.*)$")
ERROR_LINE_RE = re.compile(r"^\*\*Error:\*\* (.*)$")
FENCE_RE = re.compile(r"^```")
INPUT_KEY_RE = re.compile(r"^\*(\w+):\*\s*$")
META_RE = re.compile(r"^- \*\*(.+?):\*\* (.*)$")
SESSION_LOG_RE = re.compile(
    r"^(\d{3})_([a-z_]+?)(?:_task\d+)?_(\d{8}_\d{6})\.md$")
PLAN_LINE_RE = re.compile(r"^\[(\d+)/(\d+)\] ")
MCP_PREFIX = "mcp__predicator_tools__"

ENV_TOOLS = {"skills_invoke", "skills_execute_plan", "env_reset", "env_step"}


@dataclass
class ToolCall:
    """One tool call and its result."""
    name: str
    tool_id: str
    args: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    is_error: bool = False
    turn: int = 0

    @property
    def short_name(self) -> str:
        """The tool name without the MCP prefix."""
        return self.name[len(MCP_PREFIX):] if self.name.startswith(
            MCP_PREFIX) else self.name


@dataclass
class Turn:
    """One assistant turn: what it thought, said and called."""
    number: int
    thinking: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    calls: List[ToolCall] = field(default_factory=list)


@dataclass
class Transcript:
    """A parsed session log."""
    name: str
    number: int
    kind: str
    stamp: Optional[datetime.datetime]
    prompt: str = ""
    turns: List[Turn] = field(default_factory=list)
    result_line: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def calls(self) -> List[ToolCall]:
        """Every tool call in order."""
        return [c for t in self.turns for c in t.calls]

    @property
    def start_ts(self) -> Optional[float]:
        """The session's start as a unix timestamp."""
        return self.stamp.timestamp() if self.stamp else None


def _stamp_from_name(
        name: str) -> Tuple[int, str, Optional[datetime.datetime]]:
    m = SESSION_LOG_RE.match(name)
    if m is None:
        return 0, "", None
    try:
        stamp = datetime.datetime.strptime(m.group(3), "%Y%m%d_%H%M%S")
    except ValueError:
        stamp = None
    return int(m.group(1)), m.group(2), stamp


def parse_transcript(text: str, name: str = "") -> Transcript:
    """Parse one session markdown into a :class:`Transcript`."""
    number, kind, stamp = _stamp_from_name(name)
    tx = Transcript(name=name, number=number, kind=kind, stamp=stamp)
    lines = text.splitlines()
    # Prompt: between "## Prompt" and "## Conversation".
    try:
        p0 = lines.index("## Prompt")
    except ValueError:
        p0 = -1
    try:
        c0 = lines.index("## Conversation")
    except ValueError:
        c0 = len(lines)
    if p0 >= 0:
        tx.prompt = "\n".join(lines[p0 + 1:c0]).strip("\n")
    for line in lines[:c0 if c0 > 0 else 0]:
        m = META_RE.match(line)
        if m and m.group(1) == "Timestamp" and tx.stamp is None:
            try:
                tx.stamp = datetime.datetime.strptime(
                    m.group(2).strip(), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
    body = lines[c0 + 1:] if c0 < len(lines) else []
    _parse_conversation(body, tx)
    return tx


def _parse_conversation(lines: Sequence[str], tx: Transcript) -> None:
    """Walk the conversation lines with a small state machine."""
    calls_by_id: Dict[str, ToolCall] = {}
    turn: Optional[Turn] = None
    i = 0
    n = len(lines)

    def _read_fence(start: int) -> Tuple[str, int]:
        """Return the fenced block starting at ``start`` and the index of
        the line after its closing fence."""
        assert FENCE_RE.match(lines[start])
        j = start + 1
        block: List[str] = []
        while j < n and not FENCE_RE.match(lines[j]):
            block.append(lines[j])
            j += 1
        return "\n".join(block), min(j + 1, n)

    while i < n:
        line = lines[i]
        m = TURN_RE.match(line)
        if m:
            turn = Turn(int(m.group(1)))
            tx.turns.append(turn)
            i += 1
            continue
        if line.strip() == "*[thinking]*":
            i += 1
            quoted: List[str] = []
            while i < n and lines[i].startswith(">"):
                quoted.append(lines[i][1:].lstrip(" "))
                i += 1
            if turn is None:
                turn = Turn(len(tx.turns) + 1)
                tx.turns.append(turn)
            turn.thinking.append("\n".join(quoted).strip("\n"))
            continue
        m = ASSISTANT_RE.match(line)
        if m:
            text = [m.group(1)]
            i += 1
            while i < n and not _is_marker(lines[i]):
                text.append(lines[i])
                i += 1
            if turn is None:
                turn = Turn(len(tx.turns) + 1)
                tx.turns.append(turn)
            turn.texts.append("\n".join(text).strip("\n"))
            continue
        m = TOOL_CALL_RE.match(line)
        if m:
            call = ToolCall(name=m.group(1),
                            tool_id=m.group(2),
                            turn=turn.number if turn else 0)
            i += 1
            # Inputs: "*key:*" + fence blocks and/or one json fence.
            while i < n:
                km = INPUT_KEY_RE.match(lines[i])
                if km and i + 1 < n and FENCE_RE.match(lines[i + 1]):
                    value, i = _read_fence(i + 1)
                    call.args[km.group(1)] = value
                    continue
                if FENCE_RE.match(lines[i]):
                    value, i = _read_fence(i)
                    try:
                        parsed = json.loads(value)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        call.args.update(parsed)
                    else:
                        call.args.setdefault("_raw", value)
                    continue
                if lines[i].strip() == "":
                    i += 1
                    continue
                break
            if turn is None:
                turn = Turn(len(tx.turns) + 1)
                tx.turns.append(turn)
            turn.calls.append(call)
            if call.tool_id:
                calls_by_id[call.tool_id] = call
            continue
        m = TOOL_RESULT_RE.match(line)
        if m:
            is_error = m.group(1) == "Error"
            tool_id = m.group(2)
            i += 1
            parts: List[str] = []
            while i < n:
                if FENCE_RE.match(lines[i]):
                    value, i = _read_fence(i)
                    parts.append(value)
                    continue
                if lines[i].startswith("*[image:"):
                    parts.append(lines[i].strip("* "))
                    i += 1
                    continue
                if lines[i].strip() == "":
                    i += 1
                    continue
                break
            target = calls_by_id.get(tool_id)
            if target is not None:
                target.result = "\n".join(parts)
                target.is_error = is_error
            continue
        m = RESULT_LINE_RE.match(line)
        if m:
            tx.result_line = m.group(1).strip()
            i += 1
            continue
        m = ERROR_LINE_RE.match(line)
        if m:
            tx.errors.append(m.group(1).strip())
            i += 1
            continue
        i += 1


def _is_marker(line: str) -> bool:
    return bool(
        TURN_RE.match(line) or TOOL_CALL_RE.match(line)
        or TOOL_RESULT_RE.match(line) or ASSISTANT_RE.match(line)
        or USER_RE.match(line) or RESULT_LINE_RE.match(line)
        or ERROR_LINE_RE.match(line) or line.strip() == "*[thinking]*"
        or line.strip() == "---")


# ── Replay assembly ────────────────────────────────────────────────


def entries_consumed(call: ToolCall) -> int:
    """How many index entries one env/skill call produced."""
    name = call.short_name
    if name == "skills_invoke":
        return 0 if call.is_error and not call.result.strip() else 1
    if name == "skills_execute_plan":
        return sum(1 for ln in call.result.splitlines()
                   if PLAN_LINE_RE.match(ln))
    if name == "env_reset":
        return 1
    return 0


def _wants(call: ToolCall) -> Optional[str]:
    name = call.short_name
    if name in ("skills_invoke", "skills_execute_plan"):
        return "invoke"
    if name == "env_reset":
        return "reset"
    return None


def pair_entries(entries: Sequence[Dict[str, Any]],
                 transcripts: Sequence[Transcript]) -> List[Dict[str, Any]]:
    """Attach session, turn, call and reasoning to index entries.

    Returns a new list of entry dicts (copies) with the keys ``session``,
    ``turn``, ``call`` (``{name, args, result, is_error}``), ``thinking``
    and ``texts`` added where a pairing was found.
    """
    out = [dict(e) for e in entries]
    ordered = sorted((t for t in transcripts if t.start_ts is not None),
                     key=lambda t: t.start_ts or 0.0)
    if not ordered:
        return out
    # Bucket entries by session start times.
    starts = [t.start_ts or 0.0 for t in ordered]
    buckets: Dict[int, List[int]] = {}
    for idx, e in enumerate(out):
        t = float(e.get("t") or 0.0)
        which = -1
        for s_idx, start in enumerate(starts):
            if t >= start - 1.0:
                which = s_idx
        if which >= 0:
            buckets.setdefault(which, []).append(idx)
    for s_idx, tx in enumerate(ordered):
        pending = [
            i for i in buckets.get(s_idx, [])
            if out[i].get("event") in ("invoke", "reset") and (
                out[i].get("event") != "reset" or out[i].get("by") == "agent")
        ]
        last_turn = 0
        for call in tx.calls:
            kind = _wants(call)
            if kind is None:
                continue
            count = entries_consumed(call)
            if count <= 0:
                continue
            reasoning_turns = [
                t for t in tx.turns if last_turn < t.number <= call.turn
            ]
            thinking = [th for t in reasoning_turns for th in t.thinking]
            texts = [tx_ for t in reasoning_turns for tx_ in t.texts]
            taken = 0
            for i in list(pending):
                if taken >= count:
                    break
                if out[i].get("event") != kind:
                    continue
                pending.remove(i)
                taken += 1
                out[i]["session"] = tx.number
                out[i]["session_name"] = tx.name
                out[i]["turn"] = call.turn
                out[i]["call"] = {
                    "name": call.short_name,
                    "args": call.args,
                    "result": call.result,
                    "is_error": call.is_error,
                }
                out[i]["thinking"] = thinking
                out[i]["texts"] = texts
            last_turn = call.turn
    return out


def frames_from_entries(entries: Sequence[Dict[str, Any]],
                        render_url: Any) -> List[Dict[str, Any]]:
    """Replay frames: one per index entry, with a render URL and marker."""
    frames = []
    for i, e in enumerate(entries):
        event = str(e.get("event", ""))
        frames.append({
            "i":
            i,
            "event":
            event,
            "marker":
            event in ("level_start", "reset", "resume", "win", "game_over"),
            "episode":
            e.get("episode"),
            "skill":
            e.get("skill", ""),
            "params":
            e.get("params", []),
            "status":
            e.get("status", ""),
            "reason":
            e.get("reason", "") or e.get("episode_reason", ""),
            "steps":
            e.get("steps"),
            "level_steps":
            e.get("level_steps"),
            "run_steps":
            e.get("run_steps"),
            "state":
            e.get("state", ""),
            "by":
            e.get("by", ""),
            "expected":
            e.get("expected", []),
            "expected_absent":
            e.get("expected_absent", []),
            "missing":
            e.get("missing", []),
            "present":
            e.get("present", []),
            "atoms":
            e.get("atoms"),
            "env_atoms":
            e.get("env_atoms"),
            "note":
            e.get("note", ""),
            "goal":
            e.get("goal", []),
            "goal_nl":
            e.get("goal_nl", ""),
            "t":
            e.get("t"),
            "render":
            render_url(e.get("render")),
            "session":
            e.get("session"),
            "session_name":
            e.get("session_name"),
            "turn":
            e.get("turn"),
            "call":
            e.get("call"),
            "thinking":
            e.get("thinking", []),
            "texts":
            e.get("texts", []),
            "verified":
            e.get("verified"),
            "replayed_steps":
            e.get("replayed_steps"),
        })
    return frames
