"""Persistent per-run journal and round log.

Two markdown files in the sandbox that carry knowledge across the rounds
and levels of a continual run:

- ``journal.md`` is the AGENT's notebook. The agent appends to it with
  the ordinary file tools (no dedicated tool): short factual entries -
  what was tried with exact parameters, what was measured, what to try
  differently. The prompts ask for facts and measurements rather than
  verdicts: a recorded "X is impossible" from a failed attempt would
  re-import exactly the anchoring a fresh context is meant to shed,
  while "tried yaws 0-15 deg at x in [0.50, 0.54], all stopped >=5 cm
  short" steers the next attempt without foreclosing it.
- ``attempts.md`` is the HARNESS's log, never edited by the agent: one
  entry per round (what the agent did in the environment) and per model
  change, so the essentials of every round are on record even when the
  agent writes nothing.

Each round's query injects both (tail-capped so recent entries stay
intact), so knowledge travels through these curated channels instead of
raw transcript history.
"""

from __future__ import annotations

import os
from typing import Optional

JOURNAL_FILENAME = "journal.md"
# The harness-owned round log.
ATTEMPTS_FILENAME = "attempts.md"

# Per-entry cap for harness log entries.
MAX_ENTRY_CHARS = 4000
# Cap on how much of each file is injected into a query. Tail-biased:
# recent rounds matter most.
MAX_PROMPT_CHARS = 6000


def append_entry(sandbox_dir: str,
                 header: str,
                 body: str,
                 max_chars: int = MAX_ENTRY_CHARS,
                 filename: str = ATTEMPTS_FILENAME) -> None:
    """Append one harness entry.

    ``header`` becomes a ``### <header>`` line; ``body`` is written
    verbatim below it, truncated at ``max_chars``.
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
        body += "\n[entry truncated at the per-entry size cap]"
    with open(os.path.join(sandbox_dir, filename), "a", encoding="utf-8") as f:
        f.write(f"### {header.strip()}\n{body}\n\n")


def read_journal(sandbox_dir: Optional[str],
                 max_chars: int = MAX_PROMPT_CHARS,
                 filename: str = JOURNAL_FILENAME) -> str:
    """File content for prompt injection ('' if absent or empty).

    Over ``max_chars`` the head is dropped at an entry boundary with a
    truncation marker, keeping the most recent entries intact.
    """
    if not sandbox_dir:
        return ""
    path = os.path.join(sandbox_dir, filename)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if len(content) <= max_chars:
        return content
    tail = content[-max_chars:]
    cut = tail.find("\n### ")
    if cut != -1:
        tail = tail[cut + 1:]
    return ("[journal truncated: older entries omitted, most recent "
            f"kept]\n{tail}")
