"""Persistent per-run solve journal and attempt log.

Two markdown files in the sandbox that carry knowledge across solve
attempts, test tasks, and learning cycles:

- ``journal.md`` is the AGENT's notebook. Solve and learn sessions
  append to it with the ordinary file tools (no dedicated tool): short
  factual entries - what was tried with exact parameters, what was
  measured, what to try differently. The prompts ask for facts and
  measurements rather than verdicts: a recorded "X is impossible"
  from a failed attempt would re-import exactly the anchoring a fresh
  context is meant to shed, while "tried yaws 0-15 deg at x in
  [0.50, 0.54], all stopped >=5 cm short" steers the next attempt
  without foreclosing it.
- ``attempts.md`` is the HARNESS's log, never edited by the agent:
  each task's goal + initial state (once per task) and each
  attempt's outcome and captured or best refused plan, so the
  essentials of every attempt are on record even when the agent
  writes nothing.

Fresh-context solve sessions read both from their prompt (tail-capped
so recent attempts stay intact), so knowledge travels through these
curated channels instead of raw transcript history.

Phase lifecycle: learning-phase content persists for the whole run
and accumulates across online-learning cycles, so every evaluation
starts from all learning knowledge so far. Test-phase additions live
only for their own evaluation: at ``end_test_phase`` the approach
archives both files to the run's log dir (outside the sandbox, so the
agent cannot read them) and rolls them back to their pre-test content
via :func:`read_raw` / :func:`restore` - entries written while
solving one evaluation's test tasks must not leak into the next.
"""

from __future__ import annotations

import os
from typing import Optional

JOURNAL_FILENAME = "journal.md"
# The harness-owned attempt log (task contexts, attempt outcomes).
ATTEMPTS_FILENAME = "attempts.md"

# Per-entry cap for harness attempt-log entries: the first entry per
# task embeds the init-state feature dict (the prompt's own
# representation) and a captured plan. The writer orders the layout
# block last, so tail truncation at this cap can only ever cut layout,
# never the outcome or the captured plan.
MAX_ENTRY_CHARS = 4000
MAX_AUTO_ENTRY_CHARS = MAX_ENTRY_CHARS
# Cap on how much of each file is injected into a solve prompt.
# Tail-biased: recent attempts (usually the same task) matter most.
MAX_PROMPT_CHARS = 6000

# The learn-phase-maintained domain strategy document. Unlike the
# append-only journal (facts and measurements), strategy.md is a LIVING
# document the learn agent rewrites freely each cycle: its best current
# natural-language account of how to solve tasks in this domain. Solve
# prompts inject it as explicitly-advisory reference.
STRATEGY_FILENAME = "strategy.md"

# Cap on how much strategy is injected into a solve prompt. Head-biased
# (unlike the journal): the document is curated, so its lead carries the
# headline strategy and a tail truncation only cuts detail.
MAX_STRATEGY_PROMPT_CHARS = 4000


def journal_path(sandbox_dir: str) -> str:
    """Host path of the run's journal file."""
    return os.path.join(sandbox_dir, JOURNAL_FILENAME)


def attempts_path(sandbox_dir: str) -> str:
    """Host path of the run's harness-owned attempt log."""
    return os.path.join(sandbox_dir, ATTEMPTS_FILENAME)


def strategy_path(sandbox_dir: str) -> str:
    """Host path of the run's domain strategy document."""
    return os.path.join(sandbox_dir, STRATEGY_FILENAME)


def read_strategy(sandbox_dir: Optional[str],
                  max_chars: int = MAX_STRATEGY_PROMPT_CHARS) -> str:
    """Strategy document content for prompt injection ("" when absent).

    Head-biased truncation: the document is curated by the learn agent,
    so the front holds the headline strategy; a truncation notice marks
    the cut so readers know detail was dropped.
    """
    if not sandbox_dir:
        return ""
    path = strategy_path(sandbox_dir)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if len(content) > max_chars:
        # Cut at a line boundary, never mid-word.
        head = content[:max_chars]
        cut = head.rfind("\n")
        if cut > 0:
            head = head[:cut]
        content = (head.rstrip() +
                   "\n[strategy truncated at the prompt cap - read "
                   f"./{STRATEGY_FILENAME} for the rest]")
    return content


def append_entry(sandbox_dir: str,
                 header: str,
                 body: str,
                 max_chars: int = MAX_ENTRY_CHARS,
                 filename: str = ATTEMPTS_FILENAME) -> Optional[str]:
    """Append one harness entry; returns a truncation notice or None.

    ``header`` becomes a ``### <header>`` line; ``body`` is written
    verbatim below it, truncated at ``max_chars`` (default
    :data:`MAX_ENTRY_CHARS`; harness auto-entries pass
    :data:`MAX_AUTO_ENTRY_CHARS`).
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    note: Optional[str] = None
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
        body += "\n[entry truncated at the per-entry size cap]"
        note = (f"entry truncated to {max_chars} chars - keep journal "
                "entries short and factual")
    with open(os.path.join(sandbox_dir, filename), "a", encoding="utf-8") as f:
        f.write(f"### {header.strip()}\n{body}\n\n")
    return note


def read_raw(sandbox_dir: Optional[str],
             filename: str = JOURNAL_FILENAME) -> Optional[str]:
    """Exact file content, or None if the file does not exist.

    Unlike :func:`read_journal` there is no prompt trimming and the
    absent-file case is distinguishable from an empty file, so the
    result is a faithful snapshot for :func:`restore`.
    """
    if not sandbox_dir:
        return None
    path = os.path.join(sandbox_dir, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def restore(sandbox_dir: str,
            snapshot: Optional[str],
            filename: str = JOURNAL_FILENAME) -> None:
    """Reset the file to a :func:`read_raw` snapshot.

    A ``None`` snapshot means the file did not exist, so it is removed
    if present.
    """
    path = os.path.join(sandbox_dir, filename)
    if snapshot is None:
        if os.path.isfile(path):
            os.remove(path)
        return
    os.makedirs(sandbox_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(snapshot)


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
