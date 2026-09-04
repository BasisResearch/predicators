#!/usr/bin/env python3
"""Aggregate continual-protocol scorecards into tables.

Reads every ``scorecards/<run_id>.json`` (see ``predicators/run/
scorecard.py``) and writes two CSVs plus a Markdown summary:

* ``runs.csv``: one row per run with the run-level totals and the end
  reason;
* ``levels.csv``: one row per (run, level) with the section 4.4 base
  metrics, so any aggregate can be recomputed later without touching the
  harness;
* ``summary.md``: per env, one table with one row per arm: runs, mean
  levels won, mean steps, mean resets, mean steps before the first win
  over won levels, and the count of runs that won every level.

The protocol deliberately defines no headline score; this script reports
base metrics and leaves the aggregation to the analysis that follows.

Usage:
    python scripts/aggregate_scorecards.py [--scorecards scorecards] \\
        [--out analysis/scorecards]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from typing import Any, Callable, Dict, List, Sequence

RUN_COLUMNS = [
    "run_id", "env", "arm", "seed", "config", "git_sha", "end_reason",
    "levels_total", "levels_completed", "total_steps", "total_resets",
    "total_skill_invocations", "total_wall_clock", "total_downtime",
    "total_llm_cost", "step_cap", "started_at", "finished_at"
]

LEVEL_COLUMNS = [
    "run_id", "env", "arm", "seed", "level", "split", "task_idx", "attempted",
    "won", "won_at_step", "steps", "resets", "skill_invocations",
    "failed_skill_invocations", "game_overs", "game_over_reasons",
    "divergences", "wall_clock", "wall_clock_env", "steps_before_first_win",
    "resets_before_first_win", "preemptions", "resumes", "downtime",
    "harness_resets", "interrupted_invocations", "episodes", "llm_cost_usd",
    "sim_rollouts", "fits", "sessions"
]


def load_cards(root: str) -> List[Dict[str, Any]]:
    """Every scorecard under ``root``."""
    cards: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return cards
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(root, name), "r", encoding="utf-8") as f:
            try:
                card = json.load(f)
            except ValueError:
                continue
        card.setdefault("run_id", name[:-len(".json")])
        cards.append(card)
    return cards


def run_rows(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per run."""
    rows = []
    for card in cards:
        totals = card.get("totals") or {}
        row = {k: card.get(k, "") for k in RUN_COLUMNS}
        for key in ("levels_total", "levels_completed", "total_steps",
                    "total_resets", "total_skill_invocations",
                    "total_wall_clock", "total_downtime", "total_llm_cost"):
            row[key] = totals.get(key, "")
        rows.append(row)
    return rows


def level_rows(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (run, level)."""
    rows = []
    for card in cards:
        for lv in card.get("levels", []):
            sandbox = lv.get("sandbox") or {}
            reasons = sorted(set(str(g) for g in lv.get("game_overs", [])))
            rows.append({
                "run_id":
                card["run_id"],
                "env":
                card.get("env", ""),
                "arm":
                card.get("arm", ""),
                "seed":
                card.get("seed", ""),
                "level":
                int(lv["index"]) + 1,
                "split":
                lv.get("split", ""),
                "task_idx":
                lv.get("task_idx", ""),
                "attempted":
                lv.get("attempted", False),
                "won":
                lv.get("won", False),
                "won_at_step":
                lv.get("won_at_step", ""),
                "steps":
                lv.get("steps", 0),
                "resets":
                lv.get("resets", 0),
                "skill_invocations":
                lv.get("skill_invocations", 0),
                "failed_skill_invocations":
                lv.get("failed_skill_invocations", 0),
                "game_overs":
                len(lv.get("game_overs", [])),
                "game_over_reasons":
                ";".join(reasons),
                "divergences":
                lv.get("divergences", 0),
                "wall_clock":
                lv.get("wall_clock", 0.0),
                "wall_clock_env":
                lv.get("wall_clock_env", 0.0),
                "steps_before_first_win":
                lv.get("steps_before_first_win", ""),
                "resets_before_first_win":
                lv.get("resets_before_first_win", ""),
                "preemptions":
                lv.get("preemptions", 0),
                "resumes":
                lv.get("resumes", 0),
                "downtime":
                lv.get("downtime", 0.0),
                "harness_resets":
                lv.get("harness_resets", 0),
                "interrupted_invocations":
                lv.get("interrupted_invocations", 0),
                "episodes":
                len(lv.get("episodes", [])),
                "llm_cost_usd":
                sandbox.get("llm_cost_usd", 0.0),
                "sim_rollouts":
                sandbox.get("sim_rollouts", 0.0),
                "fits":
                sandbox.get("fits", 0.0),
                "sessions":
                sandbox.get("sessions", 0.0),
            })
    return rows


def _mean(values: Sequence[float]) -> str:
    return f"{statistics.mean(values):.1f}" if values else "-"


def _column_mean(
        totals: Sequence[Dict[str, Any]]) -> Callable[[str, float], str]:
    """A mean over one totals column, scaled (e.g. seconds to hours)."""

    def col(key: str, scale: float = 1.0) -> str:
        return _mean([float(t.get(key, 0)) / scale for t in totals])

    return col


def summary_markdown(cards: Sequence[Dict[str, Any]]) -> str:
    """Per env, one table with one row per arm."""
    by_env: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for card in cards:
        by_env.setdefault(str(card.get("env")),
                          {}).setdefault(str(card.get("arm")), []).append(card)
    lines = ["# Continual protocol scorecards", ""]
    lines.append("Base metrics only; no aggregation into a score is applied.")
    lines.append("")
    for env in sorted(by_env):
        lines.append(f"## {env}")
        lines.append("")
        lines.append("| arm | runs | all levels won | mean levels won | "
                     "mean steps | mean resets | mean steps to first win "
                     "(won levels) | mean active h | mean LLM $ |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for arm in sorted(by_env[env]):
            runs = by_env[env][arm]
            totals = [c.get("totals") or {} for c in runs]
            won_levels = [
                float(lv["steps_before_first_win"]) for c in runs
                for lv in c.get("levels", [])
                if lv.get("steps_before_first_win") is not None
            ]
            all_won = sum(1 for t in totals
                          if t.get("levels_completed") == t.get("levels_total")
                          and t.get("levels_total"))

            col = _column_mean(totals)
            cells = [
                arm,
                str(len(runs)),
                str(all_won),
                f"{col('levels_completed', 1.0)} / {col('levels_total', 1.0)}",
                col("total_steps", 1.0),
                col("total_resets", 1.0),
                _mean(won_levels),
                col("total_wall_clock", 3600.0),
                col("total_llm_cost", 1.0),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_csv(path: str, columns: Sequence[str],
              rows: Sequence[Dict[str, Any]]) -> None:
    """Write ``rows`` as CSV with ``columns``."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def aggregate(scorecards: str, out: str) -> Dict[str, str]:
    """Write the three outputs; returns their paths."""
    cards = load_cards(scorecards)
    os.makedirs(out, exist_ok=True)
    paths = {
        "runs": os.path.join(out, "runs.csv"),
        "levels": os.path.join(out, "levels.csv"),
        "summary": os.path.join(out, "summary.md"),
    }
    write_csv(paths["runs"], RUN_COLUMNS, run_rows(cards))
    write_csv(paths["levels"], LEVEL_COLUMNS, level_rows(cards))
    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write(summary_markdown(cards))
    return paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0])
    parser.add_argument("--scorecards", default="scorecards")
    parser.add_argument("--out", default="analysis/scorecards")
    args = parser.parse_args()
    paths = aggregate(args.scorecards, args.out)
    with open(paths["summary"], "r", encoding="utf-8") as f:
        print(f.read())
    print(f"wrote {paths['runs']}, {paths['levels']}, {paths['summary']}")


if __name__ == "__main__":
    main()
