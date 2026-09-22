"""Audit selected benchmark runtimes after logged account-limit waits.

Usage:
    python scripts/plotting/audit_benchmark_runtime.py

The scorecard excludes gaps between saved checkpoints and resumed processes.
It can also omit time after its last clock flush. This script subtracts only
the portion of logged sleeps overlapping recorded scorecard time. Refused-
query latency is not separately recorded and remains in the adjusted figure.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (ROOT /
                    "docs/comparisons/figures/paper-results-opus-summary.json")
WAIT_RE = re.compile(r"\b(?:resuming|retrying) in ([0-9]+(?:\.[0-9]+)?) s\b")
RESET_RE = re.compile(r"resets (\d{1,2}(?::\d{2})?(?:am|pm)) "
                      r"\(America/New_York\); stated reset in (\d+) s")
EASTERN = ZoneInfo("America/New_York")


def _run_dir(source: str) -> Path:
    """Resolve the archive's cluster path in the current checkout."""
    marker = "/predicators/logs/"
    if marker not in source:
        raise ValueError(f"Unexpected run source: {source}")
    return ROOT / "logs" / source.split(marker, 1)[1]


def _overlap(left: tuple, right: tuple) -> float:
    """Seconds shared by two epoch-time intervals."""
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _resume_gaps(run_dir: Path) -> List[tuple]:
    """Recover downtime intervals from the recording's cumulative totals."""
    gaps = []
    for index_path in sorted(run_dir.glob("L*/index.jsonl")):
        last_downtime = 0.0
        with index_path.open(encoding="utf-8") as index:
            for line in index:
                event = json.loads(line)
                if event.get("event") != "resume":
                    continue
                new_downtime = float(event["downtime"])
                delta = new_downtime - last_downtime
                if delta < 0:
                    raise ValueError(f"Decreasing downtime: {index_path}")
                gaps.append((float(event["t"]) - delta, float(event["t"])))
                last_downtime = new_downtime
    return gaps


def _warning_time(line: str, started: float, finished: float,
                  previous: float) -> float:
    """Date a limit warning using its reset clock and seconds remaining."""
    reset = RESET_RE.search(line)
    if reset is None:
        raise ValueError(f"Cannot date retry warning: {line}")
    clock = reset.group(1)
    fmt = "%I:%M%p" if ":" in clock else "%I%p"
    reset_time = datetime.strptime(clock, fmt).time()
    first_day = datetime.fromtimestamp(started, EASTERN).date()
    last_day = datetime.fromtimestamp(finished, EASTERN).date()
    candidates = []
    for offset in range(-1, (last_day - first_day).days + 3):
        day = first_day + timedelta(days=offset)
        reset_at = datetime.combine(day, reset_time, EASTERN).timestamp()
        warned_at = reset_at - float(reset.group(2))
        if max(started - 60, previous - 2) <= warned_at <= finished + 60:
            candidates.append(warned_at)
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous warning time ({candidates}): {line}")
    return candidates[0]


def _retry_waits(log_path: Path, started: float,
                 finished: float) -> List[tuple]:
    """Read sleep intervals in log order, including interrupted attempts."""
    intervals = []
    previous = started - 60
    with log_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            if "The attempt's wall-clock budget is paused meanwhile." \
                    not in line:
                continue
            if "usage limit" not in line:
                raise ValueError(f"Unknown retry type in {log_path}: {line}")
            wait = WAIT_RE.search(line)
            if wait is None:
                raise ValueError(f"Retry wait without duration in "
                                 f"{log_path}: {line}")
            try:
                warned_at = _warning_time(line, started, finished, previous)
            except ValueError as exc:
                raise ValueError(f"{log_path}: {exc}") from exc
            intervals.append((warned_at, warned_at + float(wait.group(1))))
            previous = warned_at
    return intervals


def audit(manifest: Path, arms: List[str]) -> List[dict]:
    """Return per-run scorecard time and logged retry sleeps."""
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    rows = []
    for record in records:
        if arms and record["arm"] not in arms:
            continue
        run_dir = _run_dir(record["source"])
        card = json.loads(
            (run_dir / "scorecard.json").read_text(encoding="utf-8"))
        run_started = datetime.strptime(
            run_dir.name,
            "run_%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        intervals = _retry_waits(run_dir / "info.log",
                                 min(run_started, float(card["started_at"])),
                                 float(card["finished_at"]))
        levels = []
        for lv in card["levels"]:
            if lv["started_at"] is None:
                continue
            start = float(lv["started_at"])
            # A terminal scorecard may retain the last flush's clock rather
            # than all elapsed time through finished_at. Resume downtime is
            # already excluded from wall_clock.
            counted_end = (start + float(lv["wall_clock"]) +
                           float(lv["downtime"]))
            if counted_end > float(card["finished_at"]) + 2:
                raise ValueError(f"Scorecard clock past run end: {run_dir}")
            levels.append((start, counted_end))
        gaps = _resume_gaps(run_dir)
        raw = float(card["totals"]["total_wall_clock"])
        logged = sum(stop - start for start, stop in intervals)
        paused = sum(
            sum(_overlap(wait, level) for level in levels) -
            sum(_overlap(wait, gap) for gap in gaps) for wait in intervals)
        if paused < -1 or paused > raw + 1:
            raise ValueError(f"Invalid charged wait {paused:.1f}s: {run_dir}")
        rows.append({
            "arm": record["arm"],
            "domain": record["domain"],
            "seed": record["seed"],
            "raw_seconds": raw,
            "logged_wait_seconds": logged,
            "charged_wait_seconds": paused,
            "retry_events": len(intervals),
            "adjusted_seconds": raw - paused,
        })
    return rows


def main() -> None:
    """Print medians, wait totals, and affected-run counts by arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--arms",
                        nargs="*",
                        default=[],
                        help="arm IDs to audit")
    parser.add_argument("--json",
                        action="store_true",
                        help="print per-run JSON")
    args = parser.parse_args()
    rows = audit(args.manifest, args.arms)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    print("arm\tn\traw_median_min\tadjusted_median_min\tlogged_wait_h\t"
          "charged_wait_h\taffected_runs")
    for arm, group in sorted(by_arm.items()):
        raw = statistics.median(r["raw_seconds"] for r in group) / 60
        adjusted = statistics.median(r["adjusted_seconds"] for r in group) / 60
        logged = sum(r["logged_wait_seconds"] for r in group) / 3600
        charged = sum(r["charged_wait_seconds"] for r in group) / 3600
        affected = sum(r["charged_wait_seconds"] > 0 for r in group)
        print(f"{arm}\t{len(group)}\t{raw:.2f}\t{adjusted:.2f}\t"
              f"{logged:.2f}\t{charged:.2f}\t{affected}")


if __name__ == "__main__":
    main()
