"""Recompute account-limit / budget-cap flags on finished runs.

The first sweep-2 runs were scored by a detector that matched the word
"budget" anywhere in the agent's closing message, which flagged runs costing a
third of the cap. This re-derives both flags from the transcript subtype and
the recorded cost, and rewrites results.json in place.

    uv run python -m agent_robot_control.experiments.repair_flags ~/arc_outputs2 --cap 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Rewrite the flags for every results.json under a root."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--cap", type=float, default=20.0)
    args = ap.parse_args()
    from agent_robot_control.harness.claude_code import parse_stream_json
    changed = 0
    for path in sorted(args.root.rglob("results.json")):
        res = json.loads(path.read_text())
        transcript = path.parent / "transcript.jsonl"
        if not transcript.exists():
            continue
        parsed = parse_stream_json(transcript)
        cost = res.get("cost_usd") or parsed.get("cost_usd") or 0.0
        budget = bool(parsed.get("budget_cap")) or cost >= 0.98 * args.cap
        account = bool(parsed.get("account_limit"))
        invalid = (budget or account) and not res.get("goal_reached_ever")
        before = (res.get("budget_cap_hit"), res.get("account_limit_hit"),
                  res.get("invalid"))
        after = (budget, account, invalid)
        if before != after:
            res.update(budget_cap_hit=budget, account_limit_hit=account,
                       invalid=invalid)
            path.write_text(json.dumps(res, indent=2, default=str))
            changed += 1
            print(f"{path.parent.relative_to(args.root)}: cost ${cost:.2f} "
                  f"{before} -> {after}")
    print(f"{changed} results.json rewritten")


if __name__ == "__main__":
    main()
