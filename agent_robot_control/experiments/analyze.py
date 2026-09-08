"""Aggregate results.json files into success-vs-interactions curves.

    uv run python -m agent_robot_control.experiments.analyze outputs/ --out analysis/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results(root: Path) -> List[dict]:
    """Every results.json under ``root``."""
    out = []
    for p in root.rglob("results.json"):
        if any(part in ("first_runs", "rl_pilot") or part.startswith("rl_pilot") for part in p.parts):
            continue  # pilots and archived smoke runs are not sweep data
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("invalid"):
            continue  # truncated by the account usage limit
        if "env" in r and "condition" in r:
            r["_path"] = str(p)
            # Run dirs are <env><tier>/<condition>/<harness>/seed_k; the dir
            # name carries the clearance tier that the env name lacks.
            r["env"] = p.parents[3].name if len(p.parents) > 3 else r["env"]
            out.append(r)
    return out


def success_curve(runs: List[dict], cap: int, num_points: int = 200):
    """Fraction of runs with first success <= t, for t on a grid."""
    grid = np.linspace(0, cap, num_points)
    firsts = [r.get("first_success_interaction") for r in runs]
    ys = [np.mean([f is not None and f <= t for f in firsts]) for t in grid]
    return grid, np.array(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    runs = load_results(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["env"], r["harness"], r["condition"])].append(r)
    lines = ["| env | harness | condition | runs | success | median first success | "
             "mean interactions | mean tool calls | RL calls | mean wall (min) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (env, harness, cond), rs in sorted(groups.items()):
        succ = [r for r in rs if r.get("goal_reached_ever")]
        firsts = [r["first_success_interaction"] for r in succ]
        rl_calls = sum(sum(v for k, v in (r.get("tool_call_counts") or {}).items() if "rl" in k) for r in rs)
        lines.append(f"| {env} | {harness} | {cond} | {len(rs)} | {len(succ)}/{len(rs)} | "
                     f"{np.median(firsts) if firsts else float('nan'):.0f} | "
                     f"{np.mean([r.get('interactions_used', 0) for r in rs]):.0f} | "
                     f"{np.mean([r.get('tool_calls', 0) for r in rs]):.1f} | {rl_calls} | "
                     f"{np.mean([r.get('wall_time_s', 0) for r in rs])/60:.0f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        envs = sorted({k[0] for k in groups})
        fig, axes = plt.subplots(1, max(1, len(envs)), figsize=(5 * max(1, len(envs)), 4), squeeze=False)
        for ax, env in zip(axes[0], envs):
            for (e, harness, cond), rs in sorted(groups.items()):
                if e != env:
                    continue
                cap = int(rs[0].get("interaction_cap", 100000))
                x, y = success_curve(rs, cap)
                ax.plot(x, y, label=f"{harness}/{cond} (n={len(rs)})")
            ax.set_title(env); ax.set_xlabel("env interactions"); ax.set_ylabel("success rate")
            ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(args.out / "success_vs_interactions.png", dpi=120)
        print("wrote", args.out / "success_vs_interactions.png")
    except Exception as e:  # pylint: disable=broad-except
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
