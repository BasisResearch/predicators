"""Capture and report the paired bridge, fan and balloons noise runs.

Use --refresh to read live scorecards, or omit it to reproduce the saved
snapshot. This script never launches, resumes or modifies experiment
runs.
"""
import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "logs/uncertainty_crossdomain_20260908/launch-manifest.json"
FLAGS = (
    "code_sim_learning_interval_belief",
    "agent_explorer_info_seeking_noise_aware",
    "code_sim_learning_rollout_noise_filter",
    "code_sim_learning_carry_posterior",
    "code_sim_learning_fit_evidence",
    "continual_belief_frame",
)


def capture(manifest_path, root):
    """Read only active manifest entries, preserving missing seed records."""
    plan = json.loads(manifest_path.read_text())
    rows = []
    for run in plan["runs"]:
        flags = run["flags"]
        switches = [flags[key] for key in FLAGS]
        if any(switches) != all(switches):
            raise ValueError(
                f"Mixed uncertainty flags: {run['experiment_id']}")
        if run["approach"] == "agent_continual_model_free":
            if any(switches):
                raise ValueError("MF must have all uncertainty flags off")
            arm = "MF"
        elif run["approach"] == "agent_continual":
            arm = "MB with features" if all(
                switches) else "MB without features"
        else:
            raise ValueError(f"Unexpected approach: {run['approach']}")
        if flags.get("agent_model_repair", False):
            arm += " + repair"
        domain = run["env"].removeprefix("pybullet_").title()
        expected_levels = flags["num_train_tasks"] + flags["num_test_tasks"]
        runs = []
        seeds = plan.get(
            "seeds",
            range(run.get("start_seed", 0),
                  run.get("start_seed", 0) + run.get("num_seeds", 2)))
        for seed in seeds:
            directory = (root / "logs" / run["approach"] /
                         run["experiment_id"] / f"seed{seed}")
            paths = sorted(directory.glob("run_*/scorecard.json"))
            if len(paths) > 1:
                raise ValueError(f"Ambiguous seed: {directory}: {paths}")
            record = {"seed": seed, "finished_at": None, "status": "Missing"}
            if paths:
                card = json.loads(paths[0].read_text())
                totals = card["totals"]
                if (card["seed"] != seed or card["env"] != run["env"]
                        or card["arm"] != run["approach"]
                        or not plan["base_sha"].startswith(card["git_sha"])
                        or totals["levels_total"] != expected_levels):
                    raise ValueError(
                        f"Scorecard identity mismatch: {paths[0]}")
                for suffix in ("position", "orientation", "scalar",
                               "declared"):
                    key = f"obs_noise_{suffix}"
                    if card.get(key) != flags[f"continual_{key}"]:
                        raise ValueError(f"Noise mismatch: {paths[0]}: {key}")
                for key, field in (("levels_completed", "won"),
                                   ("total_steps", "steps"), ("total_resets",
                                                              "resets")):
                    if totals[key] != sum(lv[field] for lv in card["levels"]):
                        raise ValueError(
                            f"Inconsistent totals: {paths[0]}: {key}")
                finished = card.get("finished_at")
                record.update({
                    "source": str(paths[0].relative_to(root)),
                    "source_git_sha": card["git_sha"],
                    "finished_at": finished,
                    "status": "Finished" if finished else "Unfinished",
                    "end_reason": card.get("end_reason"),
                    "levels_solved": totals["levels_completed"],
                    "levels_total": totals["levels_total"],
                    "steps": totals["total_steps"],
                    "resets": totals["total_resets"],
                })
            runs.append(record)
        rows.append({
            "domain":
            domain,
            "noise": (f"{1000 * flags['continual_obs_noise_position']:g} mm / "
                      f"{flags['continual_obs_noise_orientation']:g} rad" +
                      (f" / scalar {flags['continual_obs_noise_scalar']:g}"
                       if flags["continual_obs_noise_scalar"] else "")),
            "arm":
            arm,
            "experiment_id":
            run["experiment_id"],
            "job_id":
            run["job_id"],
            "uncertainty_flags":
            dict(zip(FLAGS, switches)),
            "runs":
            runs,
        })
    domains = {"Bridge": 0, "Fan": 1, "Balloons": 2, "Domino": 3, "Boil": 4}
    arms = {
        "MF": 0,
        "MB without features": 1,
        "MB with features": 2,
        "MB with features + repair": 3
    }
    rows.sort(key=lambda row: (domains[row["domain"]], arms[row["arm"]]))
    return {
        "generated_by": "make_crossdomain_table.py; do not edit manually",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "base_sha": plan["base_sha"],
        "source_tree": plan["tree"],
        "rows": rows,
    }


def aggregate(row):
    """Withhold averages until all seeds finish; condition steps on success."""
    runs = row["runs"]
    finished = [run for run in runs if run["finished_at"]]
    successful = [
        run for run in finished if run["levels_solved"] == run["levels_total"]
    ]
    complete = len(finished) == len(runs)
    return {
        "domain":
        row["domain"],
        "noise":
        row["noise"],
        "arm":
        row["arm"],
        "seeds":
        len(runs),
        "finished_seeds":
        len(finished),
        "successful_seeds":
        len(successful),
        "solve_rate_pct":
        (100 * mean(run["levels_solved"] / run["levels_total"]
                    for run in runs) if complete else None),
        "mean_steps_successful":
        (mean(run["steps"]
              for run in successful) if complete and successful else None),
        "mean_resets": (mean(run["resets"]
                             for run in runs) if complete else None),
        "status":
        "Complete" if complete else "Incomplete",
    }


def number(value):
    """Show undefined successful-seed cost as N/A, never as zero."""
    return "N/A" if value is None else f"{value:,.1f}".removesuffix(".0")


def atomic_write(path, contents):
    """Replace a report atomically so preemption cannot leave a partial
    file."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     delete=False) as stream:
        stream.write(contents)
        temp_path = Path(stream.name)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_report(payload, output):
    """Write the aggregate table and auditable per-seed details."""
    summaries = [aggregate(row) for row in payload["rows"]]
    finished = sum(row["finished_seeds"] for row in summaries)
    expected = sum(row["seeds"] for row in summaries)
    lines = [
        "# Bridge, fan and balloons noise performance",
        "",
        "Generated by `make_crossdomain_table.py`; do not edit manually.",
        f"Scorecards captured at {payload['observed_at']}.",
        f"Status: {finished}/{expected} seeds finished.",
        "",
        "Solve rate averages the fraction of train and test levels won over all seeds.",
        "Mean resets averages whole-run reset counts over all seeds.",
        "Mean steps uses only finished seeds that won every level, including all attempts and resets within those runs.",
        "Each arm uses seeds 0 and 1; bridge and fan have two levels per seed, and balloons has three.",
        "Incomplete rows withhold all averages until both seeds finish.",
        "N/A in a completed row means there were no successful seeds.",
        "Noise lists position and orientation standard deviations; scalar-reading noise is zero.",
        "",
        "| Domain | Noise | Arm | Finished seeds | Successful seeds | Solve rate | Mean steps (successful seeds) | Mean resets (all seeds) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        complete = row["finished_seeds"] == row["seeds"]
        solve = f"{number(row['solve_rate_pct'])}%" if complete else "Pending"
        steps = number(row["mean_steps_successful"]) if complete else "Pending"
        resets = number(row["mean_resets"]) if complete else "Pending"
        lines.append(
            f"| {row['domain']} | {row['noise']} | {row['arm']} | "
            f"{row['finished_seeds']}/{row['seeds']} | {row['successful_seeds']} | "
            f"{solve} | {steps} | {resets} |")
    lines.extend([
        "",
        "Steps are conditional on success and should be interpreted alongside solve rate.",
        "These are exploratory comparisons with two seeds per arm.",
        f"All noisy arms use frozen base `{payload['base_sha']}` plus the validated runtime patch described in [the plan](crossdomain-plan.md).",
        "The paired MB arms differ only in the six uncertainty switches listed in the plan.",
        "Existing noiseless references remain separate because their code versions differ.",
        "",
        "## Seed records",
        "",
        "Unfinished step and reset counts below are provisional and do not enter the averages.",
        "",
        "| Domain | Arm | Seed | Status / end reason | Levels won | Steps | Resets | Slurm task | Scorecard |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ])
    for row in payload["rows"]:
        for run in row["runs"]:
            status = run["end_reason"] if run["finished_at"] else run["status"]
            levels = (f"{run['levels_solved']}/{run['levels_total']}"
                      if "source" in run else "Pending")
            source = f"[JSON](../../{run['source']})" if "source" in run else "N/A"
            lines.append(
                f"| {row['domain']} | {row['arm']} | {run['seed']} | {status} | "
                f"{levels} | {number(run.get('steps'))} | {number(run.get('resets'))} | "
                f"{row['job_id']}_{run['seed']} | {source} |")
    lines.extend([
        "",
        "Data: [TSV](crossdomain-summary.tsv), [snapshot JSON](crossdomain-snapshot.json).",
        "",
        "Reproduce this saved table from the repository root:",
        "",
        "```bash",
        "python docs/uncertainty-results/make_crossdomain_table.py",
        "```",
        "",
        "Add `--refresh` to capture current scorecards, and `--require-finished` to fail the collection job if any seed remains missing or unfinished.",
    ])
    atomic_write(output / "crossdomain-table.md", "\n".join(lines) + "\n")
    # Use the CSV writer to preserve a machine-readable schema and LF endings.
    with tempfile.TemporaryFile(mode="w+") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=list(summaries[0]),
                                delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
        stream.seek(0)
        atomic_write(output / "crossdomain-summary.tsv", stream.read())
    return finished == expected


def main():
    """Refresh explicitly; otherwise reproduce the saved snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output",
                        type=Path,
                        default=ROOT / "docs/uncertainty-results")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--require-finished", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "crossdomain-snapshot.json"
    if args.refresh:
        payload = capture(args.manifest, args.root)
        atomic_write(snapshot, json.dumps(payload, indent=2) + "\n")
    else:
        payload = json.loads(snapshot.read_text())
    complete = write_report(payload, args.output)
    print(f"Wrote {args.output / 'crossdomain-table.md'}; complete={complete}")
    if args.require_finished and not complete:
        raise SystemExit(
            "Missing or unfinished seeds; see the incomplete table.")


if __name__ == "__main__":
    main()
