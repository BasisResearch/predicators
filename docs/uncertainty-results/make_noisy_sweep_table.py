"""Refresh the frozen five-domain noisy sweep, including explicit reused
seeds."""
import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "logs/noisy_sweep_20260910/launch-manifest.json"
TERMINAL_REASONS = {
    "all_levels_won", "level_lost", "level_not_won", "agent_ended", "step_cap",
    "wall_clock_cap"
}
DOMAINS = ("Bridge", "Fan", "Domino", "Boil", "Balloons")


def capture(manifest):
    """Validate scorecard identity and keep every expected seed visible."""
    rows = []
    for run in manifest["runs"]:
        for seed in run["seeds"]:
            directory = (ROOT / "logs" / run["approach"] /
                         run["experiment_id"] / f"seed{seed}")
            paths = list(directory.glob("run_*/scorecard.json"))
            if len(paths) > 1:
                raise ValueError(f"Ambiguous scorecards: {directory}")
            rows.append(
                read_card(run["env"], run["arm"], seed,
                          paths[0] if paths else None, run["source_commit"],
                          run["flags"], f"{run['job_id']}_{seed}"))
    balloons = next(r for r in manifest["runs"]
                    if r["env"] == "pybullet_balloons" and r["arm"] == "MB")
    for reused in manifest["reused"]:
        row = read_card(reused["env"], reused["arm"], reused["seed"],
                        Path(reused["scorecard"]), reused["source_commit"],
                        balloons["flags"], None)
        row["reused"] = True
        rows.append(row)
    rows.sort(key=lambda r: (DOMAINS.index(r["domain"]), r["arm"], r["seed"]))
    if len(rows) != 30 or len({(r["domain"], r["arm"], r["seed"])
                               for r in rows}) != 30:
        raise ValueError(
            "Expected exactly three distinct seeds per domain/arm")
    return rows


def read_card(env, arm, seed, path, source, flags, job):
    """An unfinished or unrecognized outcome never enters agent averages."""
    row = {
        "domain": env.removeprefix("pybullet_").title(),
        "arm": arm,
        "seed": seed,
        "source_commit": source,
        "job": job,
        "reused": False,
        "status": "Not started",
        "finished": False,
        "wins": None,
        "levels": flags["num_train_tasks"] + flags["num_test_tasks"],
        "steps": None,
        "resets": None,
        "success": False,
        "scorecard": str(path) if path else None
    }
    if path is None:
        return row
    card = json.loads(path.read_text())
    if (card["seed"] != seed or card["env"] != env or card["arm"] !=
        ("agent_continual" if arm == "MB" else "agent_continual_model_free")
            or not source.startswith(card["git_sha"])):
        raise ValueError(f"Scorecard identity mismatch: {path}")
    for suffix in ("position", "orientation", "scalar", "declared"):
        key = f"obs_noise_{suffix}"
        if card[key] != flags[f"continual_{key}"]:
            raise ValueError(f"Noise mismatch: {path}: {key}")
    totals = card["totals"]
    if totals["levels_total"] != row["levels"]:
        raise ValueError(f"Level count mismatch: {path}")
    for total, field in (("total_steps", "steps"), ("total_resets", "resets"),
                         ("levels_completed", "won")):
        if totals[total] != sum(level[field] for level in card["levels"]):
            raise ValueError(f"Inconsistent totals: {path}: {total}")
    finished = bool(card.get("finished_at"))
    if finished and card.get("end_reason") not in TERMINAL_REASONS:
        row["status"] = "Outcome needs review"
        return row
    row.update(finished=finished,
               status="Finished" if finished else "In progress",
               wins=totals["levels_completed"],
               steps=totals["total_steps"],
               resets=totals["total_resets"],
               end_reason=card.get("end_reason"),
               success=finished
               and totals["levels_completed"] == totals["levels_total"])
    return row


def aggregate(rows):
    """Use finished agent seeds for solve/reset, whole-run successes for
    steps."""
    groups = []
    for domain in DOMAINS:
        for arm in ("MB", "MF"):
            subset = [
                r for r in rows if r["domain"] == domain and r["arm"] == arm
            ]
            done = [r for r in subset if r["finished"]]
            wins = [r for r in done if r["success"]]
            groups.append({
                "domain":
                domain,
                "arm":
                arm,
                "finished":
                len(done),
                "expected":
                len(subset),
                "successful":
                len(wins),
                "whole_run_solve_pct":
                100 * len(wins) / len(done) if done else None,
                "level_solve_pct":
                100 * sum(r["wins"] for r in done) /
                sum(r["levels"] for r in done) if done else None,
                "mean_steps_successful":
                mean(r["steps"] for r in wins) if wins else None,
                "steps_n":
                len(wins),
                "mean_resets_finished":
                mean(r["resets"] for r in done) if done else None
            })
    return groups


def refresh(manifest_path=MANIFEST):
    """Write reproducible JSON, TSV and a Markdown result table."""
    manifest = json.loads(manifest_path.read_text())
    rows = capture(manifest)
    groups = aggregate(rows)
    stamp = datetime.now(timezone.utc).isoformat()
    output = Path(__file__).resolve().parent
    snapshot = {
        "generated_by": "make_noisy_sweep_table.py; do not edit manually",
        "updated_at": stamp,
        "manifest": str(manifest_path),
        "source_commit": manifest["source_commit"],
        "rows": rows,
        "aggregate": groups
    }
    (output / "noisy-sweep-snapshot.json"
     ).write_text(json.dumps(snapshot, indent=2) + "\n")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer,
                            fieldnames=list(groups[0]),
                            delimiter="\t",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows({
        key: "NA" if value is None else value
        for key, value in group.items()
    } for group in groups)
    (output / "noisy-sweep-summary.tsv").write_text(buffer.getvalue())
    lines = [
        "<!-- Generated by make_noisy_sweep_table.py; do not edit manually. -->",
        "# Five-domain noisy sweep", "", f"Updated {stamp}.", "",
        "All aggregates are provisional until all three seeds finish.",
        "Solve rates and mean resets use finished agent runs, with their count shown.",
        "Mean steps uses only whole-run successful seeds, with qualifying n.",
        "Missing, unfinished, and infrastructure outcomes do not enter these averages.",
        "Balloons MB seeds 0 and 1 are reused from the identical agent runtime and flags; all MF seeds are fresh.",
        "",
        "| Domain | Arm | Finished | Whole-run wins | Level solve rate | Steps, successes only | Resets, finished seeds |",
        "|---|---|---:|---:|---:|---:|---:|"
    ]
    for group in groups:

        def number(value):
            return "-" if value is None else f"{value:g}"

        rate = "-" if group[
            "level_solve_pct"] is None else f"{group['level_solve_pct']:.1f}%"
        lines.append(
            f"| {group['domain']} | {group['arm']} | {group['finished']}/3 | "
            f"{group['successful']}/{group['finished']} | {rate} | "
            f"{number(group['mean_steps_successful'])} (n={group['steps_n']}) | "
            f"{number(group['mean_resets_finished'])} |")
    lines += [
        "",
        "| Domain | Arm | Seed | Status | Wins | Steps | Resets | Source |",
        "|---|---|---:|---|---:|---:|---:|---|"
    ]
    for row in rows:
        win = "-" if row["wins"] is None else f"{row['wins']}/{row['levels']}"
        source = f"[scorecard]({row['scorecard']})" if row["scorecard"] else "-"
        if row["reused"]:
            source += " (reused)"
        lines.append(
            f"| {row['domain']} | {row['arm']} | {row['seed']} | {row['status']} | "
            f"{win} | {row['steps'] if row['steps'] is not None else '-'} | "
            f"{row['resets'] if row['resets'] is not None else '-'} | {source} |"
        )
    (output / "noisy-sweep-table.md").write_text("\n".join(lines) + "\n")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    refresh(parser.parse_args().manifest)
