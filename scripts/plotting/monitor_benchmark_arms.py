"""Refresh the benchmark figure and report when finished scorecards change."""

import argparse
import datetime
import fcntl
import json
import os
import re
import runpy
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/comparisons/ten-agent-opus-benchmark.md"
FIGURES = REPORT.parent / "figures"
STATE = ROOT / "logs/benchmark_monitor"
STEM = "benchmark-arms-opus"
DOMAIN_NAMES = {
    "Boil (2-jug)": "Boil (two-jug)",
    "Domino": "Domino (high-friction turn)",
    "Bridge (4-span)": "Bridge (four-span)",
}
Row = Dict[str, Any]


def timestamp() -> str:
    """Return an unambiguous timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    """Publish a complete generated file on the same filesystem."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)


def number(value: float) -> str:
    """Format a count or mean to at most one decimal place."""
    return f"{value:,.1f}".rstrip("0").rstrip(".")


def report_text(original: str, rows: List[Row], plot: Dict[str, Any],
                stamp: str) -> str:
    """Preserve methodology and settings while regenerating result tables."""
    labels = {
        arm: f"{i + 1}. {label}"
        for i, (arm, label) in enumerate(zip(plot["ARMS"], plot["LABELS"]))
    }
    averages = []
    details = []
    unfinished = []
    direct_pending = []
    for domain, directories in plot["DOMAINS"]:
        title = DOMAIN_NAMES.get(domain, domain)
        details.extend([
            f"### {title}", "",
            "| Approach | Seed | Levels won | Steps | Resets | "
            "Log directory | Scorecard |", "|---|---:|---:|---:|---:|---|---|"
        ])
        for arm in plot["ARMS"]:
            group = [
                r for r in rows if r["domain"] == domain and r["arm"] == arm
            ]
            done = {r["seed"] for r in group}
            for seed, directory in enumerate(directories[arm]):
                if seed in done:
                    continue
                run = plot["pick"](directory)
                link = f"[run logs]({run})" if run else "No run directory"
                unfinished.append(
                    f"| {title} | {labels[arm]} | {seed} | {link} |")
                if arm == "mf_scene_package":
                    direct_pending.append(f"{title} seed {seed}")
            if not group:
                continue
            successes = [r for r in group if r["won"] == r["levels"]]
            count = len(group)
            won = sum(r["won"] for r in group)
            levels = sum(r["levels"] for r in group)
            steps = (number(
                sum(r["steps"] for r in successes) /
                len(successes)) if successes else "-")
            resets = number(sum(r["resets"] for r in group) / count)
            averages.append(
                f"| {title} | {labels[arm]} | {len(successes)}/{count} "
                f"({number(100 * len(successes) / count)}%) | {won}/{levels} "
                f"({number(100 * won / levels)}%) | {steps} "
                f"(n={len(successes)}) | {resets} | {count}/3 |")
            for row in sorted(group, key=lambda r: r["seed"]):
                source = row["source"]
                details.append(
                    f"| {labels[arm]} | {row['seed']} | "
                    f"{row['won']}/{row['levels']} | {row['steps']:,} | "
                    f"{row['resets']} | [run logs]({source}) "
                    f"| [scorecard]({source}/scorecard.json) |")
        details.append("")
    prefix = original.split("## Averages across seeds", 1)[0]
    prefix = re.sub(r"^# Opus benchmark sweep:.*\n",
                    "# Opus benchmark sweep: eleven agents and r2 cohorts\n",
                    prefix,
                    count=1)
    empiric_note = (
        "EMPIRIC r2 uses seeds 3 and 4 across all five domains "
        "on the repaired runtime.\n"
        "Legacy blocking preflight is off; bounded shadow validation "
        "logs predictions "
        "and outcomes without refusing actions.\n"
        "It is a separate prospective cohort, "
        "not a matched preflight ablation.\n\n")
    if "EMPIRIC r2 uses seeds 3 and 4" not in prefix:
        prefix += empiric_note
    prefix = prefix.replace(
        "All agents are Claude Opus with the composite skill library "
        "and skill preflight off.",
        "All agents are Claude Opus with the composite skill library.\n"
        "Preflight settings differ across historical cohorts: the selected "
        "EMPIRIC Boil, Domino, and Balloons runs and Fan seeds 1-2 had "
        "preflight enabled; the selected Bridge reruns and newer comparison "
        "arms used preflight off.\n"
        "This is not a matched preflight ablation.")
    validation_link = "[Oracle repair pilot](oracle-dynamics-validation-r2.md)"
    pilot_note = (
        "Oracle dynamics r2 is a separate entry directly below "
        "Oracle dynamics, in dark grey.\n"
        "Only finished Domino and Bridge r2 runs contribute; "
        "the other domains are blank because they were not launched.\n"
        "The " + validation_link + " also shows the two Oracle "
        "rounds side by side.\n\n")
    if validation_link in prefix:
        prefix = re.sub(
            r"(?:Oracle dynamics r2 is a separate entry.*?\n)?"
            r"(?:Only finished Domino and Bridge r2 runs.*?\n)?"
            r"The \[Oracle repair pilot\].*?\n\n",
            lambda _: pilot_note,
            prefix,
            count=1)
    else:
        prefix += pilot_note
    prefix = re.sub(
        r"^Compiled .*?$",
        f"Compiled {stamp} from {len(rows)} finished scorecards on "
        "the five benchmark settings fixed on September 18.",
        prefix,
        count=1,
        flags=re.MULTILINE)
    direct_done = sum(r["arm"] == "mf_scene_package" for r in rows)
    pending = "; ".join(direct_pending) or "none"
    no_unc_done = any(r["arm"] == "no_uncertainty" and r["seed"] == 2
                      and r["domain"] == "Balloons (composition)"
                      for r in rows)
    no_unc_status = "finished" if no_unc_done else "unfinished"
    oracle_r2_done = sum(r["arm"] == "oracle_dynamics_r2" for r in rows)
    empiric_r2_done = sum(r["arm"] == "MB_r2" for r in rows)
    status = (
        "## Status at this snapshot\n\n"
        f"- Oracle dynamics r2: {oracle_r2_done}/6 seeds finished "
        "across Domino and Bridge.\n"
        f"- EMPIRIC r2: {empiric_r2_done}/10 seeds finished (seeds 3 and 4).\n"
        f"- Direct agent + scene assets: {direct_done}/15 seeds finished.\n"
        f"  Unfinished: {pending}.\n"
        "- EMPIRIC + scene package: four unfinished runs remain paused "
        "at the user's request "
        "(Balloons seed 0, Boil seeds 0 and 2, Domino seed 0).\n"
        f"- No explicit uncertainty Balloons seed 2: {no_unc_status}.\n"
        "- Other unfinished seeds are listed below; an unfinished scorecard "
        "does not establish whether a job is running.\n"
        "  Zero-shot remains paused; the monitor does not launch, resume "
        "or cancel experiments.\n\n"
        "The local monitor checks every minute and updates this report "
        "and its figure when finished results change.\n"
        "Its heartbeat and update log are in `logs/benchmark_monitor/`.\n\n")
    prefix = re.sub(
        r"## Status at this snapshot\n.*?(?=## Figure and plotting method)",
        lambda _: status,
        prefix,
        count=1,
        flags=re.DOTALL)
    average_intro = original.split("## Averages across seeds",
                                   1)[1].split("| Domain |", 1)[0]
    return (
        prefix + "## Averages across seeds" + average_intro +
        "| Domain | Approach | Whole-run successes | Levels won | "
        "Mean successful-run steps (n) | Mean resets, all finished seeds | "
        "Finished seeds |\n"
        "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(averages) +
        "\n\n## Per-seed results and log directories\n\n" +
        "\n".join(details) + "\n## Unfinished runs\n\n"
        "These seeds have no finished scorecard at this snapshot "
        "and are excluded above.\n\n"
        "| Domain | Approach | Seed | Latest run |\n|---|---|---:|---|\n" +
        "\n".join(unfinished) + "\n")


def refresh(plot: Dict[str, Any], force: bool = False) -> None:
    """Render from one snapshot; retry partial scorecard writes next poll."""
    rows = plot["records"]()
    summary = FIGURES / f"{STEM}-summary.json"
    previous = json.loads(
        summary.read_text())["records"] if summary.exists() else []
    stamp = timestamp()
    if force or rows != previous:
        original = REPORT.read_text()
        updated = report_text(original, rows, plot, stamp)
        with tempfile.TemporaryDirectory(prefix="benchmark-",
                                         dir=FIGURES) as folder:
            output = Path(folder) / STEM
            plot["render"](rows, str(output))
            # Re-read before publishing so concurrent report edits are retained.
            if REPORT.read_text() != original:
                raise RuntimeError(
                    "Report changed during rendering; retrying next poll")
            for suffix in (".png", ".pdf"):
                Path(str(output) + suffix).replace(FIGURES / (STEM + suffix))
            atomic_text(REPORT, updated)
            # Summary is the completion marker: publish it last.
            Path(str(output) + "-summary.json").replace(summary)
        old_sources = {r["source"] for r in previous}
        event = {
            "time": stamp,
            "finished": len(rows),
            "new_runs": [r for r in rows if r["source"] not in old_sources]
        }
        with (STATE / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
    atomic_text(
        STATE / "status.json",
        json.dumps(
            {
                "pid": os.getpid(),
                "checked_at": stamp,
                "finished": len(rows),
                "status": "watching",
                "report": str(REPORT)
            },
            indent=2) + "\n")


def refresh_oracle_validation(force: bool = False) -> None:
    """Keep the repair pilot separate from the historical main comparison."""
    loaded = runpy.run_path(
        str(ROOT / "scripts/plotting/plot_benchmark_arms.py"))
    plot = loaded["records"].__globals__
    domains = []
    for title, key in (("Domino", "domino_high_friction_turn"),
                       ("Bridge (4-span)", "bridge")):
        domains.append((title, {
            arm: [
                f"agent_continual_oracle_dynamics/{key}-"
                f"oracle_dynamics_opus_benchmark_{round_id}/seed{s}"
                for s in range(3)
            ]
            for arm, round_id in (("oracle_old", "r1"), ("oracle_fixed", "r2"))
        }))
    plot.update(DOMAINS=domains,
                ARMS=["oracle_old", "oracle_fixed"],
                LABELS=["Oracle dynamics", "Oracle dynamics r2"],
                COLORS=["#88929d", "#52616b"],
                GREYED=set(),
                GROUPS=[("Oracle", 0, 2)])
    rows = plot["records"]()
    stem = FIGURES / "oracle-dynamics-validation-r2"
    summary = Path(str(stem) + "-summary.json")
    previous = json.loads(summary.read_text(
        encoding="utf-8"))["records"] if summary.exists() else None
    if not force and rows == previous:
        return
    plot["render"](rows, str(stem))
    new_count = sum(r["arm"] == "oracle_fixed" for r in rows)
    body = [
        "# Oracle dynamics repair pilot", "",
        "<!-- Generated by monitor_benchmark_arms.py; do not edit. -->", "",
        f"Updated {timestamp()}; {new_count}/6 repaired-run seeds finished.",
        "Only Domino and Bridge are launched, "
        "with seeds 0-2 and preflight off.",
        "The other domains await review of this pilot.",
        "Only finished runs contribute; unfinished seeds are not failures.",
        "", "![Original and repaired Oracle]"
        "(figures/oracle-dynamics-validation-r2.png)", "",
        "The bar and curve denominators are the finished seeds of each "
        "cohort, not always three.",
        "Shared harness repairs change validation, and Bridge also has a "
        "repaired observation-memory model.",
        "These are diagnostic reruns, not yet a matched final-paper "
        "comparison.", "", "## Finished recordings", ""
    ]
    body.extend(f"- {r['domain']}, {r['arm']}, seed {r['seed']}: "
                f"[scorecard]({r['source']}/scorecard.json)." for r in rows)
    atomic_text(REPORT.parent / "oracle-dynamics-validation-r2.md",
                "\n".join(body) + "\n")


def main() -> None:
    """Run once or keep watching under an exclusive process lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plot = runpy.run_path(
            str(ROOT / "scripts/plotting/plot_benchmark_arms.py"))
        force = args.force
        while True:
            try:
                refresh(plot, force=force)
                refresh_oracle_validation(force=force)
                force = False
            except (OSError, ValueError, KeyError, RuntimeError):
                atomic_text(
                    STATE / "status.json",
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "checked_at": timestamp(),
                            "status": "retrying",
                            "error": traceback.format_exc()
                        },
                        indent=2) + "\n")
                traceback.print_exc()
                if not args.watch:
                    raise
            if not args.watch:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
