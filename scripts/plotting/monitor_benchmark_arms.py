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
            for directory in directories[arm]:
                seed = int(Path(directory).name.removeprefix("seed"))
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
                f"(n={len(successes)}) | {resets} | "
                f"{count}/{len(directories[arm])} |")
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
                    "# Opus benchmark sweep: twelve agents and r2 cohorts\n",
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
        "Preflight settings differ across historical cohorts: some selected "
        "EMPIRIC Boil, Domino, and Balloons runs had preflight enabled; "
        "the selected Bridge reruns, EMPIRIC r2 replacements, and newer "
        "comparison arms used preflight off.\n"
        "This is not a matched preflight ablation.")
    prefix = re.sub(
        r"All agents are Claude Opus with the composite skill library\.\n"
        r"Preflight settings differ across historical cohorts:.*?\n"
        r"This is not a matched preflight ablation\.",
        "All agents are Claude Opus with the composite skill library.\n"
        "Preflight settings differ across historical cohorts: some selected "
        "EMPIRIC Boil, Domino, and Balloons runs had preflight enabled; "
        "the selected Bridge reruns, EMPIRIC r2 replacements, and newer "
        "comparison arms used preflight off.\n"
        "This is not a matched preflight ablation.",
        prefix,
        count=1,
        flags=re.DOTALL)
    validation_link = "[Oracle repair pilot](oracle-dynamics-validation-r2.md)"
    pilot_note = (
        "Oracle dynamics combines r2 seeds 0-4 for Domino and Bridge "
        "with r1 seeds 0-2 and r2 seeds 3-4 for the other domains.\n"
        "Fan instead uses the fresh five-seed prompt-aligned Oracle cohort; "
        "earlier Fan Oracle results are excluded rather than pooled.\n"
        "The " + validation_link + " also shows the two Oracle "
        "rounds side by side.\n\n")
    if validation_link in prefix:
        prefix = re.sub(
            r"(?:Oracle dynamics r2 is a separate entry.*?\n)?"
            r"(?:Oracle dynamics combines.*?\n)?"
            r"(?:Fan instead uses.*?\n)?"
            r"(?:Only finished Domino and Bridge r2 runs.*?\n)?"
            r"(?:The r2 entry includes.*?\n)?"
            r"The \[Oracle repair pilot\].*?\n\n",
            lambda _: pilot_note,
            prefix,
            count=1)
    else:
        prefix += pilot_note
    prefix = re.sub(
        r"^Compiled .*?$",
        f"Compiled {stamp} from {len(rows)} finished scorecards on "
        "the current benchmark settings and archived Fan development cohorts.",
        prefix,
        count=1,
        flags=re.MULTILINE)
    direct_done = sum(
        r["arm"] == "mf_scene_package" and r["domain"] in plot["PAPER_DOMAINS"]
        for r in rows)
    pending = "; ".join(direct_pending) or "none"
    no_unc_done = any(r["arm"] == "no_uncertainty" and r["seed"] == 2
                      and r["domain"] == "Balloons (composition)"
                      for r in rows)
    no_unc_status = "finished" if no_unc_done else "unfinished"
    oracle_r2_done = sum(
        r["arm"] == "oracle_dynamics" and r["domain"] in plot["PAPER_DOMAINS"]
        for r in rows)
    empiric_done = sum(
        r["arm"] == "MB" and r["domain"] in plot["PAPER_DOMAINS"]
        for r in rows)
    transfer_done = {
        arm: sum(r["arm"] == arm and r["domain"] == plot["FAN_TRANSFER"]
                 for r in rows)
        for arm in ("MB", "MF")
    }
    expected = {
        arm: sum(
            len(dirs[arm]) for domain, dirs in plot["DOMAINS"]
            if domain in plot["PAPER_DOMAINS"])
        for arm in plot["ARMS"]
    }
    variant_status = ""
    assets_done = sum(r["arm"] == "from_assets" for r in rows)
    for domain in plot["FAN_VARIANTS"]:
        counts = []
        planned = dict(plot["DOMAINS"])[domain]
        for arm, label in zip(plot["ARMS"], plot["LABELS"]):
            if not planned[arm]:
                continue
            group = [
                r for r in rows if r["domain"] == domain and r["arm"] == arm
            ]
            wins = sum(r["won"] == r["levels"] for r in group)
            counts.append(f"{label} {wins}/{len(group)} solved, "
                          f"{len(group)}/{len(planned[arm])} finished")
        variant_status += f"- {domain}: " + "; ".join(counts) + ".\n"
    status = (
        "## Status at this snapshot\n\n"
        f"- Oracle dynamics: {oracle_r2_done}/"
        f"{expected['oracle_dynamics']} seeds finished across five domains.\n"
        f"- EMPIRIC: {empiric_done}/25 seeds finished "
        "(five selected seeds per domain; Boil and Balloons seed 2 and "
        "seeds 3-4 outside Fan use r2; Fan uses the repaired-skill cohort).\n"
        f"- EMPIRIC from assets: {assets_done}/10 seeds finished "
        "(two per domain; Fan uses the ramp variant, not the maze).\n"
        f"- Fan transfer pilot: EMPIRIC {transfer_done['MB']}/2 and "
        f"Direct agent {transfer_done['MF']}/2 seeds finished "
        "(separate from the five-domain totals above).\n"
        f"{variant_status}"
        f"- Direct + scene: {direct_done}/"
        f"{expected['mf_scene_package']} seeds finished.\n"
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
    assets_note = (
        "## EMPIRIC from assets development\n\n"
        "EMPIRIC from assets builds its own scene and mechanisms while "
        "retaining harness fitting and uncertainty.\n"
        "It is a separate magenta entry, with two seeds planned each for "
        "Domino, Bridge, Balloons, Boil and Fan ramp.\n"
        "Cancelled Fan maze pilots are excluded; the Fan maze and inertial "
        "columns have no results for this agent.\n"
        "Only finished scorecards enter the bars and curves; empty entries "
        "are pending or untested, not zero solve rates.\n"
        "These development runs are excluded from paper figures.\n"
        "See [implementation and launch notes](../amps/empiric-from-assets.md)."
        "\n\n")
    if "## EMPIRIC from assets development" not in prefix:
        prefix += assets_note
    prefix = re.sub(
        r"## Status at this snapshot\n.*?(?=## Figure and plotting method)",
        lambda _: status,
        prefix,
        count=1,
        flags=re.DOTALL)
    prefix = prefix.replace(
        "It is a separate prospective cohort, "
        "not a matched preflight ablation.",
        "The original and r2 seeds are pooled in one EMPIRIC entry, "
        "with source-cohort provenance retained.\n"
        "This is not a matched preflight ablation.")
    if "## Fan exposed-transfer pilot" not in prefix:
        prefix += (
            "## Fan exposed-transfer pilot\n\n"
            "The rightmost plot column, Fan transfer, is a separate pilot "
            "with two seeds each for EMPIRIC and the direct agent.\n"
            "The other agents have not been launched on this variant; "
            "their empty rows are missing results, not failures.\n"
            "Only finished seeds enter the bars and curves; pending runs "
            "are listed below and do not count as zero successes.\n"
            "See the [illustrated task description]"
            "(../amps/fan-exposed-transfer.md) and "
            "[launch configuration]"
            "(../../scripts/configs/predicatorv3/"
            "continual_fan_transfer_pilot_r1.yaml).\n"
            "This column is excluded from the paper figure and its "
            "data selection.\n\n")
    prefix = prefix.replace("The rightmost plot column, Fan transfer,",
                            "The Fan transfer plot column,")
    prefix = prefix.replace(
        "The Fan transfer plot column, is a separate pilot ",
        "The archived Fan transfer experiment is a separate pilot ")
    prefix = prefix.replace(
        "The Fan transfer plot column, Fan transfer, is a separate pilot ",
        "The archived Fan transfer experiment is a separate pilot ")
    prefix = prefix.replace(
        "This column is excluded from the paper figure and its data selection.",
        "This superseded pilot is omitted from the figure; its tables and "
        "logs remain archived below.\n"
        "It is also excluded from the paper figure and its data selection.")
    if "## Fan inertial and ramp development" not in prefix:
        prefix += (
            "## Fan inertial and ramp development\n\n"
            "The Fan inertial and Fan ramp columns include screening seeds 0-1 "
            "and fresh confirmation seeds 2-4 for EMPIRIC and Direct agent.\n"
            "Only finished runs enter bars, curves, and averages; unfinished "
            "runs are listed separately, not counted as failures.\n"
            "Inertial confirmation finished 3/3 for both methods, so its pilot "
            "solve-rate gap did not replicate.\n"
            "Ramp confirmation is a separate prospective test of the frozen "
            "ramp candidate; pooled development results are not "
            "independent confirmation.\n"
            "See the [development record](../amps/fan-development.md) for "
            "task illustrations, cohort provenance, and failure analysis.\n"
            "Both columns are excluded from paper figures "
            "and data selection.\n\n")
    inertial_note = (
        "Five seeds each of Oracle dynamics, Direct + scene, Standalone sim., "
        "No harness fitting, and No explicit uncertainty were also launched "
        "on the frozen no-ramp Fan inertial configuration on September 21.\n"
        "Their finished results are included in the Fan inertial column; "
        "these runs do not use the candidate shared push-skill repair.\n")
    if "were also launched on the frozen no-ramp Fan inertial" not in prefix:
        prefix = prefix.replace(
            "## Fan inertial and ramp development\n\n",
            "## Fan inertial and ramp development\n\n" + inertial_note)
    prefix = prefix.replace(
        "The Fan inertial and Fan ramp columns include screening seeds 0-1 "
        "and fresh confirmation seeds 2-4 for EMPIRIC and Direct agent.",
        "The Fan inertial column includes screening seeds 0-1 and fresh "
        "confirmation seeds 2-4 for EMPIRIC and Direct agent.")
    prefix = prefix.replace(
        "Ramp confirmation is a separate prospective test of the frozen "
        "ramp candidate; pooled development results are not independent "
        "confirmation.",
        "The Fan ramp column now uses only the matched repaired-skill "
        "cohort for the six non-Oracle methods and the fresh prompt-aligned "
        "five-seed cohort for Oracle dynamics.\n"
        "Earlier ramp results are replaced, not pooled; pending new seeds "
        "never fall back to old results.")
    prefix = prefix.replace(
        "with five seeds planned for each of the seven methods.",
        "with nine Oracle dynamics seeds (0-8) and five seeds for each "
        "of the other six methods.")
    prefix = prefix.replace(
        "with seven Oracle dynamics seeds (0-6) and five seeds for each "
        "of the other six methods.",
        "with nine Oracle dynamics seeds (0-8) and five seeds for each "
        "of the other six methods.")
    prefix = prefix.replace(
        "Both columns are excluded from paper figures and data selection.",
        "The ramp setting is now the default Fan in both figures.\n"
        "The old maze, exposed-transfer, and inertial settings remain "
        "archived in these tables but are excluded from both figures.")
    prefix = prefix.replace("Fan inertial column",
                            "archived Fan inertial cohort")
    prefix = prefix.replace("Fan ramp column", "Fan column")
    prefix = prefix.replace("Fan maze test", "Fan ramp test")
    average_intro = original.split("## Averages across seeds",
                                   1)[1].split("| Domain |", 1)[0]
    prefix = prefix.replace(
        "with eleven Oracle dynamics seeds (0-10) and five seeds for each "
        "of the other six methods.",
        "with five prompt-aligned Oracle dynamics seeds and five seeds for "
        "each of the other six methods.")
    average_intro = average_intro.replace(
        "The last column gives how many of the three seeds have finished; "
        "rows with fewer than three are provisional.",
        "The last column gives finished versus planned seeds for each "
        "entry; incomplete entries are provisional.")
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
        "This diagnostic comparison retains Domino and Bridge "
        "seeds 0-2 with preflight off.",
        "Additional seeds 3-4 across all five domains are tracked in "
        "the main benchmark, not pooled into this repair pilot.",
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
        force = args.force
        while True:
            try:
                plot = runpy.run_path(
                    str(ROOT / "scripts/plotting/plot_benchmark_arms.py"))
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
