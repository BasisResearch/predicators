"""Capture final scorecards and plot the latest five-domain comparison."""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DOMAINS = ["Balloons", "Boil", "Bridge", "Domino", "Fan"]
CONDITIONS = ["Noiseless", "Noisy"]
COLORS = {"MB": "#087f8c", "MF": "#c15b24"}
FEATURES = [
    "code_sim_learning_interval_belief",
    "agent_explorer_info_seeking_noise_aware",
    "code_sim_learning_rollout_noise_filter",
    "code_sim_learning_carry_posterior", "code_sim_learning_fit_evidence",
    "continual_belief_frame"
]
NOISE_LABELS = {
    "Balloons": "10 mm pose / 0.02 rad",
    "Boil": "12.5 mm / 0.05 rad + reading 0.07",
    "Bridge": "5 mm pose / 0.02 rad",
    "Domino": "10 mm pose / 0.04 rad",
    "Fan": "5 mm pose / 0.02 rad"
}


def capture() -> None:
    """Poll exactly the selected runs, preserving source hashes and cards."""
    manifest = json.loads((OUT / "selection.json").read_text())
    records = []
    for entry in manifest["runs"]:
        path = ROOT / entry["source"]
        raw = path.read_bytes()
        card = json.loads(raw)
        config = ""
        with path.with_name("info.log").open() as stream:
            for line in stream:
                if "Full config:" in line:
                    config = next(stream)
                    break
        flags = {}
        for flag in FEATURES:
            match = re.search(r"\b" + flag + r"=(True|False)", config)
            flags[flag] = match.group(1) == "True" if match else None
        records.append({
            **entry, "sha256": hashlib.sha256(raw).hexdigest(),
            "uncertainty_flags": flags,
            "card": card
        })
    snapshot = {
        "generated_by": "make_plot.py; do not edit manually",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "selection": manifest,
        "records": records
    }
    validate(snapshot)
    (OUT / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def validate(snapshot: Dict[str, Any]) -> None:
    """Reject missing seeds, partial runs, wrong noise, or inconsistent
    totals."""
    records = snapshot["records"]
    assert len(records) == 50
    for domain in DOMAINS:
        for condition in CONDITIONS:
            for arm in COLORS:
                group = [
                    r for r in records
                    if (r["domain"], r["condition"],
                        r["arm"]) == (domain, condition, arm)
                ]
                seeds = [r["seed"] for r in group]
                assert sorted(seeds) == ([0, 1, 2] if condition == "Noiseless"
                                         else [0, 1])
    for record in records:
        card = record["card"]
        totals = card["totals"]
        levels = card["levels"]
        assert card["finished_at"] and card["end_reason"]
        assert card["seed"] == record["seed"]
        assert card["arm"] == ("agent_continual" if record["arm"] == "MB" else
                               "agent_continual_model_free")
        assert totals["levels_total"] == len(levels)
        assert totals["levels_total"] == (3 if record["domain"] == "Balloons"
                                          else 2)
        assert totals["levels_completed"] == sum(lv["won"] for lv in levels)
        assert totals["total_steps"] == sum(lv["steps"] for lv in levels)
        assert totals["total_resets"] == sum(lv["resets"] for lv in levels)
        sigmas = [
            card.get("obs_noise_" + k, 0.0)
            for k in ["position", "orientation", "scalar"]
        ]
        expected = ([0, 0, 0] if record["condition"] == "Noiseless" else
                    snapshot["selection"]["noise_sigmas"][record["domain"]])
        assert sigmas == expected
        if record["arm"] == "MB":
            assert card["end_reason"] == "all_levels_won"
            assert totals["levels_completed"] == totals["levels_total"]
            enabled = record["uncertainty_flags"].values()
            if record["condition"] == "Noisy":
                assert all(enabled)
            else:
                assert not any(enabled)


def summarize(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep all-run effort and success-conditional cost as separate metrics."""
    rows = []
    for domain in DOMAINS:
        for condition in CONDITIONS:
            for arm in COLORS:
                records = [
                    r for r in snapshot["records"]
                    if (r["domain"], r["condition"],
                        r["arm"]) == (domain, condition, arm)
                ]
                cards = [r["card"] for r in records]
                totals = [c["totals"] for c in cards]
                successful = [
                    t for t in totals
                    if t["levels_completed"] == t["levels_total"]
                ]
                rows.append({
                    "domain":
                    domain,
                    "condition":
                    condition,
                    "arm":
                    arm,
                    "seeds":
                    len(cards),
                    "successful_seeds":
                    len(successful),
                    "solve_percent":
                    100 * mean(t["levels_completed"] / t["levels_total"]
                               for t in totals),
                    "mean_steps_all":
                    mean(t["total_steps"] for t in totals),
                    "mean_steps_success":
                    mean(t["total_steps"]
                         for t in successful) if successful else None,
                    "mean_resets_all":
                    mean(t["total_resets"] for t in totals),
                    "source_git_shas":
                    ", ".join(sorted({c["git_sha"]
                                      for c in cards})),
                    "experiments":
                    ", ".join(sorted({c["config"]
                                      for c in cards}))
                })
    return rows


def number(value: Optional[float]) -> str:
    """Format a table value while preserving missing successful costs."""
    return "N/A" if value is None else f"{value:,.1f}".removesuffix(".0")


def write_report(snapshot: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Export auditable metrics and explain the historical comparisons."""
    with (OUT / "summary.tsv").open("w") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=list(rows[0]),
                                delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Latest completed MB versus MF results across five domains", "",
        "Generated by `make_plot.py`; do not edit manually.",
        f"Scorecards polled at {snapshot['captured_at']}.", "",
        "All 50 selected runs are final: three noiseless seeds and two noisy seeds per domain and arm.",
        "MB solved every level in all 25 selected runs, including both noisy seeds in each domain.",
        "The noiseless MB references have the six added uncertainty features off; the noisy MB runs have all six on.",
        "The selection includes all seeds in these cohorts, including MF failures, and uses the archived recency decision for the duplicate noiseless Boil MF seed 2.",
        "No experiments were launched, resumed, cancelled, or changed by this collection.",
        "", "![Mean effort and solve rate](mb_vs_mf.png)", "",
        "[PDF](mb_vs_mf.pdf) | [SVG](mb_vs_mf.svg) | [Successful-run cost plot](successful_steps.png) | [TSV](summary.tsv)",
        "",
        "The main step curves use whole-run real environment steps averaged over all seeds, matching the frozen noiseless table's convention.",
        "Low effort can mean early failure, so every step panel is paired with the fraction of all train and test levels solved.",
        "The companion plot averages steps only over runs that won every level and reports N/A when none succeeded.",
        "Resets also average over all seeds, including unsuccessful runs.",
        "Lines join two categorical settings; they do not interpolate a measured noise-response curve or isolate the effect of noise.",
        "No confidence intervals are estimated from these two or three seeds.",
        "", "## Results", "",
        "| Domain | Setting | Arm | Successful runs | Levels solved | Steps, all runs | Steps, successful runs | Resets, all runs |",
        "|---|---|---|---:|---:|---:|---:|---:|"
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['condition']} | {row['arm']} | "
            f"{row['successful_seeds']}/{row['seeds']} | "
            f"{number(row['solve_percent'])}% | "
            f"{number(row['mean_steps_all'])} | "
            f"{number(row['mean_steps_success'])} | "
            f"{number(row['mean_resets_all'])} |")
    lines += [
        "", "## MB versus MF gaps", "",
        "Step gap is MF minus MB mean effort over all runs; a positive value means MF spent more real steps, irrespective of success.",
        "Solve gap is MB minus MF in percentage points over all levels.", "",
        "| Domain | Step gap, noiseless | Step gap, noisy | Solve gap, noiseless | Solve gap, noisy |",
        "|---|---:|---:|---:|---:|"
    ]
    for domain in DOMAINS:
        gaps = []
        for condition in CONDITIONS:
            mb, mf = [
                r for r in rows
                if r["domain"] == domain and r["condition"] == condition
            ]
            gaps.append((mf["mean_steps_all"] - mb["mean_steps_all"],
                         mb["solve_percent"] - mf["solve_percent"]))
        lines.append(f"| {domain} | {number(gaps[0][0])} | "
                     f"{number(gaps[1][0])} | {number(gaps[0][1])} pp | "
                     f"{number(gaps[1][1])} pp |")
    lines += [
        "", "## Versions and limits", "",
        "This is a historical comparison of the latest completed cohorts, with different source versions across domains and settings.",
        "Newly scheduled runs without final scorecards do not enter this snapshot.",
        "Noiseless references follow the 30-run selection archived at tag `noiseless-sweep-results-20260909`: Bridge and Domino use MB `uniform_r2`; the other MB domains and all MF domains use `uniform`.",
        "The archive's later Boil MF seed 2 rerun was selected by recency, irrespective of outcome; the earlier duplicate and cancelled MF r2 runs remain excluded.",
        "Some scorecard SHAs identify a base commit with runtime patches; they are provenance labels, not guarantees of identical code.",
        "**Boil changes task size:** the noiseless uniform runs test two jugs, while the latest noisy pose-plus-reading runs test one jug.",
        "Its line is therefore a comparison of available experiments, not a matched task comparison, and the lower noisy MB cost cannot be attributed to noise handling alone.",
        "The older one-jug noiseless m3 runs are excluded: they have only one seed per arm and predate exact-observation sanitization.",
        "Bridge and Fan noisy runs include later manipulation and Wait fixes, respectively; these changes also affect step costs.",
        "Balloons uses the original non-hatch task distribution; hatch and corrected-generator experiments are excluded.",
        "Its latest MB uses the simulator subclass interface and the shared timed-Wait/proprioception update, while reused noisy MF predates that interface update.",
        "The original Balloons task-generator caveat remains part of this historical dataset, and development used these seeds.",
        "Seeds 0, 1, and 2 contribute to noiseless points; seeds 0 and 1 contribute to noisy points, so the means are not paired differences.",
        "",
        "| Domain | Setting | Arm | Experiment | Scorecard source revisions |",
        "|---|---|---|---|---|"
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['condition']} | {row['arm']} | "
            f"`{row['experiments']}` | `{row['source_git_shas']}` |")
    lines += [
        "", "Noisy position, orientation and scalar standard deviations:", ""
    ]
    for domain in DOMAINS:
        lines.append(f"- {domain}: {NOISE_LABELS[domain]}.")
    lines += [
        "", "## Source scorecards", "",
        "Full parsed scorecards, SHA-256 digests, feature flags, and selection are frozen in [snapshot.json](snapshot.json).",
        "The exact paths are also in [selection.json](selection.json).", ""
    ]
    for record in snapshot["records"]:
        label = (f"{record['domain']} {record['condition']} "
                 f"{record['arm']} seed {record['seed']}")
        lines.append(f"- [{label}](../../../{record['source']})")
    lines += [
        "", "Regenerate from the saved snapshot on a compute node:", "",
        "```bash",
        "python docs/uncertainty-results/five-domain-latest/make_plot.py",
        "```", "",
        "Add `--refresh` to poll the same selected scorecards again; selection does not change automatically.",
        "Use `--data-only` to validate and export tables without rendering."
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def plot(rows: List[Dict[str, Any]], successful_only: bool = False) -> None:
    """Draw one MB and one MF series per domain, with solve rate beneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "svg.fonttype": "none"
    })
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.5))
    fig.subplots_adjust(left=.06,
                        right=.98,
                        bottom=.18,
                        top=.76,
                        hspace=.35,
                        wspace=.4)
    fig.text(.06,
             .96,
             "MB vs MF: from noiseless to noisy",
             fontsize=23,
             fontweight="bold",
             color="#172e38")
    detail = (
        "Steps among fully successful runs; N/A means none succeeded."
        if successful_only else
        "Mean real environment steps over all runs, paired with solve rate.")
    fig.text(.06, .915, detail, fontsize=12, color="#4e606a")
    fig.text(.06,
             .88, "3 noiseless seeds / 2 noisy seeds per arm. "
             "MB won every level in every selected run.",
             color="#4e606a")
    fig.legend(handles=[
        Line2D([0], [0], color=COLORS[a], marker=m, linewidth=2, label=a)
        for a, m in [("MB", "o"), ("MF", "s")]
    ],
               loc="upper right",
               bbox_to_anchor=(.98, .965),
               ncol=2,
               frameon=False,
               fontsize=12)
    metric = "mean_steps_success" if successful_only else "mean_steps_all"
    for col, domain in enumerate(DOMAINS):
        top, bottom = axes[:, col]
        title = domain + (" *" if domain == "Boil" else "")
        top.set_title(title, pad=30, fontsize=14, loc="left")
        top.text(0,
                 1.08,
                 NOISE_LABELS[domain],
                 transform=top.transAxes,
                 fontsize=8.3,
                 color="#52636c")
        domain_rows = [r for r in rows if r["domain"] == domain]
        highest = max(r[metric] or 0 for r in domain_rows)
        for arm in COLORS:
            points = [
                next(r for r in domain_rows
                     if r["condition"] == c and r["arm"] == arm)
                for c in CONDITIONS
            ]
            marker = "o" if arm == "MB" else "s"
            positions = [x + (-.025 if arm == "MB" else .025) for x in [0, 1]]
            for ax, key in [(top, metric), (bottom, "solve_percent")]:
                values = [p[key] for p in points]
                ax.plot(positions,
                        [float("nan") if v is None else v for v in values],
                        color=COLORS[arm],
                        marker=marker,
                        markersize=7,
                        linewidth=2,
                        linestyle="--" if domain == "Boil" else "-",
                        zorder=3)
                for x, value, point in zip(positions, values, points):
                    if value is None:
                        ax.text(x, (.9 if domain == "Boil" else .52) * highest,
                                "MF: N/A\n0 successful runs",
                                color=COLORS[arm],
                                ha="center",
                                fontsize=8.5)
                        continue
                    label = (f"{value:.1f}%".replace(".0%", "%")
                             if key == "solve_percent" else number(value))
                    if successful_only and key == metric:
                        label += f"\n(n={point['successful_seeds']})"
                    offset = 10 if arm == "MB" else -19
                    horizontal_offset = 0
                    alignment = "center"
                    if key == metric:
                        other = next(r[key] for r in domain_rows
                                     if r["condition"] == point["condition"]
                                     and r["arm"] != arm)
                        below = (other is not None and value < other
                                 and value > .12 * highest)
                        offset = (
                            -30 if successful_only else -19) if below else 10
                        if (other is not None and value < other
                                and value <= .12 * highest
                                and other - value < .18 * highest):
                            horizontal_offset = -22
                            alignment = "right"
                    ax.annotate(label, (x, value),
                                xytext=(horizontal_offset, offset),
                                textcoords="offset points",
                                ha=alignment,
                                color=COLORS[arm],
                                fontsize=9,
                                fontweight="medium")
        for ax in [top, bottom]:
            ax.set_xlim(-.28, 1.28)
            ax.set_xticks([0, 1], CONDITIONS)
            ax.grid(axis="y", color="#dbe2e6", linewidth=.7, zorder=0)
            ax.tick_params(length=0, labelcolor="#52636c", pad=7)
        top.set_ylim(0, highest * 1.27)
        top.yaxis.set_major_locator(MaxNLocator(4))
        top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        bottom.set_ylim(0, 119)
        bottom.set_yticks([0, 25, 50, 75, 100])
        bottom.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:.0f}%"))
        if col == 0:
            top.set_ylabel("Steps per run (lower is less effort)", labelpad=12)
            bottom.set_ylabel("Train + test levels solved", labelpad=12)
    fig.text(.06,
             .10, "Noiseless: MB without added uncertainty features. "
             "Noisy: latest completed MB with features; historical MF.",
             fontsize=10,
             color="#4e606a")
    fig.text(.06,
             .068, "Versions and seed counts differ; connecting lines are "
             "descriptive. * Boil also changes from 2 test jugs to 1.",
             fontsize=10,
             color="#4e606a")
    fig.text(
        .06,
        .036, "Balloons uses the original non-hatch distribution. "
        "Low all-run effort may reflect early failure. Polled 10 Sep 2026.",
        fontsize=10,
        color="#4e606a")
    name = "successful_steps" if successful_only else "mb_vs_mf"
    for extension in ["png", "pdf", "svg"]:
        path = OUT / f"{name}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(
                line.rstrip()
                for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    """Refresh only on request; otherwise reproduce the captured results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        capture()
    snapshot = json.loads((OUT / "snapshot.json").read_text())
    validate(snapshot)
    rows = summarize(snapshot)
    write_report(snapshot, rows)
    if not args.data_only:
        plot(rows)
        plot(rows, successful_only=True)
    print(f"Verified {len(snapshot['records'])} final scorecards and "
          f"{len(rows)} summary rows; all 25 MB runs won every level.")


if __name__ == "__main__":
    main()
