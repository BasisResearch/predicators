"""Reproduce the uncertainty comparison from a saved scorecard snapshot.

Run with --refresh to capture the selected source scorecards again. By
default, regenerate the report and figures without changing the
snapshot.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SNAPSHOT = OUT / "snapshot.json"
COLORS = {"MB": "#4169a1", "MB + uncertainty": "#008577", "MF": "#d77831"}
# These are the historical references selected for the original noise sweep.
# Do not pool them with later runs from other protocols or code trees.
EXPERIMENTS = [
    ("Domino", "No noise", "MB", "m4", 1),
    ("Domino", "No noise", "MF", "m4", 1),
    ("Domino", "Pose 0.5 cm / 0.02 rad", "MB", "noise5mm", 2),
    ("Domino", "Pose 0.5 cm / 0.02 rad", "MF", "noise5mm", 2),
    ("Domino", "Pose 1 cm / 0.04 rad", "MB", "noise10mm", 2),
    ("Domino", "Pose 1 cm / 0.04 rad", "MF", "noise10mm", 2),
    ("Domino", "Pose 1 cm / 0.04 rad", "MB + uncertainty", "noise10mm_v2", 2),
    ("Domino", "Pose 2 cm / 0.08 rad", "MB", "noise20mm", 2),
    ("Domino", "Pose 2 cm / 0.08 rad", "MF", "noise20mm", 2),
    ("Boil", "No noise", "MB", "m3", 1),
    ("Boil", "No noise", "MF", "m3", 1),
    ("Boil", "Pose 1.25 cm / 0.05 rad", "MB", "noise12mm_r2", 2),
    ("Boil", "Pose 1.25 cm / 0.05 rad", "MF", "noise12mm", 2),
    ("Boil", "Pose 2.5 cm / 0.10 rad", "MB", "noise25mm", 2),
    ("Boil", "Pose 2.5 cm / 0.10 rad", "MF", "noise25mm", 2),
    ("Boil", "Pose 1.25 cm / 0.05 rad + reading 0.07", "MB + uncertainty",
     "noise_p12_r07", 2),
    ("Boil", "Pose 1.25 cm / 0.05 rad + reading 0.07", "MF", "noise_p12_r07",
     2),
]
SIGMAS = {
    "m3": (0.0, 0.0, 0.0),
    "m4": (0.0, 0.0, 0.0),
    "noise5mm": (0.005, 0.02, 0.0),
    "noise10mm": (0.01, 0.04, 0.0),
    "noise10mm_v2": (0.01, 0.04, 0.0),
    "noise20mm": (0.02, 0.08, 0.0),
    "noise12mm": (0.0125, 0.05, 0.0),
    "noise12mm_r2": (0.0125, 0.05, 0.0),
    "noise25mm": (0.025, 0.1, 0.0),
    "noise_p12_r07": (0.0125, 0.05, 0.07),
}


def capture():
    """Read one unambiguous run per selected experiment and seed."""
    rows = []
    for domain, noise, arm, suffix, num_seeds in EXPERIMENTS:
        approach = "agent_continual_model_free" if arm == "MF" else "agent_continual"
        env = "domino_high_friction_turn" if domain == "Domino" else "boil"
        experiment = f"{env}-{approach}_{suffix}"
        runs = []
        for seed in range(num_seeds):
            directory = ROOT / "logs" / approach / experiment / f"seed{seed}"
            paths = sorted(directory.glob("run_*/scorecard.json"))
            if len(paths) != 1:
                raise ValueError(
                    f"Expected one scorecard in {directory}: {paths}")
            card = json.loads(paths[0].read_text())
            totals = card["totals"]
            assert card["seed"] == seed
            assert totals["levels_total"] == 2
            assert totals["levels_completed"] == sum(
                level["won"] for level in card["levels"])
            assert totals["total_steps"] == sum(level["steps"]
                                                for level in card["levels"])
            assert totals["total_resets"] == sum(level["resets"]
                                                 for level in card["levels"])
            expected = SIGMAS[suffix]
            sigmas = tuple(
                card.get(key, 0.0)
                for key in ("obs_noise_position", "obs_noise_orientation",
                            "obs_noise_scalar"))
            assert sigmas == expected, (experiment, sigmas)
            runs.append({
                "seed": seed,
                "source": str(paths[0].relative_to(ROOT)),
                "source_git_sha": card["git_sha"],
                "finished_at": card.get("finished_at"),
                "end_reason": card.get("end_reason"),
                "levels_solved": totals["levels_completed"],
                "levels_total": totals["levels_total"],
                "steps": totals["total_steps"],
                "resets": totals["total_resets"],
                "noise_sigmas": {
                    "position_m": sigmas[0],
                    "orientation_rad": sigmas[1],
                    "scalar": sigmas[2]
                },
            })
        rows.append({
            "domain": domain,
            "noise": noise,
            "arm": arm,
            "historical_unsanitized": domain == "Boil" and noise == "No noise",
            "runs": runs,
        })
    payload = {
        "generated_by": "make_figures.py; do not edit manually",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows
    }
    SNAPSHOT.write_text(json.dumps(payload, indent=2) + "\n")


def aggregate(row):
    """Average steps over successful seeds; other metrics use all seeds."""
    runs = row["runs"]
    successful = [
        run for run in runs if run["finished_at"] is not None
        and run["levels_solved"] == run["levels_total"]
    ]
    return {
        "domain":
        row["domain"],
        "noise":
        row["noise"],
        "arm":
        row["arm"],
        "seeds":
        len(runs),
        "successful_seeds":
        len(successful),
        "solve_rate_pct":
        100 * mean(run["levels_solved"] / run["levels_total"] for run in runs),
        "mean_steps_successful":
        mean(run["steps"] for run in successful) if successful else None,
        "mean_resets":
        mean(run["resets"] for run in runs),
        "unfinished_runs":
        sum(run["finished_at"] is None for run in runs),
        "historical_unsanitized":
        row["historical_unsanitized"],
    }


def number(value):
    """Format measured averages and explicitly mark undefined means."""
    if value is None:
        return "N/A"
    return f"{value:,.1f}".removesuffix(".0")


def write_report(payload):
    """Write the comparison table, caveats and source links."""
    summaries = [aggregate(row) for row in payload["rows"]]
    unfinished = sum(summary["unfinished_runs"] for summary in summaries)
    with (OUT / "summary.tsv").open("w") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=list(summaries[0]),
                                delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    lines = [
        "# Noiseless and noisy continual results",
        "",
        "Generated by `make_figures.py` from `snapshot.json`; do not edit this report manually.",
        f"Scorecards captured at {payload['observed_at']}.",
        "",
        "Solve rate is the fraction of the two levels solved in each run, averaged across the listed seeds.",
        "It includes the training level and the test level, not just test performance.",
        "Mean steps uses only successful seeds: completed runs that solved both the train and test levels.",
        "For each successful seed, its whole-run step total includes all attempts and resets before completion.",
        "The table reports the number of successful seeds contributing to that mean.",
        "N/A means zero successful seeds; these step values are omitted from the curves, not plotted as zero.",
        "Solve rate and mean resets continue to use all seeds, including unsuccessful runs.",
        "The noiseless references have seed 0 only; noisy conditions have seeds 0 and 1.",
        "`MB` means the earlier model-based configuration without the new uncertainty features; `MB + uncertainty` enables the new features.",
        "MF remains the plain model-free code agent.",
        "",
        f"{unfinished} selected run(s) are unfinished at the capture time."
        if unfinished else "All selected scorecards are final.",
        "",
        "| Domain | Noise | Arm | Seeds | Successful seeds | Solve rate | Mean steps (successful seeds) | Mean resets (all seeds) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        mark = " †" if summary["historical_unsanitized"] else ""
        if summary["unfinished_runs"]:
            mark += " *"
        lines.append(
            f"| {summary['domain']} | {summary['noise']} | {summary['arm']}{mark} | {summary['seeds']} | {summary['successful_seeds']} | {number(summary['solve_rate_pct'])}% | {number(summary['mean_steps_successful'])} | {number(summary['mean_resets'])} |"
        )
    lines.extend([
        "",
        "The pose settings list position sigma in cm and orientation sigma in radians.",
        "The combined boil point additionally has scalar-reading sigma 0.07.",
        "",
        "† Both historical noiseless boil runs predate exact-observation sanitization and used a protocol that exposed privileged hidden heat in agent data.",
        "They are historical references, not clean controls for the noisy runs.",
        "Their presence in this table does not establish that noise has no cost.",
        *([
            "* Averages include an unfinished run at the capture time and are provisional.",
            "An unplayed level in that run contributes zero current wins; this is not its final outcome."
        ] if unfinished else []),
        "Step costs are conditional on success and should be read alongside solve rate.",
        "The older MB comparisons were run on earlier commits; this is not a controlled ablation on one code tree.",
        "There is no MB-without-uncertainty result at the combined boil pose-plus-reading point.",
        "The pose-only boil comparison cannot isolate the effect of the new features under reading noise.",
        "",
        "Each curve figure shows one metric in one environment, with MF, MB without the new features and MB with the features as separate series.",
        "Each marker uses the metric's seed subset defined above; a series with only one reported setting is an isolated point.",
        "The x-axis is categorical: equal spacing does not imply equal noise increments.",
        "Small horizontal offsets keep coincident series visible; the y-values are unchanged.",
        "Lines connect only adjacent measured settings in the pose sweep, with no extrapolation or values inserted at unmeasured settings.",
        "The combined boil point is separated from the pose sweep because it changes noise class and uses a smaller pose sigma than the 2.5 cm point.",
        "Hollow markers with † denote historical unsanitized boil results and are not connected to the clean noisy results.",
        "The cost axes are linear and include zero.",
        "No confidence intervals are estimated from one or two seeds.",
        "",
        *[
            line for domain in ("domino", "boil")
            for metric in ("solve_rate", "steps", "resets") for line in [
                f"![{domain.title()} {metric.replace('_', ' ')}]({domain}_{metric}.png)",
                "",
                f"[{domain.title()} {metric.replace('_', ' ')} PDF]({domain}_{metric}.pdf) · [SVG]({domain}_{metric}.svg)",
                "",
            ]
        ],
        "",
        "Data: [summary TSV](summary.tsv), [snapshot JSON with source paths and commit IDs](snapshot.json).",
        "",
        "Presentation: [standalone HTML slides](../slides/uncertainty_results_slides.html), [PDF slides](../slides/uncertainty_results_slides.pdf).",
        "",
        "Reproduce this snapshot's report and figures from the repository root:",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/uncertainty-results/make_figures.py",
        "```",
        "",
        "Use `--refresh --data-only` to explicitly replace the snapshot and table with current source scorecards, then run the command above to redraw the figures.",
        "On Engaging, run figure rendering on a compute node.",
        "This script does not launch, resume or cancel experiments.",
        "",
        "Source scorecards:",
        "",
    ])
    for row in payload["rows"]:
        for run in row["runs"]:
            label = f"{row['domain']}, {row['noise']}, {row['arm']}, seed {run['seed']}"
            lines.append(f"- [{label}](../../{run['source']})")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:lines.index(
        "The pose settings list position sigma in cm and orientation sigma in radians."
    )]))


def plot_domain(payload, domain):
    """Export one curve plot per metric without inventing missing points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    rows = [row for row in payload["rows"] if row["domain"] == domain]
    settings = list(dict.fromkeys(row["noise"] for row in rows))
    assert len(settings) == 4
    labels = ([
        "No noise\nn = 1", "0.5 cm / 0.02 rad\nn = 2",
        "1 cm / 0.04 rad\nn = 2", "2 cm / 0.08 rad\nn = 2"
    ] if domain == "Domino" else [
        "No noise †\nn = 1", "1.25 cm / 0.05 rad\nn = 2",
        "2.5 cm / 0.10 rad\nn = 2",
        "1.25 cm / 0.05 rad\n+ reading 0.07 · n = 2"
    ])
    style = {
        "MF": ("o", "--", 0.065, "MF"),
        "MB": ("s", "-", -0.065, "MB w/o features"),
        "MB + uncertainty": ("D", "-", 0.0, "MB w/ features"),
    }
    metrics = [("solve_rate_pct", "Solve rate", "solve_rate"),
               ("mean_steps_successful", "Mean steps on successful seeds",
                "steps"), ("mean_resets", "Mean resets", "resets")]
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42
    })
    for key, title, filename in metrics:
        fig, ax = plt.subplots(figsize=(10.8, 7.2))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.77, bottom=0.31)
        fig.patch.set_facecolor("white")
        fig.text(0.10,
                 0.95,
                 f"{domain} | {title}",
                 fontsize=23,
                 weight="bold",
                 color="#182c3b")
        fig.text(0.10,
                 0.90,
                 "Across observation-noise settings",
                 fontsize=12,
                 color="#526475")
        legend = [
            Line2D([], [],
                   color=COLORS[arm],
                   marker=marker,
                   linestyle=line,
                   markersize=7,
                   linewidth=2,
                   label=label)
            for arm, (marker, line, _, label) in style.items()
        ]
        fig.legend(handles=legend,
                   loc="upper left",
                   bbox_to_anchor=(0.09, 0.86),
                   ncol=3,
                   frameon=False,
                   fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e9ed", linewidth=0.8)
        for name, spine in ax.spines.items():
            spine.set_visible(name == "bottom")
            spine.set_color("#ccd5dc")
        ax.set_xlim(-0.32, 3.32)
        tick_labels = labels
        if key == "mean_steps_successful":
            tick_labels = [
                label.replace("\nn = 1",
                              "").replace("\nn = 2",
                                          "").replace(" · n = 2", "")
                for label in labels
            ]
        ax.set_xticks(range(4), tick_labels)
        ax.tick_params(axis="both", length=0, pad=10, colors="#526475")
        ax.set_xlabel("Noise setting: position sigma / orientation sigma",
                      labelpad=14,
                      color="#526475")
        ax.set_ylabel("Levels solved (%)" if key == "solve_rate_pct" else
                      ("Mean steps per successful run" if key
                       == "mean_steps_successful" else "Resets per run"),
                      labelpad=12,
                      color="#526475")
        values = [
            aggregate(row)[key] for row in rows
            if aggregate(row)[key] is not None
        ]
        maximum = max(values, default=0)
        if key == "solve_rate_pct":
            ax.set_ylim(-5, 114)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:.0f}%"))
        elif maximum == 0:
            ax.set_ylim(-0.20, 0.8)
            ax.set_yticks([0, 0.5])
            ax.text(0.5,
                    0.70,
                    "No resets in any run."
                    if key == "mean_resets" else "No successful seeds.",
                    transform=ax.transAxes,
                    ha="center",
                    color="#526475")
        else:
            ax.set_ylim(-0.05 * maximum, 1.30 * maximum)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:,.0f}"))
        if domain == "Boil":
            ax.axvspan(-0.32, 0.32, color="#f4f5f6", zorder=0)
            ax.axvspan(2.5, 3.32, color="#f0f8f7", zorder=0)
            ax.axvline(2.5, color="#bccbd0", linewidth=1, linestyle=":")
            ax.text(3.0,
                    1.03,
                    "Pose + reading",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#526475")
        for arm in ("MB", "MF", "MB + uncertainty"):
            marker, line, offset, _ = style[arm]
            selected = sorted([
                row for row in rows
                if row["arm"] == arm and aggregate(row)[key] is not None
            ],
                              key=lambda row: settings.index(row["noise"]))
            assert len({row["noise"] for row in selected}) == len(selected)
            for previous, current in zip(selected, selected[1:]):
                x0 = settings.index(previous["noise"])
                x1 = settings.index(current["noise"])
                clean = not (previous["historical_unsanitized"]
                             or current["historical_unsanitized"])
                pose_only = all(row["runs"][0]["noise_sigmas"]["scalar"] == 0
                                for row in (previous, current))
                if x1 == x0 + 1 and clean and pose_only:
                    ax.plot(
                        [x0 + offset, x1 + offset],
                        [aggregate(previous)[key],
                         aggregate(current)[key]],
                        color=COLORS[arm],
                        linestyle=line,
                        linewidth=2,
                        zorder=2)
            for row in selected:
                summary = aggregate(row)
                x = settings.index(row["noise"]) + offset
                value = summary[key]
                legacy = row["historical_unsanitized"]
                ax.plot(x,
                        value,
                        marker=marker,
                        markersize=9,
                        markerfacecolor="white" if legacy else COLORS[arm],
                        markeredgecolor=COLORS[arm] if legacy else "white",
                        markeredgewidth=1.8 if legacy else 1.1,
                        linestyle="none",
                        zorder=4)
                dy = -23 if arm == "MF" else 13
                if key == "mean_resets" and arm == "MB + uncertainty":
                    dy = 33
                mark = "†" if legacy else ""
                if summary["unfinished_runs"]:
                    mark += "*"
                text = number(value) + ("%" if key == "solve_rate_pct" else
                                        "") + mark
                if key == "mean_steps_successful":
                    text += f"\nn = {summary['successful_seeds']}"
                    if arm == "MF":
                        dy = -35
                dx = {"MB": -8, "MF": 8, "MB + uncertainty": 0}[arm]
                ax.annotate(text, (x, value),
                            xytext=(dx, dy),
                            textcoords="offset points",
                            ha="center",
                            fontsize=10,
                            weight="bold",
                            color=COLORS[arm],
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.85,
                                "pad": 1
                            },
                            zorder=5)
        notes = [
            "Steps use only completed seeds solving both levels; n beside each point counts those seeds."
            if key == "mean_steps_successful" else
            "Points = all-seed means. Noiseless: n = 1; noisy: n = 2. Solve rate includes train + test."
        ]
        if key == "mean_steps_successful":
            notes.append(
                "Missing step point: unmeasured setting or zero successful seeds. Other metrics use all seeds."
            )
        if domain == "Boil":
            notes.extend([
                "† Unsanitized historical references, shown separately. The combined point is not on the pose sweep.",
                "MB w/ features has one measured setting. Read step costs alongside solve rate."
            ])
        else:
            notes.extend([
                "MB w/ features has one measured setting. Lines join measured adjacent settings; no extrapolation.",
                "Earlier MB runs use older code. Read step costs alongside solve rate."
            ])
        timestamp = datetime.fromisoformat(
            payload["observed_at"]).strftime("%Y-%m-%d %H:%M UTC")
        notes.append(
            f"Categorical x-axis; small offsets reveal overlapping points. Snapshot: {timestamp}."
        )
        fig.text(0.10,
                 0.155,
                 "\n".join(notes),
                 fontsize=8.5 if key == "mean_steps_successful" else 9,
                 color="#526475",
                 va="top",
                 linespacing=1.7)
        for extension in ("png", "pdf", "svg"):
            output = OUT / f"{domain.lower()}_{filename}.{extension}"
            fig.savefig(output, dpi=180, facecolor="white")
            if extension == "svg":
                output.write_text("\n".join(
                    line.rstrip()
                    for line in output.read_text().splitlines()) + "\n")
        plt.close(fig)


def main():
    """Capture only when requested, then regenerate derived artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        capture()
    payload = json.loads(SNAPSHOT.read_text())
    write_report(payload)
    if not args.data_only:
        for domain in ("Domino", "Boil"):
            plot_domain(payload, domain)


if __name__ == "__main__":
    main()
