"""The eight Opus arms on the five Sept 18 benchmark settings, in the style
of the paper's fig3_results: three metric rows (solve rate, successful-run
steps, resets) by five domain columns, bar = mean, dots = seeds.  Solve =
fraction of finished runs winning every level; steps over whole-run
successes only; resets over all finished runs.  Reads the scorecard.json of
the run picked per seed (finished runs preferred; pinned replays excluded).
Arms with no finished run in a domain draw nothing there.

Usage: python scripts/plotting/plot_benchmark_arms.py <output stem>
writes <stem>.png, <stem>.pdf and <stem>-summary.json (per-seed records
with the run directory each one read, plus the per-arm summary)."""
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

LOGS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "logs"))
MB = "agent_continual"
MF = "agent_continual_model_free"
# Highlighted arms first, followed by the de-emphasised arms.
# Oracle dynamics is the grey reference; no uncertainty uses green.
ARMS = [
    "oracle_dynamics", "oracle_dynamics_r2", "MB", "MB_r2", "MF",
    "mf_scene_package", "standalone", "no_fitting", "no_uncertainty",
    "mb_scene_package", "scene_only", "zero_shot", "real_to_sim"
]
LABELS = [
    "Oracle dynamics", "Oracle dynamics r2", "EMPIRIC", "EMPIRIC r2",
    "Direct agent", "Direct agent + scene assets", "Standalone sim.",
    "No harness fitting", "No explicit uncert.", "EMPIRIC + scene pkg.",
    "Scene only", "Zero-shot model", "Agentic real-to-sim"
]
COLORS = [
    "#88929d", "#52616b", "#087f8c", "#034f4f", "#bd5929", "#7a3312",
    "#7467a6", "#b19658", "#397957", "#0b4f6c", "#6b9483", "#5588ad", "#a5573f"
]
GROUPS = [
    ("Oracle reference", 0, 2),
    ("Methods", 2, 7),
    ("EMPIRIC ablations", 7, 9),
    ("Additional comparisons", 9, 13),
]
APPROACH_DIR = {
    "mb_scene_package": "agent_continual",
    "mf_scene_package": "agent_continual_model_free",
    "standalone": "agent_continual_program_world_model",
    "oracle_dynamics": "agent_continual_oracle_dynamics",
    "scene_only": "agent_continual_scene_only",
    "zero_shot": "agent_continual_zero_shot",
    "no_fitting": "agent_continual_no_fitting",
    "no_uncertainty": "agent_continual_no_uncertainty",
    "real_to_sim": "agent_continual_real_to_sim",
}
# Arms relaunched with a revised surface read their newer round
# (continual_standalone_no_uncertainty_r2.yaml).
ROUND = {"standalone": "r2", "no_uncertainty": "r2"}
# Per-domain rounds: EMPIRIC + scene package reran Boil, Bridge and Domino
# as r2 after the base-sim splits (Sept 19); r1 there was cancelled.
ROUND_BY_DOMAIN = {("mb_scene_package", d): "r2"
                   for d in ("Boil (2-jug)", "Bridge (4-span)", "Domino")}
# The Sept 18 benchmark rounds:
# logs/<approach>/<env key>-<arm>_opus_benchmark_<round>/seed<N>.
ENV_KEY = {
    "Bridge (4-span)": "bridge",
    "Fan (maze)": "fan",
    "Domino": "domino_high_friction_turn",
    "Boil (2-jug)": "boil",
    "Balloons (composition)": "balloons"
}
# domain label -> {arm: seed dirs}.  Opus MB/MF ran earlier under their own
# round keys (four-span MB: seed 0 from r2 and seeds 1-2 from the
# preflight-off rerun r3).
DOMAINS = [
    ("Bridge (4-span)", {
        "MB": [
            f"{MB}/bridge-mb_opus_span_transfer_r2/seed0",
            f"{MB}/bridge_span_transfer-mb_opus_span_transfer_r3/seed1",
            f"{MB}/bridge_span_transfer-mb_opus_span_transfer_r3/seed2"
        ],
        "MF": [f"{MF}/bridge-mf_opus_span_transfer_r2/seed0"] + [
            f"{MF}/bridge_span_transfer-mf_opus_span_transfer_r2/seed{s}"
            for s in (1, 2)
        ]
    }),
    ("Fan (maze)", {
        "MB": [f"{MB}/fan_maze-mb_opus_gate_r1/seed{s}" for s in range(3)],
        "MF": [f"{MF}/fan_maze-mf_opus_r1/seed{s}" for s in range(3)]
    }),
    ("Domino", {
        "MB": [
            f"{MB}/domino_high_friction_turn-mb_opus_gate_r1/seed{s}"
            for s in range(3)
        ],
        "MF": [
            f"{MF}/domino_high_friction_turn-mf_opus_r1/seed{s}"
            for s in range(3)
        ]
    }),
    ("Boil (2-jug)", {
        "MB": [
            f"{MB}/boil-mb_opus_gate_preflight_two_jug_tight_r1/seed{s}"
            for s in range(3)
        ],
        "MF":
        [f"{MF}/boil-mf_opus_two_jug_tight_r1/seed{s}" for s in range(3)]
    }),
    ("Balloons (composition)", {
        "MB": [f"{MB}/balloons-mb_opus_compose_r2/seed{s}" for s in range(3)],
        "MF": [f"{MF}/balloons-mf_opus_compose_r2/seed{s}" for s in range(3)]
    }),
]
for _domain, _dirs in DOMAINS:
    _dirs["MB_r2"] = [
        f"{MB}/{ENV_KEY[_domain]}-mb_opus_benchmark_r2/seed{s}" for s in (3, 4)
    ]
    for _arm, _approach in APPROACH_DIR.items():
        _round = ROUND_BY_DOMAIN.get((_arm, _domain), ROUND.get(_arm, 'r1'))
        _key = f"{ENV_KEY[_domain]}-{_arm}_opus_benchmark_{_round}"
        _dirs[_arm] = [f"{_approach}/{_key}/seed{s}" for s in range(3)]
    # Keep the repair pilot distinct from r1. Empty directories mean the
    # other domains are not launched, not three unfinished/failed seeds.
    _dirs["oracle_dynamics_r2"] = ([
        f"agent_continual_oracle_dynamics/{ENV_KEY[_domain]}-"
        f"oracle_dynamics_opus_benchmark_r2/seed{s}" for s in range(3)
    ] if _domain in ("Domino", "Bridge (4-span)") else [])
# Column order for the figure.
_ORDER = [
    "Boil (2-jug)", "Domino", "Balloons (composition)", "Bridge (4-span)",
    "Fan (maze)"
]
DOMAINS.sort(key=lambda d: _ORDER.index(d[0]))
DISPLAY_TITLE = {domain: domain.split(" (", 1)[0] for domain in _ORDER}
# Arms shown de-emphasised (grey bars and labels).
GREYED = {"mb_scene_package", "scene_only", "zero_shot", "real_to_sim"}
GREY = "#c3c9cd"
# Pale tints of each greyed arm's own hue, so they stay distinguishable.
MUTED = {
    "mb_scene_package": "#b6cbd4",
    "scene_only": "#cdbfae",
    "zero_shot": "#abc0d6",
    "real_to_sim": "#dbc3bb",
}

Row = Dict[str, Any]


def colour(arm: str, i: int) -> str:
    """The arm's bar and line colour."""
    return MUTED[arm] if arm in GREYED else COLORS[i]


RUN_PINS = {"balloons-mb_opus_compose_r2/seed2": "run_20260917_082021"}


def _card(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "scorecard.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _finished(card: Optional[Dict[str, Any]]) -> bool:
    return card is not None and card.get("end_reason") is not None


def pick(seed_dir: str) -> Optional[str]:
    """The run directory a seed reports: its newest finished run, else its
    newest run."""
    runs = sorted(glob.glob(f"{LOGS}/{seed_dir}/run_*"))
    for key, name in RUN_PINS.items():
        if key in seed_dir:
            runs = [r for r in runs if r.endswith(name)] or runs
    done = [r for r in runs if _finished(_card(r))]
    return done[-1] if done else (runs[-1] if runs else None)


def records() -> List[Row]:
    """One row per finished seed of every arm and domain."""
    rows = []
    for domain, arm_dirs in DOMAINS:
        for arm, dirs in arm_dirs.items():
            for seed, sd in enumerate(dirs):
                r = pick(sd)
                card = _card(r) if r else None
                if card is None or not _finished(card):
                    continue
                rows.append(
                    dict(domain=domain,
                         arm=arm,
                         seed=seed,
                         won=sum(bool(lv.get("won")) for lv in card["levels"]),
                         levels=len(card["levels"]),
                         steps=card["totals"]["total_steps"],
                         resets=card["totals"]["total_resets"],
                         source=r))
    return rows


def _solve_curves(ax: Any, rows: List[Row], domain: str, col: int) -> None:
    """Fraction of finished runs that solved the domain within each step
    budget: a step up at every solved run's total steps, flat after the
    last one, so the plateau height is the solve rate."""
    solved_steps = [
        r["steps"] for r in rows
        if r["domain"] == domain and r["won"] == r["levels"]
    ]
    right = max(solved_steps, default=1) * 1.12
    # Greyed arms first, so the highlighted arms draw on top.
    order = sorted(range(len(ARMS)), key=lambda i: ARMS[i] not in GREYED)
    for i in order:
        arm = ARMS[i]
        group = [r for r in rows if r["domain"] == domain and r["arm"] == arm]
        if not group:
            continue
        steps = sorted(r["steps"] for r in group if r["won"] == r["levels"])
        xs = [0] + [x for x in steps for _ in (0, 1)] + [right]
        counts = [0] + [v for k in range(len(steps))
                        for v in (k, k + 1)] + [len(steps)]
        ys = [100 * y / len(group) for y in counts]
        greyed = arm in GREYED
        ax.plot(xs,
                ys,
                color=colour(arm, i),
                lw=1.0 if greyed else 2.0,
                zorder=1 if greyed else 3,
                label=LABELS[i],
                solid_joinstyle="miter")
        ax.plot(right,
                ys[-1],
                "o",
                ms=3.5,
                color=colour(arm, i),
                zorder=1 if greyed else 3,
                clip_on=False)
    ax.set_xlim(0, right)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 50, 100])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#aebec5")
    ax.tick_params(length=0, pad=4, labelsize=10)
    ax.grid(color="#dce4e7", lw=.6)
    ax.set_xlabel("Step budget", fontsize=11)
    if col == 0:
        ax.set_ylabel("Runs solved (%)", fontsize=11)


def render(rows: List[Row], output: str) -> None:
    """Draw the figure to <output>.png/.pdf and its summary json."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42
    })
    columns = len(DOMAINS)
    fig, axes = plt.subplots(3,
                             columns,
                             figsize=(3.5 + 2 * columns,
                                      11.5 if len(ARMS) > 8 else 6.8),
                             sharey="row",
                             squeeze=False)
    summary = []
    for col, (domain, _) in enumerate(DOMAINS):
        for metric, field in enumerate(["solve", "steps", "resets"]):
            ax = axes[metric, col]
            if field == "steps":
                _solve_curves(ax, rows, domain, col)
                continue
            for i, arm in enumerate(ARMS):
                group = [
                    r for r in rows
                    if r["domain"] == domain and r["arm"] == arm
                ]
                eligible = ([r for r in group if r["won"] == r["levels"]]
                            if field == "steps" else group)
                vals = [
                    100 * int(r["won"] == r["levels"])
                    if field == "solve" else r[field] for r in eligible
                ]
                avg = float(np.mean(vals)) if vals else None
                summary.append(
                    dict(domain=domain,
                         arm=arm,
                         metric=field,
                         mean=avg,
                         n=len(vals),
                         values=vals))
                if vals:
                    ax.barh(i, avg, height=.56, zorder=2, color=colour(arm, i))
                    jitter = np.linspace(-.15, .15,
                                         len(vals)) if len(vals) > 1 else [0]
                    ax.scatter(
                        vals,
                        i + np.asarray(jitter),
                        s=12,
                        facecolors="white",
                        edgecolors=MUTED[arm] if arm in GREYED else "#263c46",
                        linewidths=.6,
                        zorder=3,
                        clip_on=False)
                if field == "steps":
                    ax.text(.98,
                            i,
                            f"n={len(vals)}" if vals else "no success",
                            transform=ax.get_yaxis_transform(),
                            ha="right",
                            va="center",
                            fontsize=7,
                            color="#344a55",
                            bbox=dict(facecolor="white",
                                      edgecolor="none",
                                      pad=.6))
            ax.set_ylim(len(ARMS) - .4, -.7)
            for group_index, (_, start, end) in enumerate(GROUPS):
                if group_index % 2 == 0:
                    ax.axhspan(start - .5, end - .5, color="#f2f4f5", zorder=0)
            ax.set_yticks(range(len(ARMS)), LABELS, fontsize=10)
            for tick, arm in zip(ax.get_yticklabels(), ARMS):
                if arm in ("MB", "MB_r2"):
                    tick.set_fontweight("bold")
                if arm in GREYED:
                    tick.set_color("#9aa3a8")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#aebec5")
            ax.tick_params(length=0, pad=4, labelsize=10)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#dce4e7", lw=.6)
            if field == "solve":
                ax.set_xlim(-3, 108)
                ax.set_xticks([0, 50, 100])
                ax.set_title(DISPLAY_TITLE[domain],
                             fontsize=12,
                             fontweight="bold",
                             color="#203744",
                             pad=12)
                ax.set_xlabel("Successful runs (%)", fontsize=11)
            elif field == "steps":
                maximum = max(
                    (r["steps"] for r in rows
                     if r["domain"] == domain and r["won"] == r["levels"]),
                    default=1)
                ax.set_xlim(0, maximum * 1.30)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f"{x/1000:g}k"
                                  if x >= 1000 else f"{x:g}"))
                ax.set_xlabel("Successful-run steps", fontsize=11)
            else:
                ax.set_xlim(left=-.05, right=max(1, ax.get_xlim()[1]))
                ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
                ax.set_xlabel("Resets (all runs)", fontsize=11)
    fig.subplots_adjust(left=.11,
                        right=.99,
                        top=.95,
                        bottom=.05,
                        wspace=.23,
                        hspace=.45)
    handles = []
    legend_labels = []
    legend_arms = []
    for _, start, end in GROUPS:
        for i in range(start, end):
            arm = ARMS[i]
            handles.append(
                plt.Line2D([], [],
                           color=colour(arm, i),
                           lw=1.0 if arm in GREYED else 2.0))
            legend_labels.append(LABELS[i])
            legend_arms.append(arm)
    leg = fig.legend(
        handles,
        legend_labels,
        loc="center right",
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(axes[1, 0].get_position().x0 - .7 / fig.get_figwidth(),
                        sum(axes[1, 0].get_position().intervaly) / 2))
    for text, arm in zip(leg.get_texts(), legend_arms):
        if arm in ("MB", "MB_r2"):
            text.set_fontweight("bold")
        if arm in GREYED:
            text.set_color("#9aa3a8")
    for ext in ("png", "pdf"):
        fig.savefig(f"{output}.{ext}",
                    dpi=180,
                    bbox_inches="tight",
                    pad_inches=.06)
    plt.close(fig)
    with open(f"{output}-summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(records=rows, summary=summary), f, indent=2)


def main() -> None:
    """Print every finished seed and draw the figure."""
    found = records()
    for row in found:
        print(row["domain"], row["arm"], f"s{row['seed']}",
              f"{row['won']}/{row['levels']}", row["steps"], "steps",
              row["resets"], "resets")
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmark_arms_opus"
    render(found, out)
    print("saved", out + ".png")


if __name__ == "__main__":
    main()
