"""Opus arms on the five benchmark settings plus a Fan transfer pilot.

In the style
of the paper's fig3_results: three metric rows (solve rate, successful-run
steps, resets) by five domain columns, bar = mean, dots = seeds.  Solve =
fraction of finished runs winning every level; steps over whole-run
successes only; resets over all finished runs.  Reads the scorecard.json of
the run picked per seed (finished runs preferred; pinned replays excluded).
Arms with no finished run in a domain draw nothing there.

Usage: python scripts/plotting/plot_benchmark_arms.py <output stem>
Add --paper after the output stem for the seven-arm, two-row paper view,
using repaired Oracle r2 cohorts in Domino and Bridge only. Add
--records=<summary.json> to redraw an archived selection, such as the
paper's, instead of scanning the logs.
writes <stem>.png, <stem>.pdf and <stem>-summary.json (per-seed records
with the run directory each one read, plus the per-arm summary).
"""
import glob
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
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
    "oracle_dynamics", "MB", "MF", "mf_scene_package", "standalone",
    "no_fitting", "no_uncertainty", "mb_scene_package", "scene_only",
    "zero_shot", "real_to_sim", "from_assets"
]
LABELS = [
    "Oracle dynamics", "EMPIRIC", "Direct agent",
    "Direct agent + scene assets", "Standalone sim.", "No harness fitting",
    "No explicit uncertainty", "EMPIRIC + scene pkg.", "Scene only",
    "Zero-shot model", "Agentic real-to-sim", "EMPIRIC from assets"
]
COLORS = [
    "#88929d", "#087f8c", "#bd5929", "#7a3312", "#7467a6", "#b19658",
    "#397957", "#0b4f6c", "#6b9483", "#5588ad", "#a5573f", "#c04483"
]
GROUPS = [
    ("Oracle reference", 0, 1),
    ("Methods", 1, 5),
    ("EMPIRIC ablations", 5, 7),
    ("Additional comparisons", 7, 12),
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
            for s in range(2)
        ] + [f"{MB}/boil-mb_opus_benchmark_r2/seed2"],
        "MF":
        [f"{MF}/boil-mf_opus_two_jug_tight_r1/seed{s}" for s in range(3)]
    }),
    ("Balloons (composition)", {
        "MB":
        [f"{MB}/balloons-mb_opus_compose_r2/seed{s}"
         for s in range(2)] + [f"{MB}/balloons-mb_opus_benchmark_r2/seed2"],
        "MF": [f"{MF}/balloons-mf_opus_compose_r2/seed{s}" for s in range(3)]
    }),
]
for _domain, _dirs in DOMAINS:
    _dirs["MB"] += [
        f"{MB}/{ENV_KEY[_domain]}-mb_opus_benchmark_r2/seed{s}" for s in (3, 4)
    ]
    for _arm, _approach in APPROACH_DIR.items():
        _round = ROUND_BY_DOMAIN.get((_arm, _domain), ROUND.get(_arm, 'r1'))
        _key = f"{ENV_KEY[_domain]}-{_arm}_opus_benchmark_{_round}"
        _dirs[_arm] = [f"{_approach}/{_key}/seed{s}" for s in range(3)]
    # Preserve historical rows, except for explicitly replaced EMPIRIC seed
    # 2 runs in Boil and Balloons, and add prospective seeds 3 and 4.
    _dirs["MF"] += [
        f"{MF}/{ENV_KEY[_domain]}-mf_opus_benchmark_r2/seed{s}" for s in (3, 4)
    ]
    for _arm in ("mf_scene_package", "standalone", "no_fitting",
                 "no_uncertainty"):
        _round = ROUND.get(_arm, "r1")
        _key = f"{ENV_KEY[_domain]}-{_arm}_opus_benchmark_{_round}"
        _dirs[_arm] += [f"{APPROACH_DIR[_arm]}/{_key}/seed{s}" for s in (3, 4)]
    # r2 contains the original six repair pilots plus ten new seeds.
    _dirs["oracle_dynamics_r2"] = ([
        f"agent_continual_oracle_dynamics/{ENV_KEY[_domain]}-"
        f"oracle_dynamics_opus_benchmark_r2/seed{s}"
        for s in (range(5) if _domain in ("Domino",
                                          "Bridge (4-span)") else (3, 4))
    ])
    _oracle_r2 = _dirs.pop("oracle_dynamics_r2")
    if _domain in ("Domino", "Bridge (4-span)"):
        _dirs["oracle_dynamics"] = _oracle_r2
    else:
        _dirs["oracle_dynamics"] += _oracle_r2
# Column order for the figure.
_ORDER = [
    "Domino", "Bridge (4-span)", "Balloons (composition)", "Boil (2-jug)",
    "Fan (maze)"
]
DOMAINS.sort(key=lambda d: _ORDER.index(d[0]))
DISPLAY_TITLE = {domain: domain.split(" (", 1)[0] for domain in _ORDER}
# Keep exploratory columns out of both paper records and paper rendering.
PAPER_DOMAINS = tuple(_ORDER[:-1] + ["Fan (ramp transfer)"])
FAN_TRANSFER = "Fan (exposed transfer)"
_transfer_dirs: Dict[str, List[str]] = {arm: [] for arm in ARMS}
for _arm, _approach, _round in (("MB", MB, "mb"), ("MF", MF, "mf")):
    _transfer_dirs[_arm] = [
        f"{_approach}/fan_transfer-{_round}_opus_transfer_pilot_r1/seed{s}"
        for s in (0, 1)
    ]
DOMAINS.append((FAN_TRANSFER, _transfer_dirs))
DISPLAY_TITLE[FAN_TRANSFER] = "Fan transfer"
FAN_VARIANTS = ("Fan (inertial transfer)", "Fan (ramp transfer)")
for _variant, _title in zip(("inertial", "ramp"), FAN_VARIANTS):
    _variant_dirs: Dict[str, List[str]] = {arm: [] for arm in ARMS}
    for _arm, _approach, _round in (("MB", MB, "mb"), ("MF", MF, "mf")):
        _variant_dirs[_arm] = [
            f"{_approach}/fan_{_variant}-{_round}_opus_{_variant}_"
            f"{'pilot' if s < 2 else 'confirmation'}_r1/seed{s}"
            for s in range(5)
        ]
    DOMAINS.append((_title, _variant_dirs))
    DISPLAY_TITLE[_title] = "Fan" if _variant == "ramp" else "Fan inertial"
    if _variant == "inertial":
        for _arm, _approach in (("oracle_dynamics",
                                 "agent_continual_oracle_dynamics"),
                                ("mf_scene_package",
                                 "agent_continual_model_free"),
                                ("standalone",
                                 "agent_continual_program_world_model"),
                                ("no_fitting", "agent_continual_no_fitting"),
                                ("no_uncertainty",
                                 "agent_continual_no_uncertainty")):
            _variant_dirs[_arm] = [
                f"{_approach}/fan_inertial-{_arm}_opus_inertial_r1/seed{s}"
                for s in range(5)
            ]
# Use only the matched repaired-skill cohort for the current ramp column.
# Pending seeds must not fall back to the earlier geometry/skill experiments.
_ramp_dirs = dict(DOMAINS)["Fan (ramp transfer)"]
for _arm, _approach, _round in (
    ("MB", MB, "mb"),
    ("MF", MF, "mf"),
    ("mf_scene_package", MF, "mf_scene_package"),
    ("standalone", "agent_continual_program_world_model", "standalone"),
    ("no_fitting", "agent_continual_no_fitting", "no_fitting"),
    ("no_uncertainty", "agent_continual_no_uncertainty", "no_uncertainty"),
):
    _ramp_dirs[_arm] = [
        f"{_approach}/fan_ramp-{_round}_opus_ramp_skill_repair_r1/seed{s}"
        for s in range(5)
    ]
# Oracle Fan now uses only the fresh prompt-aligned cohort.  Failed startup
# attempts and every earlier ramp cohort remain archived but are not pooled.
_ramp_dirs["oracle_dynamics"] = [
    "agent_continual_oracle_dynamics/"
    f"fan-oracle_dynamics_opus_fan_prompt_r2/seed{s}" for s in range(5)
]
# Assets-only EMPIRIC is a separate two-seed development arm.
# Exclude its cancelled maze runs and never substitute them for ramp data.
for _domain, _dirs in DOMAINS:
    _asset_key = ("fan_ramp" if _domain == "Fan (ramp transfer)" else
                  ENV_KEY.get(_domain))
    _dirs["from_assets"] = ([
        f"agent_continual_from_assets/{_asset_key}-"
        f"from_assets_opus_pilot_r1/seed{s}" for s in range(2)
    ] if _asset_key is not None and _domain != "Fan (maze)" else [])

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


class Fonts(NamedTuple):
    """Text sizes, in points on the figure canvas."""
    tick: float
    label: float
    title: float
    legend: float


SCREEN_FONTS = Fonts(tick=10, label=11, title=12, legend=10)
# The paper prints its 15 in wide canvas at the 5.5 in text width. These
# printed sizes match the text of Figures 1 to 3: 5.5 pt ticks, 6 pt labels
# and legend, 7.2 pt domain titles.
PAPER_FONTS = Fonts(*(size * 15.0 / 5.5 for size in (5.5, 6.0, 7.2, 6.0)))


def _longhand_fonts(svg: str) -> str:
    """Spell out the CSS font shorthand in matplotlib's SVG text, which Figma's
    SVG import ignores, as separate weight, size and family."""

    def expand(match: "re.Match[str]") -> str:
        weight, size, family = match.groups()
        parts = [f"font-family: {family}", f"font-size: {size}"]
        if weight:
            parts.append(f"font-weight: {weight}")
        return "; ".join(parts)

    return re.sub(r"font: (?:(\d+) )?([\d.]+px) ('[^']+')", expand, svg)


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
            for sd in dirs:
                seed = int(Path(sd).name.removeprefix("seed"))
                r = pick(sd)
                card = _card(r) if r else None
                if card is None or not _finished(card):
                    continue
                rows.append(
                    dict(domain=domain,
                         arm=arm,
                         source_arm=(
                             "MB_r2" if arm == "MB"
                             and "-mb_opus_benchmark_r2/" in sd else
                             "oracle_dynamics_r2" if arm == "oracle_dynamics"
                             and "benchmark_r2/" in sd else arm),
                         seed=seed,
                         won=sum(bool(lv.get("won")) for lv in card["levels"]),
                         levels=len(card["levels"]),
                         steps=card["totals"]["total_steps"],
                         resets=card["totals"]["total_resets"],
                         source=r))
    return rows


def paper_records(rows: List[Row]) -> List[Row]:
    """Select current paper cohorts, preserving every row's provenance."""
    selected = []
    for row in rows:
        if row["domain"] not in PAPER_DOMAINS:
            continue
        arm = row["arm"]
        if arm == "oracle_dynamics_r2":
            arm = "oracle_dynamics"
        if arm == "MB_r2":
            arm = "MB"
        if arm in PAPER_ARMS:
            selected.append(
                dict(row,
                     arm=arm,
                     source_arm=row.get("source_arm", row["arm"])))
    return selected


PAPER_ARMS = [
    "oracle_dynamics", "MB", "MF", "mf_scene_package", "standalone",
    "no_fitting", "no_uncertainty"
]


def _solve_curves(ax: Any,
                  rows: List[Row],
                  domain: str,
                  col: int,
                  arms: Optional[List[str]] = None,
                  curve_style: str = "default",
                  fonts: Fonts = SCREEN_FONTS) -> None:
    """Fraction of finished runs that solved the domain within each step
    budget: a step up at every solved run's total steps, flat after the
    last one, so the plateau height is the solve rate."""
    solved_steps = [
        r["steps"] for r in rows
        if r["domain"] == domain and r["won"] == r["levels"]
    ]
    right = max(solved_steps, default=1) * 1.12
    shown_arms = arms or ARMS
    if curve_style == "strips":
        gutter_left = right * 1.04
        gutter_center = right * 1.16
        plot_right = right * 1.28
        ax.axvspan(gutter_left, plot_right, color="#f3f5f6", zorder=-1)
        ax.axvline(gutter_left, color="#aebec5", lw=.8, zorder=1)
        for i, arm in enumerate(shown_arms):
            group = [
                r for r in rows if r["domain"] == domain and r["arm"] == arm
            ]
            steps = [r["steps"] for r in group if r["won"] == r["levels"]]
            failures = [r for r in group if r["won"] != r["levels"]]
            ax.hlines(i, 0, right, color="#e1e7e9", lw=.8, zorder=0)
            if steps:
                jitter = (np.linspace(-.13, .13, len(steps))
                          if len(steps) > 1 else [0])
                ax.scatter(steps,
                           i + np.asarray(jitter),
                           s=28,
                           marker="o",
                           color=colour(arm, ARMS.index(arm)),
                           edgecolor="white",
                           linewidth=.6,
                           zorder=3)
            if failures:
                failure_xs = (np.linspace(gutter_center -
                                          right * .035, gutter_center +
                                          right * .035, len(failures))
                              if len(failures) > 1 else [gutter_center])
                ax.scatter(failure_xs,
                           np.full(len(failures), i),
                           s=30,
                           marker="x",
                           color=colour(arm, ARMS.index(arm)),
                           linewidth=1.4,
                           zorder=3)
        ax.text(gutter_center,
                1.015,
                "Unsolved",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#52616b")
        ax.set_xlim(0, plot_right)
        ax.set_ylim(len(shown_arms) - .45, -.55)
        ax.set_yticks([])
        tick_locator = MaxNLocator(3)
        ax.set_xticks([
            tick for tick in tick_locator.tick_values(0, right)
            if 0 <= tick <= right
        ])
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x/1000:g}k"
                          if x >= 1000 else f"{x:g}"))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#aebec5")
        ax.tick_params(length=0, pad=4, labelsize=fonts.tick)
        ax.grid(axis="x", color="#dce4e7", lw=.6)
        ax.set_xlabel("Successful-run steps", fontsize=fonts.label)
        if col == 0:
            ax.set_ylabel("Successful seeds", fontsize=fonts.label)
        return
    # Paint in reverse legend order so the primary methods remain visible
    # where curves overlap. EMPIRIC is painted after the
    # Oracle reference without changing their displayed legend or bar order.
    draw_arms = list(reversed(shown_arms))
    if "MB" in draw_arms:
        draw_arms.remove("MB")
        draw_arms.append("MB")
    order = [ARMS.index(a) for a in draw_arms]
    offsets = dict(zip(shown_arms, np.linspace(-4.5, 4.5, len(shown_arms))))
    line_styles = [
        "-", (0, (6, 2)), (0, (1, 1)), (0, (4, 1, 1, 1)), (0, (7, 2, 1, 2)),
        (0, (2, 2)), (0, (5, 2, 2, 2))
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*"]
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
        transform = ax.transData
        if curve_style == "offset":
            transform += mtransforms.ScaledTranslation(
                0, offsets[arm] / 72, ax.figure.dpi_scale_trans)
        linestyle = (line_styles[shown_arms.index(arm)]
                     if curve_style == "dashes" else "-")
        ax.plot(xs,
                ys,
                color=colour(arm, i),
                lw=1.0 if greyed else 2.0,
                linestyle=linestyle,
                transform=transform,
                zorder=1 if greyed else 3,
                label=LABELS[i],
                solid_joinstyle="miter")
        ax.plot(right,
                ys[-1],
                "o",
                ms=3.5,
                color=colour(arm, i),
                transform=transform,
                zorder=1 if greyed else 3,
                clip_on=False)
        if curve_style == "markers" and steps:
            ax.scatter(steps,
                       [100 * (k + 1) / len(group) for k in range(len(steps))],
                       s=22,
                       marker=markers[shown_arms.index(arm)],
                       facecolor="white",
                       edgecolor=colour(arm, i),
                       linewidth=1.0,
                       zorder=1 if greyed else 4,
                       clip_on=False)
    ax.set_xlim(0, right)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 50, 100])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1000:g}k" if x >= 1000 else f"{x:g}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#aebec5")
    ax.tick_params(length=0, pad=4, labelsize=fonts.tick)
    ax.grid(color="#dce4e7", lw=.6)
    ax.set_xlabel("Step budget", fontsize=fonts.label)
    if col == 0:
        ax.set_ylabel("Runs solved (%)", fontsize=fonts.label)


def _solve_columns(ax: Any, count: int, groups: List[Tuple[str, int, int]],
                   fonts: Fonts) -> None:
    """Style the paper's solve-rate columns on the solve curves' scale, so a
    column's height matches where its curve ends."""
    ax.set_xlim(-.6, count - .4)
    for group_index, (_, start, end) in enumerate(groups):
        if group_index % 2 == 0:
            ax.axvspan(start - .5, end - .5, color="#f2f4f5", zorder=0)
    ax.axhline(0, color="#aebec5", lw=.8, zorder=1)
    ax.set_xticks([])
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 50, 100])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#aebec5")
    ax.tick_params(length=0, pad=4, labelsize=fonts.tick)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce4e7", lw=.6)


def render(rows: List[Row],
           output: str,
           paper: bool = False,
           curve_style: str = "markers") -> None:
    """Draw the figure to <output>.png/.pdf and its summary json."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        # The paper view's SVG, the source of the Figma copy, keeps its text
        # editable and its ids fixed.
        "svg.fonttype": "none",
        "svg.hashsalt": "paper-results",
    })
    fonts = PAPER_FONTS if paper else SCREEN_FONTS
    arms = PAPER_ARMS if paper else ARMS
    labels = [LABELS[ARMS.index(a)] for a in arms]
    if paper:
        labels[arms.index("mf_scene_package")] = "Direct + scene"
    groups = [("Oracle", 0, 1), ("Methods", 1, 5),
              ("Ablations", 5, 7)] if paper else GROUPS
    fields = ["solve", "steps"] if paper else ["solve", "steps", "resets"]
    # Preserve historical variants in records, never pool them into Fan.
    domains = sorted((item for item in DOMAINS if item[0] in PAPER_DOMAINS),
                     key=lambda item: PAPER_DOMAINS.index(item[0]))
    columns = len(domains)
    fig, axes = plt.subplots(len(fields),
                             columns,
                             figsize=(15.0 if paper else 3.0 * columns,
                                      5.4 if paper else 11.5),
                             sharey="row",
                             squeeze=False)
    summary = []
    for col, (domain, _) in enumerate(domains):
        for metric, field in enumerate(fields):
            ax = axes[metric, col]
            if not any(r["domain"] == domain for r in rows):
                ax.text(.5,
                        .5,
                        "Awaiting results",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        color="#647681",
                        fontsize=9,
                        zorder=5)
            if field == "steps":
                _solve_curves(ax, rows, domain, col, arms, curve_style, fonts)
                if not any(r["domain"] == domain for r in rows):
                    ax.set_xticks([])
                continue
            for i, arm in enumerate(arms):
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
                    assert avg is not None
                    if paper and field == "solve":
                        # The paper stands the solve rates up as columns, so
                        # they share the solve curves' vertical axis. Each
                        # seed solves or fails, so it shows no seed dots.
                        ax.bar(i,
                               avg,
                               width=.56,
                               zorder=2,
                               color=colour(arm, ARMS.index(arm)))
                        if avg == 0:
                            # A zero column would vanish; a stub on the
                            # baseline shows that the agent solved no run.
                            ax.hlines(0,
                                      i - .28,
                                      i + .28,
                                      color=colour(arm, ARMS.index(arm)),
                                      lw=2.4,
                                      zorder=3)
                    else:
                        ax.barh(i,
                                avg,
                                height=.56,
                                zorder=2,
                                color=colour(arm, ARMS.index(arm)))
                        jitter = (np.linspace(-.15, .15, len(vals))
                                  if len(vals) > 1 else [0])
                        ax.scatter(vals,
                                   i + np.asarray(jitter),
                                   s=12,
                                   facecolors="white",
                                   edgecolors=MUTED[arm]
                                   if arm in GREYED else "#263c46",
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
            if paper and field == "solve":
                _solve_columns(ax, len(arms), groups, fonts)
                ax.set_title(DISPLAY_TITLE[domain],
                             fontsize=fonts.title,
                             fontweight="bold",
                             color="#203744",
                             pad=6)
                if col == 0:
                    ax.set_ylabel("Runs solved (%)", fontsize=fonts.label)
                continue
            ax.set_ylim(len(arms) - .4, -.7)
            for group_index, (_, start, end) in enumerate(groups):
                if group_index % 2 == 0:
                    ax.axhspan(start - .5, end - .5, color="#f2f4f5", zorder=0)
            ax.set_yticks(range(len(arms)), labels, fontsize=fonts.tick)
            ax.tick_params(axis="y", labelleft=False)
            for tick, arm in zip(ax.get_yticklabels(), arms):
                if arm in ("MB", "MB_r2"):
                    tick.set_fontweight("bold")
                if arm in GREYED:
                    tick.set_color("#9aa3a8")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#aebec5")
            ax.tick_params(length=0, pad=4, labelsize=fonts.tick)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#dce4e7", lw=.6)
            if field == "solve":
                ax.set_xlim((0, 100) if paper else (-3, 108))
                ax.set_xticks([0, 50, 100])
                ax.set_title(DISPLAY_TITLE[domain],
                             fontsize=fonts.title,
                             fontweight="bold",
                             color="#203744",
                             pad=6 if paper else 12)
                ax.set_xlabel("Runs solved (%)", fontsize=fonts.label)
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
                ax.set_xlabel("Successful-run steps", fontsize=fonts.label)
            else:
                ax.set_xlim(left=-.05, right=max(1, ax.get_xlim()[1]))
                ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
                ax.set_xlabel("Resets (all runs)", fontsize=fonts.label)
    # The paper's printed text is larger relative to the canvas, so it needs
    # wider margins and two legend rows.
    # In the paper the plots, with their labels, are centred on the canvas,
    # like the legend above them.
    fig.subplots_adjust(left=.0605 if paper else .06,
                        right=.9755 if paper else .99,
                        top=.78 if paper else .89,
                        bottom=.13 if paper else .05,
                        wspace=.23,
                        hspace=.3 if paper else .45)
    handles = []
    legend_labels = []
    legend_arms = []
    for _, start, end in groups:
        for i in range(start, end):
            arm = arms[i]
            handles.append(
                plt.Line2D([], [],
                           color=colour(arm, ARMS.index(arm)),
                           lw=8.0 if paper else 5.0))
            legend_labels.append(labels[i])
            legend_arms.append(arm)
    if paper:
        # One centred legend per row, four agents over three, so the shorter
        # second row sits under the middle of the first.
        legend_rows = [list(range(4)), list(range(4, len(arms)))]
    else:
        # One legend; matplotlib fills its columns first, so this order reads
        # by row.
        legend_rows = [[
            i for col in range(6) for i in range(col, len(arms), 6)
        ]]
    top = .99 if paper else .985
    for entries in legend_rows:
        leg = fig.legend([handles[i] for i in entries],
                         [legend_labels[i] for i in entries],
                         loc="upper center",
                         ncol=len(entries) if paper else 6,
                         borderpad=0 if paper else .4,
                         borderaxespad=0 if paper else .5,
                         columnspacing=2.0,
                         handlelength=2.2 if paper else 2.0,
                         handletextpad=0.65 if paper else 0.8,
                         fontsize=fonts.legend,
                         frameon=False,
                         bbox_to_anchor=(.5, top))
        for text, i in zip(leg.get_texts(), entries):
            if legend_arms[i] in ("MB", "MB_r2"):
                text.set_fontweight("bold")
            if legend_arms[i] in GREYED:
                text.set_color("#9aa3a8")
        # The next row starts one line space below this one, as within a
        # legend.
        box = leg.get_window_extent(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        top = box.y0 - leg.labelspacing * fonts.legend / (72 *
                                                          fig.get_figheight())
    # Without creation dates, re-exporting an unchanged figure writes the
    # same bytes.
    undated = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}
    for ext in ("png", "pdf", "svg") if paper else ("png", "pdf"):
        target: Any = io.StringIO() if ext == "svg" else f"{output}.{ext}"
        fig.savefig(target,
                    format=ext,
                    dpi=180,
                    bbox_inches=None,
                    pad_inches=0 if paper else .06,
                    metadata=undated.get(ext))
        if ext == "svg":
            Path(f"{output}.svg").write_text(_longhand_fonts(
                target.getvalue()),
                                             encoding="utf-8")
    plt.close(fig)
    with open(f"{output}-summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(records=rows, summary=summary), f, indent=2)


def main() -> None:
    """Print every finished seed and draw the figure."""
    archive = next((arg.removeprefix("--records=")
                    for arg in sys.argv[2:] if arg.startswith("--records=")),
                   None)
    if archive is None:
        found = records()
    else:
        with open(archive, encoding="utf-8") as f:
            found = json.load(f)["records"]
    for row in found:
        print(row["domain"], row["arm"], f"s{row['seed']}",
              f"{row['won']}/{row['levels']}", row["steps"], "steps",
              row["resets"], "resets")
    out = sys.argv[1] if len(sys.argv) > 1 else "benchmark_arms_opus"
    paper = "--paper" in sys.argv[2:]
    curve_style = next(
        (arg.removeprefix("--curve-style=")
         for arg in sys.argv[2:] if arg.startswith("--curve-style=")),
        "markers")
    if curve_style not in {"default", "offset", "dashes", "markers", "strips"}:
        raise ValueError(f"Unknown curve style: {curve_style}")
    # An archived selection is drawn as recorded.
    render(paper_records(found) if paper and archive is None else found,
           out,
           paper=paper,
           curve_style=curve_style)
    if archive is not None:
        with open(f"{out}-summary.json", encoding="utf-8") as f:
            assert json.load(f)["records"] == found, (
                "the redraw changed the archived records")
    print("saved", out + ".png")


if __name__ == "__main__":
    main()
