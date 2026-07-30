"""Diagram of the physical system-identification pipeline.

One box per pipeline stage, grouped into four lanes (data collection,
fit orchestration, uncertainty accounting, consumers), each annotated
with the module that implements it. Red badges mark the failure modes
observed in run_20260724_232411 (seeds 1-2: fits 1.0358 / 0.3236 vs
true lateral_friction 0.5), with the planned stage-1 remedies listed in
the legend. Regenerate with:

    PYTHONPATH=. python docs/sysid/make_sysid_pipeline_fig.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "sysid_pipeline.png"

LANE_FILL = "#f4f4f4"
BOX_FILL = "#ffffff"
BOX_EDGE = "#555555"
HEADER_COLORS = ["#3d6b99", "#3d8a5f", "#8a6d3d", "#7a4f8a"]
FLAG_COLOR = "#c0392b"

LANES = [
    (12.0, "DATA (real env, true params)"),
    (37.0, "FIT  (code_sim_learning)"),
    (62.0, "UNCERTAINTY"),
    (87.0, "CONSUMERS"),
]
LANE_W = 23.0

# (id, lane, y_center, height, title, body, flag)
BOXES = [
    ("A1", 0, 46.0, 9.0, "Explore episodes",
     "agent_bilevel explorer, 2/cycle\n"
     "-> LowLevelTrajectory\n"
     "poses @ 20 Hz, no velocities", None),
    ("A2", 0, 31.0, 11.0, "Learn session artifact", "sandbox/simulator.py:\n"
     "PHYSICAL_PARAMS (ParamSpec,\n"
     "log scale) + PROCESS_FEATURES\n"
     "+ rules and rule params", None),
    ("B1", 1, 48.5, 7.0, "Trajectory prep", "settled-tail truncation\n"
     "-> rest-point segmentation\n"
     "-> residual scaling  [trajectory_prep]", None),
    ("B2", 1, 39.0, 8.0, "Rollout objective", "free-run replay per theta,\n"
     "fresh env per rollout  [rollout_env]\n"
     "per-step scaled SSE  [rollout_objective]",
     "1  chaotic through cascades"),
    ("B3", 1, 28.5, 8.5, "Search", "explainability trim -> per-param\n"
     "grid + flat-edge refine  [grid_seed]\n"
     "-> joint LM MAP, log space,\n"
     "Gaussian prior  [lm]", None),
    ("B4", 1, 20.0, 5.0, "Anchor ablation", "revert compensatory moves", None),
    ("B5", 1, 12.5, 7.0, "Consistency loop",
     "per-segment refits; on disagreement\n"
     "DROP least trustworthy + refit", "2  discards informative data"),
    ("C1", 2, 46.5, 8.0, "Laplace posterior", "posterior_std per param,\n"
     "floored at 0.1 (log space)", "3  floor hides observed scatter"),
    ("C2", 2, 35.5, 9.0, "Identifiability report",
     "posterior/prior contraction:\n"
     "identified / weakly / NOT /\n"
     "INCONSISTENT across cycles", "4  INCONSISTENT disarms gate"),
    ("C3", 2, 26.0, 7.0, "Trustworthy selection", "select_trustworthy_params\n"
     "-> applied dict (physical only)", None),
    ("C4", 2, 16.5, 7.5, "Margin grid", "physics_sigma_points: 32-point\n"
     "+-1 sigma fit-space grid\n"
     "-> ctx.physics_margin_provider", None),
    ("D1", 3, 47.5, 7.0, "Belief env", "applied to base env for planning;\n"
     "fresh validation envs re-apply", None),
    ("D2", 3, 36.5, 10.0, "Capture gate", "evaluate_option_plan: parse ->\n"
     "legitimacy -> 3x/6x decorrelated\n"
     "validation -> 32-pt margin sweep\n"
     "-> PARAM-SENSITIVE on failure", None),
    ("D3", 3, 27.0, 6.0, "Agent pre-check",
     "sim.run(plan, physics_sweep=True)\n"
     "same grid, one rollout per point", None),
    ("D4", 3, 18.0, 7.0, "Exploration ensemble",
     "6 Laplace members -> explorer\n"
     "info-seeking disagreement", None),
]

ARROWS = [
    ("A1", "B1"),
    ("A2", "B2"),
    ("B1", "B2"),
    ("B2", "B3"),
    ("B3", "B4"),
    ("B4", "B5"),
    ("B5", "C1"),
    ("C1", "C2"),
    ("C2", "C3"),
    ("C3", "C4"),
    ("C3", "D1"),
    ("C4", "D2"),
    ("C4", "D3"),
    ("C1", "D4"),
]

LEGEND = (
    "Red badges: failure modes observed in run_20260724_232411 "
    "(true friction 0.5; seed1 fit 1.0358, seed2 fits 0.3236 -> 0.6267, "
    "both 0/1 on the test task; seed0 fit 0.5033 solved 1/1).\n"
    "Planned stage-1 remedies:  [1] Huber-capped residuals + "
    "summary statistics (settled poses, net displacement, onset times) "
    "+ divergence guard;  [2] keep disagreeing segments and widen;\n"
    "[3] sweep the disagreement HULL (union of per-segment and per-cycle "
    "fits) instead of the floored sigma;  [4] INCONSISTENT widens the "
    "sweep instead of disarming it (and holds the last TRUSTED value).")


def _lane_x(lane):
    return LANES[lane][0]


def draw():
    fig, ax = plt.subplots(figsize=(21, 11.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    for li, (cx, title) in enumerate(LANES):
        ax.add_patch(
            FancyBboxPatch((cx - LANE_W / 2, 7.5),
                           LANE_W,
                           48.0,
                           boxstyle="round,pad=0.4",
                           facecolor=LANE_FILL,
                           edgecolor="none",
                           zorder=0))
        ax.add_patch(
            FancyBboxPatch((cx - LANE_W / 2, 53.5),
                           LANE_W,
                           3.4,
                           boxstyle="round,pad=0.3",
                           facecolor=HEADER_COLORS[li],
                           edgecolor="none",
                           zorder=2))
        ax.text(cx,
                55.2,
                title,
                ha="center",
                va="center",
                fontsize=13,
                color="white",
                fontweight="bold",
                zorder=3)

    centers = {}
    for bid, lane, yc, h, title, body, flag in BOXES:
        cx = _lane_x(lane)
        w = LANE_W - 2.0
        ax.add_patch(
            FancyBboxPatch((cx - w / 2, yc - h / 2),
                           w,
                           h,
                           boxstyle="round,pad=0.25",
                           facecolor=BOX_FILL,
                           edgecolor=BOX_EDGE,
                           linewidth=1.1,
                           zorder=2))
        ax.text(cx,
                yc + h / 2 - 1.1,
                title,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                zorder=3)
        ax.text(cx,
                yc + h / 2 - 2.2,
                body,
                ha="center",
                va="top",
                fontsize=8.6,
                zorder=3,
                linespacing=1.35)
        if flag:
            ax.text(cx - w / 2 + 0.6,
                    yc - h / 2 + 0.5,
                    flag,
                    ha="left",
                    va="bottom",
                    fontsize=8.6,
                    color=FLAG_COLOR,
                    fontweight="bold",
                    zorder=4)
        centers[bid] = (cx, yc, w, h)

    for src, dst in ARROWS:
        sx, sy, sw, sh = centers[src]
        dx, dy, dw, dh = centers[dst]
        if abs(sx - dx) < 1.0:
            start, end = (sx, sy - sh / 2 - 0.3), (dx, dy + dh / 2 + 0.3)
            style = "arc3,rad=0.0"
        else:
            start, end = (sx + sw / 2 + 0.3, sy), (dx - dw / 2 - 0.3, dy)
            style = "arc3,rad=-0.08"
        ax.add_patch(
            FancyArrowPatch(start,
                            end,
                            arrowstyle="-|>",
                            mutation_scale=13,
                            linewidth=1.3,
                            color="#666666",
                            connectionstyle=style,
                            zorder=1))

    ax.text(2.0,
            5.2,
            LEGEND,
            ha="left",
            va="top",
            fontsize=9.0,
            linespacing=1.5)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    draw()
