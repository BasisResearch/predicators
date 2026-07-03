"""Slide figure: the turn-K* candidate family (agent-buildable layouts).

Renders the REAL candidates yielded by ``_candidate_turn_layouts`` for a
canonical turn geometry (k=3): the straight-line probe plus the five
natural-yaw corner configs (``_CORNER_CONFIGS``), and — for contrast — the
generator's mirrored 45-degree pair, which is deliberately EXCLUDED from
the search because no planner would propose it. Footprints are geometry-
exact (poses come from the search code itself); no simulation is run.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu

utils.reset_config({
    "env": "pybullet_domino",
    "seed": 0,
    "domino_use_domino_blocks_as_target": True,
    "domino_true_friction": 0.1,
})
env = create_new_env("pybullet_domino", do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
W, D = comp.domino_width, comp.domino_depth

START = (0.55, 1.20, np.pi / 2)  # travel +x
TARGET = (0.85, 1.42, 0.0)  # one left turn away, faces +y


def block(ax, x, y, yaw, color, hl=False):
    tr = (Affine2D().rotate(-yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2), W, D, facecolor=color, edgecolor="k",
                  lw=1.0, transform=tr, zorder=3))
    fx, fy = 0.03 * np.sin(yaw), 0.03 * np.cos(yaw)
    ax.arrow(x, y, fx, fy, head_width=0.009, color=color, lw=1.0, zorder=4)
    if hl:
        ax.add_patch(
            Circle((x, y), 0.045, fill=False, edgecolor="#d62728", lw=2.0,
                   linestyle="--", zorder=5))


def draw_candidate(ax, od, title, tcolor="k", corner_yaw=None):
    for obj, pose in od.items():
        x, y, yaw = pose["x"], pose["y"], pose["yaw"]
        if obj.name == "domino_0":
            block(ax, x, y, yaw, "#7fc97f")
        elif obj.name == "domino_1":
            block(ax, x, y, yaw, "#c599c5")
        else:
            hl = corner_yaw is not None and abs(yaw - corner_yaw) < 1e-6
            block(ax, x, y, yaw, "#7fb2d9", hl=hl)
    ax.set_title(title, fontsize=10, color=tcolor)
    ax.set_xlim(0.47, 0.95)
    ax.set_ylim(1.10, 1.52)
    ax.set_aspect("equal")
    ax.axis("off")


cands = list(mbu._candidate_turn_layouts(comp, 3, START, TARGET))
fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
axes = axes.ravel()

# Panel 0: straight-line probe (first candidate yielded).
draw_candidate(axes[0], cands[0][0], "straight-line probe\n(corner-cheat check)")

# Panels 1-5: the natural-yaw corner configs (k1=0 candidates follow the
# straight probe in yield order; label with their (f, g1, g2)).
corner_cands = cands[1:1 + len(mbu._CORNER_CONFIGS)]
for i, ((od, _s, _t), cfg) in enumerate(zip(corner_cands,
                                            mbu._CORNER_CONFIGS)):
    f_yaw, g1, g2 = cfg
    draw_candidate(
        axes[1 + i], od,
        f"natural corner\nyaw {int(f_yaw * 90)}° · in {g1:.2f} · out {g2:.2f}",
        corner_yaw=np.pi / 2 - f_yaw * np.pi / 2)

# Panel 6: the generator's mirrored pair — excluded from the search.
ax = axes[6]
sx, sy, syaw = START
u = np.array([1.0, 0.0])
td = 1.0
half_w = W / 2
g = 0.10
d1_dir = syaw - td * np.pi / 4
d1_yaw = syaw + td * np.pi / 4
d2_rot = syaw - td * np.pi / 2
s_pt = np.array([sx, sy])
d1 = s_pt + g * u + np.array(
    [td * -half_w * np.cos(d1_dir), -td * -half_w * np.sin(d1_dir)])
d2 = d1 + g * np.array([np.sin(d1_dir), np.cos(d1_dir)]) + np.array(
    [td * -half_w * np.cos(d2_rot), -td * -half_w * np.sin(d2_rot)])
block(ax, sx, sy, syaw, "#7fc97f")
block(ax, *d1, d1_yaw, "#cccccc", hl=True)
block(ax, *d2, syaw + td * np.pi / 2, "#cccccc")
block(ax, *TARGET[:2], TARGET[2], "#c599c5")
ax.set_title("generator's mirrored pair\nEXCLUDED — agents don't build this",
             fontsize=10, color="#a01515")
ax.set_xlim(0.47, 0.95)
ax.set_ylim(1.10, 1.52)
ax.set_aspect("equal")
ax.axis("off")

# Panel 7: legend / notes.
ax = axes[7]
ax.axis("off")
ax.text(0.02, 0.85, "k = 3 candidates for one canonical task (k1 = 0 shown)", fontsize=11,
        weight="bold")
ax.text(
    0.02, 0.12,
    "green = start (pushed)   purple = target\nblue = movable blues; "
    "dashed circle = corner blue\n\nhigher k adds entry blues\n"
    "(per-gap ∈ {0.10, 0.13, 0.15} slides the corner)\nand evenly-spaced "
    "exit blues\n\ncorner configs sim-calibrated: each propagates\nat one "
    "of the two experiment frictions", fontsize=9.5, va="bottom")

fig.tight_layout()
out = Path(__file__).parent / "turn_layouts.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out, f"({len(cands)} candidates at k=3)")
