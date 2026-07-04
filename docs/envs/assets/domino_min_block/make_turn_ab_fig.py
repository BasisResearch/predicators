"""Slide figure: the 45-deg turn block's mirrored yaw is load-bearing.

Four schematic panels (footprints + facing arrows):
  1. generator convention (mirrored 45-block)  -> topples (A/B verified)
  2. "natural" aligned 45-block                -> fails at min-block gaps
  3. search family: stretched natural corner (saves a block)
  4. search family: straight-line probe (corner-cheat check)
Outcome labels come from the simulated A/B (gaps 0.098-0.13, frictions
0.1/0.5, side offsets {-W/2, 0, +W/2}); this drawing is schematic.
Scope: the panel-2 failure holds ONLY in that swept min-block band. At the
legacy generator's tighter gaps (<= ~0.09) the natural alignment DOES
propagate (legacy b2e0f244 used natural yaw + turn_shift offset; re-verified
by sim A/B on the real corner-family geometry, 2026-07-03).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

W, D = 0.07, 0.015
GAP = 0.10


def block(ax, x, y, yaw, color, hl=False):
    tr = (Affine2D().rotate(-yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2), W, D, facecolor=color, edgecolor="k",
                  lw=1.1, transform=tr, zorder=3))
    fx, fy = 0.045 * np.sin(yaw), 0.045 * np.cos(yaw)
    ax.arrow(x, y, fx, fy, head_width=0.011, color=color, lw=1.1, zorder=4)
    if hl:
        ax.add_patch(
            Circle((x, y), 0.055, fill=False, edgecolor="#d62728", lw=2.2,
                   linestyle="--", zorder=5))


def turn_chain(ax, d1_sign, title, verdict, vcolor, note=None):
    """start -> entry blue -> d1 (45 deg, sign under test) -> d2 -> exit ->
    target."""
    syaw, td = 0.0, 1.0  # travel +y, turning left (exit -x)
    u = np.array([np.sin(syaw), np.cos(syaw)])
    d1_dir = syaw - td * np.pi / 4
    d2_rot = syaw - td * np.pi / 2
    s = np.array([0.0, 0.0])
    block(ax, *s, syaw, "#7fc97f")
    e1 = s + GAP * u
    block(ax, *e1, syaw, "#7fb2d9")
    d1 = e1 + GAP * u + np.array([
        td * -(W / 2) * np.cos(d1_dir), -td * -(W / 2) * np.sin(d1_dir)])
    block(ax, *d1, syaw + d1_sign * td * np.pi / 4, "#7fb2d9", hl=True)
    d2 = d1 + GAP * np.array([np.sin(d1_dir), np.cos(d1_dir)]) + np.array([
        td * -(W / 2) * np.cos(d2_rot), -td * -(W / 2) * np.sin(d2_rot)])
    block(ax, *d2, syaw + td * np.pi / 2, "#7fb2d9")
    e_dir = np.array([np.sin(d2_rot), np.cos(d2_rot)])
    ex = d2 + GAP * e_dir
    block(ax, *ex, d2_rot, "#7fb2d9")
    t = d2 + 2 * GAP * e_dir
    block(ax, *t, d2_rot, "#c599c5")
    ax.set_title(title, fontsize=11)
    ax.text(0.02, -0.13, verdict, fontsize=12, color=vcolor, weight="bold",
            ha="center")
    if note:
        ax.text(0.02, -0.20, note, fontsize=9, color="#555555", ha="center")
    ax.set_xlim(-0.36, 0.18)
    ax.set_ylim(-0.23, 0.36)


fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))

# 1) generator convention: d1 yaw mirrored (rot + td*45)
turn_chain(axes[0], +1.0, "Generator convention\n(45°-block yaw MIRRORED)",
           "TOPPLES ✓ (gap ≤ 0.11)", "#1a7a1a")
# 2) "natural" alignment: d1 yaw = travel (rot − td*45)
turn_chain(axes[1], -1.0, '"Natural" alignment\n(45°-block faces the chain)',
           "FAILS AT MIN-BLOCK GAPS ✗\n(0.098–0.13 · frictions · offsets)",
           "#a01515",
           note="legacy generator regime (gap ≲ 0.09): propagates fine")

# 3) stretched corner (search family): ONE natural-yaw corner blue slid
# toward the start (the agent-buildable corner style; cf. the oracle's
# -36 deg corner blue), no entry blues needed.
ax = axes[2]
syaw, td = 0.0, 1.0
u = np.array([0.0, 1.0])
d2_rot = -np.pi / 2
s = np.array([0.0, 0.0])
block(ax, *s, syaw, "#7fc97f")
c_yaw = syaw - td * 0.5 * np.pi / 2  # natural: faces halfway into the turn
c_dir = np.array([np.sin(c_yaw), np.cos(c_yaw)])
c = s + 0.15 * u
block(ax, *c, c_yaw, "#7fb2d9", hl=True)
b1 = c + 0.08 * c_dir
block(ax, *b1, d2_rot, "#7fb2d9")
t = b1 + 0.11 * np.array([np.sin(d2_rot), np.cos(d2_rot)])
block(ax, *t, d2_rot, "#c599c5")
ax.annotate("stretched entry\n(no blue needed)", (0.045, 0.075), fontsize=9,
            color="#b3541e", ha="left")
ax.set_title("Search: stretched natural corner\n(slides corner toward start)",
             fontsize=11)
ax.text(-0.09, -0.13, "can SAVE a block vs the even L\n→ K* must search layouts",
        fontsize=10, color="#b3541e", ha="center")
ax.set_xlim(-0.36, 0.18)
ax.set_ylim(-0.23, 0.36)

# 4) straight-line probe
ax = axes[3]
s = np.array([0.0, 0.0])
t = np.array([-0.24, 0.24])
d = (t - s) / np.linalg.norm(t - s)
line_yaw = float(np.arctan2(d[0], d[1]))
block(ax, *s, 0.0, "#7fc97f")
for i in (1, 2):
    p = s + i * np.linalg.norm(t - s) / 3 * d
    block(ax, *p, line_yaw, "#7fb2d9")
block(ax, *t, -np.pi / 2, "#c599c5")
ax.set_title("Search: straight-line probe\n(can the corner be cheated?)",
             fontsize=11)
ax.text(-0.09, -0.13, "usually fails — exists so K*\naccounts for corner-cheats",
        fontsize=10, color="#555", ha="center")
ax.set_xlim(-0.36, 0.18)
ax.set_ylim(-0.23, 0.36)

for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")

fig.tight_layout()
out = Path(__file__).parent / "turn_ab.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
