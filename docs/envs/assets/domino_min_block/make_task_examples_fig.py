"""Slide figure: sampled tasks — calibrated vs miscalibrated solutions.

For each sampled cached test task, three columns:
  1. the staged initial state (blues parked, start/target fixed);
  2. the calibrated solution: the K*-search's winning layout at the TRUE
     friction, re-verified by simulation (should topple);
  3. the miscalibrated plan: the cheapest layout the PLANNING-friction
     model accepts (straight: believed-K chain; turn: same corner blueprint
     with legs respaced at believed counts), then EXECUTED at the true
     friction (should die short) — the baseline's predicted behaviour.

Everything is produced by the real task-gen machinery: cached tasks are
reloaded through the env, layouts come from the search code, outcomes from
sim rollouts with the real Push.
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
from predicators.settings import CFG

utils.reset_config({
    'env': 'pybullet_domino', 'seed': 0, 'num_train_tasks': 1,
    'num_test_tasks': 5, 'test_env_seed_offset': 10000,
    'max_initial_demos': 0, 'horizon': 500,
    'domino_initialize_at_finished_state': False,
    'domino_use_domino_blocks_as_target': True,
    'domino_use_continuous_place': True,
    'domino_has_glued_dominos': False,
    'domino_min_block_tasks': True,
    'domino_true_friction': 0.1, 'domino_planning_friction': 0.5,
    'domino_min_block_span_lo': 0.13, 'domino_min_block_span_hi': 0.3,
    'domino_min_block_num_blues': 4,
    'pybullet_birrt_extend_num_interp': 20,
    'pybullet_birrt_path_subsample_ratio': 2,
})
env = create_new_env('pybullet_domino', do_cache=False, use_gui=False)
comp = env._domino_component  # pylint: disable=protected-access
push_opt = mbu._get_push_option(env)
W, D = comp.domino_width, comp.domino_depth
doms = comp.dominos


def block(ax, x, y, yaw, color, hl=False):
    tr = (Affine2D().rotate(-yaw).translate(x, y) + ax.transData)
    ax.add_patch(
        Rectangle((-W / 2, -D / 2), W, D, facecolor=color, edgecolor="k",
                  lw=0.9, transform=tr, zorder=3))
    fx, fy = 0.025 * np.sin(yaw), 0.025 * np.cos(yaw)
    ax.arrow(x, y, fx, fy, head_width=0.008, color=color, lw=0.9, zorder=4)
    if hl:
        ax.add_patch(
            Circle((x, y), 0.042, fill=False, edgecolor="#d62728", lw=1.6,
                   linestyle="--", zorder=5))


def draw_state(ax, poses, title, tcolor="k"):
    """poses: list of (x, y, yaw, role) with role in start/target/blue."""
    colors = {"start": "#7fc97f", "target": "#c599c5", "blue": "#7fb2d9"}
    for x, y, yaw, role in poses:
        block(ax, x, y, yaw, colors[role])
    ax.set_title(title, fontsize=9.5, color=tcolor)
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    ax.set_xlim(min(xs) - 0.09, max(xs) + 0.09)
    ax.set_ylim(min(ys) - 0.09, max(ys) + 0.09)
    ax.set_aspect("equal")
    ax.axis("off")


def od_poses(od, start, target):
    out = []
    for obj, pose in od.items():
        role = ("start" if obj is start else
                "target" if obj is target else "blue")
        out.append((pose["x"], pose["y"], pose["yaw"], role))
    return out


def state_poses(state):
    out = []
    for d in state.get_objects(comp.domino_type):
        # pylint: disable=protected-access
        role = ("start" if comp._StartBlock_holds(state, [d]) else
                "target" if comp._TargetDomino_holds(state, [d]) else "blue")
        out.append((state.get(d, "x"), state.get(d, "y"),
                    state.get(d, "yaw"), role))
    return out


def winning_layout(state, k_max, friction):
    """First toppling layout (straight chain or turn candidate) with the
    fewest blues at ``friction``; returns (od, start, target, k) or None."""
    # pylint: disable=protected-access
    dominoes = state.get_objects(comp.domino_type)
    start = next(d for d in dominoes if comp._StartBlock_holds(state, [d]))
    target = next(d for d in dominoes if comp._TargetDomino_holds(state, [d]))
    s_pose = (state.get(start, "x"), state.get(start, "y"),
              state.get(start, "yaw"))
    t_pose = (state.get(target, "x"), state.get(target, "y"),
              state.get(target, "yaw"))
    env.set_domino_physical_params(friction=friction)
    try:
        for k in range(k_max + 1):
            for od, s_, t_ in mbu._candidate_turn_layouts(
                    comp, k, s_pose, t_pose):
                if mbu._layout_topples(env, od, s_, t_, push_opt):
                    return od, s_, t_, k
    finally:
        env.set_domino_physical_params(friction=CFG.domino_true_friction)
    return None


def executes_at_true(od, s_, t_):
    env.set_domino_physical_params(friction=CFG.domino_true_friction)
    return mbu._layout_topples(env, od, s_, t_, push_opt)


tasks = env.get_test_tasks()
picks = list(range(len(tasks)))  # every task in the live set
# Transposed layout — tasks as columns, stages as rows — so the full set
# fits a widescreen slide.
fig, axes = plt.subplots(3, len(picks), figsize=(3.4 * len(picks), 10.2))
axes = np.atleast_2d(axes).T  # axes[col] = (init, true, believed) per task

for row, ti in enumerate(picks):
    task = tasks[ti]
    k_star = task.reward_fn.max_blocks
    turn = "90-degree" in (task.goal_nl or "")
    kind = "turn" if turn else "straight"
    draw_state(
        axes[row][0], state_poses(task.init),
        f"task {ti} ({kind}) · K*={k_star}\nstaged init")

    # Search up to the full blue budget: staged poses drift a little
    # through the PyBullet round-trip, so the regenerated minimal layout
    # can land one blue off the recorded K*.
    true_win = winning_layout(task.init, CFG.domino_min_block_num_blues,
                              CFG.domino_true_friction)
    if true_win is None:
        axes[row][1].axis("off")
        axes[row][1].set_title(
            "layout not reproducible from staged poses",
            fontsize=9.5, color="#555")
        axes[row][2].axis("off")
        continue
    od, s_, t_, k_t = true_win
    draw_state(
        axes[row][1], od_poses(od, s_, t_),
        f"calibrated: {k_t} blues\n→ TOPPLES ✓",
        tcolor="#1a7a1a")

    # Believed side: what the miscalibrated (µ=0.5) model builds.
    if not turn:
        # Straight: believed evenly-spaced chain over the task's span,
        # count probed at the drift-free canonical anchor.
        s_xy = np.array([od[s_]["x"], od[s_]["y"]])
        t_xy = np.array([od[t_]["x"], od[t_]["y"]])
        span = float(np.linalg.norm(t_xy - s_xy))
        env.set_domino_physical_params(friction=CFG.domino_planning_friction)
        try:
            k_b = mbu.straight_span_k_star(env, span, budget=max(k_t - 1, 0))
        finally:
            env.set_domino_physical_params(friction=CFG.domino_true_friction)
        if k_b is None:
            axes[row][2].axis("off")
            axes[row][2].set_title("no cheaper chain validates at µ=0.5",
                                   fontsize=9.5, color="#555")
            continue
        d_dir = (t_xy - s_xy) / span
        line_yaw = float(np.arctan2(d_dir[0], d_dir[1]))
        odb = {s_: od[s_], t_: od[t_]}
        for i in range(k_b):
            pt = s_xy + (i + 1) * span / (k_b + 1) * d_dir
            odb[doms[2 + i]] = comp.place_domino(2 + i, float(pt[0]),
                                                 float(pt[1]), line_yaw)
        label = f"believed chain: {k_b} blues"
    else:
        # Turn: same route, one fewer blue — the believed model's
        # under-build, spread evenly along the winning layout's own path
        # (each blue faces its local travel direction). Uniform across
        # corner and diagonal-probe winning layouts.
        blues = [(o, pp) for o, pp in od.items() if o not in (s_, t_)]
        s_xy = np.array([od[s_]["x"], od[s_]["y"]])
        t_xy = np.array([od[t_]["x"], od[t_]["y"]])
        rest = blues[:]
        path = [s_xy]
        cur = s_xy
        while rest:
            nxt = min(rest,
                      key=lambda e: (e[1]["x"] - cur[0])**2 +
                      (e[1]["y"] - cur[1])**2)
            rest.remove(nxt)
            cur = np.array([nxt[1]["x"], nxt[1]["y"]])
            path.append(cur)
        path.append(t_xy)
        segs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        lens = [float(np.linalg.norm(b - a)) for a, b in segs]
        total = sum(lens)
        n_bel = max(len(blues) - 1, 0)
        odb = {s_: od[s_], t_: od[t_]}
        for i in range(n_bel):
            d_target = (i + 1) / (n_bel + 1) * total
            acc = 0.0
            for (a, b), seg_len in zip(segs, lens):
                if acc + seg_len >= d_target:
                    frac = (d_target - acc) / seg_len
                    pt = a + frac * (b - a)
                    dvec = (b - a) / seg_len
                    yaw = float(np.arctan2(dvec[0], dvec[1]))
                    odb[doms[2 + i]] = comp.place_domino(
                        2 + i, float(pt[0]), float(pt[1]), yaw)
                    break
                acc += seg_len
        label = f"same route: {n_bel} blues"
    ok = mbu._layout_topples(env, odb, s_, t_, push_opt)
    verdict = "→ TOPPLES (leak!)" if ok else "→ DIES SHORT ✗"
    draw_state(
        axes[row][2], od_poses(odb, s_, t_),
        f"{label}\n{verdict}",
        tcolor="#a01515" if not ok else "#b3541e")

for label, y in (("staged init", 0.86), ("calibrated solution\n@ true µ=0.1", 0.53),
                 ("µ=0.5 model's build\nrun @ true µ=0.1", 0.19)):
    fig.text(0.005, y, label, fontsize=11, rotation=90, va="center",
             weight="bold", color="#333")
fig.tight_layout(rect=(0.03, 0, 1, 1))
out = Path(__file__).parent / "task_examples.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
