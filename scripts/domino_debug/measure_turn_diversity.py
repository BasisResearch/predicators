"""Render the first 5 test tasks for seeds 0-4 (the exact tasks the agent run
used) and measure turn % with the grasp-clearance staging check (commit
50d56e940) ON ("after") vs OFF ("before").

Reproduces the env's generation path exactly: for seed N the test rng is
np.random.default_rng(N + CFG.test_env_seed_offset), 5 tasks per seed.

A task is a TURN if the purple target block's yaw differs from the green start
block's yaw by > 30 deg (a straight chain shares one yaw mod pi; a turn90 chain
ends ~90 deg rotated).
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from predicators import utils
from predicators.envs.pybullet_domino.env import PyBulletDominoEnv
from predicators.envs.pybullet_domino.task_generators.domino_task_generator import \
    DominoTaskGenerator
from predicators.settings import CFG

SEEDS = [0, 1, 2, 3, 4]
TASKS_PER_SEED = 5
OFFSET = 10000  # CFG.test_env_seed_offset


def ang_diff(a, b):
    d = (a - b) % np.pi
    return min(d, np.pi - d)


def classify(task, comp):
    st = task.init
    dominoes, sy, ty = [], None, None
    for d in st.get_objects(comp.domino_type):
        r, g, b = st.get(d, "r"), st.get(d, "g"), st.get(d, "b")
        x, y, yaw = st.get(d, "x"), st.get(d, "y"), st.get(d, "yaw")
        if abs(r - comp.start_domino_color[0]) < 1e-2 and \
           abs(g - comp.start_domino_color[1]) < 1e-2:
            role, sy = "start", yaw
        elif abs(r - comp.target_domino_color[0]) < 1e-2 and \
             abs(b - comp.target_domino_color[2]) < 1e-2:
            role, ty = "target", yaw
        else:
            role = "movable"
        dominoes.append((x, y, yaw, role))
    is_turn = (sy is not None and ty is not None
               and ang_diff(sy, ty) > np.deg2rad(30))
    return is_turn, dominoes


def build_generator(env):
    ris = {
        "x": env.robot_init_x,
        "y": env.robot_init_y,
        "z": env.robot_init_z,
        "fingers": env.open_fingers,
        "roll": env.robot_init_roll,
        "tilt": env.robot_init_tilt,
        "wrist": env.robot_init_wrist,
    }
    return DominoTaskGenerator(domino_component=env._domino_component,
                               robot=env._robot,
                               robot_init_state=ris,
                               additional_components=[])


def gen_all(env, disable_grasp):
    gen = build_generator(env)
    orig = DominoTaskGenerator._grasp_clearance_blocked
    if disable_grasp:
        DominoTaskGenerator._grasp_clearance_blocked = lambda *a, **k: False
    try:
        out = []  # (seed, task_idx, task)
        for seed in SEEDS:
            rng = np.random.default_rng(seed + OFFSET)
            tasks = gen.generate_tasks(
                num_tasks=TASKS_PER_SEED,
                rng=rng,
                possible_num_dominos=CFG.domino_test_num_dominos,
                possible_num_targets=CFG.domino_test_num_targets,
                possible_num_pivots=CFG.domino_test_num_pivots)
            for ti, t in enumerate(tasks):
                out.append((seed, ti, t))
    finally:
        DominoTaskGenerator._grasp_clearance_blocked = orig
    return out


def render(entries, comp, title, path):
    nrows, ncols = len(SEEDS), TASKS_PER_SEED
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows))
    w, dpth = comp.domino_width, comp.domino_depth
    cmap = {"start": "#2ca02c", "movable": "#6699ff", "target": "#cc66cc"}
    by_key = {(s, ti): t for (s, ti, t) in entries}
    turns = 0
    for r, seed in enumerate(SEEDS):
        for c in range(TASKS_PER_SEED):
            ax = axes[r][c]
            t = by_key.get((seed, c))
            if t is None:
                ax.axis("off")
                continue
            is_turn, dominoes = classify(t, comp)
            turns += is_turn
            for (x, y, yaw, role) in dominoes:
                rect = Rectangle((-w / 2, -dpth / 2),
                                 w,
                                 dpth,
                                 color=cmap[role])
                rect.set_transform(Affine2D().rotate(yaw).translate(x, y) +
                                   ax.transData)
                ax.add_patch(rect)
            ax.set_xlim(comp.domino_x_lb - 0.05, comp.domino_x_ub + 0.05)
            ax.set_ylim(comp.domino_y_lb - 0.05, comp.domino_y_ub + 0.05)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"seed{seed} t{c}: {'TURN' if is_turn else 'straight'}",
                fontsize=9,
                color="red" if is_turn else "black")
    n = len(entries)
    fig.suptitle(
        f"{title}  —  {turns}/{n} turns ({100.0*turns/n:.0f}%)\n"
        "green=start  blue=movable(staged)  purple=target",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=95)
    plt.close(fig)
    print(f"  {title}: {turns}/{n} turns = {100.0*turns/n:.1f}%  -> {path}")


def main():
    utils.reset_config({
        "env": "pybullet_domino",
        "seed": 0,
        "num_train_tasks": 0,
        "num_test_tasks": TASKS_PER_SEED,
        "test_env_seed_offset": OFFSET,
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_has_glued_dominos": False,
        "domino_test_num_dominos": [3],
        "domino_test_num_targets": [1, 2],
        "domino_test_num_pivots": [0],
    })
    env = PyBulletDominoEnv(use_gui=False)
    comp = env._domino_component
    base = "/Users/ycliang/Code/predicators/scripts/domino_debug/"
    for label, disable, fn in [
        ("AFTER  (grasp-clearance ON  = commit 50d56e940)", False, "after"),
        ("BEFORE (grasp-clearance OFF = parent 50d56e940~1)", True, "before"),
    ]:
        entries = gen_all(env, disable)
        render(entries, comp, label, f"{base}turn_diversity_{fn}.png")


if __name__ == "__main__":
    main()
