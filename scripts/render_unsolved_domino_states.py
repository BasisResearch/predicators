"""Render init-state PNGs for the unsolved domino tasks (oracle-samplers runs).

Uses the geometry-affecting flags from the experiment command line so the
regenerated test scenes match the runs exactly (verified: seed1 = [4,4,5,4,4]
dominoes, and the seed1.t2 grasp-infeasibility matches the run). Run ONE seed
per process (task-gen RNG is shared across seeds in one interpreter).

Usage: PYTHONPATH=. python scripts/render_unsolved_domino_states.py <seed>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from predicators import utils


def _project(xyz, view_matrix, proj_matrix, width, height):
    """World (x,y,z) -> (u,v) pixel using pybullet's column-major matrices."""
    V = np.array(view_matrix).reshape((4, 4), order="F")
    P = np.array(proj_matrix).reshape((4, 4), order="F")
    clip = P @ (V @ np.array([xyz[0], xyz[1], xyz[2], 1.0]))
    if clip[3] == 0:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * width,
            (1.0 - (ndc[1] * 0.5 + 0.5)) * height)


def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # pylint: disable=broad-except
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _caption(rgb, lines):
    """Draw a header banner (top-left) with the given text lines."""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(22)
    pad, lh = 8, 26
    w = max(draw.textlength(t, font=font) for t in lines)
    draw.rectangle([0, 0, w + 2 * pad, lh * len(lines) + pad],
                   fill=(0, 0, 0, 170))
    for i, t in enumerate(lines):
        draw.text((pad, pad + i * lh), t, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _annotate(rgb, init_state, cam):
    """Label each domino with its index at its initial-state position."""
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    font = _font(26)
    for o in sorted([o for o in init_state if o.type.name == "domino"],
                    key=lambda o: o.name):
        x, y, z = (init_state.get(o, "x"), init_state.get(o, "y"),
                   init_state.get(o, "z"))
        uv = _project((x, y, z + 0.13), *cam)
        if uv is None:
            continue
        u, v = uv
        idx = o.name.split("_")[-1]
        col = (int(init_state.get(o, "r") * 255),
               int(init_state.get(o, "g") * 255),
               int(init_state.get(o, "b") * 255))
        r = 15
        draw.ellipse([u - r, v - r, u + r, v + r],
                     fill=(0, 0, 0),
                     outline=col,
                     width=3)
        tb = draw.textbbox((0, 0), idx, font=font)
        draw.text((u - (tb[2] - tb[0]) / 2, v - (tb[3] - tb[1]) / 2 - tb[1]),
                  idx,
                  fill=(255, 255, 255),
                  font=font)
    return np.asarray(img)


# 1-indexed tasks unsolved in EITHER arm, with (arms, failure-mode) labels.
UNSOLVED = {
    0: {
        1: ("both", "push-dropped"),
        2: ("both", "place-MP+InFront"),
        3: ("no_demo", "pick+place-MP")
    },
    1: {
        1: ("demo", "exec-retreat-collision"),
        3: ("both", "pick+place-MP")
    },
    2: {
        1: ("no_demo", "pick+place-MP"),
        2: ("demo", "toppled-cascade"),
        4: ("both", "pick+place-MP"),
        5: ("both", "place-MP+toppled")
    },
    3: {
        5: ("demo", "holding+InFront+place-MP")
    },
}
FLAGS = {
    "env": "pybullet_domino",
    "num_train_tasks": 1,
    "num_test_tasks": 5,
    "pybullet_ik_validate": False,
    "pybullet_camera_width": 900,
    "pybullet_camera_height": 900,
    "domino_initialize_at_finished_state": False,
    "domino_use_domino_blocks_as_target": True,
    "domino_use_continuous_place": True,
    "domino_restricted_push": True,
    "domino_has_glued_dominos": False,
    "pybullet_birrt_extend_num_interp": 20,
    "pybullet_birrt_path_subsample_ratio": 2,
}
OUT = "logs/agent_sim_learning/unsolved_init_states"


def main():
    seed = int(sys.argv[1])
    os.makedirs(OUT, exist_ok=True)
    utils.reset_config(dict(FLAGS, seed=seed))
    from predicators.envs import create_new_env
    env = create_new_env("pybullet_domino", do_cache=False)
    tasks = env.get_test_tasks()
    counts = [
        len([o for o in t.init if o.type.name == "domino"]) for t in tasks
    ]
    print(f"seed{seed} domino counts per task = {counts}")
    cam = env._get_camera_matrices()  # pylint: disable=protected-access
    for t1, (arms, mode) in sorted(UNSOLVED.get(seed, {}).items()):
        idx = t1 - 1
        env.reset("test", idx)
        rgb = np.asarray(env.render()[0], dtype=np.uint8)
        rgb = _annotate(rgb, tasks[idx].init, cam)
        goal_ids = ",".join(
            sorted(
                str(a).split("_")[-1].rstrip(":domino)")
                for a in tasks[idx].goal))
        rgb = _caption(rgb, [
            f"seed {seed}  task {t1}  ({arms})",
            f"goal: Toppled({goal_ids})   fail: {mode}"
        ])
        fname = f"seed{seed}_task{t1}_{arms}_{mode}.png"
        Image.fromarray(rgb).save(os.path.join(OUT, fname))
        goal = sorted(str(a) for a in tasks[idx].goal)
        print(f"  saved {fname} | {counts[idx]} dominoes | goal={goal}")


if __name__ == "__main__":
    main()
