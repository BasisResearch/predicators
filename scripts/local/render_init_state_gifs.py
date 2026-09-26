"""Render a looping GIF of initial states for each RoboDisco environment.

Each GIF cycles through the initial states of several train and test tasks,
one frame per task, with a small caption naming the split and task index.
Flags come from the config menus, so the rendered tasks are the ones the
experiments use:

- the five benchmark domains take their entry in
  scripts/configs/predicatorv3/envs/continual.yaml,
- the other recent domains take their entry in envs/all.yaml,
- the older domains take their entry in random_actions_pybullet.yaml.

Observation noise flags are irrelevant here (states are rendered, not
observed). Run on a compute node, one env per call:

    PYTHONPATH=. python scripts/local/render_init_state_gifs.py \
        --env pybullet_boil --out-dir <dir>
"""
import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from predicators import utils
from predicators.envs import create_new_env

_CONFIG_DIR = "scripts/configs/predicatorv3"

# env name -> (menu file, menu key)
_MENUS: Dict[str, Tuple[str, str]] = {
    "pybullet_balloons": ("envs/continual.yaml", "balloons"),
    "pybullet_bridge": ("envs/continual.yaml", "bridge"),
    "pybullet_boil": ("envs/continual.yaml", "boil"),
    "pybullet_fan": ("envs/continual.yaml", "fan"),
    "pybullet_domino": ("envs/continual.yaml", "domino_high_friction_turn"),
    "pybullet_busyboard": ("envs/all.yaml", "busyboard"),
    "pybullet_crane": ("envs/all.yaml", "crane"),
    "pybullet_icerink": ("envs/all.yaml", "icerink"),
    "pybullet_launcher": ("envs/all.yaml", "launcher"),
    "pybullet_magnets": ("envs/all.yaml", "magnets"),
}
_LEGACY_MENU = "random_actions_pybullet.yaml"


def _menu_flags(env_name: str) -> Dict[str, Any]:
    """Return the FLAGS of the menu entry that defines this env's tasks."""
    menu_file, key = _MENUS.get(env_name, (_LEGACY_MENU, ""))
    with open(os.path.join(_CONFIG_DIR, menu_file), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    envs = config["ENVS"]
    if not key:
        key = next(k for k, v in envs.items() if v["NAME"] == env_name)
    flags = dict(envs[key].get("FLAGS", {}))
    return {k: v for k, v in flags.items() if not k.startswith("continual_")}


def _caption(frame: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(  # type: ignore[no-untyped-call]
        frame[:, :, :3].astype(np.uint8))
    draw = ImageDraw.Draw(image)
    size = max(14, image.height // 18)
    # Pillow's bundled scalable font; system fonts vary across nodes.
    font: Any = ImageFont.load_default(size=size)
    pad = size // 2
    box = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((box[0] - pad // 2, box[1] - pad // 2, box[2] + pad // 2,
                    box[3] + pad // 2),
                   fill=(15, 23, 42))
    draw.text((pad, pad), text, fill=(241, 245, 249), font=font)
    return image


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-train", type=int, default=4)
    parser.add_argument("--num-test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1008)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame-ms", type=int, default=1200)
    args = parser.parse_args()

    config = _menu_flags(args.env)
    config.update({
        "env": args.env,
        "seed": args.seed,
        "num_train_tasks": args.num_train,
        "num_test_tasks": args.num_test,
        "pybullet_camera_width": args.width,
        "pybullet_camera_height": args.height,
    })
    utils.reset_config(config)
    env = create_new_env(args.env, do_cache=True, use_gui=False)

    frames: List[Image.Image] = []
    seen: List[np.ndarray] = []
    for split, num in (("train", args.num_train), ("test", args.num_test)):
        tasks = env.get_train_tasks() if split == "train" else \
            env.get_test_tasks()
        for idx in range(min(num, len(tasks))):
            env.reset(split, idx)
            frame = np.asarray(env.render()[0])
            if any(np.array_equal(frame, s) for s in seen):
                continue  # fixed-layout envs repeat the same scene
            seen.append(frame)
            frames.append(_caption(frame, f"{split.capitalize()} task {idx}"))
            print(f"rendered {split} task {idx}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.env}.gif")
    frames[0].save(out,
                   save_all=True,
                   append_images=frames[1:],
                   duration=args.frame_ms,
                   loop=0,
                   optimize=True)
    first = os.path.join(args.out_dir, f"{args.env}_init0.png")
    frames[0].save(first)
    print(f"wrote {out} ({len(frames)} distinct initial states)")


if __name__ == "__main__":
    _main()
