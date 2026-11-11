"""Render a video's exported frame scenes with Blender Cycles.

Reads a manifest of predicators/run/cycles_video.py, such as the one
export_run_video_scenes.py writes, and renders each frame's scene as
render_cycles_scene.py renders a figure scene: the same materials, lights,
camera and AgX look. Frames already rendered are skipped, so a job can
resume, and ``--task``/``--tasks`` split the frames across array jobs. On
an L40S, ``--device OPTIX`` takes about three seconds a frame, scene build
included; Cycles CPU, as for the figures, takes far longer.
scripts/cycles_video.py runs this script and assembles the videos.

Run in the Blender environment, from the repository root:
    uv run --no-project --python 3.11 --with bpy==4.5.3 --with pycollada \\
        python scripts/paper_figures/render_cycles_frames.py \\
        logs/paper_run_videos_cycles/domino/manifest-all.json --device OPTIX
"""
import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict

import bpy  # type: ignore[import-not-found]  # pylint: disable=import-error

# Importing the figure renderer limits threads and loads bpy first.
import render_cycles_scene as rcs  # isort: skip


def main() -> None:
    """Render the selected frames of one manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device",
                        choices=("CPU", "OPTIX", "CUDA"),
                        default="CPU")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="render at most this many frames (a timing probe)")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="render the selection from its end, so a second "
        "job can finish a slow job's share from the other side")
    parser.add_argument("--pick",
                        type=int,
                        default=None,
                        help="render only this many evenly spaced frames, "
                        "first and last included (a preview)")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    out = root / "frames"
    out.mkdir(exist_ok=True)
    names = list(dict.fromkeys(f["scene"] for f in manifest["frames"]))
    if args.pick:
        last = len(names) - 1
        names = [
            names[round(i * last / (args.pick - 1))] for i in range(args.pick)
        ]
    mine = names[args.task::args.tasks][:args.limit]
    if args.reverse:
        mine.reverse()
    verified: Dict[str, str] = {}
    device = args.device
    if device == "OPTIX" and not rcs.use_gpu("OPTIX"):
        device = "CUDA"  # OptiX needs a driver and GPU that support it.
    print(f"Rendering on {device}", flush=True)
    for name in mine:
        target = out / name.replace(".json.gz", ".png")
        if target.exists():
            continue
        start = time.time()
        with gzip.open(root / "scenes" / name, "rt") as f:
            exported = json.load(f)
        for mesh, digest in exported["mesh_sha256"].items():
            if mesh not in verified:
                verified[mesh] = hashlib.sha256(
                    rcs.resolve_mesh(mesh).read_bytes()).hexdigest()
            assert verified[mesh] == digest, mesh
        rcs.build_scene(exported, args.samples, args.threads, args.scale,
                        device)
        built = time.time()
        # Per-process name: two jobs may render the same frame where their
        # shares meet, and each must finish its own file before the rename.
        partial = target.with_name(f"{target.stem}.{os.getpid()}.partial.png")
        bpy.context.scene.render.filepath = str(partial.resolve())
        bpy.ops.render.render(write_still=True)
        partial.replace(target)
        print(
            f"{name}: build {built - start:.1f} s, render "
            f"{time.time() - built:.1f} s",
            flush=True)
    print(
        f"Task {args.task}/{args.tasks}: {len(mine)} frames of "
        f"{manifest['domain']} done",
        flush=True)


if __name__ == "__main__":
    main()
