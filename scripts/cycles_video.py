"""Render a run's Blender Cycles scenes and assemble its videos.

The scenes come from ``predicators/run/cycles_video.py``. Under
``--video_cycles_scenes True``, a continual run writes them to
``<run_dir>/cycles`` and test and failure videos write them to
``<video_dir>/<video name>_cycles``;
``scripts/continual_video.py --run_dir <run_dir> --cycles`` writes them
for an earlier continual run. On a GPU node::

    python scripts/cycles_video.py <run_dir>/cycles/manifest-all.json

renders every frame with Blender Cycles, skipping frames already
rendered so that a preempted job resumes where it stopped, then writes
beside the manifest ``<domain>.mp4``, the scene with the run video's
label panel, and ``<domain>-test-scene.mp4``, the test task alone. A
manifest of an evaluation trajectory, which has no labels, gives
``<domain>-scene.mp4``. ``--tasks N --task k`` renders every N-th frame
from the k-th, for N jobs at once; such a job does not assemble, so run
once more without them when all have finished.

Blender runs in its own Python, with bpy 4.5.3 and pycollada on Python
3.11: through ``uv run`` by default, or the interpreter that
``--blender-python`` names. On an L40S a frame takes about three
seconds; ``--device CPU`` works anywhere but takes far longer.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable-next=wrong-import-position
from predicators.run.cycles_video import compose_video  # noqa: E402

RENDERER = Path(__file__).resolve().parent / "paper_figures" / \
    "render_cycles_frames.py"


def blender_command(blender_python: str) -> List[str]:
    """The command prefix that runs a script in Blender's Python."""
    if blender_python:
        return [blender_python]
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    return [
        uv, "run", "--no-project", "--python", "3.11", "--with", "bpy==4.5.3",
        "--with", "pycollada", "python"
    ]


def main() -> None:
    """Render the manifest's frames, then assemble its videos."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("manifest",
                        type=Path,
                        help="a manifest, or the directory holding "
                        "manifest-all.json")
    parser.add_argument("--device",
                        choices=("OPTIX", "CUDA", "CPU"),
                        default="OPTIX")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument("--blender-python", default="")
    parser.add_argument("--no-render",
                        action="store_true",
                        help="only assemble the videos")
    args = parser.parse_args()
    manifest = args.manifest
    if manifest.is_dir():
        manifest = manifest / "manifest-all.json"
    if not args.no_render:
        subprocess.run(blender_command(args.blender_python) + [
            str(RENDERER),
            str(manifest), "--device", args.device, "--samples",
            str(args.samples), "--threads",
            str(args.threads), "--task",
            str(args.task), "--tasks",
            str(args.tasks)
        ],
                       check=True,
                       env=dict(os.environ,
                                EMPIRIC_RENDER_THREADS=str(args.threads)))
        if args.tasks > 1:
            print("Rendered this job's share; run again without --tasks to "
                  "assemble the videos once every share is done.")
            return
    frames = json.loads(manifest.read_text())["frames"]
    if "label" in frames[0]:
        print(f"Wrote {compose_video(manifest, panel=True)}")
    else:
        print(f"Wrote {compose_video(manifest)}")
    if frames[-1]["split"] == "test" and "label" in frames[0]:
        print(f"Wrote {compose_video(manifest, test_only=True)}")


if __name__ == "__main__":
    main()
