"""Archive the real-robot Fan-Domino frames and measurements for Figure 3.

The source is the shared folder of the 2026-09-22 cascade run,
downloaded to ``logs/real_robot/fan_domino_drive`` (for example with
``gdown``). Frames come from the gust camera's tracking videos, which
overlay the fitted box of the probed or goal block.
"""
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# Optional figure-authoring dependencies, separate from benchmark runtime.
import imageio_ffmpeg  # type: ignore # pylint: disable=import-error
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parents[1] / "logs/real_robot/fan_domino_drive"
DRIVE = ("https://drive.google.com/drive/folders/"
         "1bcFFkaMb1ZKa0p1sK5KuMdQQ92xojnBO")
# Two exploration probes, then the test after the patch was moved.
SELECTION = [
    ("ep01_gust_tracked.mp4", 106, 1, "Probe: green stays"),
    ("ep02_gust_tracked.mp4", 104, 2, "Probe: grey slides"),
    ("ep03_gust_tracked.mp4", 45, 3, "Test: grey upwind"),
    ("ep03_gust_tracked.mp4", 63, 3, "Grey knocks green"),
    ("ep03_gust_tracked.mp4", 104, 3, "Green in patch"),
]
# Keeps the gripper at the button, the fan, both blocks, and both patch
# placements in every frame.
CROP = (230, 400, 1210, 1080)
FPS = 15


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(video: Path, index: int) -> Image.Image:
    png = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i",
        str(video), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", "-f",
        "image2pipe", "-vcodec", "png", "-"
    ],
                         capture_output=True,
                         check=True).stdout
    return Image.open(io.BytesIO(png)).convert("RGB")


def main() -> None:
    """Extract the selected frames and record their provenance."""
    rows = [
        json.loads(line)
        for line in (RUN / "calibration.jsonl").read_text().splitlines()
    ]
    decisions = [
        json.loads(line)
        for line in (RUN / "decisions.jsonl").read_text().splitlines()
    ]
    test = next(d["decision"] for d in decisions if d.get("test"))
    posterior = json.loads((RUN / "posterior.json").read_text())
    frames: List[Dict[str, Any]] = []
    for i, (video, index, episode, label) in enumerate(SELECTION):
        image = _frame(RUN / video, index)
        assert image.size == (1920, 1080), image.size
        dest = ROOT / "figures/sources" / f"real_fan_domino_{i}.png"
        image.crop(CROP).save(dest)
        frames.append(
            dict(name=dest.stem,
                 label=label,
                 episode=episode,
                 video=video,
                 video_sha256=_digest(RUN / video),
                 frame_index=index,
                 time_s=round(index / FPS, 3),
                 sha256=_digest(dest)))
    archive = dict(
        generated_by="scripts/paper_figures/import_real_robot_frames.py; "
        "do not edit manually",
        description="Recorded gust-camera frames from one real-robot run; "
        "overlays are the tracker's fitted boxes, not predictions.",
        experiment="exp_20260922_134142",
        seed=int((RUN / "seed").read_text()),
        source_folder=DRIVE,
        report_sha256=_digest(RUN / "REPORT.md"),
        agent_log_sha256=_digest(RUN / "agent.log"),
        crop=CROP,
        measured=[
            dict(episode=k + 1,
                 block=r["block"],
                 distance_m=round(r["dist_m"], 3),
                 slide_cm=round(100 * r["slide_m"], 1),
                 fall_deg=r["fall_deg"],
                 flat_in_patch=r["solved"]) for k, r in enumerate(rows)
        ],
        test_plan=dict(block=test["block"],
                       at=test["at"],
                       also=test["also"],
                       predicted_slide_cm=[
                           round(100 * v, 1) for v in test["slide_pred_m"]
                       ],
                       p_success=round(test["p_success"], 2)),
        posterior=dict(kept=posterior["kept"],
                       draws=posterior["draws"],
                       masses=posterior["masses"],
                       p_green_lighter=posterior["p_green_lighter"]),
        frames=frames)
    (ROOT / "data/trajectories/real_fan_domino.json"
     ).write_text(json.dumps(archive, indent=2) + "\n")


if __name__ == "__main__":
    main()
