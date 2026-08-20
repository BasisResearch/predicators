"""Build the bridge 1x3 mosaics for the 2026-08-20 weekly-sync deck.

Two grids, one row of three seeds each:

  bridge_oracle_hybrid_1x3.mp4   the agent_oracle_hybrid_sim arm's solved
                                 test episodes (opus replication, seeds
                                 0/1/2, all first-attempt solves)
  bridge_explore_1x3.mp4         cycle-0 explore episodes of the learning
                                 arm's 2026-08-19 diagnostic runs (seeds
                                 0/1/2) - the agent experimenting with
                                 glue before it has any cure model

Sources are 900x900; same recipe as make_al_margin_mosaics.py minus the
crop (the bridge camera has no dead floor region): lanczos downscale,
hairline border, short clips hold their last frame so the row loops
together.

Usage:
    python docs/slides/assets/make_bridge_mosaics.py
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VIDEOS = REPO / "videos"

ORACLE_HYBRID = (VIDEOS / "agent_sim_learning" /
                 "bridge-agent_oracle_hybrid_sim_opus")
ORACLE_RUNS = {
    0: "run_20260819_104107",
    1: "run_20260819_104104",
    2: "run_20260819_104101",
}
LEARNING = (VIDEOS / "agent_sim_predicate_invention" /
            "bridge-agent_po_predicate_invention_al")

TILE = 424  # px per tile after downscale; 3 tiles -> ~1.3k row
BORDER = 3
FPS = 20


def _oracle_clip(seed: int) -> Path:
    run_dir = ORACLE_HYBRID / f"seed{seed}" / ORACLE_RUNS[seed]
    matches = sorted(run_dir.glob("*__task1__cycleNone.mp4"))
    assert len(matches) == 1, (run_dir, matches)
    return matches[0]


def _explore_clip(seed: int) -> Path:
    """First cycle-0 explore episode of the seed's 08-19 evening run."""
    run_dirs = sorted((LEARNING / f"seed{seed}").glob("run_20260819_18*"))
    assert run_dirs, f"no 08-19 evening run for seed{seed}"
    matches = sorted(run_dirs[-1].glob("*__ep0__cycle0.mp4"))
    assert len(matches) == 1, (run_dirs[-1], matches)
    return matches[0]


def _row_filter(n_inputs: int) -> str:
    step = TILE + 2 * BORDER
    per_input = "".join(
        f"[{i}:v]scale={TILE}:{TILE}:flags=lanczos,fps={FPS},"
        f"pad={step}:{step}:{BORDER}:{BORDER}:white,"
        f"tpad=stop_mode=clone:stop_duration=600[v{i}];"
        for i in range(n_inputs))
    layout = "|".join(f"{i * step}_0" for i in range(n_inputs))
    return (f"{per_input}"
            f"{''.join(f'[v{i}]' for i in range(n_inputs))}"
            f"xstack=inputs={n_inputs}:layout={layout}:shortest=0[grid]")


def _duration(ffmpeg: str, path: Path) -> float:
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                         capture_output=True,
                         text=True,
                         check=False).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise RuntimeError(f"no duration reported for {path}")


def build(stem: str, clips: list[Path]) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    longest = max(_duration(ffmpeg, c) for c in clips)
    mp4 = HERE / f"{stem}.mp4"
    inputs = [arg for c in clips for arg in ("-i", str(c))]
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *inputs,
        "-filter_complex",
        _row_filter(len(clips)), "-map", "[grid]", "-t", f"{longest:.2f}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt",
        "yuv420p", "-movflags", "+faststart",
        str(mp4)
    ],
                   check=True)
    print(f"wrote {mp4} ({mp4.stat().st_size // 1024} KB)")


def main() -> int:
    build("bridge_oracle_hybrid_1x3", [_oracle_clip(s) for s in (0, 1, 2)])
    build("bridge_explore_1x3", [_explore_clip(s) for s in (0, 1, 2)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
