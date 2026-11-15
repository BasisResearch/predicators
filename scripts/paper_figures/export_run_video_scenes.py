"""Export one Cycles scene per frame of the paper runs' videos.

The project page and the X thread show the runs behind the paper's
trajectory figures (data/trajectories/stripes.json) rendered with Blender
Cycles. predicators/run/cycles_video.py does the export, as it does for
any run under ``video_cycles_scenes``; this script adds what
the paper runs need: their run directories, Figure 1's moves of the Fan
and Balloons states into the current layouts, and no generation of
Domino's turn tasks, which take minutes of probe simulations and which
the restored recording replaces anyway.

Writes ``<out-dir>/<domain>/scenes/`` and
``<out-dir>/<domain>/manifest-<levels>.json``. Usage (repository root,
PYTHONPATH=.):
    python scripts/paper_figures/export_run_video_scenes.py \
        --domains Domino --levels all
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from export_static_scenes import LOGS, _load_run_config
from render_run_videos import ADJUST, RUNS

from predicators.run import paths
from predicators.run.cycles_video import export_run_scenes
from predicators.run.scorecard import RunCard
from predicators.settings import CFG

# The runs whose videos the project page and the X thread show.
VIDEO_RUNS: Dict[str, str] = {
    "Domino": "domino_high_friction_turn-mb_opus_gate_r1/seed0/"
    "run_20260917_082017",
    "Bridge": "bridge-mb_opus_benchmark_r2/seed3/run_20260919_124955",
    "Balloons": RUNS["Balloons"],
    "Boil": "boil-mb_opus_gate_preflight_two_jug_tight_r1/seed1/"
    "run_20260917_082805",
    "Fan": RUNS["Fan"],
}
# Settings that change only how the environment generates tasks.
GENERATION_FLAGS: Dict[str, Dict[str, Any]] = {
    "Domino": dict(domino_train_turn_ratio=0.0, domino_test_turn_ratio=0.0),
}


def export(domain: str, out_dir: Path, levels: str) -> Path:
    """Export the scenes and manifest of one paper run's video."""
    run_dir = LOGS / VIDEO_RUNS[domain]
    _load_run_config(run_dir)
    for key, value in GENERATION_FLAGS.get(domain, {}).items():
        setattr(CFG, key, value)
    card = RunCard.load(paths.scorecard_path(str(run_dir)))
    return export_run_scenes(card,
                             str(run_dir),
                             out_dir=str(out_dir / domain.lower()),
                             levels=levels,
                             adjust=ADJUST.get(domain),
                             name=domain.lower())


def main() -> None:
    """Export the selected domains' video scenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=list(VIDEO_RUNS))
    parser.add_argument("--levels", choices=("test", "all"), default="all")
    parser.add_argument("--out-dir",
                        type=Path,
                        default=LOGS.parent / "paper_run_videos_cycles")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for domain in args.domains:
        print(
            f"Exported {domain}: {export(domain, args.out_dir, args.levels)}",
            flush=True)


if __name__ == "__main__":
    main()
