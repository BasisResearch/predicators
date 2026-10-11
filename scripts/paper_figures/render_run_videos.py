"""Render labelled run videos from recorded states in the current scene
layouts.

The harness's run.mp4 (predicators/run/continual_video.py) replays a
run's actions through the env of the run's own code. The Balloons and
Fan scenes have changed since their paper runs were recorded: the Fan
platforms, fan banks and camera moved, and the Balloons chute moved and
gained a red cap for its ceiling. Replaying those actions in the current
env would not reproduce the runs. This script instead restores every
recorded state, moves it into the current layout exactly as the paper
figures do (export_static_scenes.py), renders it with the env's own
PyBullet camera, and writes the harness video's labelled frames: the
render with a panel showing the level, the skill and the agent's note
for it, the steps and the resets.

Usage (from the repository root, with PYTHONPATH=.):
    python scripts/paper_figures/render_run_videos.py --domains Fan \
        --out-dir logs/paper_run_videos
"""
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from export_static_scenes import LOGS, _load_run_config, \
    _migrate_balloons_layout, _migrate_fan_layout

from predicators.envs import create_new_env
from predicators.run import paths
from predicators.run.continual_video import _Writer, compose_frame, \
    iter_recorded_level_frames, read_level_episodes
from predicators.run.cycles_video import canonical_state
from predicators.run.scorecard import RunCard
from predicators.settings import CFG
from predicators.structs import State

# The runs behind the paper's trajectory figures (data/trajectories/
# stripes.json), whose videos the project page shows.
RUNS: Dict[str, str] = {
    "Balloons": "balloons-mb_opus_benchmark_r2/seed3/run_20260919_124956",
    "Fan": "fan_ramp-mb_opus_ramp_skill_repair_r1/seed2/run_20260921_090827",
}

Adjust = Callable[[Any, State, State], State]


def _move_fan(env: Any, state: State, initial: State) -> State:
    return _migrate_fan_layout(env, state, initial)


def _move_balloons(env: Any, state: State, initial: State) -> State:
    del initial  # The chute layout needs no reference state.
    return _migrate_balloons_layout(env, state)


ADJUST: Dict[str, Adjust] = {"Balloons": _move_balloons, "Fan": _move_fan}


def _show(env: Any, recorded: State, initial: State,
          adjust: Adjust) -> np.ndarray:
    """Render one recorded state in the current layout."""
    state = adjust(env, canonical_state(env, recorded), initial)
    env._set_state(state)  # pylint: disable=protected-access
    env._current_observation = state  # pylint: disable=protected-access
    return np.asarray(env.render()[0])


def render_run(domain: str, out_dir: Path, stride: int, fps: int) -> Path:
    """Write ``out_dir/<domain>.mp4`` for the domain's paper run."""
    run_dir = LOGS / RUNS[domain]
    _load_run_config(run_dir)
    card = RunCard.load(paths.scorecard_path(str(run_dir)))
    env: Any = create_new_env(CFG.env, do_cache=False, use_gui=False)
    out = out_dir / f"{domain.lower()}.mp4"
    writer = _Writer(str(out), fps)
    hold = max(1, fps)
    try:
        for level in card.levels:
            level_dir = Path(paths.level_dir(str(run_dir), level.index))
            if not level.attempted or not level_dir.is_dir():
                continue
            episodes = read_level_episodes(str(level_dir))
            # Trusted local experiment record.
            with open(level_dir / "episodes.pkl", "rb") as f:
                recorded = {int(ep["episode"]): ep for ep in pickle.load(f)}
            adjust = ADJUST[domain]

            def show(state: State,
                     initial: State,
                     adjust: Adjust = adjust) -> np.ndarray:
                return _show(env, state, initial, adjust)

            for frame, label, repeat in iter_recorded_level_frames(
                    env, card, level, episodes, recorded, show, stride, hold):
                writer.append(compose_frame(frame, label), repeat)
            logging.info("%s level %d: %d frames so far", domain,
                         level.index + 1, writer.frames)
    finally:
        writer.close()
        env.dispose()
    print(f"Wrote {out} ({writer.frames} frames at {fps} fps)", flush=True)
    return out


def main() -> None:
    """Render the selected domains' run videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="+", default=list(RUNS))
    parser.add_argument("--out-dir",
                        type=Path,
                        default=LOGS.parent / "paper_run_videos")
    parser.add_argument("--stride",
                        type=int,
                        default=None,
                        help="steps per frame; defaults to the run's own")
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for domain in args.domains:
        # The run's own flags decide the stride and frame rate unless
        # given here, as for the harness video.
        _load_run_config(LOGS / RUNS[domain])
        stride = args.stride or int(CFG.continual_video_stride)
        fps = args.fps or int(CFG.video_fps)
        render_run(domain, args.out_dir, stride, fps)


if __name__ == "__main__":
    main()
