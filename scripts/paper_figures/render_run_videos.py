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
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple

import numpy as np
from export_static_scenes import LOGS, _load_run_config, \
    _migrate_balloons_layout, _migrate_fan_layout
from export_trajectory_scenes import _canonical_state

from predicators.envs import create_new_env
from predicators.run import paths
from predicators.run.continual_video import PANEL_ACCENT, PANEL_BAD, \
    PANEL_GOOD, PANEL_WARN, EpisodeRecord, FrameLabel, _Writer, \
    compose_frame, read_level_episodes, reset_cost_of
from predicators.run.episode import EpisodeState
from predicators.run.scorecard import LevelCard, RunCard
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
    state = adjust(env, _canonical_state(env, recorded), initial)
    env._set_state(state)  # pylint: disable=protected-access
    env._current_observation = state  # pylint: disable=protected-access
    return np.asarray(env.render()[0])


def _level_frames(env: Any, card: RunCard, level: LevelCard,
                  episodes: Sequence[EpisodeRecord],
                  recorded: Dict[int, Dict[str, Any]], adjust: Adjust,
                  stride: int, hold: int) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield ``(frame, repeat)`` pairs for one level, choosing frames and
    labels as continual_video.iter_level_frames does."""
    run_steps_before = sum(lv.steps for lv in card.levels[:level.index])
    run_resets_before = sum(lv.resets for lv in card.levels[:level.index])
    reset_cost = reset_cost_of(level)
    resets_allowed = level.split == "train" or bool(
        CFG.continual_allow_test_resets)
    initial = env.reset(level.split, level.task_idx)
    level_steps = 0
    level_resets = 0
    for ep in episodes:
        states: List[State] = recorded[ep.index]["states"]
        end = str(recorded[ep.index]["end"])
        assert len(states) == len(ep.actions) + 1, (level.index, ep.index)
        if ep.opened_by == "agent":
            level_resets += 1
            level_steps += reset_cost

        def label(step: int,
                  outcome: EpisodeState = EpisodeState.NOT_FINISHED,
                  reason: str = "",
                  banner: str = "",
                  color: Tuple[int, int, int] = PANEL_ACCENT,
                  ep: EpisodeRecord = ep) -> FrameLabel:
            inv = ep.invocation_at(step)
            ended = None
            if inv is None and step > 0:
                ended = ep.invocation_at(step - 1)
            cur = inv if inv is not None else ended
            return FrameLabel(
                env=card.env,
                arm=card.arm,
                seed=card.seed,
                level_index=level.index,
                levels_total=card.levels_total,
                split=level.split,
                task_idx=level.task_idx,
                goal_nl=level.goal_nl,
                goal=list(level.goal),
                episode=ep.index,
                opened_by=ep.opened_by,
                skill=cur.skill if cur is not None else "",
                note=cur.note if cur is not None else "",
                skill_status=(ended.status if ended is not None else ""),
                level_steps=level_steps,
                run_steps=run_steps_before + level_steps,
                step_cap=card.step_cap,
                level_resets=level_resets,
                run_resets=run_resets_before + level_resets,
                reset_cost=reset_cost,
                resets_allowed=resets_allowed,
                state=outcome.value,
                reason=reason,
                banner=banner,
                banner_color=color,
            )

        first = _show(env, states[0], initial, adjust)
        if ep.opened_by == "level_start":
            banner = f"Level {level.index + 1}: {level.split} task " \
                f"{level.task_idx}"
            yield compose_frame(first, label(0, banner=banner)), hold
        elif ep.opened_by == "agent":
            yield compose_frame(
                first, label(0, banner="RESET by agent",
                             color=PANEL_WARN)), hold
        else:
            yield compose_frame(
                first, label(0, banner="RESET by harness",
                             color=PANEL_WARN)), hold
        n = len(ep.actions)
        for i in range(n):
            level_steps += 1
            last = i + 1 == n
            boundary = any(inv.end == i + 1 for inv in ep.invocations)
            if not (last or boundary or (i + 1) % stride == 0):
                continue
            render = _show(env, states[i + 1], initial, adjust)
            if last and end == "win":
                yield compose_frame(
                    render,
                    label(i,
                          EpisodeState.WIN,
                          banner="LEVEL WON",
                          color=PANEL_GOOD)), 2 * hold
            elif last and end.startswith("game_over"):
                reason = end.split(":", 1)[1] if ":" in end else ""
                yield compose_frame(
                    render,
                    label(i,
                          EpisodeState.GAME_OVER,
                          reason,
                          banner=f"GAME OVER: {reason}",
                          color=PANEL_BAD)), 2 * hold
            else:
                yield compose_frame(render, label(i)), 1


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
            for frame, repeat in _level_frames(env, card, level, episodes,
                                               recorded, ADJUST[domain],
                                               stride, hold):
                writer.append(frame, repeat)
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
