"""A labelled video of one continual-protocol run (docs/continual-protocol.md).

The recording of a run holds every applied primitive action
(``actions.jsonl``) and every skill invocation, reset, win and game
over (``index.jsonl``) of each level. This module replays those
actions through the env, level by level, the way a resume does, and
writes one frame every ``continual_video_stride`` steps: the env
render on the left and, on the right, a panel with the level and its
goal, the skill running at that step and the agent's note for it, the
episode, the steps used against the cap and the resets. Level starts,
resets, wins and game overs hold a banner for a moment.

The video is ``run.mp4`` in the run's directory
(``predicators/run/paths.py``), beside the scorecard and the recordings
it was built from, where the continual viewer looks for it.

Run in-process at the end of a run under ``continual_make_video``, or
offline for a finished run with ``scripts/continual_video.py``.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from predicators.envs import BaseEnv
from predicators.run import paths
from predicators.run.episode import EpisodeRunner, EpisodeState
from predicators.run.recording import ACTIONS_FILENAME, INDEX_FILENAME
from predicators.run.scorecard import LevelCard, RunCard
from predicators.settings import CFG
from predicators.structs import Action

# Panel geometry. The panel is as tall as the render; its width keeps
# the whole frame a multiple of 16 for a 900 px render.
PANEL_WIDTH = 620
PANEL_BG = (24, 24, 30)
PANEL_FG = (235, 235, 240)
PANEL_MUTED = (150, 150, 160)
PANEL_ACCENT = (110, 170, 255)
PANEL_GOOD = (90, 200, 120)
PANEL_BAD = (235, 90, 90)
PANEL_WARN = (245, 180, 70)


@dataclass
class Invocation:
    """One skill invocation of an episode, in episode-local steps."""
    skill: str
    note: str
    status: str
    start: int
    end: int


@dataclass
class EpisodeRecord:
    """One recorded episode of a level: how it opened, its actions and the
    invocations that produced them."""
    index: int
    opened_by: str  # "level_start" | "agent" | "harness_reset"
    actions: List[np.ndarray] = field(default_factory=list)
    invocations: List[Invocation] = field(default_factory=list)

    def invocation_at(self, step: int) -> Optional[Invocation]:
        """The invocation whose steps cover episode step ``step`` (the step
        index of the action just applied), or the one that just ended."""
        for inv in self.invocations:
            if inv.start <= step < inv.end:
                return inv
        return None


def read_level_episodes(level_dir: str) -> List[EpisodeRecord]:
    """The episodes of a level from its action log, with the invocations of its
    index attached."""
    episodes: List[EpisodeRecord] = []
    actions_path = os.path.join(level_dir, ACTIONS_FILENAME)
    if os.path.isfile(actions_path):
        with open(actions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("event") == "reset":
                    episodes.append(
                        EpisodeRecord(int(rec["ep"]),
                                      str(rec.get("by", "agent"))))
                elif episodes:
                    episodes[-1].actions.append(
                        np.array(rec["a"], dtype=np.float32))
    by_index = {ep.index: ep for ep in episodes}
    # Invocations partition an episode's steps in order: their episode-
    # local ranges follow from their step counts, whichever convention
    # the entry's start_step/end_step used.
    cursor: Dict[int, int] = {}
    index_path = os.path.join(level_dir, INDEX_FILENAME)
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("event") != "invoke":
                    continue
                ep = by_index.get(int(entry.get("episode", 0)))
                if ep is None:
                    continue
                start = cursor.get(ep.index, 0)
                end = start + int(entry.get("steps", 0))
                cursor[ep.index] = end
                ep.invocations.append(
                    Invocation(str(entry.get("skill", "")),
                               str(entry.get("note", "")),
                               str(entry.get("status", "")), start, end))
    return episodes


def reset_cost_of(level: LevelCard) -> int:
    """The steps one agent reset was charged on ``level``, recovered from its
    card so an offline replay needs no flag (0 when the level had none)."""
    if level.resets <= 0:
        return 0
    charged = level.steps - sum(ep.steps for ep in level.episodes)
    return max(0, charged // level.resets)


@dataclass
class FrameLabel:
    """Everything the panel shows for one frame."""
    env: str
    arm: str
    seed: int
    level_index: int
    levels_total: int
    split: str
    task_idx: int
    goal_nl: str
    goal: List[str]
    episode: int
    opened_by: str
    skill: str
    note: str
    skill_status: str  # "" while running, else the invocation's status
    level_steps: int
    run_steps: int
    step_cap: int
    level_resets: int
    run_resets: int
    reset_cost: int
    resets_allowed: bool
    state: str  # NOT_FINISHED | WIN | GAME_OVER
    reason: str
    banner: str = ""
    banner_color: Tuple[int, int, int] = PANEL_ACCENT


class _Fonts:
    """DejaVu from matplotlib's bundle (present wherever matplotlib is),
    falling back to PIL's bitmap font."""

    def __init__(self) -> None:
        self.title = self._load("DejaVuSans-Bold.ttf", 30)
        self.head = self._load("DejaVuSans-Bold.ttf", 23)
        self.body = self._load("DejaVuSans.ttf", 21)
        self.small = self._load("DejaVuSans.ttf", 18)
        self.banner = self._load("DejaVuSans-Bold.ttf", 34)

    @staticmethod
    def _load(name: str, size: int) -> Any:
        try:
            # pylint: disable-next=import-outside-toplevel
            import matplotlib
            path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf",
                                name)
            return ImageFont.truetype(path, size)  # type: ignore
        except Exception:  # pylint: disable=broad-except
            try:
                return ImageFont.load_default(size)
            except TypeError:  # Pillow < 10.1 has no size argument
                return ImageFont.load_default()


_FONTS: Optional[_Fonts] = None


def _fonts() -> _Fonts:
    global _FONTS  # pylint: disable=global-statement
    if _FONTS is None:
        _FONTS = _Fonts()
    return _FONTS


def _wrap(draw: Any, text: str, font: Any, width: int,
          max_lines: int) -> List[str]:
    """Word-wrap ``text`` to ``width`` pixels, at most ``max_lines`` lines, an
    ellipsis on the last line when cut."""
    lines: List[str] = []
    for para in text.splitlines() or [""]:
        words = para.split()
        line = ""
        for word in words:
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=font) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".…") + "…"
    return lines


def render_panel(label: FrameLabel,
                 height: int,
                 width: int = PANEL_WIDTH) -> np.ndarray:
    """The label panel of one frame as an RGB array."""
    fonts = _fonts()
    img = Image.new("RGB", (width, height), PANEL_BG)
    draw: Any = ImageDraw.Draw(img)
    pad = 24
    text_w = width - 2 * pad
    y = pad

    def line(text: str,
             font: Any,
             color: Tuple[int, int, int] = PANEL_FG,
             gap: int = 6) -> None:
        nonlocal y
        draw.text((pad, y), text, font=font, fill=color)
        y += int(font.size * 1.25) + gap

    def block(text: str,
              font: Any,
              color: Tuple[int, int, int],
              max_lines: int,
              gap: int = 6) -> None:
        for row in _wrap(draw, text, font, text_w, max_lines):
            line(row, font, color, gap=0)
        y_add(gap)

    def y_add(dy: int) -> None:
        nonlocal y
        y += dy

    def rule() -> None:
        nonlocal y
        draw.line([(pad, y), (width - pad, y)], fill=(60, 60, 70), width=1)
        y += 14

    # Header: env, arm and seed.
    line(f"{label.env}  ·  {label.arm}  ·  seed {label.seed}", fonts.small,
         PANEL_MUTED)
    line(f"Level {label.level_index + 1} / {label.levels_total}",
         fonts.title,
         PANEL_FG,
         gap=0)
    line(f"{label.split} task {label.task_idx}", fonts.small, PANEL_MUTED)
    rule()

    # Goal.
    line("Goal", fonts.head, PANEL_ACCENT, gap=2)
    if label.goal_nl:
        block(label.goal_nl, fonts.body, PANEL_FG, max_lines=4)
    block(", ".join(label.goal), fonts.small, PANEL_MUTED, max_lines=3)
    rule()

    # Current action.
    line("Action", fonts.head, PANEL_ACCENT, gap=2)
    if label.skill:
        block(label.skill, fonts.body, PANEL_FG, max_lines=2, gap=2)
        if label.skill_status:
            color = {
                "succeeded": PANEL_GOOD,
                "failed": PANEL_BAD,
                "interrupted": PANEL_WARN
            }.get(label.skill_status, PANEL_MUTED)
            line(label.skill_status, fonts.small, color)
        if label.note:
            block(label.note, fonts.small, PANEL_MUTED, max_lines=4)
    else:
        line("(no skill running)", fonts.body, PANEL_MUTED)
    rule()

    # Budget.
    line("Budget", fonts.head, PANEL_ACCENT, gap=2)
    line(f"Steps  {label.run_steps} / {label.step_cap}", fonts.body)
    bar_h = 12
    frac = 0.0 if label.step_cap <= 0 else min(
        1.0, label.run_steps / label.step_cap)
    draw.rectangle([pad, y, width - pad, y + bar_h], fill=(50, 50, 60))
    if frac > 0:
        draw.rectangle([pad, y, pad + int(text_w * frac), y + bar_h],
                       fill=PANEL_ACCENT)
    y_add(bar_h + 12)
    line(f"this level {label.level_steps} steps", fonts.small, PANEL_MUTED)
    resets = f"Resets  level {label.level_resets}, run {label.run_resets}"
    line(resets, fonts.body)
    if label.resets_allowed and label.reset_cost > 0:
        unit = "step" if label.reset_cost == 1 else "steps"
        line(f"a reset costs {label.reset_cost} {unit}", fonts.small,
             PANEL_MUTED)
    elif not label.resets_allowed:
        line("no resets on this level", fonts.small, PANEL_MUTED)
    rule()

    # Episode and its state.
    opened = {
        "level_start": "level start",
        "agent": "agent reset",
        "harness_reset": "harness reset"
    }.get(label.opened_by, label.opened_by)
    line(f"Episode {label.episode}  ({opened})", fonts.body)
    if label.state == "WIN":
        line("WIN", fonts.head, PANEL_GOOD)
    elif label.state == "GAME_OVER":
        block(f"GAME OVER: {label.reason}" if label.reason else "GAME OVER",
              fonts.head,
              PANEL_BAD,
              max_lines=2)
    else:
        line("in progress", fonts.small, PANEL_MUTED)

    # Banner, bottom of the panel.
    if label.banner:
        rows = _wrap(draw, label.banner, fonts.banner, text_w, 2)
        box_h = len(rows) * int(fonts.banner.size * 1.3) + 2 * 14
        top = height - pad - box_h
        draw.rectangle([pad, top, width - pad, top + box_h],
                       fill=label.banner_color)
        yy = top + 14
        for row in rows:
            draw.text((pad + 14, yy), row, font=fonts.banner, fill=PANEL_BG)
            yy += int(fonts.banner.size * 1.3)
    return np.asarray(img, dtype=np.uint8)


def compose_frame(render: np.ndarray, label: FrameLabel) -> np.ndarray:
    """The render with the label panel to its right, even-sized for the
    encoder."""
    rgb = np.asarray(render)[..., :3].astype(np.uint8)
    panel = render_panel(label, rgb.shape[0])
    frame = np.concatenate([rgb, panel], axis=1)
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        frame = np.pad(frame, ((0, h % 2), (0, w % 2), (0, 0)))
    return frame


class _Writer:
    """Streams frames to an mp4 so a long run never sits in memory."""

    def __init__(self, path: str, fps: int) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Per-process temp name: two writers of the same run (a rerun
        # racing a run-end hook) must never interleave into one file.
        self._tmp = f"{path}.{os.getpid()}.tmp.mp4"
        self._path = path
        self._writer = imageio.get_writer(
            self._tmp,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=1,
            # moov up front so a browser can seek before the download ends.
            output_params=["-movflags", "+faststart"])
        self.frames = 0

    def append(self, frame: np.ndarray, repeat: int = 1) -> None:
        """Write ``frame`` ``repeat`` times."""
        for _ in range(max(1, repeat)):
            self._writer.append_data(frame)
            self.frames += 1

    def close(self) -> None:
        """Finish the file and move it into place."""
        self._writer.close()
        os.replace(self._tmp, self._path)


def _level_env(env: BaseEnv) -> Tuple[BaseEnv, Optional[BaseEnv]]:
    """The env a level is replayed in: a fresh instance when the run used one
    per level, else the run env; the fresh instance to dispose."""
    if CFG.test_fresh_env_per_episode:
        fresh = env.make_fresh_test_instance()
        if fresh is not None:
            return fresh, fresh
    return env, None


def _render(env: BaseEnv) -> Optional[np.ndarray]:
    try:
        frames = env.render()
    except Exception as e:  # pylint: disable=broad-except
        logging.debug("[Continual video] render skipped: %s", e)
        return None
    if not frames:
        return None
    return np.asarray(frames[0])


def iter_level_frames(env: BaseEnv, card: RunCard, level: LevelCard,
                      episodes: Sequence[EpisodeRecord], stride: int,
                      hold: int) -> Iterator[Tuple[np.ndarray, int]]:
    """Replay one level and yield ``(frame, repeat)`` pairs.

    ``hold`` is how many times a banner frame repeats. Steps of previous
    levels come from the card; the level's own steps and resets are
    recounted from the replay so the panel matches what the agent saw.
    """
    run_steps_before = sum(lv.steps for lv in card.levels[:level.index])
    run_resets_before = sum(lv.resets for lv in card.levels[:level.index])
    reset_cost = reset_cost_of(level)
    resets_allowed = level.split == "train" or bool(
        CFG.continual_allow_test_resets)
    level_env, fresh = _level_env(env)
    runner = EpisodeRunner(level_env,
                           horizon=CFG.continual_episode_horizon,
                           max_option_steps=CFG.max_num_steps_option_rollout)
    level_steps = 0
    level_resets = 0
    # How the scorecard says each episode ended ("game_over:<reason>",
    # "win", "reset", "in_progress"), for a game over the replay's
    # runner does not raise itself.
    recorded_end = {rec.index: rec.end for rec in level.episodes}
    try:
        for ep in episodes:
            runner.finish()
            runner.reset(level.split, level.task_idx)
            if ep.opened_by == "agent":
                level_resets += 1
                level_steps += reset_cost

            def label(step: int,
                      banner: str = "",
                      color: Tuple[int, int, int] = PANEL_ACCENT,
                      ep: EpisodeRecord = ep) -> FrameLabel:
                inv = ep.invocation_at(step)
                ended = None
                if inv is None and step > 0:
                    # Between invocations: show the one that just ended.
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
                    state=runner.episode_state.value,
                    reason=runner.reason,
                    banner=banner,
                    banner_color=color,
                )

            first = _render(level_env)
            if first is not None:
                if ep.opened_by == "level_start":
                    yield compose_frame(
                        first,
                        label(
                            0, f"Level {level.index + 1}: "
                            f"{level.split} task {level.task_idx}")), hold
                elif ep.opened_by == "agent":
                    yield compose_frame(first,
                                        label(0, "RESET by agent",
                                              PANEL_WARN)), hold
                else:
                    yield compose_frame(
                        first, label(0, "RESET by harness", PANEL_WARN)), hold
            n = len(ep.actions)
            for i, arr in enumerate(ep.actions):
                if runner.episode_state is not EpisodeState.NOT_FINISHED:
                    logging.warning(
                        "[Continual video] level %d episode %d: replay "
                        "ended after %d of %d steps (%s); the rest is "
                        "skipped", level.index + 1, ep.index, i, n,
                        runner.reason)
                    break
                outcome = runner.step(Action(arr))
                level_steps += 1
                terminal = outcome.state is not EpisodeState.NOT_FINISHED
                boundary = any(inv.end == i + 1 for inv in ep.invocations)
                if not (terminal or boundary or i + 1 == n or
                        (i + 1) % stride == 0):
                    continue
                render = _render(level_env)
                if render is None:
                    continue
                if outcome.state is EpisodeState.WIN:
                    yield compose_frame(render,
                                        label(i, "LEVEL WON",
                                              PANEL_GOOD)), 2 * hold
                elif outcome.state is EpisodeState.GAME_OVER:
                    yield compose_frame(
                        render,
                        label(i, f"GAME OVER: {outcome.reason}",
                              PANEL_BAD)), 2 * hold
                elif i + 1 == n and recorded_end.get(
                        ep.index, "").startswith("game_over:"):
                    # A recording made under an episode horizon the
                    # current run has none of: the runner replays past
                    # it, the scorecard still says how the episode ended.
                    reason = recorded_end[ep.index].split(":", 1)[1]
                    yield compose_frame(
                        render, label(i, f"GAME OVER: {reason}",
                                      PANEL_BAD)), 2 * hold
                else:
                    yield compose_frame(render, label(i)), 1
    finally:
        runner.finish()
        if fresh is not None:
            fresh.dispose()


def make_run_video(env: BaseEnv,
                   card: RunCard,
                   run_dir: str,
                   out_path: Optional[str] = None,
                   stride: Optional[int] = None,
                   fps: Optional[int] = None) -> Optional[str]:
    """Write the video of the run in ``run_dir`` (its level recordings) to
    ``run_dir/run.mp4``, or ``out_path``; returns the path, or ``None`` when
    the run has no attempted level with a recording."""
    stride = int(CFG.continual_video_stride if stride is None else stride)
    fps = int(CFG.video_fps if fps is None else fps)
    out_path = out_path or paths.video_path(run_dir)
    hold = max(1, fps)
    writer: Optional[_Writer] = None
    try:
        for level in card.levels:
            if not level.attempted:
                continue
            level_dir = paths.level_dir(run_dir, level.index)
            if not os.path.isdir(level_dir):
                continue
            episodes = read_level_episodes(level_dir)
            if not episodes:
                continue
            logging.info("[Continual video] level %d: %d episodes, %d steps",
                         level.index + 1, len(episodes),
                         sum(len(ep.actions) for ep in episodes))
            for frame, repeat in iter_level_frames(env, card, level, episodes,
                                                   stride, hold):
                if writer is None:
                    writer = _Writer(out_path, fps)
                writer.append(frame, repeat)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        logging.info("[Continual video] nothing to replay for %s", card.run_id)
        return None
    logging.info("[Continual video] wrote %s (%d frames at %d fps)", out_path,
                 writer.frames, fps)
    return out_path
