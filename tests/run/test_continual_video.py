"""Tests for predicators/run/continual_video.py and scripts/continual_video.py
against a real cover run."""
import os
from typing import Any

import imageio
import numpy as np

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run import continual_video as cv
from predicators.run.continual import ContinualRun
from predicators.run.controllers import create_controller
from predicators.run.scorecard import RunCard
from predicators.settings import CFG
from scripts import continual_video as script


def _config(tmp_path: Any, **overrides: Any) -> None:
    utils.reset_config({
        "env":
        "cover",
        "approach":
        "oracle",
        "seed":
        3,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "horizon":
        30,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        40,
        "continual_render":
        False,
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
        "experiment_id":
        "video",
        "video_fps":
        4,
        "video_dir":
        os.path.join(str(tmp_path), "videos"),
        **overrides,
    })


def _num_frames(path: str) -> int:
    reader: Any = imageio.get_reader(path)
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


def _finished_run(tmp_path: Any, **overrides: Any) -> ContinualRun:
    _config(tmp_path, **overrides)
    env = create_new_env("cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    run = ContinualRun(env, approach, create_controller(env, approach))
    run.run()
    return run


def test_read_level_episodes_matches_the_card(tmp_path: Any) -> None:
    """The action log and index reconstruct the card's episodes, with one
    invocation per skill the oracle ran."""
    run = _finished_run(tmp_path)
    card = run.card
    assert card.end_reason == "all_levels_won"
    for lv in card.levels:
        episodes = cv.read_level_episodes(
            os.path.join(run.run_dir, f"L{lv.index + 1:02d}"))
        assert [ep.index
                for ep in episodes] == [ep.index for ep in lv.episodes]
        assert episodes[0].opened_by == "level_start"
        for rec, ep in zip(lv.episodes, episodes):
            assert len(ep.actions) == rec.steps
            # Invocations tile the episode's steps in order.
            assert [inv.start for inv in ep.invocations] == \
                [0] + [inv.end for inv in ep.invocations[:-1]]
            assert ep.invocations[-1].end == rec.steps
            covered = {
                s
                for inv in ep.invocations for s in range(inv.start, inv.end)
            }
            assert covered == set(range(rec.steps))
            assert all(inv.skill for inv in ep.invocations)
        assert sum(len(ep.invocations) for ep in episodes) == \
            lv.skill_invocations
    assert cv.reset_cost_of(card.levels[0]) == 0


def test_invocation_lookup_and_reset_cost() -> None:
    """invocation_at covers [start, end); the reset price is recovered from a
    level's charged steps."""
    ep = cv.EpisodeRecord(0, "level_start")
    ep.invocations = [
        cv.Invocation("A(x)", "first", "succeeded", 0, 3),
        cv.Invocation("B(y)", "second", "failed", 3, 5),
    ]
    assert [(inv.skill if inv is not None else None)
            for inv in map(ep.invocation_at, range(6))] == \
        ["A(x)", "A(x)", "A(x)", "B(y)", "B(y)", None]
    lv = RunCard.from_dict({
        "run_id":
        "r",
        "env":
        "e",
        "seed":
        0,
        "arm":
        "a",
        "step_cap":
        10,
        "wall_clock_cap":
        1.0,
        "levels": [{
            "index":
            0,
            "split":
            "train",
            "task_idx":
            0,
            "goal": [],
            "steps":
            2 * 7 + 12,
            "resets":
            2,
            "episodes": [{
                "index": 0,
                "steps": 5
            }, {
                "index": 1,
                "steps": 4
            }, {
                "index": 2,
                "steps": 3
            }],
        }],
    }).levels[0]
    assert cv.reset_cost_of(lv) == 7


def test_panel_and_frame_geometry() -> None:
    """The panel is as tall as the render, sits to its right, and every label
    field renders (long text wraps and is cut)."""
    label = cv.FrameLabel(env="pybullet_x",
                          arm="agent",
                          seed=0,
                          level_index=1,
                          levels_total=2,
                          split="test",
                          task_idx=0,
                          goal_nl="Light the lamp " * 30,
                          goal=["LampOn(lamp0:lamp)"] * 12,
                          episode=2,
                          opened_by="agent",
                          skill="PressButton(robot, b0)",
                          note="Check whether the lamp charges. " * 20,
                          skill_status="succeeded",
                          level_steps=120,
                          run_steps=1120,
                          step_cap=10000,
                          level_resets=1,
                          run_resets=1,
                          reset_cost=1000,
                          resets_allowed=True,
                          state="GAME_OVER",
                          reason="horizon",
                          banner="GAME OVER: horizon",
                          banner_color=cv.PANEL_BAD)
    panel = cv.render_panel(label, 300)
    assert panel.shape == (300, cv.PANEL_WIDTH, 3)
    render = np.zeros((301, 251, 4), dtype=np.uint8)
    frame = cv.compose_frame(render, label)
    assert frame.shape == (302, 252 + cv.PANEL_WIDTH, 3)
    # The render occupies the left, the panel background the right.
    assert frame[10, 10].tolist() == [0, 0, 0]
    assert frame[10, 300].tolist() == list(cv.PANEL_BG)
    # A label with nothing running and a win state draws too.
    quiet = cv.FrameLabel(
        **{
            **label.__dict__, "skill": "",
            "note": "",
            "skill_status": "",
            "state": "WIN",
            "reason": "",
            "banner": "",
            "resets_allowed": False,
            "opened_by": "level_start"
        })
    assert cv.render_panel(quiet, 200).shape == (200, cv.PANEL_WIDTH, 3)


def test_make_run_video_replays_every_level(tmp_path: Any) -> None:
    """The video holds one frame per step (stride 1) plus the held banners and
    lands in the run directory as run.mp4; stride thins the steps."""
    run = _finished_run(tmp_path)
    card = run.card
    # pylint: disable-next=protected-access
    env = run._env
    path = cv.make_run_video(env, card, run.run_dir, stride=1, fps=4)
    assert path == os.path.join(run.run_dir, "run.mp4")
    assert os.path.isfile(path)
    n_frames = _num_frames(path)
    steps = card.total_steps
    hold = 4
    # Per level: a start banner (hold) and a win banner (2 * hold) replace
    # the plain frame of the winning step.
    expected = steps + sum(hold + 2 * hold - 1 for _ in card.levels)
    assert n_frames == expected
    # Cover's skills are one step each, so every step is an invocation
    # boundary and a stride cannot thin them: merge each episode's
    # invocations into one and only banners and episode ends remain.
    level = card.levels[0]
    episodes = cv.read_level_episodes(os.path.join(run.run_dir, "L01"))
    for ep in episodes:
        ep.invocations = [
            cv.Invocation("All", "", "succeeded", 0, len(ep.actions))
        ]
    frames = list(
        cv.iter_level_frames(env,
                             card,
                             level,
                             episodes,
                             stride=1000,
                             hold=hold))
    assert [repeat for _, repeat in frames] == [hold, 2 * hold]
    assert frames[0][0].shape == frames[1][0].shape


def test_make_run_video_with_nothing_recorded(tmp_path: Any) -> None:
    """A card whose levels have no recording yields no video."""
    run = _finished_run(tmp_path)
    card = run.card
    # pylint: disable-next=protected-access
    env = run._env
    assert cv.make_run_video(
        env, card, os.path.join(str(tmp_path), "none"), fps=4) is None


def test_script_from_flags_and_from_log(tmp_path: Any) -> None:
    """The offline script takes the run directory, rebuilds the env from the
    'Running command' line of its info.log (or from main.py flags when there is
    none), and writes run.mp4 into it."""
    run = _finished_run(tmp_path)
    flags = [
        "--env",
        "cover",
        "--approach",
        "oracle",
        "--seed",
        "3",
        "--experiment_id",
        "video",
        "--num_train_tasks",
        "1",
        "--num_test_tasks",
        "1",
        "--horizon",
        "30",
        "--experiment_protocol",
        "continual",
        "--continual_steps_per_level",
        "40",
        "--continual_runs_dir",
        os.path.join(str(tmp_path), "runs"),
        "--video_fps",
        "4",
    ]
    # No info.log in the run directory: the flags come from the command
    # line, and --out picks the file.
    out = os.path.join(str(tmp_path), "from_flags.mp4")
    assert script.main(["--run_dir", run.run_dir, *flags, "--out", out]) == out
    assert os.path.isfile(out)
    log = os.path.join(run.run_dir, "info.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write("\x1b[32mINFO: Running command: python predicators/main.py " +
                " ".join(flags) + "\x1b[0m\n")
        f.write("INFO: something else\n")
    assert script.command_flags_from_log(log) == flags
    expected = os.path.join(run.run_dir, "run.mp4")
    assert script.main(["--run_dir", run.run_dir]) == expected
    assert os.path.isfile(expected)
    bad = os.path.join(str(tmp_path), "bad.log")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("nothing here\n")
    try:
        script.command_flags_from_log(bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_end_hook_writes_the_video(tmp_path: Any) -> None:
    """Under continual_make_video, run_continual writes the video when the run
    ends."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.run.continual import run_continual
    _config(tmp_path, continual_make_video=True)
    env = create_new_env("cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    card = run_continual(env, approach)
    assert card.is_finished
    assert os.path.isfile(
        os.path.join(CFG.continual_runs_dir, CFG.run_subdir, "run.mp4"))
