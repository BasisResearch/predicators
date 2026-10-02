"""Tests for predicators/run/cycles_video.py and scripts/cycles_video.py.

A real pybullet_cover run supplies the recorded states; the Blender
render itself needs bpy on Python 3.11, so these tests stand in
placeholder frames for its output and check everything around it.
"""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List

import imageio
import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run import cycles_video as cyc
from predicators.run import paths
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.settings import CFG
from predicators.structs import State
from scripts import continual_video as continual_script
from scripts import cycles_video as script


def _config(tmp_path: Any, **overrides: Any) -> None:
    utils.reset_config({
        "env":
        "pybullet_cover",
        "approach":
        "oracle",
        "seed":
        0,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "horizon":
        300,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        400,
        "continual_render":
        False,
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
        "experiment_id":
        "cycles",
        "video_fps":
        4,
        "pybullet_camera_width":
        64,
        "pybullet_camera_height":
        48,
        **overrides,
    })


def _finished_run(tmp_path: Any, **overrides: Any) -> ContinualRun:
    _config(tmp_path, **overrides)
    env = create_new_env("pybullet_cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    run = ContinualRun(env, approach, create_level_player(env, approach))
    run.run()
    assert run.card.is_finished
    return run


def _fake_render(manifest_path: Path) -> None:
    """Stand in for render_cycles_frames.py: one PNG per scene."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest_path.parent / "frames"
    frames.mkdir(exist_ok=True)
    image = np.zeros((manifest["height"], manifest["width"], 3), np.uint8)
    for entry in manifest["frames"]:
        imageio.imwrite(frames / entry["scene"].replace(".json.gz", ".png"),
                        image)


def _num_frames(path: Path) -> int:
    reader: Any = imageio.get_reader(str(path))
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


def test_export_run_scenes_matches_the_run_video(tmp_path: Any) -> None:
    """The manifest holds the run video's frames, labels and holds, one scene
    per recorded step shown; the videos assemble from rendered frames."""
    run = _finished_run(tmp_path)
    manifest_path = cyc.export_run_scenes(run.card, run.run_dir)
    assert manifest_path == Path(paths.cycles_dir(run.run_dir),
                                 "manifest-all.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    assert manifest["domain"] == "cover"
    assert manifest["fps"] == 4
    # Each level opens with its start banner, held a second, and ends on
    # the win, held two.
    for level in run.card.levels:
        own = [f for f in frames if f["level"] == level.index]
        assert own[0]["repeat"] == 4
        assert own[0]["label"]["banner"].startswith(f"Level {level.index + 1}")
        assert own[-1]["repeat"] == 8
        assert own[-1]["label"]["banner"] == "LEVEL WON"
    # Scenes are named by the recorded step they show and hold the camera
    # and the visual shapes.
    names = {f["scene"] for f in frames}
    assert all(name.startswith("L0") for name in names)
    with gzip.open(manifest_path.parent / "scenes" / sorted(names)[0],
                   "rt") as f:
        scene = json.load(f)
    assert scene["camera"]["width"] == 64
    assert any(shape["name"] for shape in scene["shapes"])
    # The test-only export shares the run export's scenes.
    test_path = cyc.export_run_scenes(run.card, run.run_dir, levels="test")
    test_frames = json.loads(test_path.read_text(encoding="utf-8"))["frames"]
    assert {f["scene"] for f in test_frames} <= names
    # Assemble from placeholder frames: the panel video holds every frame
    # its manifest holds, and the test clip only the last level.
    with pytest.raises(AssertionError):
        cyc.compose_video(manifest_path)
    _fake_render(manifest_path)
    video = cyc.compose_video(manifest_path, panel=True)
    assert video.name == "cover.mp4"
    assert _num_frames(video) == sum(f["repeat"] for f in frames)
    clip = cyc.compose_video(manifest_path, test_only=True)
    assert clip.name == "cover-test-scene.mp4"
    assert _num_frames(clip) == sum(f["repeat"] for f in test_frames)


def test_export_state_scenes_and_script(tmp_path: Any,
                                        monkeypatch: Any) -> None:
    """A recorded sequence of states exports one frame per state, the last
    held; scripts/cycles_video.py assembles its scene video."""
    _config(tmp_path)
    env = create_new_env("pybullet_cover", do_cache=False)
    state = env.reset("test", 0)
    states: List[State] = [state, state.copy()]
    out = Path(tmp_path, "trajectory")
    manifest_path = cyc.export_state_scenes(states,
                                            str(out),
                                            fps=4,
                                            name="cover-trajectory",
                                            hold_last=3)
    frames = json.loads(manifest_path.read_text(encoding="utf-8"))["frames"]
    assert [f["repeat"] for f in frames] == [1, 4]
    assert len({f["scene"] for f in frames}) == 2
    _fake_render(manifest_path)
    monkeypatch.setattr(
        sys, "argv",
        ["cycles_video.py", str(out), "--no-render"])
    script.main()
    assert _num_frames(out / "cover-trajectory-scene.mp4") == 5
    assert script.blender_command("/opt/blender/python") == [
        "/opt/blender/python"
    ]
    assert script.blender_command("")[-1] == "python"


def test_boil_heat_follows_the_burner() -> None:
    """Heat accumulates on a lit burner at heating_speed per step and must
    agree with the recorded bubbling level."""
    utils.reset_config({"env": "pybullet_boil", "num_test_tasks": 1})
    env: Any = create_new_env("pybullet_boil", do_cache=False)
    state = env.reset("test", 0)
    jug = state.get_objects(env._jug_type)[0]  # pylint: disable=protected-access
    burner = state.get_objects(env._burner_type)[0]  # pylint: disable=protected-access
    lit = state.copy()
    lit.set(jug, "x", lit.get(burner, "x"))
    lit.set(jug, "y", lit.get(burner, "y"))
    lit.set(jug, "water_volume", 1.0)
    lit.set(burner, "is_on", 1.0)
    track = cyc.boil_heat(env, [state, lit, lit.copy(), lit.copy()])
    # The burner is off before the first step, so heating starts at the
    # second.
    assert track[1].get(jug.name, 0.0) == 0.0
    assert track[3][jug.name] == pytest.approx(2 * env.heating_speed)
    bubbling = lit.copy()
    bubbling.set(jug, "bubbling_level", 0.5)
    with pytest.raises(AssertionError):
        cyc.boil_heat(env, [state, bubbling])
    env.dispose()


def test_run_end_hook_and_offline_export(tmp_path: Any) -> None:
    """Under video_cycles_scenes, run_continual exports the scenes when the run
    ends; scripts/continual_video.py --cycles exports them for an earlier
    run."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.run.continual import run_continual
    _config(tmp_path, video_cycles_scenes=True)
    env = create_new_env("pybullet_cover", do_cache=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    card = run_continual(env, approach)
    assert card.is_finished
    run_dir = os.path.join(CFG.continual_runs_dir, CFG.run_subdir)
    assert os.path.isfile(
        os.path.join(paths.cycles_dir(run_dir), "manifest-all.json"))
    out = os.path.join(str(tmp_path), "offline")
    manifest = continual_script.main([
        "--run_dir", run_dir, "--cycles", "--out", out, "--env",
        "pybullet_cover", "--approach", "oracle", "--seed", "0",
        "--num_train_tasks", "1", "--num_test_tasks", "1",
        "--experiment_protocol", "continual", "--pybullet_camera_width", "64",
        "--pybullet_camera_height", "48"
    ])
    assert manifest == os.path.join(out, "manifest-all.json")


def test_test_videos_record_cycles_scenes(tmp_path: Any) -> None:
    """Under video_cycles_scenes, a test video's monitor records a Cycles scene
    of every frame as it renders, saved beside the video."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.run import testing
    _config(tmp_path,
            make_test_videos=True,
            video_cycles_scenes=True,
            video_dir=os.path.join(str(tmp_path), "videos"))
    with cyc.record_procedural_meshes():
        env = create_new_env("pybullet_cover", do_cache=False)
        monitor = testing._make_monitor(env)  # pylint: disable=protected-access
        assert isinstance(monitor, cyc.CyclesSceneMonitor)
        obs = env.reset("test", 0)
        monitor.reset("test", 0)
        monitor.observe(obs, None)
        monitor.observe(obs, None)
        artifacts = testing.TestArtifacts(None)
        artifacts.save_video(monitor, False, 0)
    video = Path(CFG.video_dir).glob("*.mp4")
    stem = next(video).name[:-len(".mp4")]
    manifest = Path(CFG.video_dir, f"{stem}_cycles", "manifest-all.json")
    frames = json.loads(manifest.read_text(encoding="utf-8"))["frames"]
    assert [f["repeat"] for f in frames] == [1, 1]
    assert all(
        (manifest.parent / "scenes" / f["scene"]).is_file() for f in frames)
    # Without the flag the monitor is the plain video monitor.
    utils.update_config({"video_cycles_scenes": False})
    assert not isinstance(
        testing._make_monitor(env),  # pylint: disable=protected-access
        cyc.CyclesSceneMonitor)


def test_kept_recording_closes_at_exit() -> None:
    """A process that keeps the mesh recording exits without the error its
    cleanup raised when the interpreter's teardown ran it."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        sys.executable, "-c",
        "from predicators.run.cycles_video import keep_procedural_meshes; "
        "keep_procedural_meshes()"
    ],
                            cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root)),
                            capture_output=True,
                            text=True,
                            check=True)
    assert "Exception ignored" not in result.stderr
