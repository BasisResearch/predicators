"""Named particles: every env labels its objects, points are sane, files
round-trip. Guards DEBUG_LOG entries 1, 2 and 5."""
import json

import numpy as np
import pytest

from predicators.envs.pybullet_airport import PyBulletAirportEnv
from predicators.envs.pybullet_donut import PyBulletDonutEnv
from predicators.envs.pybullet_plug_outlet import PyBulletPlugOutletEnv

from agent_robot_control.sim.perception import extract_named_particles, \
    farthest_point_downsample, flatten_particles
from agent_robot_control.tests.conftest import make_env

ENVS = [
    ("pybullet_donut", PyBulletDonutEnv, {"donut_0", "target", "robot", "table"}),
    ("pybullet_airport", PyBulletAirportEnv,
     {"item_0", "conveyor", "button", "table", "robot"}),
    ("pybullet_plug_outlet", PyBulletPlugOutletEnv,
     {"plug", "outlet", "holder", "robot", "table"}),
]


@pytest.mark.parametrize("name,cls,expected", ENVS)
def test_named_particles_are_sane(name, cls, expected):
    env = make_env(name, cls, num_train_tasks=1)
    s = env.reset("train", 0)
    snap = extract_named_particles(env, max_points_per_object=32,
                                   rng=np.random.default_rng(0))
    assert expected <= set(snap.names)
    visible = [n for n in snap.names if snap.num_points(n) > 0]
    assert len(visible) >= 3
    cam = np.array(env._camera_target)
    for n in visible:
        pts = snap.points[n]
        assert pts.shape[1] == 3
        # Guard against the OpenBLAS back-projection bug (DEBUG_LOG 1).
        assert np.linalg.norm(pts - cam, axis=1).max() < 2.0, n
    # Requested count honoured for a large object (DEBUG_LOG 2).
    assert snap.num_points("table") == 32
    # Centroid of a small object is near its true position.
    for obj in env._objects:
        if obj.name in visible and "x" in obj.type.feature_names \
                and obj.type.name != "robot":
            gt = np.array([s.get(obj, f) for f in "xyz"])
            err = np.linalg.norm(snap.points[obj.name].mean(0) - gt)
            assert err < 0.05, (obj.name, err)
            break


def test_snapshot_save_roundtrip(tmp_path):
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    env.reset("train", 0)
    snap = extract_named_particles(env, max_points_per_object=16,
                                   interaction_count=7)
    npz, js = tmp_path / "p.npz", tmp_path / "p.json"
    summary = snap.save(npz, js)
    data = np.load(npz)
    assert data["points_donut_0"].shape == (16, 3)
    assert int(data["interaction_count"]) == 7
    loaded = json.loads(js.read_text())
    assert loaded["objects"]["donut_0"]["num_points"] == 16
    assert summary["objects"]["donut_0"]["visible"] is True
    assert "target" in loaded["objects"]


def test_farthest_point_downsample_exact_count():
    rng = np.random.default_rng(0)
    pts = rng.random((5000, 3))
    out, cols = farthest_point_downsample(pts, pts, 64, rng)
    assert out.shape == (64, 3) and cols.shape == (64, 3)
    small = rng.random((10, 3))
    out, _ = farthest_point_downsample(small, small, 64, rng)
    assert out.shape == (10, 3)


def test_flatten_particles_pads_and_masks():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    env.reset("train", 0)
    snap = extract_named_particles(env, max_points_per_object=8)
    names = ["donut_0", "donut_7", "target"]
    pts, vis = flatten_particles(snap, names, 8, origin=np.zeros(3))
    assert pts.shape == (3, 8, 3) and vis.shape == (3, 8)
    assert vis[0].sum() == 8 and vis[1].sum() == 0  # donut_7 parked far away


def test_video_recorder_handles_odd_frame_sizes(tmp_path):
    """libx264 rejects odd dimensions and the default camera is 335 px wide.

    Without padding, ffmpeg dies with a broken pipe on the first frame, which
    is what killed every RL call in sweep 2 (DEBUG_LOG 25).
    """
    from agent_robot_control.experiments.replay import VideoRecorder
    rng = np.random.default_rng(0)
    out = tmp_path / "odd.mp4"
    rec = VideoRecorder(out, fps=20)
    for i in range(6):
        frame = rng.integers(0, 255, (180, 335, 3), dtype=np.uint8)
        rec.add(frame, f"frame {i}")
    rec.close()
    assert out.exists() and out.stat().st_size > 2000, out.stat().st_size
    assert rec.frames == 6
