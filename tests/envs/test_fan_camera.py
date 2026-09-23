"""Regression checks for ramp camera framing."""
# pylint: disable=protected-access
import itertools

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv


def _body_aabb_corners(body_id, client_id):
    """Return corners covering the base and every articulated link."""
    corners = []
    for link in range(-1, p.getNumJoints(body_id, physicsClientId=client_id)):
        lower, upper = p.getAABB(body_id, link, physicsClientId=client_id)
        corners.extend(itertools.product(*zip(lower, upper)))
    return corners


def _project(points, view, projection):
    homogeneous = np.column_stack(
        (np.asarray(points), np.ones(len(points), dtype=float)))
    clip = (np.asarray(projection).reshape(
        (4, 4), order="F") @ np.asarray(view).reshape(
            (4, 4), order="F") @ homogeneous.T).T
    assert np.all(clip[:, 3] > 0)
    return clip[:, :2] / clip[:, 3, None]


@pytest.mark.parametrize("seed", range(5))
def test_ramp_target_visible(seed):
    """Both levels frame the target, every fan bank, and all switches."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": seed,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": True,
        "fan_ramp_transfer": True,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
        "fan_test_num_pos_x": 3,
        "fan_test_num_pos_y": 3,
    })
    env = PyBulletFanEnv(use_gui=False)
    try:
        for split in ("train", "test"):
            state = env.reset(split, 0)
            view, projection, _, _ = env._get_camera_matrices()
            target = next(o for o in state if o.type.name == "target")
            target_point = [[state.get(target, c) for c in ("x", "y", "z")]]
            assert np.all(
                np.abs(_project(target_point, view, projection)) < 0.75)
            body_ids = [
                body_id for fan in env._fans for body_id in fan.fan_ids
            ] + [switch.id for switch in env._switches]
            corners = [
                corner
                for body_id in body_ids for corner in _body_aabb_corners(
                    body_id, env._physics_client_id)
            ]
            assert np.all(np.abs(_project(corners, view, projection)) < 0.96)
    finally:
        p.disconnect(env._physics_client_id)
