"""Regression checks for ramp camera framing."""
# pylint: disable=protected-access
import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv


@pytest.mark.parametrize("seed", range(5))
def test_ramp_target_visible(seed):
    """Both levels keep the target comfortably inside the rendered frame."""
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
            point = np.array([state.get(target, c)
                              for c in ("x", "y", "z")] + [1])
            clip = (np.array(projection).reshape(
                (4, 4), order="F") @ np.array(view).reshape(
                    (4, 4), order="F") @ point)
            assert clip[3] > 0
            assert np.all(np.abs(clip[:2] / clip[3]) < 0.85)
    finally:
        p.disconnect(env._physics_client_id)
