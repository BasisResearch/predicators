"""Tests for the PyBullet Laser env."""
# pylint: disable=protected-access

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.envs.pybullet_laser import PyBulletLaserEnv
from predicators.structs import Action


def test_laser_beam_bodies_stay_in_their_own_world():
    """A powered station draws its beam bodies in the Laser env's own world.

    The beam bodies used to be created without naming a PyBullet client,
    so a Laser env built after another env drew them into the first
    env's world.
    """
    utils.reset_config({
        "env": "pybullet_laser",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    other = PyBulletBlocksEnv(use_gui=False)
    env = PyBulletLaserEnv(use_gui=False)
    env.reset("train", 0)
    env._set_station_powered_on(True)
    other_bodies = p.getNumBodies(physicsClientId=other._physics_client_id)
    laser_bodies = p.getNumBodies(physicsClientId=env._physics_client_id)
    env.step(
        Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32)))
    assert p.getNumBodies(
        physicsClientId=other._physics_client_id) == other_bodies
    assert p.getNumBodies(
        physicsClientId=env._physics_client_id) > laser_bodies
