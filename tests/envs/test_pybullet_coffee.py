"""Tests for the PyBullet coffee environment."""
# pylint: disable=protected-access

import numpy as np

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.structs import Action


def _noop(env: PyBulletCoffeeEnv) -> Action:
    return Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))


def test_second_world_simulates_with_its_own_cups():
    """A second Coffee world, like the planner's, simulates the first world's
    task with its own cup bodies.

    Coffee recreates its cups for every state, so their body ids differ
    between worlds. The second world used to move the cups by the ids on
    the task's Objects, which belong to the first world, and to write
    its own new ids onto them; the oracle then failed with "Failed to
    get pose for object cup0".
    """
    utils.reset_config({
        "env": "pybullet_coffee",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = PyBulletCoffeeEnv(use_gui=False)
    init = env.get_test_tasks()[0].init
    cups = init.get_objects(env._cup_type)
    cup_ids = [cup.id for cup in cups]
    planner_world = PyBulletCoffeeEnv(use_gui=False)
    state = planner_world.simulate(init, _noop(planner_world))
    assert [cup.id for cup in cups] == cup_ids
    for cup in cups:
        for feature in ("x", "y"):
            assert np.isclose(state.get(cup, feature),
                              init.get(cup, feature),
                              atol=1e-3)
    env.reset("test", 0)
    env.step(_noop(env))
