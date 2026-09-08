"""Donut: live-donut cap, recycling, object ids."""
import numpy as np

from predicators.envs.pybullet_donut import PyBulletDonutEnv
from predicators.structs import Action

from agent_robot_control.tests.conftest import make_env


def test_live_donuts_capped_and_ids_assigned():
    env = make_env("pybullet_donut", PyBulletDonutEnv, num_train_tasks=1)
    env.reset("train", 0)
    assert all(d.id is not None for d in env._donuts)
    assert env._target.id is not None
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    max_live = 0
    for t in range(1500):
        env.step(hold)
        if t % 50 == 0:
            live = sum(not env._is_out_of_view(d) for d in env._donut_ids)
            max_live = max(max_live, live)
    assert max_live == env.num_donuts
    assert not env._is_out_of_view(env._donut_ids[0]), "donut_0 was recycled"
    assert len(env._spawn_order) == env.num_donuts
    assert not env.goal_reached()
