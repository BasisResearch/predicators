"""A task cache preserves the exact robot configuration across fresh worlds."""
# pylint: disable=protected-access
import json
from typing import cast

import numpy as np
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv
from predicators.envs.pybullet_domino.task_generators import \
    min_block_generation as task_cache
from predicators.structs import Action, EnvironmentTask


@pytest.mark.parametrize("legacy", [False, True])
def test_task_cache_preserves_joint_configuration(tmp_path, legacy):
    """Save, load, reset, and act through the real simulator and disk cache."""
    utils.reset_config({
        "env": "pybullet_domino",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "domino_min_block_tasks": False,
        "domino_initialize_at_finished_state": False,
    })
    source = cast(PyBulletDominoComposedEnv,
                  create_new_env("pybullet_domino", do_cache=False))
    fresh = cast(PyBulletDominoComposedEnv,
                 create_new_env("pybullet_domino", do_cache=False))
    try:
        task = source.get_train_tasks()[0]
        source.reset("train", 0)
        joints = source._pybullet_robot.get_joints()
        joints[0] += .15
        source._pybullet_robot.set_joints(joints)
        state = source._get_state()
        cache_task = EnvironmentTask(state, task.goal, goal_nl=task.goal_nl)
        cache = tmp_path / "tasks.json"
        task_cache._save_min_block_cache(cache, [cache_task], 1)
        if legacy:
            payload = json.loads(cache.read_text())
            payload["tasks"][0].pop("simulator_state", None)
            cache.write_text(json.dumps(payload))
        loaded = task_cache._load_min_block_cache(fresh, cache)
        assert loaded is not None and len(loaded) == 1
        assert loaded[0].goal == task.goal
        if legacy:
            return
        # Exact joint data is observed and must survive, even where the
        # end-effector feature vector admits multiple inverse solutions.
        assert loaded[0].init.simulator_state[
            "joint_positions"] == state.simulator_state["joint_positions"]
        source._train_tasks = [cache_task]
        fresh._train_tasks = loaded
        left = source.reset("train", 0)
        right = fresh.reset("train", 0)
        assert left.simulator_state[
            "joint_positions"] == right.simulator_state["joint_positions"]
        hold = Action(
            np.array(state.simulator_state["joint_positions"],
                     dtype=np.float32))
        for _ in range(3):
            left = source.step(hold)
            right = fresh.step(hold)
            np.testing.assert_allclose(
                left.simulator_state["joint_positions"],
                right.simulator_state["joint_positions"],
                rtol=0,
                atol=1e-10)
    finally:
        source.dispose()
        fresh.dispose()
