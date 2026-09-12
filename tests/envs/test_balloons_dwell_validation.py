"""Mechanical validation of sustained hovering on validated Balloons tasks."""
# pylint: disable=protected-access
import itertools
from typing import Any

import pybullet as p
import pytest

from predicators import utils
from predicators.envs import create_new_env


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_validated_tasks_admit_sustained_hover(seed: int) -> None:
    """Each pilot task has a public-controller sequence that sustains the goal.

    This is mechanical validation with known physics, not an agent
    result. Task sampling remains the original distribution; only
    acceptance at execution time now requires twenty-five full intervals
    of quiet hovering.
    """
    utils.reset_config({
        'env': 'pybullet_balloons',
        'seed': seed,
        'num_train_tasks': 2,
        'num_test_tasks': 1,
        'balloons_scene': 'chute',
        'balloons_task_generation': 'validated',
        'balloons_goal_dwell_steps': 25,
        'partially_observable': True,
        'skill_phase_use_motion_planning': True,
        'pybullet_ik_validate': False,
        'pybullet_birrt_path_subsample_ratio': 2
    })
    env: Any = create_new_env('pybullet_balloons', do_cache=False)
    try:
        for index, task in enumerate(env.get_train_tasks() +
                                     env.get_test_tasks()):
            subset = env.solution_subset(task.init)
            assert subset is not None
            results = []
            for order in itertools.permutations(subset):
                outcome = env._run_release_sequence(task.init, order)
                results.append((order, outcome.status, outcome.steps))
                if outcome.won:
                    break
            print('sustained-hover', seed, index, results, flush=True)
            assert any(status == 'won' for _, status, _ in results), results
    finally:
        p.disconnect(env._physics_client_id)
