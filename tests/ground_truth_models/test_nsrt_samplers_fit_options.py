"""The ground-truth NSRTs sample parameters their options accept."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


@pytest.mark.parametrize("env_name", ["pybullet_coffee", "pybullet_grow"])
def test_nsrt_samplers_fit_their_options(env_name):
    """Each NSRT samples parameters that its option's parameter space holds.

    The default skill-factory options take a grasp height for PickJug
    and a placement target for Place, but these NSRTs sampled for the
    legacy options, which take none: the oracle ran PickJug without a
    grasp height and raised an IndexError.
    """
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    options = get_gt_options(env_name)
    nsrts = get_gt_nsrts(env_name, env.predicates, options)
    task = env.get_train_tasks()[0].task
    rng = np.random.default_rng(0)
    for nsrt in nsrts:
        objects = [
            next(obj for obj in task.init if obj.is_instance(var.type))
            for var in nsrt.parameters
        ]
        option = nsrt.ground(objects).sample_option(task.init, task.goal, rng)
        assert nsrt.option.params_space.contains(option.params), nsrt.name
