"""The ground-truth processes' samplers return params of their options' shapes,
which ParameterizedOption.ground() requires."""
import itertools

import numpy as np
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_helper_predicates, \
    get_gt_options, get_gt_processes, nsrts_from_processes


@pytest.mark.parametrize("env_name, flags", [
    ("pybullet_grow", {}),
    ("pybullet_grow", {
        "grow_use_skill_factories": False,
        "grow_place_option_no_sampler": True
    }),
    ("pybullet_boil", {}),
    ("pybullet_boil", {
        "boil_use_skill_factories": False
    }),
    ("pybullet_coffee", {}),
    ("pybullet_coffee", {
        "coffee_use_skill_factories": False
    }),
    ("pybullet_coffee", {
        "coffee_twist_sampler": False
    }),
])
def test_process_samplers_match_option_params(env_name, flags):
    """Each endogenous process samples params of its option's shape (empty
    params count when the option has defaults), and its sampled option grounds.

    Regression: with legacy options, whose PickJug takes no params,
    the Grow, Boil and Coffee pick samplers returned a grasp offset
    that the clip before grounding dropped; Coffee's TwistJug process
    gave no twist amount to a TwistJug that takes one, which grounded
    only because the bounds fallback passed empty params.
    """
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        **flags
    })
    env = create_new_env(env_name, do_cache=True, use_gui=False)
    options = get_gt_options(env_name)
    predicates = env.predicates | get_gt_helper_predicates(env_name)
    processes = get_gt_processes(env_name,
                                 predicates,
                                 options,
                                 only_endogenous=True)
    task = env.get_test_tasks()[0].task
    objects = list(task.init)
    rng = np.random.default_rng(0)
    nsrts = sorted(nsrts_from_processes(processes), key=lambda n: n.name)
    assert nsrts
    for nsrt in nsrts:
        candidates = [[o for o in objects if o.is_instance(v.type)]
                      for v in nsrt.parameters]
        objs = next(
            list(c) for c in itertools.product(*candidates)
            if len(set(c)) == len(c))
        params = nsrt.sampler(task.init, task.goal, rng, objs)
        option = nsrt.option
        if not (np.size(params) == 0 and option.default_params is not None):
            assert np.shape(params) == option.params_space.shape, nsrt.name
        ground_option = nsrt.ground(objs).sample_option(
            task.init, task.goal, rng)
        assert ground_option.params.shape == option.params_space.shape
