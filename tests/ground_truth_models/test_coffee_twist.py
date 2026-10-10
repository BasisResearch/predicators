"""Tests for the combined Twist option of the PyBullet coffee environment."""

import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options, \
    get_gt_processes


def test_combined_twist_turns_a_rotated_jug_until_pickable():
    """Twist, its NSRT and its process take no params, and the process's option
    turns a rotated jug until JugPickable holds.

    Regression: Twist chained MoveToTwistJug, which takes no params,
    with a TwistJug that takes a twist amount. The chain's check of its
    children's params spaces compared bounds with np.allclose, which
    broadcast the (0,) and (1,) bounds and let the mismatch through.
    """
    utils.reset_config({
        "env": "pybullet_coffee",
        "coffee_combined_move_and_twist_policy": True,
        "coffee_jug_pickable_pred": True,
        "coffee_rotated_jug_ratio": 1.0,
        # The position controller's descent onto the jug knocks it aside.
        "pybullet_control_mode": "reset",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = create_new_env("pybullet_coffee", do_cache=True, use_gui=False)
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    processes = get_gt_processes(env.get_name(),
                                 env.predicates,
                                 options,
                                 only_endogenous=True)
    twist = utils.get_parameterized_option_by_name(options, "Twist")
    assert twist is not None
    assert twist.params_space.shape == (0, )
    nsrt, = [n for n in nsrts if n.name == "Twist"]
    process, = [p for p in processes if p.name == "Twist"]
    types = {t.name: t for t in env.types}
    JugPickable, = [p for p in env.predicates if p.name == "JugPickable"]
    task = env.get_test_tasks()[0].task
    state = env.reset("test", 0)
    robot, = state.get_objects(types["robot"])
    jug, = state.get_objects(types["jug"])
    assert not JugPickable.holds(state, [jug])
    rng = np.random.default_rng(0)
    objects = [robot, jug]
    nsrt_option = nsrt.ground(objects).sample_option(state, task.goal, rng)
    assert nsrt_option.params.shape == (0, )
    option = process.ground(objects).sample_option(state, task.goal, rng)
    assert option.params.shape == (0, )
    assert option.initiable(state)
    for _ in range(100):
        state = env.step(option.policy(state))
        if option.terminal(state):
            break
    assert option.terminal(state)
    assert JugPickable.holds(state, [jug])
