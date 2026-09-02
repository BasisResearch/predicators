"""Tests for the GNN dynamics + shooting baseline (paper arm C5)."""
# pylint: disable=protected-access
import numpy as np
import pytest

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import InteractionResult


def _setup(env_name: str = "cover"):
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 3,
        "num_test_tasks": 2,
        "gnn_num_epochs": 20,
        "gnn_use_validation_set": False,
        "gnn_do_normalization": True,
        "gnn_dynamics_history_len": 2,
        "gnn_dynamics_shooting_max_tries": 5,
        "gnn_dynamics_max_plan_length": 4,
        "explorer": "random_options",
        "online_nsrt_learning_requests_per_cycle": 1,
        "max_num_steps_interaction_request": 5,
        "horizon": 10,
        "timeout": 5,
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = get_gt_options(env.get_name())
    approach = create_approach("gnn_dynamics_shooting", env.predicates,
                               options, env.types, env.action_space,
                               train_tasks)
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, options, predicates)
    return env, train_tasks, options, approach, dataset


def test_gnn_dynamics_learns_predicts_and_plans():
    """Learn from demos, predict a transition, shoot a plan, save and load."""
    env, train_tasks, options, approach, dataset = _setup()
    assert approach.is_learning_based
    task = env.get_test_tasks()[0].task
    with pytest.raises(ApproachFailure):  # nothing learned yet
        approach.solve(task, timeout=CFG.timeout)
    approach.learn_from_offline_dataset(dataset)
    assert approach._gnn is not None
    # Option-level examples came out of the demos with history attached.
    examples = approach._generate_examples()
    assert examples
    assert all(len(h) <= 2 for h, *_ in examples)
    history, state, option, next_state, num_actions = examples[-1]
    pred_state, pred_steps = approach.predict_next_state(
        history, state, option)
    assert set(pred_state) == set(state)
    assert pred_steps >= 1
    assert isinstance(num_actions, int)
    for obj in state:
        assert pred_state[obj].shape == next_state[obj].shape
    # A shooting policy is returned; executing it either reaches the goal
    # or fails honestly (the tiny model is not expected to be accurate).
    try:
        policy = approach.solve(task, timeout=CFG.timeout)
        utils.run_policy_with_simulator(policy,
                                        env.simulate,
                                        task.init,
                                        task.goal_holds,
                                        max_num_steps=CFG.horizon,
                                        exceptions_to_break_on={
                                            utils.OptionExecutionFailure,
                                            ApproachFailure,
                                        })
    except (ApproachFailure, ApproachTimeout):
        pass
    # Save / load round trip rebuilds the model.
    approach2 = create_approach("gnn_dynamics_shooting", env.predicates,
                                options, env.types, env.action_space,
                                train_tasks)
    approach2.load(online_learning_cycle=None)
    assert approach2._gnn is not None
    assert approach2._feat_to_index == approach._feat_to_index
    s2, _ = approach2.predict_next_state(history, state, option)
    for obj in state:
        assert np.allclose(s2[obj], pred_state[obj])


def test_gnn_dynamics_online_cycle():
    """Interaction requests come from the configured explorer and the results
    extend the data the next model is trained on."""
    env, _, _, approach, dataset = _setup()
    approach.learn_from_offline_dataset(dataset)
    n_before = len(approach._trajectories)
    requests = approach.get_interaction_requests()
    assert len(requests) == 1
    request = requests[0]
    task = approach._train_tasks[request.train_task_idx]
    traj, _ = utils.run_policy(request.act_policy,
                               env,
                               "train",
                               request.train_task_idx,
                               request.termination_function,
                               max_num_steps=5,
                               exceptions_to_break_on={
                                   utils.RequestActPolicyFailure,
                               })
    del task
    result = InteractionResult(traj.states, traj.actions,
                               [None] * len(traj.states))
    approach.learn_from_interaction_results([result])
    assert len(approach._trajectories) == n_before + 1
    assert approach._online_learning_cycle == 1
    # The interaction trajectory is not a demo and keeps its task index.
    assert not approach._trajectories[-1].is_demo
    assert approach._trajectories[-1].train_task_idx == request.train_task_idx


def test_gnn_dynamics_no_transitions_keeps_model():
    """With no option-bearing data the previous model is kept."""
    _, _, _, approach, dataset = _setup()
    approach.learn_from_offline_dataset(dataset)
    gnn = approach._gnn
    approach._trajectories = []
    approach._learn_model()
    assert approach._gnn is gnn
