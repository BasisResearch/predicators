"""Test cases for the interactive learning approach."""

import pytest
from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src.structs import Dataset
from predicators.src.main import _generate_interaction_results
from predicators.src.teacher import Teacher
from predicators.src import utils


def test_interactive_learning_approach():
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "interactive_learning",
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "Covers,Holding",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 100,
        "predicate_mlp_classifier_max_itr": 100,
        "neural_gaus_regressor_max_itr": 100,
        "num_online_learning_cycles": 1,
        "teacher_dataset_label_ratio": 1.0,
        "num_train_tasks": 5,
        "num_test_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    initial_predicates = {
        p
        for p in env.predicates if p.name not in ["Covers", "Holding"]
    }
    approach = InteractiveLearningApproach(initial_predicates, env.options,
                                           env.types, env.action_space,
                                           train_tasks)
    teacher = Teacher()
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    # Learning with an empty dataset should not crash.
    approach.learn_from_offline_dataset(Dataset([]))
    # Learning with the actual dataset.
    approach.learn_from_offline_dataset(dataset)
    approach.load(online_learning_cycle=None)
    interaction_requests = approach.get_interaction_requests()
    interaction_results = _generate_interaction_results(
        env.simulate, teacher, train_tasks, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    approach.load(online_learning_cycle=0)
    with pytest.raises(FileNotFoundError):
        approach.load(online_learning_cycle=1)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
    # Test with a query policy that always queries about every atom.
    utils.update_config({
        "interactive_query_policy": "nonstrict_best_seen",
        "interactive_score_function": "trivial",
    })
    interaction_requests = approach.get_interaction_requests()
    _generate_interaction_results(env.simulate, teacher, train_tasks,
                                  interaction_requests)
    # Cover unrecognized interactive_action_strategy.
    utils.update_config({
        "interactive_action_strategy": "not a real action strategy",
        "interactive_query_policy": "strict_best_seen",
        "interactive_score_function": "frequency",
    })
    with pytest.raises(NotImplementedError) as e:
        approach.get_interaction_requests()
    assert "Unrecognized interactive_action_strategy" in str(e)
    # Cover unrecognized interactive_query_policy.
    utils.update_config({
        "interactive_action_strategy": "glib",
        "interactive_query_policy": "not a real query policy",
        "interactive_score_function": "frequency",
    })
    with pytest.raises(NotImplementedError) as e:
        approach.get_interaction_requests()
    assert "Unrecognized interactive_query_policy" in str(e)
    # Cover unrecognized interactive_score_function.
    utils.update_config({
        "interactive_action_strategy":
        "glib",
        "interactive_query_policy":
        "strict_best_seen",
        "interactive_score_function":
        "not a real score function",
    })
    with pytest.raises(NotImplementedError) as e:
        approach._score_atom_set(set(), train_tasks[0].init)  # pylint:disable=protected-access
    assert "Unrecognized interactive_score_function" in str(e)
