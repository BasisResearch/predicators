"""Test cases for the NSRT learning approach."""

import pytest
from predicators.src.envs import create_env
from predicators.src.approaches import create_approach
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src import utils


def _test_approach(env_name,
                   approach_name,
                   excluded_predicates="",
                   try_solving=True,
                   check_solution=False,
                   sampler_learner="neural",
                   option_learner="no_learning",
                   learn_side_predicates=False):
    """Integration test for the given approach."""
    utils.flush_cache()  # Some extremely nasty bugs arise without this.
    utils.update_config({
        "env": env_name,
        "approach": approach_name,
        "seed": 123
    })
    utils.update_config({
        "timeout": 10,
        "max_samples_per_step": 10,
        "neural_gaus_regressor_max_itr": 100,
        "sampler_mlp_classifier_max_itr": 100,
        "predicate_mlp_classifier_max_itr": 100,
        "mlp_regressor_max_itr": 100,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "offline_data_num_replays": 50,
        "excluded_predicates": excluded_predicates,
        "learn_side_predicates": learn_side_predicates,
        "option_learner": option_learner,
        "sampler_learner": sampler_learner
    })
    env = create_env(env_name)
    assert env.goal_predicates.issubset(env.predicates)
    if CFG.excluded_predicates:
        excludeds = set(CFG.excluded_predicates.split(","))
        assert excludeds.issubset({pred.name for pred in env.predicates}), \
            "Unrecognized excluded_predicates!"
        preds = {pred for pred in env.predicates if pred.name not in excludeds}
        assert env.goal_predicates.issubset(preds), \
            "Can't exclude a goal predicate!"
    else:
        preds = env.predicates
    approach = create_approach(approach_name, env.simulate, preds, env.options,
                               env.types, env.action_space)
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    task = env.get_test_tasks()[0]
    if try_solving:
        policy = approach.solve(task, timeout=CFG.timeout)
        if check_solution:
            assert utils.policy_solves_task(policy, task, env.simulate)
    # We won't check the policy here because we don't want unit tests to
    # have to train very good models, since that would be slow.
    # Now test loading NSRTs & predicates.
    approach2 = create_approach(approach_name, env.simulate, preds,
                                env.options, env.types, env.action_space)
    approach2.load()
    if try_solving:
        policy = approach2.solve(task, timeout=CFG.timeout)
        if check_solution:
            assert utils.policy_solves_task(policy, task, env.simulate)


def test_nsrt_learning_approach():
    """Tests for NSRTLearningApproach class."""
    _test_approach(env_name="blocks", approach_name="nsrt_learning")
    with pytest.raises(NotImplementedError):
        _test_approach(env_name="repeated_nextto",
                       approach_name="nsrt_learning",
                       try_solving=False,
                       sampler_learner="random",
                       learn_side_predicates=True)

def test_neural_option_learning():
    """Tests for NeuralOptionLearner class.
    """
    _test_approach(env_name="cover_multistep_options",
                   approach_name="nsrt_learning",
                   try_solving=False,
                   sampler_learner="random",
                   option_learner="neural",
                   check_solution=False)


def test_oracle_samplers():
    """Test NSRTLearningApproach with oracle samplers."""
    # Oracle sampler learning should work (and be fast) in cover and blocks.
    # We can even check that the policy succeeds!
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   check_solution=True)
    _test_approach(env_name="blocks",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   check_solution=True)
    # Test oracle samplers + option learning.
    _test_approach(env_name="cover",
                   approach_name="nsrt_learning",
                   sampler_learner="oracle",
                   option_learner="oracle",
                   check_solution=True)
    with pytest.raises(Exception) as e:
        # In painting, we learn operators that are different from the
        # oracle ones, so oracle sampler learning is not possible.
        _test_approach(env_name="painting",
                       approach_name="nsrt_learning",
                       sampler_learner="oracle",
                       check_solution=True)
    assert "no match for ground truth NSRT" in str(e)


def test_iterative_invention_approach():
    """Tests for IterativeInventionApproach class."""
    _test_approach(env_name="cover",
                   approach_name="iterative_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")
    _test_approach(env_name="blocks",
                   approach_name="iterative_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")


def test_grammar_search_invention_approach():
    """Tests for GrammarSearchInventionApproach class.

    Keeping this here because we can't import test files in github
    checks.
    """
    utils.update_config({
        "grammar_search_true_pos_weight": 10,
        "grammar_search_false_pos_weight": 1,
        "grammar_search_operator_size_weight": 1e-2,
        "grammar_search_max_predicates": 10,
        "grammar_search_predicate_cost_upper_bound": 6,
        "grammar_search_score_function": "prediction_error",
        "grammar_search_search_algorithm": "hill_climbing",
    })
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")
    # Test approach with unrecognized search algorithm.
    utils.update_config({
        "grammar_search_search_algorithm": "not a real search algorithm",
        "grammar_search_gbfs_num_evals": 10,
    })
    with pytest.raises(Exception) as e:
        _test_approach(env_name="cover",
                       approach_name="grammar_search_invention",
                       excluded_predicates="Holding",
                       try_solving=False,
                       sampler_learner="random")
    assert "Unrecognized grammar_search_search_algorithm" in str(e.value)
    # Test approach with gbfs.
    utils.update_config({
        "grammar_search_search_algorithm": "gbfs",
        "grammar_search_gbfs_num_evals": 10,
    })
    _test_approach(env_name="cover",
                   approach_name="grammar_search_invention",
                   excluded_predicates="Holding",
                   try_solving=False,
                   sampler_learner="random")
