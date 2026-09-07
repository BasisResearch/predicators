"""Tests for run.early_stopping.EarlyStopping.

The loop in main.py only exercises these verdicts end to end; here each
mode's gate is pinned directly.
"""
from collections import defaultdict
from typing import Any

from predicators import utils
from predicators.run.early_stopping import EarlyStopping
from predicators.structs import Metrics


def _test_results(num_solved: int, num_total: int) -> Metrics:
    results: Metrics = defaultdict(float)
    results["num_solved"] = num_solved
    results["num_total"] = num_total
    return results


def _config(**overrides: Any) -> None:
    utils.reset_config({
        "online_learning_early_stopping":
        True,
        "online_learning_early_stopping_by_test_solve_rate":
        False,
        "online_learning_early_stopping_require_all_attempts":
        False,
        "online_learning_early_stopping_consecutive_perfect_tests":
        1,
        "online_learning_early_stopping_skip_redundant_test":
        False,
        "skip_test_until_last_ite_or_early_stopping":
        False,
        **overrides,
    })


def test_train_driven_first_attempt_per_task():
    """Mode A (legacy): the first attempt per task must succeed and every train
    task must be covered."""
    _config()
    stopping = EarlyStopping(num_train_tasks=2, model_has_learned=True)
    # Task 1 never attempted.
    assert stopping.record_train_attempts([0], [True]) == 1.0
    assert not stopping.train_driven_stop()
    # A later failed retry of task 0 does not matter in legacy mode.
    assert stopping.record_train_attempts([0, 1, 0],
                                          [True, True, False]) == 1.0
    assert stopping.train_driven_stop()
    # The previous cycle's evidence is replaced, not accumulated.
    assert stopping.record_train_attempts([0, 1], [True, False]) == 0.5
    assert not stopping.train_driven_stop()


def test_train_driven_require_all_attempts():
    """Mode A with require_all_attempts scores every request."""
    _config(online_learning_early_stopping_require_all_attempts=True)
    stopping = EarlyStopping(num_train_tasks=2, model_has_learned=True)
    rate = stopping.record_train_attempts([0, 1, 0], [True, True, False])
    assert abs(rate - 2 / 3) < 1e-9
    assert not stopping.train_driven_stop()
    assert stopping.record_train_attempts([0, 1, 0], [True, True, True]) == 1.0
    assert stopping.train_driven_stop()


def test_train_driven_requires_a_learned_model():
    """All tasks solved by a model that never learned is not eligible; it
    becomes eligible once record_learned() is called."""
    _config()
    stopping = EarlyStopping(num_train_tasks=1, model_has_learned=False)
    stopping.record_train_attempts([0], [True])
    assert not stopping.train_driven_stop()
    stopping.record_learned()
    assert stopping.model_has_learned
    assert stopping.train_driven_stop()


def test_train_driven_disabled_by_flags():
    """Mode A is off without online_learning_early_stopping, and yields to mode
    B when by_test_solve_rate is set."""
    _config(online_learning_early_stopping=False)
    stopping = EarlyStopping(num_train_tasks=1, model_has_learned=True)
    stopping.record_train_attempts([0], [True])
    assert not stopping.train_driven_stop()
    _config(online_learning_early_stopping_by_test_solve_rate=True)
    assert not stopping.train_driven_stop()


def test_test_driven_consecutive_perfect_tests():
    """Mode B counts consecutive perfect test phases; an imperfect one resets
    the streak, an empty test set never counts as perfect."""
    _config(online_learning_early_stopping_by_test_solve_rate=True,
            online_learning_early_stopping_consecutive_perfect_tests=2)
    stopping = EarlyStopping(num_train_tasks=1, model_has_learned=True)
    assert not stopping.test_driven_stop()
    stopping.record_test("cycle 0", _test_results(3, 3))
    assert stopping.perfect_test_streak == 1
    assert not stopping.test_driven_stop()
    stopping.record_test("cycle 1", _test_results(2, 3))
    assert stopping.perfect_test_streak == 0
    stopping.record_test("cycle 2", _test_results(0, 0))
    assert stopping.perfect_test_streak == 0
    stopping.record_test("cycle 3", _test_results(3, 3))
    stopping.record_test("cycle 4", _test_results(3, 3))
    assert stopping.perfect_test_streak == 2
    assert stopping.test_driven_stop()
    assert stopping.last_test_summary is not None
    assert stopping.last_test_summary[0] == "cycle 4"
    # Mode B ignores the train evidence entirely.
    _config(online_learning_early_stopping_by_test_solve_rate=False)
    assert not stopping.test_driven_stop()


def test_force_final_test():
    """The stopping cycle re-tests unless the redundant re-test is opted out of
    AND every cycle is tested AND some test already ran."""
    _config(online_learning_early_stopping_skip_redundant_test=True)
    untested = EarlyStopping(num_train_tasks=1, model_has_learned=True)
    assert untested.force_final_test
    tested = EarlyStopping(num_train_tasks=1,
                           model_has_learned=True,
                           initial_test_summary=("pre-loop",
                                                 _test_results(1, 1)))
    assert not tested.force_final_test
    # Only-test-at-the-end runs must test the final model.
    _config(online_learning_early_stopping_skip_redundant_test=True,
            skip_test_until_last_ite_or_early_stopping=True)
    assert tested.force_final_test
    # Without the opt-out the re-test always runs.
    _config()
    assert tested.force_final_test
