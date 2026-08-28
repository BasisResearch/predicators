"""Tests for main.py."""
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from typing import Callable, Dict, List

import pytest

import predicators.ground_truth_models
from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach, create_approach
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.cogman import CogMan
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _discard_inflight_interactions, \
    _early_stop_below_bar_msg, _inflight_interactions_path, \
    _load_inflight_interactions, _load_test_solve_rate, \
    _perfect_test_streak_from_disk, _run_testing, \
    _save_inflight_interactions, _save_test_results, discover_resume_cycles, \
    main
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Action, DefaultState, EnvironmentTask, State, \
    Task

_GROUND_TRUTH_MODULE_PATH = predicators.ground_truth_models.__name__


class _DummyFailureApproach(BaseApproach):
    """Dummy approach that raises ApproachFailure for testing."""

    @classmethod
    def get_name(cls) -> str:
        return "dummy_failure"

    @property
    def is_learning_based(self):
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def _policy(s: State) -> Action:
            raise ApproachFailure("Option plan exhausted.")

        return _policy


class _DummySolveTimeoutApproach(BaseApproach):
    """Dummy approach that raises ApproachTimeout during planning for
    testing."""

    @classmethod
    def get_name(cls) -> str:
        return "dummy_solve_timeout"

    @property
    def is_learning_based(self):
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        raise ApproachTimeout("Planning timed out.")


class _DummyExecutionTimeoutApproach(BaseApproach):
    """Dummy approach that raises ApproachTimeout during execution for
    testing."""

    @classmethod
    def get_name(cls) -> str:
        return "dummy_execution_timeout"

    @property
    def is_learning_based(self):
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def _policy(s: State) -> Action:
            raise ApproachTimeout("Policy timed out.")

        return _policy


class _DummyCoverEnv(CoverEnv):
    """Dummy cover environment that raises EnvironmentFailure for testing."""

    @classmethod
    def get_name(cls) -> str:
        return "dummy"

    def simulate(self, state, action):
        raise utils.EnvironmentFailure("", {"offending_objects": set()})


def test_main():
    """Tests for main.py."""
    utils.reset_config()
    sys.argv = [
        "dummy", "--env", "my_env", "--approach", "my_approach", "--seed",
        "123", "--num_test_tasks", "3"
    ]
    with pytest.raises(NotImplementedError):
        main()  # invalid env
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "my_approach", "--seed",
        "123", "--num_test_tasks", "3"
    ]
    with pytest.raises(NotImplementedError):
        main()  # invalid approach
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "random_actions", "--seed",
        "123", "--not-a-real-flag", "0"
    ]
    with pytest.raises(ValueError):
        main()  # invalid flag
    parent_dir = os.path.dirname(__file__)
    video_dir = os.path.join(parent_dir, "_fake_videos")
    results_dir = os.path.join(parent_dir, "_fake_results")
    eval_traj_dir = os.path.join(parent_dir, "_fake_trajs")
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "oracle", "--seed", "123",
        "--make_test_videos", "--make_cogman_videos", "--num_test_tasks", "1",
        "--video_dir", video_dir, "--results_dir", results_dir,
        "--eval_trajectories_dir", eval_traj_dir
    ]
    main()
    # Test making videos of failures and local logging.
    temp_log_file = tempfile.NamedTemporaryFile(delete=False).name
    sys.argv = [
        "dummy", "--env", "painting", "--approach", "oracle", "--seed", "123",
        "--num_test_tasks", "1", "--video_dir", video_dir, "--results_dir",
        results_dir, "--eval_trajectories_dir", eval_traj_dir,
        "--sesame_max_skeletons_optimized", "1", "--painting_lid_open_prob",
        "0.0", "--make_failure_videos", "--log_file", temp_log_file
    ]
    main()
    shutil.rmtree(video_dir)
    shutil.rmtree(results_dir)
    shutil.rmtree(eval_traj_dir)
    # Run NSRT learning, but without sampler learning.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--sampler_learner", "random", "--cover_initial_holding_prob",
        "0.0", "--num_train_tasks", "1", "--num_test_tasks", "1",
        "--experiment_id", "foobar"
    ]
    main()
    # Try loading approaches and data.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--load_approach", "--load_data",
        "--cover_initial_holding_prob", "0.0", "--experiment_id", "foobar"
    ]
    main()
    # Try loading with a bad experiment id.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--load_approach", "--cover_initial_holding_prob", "0.0",
        "--experiment_id", "baz"
    ]
    with pytest.raises(FileNotFoundError):
        main()
    # Try loading with load experiment id.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "nsrt_learning", "--seed",
        "123", "--load_approach", "--cover_initial_holding_prob", "0.0",
        "--load_experiment_id", "foobar", "--experiment_id", "baz"
    ]
    main()
    # Run NSRT learning with option learning.
    sys.argv = [
        "dummy", "--env", "blocks", "--approach", "nsrt_learning", "--seed",
        "123", "--sampler_learner", "random", "--num_train_tasks", "1",
        "--num_test_tasks", "1", "--option_learner", "direct_bc",
        "--segmenter", "atom_changes", "--mlp_regressor_max_itr", "1"
    ]
    main()
    # Try running interactive approach with no online learning, to make sure
    # it doesn't crash. This is also an important test of the full pipeline
    # in the case where a goal predicate is excluded. No online learning occurs
    # because max number of transitions is set.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "interactive_learning",
        "--seed", "123", "--num_online_learning_cycles", "1",
        "--online_learning_max_transitions", "0", "--excluded_predicates",
        "Covers", "--interactive_num_ensemble_members", "1",
        "--num_train_tasks", "3", "--num_test_tasks", "3",
        "--predicate_mlp_classifier_max_itr", "lambda n: n * 50"
    ]
    main()
    # Tests for --crash_on_failure flag.
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "oracle", "--seed", "123",
        "--num_test_tasks", "3", "--timeout", "0", "--crash_on_failure"
    ]
    with pytest.raises(ApproachTimeout) as e:
        main()  # should time out
    assert "Planning timed out in grounding!" in str(e)
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "random_actions", "--seed",
        "123", "--num_test_tasks", "3", "--crash_on_failure"
    ]
    with pytest.raises(RuntimeError) as e:
        main()  # should fail to solve the task
    assert "Policy failed to reach goal" in str(e)
    # Test approach wrapping with the approach_wrapper flag.
    sys.argv = [
        "dummy",
        "--env",
        "noisy_button",
        "--approach",
        "oracle",
        "--seed",
        "123",
        "--approach_wrapper",
        "noisy_button_wrapper",
        "--num_train_tasks",
        "1",
        "--num_test_tasks",
        "1",
    ]
    main()


def test_bilevel_planning_approach_failure_and_timeout():
    """Test coverage for ApproachFailure and ApproachTimeout in
    run_testing()."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "timeout": 10,
        "make_test_videos": False,
        "num_test_tasks": 1,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = _DummyFailureApproach(env.predicates,
                                     get_gt_options(env.get_name()), env.types,
                                     env.action_space, train_tasks)
    assert not approach.is_learning_based
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    _run_testing(env, cogman)

    approach = _DummySolveTimeoutApproach(env.predicates,
                                          get_gt_options(env.get_name()),
                                          env.types, env.action_space,
                                          train_tasks)
    assert not approach.is_learning_based
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    _run_testing(env, cogman)

    approach = _DummyExecutionTimeoutApproach(env.predicates,
                                              get_gt_options(env.get_name()),
                                              env.types, env.action_space,
                                              train_tasks)
    assert not approach.is_learning_based
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    _run_testing(env, cogman)


def test_early_stop_below_bar_msg():
    """A solved episode counts toward early stopping only when its reward
    clears the task's early_stop_min_reward bar (minus slack)."""
    utils.reset_config({"online_learning_early_stopping_reward_slack": 0.0})
    # No bar set: never gated.
    no_bar_task = EnvironmentTask(DefaultState, set())
    assert _early_stop_below_bar_msg(-1.0, no_bar_task) is None
    # Bar set (e.g. domino optimal reward 1 - 0.05 * 3 = 0.85).
    task = EnvironmentTask(DefaultState, set(), early_stop_min_reward=0.85)
    # Over-built solve falls short.
    msg = _early_stop_below_bar_msg(0.75, task)
    assert msg is not None and "0.75" in msg and "0.85" in msg
    # Reward computed exactly at the bar clears it despite float rounding
    # (1 - 0.05 * 3 != 0.85 in binary).
    assert _early_stop_below_bar_msg(1.0 - 0.05 * 3, task) is None
    assert _early_stop_below_bar_msg(0.9, task) is None
    # Slack relaxes the bar (one spare block at 0.05 block cost).
    utils.update_config({"online_learning_early_stopping_reward_slack": 0.05})
    assert _early_stop_below_bar_msg(0.80, task) is None
    assert _early_stop_below_bar_msg(0.75, task) is not None
    # Ignoring the bar makes any solved episode count, regardless of slack.
    utils.update_config({
        "online_learning_early_stopping_reward_slack":
        0.0,
        "online_learning_early_stopping_ignore_reward_bar":
        True,
    })
    assert _early_stop_below_bar_msg(0.75, task) is None
    assert _early_stop_below_bar_msg(-1.0, task) is None


def test_env_failure():
    """Test coverage for EnvironmentFailure in run_testing()."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "timeout": 10,
        "make_test_videos": False,
        "cover_initial_holding_prob": 0.0,
        "num_test_tasks": 1,
    })
    cover_options = get_gt_options("cover")
    env = _DummyCoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("random_actions", env.predicates, cover_options,
                               env.types, env.action_space, train_tasks)
    assert not approach.is_learning_based
    task = train_tasks[0]
    approach.solve(task, timeout=500)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    _run_testing(env, cogman)


def test_skip_initial_test():
    """--skip_initial_test skips only the pre-loop test; per-cycle tests
    still run and save results."""
    utils.reset_config()
    parent_dir = os.path.dirname(__file__)
    results_dir = os.path.join(parent_dir, "_fake_results_skip_initial")
    sys.argv = [
        "dummy", "--env", "cover", "--approach", "interactive_learning",
        "--seed", "123", "--num_online_learning_cycles", "1",
        "--excluded_predicates", "Covers",
        "--interactive_num_ensemble_members", "1", "--num_train_tasks", "3",
        "--num_test_tasks", "1", "--predicate_mlp_classifier_max_itr",
        "lambda n: n * 50", "--skip_initial_test", "True", "--results_dir",
        results_dir
    ]
    main()
    saved = os.listdir(results_dir)
    assert not any(f.endswith("__None.pkl") for f in saved)
    assert any(f.endswith("__0.pkl") for f in saved)
    shutil.rmtree(results_dir)


def test_perfect_test_streak_from_disk():
    """Test-driven early stopping's consecutive-perfect-test streak is re-
    derived from the saved per-cycle results, so an --auto_resume relaunch
    continues the count instead of restarting it."""
    parent_dir = os.path.dirname(__file__)
    results_dir = os.path.join(parent_dir, "_fake_results_streak")
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 123,
        "results_dir": results_dir,
    })

    def _fake_results(num_solved: int, num_total: int) -> dict:
        results: Dict[str, float] = defaultdict(float)
        results["num_solved"] = num_solved
        results["num_total"] = num_total
        return results

    # Fresh run: nothing on disk, seed is 0.
    assert _perfect_test_streak_from_disk(-1) == 0
    assert _perfect_test_streak_from_disk(2) == 0
    assert _load_test_solve_rate(0) is None
    # Cycle 0 imperfect, cycles 1-2 perfect: the walk stops at cycle 0.
    _save_test_results(_fake_results(0, 1), online_learning_cycle=0)
    _save_test_results(_fake_results(1, 1), online_learning_cycle=1)
    _save_test_results(_fake_results(1, 1), online_learning_cycle=2)
    assert _load_test_solve_rate(0) == 0.0
    assert _load_test_solve_rate(1) == 1.0
    assert _perfect_test_streak_from_disk(2) == 2
    assert _perfect_test_streak_from_disk(1) == 1
    assert _perfect_test_streak_from_disk(0) == 0
    # A missing cycle (3) breaks the streak even with cycle 4 perfect.
    _save_test_results(_fake_results(1, 1), online_learning_cycle=4)
    assert _perfect_test_streak_from_disk(4) == 1
    # An empty test set never counts as perfect.
    _save_test_results(_fake_results(0, 0), online_learning_cycle=5)
    assert _perfect_test_streak_from_disk(5) == 0
    shutil.rmtree(results_dir)


def test_inflight_interactions_roundtrip(tmp_path):
    """A cycle's episodes persisted before LEARN survive a mid-learn death:

    reloadable at the same cycle, invisible to the checkpoint scanner,
    ignored when stale, and gone once discarded.
    """
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 0,
        "approach_dir": str(tmp_path),
    })

    class _FakeApproach:
        _save_suffix = "test_ckpt"

    class _FakeCogman:
        _approach = _FakeApproach()

    cogman = _FakeCogman()
    results: List[Dict[str, int]] = [{
        "episode": 1
    }, {
        "episode": 2
    }]  # picklable stand-ins
    _save_inflight_interactions(3, cogman, results, [0, 0], [True, False], 1.5)
    # Wrong cycle finds nothing.
    assert _load_inflight_interactions(2, cogman) is None
    data = _load_inflight_interactions(3, cogman)
    assert data is not None
    assert data["interaction_results"] == results
    assert data["task_idxs"] == [0, 0]
    assert data["task_solved_status"] == [True, False]
    assert data["query_cost"] == 1.5
    # The checkpoint scanner must not mistake the stash for a checkpoint
    # (its cycle token is non-integer by construction).
    load_path = utils.get_approach_load_path_str()
    found, max_cycle = discover_resume_cycles(load_path)
    assert not found
    assert max_cycle is None
    # A stale stash (older than the auto-resume gate) is ignored.
    path = _inflight_interactions_path(3)
    old_ts = time.time() - CFG.auto_resume_max_age_hours * 3600.0 - 10
    os.utime(path, (old_ts, old_ts))
    assert _load_inflight_interactions(3, cogman) is None
    os.utime(path, None)
    assert _load_inflight_interactions(3, cogman) is not None
    # Discard removes it.
    _discard_inflight_interactions(3, cogman)
    assert _load_inflight_interactions(3, cogman) is None

    # A non-checkpointing approach neither saves nor loads a stash.
    class _NoCkptApproach:
        _save_suffix = None

    cogman_nockpt = _FakeCogman()
    cogman_nockpt._approach = _NoCkptApproach()  # pylint: disable=protected-access
    _save_inflight_interactions(4, cogman_nockpt, results, [0], [True], 0.0)
    assert not os.path.exists(_inflight_interactions_path(4))


def test_stash_resume_restores_request_bookkeeping():
    """A resume that reuses a cycle's persisted episodes never calls
    get_interaction_requests, so the result->train-task pairing that
    learn_from_interaction_results needs must come from
    restore_interaction_requests (run_20260828_173451 asserted on it)."""
    # The model-free family records the pairing in get_interaction_requests
    # and asserts on it in learn_from_interaction_results.
    approach = object.__new__(AgentModelFreeApproach)
    approach._requests_train_task_idxs = None  # pylint: disable=protected-access
    approach.restore_interaction_requests([0, 0])
    assert approach._requests_train_task_idxs == [0, 0]  # pylint: disable=protected-access

    # CogMan forwards to whatever approach it wraps.
    class _RecordingApproach:
        restored = None

        def restore_interaction_requests(self, train_task_idxs):
            """Record what CogMan forwarded."""
            self.restored = list(train_task_idxs)

    rec = _RecordingApproach()
    cogman = CogMan(rec, create_perceiver("trivial"),
                    create_execution_monitor("trivial"))
    cogman.restore_interaction_requests([1, 0])
    assert rec.restored == [1, 0]
