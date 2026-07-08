"""Tests for cogman.py."""

import time
from typing import Any, List

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs import get_or_create_env
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.structs import Action, DefaultState, EnvironmentTask


@pytest.mark.parametrize("exec_monitor_name", ["trivial", "mpc"])
def test_cogman(exec_monitor_name):
    """Tests for CogMan()."""
    env_name = "cover"
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 0,
        "num_test_tasks": 2,
    })
    env = get_or_create_env(env_name)
    env_train_tasks = env.get_train_tasks()
    env_test_tasks = env.get_test_tasks()
    train_tasks = [t.task for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor(exec_monitor_name)
    approach = create_approach("random_actions", env.predicates, options,
                               env.types, env.action_space, train_tasks)
    cogman = CogMan(approach, perceiver, exec_monitor)
    env.reset("test", 0)
    env_task = env_test_tasks[0]
    cogman.reset(env_task)
    obs = env_task.init_obs
    act = cogman.step(obs)
    assert env.action_space.contains(act.arr)
    next_obs = env.step(act)
    next_act = cogman.step(next_obs)
    assert not np.allclose(act.arr, next_act.arr)


def test_cogman_with_expected_atoms_monitor():
    """Tests for CogMan() with bilevel planning and the 'expected_atoms'
    execution monitor."""
    env_name = "cover"
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 0,
        "num_test_tasks": 2,
        "bilevel_plan_without_sim": True,
        "approach": "oracle"
    })
    env = get_or_create_env(env_name)
    env_train_tasks = env.get_train_tasks()
    env_test_tasks = env.get_test_tasks()
    train_tasks = [t.task for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("expected_atoms")
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space, train_tasks)
    cogman = CogMan(approach, perceiver, exec_monitor)
    env.reset("test", 0)
    env_task = env_test_tasks[0]
    cogman.reset(env_task)
    obs = env_task.init_obs
    act = cogman.step(obs)
    assert env.action_space.contains(act.arr)
    next_obs = env.step(act)
    next_act = cogman.step(next_obs)
    assert not np.allclose(act.arr, next_act.arr)


def test_run_episode_and_get_observations():
    """Tests for run_episode_and_get_observations()."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    cover_options = get_gt_options(env.get_name())
    task = env.get_task("test", 0)
    approach = create_approach("random_options", env.predicates, cover_options,
                               env.types, env.action_space, train_tasks)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)
    (states, actions), solved, metrics = run_episode_and_get_observations(
        cogman, env, "test", 0, max_num_steps=5)
    assert not solved
    assert len(states) == 6
    assert len(actions) == 5
    assert "policy_call_time" in metrics
    assert metrics["policy_call_time"] > 0.0
    assert metrics["num_options_executed"] > 0.0

    # Test exceptions_to_break_on.
    def _value_error_policy(_):
        raise ValueError("mock error")

    class _MockApproach:

        def __init__(self, policy):
            self._policy = policy

        def solve(self, task, timeout):
            """Just use the given policy."""
            del task, timeout  # unused
            return self._policy

        @classmethod
        def get_name(cls) -> str:
            """Return mock approach name."""
            return "mock"

        def get_execution_monitoring_info(self) -> List[Any]:
            """Just return empty list."""
            return []

        def reset_for_new_episode(self) -> None:
            """No per-episode state."""

    class _CountingMonitor(utils.LoggingMonitor):

        def __init__(self):
            self.num_observations = 0

        def reset(self, train_or_test, task_idx):
            self.num_observations = 0

        def observe(self, obs, action):
            self.num_observations += 1

    approach = _MockApproach(_value_error_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)

    with pytest.raises(ValueError) as e:
        _, _, _ = run_episode_and_get_observations(cogman,
                                                   env,
                                                   "test",
                                                   0,
                                                   max_num_steps=5)
    assert "mock error" in str(e)

    monitor = _CountingMonitor()
    (states, _), _, _ = run_episode_and_get_observations(
        cogman,
        env,
        "test",
        0,
        max_num_steps=5,
        exceptions_to_break_on={ValueError},
        monitor=monitor)

    assert len(states) == 1
    assert monitor.num_observations == 1

    class _MockEnv:

        predicates = set()

        @staticmethod
        def reset(train_or_test, task_idx):
            """Reset the mock environment."""
            del train_or_test, task_idx  # unused
            return DefaultState

        @staticmethod
        def step(action):
            """Step the mock environment."""
            del action  # unused
            raise utils.EnvironmentFailure("mock failure")

        def get_observation(self):
            """Gets currrent observation in mock environment."""
            return DefaultState

        def goal_reached(self):
            """Goal never reached."""
            return False

    mock_env = _MockEnv()
    ones_policy = lambda _: Action(np.zeros(1, dtype=np.float32))
    approach = _MockApproach(ones_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)
    monitor = _CountingMonitor()
    (states, actions), _, _ = run_episode_and_get_observations(
        cogman,
        mock_env,
        "test",
        0,
        max_num_steps=5,
        exceptions_to_break_on={utils.EnvironmentFailure},
        monitor=monitor)
    assert len(states) == 1
    assert len(actions) == 0
    assert monitor.num_observations == 1

    # Test policy call time.
    def _policy(_):
        time.sleep(0.1)
        return Action(env.action_space.sample())

    approach = _MockApproach(_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)

    _, _, metrics = run_episode_and_get_observations(cogman,
                                                     env,
                                                     "test",
                                                     0,
                                                     max_num_steps=3)
    assert metrics["policy_call_time"] >= 3 * 0.1
    assert metrics["num_options_executed"] == 0

    # Test with monitor in case where an uncaught exception is raised.

    def _policy(_):
        raise ValueError("mock error")

    monitor = _CountingMonitor()
    approach = _MockApproach(_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)

    try:
        run_episode_and_get_observations(cogman,
                                         mock_env,
                                         "test",
                                         0,
                                         max_num_steps=3,
                                         monitor=monitor)
    except ValueError:
        pass
    assert monitor.num_observations == 1


def test_run_episode_trajectory_certificate():
    """Goal atoms holding is not enough when the env rejects the episode
    trajectory via check_episode_trajectory."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    # With no certifying reward on the task, any trajectory is accepted.
    assert env.check_episode_trajectory([DefaultState], []) == (True, "")
    task = env.get_task("test", 0)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")

    class _MockApproach:

        def solve(self, task_, timeout):
            """Return a constant policy."""
            del task_, timeout  # unused
            return lambda _: Action(np.zeros(1, dtype=np.float32))

        @classmethod
        def get_name(cls) -> str:
            """Return mock approach name."""
            return "mock"

        def get_execution_monitoring_info(self) -> List[Any]:
            """Just return empty list."""
            return []

        def reset_for_new_episode(self) -> None:
            """No per-episode state."""

    class _CertifyingEnv:
        """Goal atoms always hold; the trajectory check decides."""

        predicates = set()

        def __init__(self, ok, reason="", step_raises=False):
            self._ok = ok
            self._reason = reason
            self._step_raises = step_raises
            self.checked_with = None

        def reset(self, train_or_test, task_idx):
            """Reset the mock environment."""
            del train_or_test, task_idx  # unused
            return DefaultState

        def step(self, action):
            """Step the mock environment."""
            del action  # unused
            if self._step_raises:
                raise utils.EnvironmentFailure("mock failure")
            return DefaultState

        def get_observation(self):
            """Get current observation in mock environment."""
            return DefaultState

        def goal_reached(self):
            """Goal atoms always hold."""
            return True

        def check_episode_trajectory(self, observations, actions):
            """Record the call and return the configured verdict."""
            self.checked_with = (len(observations), len(actions))
            return self._ok, self._reason

    # Rejecting certificate => not solved, despite goal_reached() == True.
    rejecting_env = _CertifyingEnv(False, "robot knocked the target")
    cogman = CogMan(_MockApproach(), perceiver, exec_monitor)
    cogman.reset(task)
    (states,
     actions), solved, _ = run_episode_and_get_observations(cogman,
                                                            rejecting_env,
                                                            "test",
                                                            0,
                                                            max_num_steps=2)
    assert not solved
    # The certificate saw the full per-step history.
    assert rejecting_env.checked_with == (len(states), len(actions))

    # Accepting certificate => solved.
    accepting_env = _CertifyingEnv(True)
    cogman = CogMan(_MockApproach(), perceiver, exec_monitor)
    cogman.reset(task)
    _, solved, _ = run_episode_and_get_observations(cogman,
                                                    accepting_env,
                                                    "test",
                                                    0,
                                                    max_num_steps=2)
    assert solved

    # The keep_failed_demos early-return path is gated too.
    utils.reset_config({"env": "cover", "keep_failed_demos": True})
    failing_env = _CertifyingEnv(False, "rejected", step_raises=True)
    cogman = CogMan(_MockApproach(), perceiver, exec_monitor)
    cogman.reset(task)
    _, solved, _ = run_episode_and_get_observations(cogman,
                                                    failing_env,
                                                    "test",
                                                    0,
                                                    max_num_steps=2)
    assert not solved
    assert failing_env.checked_with is not None


def test_check_episode_trajectory_delegates_to_reward():
    """BaseEnv.check_episode_trajectory delegates to the task reward's
    certify_trajectory, passing the per-step States and per-action option
    labels."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    task = env.get_task("test", 0)

    class _CertifyingReward:
        """Final-state reward with a trajectory-level side-condition."""

        def __init__(self, ok, reason=""):
            self._verdict = (ok, reason)
            self.seen = None

        def __call__(self, state):
            """Goal atoms always hold."""
            del state  # unused
            return True

        def certify_trajectory(self, states, step_options):
            """Record the call and return the configured verdict."""
            self.seen = (list(states), list(step_options))
            return self._verdict

    reward = _CertifyingReward(False, "robot knocked the target")
    env._current_task = EnvironmentTask(  # pylint: disable=protected-access
        task.init_obs,
        task.goal_description,
        reward_fn=reward)
    push = utils.SingletonParameterizedOption(
        "Push", lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)))
    act_with_option = Action(np.zeros(1, dtype=np.float32))
    act_with_option.set_option(push.ground([], np.zeros(0, dtype=np.float32)))
    act_without_option = Action(np.zeros(1, dtype=np.float32))
    init = task.init_obs
    ok, reason = env.check_episode_trajectory(
        [init, init, init], [act_with_option, act_without_option])
    assert (ok, reason) == (False, "robot knocked the target")
    states_seen, options_seen = reward.seen
    assert len(states_seen) == 3
    assert options_seen == [("Push", ()), None]
    # Non-State observations: the check is skipped, not run on garbage.
    reward.seen = None
    assert env.check_episode_trajectory(["not a state"], []) == (True, "")
    assert reward.seen is None
