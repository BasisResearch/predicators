"""An option that never terminates at test time fails the task, not the run
(run.testing._execute_policy)."""
from types import SimpleNamespace
from typing import Any

import numpy as np

from predicators import utils
from predicators.run import testing
from predicators.run.testing import TestArtifacts, TestMetrics, _execute_policy
from predicators.structs import Object, State, Type


def test_option_execution_failure_counts_as_a_failed_task(
        monkeypatch, tmp_path) -> None:
    """OptionExecutionFailure (e.g. an option past max_option_steps from an
    approach that hands the raw option-policy wrapper to the cogman) is caught
    like an ApproachFailure."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 0,
        "results_dir": str(tmp_path),
        "eval_trajectories_dir": str(tmp_path),
    })

    def _boom(*args, **kwargs):
        del args, kwargs
        raise utils.OptionTimeoutFailure("Exceeded max option steps.")

    monkeypatch.setattr(testing, "run_episode_and_get_observations", _boom)
    cup = Object("cup", Type("cup", ["x"]))
    obs = State({cup: np.zeros(1, dtype=np.float32)})
    episode_env: Any = SimpleNamespace(get_observation=lambda: obs)
    cogman: Any = SimpleNamespace(_approach=SimpleNamespace())
    env_task: Any = None
    metrics = TestMetrics()
    outcome = _execute_policy(cogman, 0, env_task, episode_env, None, metrics,
                              TestArtifacts(None))
    assert not outcome.solved
    assert outcome.caught_exception
    assert metrics.num_execution_failures == 1
