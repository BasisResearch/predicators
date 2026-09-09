"""Evaluation of the deployed approach on the environment's test tasks.

``run_testing`` solves and executes every test task, rendering videos
or images when configured, and returns one ``Metrics`` round;
``save_test_results`` writes that round next to the config so later
cycles (and ``run.checkpoints``) can find it.

Per task the phases are:

1. solve - ``cogman.reset(task)`` plans; a timeout/failure is counted
   and the task skipped (``_solve_task``);
2. execute - the policy runs in the (possibly fresh) env under a video
   monitor; the env's task evaluator scores the trajectory
   (``_execute_task``);
3. record - the outcome updates ``TestMetrics`` and the monitor's
   frames are saved by ``TestArtifacts`` when the config asks for them.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import dill as pkl

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs import BaseEnv
from predicators.run.checkpoints import test_results_path
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Metrics, Observation

Trajectory = Tuple[List[Observation], List[Action]]

# CogMan totals reported as per-found-policy averages.
_AVERAGED_COGMAN_METRICS = (
    "num_samples",
    "num_skeletons_optimized",
    "num_nodes_expanded",
    "num_nodes_created",
    "num_nsrts",
    "num_preds",
    "plan_length",
    "num_failures_discovered",
)

# ── Accumulators ─────────────────────────────────────────────────


@dataclass
class TestMetrics:
    """Running totals of one test round, finalized into a ``Metrics``.

    ``per_task`` holds the ``PER_TASK_task{i}_*`` entries; everything
    else is a sum over tasks. Tasks that never execute (solve
    failure/timeout) contribute 0.0 reward, so ``avg_test_reward`` is
    over ALL test tasks.
    """
    per_task: Metrics = field(default_factory=lambda: defaultdict(float))
    num_found_policy: int = 0
    num_solved: int = 0
    total_test_reward: float = 0.0
    total_suc_time: float = 0.0
    total_low_level_action_cost: float = 0.0
    num_solve_timeouts: int = 0
    num_solve_failures: int = 0
    num_execution_timeouts: int = 0
    num_execution_failures: int = 0
    # CogMan's node counters are cumulative over the round; these hold
    # the values at the previous task so per-task deltas can be taken.
    _nodes_created_so_far: float = 0.0
    _nodes_expanded_so_far: float = 0.0

    def record_solve(self, task_idx: int, solve_time: float,
                     cogman: CogMan) -> None:
        """A task was solved (a policy was found) in ``solve_time``."""
        self.per_task[f"PER_TASK_task{task_idx}_solve_time"] = solve_time
        created = cogman.metrics["total_num_nodes_created"]
        expanded = cogman.metrics["total_num_nodes_expanded"]
        self.per_task[f"PER_TASK_task{task_idx}_nodes_created"] = \
            created - self._nodes_created_so_far
        self.per_task[f"PER_TASK_task{task_idx}_nodes_expanded"] = \
            expanded - self._nodes_expanded_so_far
        self._nodes_created_so_far = created
        self._nodes_expanded_so_far = expanded
        self.num_found_policy += 1

    def record_solve_exception(
            self, e: Union[ApproachTimeout, ApproachFailure]) -> None:
        """The solve step raised."""
        if isinstance(e, ApproachTimeout):
            self.num_solve_timeouts += 1
        else:
            self.num_solve_failures += 1

    def finalize(self, cogman: CogMan, num_test_tasks: int) -> Metrics:
        """The round's ``Metrics``: per-task entries plus aggregates."""
        metrics: Metrics = defaultdict(float)
        metrics.update(self.per_task)
        num_solved = self.num_solved
        metrics["num_solved"] = num_solved
        metrics["num_total"] = num_test_tasks
        metrics["avg_test_reward"] = (self.total_test_reward / num_test_tasks
                                      if num_test_tasks else 0.0)
        metrics["avg_suc_time"] = (self.total_suc_time / num_solved
                                   if num_solved > 0 else float("inf"))
        metrics["avg_ref_cost"] = ((self.total_low_level_action_cost +
                                    cogman.metrics["total_refinement_time"]) /
                                   num_solved
                                   if num_solved > 0 else float("inf"))

        # Skeleton / sample info
        metrics["min_num_samples"] = (
            cogman.metrics["min_num_samples"]
            if cogman.metrics["min_num_samples"] < float("inf") else 0)
        metrics["max_num_samples"] = cogman.metrics["max_num_samples"]
        metrics["min_skeletons_optimized"] = (
            cogman.metrics["min_num_skeletons_optimized"]
            if cogman.metrics["min_num_skeletons_optimized"] < float("inf")
            else 0)
        metrics["max_skeletons_optimized"] = cogman.metrics[
            "max_num_skeletons_optimized"]

        # Failure/timeouts
        metrics["num_solve_timeouts"] = self.num_solve_timeouts
        metrics["num_solve_failures"] = self.num_solve_failures
        metrics["num_execution_timeouts"] = self.num_execution_timeouts
        metrics["num_execution_failures"] = self.num_execution_failures

        # Averages of certain CogMan metrics wrt # of found policies
        for metric_name in _AVERAGED_COGMAN_METRICS:
            total = cogman.metrics[f"total_{metric_name}"]
            metrics[f"avg_{metric_name}"] = (total / self.num_found_policy
                                             if self.num_found_policy > 0 else
                                             float("inf"))
        return metrics


# ── Rendered artifacts ───────────────────────────────────────────


class TestArtifacts:
    """Names and saves one test round's videos and image sequences.

    ``online_learning_cycle`` is woven into every filename so successive
    test rounds do NOT overwrite each other, matching how
    ``save_test_results`` suffixes the metrics pkl with the cycle.
    """

    def __init__(self, online_learning_cycle: Optional[int]) -> None:
        self._cycle = online_learning_cycle
        self._save_prefix = utils.get_config_path_str()
        self._cycle_tag = f"__cycle{online_learning_cycle}"

    @property
    def save_prefix(self) -> str:
        """The config path string every artifact name starts with."""
        return self._save_prefix

    def _task_stem(self, task_idx: int, is_failure: bool) -> str:
        suffix = "_failure" if is_failure else ""
        return f"{self._save_prefix}__task{task_idx+1}{suffix}{self._cycle_tag}"

    def _query_image_dir(self, task_idx: int) -> str:
        experiment_id = CFG.experiment_id.split("-")[0]
        return (f"{experiment_id}/seed{CFG.seed}/query/"
                f"cycle{self._cycle}/task{task_idx+1}/")

    def save_video(self, monitor: Optional[utils.LoggingMonitor],
                   is_failure: bool, task_idx: int) -> None:
        """Save the monitor's video, if there is a monitor."""
        if monitor is None:
            return
        if CFG.use_counterfactual_dataset_path_name:
            is_failure = False
        outfile = f"{self._task_stem(task_idx, is_failure)}.mp4"
        if isinstance(monitor, utils.StreamingVideoMonitor):
            monitor.finalize(outfile)
        else:
            assert isinstance(monitor, utils.VideoMonitor)
            utils.save_video(outfile, monitor.get_video())

    def save_images(self, monitor: Optional[utils.LoggingMonitor],
                    is_failure: bool, task_idx: int) -> None:
        """Save the monitor's frames as images, if there is a monitor."""
        if monitor is None:
            return
        assert isinstance(monitor, utils.VideoMonitor)
        video = monitor.get_video()
        if CFG.use_counterfactual_dataset_path_name:
            outfile = self._query_image_dir(task_idx)
        else:
            outfile = self._task_stem(task_idx, is_failure)
        utils.save_images(outfile, video)

    def save_partial_refinements(self, env: BaseEnv, task_idx: int,
                                 partial_refinements: Any) -> None:
        """Render a failed solve's partial refinements, if configured."""
        if not (CFG.make_failure_videos or CFG.make_failure_images):
            return
        if not partial_refinements:
            return
        logging.info("Creating video from partial refinements...")
        video = utils.create_video_from_partial_refinements(
            partial_refinements, env, "test", task_idx, CFG.horizon)
        if CFG.make_failure_images:
            utils.save_images(self._query_image_dir(task_idx), video)
        if CFG.make_failure_videos:
            utils.save_video(f"{self._task_stem(task_idx, True)}.mp4", video)


# ── Per-task phases ──────────────────────────────────────────────


@dataclass(frozen=True)
class ExecutionOutcome:
    """What running the policy on one task produced."""
    solved: bool
    caught_exception: bool
    exec_time: float
    num_options_executed: int
    traj: Trajectory


def _solve_task(cogman: CogMan, env_task: EnvironmentTask) -> float:
    """Plan for ``env_task``; returns the solve time.

    May raise ApproachTimeout or ApproachFailure.
    """
    solve_start = time.perf_counter()
    logging.debug(f"[main.py] Solving task w. goal: {env_task.goal}")
    cogman.reset(env_task)
    return time.perf_counter() - solve_start


def _save_eval_trajectory(save_prefix: str, task_idx: int,
                          env_task: EnvironmentTask, traj: Trajectory) -> None:
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)
    traj_file = f"{save_prefix}__task{task_idx+1}.traj"
    traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
    traj_data = {
        "task": env_task,
        "trajectory": traj,
        "pybullet_robot": CFG.pybullet_robot
    }
    with open(traj_file_path, "wb") as f:
        pkl.dump(traj_data, f)


def _log_final_state(cogman: CogMan, episode_env: BaseEnv) -> None:
    # pylint: disable=protected-access
    if hasattr(cogman._approach, "_get_current_predicates"):
        abstract_state = utils.abstract(
            episode_env.get_observation(),
            cogman._approach._get_current_predicates())
        logging.debug(f"Final abstract state:\n{abstract_state}")
    logging.debug(
        f"Final state:\n{episode_env.get_observation().pretty_str()}")


def _execute_policy(cogman: CogMan, task_idx: int, env_task: EnvironmentTask,
                    episode_env: BaseEnv,
                    monitor: Optional[utils.LoggingMonitor],
                    metrics: TestMetrics,
                    artifacts: TestArtifacts) -> ExecutionOutcome:
    """Run the cogman policy in ``episode_env`` to see if the goal is solved.

    Execution-time approach timeouts/failures are counted on ``metrics``
    and reported as ``caught_exception``.
    """
    solved = False
    caught_exception = False
    exec_time = 0.0
    num_options_executed = 0
    traj: Trajectory = ([], [])
    try:
        traj, solved, execution_metrics = run_episode_and_get_observations(
            cogman,
            episode_env,
            "test",
            task_idx,
            max_num_steps=CFG.horizon,
            monitor=monitor,
            terminate_on_goal_reached=CFG.terminate_on_goal_reached)
        exec_time = execution_metrics["policy_call_time"]
        num_options_executed = int(execution_metrics["num_options_executed"])
        if CFG.save_eval_trajs:
            _save_eval_trajectory(artifacts.save_prefix, task_idx, env_task,
                                  traj)
    except utils.EnvironmentFailure as e:
        logging.info(f"Environment failed with error: {e}")
        caught_exception = True
    except (ApproachTimeout, ApproachFailure,
            utils.OptionExecutionFailure) as e:
        # OptionExecutionFailure (an option that never terminated or a
        # plan that ran out) reaches here from approaches that hand the
        # option-policy wrapper straight to the cogman without wrapping
        # its failures; it is this task failing, not the run.
        logging.info(f"Approach failed at execution time with error: {e}")
        if isinstance(e, ApproachTimeout):
            metrics.num_execution_timeouts += 1
        else:
            metrics.num_execution_failures += 1
        caught_exception = True
    _log_final_state(cogman, episode_env)
    return ExecutionOutcome(solved, caught_exception, exec_time,
                            num_options_executed, traj)


def _make_monitor(episode_env: BaseEnv) -> Optional[utils.LoggingMonitor]:
    """The frame monitor the config calls for, if any.

    Image saving needs the raw frames after the episode, so it gets the
    buffering monitor; video-only runs stream frames to disk as they are
    rendered, keeping peak memory at one frame instead of a whole
    episode.
    """
    if CFG.make_test_images or CFG.make_failure_images:
        return utils.VideoMonitor(episode_env.render)
    if CFG.make_test_videos or CFG.make_failure_videos:
        return utils.StreamingVideoMonitor(episode_env.render)
    return None


def _execute_task(env: BaseEnv, cogman: CogMan, task_idx: int,
                  env_task: EnvironmentTask, solve_time: float,
                  metrics: TestMetrics, artifacts: TestArtifacts) -> str:
    """Execution phase for a task whose solve found a policy.

    Returns the outcome label to log.
    """
    # Run the episode in a fresh env instance when the env supports it
    # (see BaseEnv.make_fresh_test_instance): a long-lived PyBullet
    # world carries history that state-level resets do not clear, so
    # the episode's physics would depend on everything the run executed
    # before it.
    episode_env: BaseEnv = env
    fresh_env: Optional[BaseEnv] = None
    if CFG.test_fresh_env_per_episode:
        fresh_env = env.make_fresh_test_instance()
        if fresh_env is not None:
            episode_env = fresh_env
        else:
            logging.info(
                "test_fresh_env_per_episode: env does not support a fresh "
                "instance here (GUI/real-robot/base env); executing in the "
                "shared long-lived env.")

    monitor: Optional[utils.LoggingMonitor] = None
    try:
        monitor = _make_monitor(episode_env)
        logging.info("Executing policy...")
        outcome = _execute_policy(cogman, task_idx, env_task, episode_env,
                                  monitor, metrics, artifacts)
        traj = outcome.traj
        per_task = metrics.per_task
        per_task[f"PER_TASK_task{task_idx}_exec_time"] = outcome.exec_time
        per_task[f"PER_TASK_task{task_idx}_options_executed"] = \
            outcome.num_options_executed

        # Task-evaluator verdict + offline metrics (e.g. domino k_used),
        # plus per-task oracle quantities (e.g. domino k_star) stored on
        # the EnvironmentTask. Offline-only: reported in results, never
        # agent-visible.
        if traj[0]:
            episode_eval = episode_env.evaluate_episode(traj[0], traj[1])
            per_task[f"PER_TASK_task{task_idx}_reward"] = episode_eval.reward
            metrics.total_test_reward += episode_eval.reward
            for metric_name, value in episode_eval.offline_metrics.items():
                per_task[f"PER_TASK_task{task_idx}_{metric_name}"] = value
            for metric_name, value in env_task.offline_task_metrics.items():
                per_task[f"PER_TASK_task{task_idx}_{metric_name}"] = value

        if CFG.refinement_data_include_execution_cost:
            metrics.total_low_level_action_cost += (
                len(traj[1]) * CFG.refinement_data_low_level_execution_cost)

        if outcome.solved and not outcome.caught_exception:
            log_msg = "SOLVED"
            metrics.num_solved += 1
            metrics.total_suc_time += solve_time + outcome.exec_time
            if CFG.make_test_videos:
                artifacts.save_video(monitor, False, task_idx)
            if CFG.make_test_images:
                artifacts.save_images(monitor, False, task_idx)
            per_task[f"PER_TASK_task{task_idx}_num_steps"] = len(traj[1])
        else:
            if not outcome.caught_exception:
                log_msg = "Policy failed to reach goal"
            else:
                log_msg = "Policy/Env encountered an exception"
            if CFG.crash_on_failure:
                raise RuntimeError(log_msg)
            if CFG.make_failure_videos:
                artifacts.save_video(monitor, True, task_idx)
            if CFG.make_failure_images:
                artifacts.save_images(monitor, True, task_idx)
    finally:
        # Drop the streamed clip when no branch above finalized it (a
        # solved episode with only make_failure_videos on, or an
        # exception past the save calls); no-op otherwise. In the
        # finally so a raise inside the try cannot leak the monitor's
        # temp file and open writer.
        if isinstance(monitor, utils.StreamingVideoMonitor):
            monitor.discard()
        if fresh_env is not None:
            fresh_env.dispose()
    return log_msg


def _evaluate_task(env: BaseEnv, cogman: CogMan, task_idx: int, num_tasks: int,
                   env_task: EnvironmentTask, metrics: TestMetrics,
                   artifacts: TestArtifacts) -> None:
    """Solve then execute one test task, recording the outcome."""
    try:
        logging.info(f"[main.py] Solving task {task_idx+1}/{num_tasks}...")
        solve_time = _solve_task(cogman, env_task)
    except (ApproachTimeout, ApproachFailure) as e:
        logging.info(f"[main.py] Task {task_idx+1} / {num_tasks}: approach "
                     f"failed with error: {e}")
        metrics.record_solve_exception(e)
        partial_refinements = getattr(e, "info", {}).get("partial_refinements")
        artifacts.save_partial_refinements(env, task_idx, partial_refinements)
        if CFG.crash_on_failure:
            raise e
        # Recognizing an impossible goal counts as solving it.
        if CFG.env_has_impossible_goals and \
                not env.is_task_solvable(env_task) and \
                "not dr-reachable" in str(e):
            logging.info("[main.py] Task is unsolvable and is recognized")
            metrics.num_solved += 1
            logging.info(f"Task {task_idx+1} / {num_tasks}: SOLVED")
        return
    metrics.record_solve(task_idx, solve_time, cogman)
    log_msg = _execute_task(env, cogman, task_idx, env_task, solve_time,
                            metrics, artifacts)
    logging.info(f"Task {task_idx+1} / {num_tasks}: {log_msg}")


# ── Public API ───────────────────────────────────────────────────


def run_testing(env: BaseEnv,
                cogman: CogMan,
                online_learning_cycle: Optional[int] = None) -> Metrics:
    """Evaluate the cogman approach on the environment's test tasks.

    ``online_learning_cycle`` is the cycle this test round belongs to
    (``None`` for the pre-learning baseline, ``i`` for the test after
    cycle ``i``); it is woven into the saved image/video filenames so
    successive test rounds do NOT overwrite each other.

    Returns the round's aggregated ``Metrics``.
    """
    test_tasks: Sequence[EnvironmentTask] = env.get_test_tasks()
    if CFG.approach != "oracle":
        test_tasks = [task.replace_goal_with_alt_goal() for task in test_tasks]
    cogman.reset_metrics()
    metrics = TestMetrics()
    artifacts = TestArtifacts(online_learning_cycle)
    cogman._approach.begin_test_phase()  # pylint: disable=protected-access
    for task_idx, env_task in enumerate(test_tasks):
        _evaluate_task(env, cogman, task_idx, len(test_tasks), env_task,
                       metrics, artifacts)
    cogman._approach.end_test_phase()  # pylint: disable=protected-access
    return metrics.finalize(cogman, len(test_tasks))


def format_per_task_rewards(results: Metrics) -> str:
    """Comma-joined per-task episode rewards of one test round.

    Tasks that never executed (solve failure/timeout) have no reward
    entry and show as ``n/a``.
    """
    parts = []
    for i in range(int(results["num_total"])):
        reward = results.get(f"PER_TASK_task{i}_reward")
        parts.append(
            f"task{i}={reward:.2f}" if reward is not None else f"task{i}=n/a")
    return ", ".join(parts)


def format_test_results_line(results: Metrics) -> str:
    """Summarize a test round: solve rate, average reward, per-task rewards."""
    num_solved = int(results["num_solved"])
    num_total = int(results["num_total"])
    rate = num_solved / num_total if num_total else 0.0
    return (f"solve rate {rate:.3f} ({num_solved} / {num_total}), "
            f"avg reward {results['avg_test_reward']:.3f}, "
            f"per-task rewards: {format_per_task_rewards(results)}")


def save_test_results(results: Metrics,
                      online_learning_cycle: Optional[int]) -> None:
    """Log a test round and pickle it (with CFG and the git hash) to
    ``test_results_path(online_learning_cycle)``.

    Strips the ``PER_TASK_`` entries from ``results`` in place after
    saving, so the logged summary stays short.
    """
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average test reward: {results['avg_test_reward']:.3f}")
    logging.info(f"Per-task rewards: {format_per_task_rewards(results)}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    os.makedirs(CFG.results_dir, exist_ok=True)
    outfile = test_results_path(online_learning_cycle)
    outdata = {
        "config": CFG,
        "results": results.copy(),
        "git_commit_hash": utils.get_git_commit_hash()
    }
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    del_keys = [k for k in results if k.startswith("PER_TASK_")]
    for k in del_keys:
        del results[k]
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")
