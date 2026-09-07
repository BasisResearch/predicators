"""The learning pipeline: offline learning, the online learning loop, and the
interaction episodes it learns from.

``run_pipeline`` is what ``main`` hands the constructed cogman to.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

from predicators import utils
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs import BaseEnv
from predicators.run.checkpoints import ApproachCheckpoints, \
    InflightInteractions, test_results_exist
from predicators.run.early_stopping import EarlyStopping, below_reward_bar_msg
from predicators.run.testing import format_test_results_line, run_testing, \
    save_test_results
from predicators.settings import CFG, get_allowed_query_type_names
from predicators.structs import Dataset, InteractionRequest, \
    InteractionResult, Metrics, Response, Task
from predicators.teacher import Teacher, TeacherInteractionMonitorWithVideo


def run_pipeline(env: BaseEnv,
                 cogman: CogMan,
                 train_tasks: List[Task],
                 offline_dataset: Optional[Dataset] = None) -> None:
    """Main pipeline for running the learning and testing process."""
    if cogman.is_learning_based:
        assert offline_dataset is not None, "Missing offline dataset"

        # Handle offline learning phase
        num_offline_trans, num_online_trans, learning_time, offline_metrics = \
            _handle_offline_learning(cogman, offline_dataset)

        # Run initial evaluation if needed
        initial_test_summary: Optional[Tuple[str, Metrics]] = None
        if CFG.skip_until_cycle < 0 and \
           not CFG.skip_test_until_last_ite_or_early_stopping and \
           not CFG.skip_initial_test:
            results = run_testing(env, cogman, online_learning_cycle=None)
            results.update({
                "num_offline_transitions": num_offline_trans,
                "num_online_transitions": num_online_trans,
                "query_cost": 0.0,
                "learning_time": learning_time,
                **offline_metrics
            })
            save_test_results(results, online_learning_cycle=None)
            initial_test_summary = ("the pre-loop test", results)

        # Run online learning loop
        run_online_learning_loop(env, cogman, train_tasks, num_offline_trans,
                                 learning_time, offline_metrics,
                                 initial_test_summary)
    else:
        # Handle non-learning case
        results = run_testing(env, cogman, online_learning_cycle=None)
        results.update({
            "num_offline_transitions": 0,
            "num_online_transitions": 0,
            "query_cost": 0.0,
            "learning_time": 0.0
        })
        save_test_results(results, online_learning_cycle=None)


def _handle_offline_learning(
        cogman: CogMan,
        offline_dataset: Dataset) -> Tuple[int, float, float, dict]:
    """Handle offline learning phase and initial evaluation."""
    num_offline_transitions = sum(
        len(traj.actions) for traj in offline_dataset.trajectories)
    auto_resume = bool(getattr(CFG, "auto_resume", False))
    if CFG.load_approach and (
            not auto_resume
            or ApproachCheckpoints.for_cogman(cogman).exists(None)):
        # Plain --load_approach stays strict (a missing file raises).
        cogman.load(online_learning_cycle=None)
        learning_time = 0.0  # ignore loading time
    elif CFG.load_approach:
        # --auto_resume over checkpoints from before the post-offline
        # ``_None`` file existed (or a deleted one): re-run offline
        # learning rather than crashing the resume; the online loop still
        # loads the per-cycle checkpoint it skips to.
        logging.warning(
            "--auto_resume: no post-offline (_None) checkpoint found; "
            "running offline learning instead of loading it.")
        learning_start = time.perf_counter()
        cogman.learn_from_offline_dataset(offline_dataset)
        learning_time = time.perf_counter() - learning_start
    else:
        learning_start = time.perf_counter()
        cogman.learn_from_offline_dataset(offline_dataset)
        learning_time = time.perf_counter() - learning_start

    offline_learning_metrics = {
        f"offline_learning_{k}": v
        for k, v in cogman.metrics.items()
    }

    return num_offline_transitions, 0.0, learning_time, offline_learning_metrics


def run_online_learning_loop(
        env: BaseEnv,
        cogman: CogMan,
        train_tasks: List[Task],
        num_offline_transitions: int,
        learning_time: float,
        offline_learning_metrics: dict,
        initial_test_summary: Optional[Tuple[str, Metrics]] = None) -> None:
    """Run the online learning loop.

    Each cycle: (resume bookkeeping) -> test-driven early stop? ->
    interact -> train-driven early stop? -> learn -> test. See
    ``run.early_stopping`` for the two stopping modes and
    ``run.checkpoints`` for what a resumed run picks up from disk.

    ``initial_test_summary`` is ``(label, results)`` from the pre-loop
    test, if one ran; it seeds the last-test summary that gets logged
    when early stopping triggers.
    """
    num_online_transitions = 0
    total_query_cost = 0.0
    checkpoints = ApproachCheckpoints.for_cogman(cogman)
    early_stopping = EarlyStopping(
        num_train_tasks=len(train_tasks),
        # Train-driven early stopping certifies the *learned* model:
        # offline demos or a loaded approach count as having learned.
        model_has_learned=CFG.load_approach or num_offline_transitions > 0,
        initial_test_summary=initial_test_summary)
    early_stopping.seed_test_streak_from_disk(CFG.skip_until_cycle - 1)

    def _test_and_save(cycle: int) -> None:
        """Test the current model as ``cycle``'s evaluation and record it."""
        results = run_testing(env, cogman, online_learning_cycle=cycle)
        results.update({
            "num_offline_transitions": num_offline_transitions,
            "num_online_transitions": num_online_transitions,
            "query_cost": total_query_cost,
            "learning_time": learning_time,
            **offline_learning_metrics
        })
        save_test_results(results, online_learning_cycle=cycle)
        early_stopping.record_test(f"cycle {cycle}", results)

    def _log_last_test() -> None:
        if early_stopping.last_test_summary is not None:
            label, test_results = early_stopping.last_test_summary
            logging.info(f"Early stopping: last test evaluation ({label}): "
                         f"{format_test_results_line(test_results)}")

    # Create teacher if needed
    teacher = Teacher(train_tasks) if get_allowed_query_type_names() else None
    load_approach = CFG.load_approach
    ran_any_cycle = False

    for i in range(CFG.num_online_learning_cycles):
        if i < CFG.skip_until_cycle:
            continue
        ran_any_cycle = True

        # Handle loading approach
        if load_approach and i > 0:
            cogman.load(online_learning_cycle=i - 1)
            if CFG.restart_learning:
                load_approach = False
            # A cycle's checkpoint is written at the end of its LEARN,
            # before its test; a kill during the test phase leaves a
            # loadable cycle i-1 with no test results. Run that test
            # first so the resumed run loses no evaluation datapoint.
            if (i == CFG.skip_until_cycle
                    and not CFG.skip_test_until_last_ite_or_early_stopping
                    and not test_results_exist(i - 1)):
                logging.info(
                    "Resumed past cycle %d whose test never ran; testing "
                    "it now before continuing.", i - 1)
                _test_and_save(i - 1)
                # Cycle i-1's result now completes the on-disk record, so
                # re-derive the streak from it (the pre-loop seed stopped
                # at the then-missing cycle).
                early_stopping.seed_test_streak_from_disk(i - 1)

        # Run online interaction
        logging.info(f"\n\nONLINE LEARNING CYCLE {i}\n")
        if num_online_transitions >= CFG.online_learning_max_transitions:
            logging.info(
                "Reached online_learning_max_transitions, terminating")
            break

        if early_stopping.test_driven_stop():
            if early_stopping.force_final_test:
                _test_and_save(i)
            _log_last_test()
            break

        # A resumed-into cycle whose previous incarnation died during
        # LEARN reuses the episodes persisted just before that learn
        # (see ApproachCheckpoints.save_inflight) instead of re-exploring.
        inflight: Optional[InflightInteractions] = None
        if getattr(CFG, "auto_resume", False) and i == CFG.skip_until_cycle:
            inflight = checkpoints.load_inflight(i)
        if inflight is not None:
            interaction_results = inflight.interaction_results
            task_idxs = inflight.task_idxs
            task_solved_status = inflight.task_solved_status
            query_cost = inflight.query_cost
            # get_interaction_requests is skipped, so hand the approach
            # the result->train-task pairing it would have recorded.
            cogman.restore_interaction_requests(task_idxs)
            logging.info(
                "Resuming cycle %d from %d persisted interaction "
                "episode(s); skipping this cycle's exploration and "
                "going straight to learning.", i, len(interaction_results))
        else:
            interaction_requests = cogman.get_interaction_requests()
            if not interaction_requests:
                logging.info(
                    "Did not receive any interaction requests, terminating")
                break

            (interaction_results, query_cost,
             task_solved_status) = \
                generate_interaction_results(
                    cogman, env, teacher,
                    interaction_requests, i)
            task_idxs = [req.train_task_idx for req in interaction_requests]
            checkpoints.save_inflight(
                InflightInteractions(cycle=i,
                                     interaction_results=interaction_results,
                                     task_idxs=task_idxs,
                                     task_solved_status=task_solved_status,
                                     query_cost=query_cost))

        num_online_transitions += sum(
            len(result.actions) for result in interaction_results)
        total_query_cost += query_cost
        logging.info(f"Query cost incurred this cycle: {query_cost}")
        early_stopping.record_train_attempts(task_idxs, task_solved_status)

        is_last_iteration = (i == CFG.num_online_learning_cycles - 1)
        should_run_testing = (
            is_last_iteration
            or not CFG.skip_test_until_last_ite_or_early_stopping)
        stop_now = early_stopping.train_driven_stop()
        if stop_now:
            should_run_testing = early_stopping.force_final_test
        # Learn from results if appropriate
        if (not CFG.load_approach or CFG.restart_learning) and \
            not stop_now:
            learning_start = time.perf_counter()
            logging.info("Learning from interaction results...")
            cogman.learn_from_interaction_results(interaction_results)
            learning_time += time.perf_counter() - learning_start
            early_stopping.record_learned()
        # The cycle's episodes are consumed (and, when learning ran, the
        # per-cycle checkpoint was written inside it): drop the stash.
        checkpoints.discard_inflight(i)

        # Evaluate if needed
        if should_run_testing:
            _test_and_save(i)
        elif stop_now:
            prev_test_label = f"cycle {i - 1}" if i > 0 else "the pre-loop test"
            logging.info(
                f"Skipping testing for early-stopping cycle {i}: model is "
                f"unchanged from {prev_test_label} (learning skipped this "
                "cycle), which was already tested. See "
                "online_learning_early_stopping_skip_redundant_test.")
        else:
            logging.info("Skipping testing for cycle "
                         f"{i} due to "
                         "skip_test_until_last_ite_or"
                         "_early_stopping flag")

        if stop_now:
            _log_last_test()
            break

    # Resume landed past the final cycle (e.g. preempted during or after
    # the last cycle's test phase): every learn is checkpointed but the
    # loop body never ran, so no final test would ever be produced. Load
    # the last cycle's checkpoint and run just the test.
    if (not ran_any_cycle and CFG.load_approach
            and CFG.skip_until_cycle >= CFG.num_online_learning_cycles > 0):
        last_cycle = CFG.num_online_learning_cycles - 1
        logging.info(
            "All %d online learning cycles were already completed at "
            "resume; loading cycle %d and running the final test only.",
            CFG.num_online_learning_cycles, last_cycle)
        cogman.load(online_learning_cycle=last_cycle)
        _test_and_save(last_cycle)


def generate_interaction_results(
    cogman: CogMan,
    env: BaseEnv,
    teacher: Optional[Teacher],
    requests: Sequence[InteractionRequest],
    cycle_num: Optional[int] = None
) -> Tuple[List[InteractionResult], float, List[bool]]:
    """Given a sequence of InteractionRequest objects, handle the requests and
    return a list of InteractionResult objects."""
    logging.info("Generating interaction results...")
    results = []
    query_cost = 0.0
    task_solved_status = []
    for episode_idx, request in enumerate(requests):
        if request.train_task_idx < CFG.max_initial_demos and \
            not CFG.allow_interaction_in_demo_tasks:
            raise RuntimeError("Interaction requests cannot be on demo tasks "
                               "if allow_interaction_in_demo_tasks is False.")
        monitor: Optional[utils.VideoMonitor] = None
        if teacher is not None:
            monitor = TeacherInteractionMonitorWithVideo(
                env.render, request, teacher)
        elif CFG.make_interaction_videos:
            monitor = utils.VideoMonitor(env.render)

        # Used to check if our think the approach is unsolvable.
        if CFG.env_has_impossible_goals:
            planning_explorer_generated_a_plan = True
            if 'RandomNSRTsExplorer' in request.act_policy.__qualname__:
                planning_explorer_generated_a_plan = False
        cogman.set_override_policy(request.act_policy)
        cogman.set_termination_function(request.termination_function)
        env_task = env.get_train_tasks()[request.train_task_idx]
        cogman.reset(env_task)
        observed_traj, solved, _ = run_episode_and_get_observations(
            cogman,
            env,
            "train",
            request.train_task_idx,
            max_num_steps=(CFG.max_num_steps_interaction_request + 1),
            terminate_on_goal_reached=False,
            exceptions_to_break_on={
                utils.EnvironmentFailure,
                utils.OptionExecutionFailure,
                utils.RequestActPolicyFailure,
            },
            monitor=monitor)
        if CFG.env_has_impossible_goals:
            task_solvable = env.is_task_solvable(env_task)
            if not task_solvable:
                solved = not planning_explorer_generated_a_plan
        # A planning explorer may report that its mental model could NOT
        # reach the goal during refinement (it then ran the plan as an
        # experiment). Don't certify such a task as solved for early
        # stopping even if real-env execution happened to reach the goal —
        # the learned model still can't be planned with. None ⇒ no verdict.
        if request.mental_model_solved is False:
            solved = False
        task_solved_status.append(solved)

        # Debug final state (mirrors run.testing). Lets us inspect the real
        # env state at the end of the rollout — e.g. whether SwitchBurnerOff
        # actually flipped the burner — separately from what the agent's
        # mental model believes happened.
        # pylint: disable=protected-access
        final_obs = env.get_observation()
        logging.debug(f"Interaction goal:\n{env_task.task.goal}")
        if hasattr(cogman._approach, "_get_current_predicates"):
            abstract_state = utils.abstract(
                final_obs, cogman._approach._get_current_predicates())
            logging.debug(f"Interaction final abstract state:\n"
                          f"{abstract_state}")
        # pylint: enable=protected-access
        logging.debug(f"Interaction final state (solved={solved}):\n"
                      f"{final_obs.pretty_str()}")
        cogman.unset_override_policy()
        cogman.unset_termination_function()
        traj = cogman.get_current_history()
        request_responses: List[Optional[Response]] = [
            None for _ in traj.states
        ]
        if isinstance(monitor, TeacherInteractionMonitorWithVideo):
            request_responses = monitor.get_responses()
            query_cost += monitor.get_query_cost()
        assert len(traj.states) == len(observed_traj[0])
        assert len(traj.actions) == len(observed_traj[1])
        # The env evaluator's verdict on this episode. Only the (reward,
        # terminated) pair travels to the agent side (rejection is
        # decodable from it) - the agent must infer the violated rule
        # from the task's NL description; the specific reason stays here
        # in the logs.
        episode_eval = env.evaluate_episode(observed_traj[0], observed_traj[1])
        accepted = episode_eval.terminated and not episode_eval.rejected
        logging.info(
            "Interaction episode on train task %d: reward=%.2f, "
            "terminated=%s, accepted=%s", request.train_task_idx,
            episode_eval.reward, episode_eval.terminated, accepted)
        if episode_eval.rejected:
            logging.info(
                "Interaction episode on train task %d REJECTED by the "
                "env: %s", request.train_task_idx, episode_eval.reason)
            # A rule-breaking episode must never count as solved for the
            # early-stopping criterion, regardless of how the cogman solve
            # gate scored it (today the gate runs the same certificate, so
            # this is belt-and-braces; it keeps the invariant local and
            # explicit).
            task_solved_status[-1] = False
        if task_solved_status[-1]:
            below_bar_msg = below_reward_bar_msg(episode_eval.reward, env_task)
            if below_bar_msg is not None:
                logging.info(
                    "Interaction episode on train task %d solved but %s: "
                    "does NOT count as solved for early stopping.",
                    request.train_task_idx, below_bar_msg)
                task_solved_status[-1] = False
        result = InteractionResult(traj.states,
                                   traj.actions,
                                   request_responses,
                                   episode_reward=episode_eval.reward,
                                   episode_terminated=episode_eval.terminated)
        results.append(result)
        if CFG.make_interaction_videos:
            assert monitor is not None
            # One video per interaction episode, saved as soon as the
            # episode ends so a mid-cycle crash keeps earlier footage.
            # scripts/log_viewer.py parses the __ep<i>__cycle<C>.mp4 tail
            # to pair each explore transcript with its episode's video.
            save_prefix = utils.get_config_path_str()
            outfile = f"{save_prefix}__ep{episode_idx}__cycle{cycle_num}.mp4"
            utils.save_video(outfile, monitor.get_video())
    return results, query_cost, task_solved_status
