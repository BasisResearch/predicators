"""Main entry point for running approaches in environments.

Example usage with learning NSRTs:
    python predicators/main.py --env cover --approach nsrt_learning --seed 0

Example usage with oracle NSRTs:
    python predicators/main.py --env cover --approach oracle --seed 0

Example with verbose logging:
    python predicators/main.py --env cover --approach oracle --seed 0 --debug

To load a saved approach:
    python predicators/main.py --env cover --approach nsrt_learning --seed 0 \
        --load_approach

To load saved data:
    python predicators/main.py --env cover --approach nsrt_learning --seed 0 \
        --load_data

To make videos of test tasks:
    python predicators/main.py --env cover --approach oracle --seed 0 \
        --make_test_videos --num_test_tasks 1

To run interactive learning approach:
    python predicators/main.py --env cover --approach interactive_learning \
         --seed 0

To exclude predicates:
    python predicators/main.py --env cover --approach oracle --seed 0 \
         --excluded_predicates Holding

To run grammar search predicate invention (example):
    python predicators/main.py --env cover --approach grammar_search_invention \
        --seed 0 --excluded_predicates all
"""

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Any

import dill as pkl

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv, create_new_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options, \
    parse_config_included_options
from predicators.perception import create_perceiver
from predicators.settings import CFG, get_allowed_query_type_names
from predicators.structs import Dataset, InteractionRequest, \
    InteractionResult, Metrics, Response, Task, Video
from predicators.teacher import Teacher, TeacherInteractionMonitorWithVideo
from predicators.approaches.base_approach import BaseApproach

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"



def setup_environment() -> Tuple[BaseEnv, List[Task], List[Task]]:
    """Create and setup the environment and tasks.
    
    Returns:
        Tuple containing:
        - The environment
        - The training tasks for the approach
        - The original training tasks
    """
    # Create environment
    env = create_new_env(CFG.env, do_cache=True, use_gui=CFG.use_gui)
    env.action_space.seed(CFG.seed)
    assert env.goal_predicates.issubset(env.predicates)

    # Setup predicates
    included_preds, excluded_preds = utils.parse_config_excluded_predicates(env)
    preds = utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds,
        env) if CFG.approach != "oracle" else included_preds

    # Create train tasks
    env_train_tasks = env.get_train_tasks()
    perceiver = create_perceiver(CFG.perceiver)
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]
    
    # Strip excluded predicates and prepare approach tasks
    stripped_train_tasks = [utils.strip_task(task, preds) for task in train_tasks]
    approach_train_tasks = [
        task.replace_goal_with_alt_goal() for task in stripped_train_tasks
    ]

    return env, approach_train_tasks, train_tasks


def setup_approach(env: BaseEnv, 
                  preds: set, 
                  approach_train_tasks: List[Task]) -> 'BaseApproach':
    """Create and setup the approach/agent.
    
    Returns:
        The configured approach
    """
    # Setup options
    if CFG.option_learner == "no_learning":
        options = get_gt_options(env.get_name())
    else:
        options = parse_config_included_options(env)

    # Create approach
    approach_name = CFG.approach
    if CFG.approach_wrapper:
        approach_name = f"{CFG.approach_wrapper}[{approach_name}]"
    
    return create_approach(approach_name, preds, options, env.types,
                         env.action_space, approach_train_tasks)


def create_offline_dataset(env: BaseEnv, 
                         train_tasks: List[Task],
                         preds: set,
                         approach: BaseApproach) -> Optional[Dataset]:
    """Create offline dataset if needed.
    
    Returns:
        Dataset if required, None otherwise
    """
    if approach.is_learning_based or CFG.make_demo_videos or \
        CFG.make_demo_images:

        options = get_gt_options(env.get_name()) if \
                        CFG.option_learner == "no_learning" \
                    else parse_config_included_options(env)
        return create_dataset(env, train_tasks, options, preds)
    return None


def main() -> None:
    """Main entry point for running approaches in environments."""
    script_start = time.perf_counter()
    
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    str_args = " ".join(sys.argv)

    # Setup logging and directories
    utils.configure_logging()
    os.makedirs(CFG.results_dir, exist_ok=True)
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)

    # Log initial info
    utils.log_initial_info(str_args)

    # Setup environment and tasks
    env, approach_train_tasks, train_tasks = setup_environment()
    
    # Setup predicates
    included_preds, excluded_preds = utils.parse_config_excluded_predicates(env)
    preds = utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds,
        env) if CFG.approach != "oracle" else included_preds

    # Create approach
    approach = setup_approach(env, preds, approach_train_tasks)

    # Create dataset and cognitive manager
    offline_dataset = create_offline_dataset(env, train_tasks, preds, approach)
    execution_monitor = create_execution_monitor(CFG.execution_monitor)
    cogman = CogMan(approach, create_perceiver(CFG.perceiver), execution_monitor)

    # Run pipeline
    _run_pipeline(env, cogman, approach_train_tasks, offline_dataset)
    
    # Log completion
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")



def _run_pipeline(env: BaseEnv,
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
        if CFG.skip_until_cycle < 0:
            results = _run_testing(env, cogman)
            results.update({
                "num_offline_transitions": num_offline_trans,
                "num_online_transitions": num_online_trans,
                "query_cost": 0.0,
                "learning_time": learning_time,
                **offline_metrics
            })
            _save_test_results(results, online_learning_cycle=None)

        # Run online learning loop
        _run_online_learning_loop(env, cogman, train_tasks, num_offline_trans,
                                learning_time, offline_metrics)
    else:
        # Handle non-learning case
        results = _run_testing(env, cogman)
        results.update({
            "num_offline_transitions": 0,
            "num_online_transitions": 0, 
            "query_cost": 0.0,
            "learning_time": 0.0
        })
        _save_test_results(results, online_learning_cycle=None)


def _handle_offline_learning(cogman: CogMan,
                           offline_dataset: Dataset) -> Tuple[float, float, float, dict]:
    """Handle offline learning phase and initial evaluation."""
    num_offline_transitions = sum(len(traj.actions) 
                                for traj in offline_dataset.trajectories)
    if CFG.load_approach:
        cogman.load(online_learning_cycle=None)
        learning_time = 0.0  # ignore loading time
    else:
        learning_start = time.perf_counter()
        cogman.learn_from_offline_dataset(offline_dataset)
        learning_time = time.perf_counter() - learning_start
    
    offline_learning_metrics = {
        f"offline_learning_{k}": v
        for k, v in cogman.metrics.items()
    }
    
    return num_offline_transitions, 0.0, learning_time, offline_learning_metrics

def _run_online_learning_loop(env: BaseEnv,
                            cogman: CogMan,
                            train_tasks: List[Task],
                            num_offline_transitions: int,
                            learning_time: float,
                            offline_learning_metrics: dict) -> None:
    """Run the online learning loop."""
    num_online_transitions = 0
    total_query_cost = 0.0
    
    # Create teacher if needed
    teacher = Teacher(train_tasks) if get_allowed_query_type_names() else None
    load_approach = CFG.load_approach

    for i in range(CFG.num_online_learning_cycles):
        if i < CFG.skip_until_cycle:
            continue

        # Handle loading approach
        if load_approach and i > 0:
            cogman.load(online_learning_cycle=i-1)
            if CFG.restart_learning:
                load_approach = False

        # Run online interaction
        logging.info(f"\n\nONLINE LEARNING CYCLE {i}\n")
        if num_online_transitions >= CFG.online_learning_max_transitions:
            logging.info("Reached online_learning_max_transitions, terminating")
            break

        interaction_requests = cogman.get_interaction_requests()
        if not interaction_requests:
            logging.info("Did not receive any interaction requests, terminating") 
            break

        interaction_results, query_cost = _generate_interaction_results(
            cogman, env, teacher, interaction_requests, i)
        
        num_online_transitions += sum(len(result.actions) 
                                    for result in interaction_results)
        total_query_cost += query_cost
        logging.info(f"Query cost incurred this cycle: {query_cost}")

        # Learn from results if appropriate
        if not CFG.load_approach or CFG.restart_learning:
            learning_start = time.perf_counter()
            logging.info("Learning from interaction results...")
            cogman.learn_from_interaction_results(interaction_results)
            learning_time += time.perf_counter() - learning_start

        # Evaluate
        results = _run_testing(env, cogman)
        results.update({
            "num_offline_transitions": num_offline_transitions,
            "num_online_transitions": num_online_transitions,
            "query_cost": total_query_cost,
            "learning_time": learning_time,
            **offline_learning_metrics
        })
        _save_test_results(results, online_learning_cycle=i)


def _generate_interaction_results(
        cogman: CogMan,
        env: BaseEnv,
        teacher: Optional[Teacher],
        requests: Sequence[InteractionRequest],
        cycle_num: Optional[int] = None
) -> Tuple[List[InteractionResult], float]:
    """Given a sequence of InteractionRequest objects, handle the requests and
    return a list of InteractionResult objects."""
    logging.info("Generating interaction results...")
    results = []
    query_cost = 0.0
    if CFG.make_interaction_videos:
        video: Video = []
    for request in requests:
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
        cogman.set_override_policy(request.act_policy)
        cogman.set_termination_function(request.termination_function)
        env_task = env.get_train_tasks()[request.train_task_idx]
        cogman.reset(env_task)
        observed_traj, _, _ = run_episode_and_get_observations(
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
        result = InteractionResult(traj.states, traj.actions,
                                   request_responses)
        results.append(result)
        if CFG.make_interaction_videos:
            assert monitor is not None
            video.extend(monitor.get_video())
    if CFG.make_interaction_videos:
        save_prefix = utils.get_config_path_str()
        outfile = f"{save_prefix}__cycle{cycle_num}.mp4"
        utils.save_video(outfile, video)
    return results, query_cost


def _run_testing(env: BaseEnv, cogman: CogMan) -> Metrics:
    """Run testing on the environment's test tasks using the cogman approach,
    measuring both solve and execution metrics, and recording successes/failures.
    Returns a Metrics object populated with aggregated statistics.
    """
    test_tasks = env.get_test_tasks()
    if CFG.approach != "oracle":
        test_tasks = [task.replace_goal_with_alt_goal() for task in test_tasks]

    # Initialize counters and per-run metrics
    cogman.reset_metrics()
    save_prefix = utils.get_config_path_str()
    metrics: Metrics = defaultdict(float)

    num_found_policy = 0
    num_solved = 0
    total_suc_time = 0.0
    total_low_level_action_cost = 0.0

    # Summaries for approach/execution failures
    total_num_solve_timeouts = 0
    total_num_solve_failures = 0
    total_num_execution_timeouts = 0
    total_num_execution_failures = 0

    # Track the running totals for nodes created/expanded
    curr_num_nodes_created = 0.0
    curr_num_nodes_expanded = 0.0

    # --------------------------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------------------------
    def _save_video(monitor: Optional[utils.VideoMonitor],
                              is_failure: bool,
                              task_idx: int) -> None:
        """Save a video from the monitor if the current config calls for it."""
        if monitor is None:
            return
        video = monitor.get_video()
        if CFG.use_counterfactual_dataset_path_name:
            suffix = ""
        else:
            suffix = "_failure" if is_failure else ""
        outfile = f"{save_prefix}__task{task_idx+1}{suffix}.mp4"
        utils.save_video(outfile, video)
    
    def _save_images(monitor: Optional[utils.VideoMonitor],
                              is_failure: bool,
                              task_idx: int) -> None:
        """Save images from the monitor if the current config calls for it."""
        if monitor is None:
            return
        video = monitor.get_video()
        if CFG.use_counterfactual_dataset_path_name:
            experiment_id = CFG.experiment_id.split("-")[0]
            outfile = f"{experiment_id}/seed{CFG.seed}/query/task{task_idx+1}/"
        else:
            suffix = "_failure" if is_failure else ""
            outfile = f"{save_prefix}__task{task_idx+1}{suffix}"
        utils.save_images(outfile, video)

    def _handle_solve_exception(
        e: Union[ApproachTimeout, ApproachFailure],
        task_idx: int,
        partial_refinements: Any,
    ) -> Tuple[int, int]:
        """Handle approach exceptions during the solve step, returning
        (updated_num_solve_timeouts, updated_num_solve_failures)."""
        nonlocal total_num_solve_timeouts, total_num_solve_failures
        if isinstance(e, ApproachTimeout):
            total_num_solve_timeouts += 1
        else:
            total_num_solve_failures += 1

        # Optionally save partial-refinement-based video
        if (CFG.make_failure_videos or CFG.make_failure_images) and\
              partial_refinements:
            logging.info(f"Creating video from partial refinements...")
            video = utils.create_video_from_partial_refinements(
                partial_refinements, env, "test", task_idx, CFG.horizon)
            if CFG.make_failure_images:
                experiment_id = CFG.experiment_id.split("-")[0]
                outfile = f"{experiment_id}/seed{CFG.seed}/query/"+\
                            f"task{task_idx+1}/"
                utils.save_images(outfile, video)                
            if CFG.make_failure_videos:
                outfile = f"{save_prefix}__task{task_idx+1}_failure.mp4"
                utils.save_video(outfile, video)

        if CFG.crash_on_failure:
            raise e
        return total_num_solve_timeouts, total_num_solve_failures

    def _solve_task(task_idx: int, env_task: Task) -> float:
        """Try to solve the given env_task using cogman, returning the solve time."""
        solve_start = time.perf_counter()
        cogman.reset(env_task)  # May raise ApproachTimeout or ApproachFailure
        return time.perf_counter() - solve_start

    def _execute_policy(
        task_idx: int,
        env_task: Task,
        monitor: Optional[utils.VideoMonitor] = None
    ) -> Tuple[bool, bool, float, int, float]:
        """Execute the cogman policy in the environment to see if the goal is solved.
        Returns:
            (solved, caught_exception, exec_time, num_options_executed, low_level_action_cost)
        """
        solved = False
        caught_exception = False
        exec_time = 0.0
        num_options_executed = 0

        try:
            traj, solved, execution_metrics = run_episode_and_get_observations(
                cogman,
                env,
                "test",
                task_idx,
                max_num_steps=CFG.horizon,
                monitor=monitor,
                terminate_on_goal_reached=CFG.terminate_on_goal_reached)
            exec_time = execution_metrics["policy_call_time"]
            num_options_executed = execution_metrics["num_options_executed"]

            # Optionally save a successful trajectory
            if CFG.save_eval_trajs:
                traj_file = f"{save_prefix}__task{task_idx+1}.traj"
                traj_file_path = Path(CFG.eval_trajectories_dir) / traj_file
                traj_data = {
                    "task": env_task,
                    "trajectory": traj,
                    "pybullet_robot": CFG.pybullet_robot
                }
                with open(traj_file_path, "wb") as f:
                    pkl.dump(traj_data, f)
        except utils.EnvironmentFailure as e:
            logging.info(f"Environment failed with error: {e}")
            caught_exception = True
        except (ApproachTimeout, ApproachFailure) as e:
            logging.info(f"Approach failed at execution time with error: {e}")
            if isinstance(e, ApproachTimeout):
                nonlocal total_num_execution_timeouts
                total_num_execution_timeouts += 1
            else:
                nonlocal total_num_execution_failures
                total_num_execution_failures += 1
            caught_exception = True
        
        # if traj is defined
        if 'traj' not in locals():
            traj = ([],[])

        return solved, caught_exception, exec_time, num_options_executed, traj

    # --------------------------------------------------------------------------
    # Main testing loop
    # --------------------------------------------------------------------------
    for test_task_idx, env_task in enumerate(test_tasks):
        # ---------------------
        # 1) Solve phase
        # ---------------------
        try:
            logging.info(f"[main.py] Solving task {test_task_idx+1}/"
                         f"{len(test_tasks)}...")
            solve_time = _solve_task(test_task_idx, env_task)
        except (ApproachTimeout, ApproachFailure) as e:
            # Handle solve failure/timeouts
            partial_refinements = getattr(e, "info", {}
                                          ).get("partial_refinements")
            logging.info(f"[main.py] Task {test_task_idx+1} / "
                         f"{len(test_tasks)}: approach failed with error: {e}")
            _handle_solve_exception(e, test_task_idx, partial_refinements)
            continue

        # Update solve-time metrics
        metrics[f"PER_TASK_task{test_task_idx}_solve_time"] = solve_time
        created = cogman.metrics["total_num_nodes_created"]
        expanded = cogman.metrics["total_num_nodes_expanded"]
        metrics[f"PER_TASK_task{test_task_idx}_nodes_created"] = created - curr_num_nodes_created
        metrics[f"PER_TASK_task{test_task_idx}_nodes_expanded"] = expanded - curr_num_nodes_expanded
        curr_num_nodes_created, curr_num_nodes_expanded = created, expanded

        num_found_policy += 1

        # ---------------------
        # 2) Execution phase
        # ---------------------
        # Decide if we need to record video
        need_video = (CFG.make_test_videos or CFG.make_failure_videos or
                      CFG.make_test_images or CFG.make_failure_images)
        monitor = utils.VideoMonitor(env.render) if need_video else None

        logging.info(f"Executing policy...")
        solved, caught_exception, exec_time, num_opts, traj = _execute_policy(
            test_task_idx, env_task, monitor)

        # Record execution metrics
        metrics[f"PER_TASK_task{test_task_idx}_exec_time"] = exec_time
        metrics[f"PER_TASK_task{test_task_idx}_options_executed"] = num_opts

        # Add cost for low-level actions if configured
        if CFG.refinement_data_include_execution_cost:
            total_low_level_action_cost += (len(traj[1]) *
                                CFG.refinement_data_low_level_execution_cost)

        # ---------------------
        # 3) Post-execution handling
        # ---------------------
        if solved and not caught_exception:
            # The plan reached the goal
            log_msg = "SOLVED"
            num_solved += 1
            total_suc_time += (solve_time + exec_time)
            # If solved, we may want to save a video if make_test_videos is True
            if CFG.make_test_videos:
                _save_video(monitor, is_failure=False, 
                                      task_idx=test_task_idx)
            if CFG.make_test_images:
                _save_images(monitor, is_failure=False,
                                        task_idx=test_task_idx)
            # Count how many steps we took
            # (We rely on the last trajectory from run_episode_and_get_observations)
            # If you need the real trajectory, you'd store it as in `_execute_policy`.
            # Suppose we do that here (execution_metrics / logging):
            metrics[f"PER_TASK_task{test_task_idx}_num_steps"] = len(traj[1])
        else:
            # The plan did not reach the goal, or an exception occurred
            if not caught_exception:
                log_msg = "Policy failed to reach goal"
            else:
                log_msg = "Policy/Env encountered an exception"
            if CFG.crash_on_failure:
                raise RuntimeError(log_msg)
            if CFG.make_failure_videos:
                _save_video(monitor, is_failure=True, 
                                      task_idx=test_task_idx)
            if CFG.make_failure_images:
                _save_images(monitor, is_failure=True,
                                        task_idx=test_task_idx)

            

        logging.info(f"Task {test_task_idx+1} / {len(test_tasks)}: {log_msg}")

    # --------------------------------------------------------------------------
    # Aggregate final metrics
    # --------------------------------------------------------------------------
    metrics["num_solved"] = num_solved
    metrics["num_total"] = len(test_tasks)
    metrics["avg_suc_time"] = (
        total_suc_time / num_solved if num_solved > 0 else float("inf"))
    metrics["avg_ref_cost"] = (
        (total_low_level_action_cost + cogman.metrics["total_refinement_time"])
        / num_solved if num_solved > 0 else float("inf"))

    # Skeleton / sample info
    metrics["min_num_samples"] = (
        cogman.metrics["min_num_samples"]
        if cogman.metrics["min_num_samples"] < float("inf") else 0)
    metrics["max_num_samples"] = cogman.metrics["max_num_samples"]
    metrics["min_skeletons_optimized"] = (
        cogman.metrics["min_num_skeletons_optimized"]
        if cogman.metrics["min_num_skeletons_optimized"] < float("inf") else 0)
    metrics["max_skeletons_optimized"] = cogman.metrics["max_num_skeletons_optimized"]

    # Failure/timeouts
    metrics["num_solve_timeouts"] = total_num_solve_timeouts
    metrics["num_solve_failures"] = total_num_solve_failures
    metrics["num_execution_timeouts"] = total_num_execution_timeouts
    metrics["num_execution_failures"] = total_num_execution_failures

    # Compute averages of certain CogMan metrics wrt # of found policies
    for metric_name in [
        "num_samples", "num_skeletons_optimized", "num_nodes_expanded",
        "num_nodes_created", "num_nsrts", "num_preds", "plan_length",
        "num_failures_discovered"
    ]:
        total = cogman.metrics[f"total_{metric_name}"]
        metrics[f"avg_{metric_name}"] = (
            total / num_found_policy if num_found_policy > 0 else float("inf"))

    return metrics

def _save_test_results(results: Metrics,
                       online_learning_cycle: Optional[int]) -> None:
    num_solved = results["num_solved"]
    num_total = results["num_total"]
    avg_suc_time = results["avg_suc_time"]
    logging.info(f"Tasks solved: {num_solved} / {num_total}")
    logging.info(f"Average time for successes: {avg_suc_time:.5f} seconds")
    outfile = (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
               f"{online_learning_cycle}.pkl")
    # Save CFG alongside results.
    outdata = {
        "config": CFG,
        "results": results.copy(),
        "git_commit_hash": utils.get_git_commit_hash()
    }
    # Dump the CFG, results, and git commit hash to a pickle file.
    with open(outfile, "wb") as f:
        pkl.dump(outdata, f)
    # Before printing the results, filter out keys that start with the
    # special prefix "PER_TASK_", to prevent an annoyingly long printout.
    del_keys = [k for k in results if k.startswith("PER_TASK_")]
    for k in del_keys:
        del results[k]
    logging.info(f"Test results: {results}")
    logging.info(f"Wrote out test results to {outfile}")


if __name__ == "__main__":  # pragma: no cover
    # Write out the exception to the log file.
    try:
        main()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("main.py crashed")
        raise _err
