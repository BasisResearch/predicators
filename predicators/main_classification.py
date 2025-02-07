import os
import time

import logging

from predicators import utils
from predicators.settings import CFG
from predicators.main import setup_environment, setup_approach, \
    create_offline_dataset, create_perceiver
from predicators.cogman import CogMan
from predicators.envs import BaseEnv

def main() -> None:
    """Main entry point for running classification approaches.
    """
    script_start = time.perf_counter()

    # Parse args
    args = utils.parse_args()
    utils.update_config(args)
    str_args = utils.get_str_args(args)

    # Set up logging
    utils.configure_logging()
    os.makedirs(CFG.results_dir, exist_ok=True)
    os.makedirs(CFG.eval_trajectories_dir, exist_ok=True)

    # Log initial info
    utils.log_initial_info(str_args)

    # Setup environment
    env, approach_train_tasks, train_tasks = setup_environment()

    # Setup predicates
    included_preds, excluded_preds = utils.parse_config_excluded_predicates(env)
    preds = utils.replace_goals_with_agent_specific_goals(
        included_preds, excluded_preds, env
        ) if CFG.approach != "oracle" else included_preds
    
    # Create approach
    approach = setup_approach(env, preds, approach_train_tasks)

    """
    --- Create dataset
    In a meta learning setting, we have meta-train and meta-test datasets but we
    only have meta-test now.
    Each dataset contains multiple tasks. Each task contains a support and query
    set.
    For now, there are 1-2 support videos and 2 query videos per task.
    ---
    Alternatively, with the current design, there is just 1 kind of 
    counterfactual per env. So we only have 1 task in the meta-test split. 
    In each task, we will have 1 or more training samples and multiple test
    samples.
    Each sample will have a (state, action) traj and a label for whether it's
    from the standard world.
    """
    train_dataset, test_dataset = create_datasets(env, train_tasks, preds)
    execution_monitor = ...
    cogman = CogMan(approach, create_perceiver(CFG.perceiver),
                    execution_monitor)

    _run_pipeline(env, cogman, train_dataset, test_dataset)

    # Log completion
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script completed in {script_time:.2f} seconds.")

def create_datasets(env: BaseEnv, tasks: List[Task], preds: List[str], 
                    approach: Approach) -> Tuple[Dataset, Dataset]:
    """how it's currently done"""
    option = get_gt_option(env.get_name()) if CFG.option_learner == \
                "no_learning" else parse_config_included_options(env)
    return create_dataset(env, train_tasks, options, preds)


def _run_pipeline(env: BaseEnv, cogman: CogMan,) -> None:
    ...



if __name__ == "__main__": # pragma: no cover
    try:
        main()
    except Exception as _err: # pylint: disable=broad-except
        logging.exception("main_classification.py crashed")
        raise _err