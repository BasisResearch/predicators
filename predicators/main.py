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

from predicators import utils
from predicators.cogman import CogMan
from predicators.execution_monitoring import create_execution_monitor
from predicators.perception import create_perceiver
from predicators.run.checkpoints import maybe_auto_resume
from predicators.run.online_learning import run_pipeline
from predicators.run.setup import create_offline_dataset, setup_approach, \
    setup_environment
from predicators.settings import CFG

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"


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

    # Setup environment, tasks, and approach
    setup = setup_environment()
    env, preds = setup.env, setup.preds
    approach = setup_approach(env, preds, setup.approach_train_tasks)

    # Self-resume (Slurm requeue / resubmission of the same command);
    # needs the approach for its checkpoint suffix.
    maybe_auto_resume(approach)

    # Create dataset and cognitive manager
    offline_dataset = create_offline_dataset(env, setup.train_tasks, preds,
                                             approach)
    execution_monitor = create_execution_monitor(CFG.execution_monitor)
    cogman = CogMan(approach, create_perceiver(CFG.perceiver),
                    execution_monitor)

    # Run pipeline
    run_pipeline(env, cogman, setup.approach_train_tasks, offline_dataset)

    # Log completion
    script_time = time.perf_counter() - script_start
    logging.info(f"\n\nMain script terminated in {script_time:.5f} seconds")


if __name__ == "__main__":  # pragma: no cover
    # Write out the exception to the log file.
    try:
        main()
    except Exception as _err:  # pylint: disable=broad-except
        logging.exception("main.py crashed")
        raise _err
