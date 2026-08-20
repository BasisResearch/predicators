"""Launch Engaging (ORCD) experiments defined by config files, adapted from
openmind/launch.py.

Each experiment (approach x env combo) is submitted as its own Slurm array
job, with one array task per seed, so all experiments run concurrently on
compute nodes rather than in the current terminal/login node.

Usage example:

    python scripts/engaging/launch.py -c predicatorv3/exp_domino.yaml

mit_normal is often saturated. To run on the much larger (but evictable)
preemptable partition instead:

    python scripts/engaging/launch.py -c predicatorv3/exp_domino.yaml \
        --partition mit_preemptable
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path so `scripts` is importable without PYTHONPATH=.
# parents[0] = scripts/engaging, parents[1] = scripts, parents[2] = repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from scripts.cluster_utils import BatchSeedRunConfig, config_to_cmd_flags, \
    config_to_logfile, generate_run_configs
from scripts.engaging.submit_engaging_job import submit_engaging_job


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        default=None,
        help="Slurm partition to submit to, e.g. mit_preemptable. Defaults "
        "to mit_normal, or mit_normal_gpu for GPU experiments.")
    requeue_group = parser.add_mutually_exclusive_group()
    requeue_group.add_argument(
        "--requeue",
        dest="requeue",
        action="store_true",
        help="Requeue jobs when they are preempted, instead of killing them. "
        "On by default for preemptable partitions. Note that a requeued job "
        "restarts from the beginning rather than resuming.")
    requeue_group.add_argument("--no-requeue",
                               dest="requeue",
                               action="store_false",
                               help="Never requeue jobs on preemption.")
    parser.set_defaults(requeue=None)
    args = parser.parse_args()
    _launch_experiments(args.config, args.partition, args.requeue)


def _launch_experiments(config_file: str,
                        partition: Optional[str] = None,
                        requeue: Optional[bool] = None) -> None:
    # Loop over run configs.
    for cfg in generate_run_configs(config_file, batch_seeds=True):
        assert isinstance(cfg, BatchSeedRunConfig)
        cmd_flags = config_to_cmd_flags(cfg)
        log_dir = "logs"
        log_prefix = config_to_logfile(cfg, suffix="")
        # Launch a job for this experiment.

        if "use_classification_problem_setting" in cfg.flags:
            use_classification_problem_setting = cfg.flags[
                'use_classification_problem_setting']
        else:
            use_classification_problem_setting = False

        if use_classification_problem_setting:
            entry_point = "main_classification.py"
        else:
            entry_point = "main.py"
        submit_engaging_job(entry_point, cfg.experiment_id, log_dir,
                            log_prefix, cmd_flags, cfg.start_seed,
                            cfg.num_seeds, cfg.use_gpu, cfg.use_mujoco,
                            partition, requeue)


if __name__ == "__main__":
    _main()
