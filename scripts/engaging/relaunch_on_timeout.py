"""Watch Slurm jobs and relaunch each one's experiment once on TIMEOUT.

Preemption self-heals via ``sbatch --requeue``, but a job that hits its
wall-clock limit is simply killed. For auto_resume experiments the fix is
to resubmit the identical command so the run continues from its latest
checkpoint. This watcher does that, once per watched job, and exits when
every watched job has reached a terminal state.

Each watched job maps to ONE experiment (approach x env arm) inside its
launch config, and only that experiment is resubmitted. Relaunching the
whole config would also resubmit sibling arms that may still be running
(jobs in one generation drift apart through preemption requeues), and a
duplicate arm races the live one on the same auto_resume checkpoints.

Usage (detach with nohup; poll every 10 min):

    nohup python scripts/engaging/relaunch_on_timeout.py \
        -p mit_preemptable \
        21338737:predicatorv3/exp_bridge_v2.yaml:bridge_v2-agent_pi_al \
        21338740:predicatorv3/exp_bridge_v2.yaml:bridge_v2-agent_pi_al_pol \
        >> logs/auto_relaunch.log 2>&1 &
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path so `scripts` is importable without PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from scripts.cluster_utils import BatchSeedRunConfig, config_to_cmd_flags, \
    config_to_logfile, generate_run_configs
from scripts.engaging.submit_engaging_job import submit_engaging_job


@dataclass
class _WatchedJob:
    job_id: str
    config_file: str
    experiment_id: str
    resolved: bool = False


def _parse_spec(spec: str) -> _WatchedJob:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected JOBID:CONFIG:EXPERIMENT_ID, got {spec!r}")
    return _WatchedJob(job_id=parts[0],
                       config_file=parts[1],
                       experiment_id=parts[2])


def _in_queue(job_id: str) -> bool:
    out = subprocess.run(["squeue", "-h", "-j", job_id],
                         capture_output=True,
                         text=True,
                         check=False)
    return bool(out.stdout.strip())


def _terminal_state(job_id: str) -> Optional[str]:
    """The job's sacct state, or None if sacct has nothing yet."""
    out = subprocess.run(["sacct", "-j", job_id, "-X", "-n", "-o", "State%30"],
                         capture_output=True,
                         text=True,
                         check=False)
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else None


def _relaunch_experiment(config_file: str, experiment_id: str,
                         partition: str) -> None:
    """Resubmit only ``experiment_id`` from ``config_file``."""
    for cfg in generate_run_configs(config_file, batch_seeds=True):
        assert isinstance(cfg, BatchSeedRunConfig)
        if cfg.experiment_id != experiment_id:
            continue
        cmd_flags = config_to_cmd_flags(cfg)
        log_prefix = config_to_logfile(cfg, suffix="")
        submit_engaging_job("main.py", cfg.experiment_id, "logs", log_prefix,
                            cmd_flags, cfg.start_seed, cfg.num_seeds,
                            cfg.use_gpu, cfg.use_mujoco, partition, True)
        return
    print(
        f"WARNING: experiment {experiment_id} not found in {config_file}; "
        "nothing relaunched",
        flush=True)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs",
                        nargs="+",
                        help="JOBID:CONFIG:EXPERIMENT_ID triples")
    parser.add_argument("-p", "--partition", default="mit_preemptable")
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args()
    jobs: List[_WatchedJob] = [_parse_spec(s) for s in args.specs]
    stamp = time.strftime("%m-%d %H:%M")
    print(
        f"{stamp} watching {len(jobs)} jobs: "
        f"{', '.join(j.job_id for j in jobs)}",
        flush=True)
    while not all(j.resolved for j in jobs):
        time.sleep(args.poll_seconds)
        for job in jobs:
            if job.resolved or _in_queue(job.job_id):
                continue
            state = _terminal_state(job.job_id)
            if state is None:
                continue
            stamp = time.strftime("%m-%d %H:%M")
            if state.startswith("TIMEOUT"):
                print(
                    f"{stamp} job {job.job_id} TIMEOUT -> relaunching "
                    f"{job.experiment_id} from {job.config_file}",
                    flush=True)
                _relaunch_experiment(job.config_file, job.experiment_id,
                                     args.partition)
            else:
                print(
                    f"{stamp} job {job.job_id} ended {state}; "
                    "no relaunch",
                    flush=True)
            job.resolved = True
    print(
        f"{time.strftime('%m-%d %H:%M')} all watched jobs resolved; "
        "exiting",
        flush=True)


if __name__ == "__main__":
    _main()
