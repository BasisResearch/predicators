"""Script for submitting jobs on the MIT ORCD Engaging cluster."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

START_SEED = 456
NUM_SEEDS = 10

# parents[0] = scripts/engaging, parents[1] = scripts, parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The conda env this repo is set up in on Engaging (see repo setup docs).
_CONDA_ENV = "pred"
_MINIFORGE_MODULE = "miniforge/25.11.0-0"

# Default partitions. mit_normal and mit_normal_gpu are the dedicated MIT
# partitions; they are frequently saturated, in which case mit_preemptable
# (much larger, and carries both CPU and GPU nodes) usually starts far sooner
# at the cost of being evictable. See `sinfo` for current partition state.
_CPU_PARTITION = "mit_normal"
_GPU_PARTITION = "mit_normal_gpu"
_CPU_TIME = "12:00:00"
_GPU_TIME = "06:00:00"

# Slurm max wall time per partition, from `sinfo`. Requested time limits are
# clamped to these, since sbatch rejects a job that asks for more.
_PARTITION_TIME_LIMITS = {
    "mit_normal": "12:00:00",
    "mit_normal_gpu": "06:00:00",
    "mit_quicktest": "00:15:00",
    "mit_preemptable": "2-00:00:00",
    "mit_data_transfer": "3-00:00:00",
}

# Partitions whose jobs can be evicted by higher-priority work. Jobs here get
# --requeue by default, so preemption reschedules the job instead of killing
# it. Note that a requeued job restarts from the beginning rather than
# resuming, so this only pays off for runs that checkpoint or are cheap to
# redo.
_PREEMPTABLE_PARTITIONS = {"mit_preemptable"}

# Requeue-enabled jobs also heal wall-time expiry without an external
# watcher: sbatch's --signal delivers SIGUSR1 to the batch shell this many
# seconds before the limit, and the generated script's trap requeues the
# still-running task (see _SELF_REQUEUE_BLOCK). Slurm only guarantees
# delivery within about a minute of the requested lead.
_TIMEOUT_SIGNAL_LEAD_SECS = 600
# A run that keeps exhausting its wall time requeues itself at most this
# many times. Preemption requeues share SLURM_RESTART_COUNT, so a heavily
# preempted run spends the same budget - generous on purpose.
_MAX_SELF_REQUEUES = 8

# Bash appended to the batch script when requeue is enabled. The trap makes
# a TIMEOUT behave exactly like a preemption: the task requeues itself
# (same job id, same command) and --auto_resume restores the latest
# checkpoint on restart. Without this, --requeue only covers preemption and
# a TIMEOUT silently ends the run.
_SELF_REQUEUE_BLOCK = """\
# Self-requeue on imminent wall-time expiry (--signal sends USR1 before
# the limit). If python is still running then, requeue this task; on
# restart --auto_resume restores the latest per-cycle checkpoint - the
# same healing path a preemption requeue already uses.
_MAX_RESTARTS={max_restarts}
_PY_PID=""
_requeue_on_timeout() {{
  if [ -z "$_PY_PID" ] || ! kill -0 "$_PY_PID" 2>/dev/null; then
    return  # python already finished; exit normally
  fi
  if [ "${{SLURM_RESTART_COUNT:-0}}" -ge "$_MAX_RESTARTS" ]; then
    echo "[self-requeue] restart cap $_MAX_RESTARTS reached; not requeueing"
    return
  fi
  if [ -n "$SLURM_ARRAY_JOB_ID" ]; then
    _TASK_ID="${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
  else
    _TASK_ID="$SLURM_JOB_ID"
  fi
  echo "[self-requeue] wall limit imminent (restart" \\
       "${{SLURM_RESTART_COUNT:-0}}); requeueing $_TASK_ID"
  scontrol requeue "$_TASK_ID"
}}
trap _requeue_on_timeout USR1
{run_cmd} &
_PY_PID=$!
# A trapped signal interrupts the first wait; wait again to reap python
# (after a requeue Slurm kills this run shortly, so the second wait just
# collects the status; with the cap reached it lets python use the rest
# of the wall time).
wait "$_PY_PID"
_RC=$?
if [ "$_RC" -gt 128 ]; then
  wait "$_PY_PID"
  _RC=$?
fi
exit "$_RC"
"""


def _build_batch_script(entry_point: str, args_and_flags_str: str,
                        requeue: bool) -> str:
    """Compose the batch script sbatch runs for every array task.

    With ``requeue``, the script backgrounds python and installs the
    self-requeue trap (see ``_SELF_REQUEUE_BLOCK``); without it, the
    python command simply runs in the foreground.
    """
    header = [
        "#!/bin/bash -l",  # -l => login shell, so /etc/profile.d (module) loads
        # main.py asserts on this. sbatch propagates the submitting shell's
        # environment, so inheriting it works only if whoever submits happens
        # to have it exported; set it here so the job is self-contained. This
        # matches what scripts/local/launch.py writes into its run scripts.
        "export PYTHONHASHSEED=0",
        f"module load {_MINIFORGE_MODULE}",
        f"conda activate {_CONDA_ENV}",
        f"cd {_REPO_ROOT}",
    ]
    run_cmd = (f"python predicators/{entry_point} "
               f"{args_and_flags_str} --seed $SLURM_ARRAY_TASK_ID")
    if not requeue:
        return "\n".join(header + [run_cmd])
    block = _SELF_REQUEUE_BLOCK.format(max_restarts=_MAX_SELF_REQUEUES,
                                       run_cmd=run_cmd)
    return "\n".join(header + ["", block])


# Overrides for callers that do not go through launch.py (e.g. running this
# module directly), where predicators' own arg parser owns the command line.
_PARTITION_ENV_VAR = "PREDICATORS_ENGAGING_PARTITION"
_REQUEUE_ENV_VAR = "PREDICATORS_ENGAGING_REQUEUE"


def _run() -> None:
    # Imported here rather than at module level so that importers of
    # submit_engaging_job() (e.g. launch.py) do not pay for the full
    # predicators import chain, which pulls in gym, pybullet and genai.
    # pylint: disable=import-outside-toplevel
    from predicators import utils
    from predicators.settings import CFG

    args = utils.parse_args(seed_required=False)
    utils.update_config(args)
    assert CFG.seed is None, "Do not pass in a seed to this script!"
    job_name = CFG.experiment_id
    log_dir = CFG.log_dir
    logfile_prefix = utils.get_config_path_str()
    args_and_flags_str = " ".join(sys.argv[1:])
    submit_engaging_job("main.py", job_name, log_dir, logfile_prefix,
                        args_and_flags_str, START_SEED, NUM_SEEDS)


def _duration_to_seconds(duration: str) -> int:
    """Parse a Slurm time string (e.g. "6:00:00", "2-00:00:00") to seconds."""
    days = 0
    if "-" in duration:
        day_str, _, duration = duration.partition("-")
        days = int(day_str)
    parts = [int(p) for p in duration.split(":")]
    # Slurm reads a bare number as minutes, and a colon-separated string as
    # right-aligned to seconds -- except after a day field, where it is
    # left-aligned to hours.
    if len(parts) == 1:
        hours, minutes, seconds = (parts[0], 0, 0) if days else (0, parts[0],
                                                                 0)
    elif len(parts) == 2:
        hours, minutes, seconds = ((parts[0], parts[1], 0) if days else
                                   (0, parts[0], parts[1]))
    else:
        hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _seconds_to_duration(seconds: int) -> str:
    """Format seconds as a Slurm time string."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    stamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{stamp}" if days else stamp


def _resolve_partition(partition: Optional[str], use_gpu: bool) -> str:
    """Pick the partition: explicit argument, then env var, then default."""
    if partition:
        return partition
    from_env = os.environ.get(_PARTITION_ENV_VAR, "").strip()
    if from_env:
        return from_env
    return _GPU_PARTITION if use_gpu else _CPU_PARTITION


def _resolve_requeue(requeue: Optional[bool], partition: str) -> bool:
    """Decide --requeue: explicit argument, then env var, then whether the
    partition is preemptable."""
    if requeue is not None:
        return requeue
    from_env = os.environ.get(_REQUEUE_ENV_VAR, "").strip().lower()
    if from_env:
        return from_env in ("1", "true", "yes", "on")
    return partition in _PREEMPTABLE_PARTITIONS


def _clamp_time_limit(time_limit: str, partition: str) -> str:
    """Shrink the requested time limit to the partition max, if known."""
    partition_max = _PARTITION_TIME_LIMITS.get(partition)
    if partition_max is None:
        # An unrecognized partition (new, or a cluster change); let sbatch
        # be the judge rather than guessing a limit.
        return time_limit
    requested_secs = _duration_to_seconds(time_limit)
    max_secs = _duration_to_seconds(partition_max)
    if requested_secs <= max_secs:
        return time_limit
    clamped = _seconds_to_duration(max_secs)
    print(f"Warning: {partition} caps wall time at {partition_max}; "
          f"reducing requested {time_limit} to {clamped}.")
    return clamped


def submit_engaging_job(entry_point: str,
                        job_name: str,
                        log_dir: str,
                        logfile_prefix: str,
                        args_and_flags_str: str,
                        start_seed: int,
                        num_seeds: int,
                        use_gpu: bool = False,
                        use_mujoco: bool = False,
                        partition: Optional[str] = None,
                        requeue: Optional[bool] = None) -> None:
    """Launch one Slurm array job (one array task per seed) on Engaging.

    partition defaults to mit_normal (or mit_normal_gpu for GPU jobs),
    and requeue defaults to True on preemptable partitions. Both can
    also be set via the PREDICATORS_ENGAGING_PARTITION and
    PREDICATORS_ENGAGING_REQUEUE environment variables.
    """
    del use_mujoco  # unused
    os.makedirs(log_dir, exist_ok=True)
    logfile_pattern = os.path.join(log_dir, f"{logfile_prefix}__%j.log")
    assert logfile_pattern.count("None") == 1
    logfile_pattern = logfile_pattern.replace("None", "%a")

    partition = _resolve_partition(partition, use_gpu)
    requeue = _resolve_requeue(requeue, partition)
    time_limit = _clamp_time_limit(_GPU_TIME if use_gpu else _CPU_TIME,
                                   partition)

    mystr = _build_batch_script(entry_point, args_and_flags_str, requeue)
    # A unique name per submission, so that a leftover file from a crashed
    # launch cannot block later ones and so concurrent launches cannot race.
    fd, temp_run_file = tempfile.mkstemp(prefix="temp_run_file_",
                                         suffix=".sh",
                                         dir=".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(mystr)

        cmd = f"sbatch --time={time_limit} --partition={partition} "
        if use_gpu:
            cmd += "--gres=gpu:1 "
        if requeue:
            # --signal feeds the script's self-requeue trap (TIMEOUT heals
            # like a preemption); --open-mode=append keeps a requeued
            # task's log instead of truncating it on restart.
            cmd += ("--requeue "
                    f"--signal=B:USR1@{_TIMEOUT_SIGNAL_LEAD_SECS} "
                    "--open-mode=append ")
        # 8 CPUs so agent_validation_parallel_workers has headroom:
        # validation/margin rollouts fork up to that many children
        # (benchmark 21336169: 3.74x at 4 workers, 4.81x at 8).
        cmd += ("--nodes=1 "
                "--cpus-per-task=8 "
                "--mem=16G "
                f"--job-name={job_name} "
                f"--array={start_seed}-{start_seed + num_seeds - 1} "
                f"-o {logfile_pattern} {temp_run_file}")
        print(f"Running command: {cmd}")
        output = subprocess.getoutput(cmd)
        print(output)
        if "command not found" in output:
            raise Exception("Are you logged into the Engaging cluster?")
    finally:
        os.remove(temp_run_file)


if __name__ == "__main__":
    _run()
