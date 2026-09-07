"""Tests for the Engaging batch-script builder.

The self-requeue trap is what lets a requeue-enabled job heal a wall-
time TIMEOUT by itself (requeue + --auto_resume), instead of relying on
an external watcher process to notice the TIMEOUT and resubmit.
"""

import subprocess
from typing import Tuple

from scripts.engaging.submit_engaging_job import _build_batch_script


def _bash_syntax_ok(script: str, tmp_path) -> Tuple[bool, str]:
    path = tmp_path / "job.sh"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(["bash", "-n", str(path)],
                            capture_output=True,
                            check=False)
    return result.returncode == 0, result.stderr.decode("utf-8")


def test_plain_script_without_requeue(tmp_path) -> None:
    """Without requeue the script is the simple foreground command."""
    script = _build_batch_script("main.py",
                                 "--env pybullet_bridge",
                                 requeue=False)
    assert script.rstrip().endswith("--seed $SLURM_ARRAY_TASK_ID")
    assert "scontrol requeue" not in script
    assert "trap" not in script
    ok, err = _bash_syntax_ok(script, tmp_path)
    assert ok, err


def test_requeue_script_installs_self_requeue_trap(tmp_path) -> None:
    """With requeue the script traps USR1, requeues its own array task, caps
    restarts via SLURM_RESTART_COUNT, and backgrounds python so the trap can
    fire while it runs."""
    script = _build_batch_script("main.py",
                                 "--env pybullet_bridge",
                                 requeue=True)
    assert "trap _requeue_on_timeout USR1" in script
    assert "scontrol requeue" in script
    assert "SLURM_RESTART_COUNT" in script
    # python is backgrounded and reaped, so the trap can interrupt wait.
    assert "--seed $SLURM_ARRAY_TASK_ID &" in script
    assert script.count('wait "$_PY_PID"') == 2
    # Array tasks requeue by <array_job>_<task>; plain jobs by job id.
    assert "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" in script
    ok, err = _bash_syntax_ok(script, tmp_path)
    assert ok, err
