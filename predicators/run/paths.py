"""Where a continual-protocol run keeps its files: one directory per run.

A run is one directory,
``<continual_runs_dir>/<approach>/<experiment_id>/seed<k>/run_<stamp>/``,
the launch's log directory (``utils.configure_logging`` names it from
the same ``CFG.run_subdir``), holding:

* ``info.log`` and ``debug.log``, the launch's logs; a launch that
  resumes the run appends to them;
* ``scorecard.json``, the ``RunCard`` (``predicators/run/scorecard.py``),
  rewritten after every skill invocation, reset and level event;
* ``L<k>/``, one recording per level (``predicators/run/recording.py``);
* ``agent/``, the agent arm's stable directory: the system prompt, one
  transcript per session, and its sandbox;
* ``run.mp4``, the labelled replay video written at run end under
  ``continual_make_video`` or by ``scripts/continual_video.py``.

A launch is a new run directory unless it resumes (``--auto_resume``,
section 6.6 of docs/protocol/design.md): then it adopts the newest
run directory of the same approach, experiment id and seed whose
scorecard is unfinished. Nothing is keyed on the run id alone, so
relaunching an experiment never writes into an earlier run's files, and
old runs stay where they are for the viewer.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

from predicators import utils
from predicators.run.scorecard import RunCard
from predicators.settings import CFG

SCORECARD_FILENAME = "scorecard.json"
VIDEO_FILENAME = "run.mp4"
AGENT_DIRNAME = "agent"
RUN_DIR_RE = re.compile(r"^run_\d{8}_\d{6}$")


def level_dirname(level_index: int) -> str:
    """``L01`` for the level at index 0."""
    return f"L{level_index + 1:02d}"


def experiment_dir() -> str:
    """The directory holding every run of this approach, experiment id and
    seed."""
    return os.path.join(CFG.continual_runs_dir, CFG.approach,
                        CFG.experiment_id, f"seed{CFG.seed}")


def _newest_run() -> Optional[Tuple[str, Optional[RunCard]]]:
    """Under ``--auto_resume`` in the continual protocol, the newest run of
    this approach, experiment id and seed that has a scorecard, with that
    scorecard, or None for an unreadable one; None when there is no such run or
    no ``--auto_resume``."""
    if not getattr(CFG, "auto_resume", False) or \
            CFG.experiment_protocol != "continual":
        return None
    parent = experiment_dir()
    try:
        names = sorted(n for n in os.listdir(parent) if RUN_DIR_RE.match(n))
    except OSError:
        return None
    for name in reversed(names):
        card_path = os.path.join(parent, name, SCORECARD_FILENAME)
        if not os.path.isfile(card_path):
            continue
        try:
            return name, RunCard.load(card_path)
        except (OSError, ValueError, KeyError, TypeError):
            logging.warning("--auto_resume: %s is unreadable.", card_path)
            return name, None
    return None


def resumable_run_subdir() -> Optional[str]:
    """The ``run_subdir`` a resuming launch adopts, or ``None`` for a new run.

    Under ``--auto_resume`` in the continual protocol, the newest run of
    this approach, experiment id and seed that has a scorecard is the
    one to continue when that scorecard is unfinished. When it is over,
    or there is none, or it cannot be read, the launch starts a new run.
    """
    newest = _newest_run()
    if newest is None or newest[1] is None or newest[1].is_finished:
        return None
    return f"{CFG.approach}/{CFG.experiment_id}/seed{CFG.seed}/{newest[0]}/"


def finished_run_requeued() -> Optional[str]:
    """The newest run's directory when this process is a requeued Slurm job
    (``SLURM_RESTART_COUNT`` of 1 or more) under ``--auto_resume`` whose run
    has already finished, else None.

    Such a job was requeued after its run ended (a time limit's requeue
    signal or a preemption came while it wrapped up), so it has nothing
    to resume, and a new run would repeat the experiment under the same
    round. A first launch still starts a new run after a finished one.
    """
    try:
        restarts = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    except ValueError:
        return None
    if restarts < 1:
        return None
    newest = _newest_run()
    if newest is None or newest[1] is None or not newest[1].is_finished:
        return None
    return os.path.join(experiment_dir(), newest[0])


def ensure_run_subdir() -> str:
    """``CFG.run_subdir``, set on first use.

    ``configure_logging`` sets it for a launch with a log file; without
    one (tests, a bare process) the first caller does: the resumable
    run's under ``--auto_resume``, else a new one that no directory
    under ``continual_runs_dir`` holds yet.
    """
    if not CFG.run_subdir:
        CFG.run_subdir = resumable_run_subdir() or utils.new_run_subdir(
            CFG.continual_runs_dir)
    return CFG.run_subdir


def run_dir() -> str:
    """This run's directory, created."""
    path = os.path.normpath(
        os.path.join(CFG.continual_runs_dir, ensure_run_subdir()))
    os.makedirs(path, exist_ok=True)
    return path


def scorecard_path(run_directory: str) -> str:
    """The run's scorecard."""
    return os.path.join(run_directory, SCORECARD_FILENAME)


def video_path(run_directory: str) -> str:
    """The run's labelled replay video."""
    return os.path.join(run_directory, VIDEO_FILENAME)


def agent_dir(run_directory: str) -> str:
    """The agent arm's directory of a run."""
    return os.path.join(run_directory, AGENT_DIRNAME)


def level_dir(run_directory: str, level_index: int) -> str:
    """A level's recording directory."""
    return os.path.join(run_directory, level_dirname(level_index))
