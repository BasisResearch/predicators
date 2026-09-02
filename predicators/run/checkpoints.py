"""On-disk state a run leaves behind between online-learning cycles.

Three kinds of files live here, all keyed by the config path so a
Slurm requeue / resubmission of the identical command finds them:

* approach checkpoints ``{load_path}_{cycle}.{suffix}``, written by
  checkpointing approaches at the end of each LEARN;
* the in-flight interaction stash ``{load_path}_inflight_interactions_
  {cycle}.pkl``, written just BEFORE a cycle's learn so a run that dies
  mid-learn resumes at learn instead of re-exploring;
* per-cycle test results ``{results_dir}/{config}__{cycle}.pkl``.

``ApproachCheckpoints`` owns the first two (they share the approach's
load path and checkpoint suffix); the module-level ``test_results_*``
functions own the third. ``maybe_auto_resume`` ties them together for
``--auto_resume``.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import dill as pkl

from predicators import utils
from predicators.settings import CFG

# ── Approach checkpoints and the in-flight stash ─────────────────


@dataclass(frozen=True)
class InflightInteractions:
    """A cycle's completed interaction episodes, persisted before LEARN."""
    cycle: int
    interaction_results: List[Any]
    task_idxs: List[int]
    task_solved_status: List[bool]
    query_cost: float


class ApproachCheckpoints:
    """Locates one run's approach checkpoints and its in-flight stash.

    ``suffix`` is the approach's checkpoint filename suffix; ``None``
    means the approach does not checkpoint, in which case a resume never
    skips cycles and the stash is neither written nor read.
    """

    def __init__(self, load_path: str, suffix: Optional[str]) -> None:
        self._load_path = load_path
        self._suffix = suffix

    @classmethod
    def for_approach(cls, approach: Any) -> "ApproachCheckpoints":
        """Checkpoints of ``approach`` under the current config path."""
        # pylint: disable-next=protected-access
        suffix = getattr(approach, "_save_suffix", None)
        return cls(utils.get_approach_load_path_str(), suffix)

    @classmethod
    def for_cogman(cls, cogman: Any) -> "ApproachCheckpoints":
        """Checkpoints of the approach ``cogman`` wraps."""
        # pylint: disable-next=protected-access
        return cls.for_approach(cogman._approach)

    @property
    def load_path(self) -> str:
        """The checkpoint path prefix (no cycle, no suffix)."""
        return self._load_path

    @property
    def suffix(self) -> Optional[str]:
        """The approach's checkpoint suffix, or None if it never saves."""
        return self._suffix

    @property
    def checkpointing(self) -> bool:
        """Whether the approach writes checkpoints at all."""
        return self._suffix is not None

    def _paths(self, cycle: Optional[int]) -> List[str]:
        pattern = glob.escape(f"{self._load_path}_{cycle}.") + (glob.escape(
            self._suffix) if self._suffix else "*")
        return glob.glob(pattern)

    def exists(self, cycle: Optional[int]) -> bool:
        """Whether a checkpoint file exists for ``cycle``."""
        return bool(self._paths(cycle))

    def mtime(self, cycle: Optional[int]) -> Optional[float]:
        """Modification time of the newest checkpoint file for ``cycle``, or
        None when there is none."""
        paths = self._paths(cycle)
        if not paths:
            return None
        return max(os.path.getmtime(p) for p in paths)

    def discover_resume_cycles(
            self,
            max_age_seconds: Optional[float] = None,
            now: Optional[float] = None) -> Tuple[bool, Optional[int]]:
        """See the module-level ``discover_resume_cycles``."""
        return discover_resume_cycles(self._load_path,
                                      suffix=self._suffix,
                                      max_age_seconds=max_age_seconds,
                                      now=now)

    # In-flight interaction stash.

    def inflight_path(self, cycle: int) -> str:
        """Path of the mid-cycle interaction-episodes stash for ``cycle``.

        The cycle token in the name ("inflight_interactions_<i>") is
        deliberately non-integer, so ``discover_resume_cycles`` ignores
        these files when deciding which cycle to resume at.
        """
        return f"{self._load_path}_inflight_interactions_{cycle}.pkl"

    def save_inflight(self, stash: InflightInteractions) -> None:
        """Persist a cycle's completed interaction episodes before LEARN.

        The per-cycle checkpoint is only written at the END of learn, so
        dying during the (hours-long) learn used to discard the cycle's
        real episodes and force the resumed run to redo the cycle's
        whole exploration (run_20260827_032234 re-ran both of cycle 2's
        explores after the 12h wall killed its learn 16 minutes in).
        With this stash the resume reuses the episodes and jumps
        straight to learn. Best-effort: failures are logged, never
        fatal.
        """
        if not self.checkpointing:
            return
        path = self.inflight_path(stash.cycle)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                utils.pkl_dump_all_or_nothing(
                    {
                        "cycle": stash.cycle,
                        "interaction_results": stash.interaction_results,
                        "task_idxs": stash.task_idxs,
                        "task_solved_status": stash.task_solved_status,
                        "query_cost": stash.query_cost,
                    }, f)
            logging.info(
                "Saved %d in-flight interaction episode(s) to %s (reused "
                "if this cycle's learn dies before its checkpoint).",
                len(stash.interaction_results), path)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Could not save in-flight interactions: %s", e)

    def load_inflight(self, cycle: int) -> Optional[InflightInteractions]:
        """Reload the resumed-into cycle's persisted episodes, if fresh.

        Returns ``None`` when there is nothing (or nothing trustworthy)
        to reuse; freshness uses the same ``auto_resume_max_age_hours``
        gate as the checkpoint scan, so a relaunch of a long-finished
        experiment cannot silently replay stale episodes.
        """
        if not self.checkpointing:
            return None
        path = self.inflight_path(cycle)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > \
                CFG.auto_resume_max_age_hours * 3600.0:
            return None
        try:
            with open(path, "rb") as f:
                data = pkl.load(f)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Could not load in-flight interactions at %s: %s",
                            path, e)
            return None
        if data.get("cycle") != cycle:
            return None
        return InflightInteractions(
            cycle=cycle,
            interaction_results=data["interaction_results"],
            task_idxs=data["task_idxs"],
            task_solved_status=data["task_solved_status"],
            query_cost=data["query_cost"])

    def discard_inflight(self, cycle: int) -> None:
        """Drop the stash once its cycle's episodes are consumed."""
        if not self.checkpointing:
            return
        try:
            path = self.inflight_path(cycle)
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def discover_resume_cycles(
        load_path: str,
        suffix: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        now: Optional[float] = None) -> Tuple[bool, Optional[int]]:
    """Scan for ``{load_path}_{cycle}.{suffix}`` approach checkpoints.

    Returns ``(found_any, max_int_cycle)``: ``max_int_cycle`` is the
    highest completed online-learning cycle with a checkpoint, or None
    when only the post-offline (``_None``) checkpoint exists.

    ``suffix`` restricts the scan to files this approach class can load
    (another approach family's checkpoints under the same config path
    must not steer the resume). ``max_age_seconds`` ignores checkpoints
    older than that: the checkpoint path ignores the run timestamp, so
    without it a RELAUNCH of a finished experiment under the same
    experiment_id would silently "resume" the old run instead of
    starting fresh; a requeue/resubmission of a live run is recent.
    """
    found = False
    max_cycle: Optional[int] = None
    prefix_len = len(os.path.basename(load_path)) + 1
    now_ts = time.time() if now is None else now
    for path in glob.glob(glob.escape(load_path) + "_*"):
        name = os.path.basename(path)[prefix_len:]
        cycle_token, _, file_suffix = name.partition(".")
        if suffix is not None and file_suffix != suffix:
            continue
        if max_age_seconds is not None and \
                now_ts - os.path.getmtime(path) > max_age_seconds:
            continue
        if cycle_token == "None":
            found = True
            continue
        try:
            cycle = int(cycle_token)
        except ValueError:
            continue
        found = True
        max_cycle = cycle if max_cycle is None else max(max_cycle, cycle)
    return found, max_cycle


def maybe_auto_resume(approach: Any) -> None:
    """Under ``--auto_resume``, continue from the latest checkpoint.

    Sets ``load_approach`` (so the offline phase loads instead of re-
    learning), ``restart_learning`` (without it the online loop's
    learning gate skips learning on EVERY cycle of a loaded run), and
    ``skip_until_cycle`` past the last completed cycle. A run with no
    checkpoint starts fresh. This makes a Slurm requeue / resubmission
    of the identical command self-resuming.
    """
    if not getattr(CFG, "auto_resume", False):
        return
    checkpoints = ApproachCheckpoints.for_approach(approach)
    max_age = CFG.auto_resume_max_age_hours * 3600.0
    found, max_cycle = checkpoints.discover_resume_cycles(
        max_age_seconds=max_age)
    if not found:
        logging.info(
            "--auto_resume: no checkpoint at %s_*.%s newer than %.1f h; "
            "starting fresh. NOTE: the checkpoint path ignores the run "
            "timestamp, so concurrent launches of the same "
            "config/seed/experiment_id would overwrite each other's "
            "checkpoints - keep experiment_id unique per concurrent "
            "launch.", checkpoints.load_path, checkpoints.suffix or "*",
            CFG.auto_resume_max_age_hours)
        return
    CFG.load_approach = True
    CFG.restart_learning = True
    CFG.skip_until_cycle = 0 if max_cycle is None else max_cycle + 1
    logging.info(
        "--auto_resume: checkpoint(s) found at %s_* (last completed "
        "cycle: %s); resuming with load_approach + restart_learning, "
        "skip_until_cycle=%d.", checkpoints.load_path, max_cycle,
        CFG.skip_until_cycle)


# ── Per-cycle test results ───────────────────────────────────────


def test_results_path(online_learning_cycle: Optional[int]) -> str:
    """Where a cycle's test results pickle is written."""
    return (f"{CFG.results_dir}/{utils.get_config_path_str()}__"
            f"{online_learning_cycle}.pkl")


def test_results_exist(online_learning_cycle: Optional[int]) -> bool:
    """Whether the results pickle for a cycle's test was written."""
    return os.path.isfile(test_results_path(online_learning_cycle))


def load_test_solve_rate(
        online_learning_cycle: Optional[int]) -> Optional[float]:
    """num_solved / num_total from a cycle's saved test results, or None if the
    cycle has no saved results (or an empty test set)."""
    outfile = test_results_path(online_learning_cycle)
    if not os.path.isfile(outfile):
        return None
    with open(outfile, "rb") as f:
        results = pkl.load(f)["results"]
    if results["num_total"] <= 0:
        return None
    return results["num_solved"] / results["num_total"]


def perfect_test_streak_from_disk(newest_cycle: int) -> int:
    """Consecutive test phases ending at ``newest_cycle`` whose saved results
    solved every test task.

    Walks the per-cycle results pickles backward from ``newest_cycle``,
    so an --auto_resume relaunch continues the consecutive-perfect-test
    count that test-driven early stopping requires instead of restarting
    it. A fresh run (newest_cycle -1) seeds 0.
    """
    streak = 0
    for cycle in range(newest_cycle, -1, -1):
        rate = load_test_solve_rate(cycle)
        if rate is None or rate < 1.0:
            break
        streak += 1
    return streak
