"""Early stopping of the online learning loop.

Two mutually exclusive modes, selected by
``CFG.online_learning_early_stopping_by_test_solve_rate``:

(A) Train-driven (default; requires ``online_learning_early_stopping``).
    Stop once a cycle's interaction requests cover every train task and
    all of those attempts succeeded, provided the model that generated
    the attempts has learned at least once (from offline demos, a loaded
    approach, or a prior online update). Otherwise (e.g. cycle 0 with no
    demos) the explorer's successes reflect only the initial mental
    model, and stopping would skip learning entirely. Sub-mode
    ``online_learning_early_stopping_require_all_attempts``:

    - False: only the first attempt per task must succeed (legacy).
    - True: every attempt must succeed. Combined with multiple
      interaction requests per cycle and the explorer's advancing rng
      (so each request samples differently) this guards against a
      single lucky sample masking a buggy learned model.

    Checked AFTER the cycle's interactions - they are its evidence.

(B) Test-driven. Stop once the last
    ``online_learning_early_stopping_consecutive_perfect_tests`` test
    phases each solved every test task. In a stochastic environment a
    single perfect phase over a small test set can be one lucky rollout;
    the consecutive requirement demands repeated evidence. This mode
    ignores ``online_learning_early_stopping`` itself and is checked at
    the TOP of a cycle, before the interaction phase (whose results it
    does not read): on the stopping cycle learning is skipped, which
    would make the interactions pure cost. The streak is seeded from the
    per-cycle results on disk so an ``--auto_resume`` relaunch continues
    it.

Either way the stopping cycle skips learning, so the model is the one
the previous test already measured; ``force_final_test`` says whether
that test must nevertheless be repeated.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from predicators.run.checkpoints import perfect_test_streak_from_disk
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Metrics

TestSummary = Tuple[str, Metrics]


class EarlyStopping:
    """Accumulates the evidence early stopping reads and renders verdicts.

    The loop feeds it three things - train attempts (``record_train_
    attempts``), test results (``record_test``) and learning updates
    (``record_learned``) - and asks ``test_driven_stop`` at the top of
    a cycle and ``train_driven_stop`` after the interactions.
    """

    def __init__(self,
                 num_train_tasks: int,
                 model_has_learned: bool,
                 initial_test_summary: Optional[TestSummary] = None) -> None:
        self._num_train_tasks = num_train_tasks
        self._model_has_learned = model_has_learned
        # (label, results) of the most recent test evaluation, re-logged
        # on early stopping so the final solve rate and rewards are
        # visible at the end of the log instead of cycles back.
        self._last_test_summary = initial_test_summary
        # Consecutive test phases (up to and including the most recent
        # one) that solved every test task - mode B's evidence.
        self._perfect_test_streak = 0
        # This cycle's attempts, mode A's evidence.
        self._first_attempts: Dict[int, bool] = {}
        self._all_attempts: Dict[int, List[bool]] = {}

    # ── Evidence ─────────────────────────────────────────────────

    @property
    def last_test_summary(self) -> Optional[TestSummary]:
        """The most recent test evaluation as ``(label, results)``."""
        return self._last_test_summary

    @property
    def model_has_learned(self) -> bool:
        """Whether the model has learned at least once (mode A's gate)."""
        return self._model_has_learned

    @property
    def perfect_test_streak(self) -> int:
        """Consecutive most-recent test phases that solved every task."""
        return self._perfect_test_streak

    def seed_test_streak_from_disk(self, newest_cycle: int) -> None:
        """Continue the streak recorded by earlier incarnations of this run.

        ``newest_cycle`` is the last cycle whose test results are on
        disk (-1 for a fresh run, which seeds 0).
        """
        self._perfect_test_streak = perfect_test_streak_from_disk(newest_cycle)

    def record_learned(self) -> None:
        """The model just learned from interaction results."""
        self._model_has_learned = True

    def record_test(self, label: str, results: Metrics) -> None:
        """A test phase finished; extend or reset the perfect streak."""
        # results is a defaultdict(float): reading a key the test never
        # wrote would silently yield 0.0, so derive the verdict from the
        # counts it does write.
        solved_all = (results["num_total"] > 0
                      and results["num_solved"] == results["num_total"])
        self._perfect_test_streak = (self._perfect_test_streak +
                                     1 if solved_all else 0)
        self._last_test_summary = (label, results)

    def record_train_attempts(self, task_idxs: Sequence[int],
                              task_solved_status: Sequence[bool]) -> float:
        """A cycle's interaction attempts, one entry per request.

        Replaces the previous cycle's attempts. Returns and logs the
        train task solve rate: over every attempt when
        ``require_all_attempts`` is on, so the denominator matches the
        stopping criterion, else over the first attempt per task (the
        legacy metric).
        """
        self._first_attempts = {}
        self._all_attempts = {}
        for task_idx, solved in zip(task_idxs, task_solved_status):
            self._all_attempts.setdefault(task_idx, []).append(solved)
            self._first_attempts.setdefault(task_idx, solved)
        if CFG.online_learning_early_stopping_require_all_attempts:
            scored = [
                solved for attempts in self._all_attempts.values()
                for solved in attempts
            ]
        else:
            scored = list(self._first_attempts.values())
        if not scored:
            return 0.0
        rate = sum(scored) / len(scored)
        logging.info(f"Train task solve rate: {rate:.3f} "
                     f"({sum(scored)}/{len(scored)})")
        return rate

    # ── Verdicts ─────────────────────────────────────────────────

    def test_driven_stop(self) -> bool:
        """Mode B: enough consecutive perfect tests?

        Logs when stopping.
        """
        if not CFG.online_learning_early_stopping_by_test_solve_rate:
            return False
        if self._perfect_test_streak < \
                CFG.online_learning_early_stopping_consecutive_perfect_tests:
            return False
        logging.info(
            f"The last {self._perfect_test_streak} test phase(s) solved "
            "every test task, triggering early stopping.\n")
        return True

    def train_driven_stop(self) -> bool:
        """Mode A: every train task solved this cycle by a learned model?

        Logs the verdict, including the ineligible case (all solved but
        the model has not learned yet).
        """
        if not CFG.online_learning_early_stopping or \
                CFG.online_learning_early_stopping_by_test_solve_rate:
            return False
        if CFG.online_learning_early_stopping_require_all_attempts:
            all_solved = (len(self._all_attempts) == self._num_train_tasks
                          and all(attempts and all(attempts)
                                  for attempts in self._all_attempts.values()))
            stop_msg = ("All training tasks solved on every attempt this "
                        "cycle, triggering early stopping.\n")
        else:
            all_solved = (len(self._first_attempts) == self._num_train_tasks
                          and all(self._first_attempts.values()))
            stop_msg = ("All training tasks solved on first attempt, "
                        "triggering early stopping.\n")
        if not all_solved:
            return False
        if not self._model_has_learned:
            logging.info(
                "All training tasks solved this cycle, but the model has "
                "not learned yet, so early stopping is not eligible; "
                "continuing to learning.\n")
            return False
        logging.info(stop_msg)
        return True

    @property
    def force_final_test(self) -> bool:
        """Whether the stopping cycle must test the final model.

        Learning is skipped on the stopping cycle, so the model is
        identical to the one the previous cycle tested; re-testing only
        re-samples test-time stochasticity at full test-set cost, and
        the user may opt out of it (``skip_redundant_test``). The re-
        test is still forced when the stopping cycle would be the
        model's only test (``skip_test_until_last_ite_or_early_
        stopping``) or when no test has run at all yet (e.g.
        ``skip_initial_test`` and stopping on cycle 0).
        """
        return not (CFG.online_learning_early_stopping_skip_redundant_test
                    and not CFG.skip_test_until_last_ite_or_early_stopping
                    and self._last_test_summary is not None)


def below_reward_bar_msg(episode_reward: float,
                         env_task: EnvironmentTask) -> Optional[str]:
    """Check a solved episode's reward against the task's early-stopping bar.

    Returns a log-ready description when the episode reward falls short
    of ``env_task.early_stop_min_reward`` (minus the configured slack),
    meaning the solve must NOT count toward early stopping; returns None
    when the task sets no bar, the bar is ignored via
    ``CFG.online_learning_early_stopping_ignore_reward_bar``, or the
    reward clears it. The comparison carries a small tolerance so a
    reward computed exactly at the bar is never rejected on float
    rounding.
    """
    reward_bar = env_task.early_stop_min_reward
    if reward_bar is None:
        return None
    if CFG.online_learning_early_stopping_ignore_reward_bar:
        return None
    slack = CFG.online_learning_early_stopping_reward_slack
    if episode_reward >= reward_bar - slack - 1e-9:
        return None
    return (f"below the early-stop reward bar (reward={episode_reward:g} < "
            f"min_reward={reward_bar:g} - slack {slack:g})")
