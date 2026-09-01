"""Fork-based parallel execution of independent validation rollouts.

The capture gate's margin sweep, its repeat rollouts, and the belief
probe's ``trials=N`` / ``physics_sweep`` modes all run the same shape of
work: N independent rollouts, each on a freshly constructed env under
its own seed / override scope, in one CPU-bound process. This module
fans them out as forked children.

Fork (never spawn) is load-bearing: each job is a CLOSURE over live
session state - the approach object, the dynamically loaded
``simulator.py`` module, the option model - none of which can be
pickled to a worker pool. A forked child inherits it all copy-on-write,
applies its own scope in its private copy, and ships only the (small,
picklable) result back through a queue.

Two rules learned from benchmark run 21335464 (see
``scripts/parallel_rollout_bench.py``), both load-bearing:

* Children exit with ``os._exit`` right after a SYNCHRONOUS
  ``SimpleQueue.put``. Normal interpreter teardown runs PyBullet's
  atexit handler against the forked copy of the parent's live client
  and lags exit by seconds; ``mp.Queue`` would instead hand the result
  to a feeder thread that ``os._exit`` could kill mid-write.
* The parent's window is bookkept by OUTSTANDING RESULTS, never by
  process liveness: a child that delivered its result frees its slot
  even while its exit lags, and a liveness-based window deadlocks.

The parallel path is strictly an optimization: any child that fails or
times out yields ``None`` at its index, and callers re-run those jobs
sequentially, so a broken parallel path can cost time but never
correctness.
"""

import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def parallel_rollouts_available() -> bool:
    """Fork-based parallelism needs a real fork (Linux/macOS)."""
    return sys.platform.startswith("linux") and hasattr(os, "fork")


def prefetch_parallel(jobs: Sequence[Callable[[], Any]],
                      label: str) -> List[Optional[Any]]:
    """Pre-run independent rollout jobs as forked children when enabled.

    Returns an index-aligned result list, or all-``None`` when parallel
    execution is disabled (``CFG.agent_validation_parallel_workers`` <=
    1), unavailable, or pointless (one job). Callers consume results
    positionally and re-run any ``None`` entry through their sequential
    path, so verdict semantics (seeds, scopes, bookkeeping, early
    breaks) are identical with the flag on or off - the parallel pass
    only prepays the rollouts.
    """
    # Deferred: settings must stay import-cycle-free from tool modules.
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    workers = min(int(CFG.agent_validation_parallel_workers), len(jobs))
    if workers <= 1 or len(jobs) <= 1 or not parallel_rollouts_available():
        return [None] * len(jobs)
    logger.info("[%s] prefetching %d rollouts across %d forked children.",
                label, len(jobs), workers)
    return run_forked_rollouts(jobs, workers, label)


def _child_main(idx: int, job: Callable[[], Any], q: Any,
                quiet_child_logging: bool) -> None:
    """Run one job in the forked child and exit without teardown."""
    exit_code = 0
    try:
        if quiet_child_logging:
            # The child inherits the parent's log handlers (the run's
            # info.log); N interleaved copies of per-option INFO lines
            # would make it unreadable. Failures still come through as
            # part of the returned result.
            logging.disable(logging.WARNING)
        result = job()
        q.put((idx, True, result))
    except BaseException as e:  # pylint: disable=broad-except
        exit_code = 1
        try:
            q.put((idx, False,
                   f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        except BaseException:  # pylint: disable=broad-except
            exit_code = 2
    finally:
        os._exit(exit_code)  # pylint: disable=protected-access


def run_forked_rollouts(
    jobs: Sequence[Callable[[], Any]],
    max_workers: int,
    label: str,
    per_job_timeout: float = 900.0,
    quiet_child_logging: bool = True,
) -> List[Optional[Any]]:
    """Run ``jobs`` in forked children, at most ``max_workers`` at once.

    Returns an index-aligned list: each entry is the job's return value,
    or ``None`` when that child failed or the deadline passed (the
    caller decides whether to re-run those sequentially). Results must
    be picklable; jobs must not depend on each other or on shared
    mutable state surviving into the parent (child-side mutations stay
    in the child).
    """
    n = len(jobs)
    if n == 0:
        return []
    assert max_workers >= 1
    ctx = mp.get_context("fork")
    q = ctx.SimpleQueue()
    procs: dict = {}
    out: List[Optional[Any]] = [None] * n
    got = 0
    next_idx = 0
    outstanding = 0
    deadline = time.monotonic() + per_job_timeout * max(
        1, (n + max_workers - 1) // max_workers)
    try:
        while got < n:
            while next_idx < n and outstanding < max_workers:
                p = ctx.Process(target=_child_main,
                                args=(next_idx, jobs[next_idx], q,
                                      quiet_child_logging))
                p.start()
                procs[next_idx] = p
                outstanding += 1
                next_idx += 1
            timed_out = False
            while q.empty():
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                time.sleep(0.05)
            if timed_out:
                logger.warning(
                    "[%s] parallel rollouts deadline exceeded with %d/%d "
                    "results; abandoning %d outstanding children.", label, got,
                    n, outstanding)
                break
            idx, ok, payload = q.get()
            got += 1
            outstanding -= 1
            if ok:
                out[idx] = payload
            else:
                logger.warning("[%s] parallel rollout %d failed in child:\n%s",
                               label, idx, payload)
            done = procs.pop(idx, None)
            if done is not None:
                done.join(timeout=30)
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)
    return out
