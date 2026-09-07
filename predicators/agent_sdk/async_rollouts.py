"""Async fork-based rollouts: launch now, collect later.

``parallel_rollouts.run_forked_rollouts`` blocks its caller for the
whole batch. This module decouples the two ends so an agent can LAUNCH
independent rollouts inside one ``run_python`` call, keep working (edit
the model, fit, think) while they run, and GATHER the results in a
later call: handles live in the session's persistent exec namespace.

The proven fork rules carry over unchanged (see ``parallel_rollouts``):
fork (never spawn) because jobs are closures over live session state;
children ``os._exit`` right after a synchronous ``SimpleQueue.put``;
the worker window is bookkept by outstanding results, never process
liveness. Two rules are new and load-bearing here:

* Only the REAPER THREAD reads the queue, and it never logs and never
  forks - it drains results into the registry and joins exited
  children. Forking stays on the caller's thread (``launch`` /
  ``poll`` / ``gather`` pump queued jobs into free slots), so a child
  can never inherit a lock held by a thread that does not exist in it.
* Collection is cooperative: with more jobs than workers, queued jobs
  are forked when a pump call finds free slots. A caller that launches
  a big batch and never polls keeps only the first window running -
  ``gather`` pumps continuously, so the natural launch/think/gather
  usage drains everything.

The async path is strictly an optimization: a handle whose child
failed, timed out, or was shut down reports ``ok=False`` with the
reason, and the caller re-runs that job synchronously if it still
wants the result.
"""

import multiprocessing as mp
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from predicators.agent_sdk.parallel_rollouts import _child_main, \
    parallel_rollouts_available


class CompletedRollout:
    """Pre-completed handle for the synchronous fallback path.

    Presents the same interface as ``AsyncRollout`` so caller code is
    identical whether the launch actually forked (workers > 1 on linux)
    or ran inline.
    """

    index = -1

    def __init__(self,
                 result: Any = None,
                 error: Optional[str] = None,
                 tag: Optional[str] = None) -> None:
        self.tag = tag
        self._result = result
        self._error = error

    def done(self) -> bool:
        """Always finished."""
        return True

    @property
    def ok(self) -> bool:
        """Whether the inline run returned a result."""
        return self._error is None

    @property
    def result(self) -> Any:
        """The inline run's return value (None on failure)."""
        return self._result

    @property
    def error(self) -> Optional[str]:
        """The failure reason, or None on success."""
        return self._error


class AsyncRollout:
    """Handle to one launched rollout.

    ``done()`` is non-blocking. After completion, ``ok`` says whether
    the child returned normally; ``result`` holds its (picklable) return
    value on success and ``error`` the failure reason otherwise. ``tag``
    is caller-provided context stamped at launch (e.g. the model version
    the rollout ran under).

    Every accessor pumps the registry's scheduler first, so a caller
    that waits by polling ``done()`` still gets queued launches forked
    into freed worker slots (queued jobs are only forked from caller
    threads, never from the reaper). ``gather`` remains the preferred
    wait: it also bounds the wait and reports staleness.
    """

    def __init__(self, index: int, registry: "AsyncRolloutRegistry",
                 tag: Optional[str]) -> None:
        self.index = index
        self.tag = tag
        self._registry = registry

    def _entry(self) -> Tuple[bool, bool, Any]:
        """(done, ok, payload), pumping queued launches first."""
        self._registry.poll()
        return self._registry.entry(self.index)

    def done(self) -> bool:
        """Whether the rollout has finished (successfully or not)."""
        return self._entry()[0]

    @property
    def ok(self) -> bool:
        """Whether the child returned a result (False while pending)."""
        entry = self._entry()
        return entry[0] and entry[1]

    @property
    def result(self) -> Any:
        """The child's return value, or None while pending / on failure."""
        entry = self._entry()
        return entry[2] if entry[0] and entry[1] else None

    @property
    def error(self) -> Optional[str]:
        """The failure reason, or None while pending / on success."""
        entry = self._entry()
        return entry[2] if entry[0] and not entry[1] else None


class AsyncRolloutRegistry:
    """Launch forked rollout children and collect their results later.

    One registry per session. ``max_workers`` bounds concurrent
    children; further launches queue and are forked by later pump calls.
    ``per_job_timeout`` bounds each child's wall clock from its fork;
    the pump terminates overdue children and fails their handles.
    """

    def __init__(self,
                 max_workers: int,
                 per_job_timeout: float = 900.0,
                 quiet_child_logging: bool = True) -> None:
        assert max_workers >= 1
        self._max_workers = max_workers
        self._per_job_timeout = per_job_timeout
        self._quiet_child_logging = quiet_child_logging
        self._ctx = mp.get_context("fork")
        self._queue = self._ctx.SimpleQueue()
        self._lock = threading.Lock()
        # index -> (done, ok, payload); payload is the result or the
        # failure reason. Pending entries are absent.
        self._entries: Dict[int, Tuple[bool, bool, Any]] = {}
        self._procs: Dict[int, Any] = {}
        self._deadlines: Dict[int, float] = {}
        self._pending_jobs: List[Tuple[int, Callable[[], Any]]] = []
        self._next_index = 0
        self._outstanding = 0
        self._stop = False
        self._reaper: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def launch(self,
               job: Callable[[], Any],
               tag: Optional[str] = None) -> AsyncRollout:
        """Register ``job`` and fork it if a worker slot is free.

        The job must be independent of other jobs and return a picklable
        value; child-side mutations stay in the child.
        """
        assert not self._stop, "registry is shut down"
        with self._lock:
            index = self._next_index
            self._next_index += 1
            self._pending_jobs.append((index, job))
        self._ensure_reaper()
        self._pump()
        return AsyncRollout(index, self, tag)

    def poll(self) -> Tuple[int, int, int]:
        """Pump the scheduler; return (n_done, n_running, n_queued)."""
        self._pump()
        with self._lock:
            return (len(self._entries), self._outstanding,
                    len(self._pending_jobs))

    def gather(
        self,
        handles: Sequence[AsyncRollout],
        timeout: Optional[float] = None
    ) -> Tuple[List[AsyncRollout], List[AsyncRollout]]:
        """Wait for ``handles`` (pumping queued jobs), up to ``timeout``.

        Returns ``(done, pending)`` in the input order; with no timeout
        it blocks until every handle is done (each child is still
        individually bounded by ``per_job_timeout``).
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            self._pump()
            if all(h.done() for h in handles):
                break
            if deadline is not None and time.monotonic() > deadline:
                break
            time.sleep(0.05)
        done = [h for h in handles if h.done()]
        pending = [h for h in handles if not h.done()]
        return done, pending

    def shutdown(self) -> None:
        """Terminate live children and fail every unfinished handle."""
        self._stop = True
        if self._reaper is not None:
            self._reaper.join(timeout=5)
        with self._lock:
            for index, proc in self._procs.items():
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=10)
                if index not in self._entries:
                    self._entries[index] = (True, False,
                                            "shut down before completion")
            self._procs.clear()
            for index, _ in self._pending_jobs:
                self._entries[index] = (True, False, "shut down before launch")
            self._pending_jobs.clear()
            self._outstanding = 0

    def entry(self, index: int) -> Tuple[bool, bool, Any]:
        """(done, ok, payload) for a handle index; pump-free peek."""
        with self._lock:
            return self._entries.get(index, (False, False, None))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_reaper(self) -> None:
        if self._reaper is None or not self._reaper.is_alive():
            self._reaper = threading.Thread(target=self._reap,
                                            name="async-rollout-reaper",
                                            daemon=True)
            self._reaper.start()

    def _reap(self) -> None:
        """Drain the queue into the registry.

        NEVER logs, NEVER forks: a child forked while this thread runs
        must not be able to inherit a lock this thread could hold
        (children touch neither the registry lock nor the queue's read
        lock). Idle passes also enforce per-child deadlines, so an
        overdue child dies even when the caller never pumps again (a
        session abandoned mid-flight).
        """
        while not self._stop:
            if self._queue.empty():
                self._fail_overdue()
                time.sleep(0.05)
                continue
            index, ok, payload = self._queue.get()
            with self._lock:
                # A result can arrive for a child the pump already
                # marked timed out (put raced the terminate): keep the
                # real result, but never decrement the window twice.
                already = index in self._entries
                self._entries[index] = (True, ok, payload)
                if not already:
                    self._outstanding -= 1
                self._deadlines.pop(index, None)
                proc = self._procs.pop(index, None)
            if proc is not None:
                proc.join(timeout=30)

    def _fail_overdue(self) -> None:
        """Terminate children past their deadline and fail their handles.

        Lock-guarded and log-free, so it is safe from both the caller
        threads (via ``_pump``) and the reaper's idle loop.
        """
        now = time.monotonic()
        with self._lock:
            overdue = [(i, p) for i, p in self._procs.items()
                       if now > self._deadlines.get(i, now)]
            for index, proc in overdue:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=10)
                del self._procs[index]
                self._deadlines.pop(index, None)
                if index not in self._entries:
                    self._entries[index] = (
                        True, False,
                        f"timed out after {self._per_job_timeout:.0f}s")
                    self._outstanding -= 1

    def _pump(self) -> None:
        """Fork queued jobs into free slots; fail overdue children.

        Runs only on caller threads (launch/poll/gather), never on the
        reaper.
        """
        if self._stop:
            return
        self._fail_overdue()
        now = time.monotonic()
        with self._lock:
            while self._pending_jobs and self._outstanding < self._max_workers:
                index, job = self._pending_jobs.pop(0)
                proc = self._ctx.Process(target=_child_main,
                                         args=(index, job, self._queue,
                                               self._quiet_child_logging))
                proc.start()
                self._procs[index] = proc
                self._deadlines[index] = now + self._per_job_timeout
                self._outstanding += 1


def async_rollouts_available() -> bool:
    """Same platform requirement as the synchronous path."""
    return parallel_rollouts_available()
