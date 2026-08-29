"""Solve-attempt budget surfaces: the [budget] footer and watchdog."""
import contextlib
import threading
import time
from typing import Callable, Iterator, List, Optional

from predicators.agent_sdk.tools.context import ToolContext


def _budget_footer(ctx: ToolContext, rollouts_before: int = 0) -> str:
    """``[budget]`` line appended to tool results during a solve attempt.

    Shows attempt wall-clock (elapsed, and the budget when one is set)
    and the attempt's cumulative sim-rollout count (plus this call's
    delta). Agents pace well when they can see a clock and terribly when
    they can't: the 47k-rollout single-call sweep of run_20260717_230436
    ran 7 h with zero cost feedback. Empty when no attempt is in flight.
    """
    start = ctx.attempt_start
    if start is None:
        return ""
    parts = []
    elapsed_min = (time.monotonic() - start) / 60.0
    deadline = ctx.attempt_deadline
    if deadline is not None:
        total_min = (deadline - start) / 60.0
        parts.append(f"attempt time {elapsed_min:.1f}/{total_min:.0f} min")
    else:
        parts.append(f"attempt time {elapsed_min:.1f} min")
    total_rollouts = ctx.attempt_rollout_count
    delta = total_rollouts - rollouts_before
    rollout_part = f"sim rollouts this attempt: {total_rollouts}"
    if delta > 0:
        rollout_part += f" (+{delta} this call)"
    parts.append(rollout_part)
    return "\n\n[budget] " + "; ".join(parts)


class _Watchdog:
    """One armed budget watchdog: a timer that injects ``ProbeBudgetExceeded``
    into the arming thread, pausable so a block that owns its own budget (a
    canonical fit) does not count against the call that contains it."""

    def __init__(self, seconds: float, target_id: int) -> None:
        self._target_id = target_id
        self._lock = threading.Lock()
        self._armed = False
        self._timer: Optional[threading.Timer] = None
        self._deadline = 0.0
        self._start(seconds)

    def _start(self, seconds: float) -> None:
        # pylint: disable-next=import-outside-toplevel
        import ctypes

        def _fire() -> None:
            # The lock makes fire/disarm mutually exclusive so the async
            # exception cannot be injected after the call has already
            # returned and disarmed.
            with self._lock:
                if not self._armed:
                    return
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(self._target_id),
                    ctypes.py_object(_probe_budget_exceeded()))

        with self._lock:
            self._armed = True
            self._deadline = time.monotonic() + max(0.0, seconds)
            self._timer = threading.Timer(max(0.0, seconds), _fire)
            self._timer.daemon = True
            self._timer.start()

    def disarm(self) -> None:
        """Idempotent: cancel the timer and forget the deadline."""
        with self._lock:
            self._armed = False
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        _pop_active(self)

    def pause(self) -> float:
        """Cancel the timer; returns the remaining seconds (0 if spent)."""
        with self._lock:
            self._armed = False
            timer, self._timer = self._timer, None
            remaining = max(0.0, self._deadline - time.monotonic())
        if timer is not None:
            timer.cancel()
        return remaining

    def resume(self, seconds: float) -> None:
        """Re-arm for ``seconds`` (a paused watchdog's remaining time)."""
        self._start(seconds)


def _probe_budget_exceeded() -> type:
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.belief_probe import ProbeBudgetExceeded
    return ProbeBudgetExceeded


_ACTIVE = threading.local()


def _active_stack() -> List[_Watchdog]:
    stack = getattr(_ACTIVE, "stack", None)
    if stack is None:
        stack = []
        _ACTIVE.stack = stack
    return stack


def _pop_active(wd: _Watchdog) -> None:
    stack = _active_stack()
    if wd in stack:
        stack.remove(wd)


def _arm_budget_watchdog(seconds: float) -> Callable[[], None]:
    """Schedule a ProbeBudgetExceeded in the CALLING thread after ``seconds``;
    returns an idempotent disarm callable.

    ``run_python``'s exec() runs on the event-loop thread, so
    pure-Python code that never reaches a probe checkpoint blocks every
    cooperative deadline check AND the sandbox's message-stream
    interrupt backstop - an async exception from a watchdog timer is the
    only preemption that reaches it. Delivery happens at the next
    bytecode boundary, so a long blocking C call (a physics step) defers
    it; those paths are exactly the ones the cooperative probe checks
    already cover.
    """
    wd = _Watchdog(seconds, threading.get_ident())
    _active_stack().append(wd)
    return wd.disarm


@contextlib.contextmanager
def suspend_budget_watchdog(own_timeout: float = 0.0) -> Iterator[None]:
    """Run a block that owns its own wall-clock budget.

    The calling thread's active watchdogs (the enclosing tool call's
    per- call cap) are paused for the block and re-armed afterwards with
    the time they had left, so the block's duration does not count
    against the call that contains it. ``own_timeout > 0`` arms a
    private watchdog for the block itself. Made for the canonical
    ``sim.fit``: a rollout system-ID fit can legitimately take longer
    than one ``run_python`` call is allowed, and being stopped mid-fit
    used to surface as an empty "param fitting failed:" (sketch seed1,
    2026-08-28 learn 011).
    """
    paused = [(wd, wd.pause()) for wd in list(_active_stack())]
    own: Optional[_Watchdog] = None
    if own_timeout > 0:
        own = _Watchdog(own_timeout, threading.get_ident())
    try:
        yield
    finally:
        if own is not None:
            own.disarm()
        for wd, remaining in paused:
            wd.resume(remaining)
