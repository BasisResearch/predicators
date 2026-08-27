"""Tests for the async rollout registry (launch now, gather later).

Cheap jobs only - the production-topology validation (live PyBullet
parent, real bridge plans) lives in ``scripts/async_rollout_bench.py``
and runs on a compute node.
"""

import time

import pytest

from predicators.agent_sdk.async_rollouts import AsyncRolloutRegistry, \
    async_rollouts_available

pytestmark = pytest.mark.skipif(not async_rollouts_available(),
                                reason="fork-based rollouts need linux")


def test_launch_and_gather_returns_results_in_handle_order():
    """Results land on the right handles regardless of finish order."""
    reg = AsyncRolloutRegistry(max_workers=4)
    try:
        # Later jobs sleep less, so completion order inverts launch
        # order - results must still land by index.
        handles = [
            reg.launch(lambda i=i: (time.sleep(0.3 - 0.05 * i), i * 2)[1])
            for i in range(4)
        ]
        done, pending = reg.gather(handles, timeout=30)
        assert not pending
        assert [h.result for h in done] == [0, 2, 4, 6]
        assert all(h.ok for h in done)
        assert all(h.error is None for h in done)
    finally:
        reg.shutdown()


def test_overlap_hides_rollout_time_behind_parent_work():
    """launch -> parent work -> gather runs in ~max of the two, not sum."""
    reg = AsyncRolloutRegistry(max_workers=4)
    try:
        t0 = time.monotonic()
        handles = [
            reg.launch(lambda: time.sleep(0.5) or "ok") for _ in range(4)
        ]
        time.sleep(0.5)  # the agent "thinking" between run_python calls
        done, pending = reg.gather(handles, timeout=30)
        wall = time.monotonic() - t0
        assert not pending and all(h.ok for h in done)
        # Sequential would be 4*0.5 + 0.5 = 2.5s; overlap target ~0.5s.
        assert wall < 1.5, f"no overlap: wall={wall:.2f}s"
    finally:
        reg.shutdown()


def test_cap_queues_excess_jobs_and_gather_drains_them():
    """More jobs than workers: the rest queue and finish via pumping."""
    reg = AsyncRolloutRegistry(max_workers=2)
    try:
        handles = [reg.launch(lambda i=i: i) for i in range(5)]
        n_done, n_running, n_queued = reg.poll()
        assert n_running <= 2
        assert n_done + n_running + n_queued == 5
        done, pending = reg.gather(handles, timeout=30)
        assert not pending
        assert sorted(h.result for h in done) == [0, 1, 2, 3, 4]
    finally:
        reg.shutdown()


def test_second_wave_after_reaper_thread_exists():
    """Forking after the reaper thread is live must not deadlock."""
    reg = AsyncRolloutRegistry(max_workers=2)
    try:
        first, _ = reg.gather([reg.launch(lambda: "one")], timeout=30)
        assert first[0].result == "one"
        second, pending = reg.gather(
            [reg.launch(lambda i=i: i) for i in range(3)], timeout=30)
        assert not pending
        assert sorted(h.result for h in second) == [0, 1, 2]
    finally:
        reg.shutdown()


def test_child_exception_fails_only_its_handle():
    """A raising job reports its error; siblings are unaffected."""
    reg = AsyncRolloutRegistry(max_workers=2)
    try:

        def _boom():
            raise ValueError("deliberate")

        bad = reg.launch(_boom)
        good = reg.launch(lambda: 7)
        done, pending = reg.gather([bad, good], timeout=30)
        assert not pending and len(done) == 2
        assert not bad.ok and "deliberate" in bad.error
        assert bad.result is None
        assert good.ok and good.result == 7
    finally:
        reg.shutdown()


def test_per_job_timeout_fails_overdue_child():
    """A child overrunning per_job_timeout is terminated and failed."""
    reg = AsyncRolloutRegistry(max_workers=2, per_job_timeout=0.5)
    try:
        slow = reg.launch(lambda: time.sleep(30))
        done, pending = reg.gather([slow], timeout=30)
        assert done and not pending
        assert not slow.ok and "timed out" in slow.error
    finally:
        reg.shutdown()


def test_gather_timeout_returns_partial():
    """gather with a timeout hands back (done, pending) without blocking."""
    reg = AsyncRolloutRegistry(max_workers=2)
    try:
        fast = reg.launch(lambda: "fast")
        slow = reg.launch(lambda: time.sleep(10) or "slow")
        done, pending = reg.gather([fast, slow], timeout=2)
        assert fast in done
        assert slow in pending
    finally:
        reg.shutdown()


def test_shutdown_fails_running_and_queued_handles():
    """shutdown terminates children and fails every unfinished handle."""
    reg = AsyncRolloutRegistry(max_workers=1)
    running = reg.launch(lambda: time.sleep(30))
    queued = reg.launch(lambda: "never")
    reg.shutdown()
    assert running.done() and not running.ok
    assert "shut down" in running.error
    assert queued.done() and not queued.ok
    assert "shut down" in queued.error


def test_handles_cross_call_scopes():
    """Handles created in one scope gather in another (the run_python
    usage: launch in call N, collect in call N+k via the persistent
    namespace)."""
    reg = AsyncRolloutRegistry(max_workers=2)
    try:

        def _call_one():
            return [reg.launch(lambda i=i: i * 10) for i in range(2)]

        def _call_two(handles):
            done, pending = reg.gather(handles, timeout=30)
            assert not pending
            return sorted(h.result for h in done)

        assert _call_two(_call_one()) == [0, 10]
    finally:
        reg.shutdown()
