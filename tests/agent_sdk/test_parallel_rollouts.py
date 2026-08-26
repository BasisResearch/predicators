"""Tests for the fork-based parallel rollout helper."""

import time

import pytest

from predicators.agent_sdk.parallel_rollouts import \
    parallel_rollouts_available, run_forked_rollouts


def test_results_are_index_aligned() -> None:
    """Results land at their job's index regardless of finish order."""
    if not parallel_rollouts_available():
        pytest.skip("fork not available on this platform")

    def make_job(i: int):

        def job() -> int:
            # Later jobs finish first, so delivery order is reversed.
            time.sleep(0.2 * (3 - i))
            return i * 10

        return job

    out = run_forked_rollouts([make_job(i) for i in range(4)],
                              max_workers=4,
                              label="test")
    assert out == [0, 10, 20, 30]


def test_child_exception_yields_none_slot() -> None:
    """A raising job costs only its own slot."""
    if not parallel_rollouts_available():
        pytest.skip("fork not available on this platform")

    def ok_job() -> str:
        return "fine"

    def bad_job() -> str:
        raise ValueError("boom")

    out = run_forked_rollouts([ok_job, bad_job, ok_job],
                              max_workers=2,
                              label="test")
    assert out == ["fine", None, "fine"]


def test_deadline_abandons_hung_child() -> None:
    """A hung child is abandoned at the deadline; fast peers survive."""
    if not parallel_rollouts_available():
        pytest.skip("fork not available on this platform")

    def fast_job() -> str:
        return "done"

    def hung_job() -> str:
        time.sleep(30)
        return "never"

    start = time.monotonic()
    out = run_forked_rollouts([fast_job, hung_job],
                              max_workers=2,
                              label="test",
                              per_job_timeout=1.0)
    assert out == ["done", None]
    assert time.monotonic() - start < 10


def test_child_mutations_stay_in_child() -> None:
    """Fork isolation: a child's writes never reach the parent."""
    if not parallel_rollouts_available():
        pytest.skip("fork not available on this platform")

    shared = {"value": 1}

    def job() -> int:
        shared["value"] = 99
        return shared["value"]

    out = run_forked_rollouts([job], max_workers=1, label="test")
    assert out == [99]
    assert shared["value"] == 1
