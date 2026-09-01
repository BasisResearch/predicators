"""Tests for ``BeliefProbe.run_async`` / ``gather``.

The probe's ``run`` is monkeypatched to a cheap stub so these cover the
async wiring (registry lifecycle, fallback path, staleness tags, gather
summaries) without an env; the machinery itself is covered by
``test_async_rollouts.py`` and the compute-node benchmark
(``scripts/async_rollout_bench.py``).
"""
# pylint: disable=protected-access
from __future__ import annotations

import os
import time

import pytest

from predicators import utils
from predicators.agent_sdk.async_rollouts import async_rollouts_available
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools import ToolContext

pytestmark = pytest.mark.skipif(not async_rollouts_available(),
                                reason="fork-based rollouts need linux")


def _probe(monkeypatch, workers, run_fn) -> BeliefProbe:
    utils.reset_config({"agent_validation_parallel_workers": workers})
    monkeypatch.setattr(BeliefProbe, "run", run_fn)
    return BeliefProbe(ToolContext())


def test_fallback_runs_inline_and_returns_done_handle(monkeypatch):
    """workers<=1: run_async executes inline; the handle is already done."""
    calls = []

    def _run(self, plan_text, **kwargs):
        del self
        calls.append((plan_text, kwargs))
        return "INLINE"

    sim = _probe(monkeypatch, 0, _run)
    # No state to snapshot around: stub run never touches it, and
    # snapshot() auto-resets via _require_state, which needs a task -
    # give the probe a state directly.
    sim._state = None
    monkeypatch.setattr(BeliefProbe, "snapshot", lambda self: 1)
    monkeypatch.setattr(BeliefProbe, "restore", lambda self, sid: self)
    monkeypatch.setattr(BeliefProbe, "drop", lambda self, sid: self)
    h = sim.run_async("PLAN")
    assert h.done() and h.ok and h.result == "INLINE"
    assert calls[0][0] == "PLAN"
    assert calls[0][1]["render"] is False  # rendering forced off
    out = sim.gather([h])
    assert "1 done, 0 pending" in repr(out)


def test_parallel_launch_and_gather(monkeypatch):
    """workers>1: children run the (stubbed) run; gather summarizes."""

    def _run(self, plan_text, **kwargs):
        del self, kwargs
        time.sleep(0.2)
        return f"ran {plan_text}"

    sim = _probe(monkeypatch, 2, _run)
    handles = [sim.run_async(f"PLAN{i}") for i in range(2)]
    assert any(not h.done() for h in handles)  # actually asynchronous
    out = sim.gather(handles, timeout=30)
    assert not out.pending
    assert sorted(h.result for h in handles) == ["ran PLAN0", "ran PLAN1"]
    assert "2 done, 0 pending" in repr(out)
    sim._async_registry.shutdown()


def test_polling_done_forks_queued_launches(monkeypatch):
    """More launches than workers, waited on with done() only.

    Queued jobs are forked only from caller threads, so a poll loop that
    never calls gather must still drive the scheduler.
    """

    def _run(self, plan_text, **kwargs):
        del self, kwargs
        time.sleep(0.1)
        return f"ran {plan_text}"

    workers = 2
    sim = _probe(monkeypatch, workers, _run)
    handles = [sim.run_async(f"PLAN{i}") for i in range(workers + 2)]
    assert sim._async_registry.poll()[2] == 2  # two still queued
    deadline = time.monotonic() + 30
    while not all(h.done() for h in handles):
        assert time.monotonic() < deadline, "queued launches never ran"
        time.sleep(0.02)
    assert sorted(h.result for h in handles) == [
        f"ran PLAN{i}" for i in range(workers + 2)
    ]
    sim._async_registry.shutdown()


def test_child_failure_lands_on_its_handle(monkeypatch):
    """A raising run fails only its own handle; gather names it."""

    def _run(self, plan_text, **kwargs):
        del self, kwargs
        if plan_text == "BAD":
            raise ValueError("boom")
        return "fine"

    sim = _probe(monkeypatch, 2, _run)
    bad = sim.run_async("BAD")
    good = sim.run_async("GOOD")
    out = sim.gather([bad, good], timeout=30)
    assert not out.pending
    assert not bad.ok and "boom" in bad.error
    assert good.ok and good.result == "fine"
    assert "FAILED" in repr(out)
    sim._async_registry.shutdown()


def test_gather_flags_results_from_an_older_model(monkeypatch, tmp_path):
    """Editing simulator.py after launch marks earlier results stale."""
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text("# v1\n")

    def _run(self, plan_text, **kwargs):
        del self, plan_text, kwargs
        return "ok"

    sim = _probe(monkeypatch, 2, _run)
    sim._ctx.sandbox_dir = str(tmp_path)
    h = sim.run_async("PLAN")
    # The agent edits the model while the rollout runs.
    os.utime(sim_file, (time.time() + 100, time.time() + 100))
    out = sim.gather([h], timeout=30)
    assert h in out.stale
    assert "OLDER simulator.py" in repr(out)
    sim._async_registry.shutdown()
