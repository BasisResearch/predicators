"""Tests for the standalone per-call timeout on python exec tools.

The synthesis-session ``run_python`` has no solve ToolContext, so until
``call_timeout_s`` it had NO cap at all: run_20260826_151728's cycle-1
learn spent 2+ hours inside one uncapped grid sweep. The standalone cap
arms only the hard watchdog and returns the call's partial output.
"""

import asyncio
from types import SimpleNamespace

from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool


def _fake_tool(name, description, schema):
    del description, schema

    def deco(fn):
        return SimpleNamespace(name=name, handler=fn)

    return deco


def _text(s):
    return {"content": [{"type": "text", "text": s}]}


def _make(tmp_path, timeout):
    return _make_python_exec_tool(_fake_tool,
                                  name="run_python",
                                  description="d",
                                  exec_ns={},
                                  sandbox_dir=str(tmp_path),
                                  text_result=_text,
                                  call_timeout_s=timeout)


def _run(handler, code):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result = loop.run_until_complete(handler({"code": code}))
    return result["content"][0]["text"]


def test_standalone_timeout_stops_runaway_call(tmp_path):
    """A busy loop is stopped at the cap; printed output survives."""
    t = _make(tmp_path, timeout=1.0)
    out = _run(t.handler, "print('partial result')\nwhile True:\n    pass\n")
    assert "TIME BUDGET" in out
    assert "partial result" in out


def test_fast_call_unaffected_by_cap(tmp_path):
    """A call finishing inside the cap sees no interference."""
    t = _make(tmp_path, timeout=5.0)
    out = _run(t.handler, "print('done quickly')")
    assert "done quickly" in out
    assert "TIME BUDGET" not in out


def test_zero_cap_disables_watchdog(tmp_path):
    """call_timeout_s=0 means no standalone watchdog (legacy behavior)."""
    t = _make(tmp_path, timeout=0.0)
    out = _run(t.handler, "print('unbounded ok')")
    assert "unbounded ok" in out
