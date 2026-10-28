"""Tests for the python exec tool's output observer.

The observer sees each result as the agent is given it, so a harness
check can tell what the agent's own code printed: the rehearsal notes
gate lets a test-level request through only once the notes of its
rehearsal have been in such an output.
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


def _run(handler, code):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result = loop.run_until_complete(handler({"code": code}))
    return result["content"][0]["text"]


def test_the_observer_sees_each_result_as_the_agent_is_given_it(tmp_path):
    """Printed output, an error with its partial output, and the preview of a
    spilled output, which leaves the middle lines out."""
    seen = []
    t = _make_python_exec_tool(_fake_tool,
                               name="run_python",
                               description="d",
                               exec_ns={},
                               sandbox_dir=str(tmp_path),
                               text_result=_text,
                               output_observer=seen.append)
    out = _run(t.handler, "print('NOTE: shown')")
    assert seen == [out] and "NOTE: shown" in out
    out = _run(t.handler, "print('before')\nraise ValueError('x')")
    assert seen[-1] == out and "before" in out and "Error" in out
    out = _run(t.handler,
               "for i in range(3000):\n    print('line %d ' % i + 'x' * 20)")
    assert seen[-1] == out and "too large to inline" in out
    assert "line 1500 " not in out
    assert len(seen) == 3
