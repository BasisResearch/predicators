"""Tests for fatal-session detection and run termination.

run_20260721_161159 spent 10 online cycles on ~300 one-second queries
that each returned only the "organization has disabled Claude
subscription access" banner: every solve attempt died instantly at
$0.00, and the restart/replan/cycle budgets absorbed them all as
ordinary no-capture attempts. ``query_fatal_error`` recognizes such
responses, ``_track_fatal_response`` counts consecutive ones across
manager instances, and ``AgentSessionFatalError`` (not an
ApproachFailure) propagates past the per-task handlers and ends the run.

The end-to-end test monkeypatches ``ClaudeSDKClient`` with a fake that
yields real SDK dataclasses; no network, no subprocess.
"""
# pylint: disable=protected-access
import asyncio
import datetime
from typing import Any, List
from zoneinfo import ZoneInfo

import claude_agent_sdk
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

import predicators.agent_sdk.session_base as sb
from predicators import utils
from predicators.agent_sdk.session_base import _LIMIT_DEFAULT_WAIT_SECS, \
    _LIMIT_MAX_WAIT_SECS, AgentSessionFatalError, BaseAgentSessionManager, \
    query_fatal_error, usage_limit_wait_seconds
from predicators.agent_sdk.session_manager import AgentSessionManager

# The exact assistant text run_20260721_161159's queries came back with.
_AUTH_BANNER = ("Your organization has disabled Claude subscription access "
                "for Claude Code · Use an Anthropic API key instead, or ask "
                "your admin to enable access")


def _text_entry(text):
    return {"type": "assistant", "content": [{"type": "text", "text": text}]}


def _tool_use_entry():
    return {
        "type":
        "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "run_python",
            "input": {}
        }],
    }


def _result_entry(subtype="success", is_error=False, result=None):
    return {
        "type": "result",
        "subtype": subtype,
        "num_turns": 1,
        "total_cost_usd": 0.0,
        "is_error": is_error,
        "result": result,
    }


def _auth_response():
    return [_text_entry(_AUTH_BANNER), _result_entry()]


# ---------------------------------------------------------------------------
# query_fatal_error
# ---------------------------------------------------------------------------


def test_auth_banner_is_fatal():
    """The observed auth banner (text-only, success result) is fatal."""
    reason = query_fatal_error(_auth_response())
    assert reason == _AUTH_BANNER


def test_other_fatal_banners_matched_case_insensitively():
    """Billing/key banners match regardless of case."""
    assert query_fatal_error([_text_entry("Credit balance is too LOW")])
    assert query_fatal_error([_text_entry("Error: Invalid API key")])


def test_tool_use_gates_out_all_signals():
    """Any tool call means the agent ran: never fatal, even with an error
    result and banner-looking text later in the stream."""
    response = [
        _tool_use_entry(),
        _text_entry(_AUTH_BANNER),
        {
            "type": "error",
            "error": "transport dropped"
        },
        _result_entry(subtype="error_during_execution", is_error=True),
    ]
    assert query_fatal_error(response) is None


def test_turn_cap_result_is_not_fatal():
    """error_max_turns is the ordinary turn-cap budget end."""
    response = [_result_entry(subtype="error_max_turns", is_error=True)]
    assert query_fatal_error(response) is None


def test_error_result_is_fatal():
    """A non-turn-cap error result is fatal, reported with its text."""
    response = [
        _result_entry(subtype="error_during_execution",
                      is_error=True,
                      result="CLI crashed on startup")
    ]
    assert query_fatal_error(response) == \
        "error result: CLI crashed on startup"


def test_stream_error_entry_is_fatal():
    """A stream error before any tool call is fatal."""
    response = [{"type": "error", "error": "scripted SDK failure"}]
    assert query_fatal_error(response) == \
        "stream error: scripted SDK failure"


def test_healthy_responses_are_not_fatal():
    """Ordinary text answers and empty responses pass."""
    assert query_fatal_error(
        [_text_entry("plan: Push(green)"),
         _result_entry()]) is None
    assert query_fatal_error([]) is None


# ---------------------------------------------------------------------------
# _track_fatal_response counter
# ---------------------------------------------------------------------------


def _make_base_manager(tmp_path):
    # reset_config also zeroes the process-wide fatal-query counter.
    utils.reset_config()
    return BaseAgentSessionManager(system_prompt="You are a test agent.",
                                   log_dir=str(tmp_path),
                                   model_name="claude-fable-5")


def test_counter_raises_at_limit(tmp_path):
    """The default limit (3) raises on the third consecutive fatal query."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response(_auth_response())
    assert BaseAgentSessionManager._consecutive_fatal_queries == 2
    with pytest.raises(AgentSessionFatalError,
                       match="disabled Claude subscription access"):
        mgr._track_fatal_response(_auth_response())


def test_healthy_query_resets_counter(tmp_path):
    """One healthy query wipes the streak."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response([_text_entry("fine"), _result_entry()])
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0
    mgr._track_fatal_response(_auth_response())  # streak restarts at 1


def test_counter_shared_across_manager_instances(tmp_path):
    """Fresh-context restarts recreate the manager; the streak survives."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    other = BaseAgentSessionManager(system_prompt="restarted",
                                    log_dir=str(tmp_path),
                                    model_name="claude-fable-5")
    other._track_fatal_response(_auth_response())
    with pytest.raises(AgentSessionFatalError):
        other._track_fatal_response(_auth_response())


def test_limit_zero_disables_check(tmp_path):
    """agent_sdk_max_consecutive_fatal_queries=0 turns the check off."""
    mgr = _make_base_manager(tmp_path)
    utils.reset_config({"agent_sdk_max_consecutive_fatal_queries": 0})
    for _ in range(10):
        mgr._track_fatal_response(_auth_response())
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0


def test_reset_config_zeroes_counter(tmp_path):
    """A config reset (new run/test) must not inherit an old streak."""
    mgr = _make_base_manager(tmp_path)
    mgr._track_fatal_response(_auth_response())
    mgr._track_fatal_response(_auth_response())
    utils.reset_config()
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0


# ---------------------------------------------------------------------------
# End-to-end through the streamed query path
# ---------------------------------------------------------------------------


def _make_fake_client_cls():

    class _FakeClient:
        """Fake ClaudeSDKClient scripted to return the auth banner."""

        instances: List[Any] = []

        def __init__(self, options=None):
            self.options = options
            self.scripts: List[List[Any]] = []

        async def connect(self):
            """No-op connect."""

        async def disconnect(self):
            """No-op disconnect."""

        async def query(self, message):
            """No-op query."""

        async def receive_response(self):
            """Yield the next scripted message batch."""
            for msg in (self.scripts.pop(0) if self.scripts else []):
                yield msg

    def _factory(options=None):
        client = _FakeClient(options)
        _FakeClient.instances.append(client)
        return client

    _factory.instances = _FakeClient.instances  # type: ignore[attr-defined]
    return _factory


def _auth_banner_messages():
    return [
        AssistantMessage(content=[TextBlock(text=_AUTH_BANNER)],
                         model="test-model"),
        ResultMessage(subtype="success",
                      duration_ms=10,
                      duration_api_ms=8,
                      is_error=False,
                      num_turns=1,
                      session_id="sess",
                      total_cost_usd=0.0),
    ]


def test_third_dead_query_raises_through_manager(monkeypatch, tmp_path):
    """Three consecutive auth-banner queries terminate via the real streamed-
    query path (detection is hooked in _run_streamed_query)."""
    fake_cls = _make_fake_client_cls()
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", fake_cls)
    utils.reset_config()
    mgr = AgentSessionManager(system_prompt="You are a test agent.",
                              mcp_server=object(),
                              log_dir=str(tmp_path),
                              model_name="claude-fable-5",
                              allowed_tools=["mcp__predicator_tools__probe"])
    asyncio.run(mgr.start_session())
    client = fake_cls.instances[-1]
    client.scripts = [
        _auth_banner_messages(),
        _auth_banner_messages(),
        _auth_banner_messages(),
    ]
    asyncio.run(mgr.query("first"))
    asyncio.run(mgr.query("second"))
    with pytest.raises(AgentSessionFatalError,
                       match="3 consecutive agent queries"):
        asyncio.run(mgr.query("third"))


# ── usage-limit wait-and-retry ───────────────────────────────────────
# A session/usage-limit banner states its own reset time, so it is a
# transient outage to wait out, not a reason to terminate: the
# 2026-08-27 limit killed six jobs and silently burned two arms'
# cycle-1 learn phases as one-turn $0 "completed" sessions.


def test_limit_wait_parses_reset_time():
    """The wait runs to the banner's stated reset (plus slack)."""
    tz = ZoneInfo("America/New_York")
    now = datetime.datetime(2026, 8, 27, 16, 30, tzinfo=tz).timestamp()
    wait = usage_limit_wait_seconds(
        "You've hit your session limit · resets 5:10pm (America/New_York)",
        now=now)
    # 40 min to 5:10pm + 180 s slack.
    assert wait == pytest.approx(40 * 60 + 180)


def test_limit_wait_wraps_to_next_day():
    """A reset time already past today means tomorrow's occurrence."""
    tz = ZoneInfo("America/New_York")
    now = datetime.datetime(2026, 8, 27, 17, 30, tzinfo=tz).timestamp()
    wait = usage_limit_wait_seconds(
        "You've hit your usage limit · resets 5:10pm (America/New_York)",
        now=now)
    # Tomorrow 5:10pm is ~23.7h away; the wait clamps to the cap.
    assert wait == _LIMIT_MAX_WAIT_SECS


def test_limit_wait_defaults_without_reset_time():
    """A limit banner without a stated reset waits the default period."""
    assert usage_limit_wait_seconds(
        "You've hit your session limit.") == _LIMIT_DEFAULT_WAIT_SECS


def test_non_limit_reasons_do_not_wait():
    """Non-limit fatal reasons return None: no wait-and-retry."""
    assert usage_limit_wait_seconds(None) is None
    assert usage_limit_wait_seconds("invalid api key") is None
    assert usage_limit_wait_seconds(
        "error result: something else broke") is None


def test_run_streamed_query_retries_after_limit(tmp_path, monkeypatch):
    """A limit-dead query is re-issued after the wait; the healthy retry is
    what the caller receives, and the fatal streak never advances."""
    mgr = _make_base_manager(tmp_path)
    limit_resp = [
        _text_entry("You've hit your session limit · resets 5:10pm "
                    "(America/New_York)"),
    ]
    healthy_resp = [_text_entry("fine"), _result_entry()]
    responses = [list(limit_resp), list(healthy_resp)]
    calls = []

    async def _fake_stream(_client, message, **_kwargs):
        calls.append(message)
        return responses.pop(0)

    sleeps = []

    async def _fake_sleep(secs):
        sleeps.append(secs)

    monkeypatch.setattr(sb, "stream_agent_response", _fake_stream)
    monkeypatch.setattr(sb.asyncio, "sleep", _fake_sleep)
    mgr._client = object()
    collected = asyncio.run(
        mgr._run_streamed_query("do the thing", log_path=None, kind="test"))
    assert len(calls) == 2 and calls[0] == calls[1] == "do the thing"
    assert len(sleeps) == 1 and sleeps[0] > 0
    assert collected == healthy_resp
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0


def test_run_streamed_query_polls_after_the_reset(tmp_path, monkeypatch):
    """A limited query is polled every _LIMIT_POLL_SECS regardless of the
    banner's stated reset (the limit can lift earlier), every wait pauses the
    attempt clock by the same amount, and the total wait is capped."""
    mgr = _make_base_manager(tmp_path)
    limit_resp = [
        _text_entry("You've hit your session limit · resets 5:10pm "
                    "(America/New_York)"),
    ]
    healthy_resp = [_text_entry("fine"), _result_entry()]
    responses = [
        list(limit_resp),
        list(limit_resp),
        list(limit_resp),
        list(healthy_resp)
    ]

    async def _fake_stream(_client, message, **_kwargs):
        del message
        return responses.pop(0)

    sleeps = []

    async def _fake_sleep(secs):
        sleeps.append(secs)

    class _Ctx:
        attempt_start = 100.0
        attempt_deadline = 2800.0
        python_call_deadline = None
        paused = 0.0

        def pause_attempt_clock(self, seconds):
            """Shift the armed marks like the real ToolContext does."""
            self.paused += seconds
            self.attempt_start += seconds
            self.attempt_deadline += seconds

    ctx = _Ctx()
    monkeypatch.setattr(sb, "stream_agent_response", _fake_stream)
    monkeypatch.setattr(sb.asyncio, "sleep", _fake_sleep)
    # Refusals take no time here, so only the sleeps are paused.
    monkeypatch.setattr(sb.time, "perf_counter", lambda: 0.0)
    mgr._client = object()
    mgr._tool_context = ctx
    collected = asyncio.run(
        mgr._run_streamed_query("do the thing", log_path=None, kind="test"))
    assert collected == healthy_resp
    assert sleeps == [sb._LIMIT_POLL_SECS] * 3
    assert ctx.paused == sum(sleeps)
    assert ctx.attempt_deadline == 2800.0 + sum(sleeps)
    # Past the total cap the limited response is handed back as-is.
    monkeypatch.setattr(sb, "_LIMIT_MAX_TOTAL_WAIT_SECS", 1.0)
    responses[:] = [list(limit_resp), list(healthy_resp)]
    sleeps.clear()
    collected = asyncio.run(
        mgr._run_streamed_query("again", log_path=None, kind="test"))
    assert collected == limit_resp and not sleeps


def test_run_streamed_query_retries_server_overload(tmp_path, monkeypatch):
    """A query refused with a server-side error (529 Overloaded) is re-issued
    every _OVERLOAD_POLL_SECS with the attempt clock paused, and the healthy
    retry is what the caller receives."""
    mgr = _make_base_manager(tmp_path)
    overloaded = [
        _result_entry(subtype="error",
                      is_error=True,
                      result="API Error: 529 Overloaded. This is a "
                      "server-side issue, usually temporary.")
    ]
    healthy_resp = [_text_entry("fine"), _result_entry()]
    responses = [list(overloaded), list(overloaded), list(healthy_resp)]
    # The SDK takes a while to give up on an overloaded backend (~200 s
    # of internal retries per query in the 2026-09-03 storm); that time
    # did no work and must be paused along with the sleep.
    clock = [0.0]
    refusal_secs = 200.0

    async def _fake_stream(_client, message, **_kwargs):
        del message
        response = responses.pop(0)
        if response is not healthy_resp:
            clock[0] += refusal_secs
        return response

    sleeps = []

    async def _fake_sleep(secs):
        sleeps.append(secs)

    paused = []
    monkeypatch.setattr(sb, "stream_agent_response", _fake_stream)
    monkeypatch.setattr(sb.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(sb.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(mgr, "_pause_attempt_clock", paused.append)
    mgr._client = object()
    collected = asyncio.run(
        mgr._run_streamed_query("learn", log_path=None, kind="test"))
    assert collected == healthy_resp
    assert sleeps == [sb._OVERLOAD_POLL_SECS] * 2
    assert paused == [sb._OVERLOAD_POLL_SECS + refusal_secs] * 2
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0
    assert sb.transient_retry_wait_seconds(
        "error result: invalid api key") is None


def test_run_streamed_query_resumes_a_turn_cut_short(tmp_path, monkeypatch):
    """A turn that already made tool calls and then died on a 5xx is not the
    agent's final answer: it is resumed in the same session with the
    continuation message, its entries kept ahead of the resumed turn's (in the
    returned response and in every flush), and only the sleep is paused since
    the cut-short turn did real work."""
    mgr = _make_base_manager(tmp_path)
    banner = ("API Error: 500 Internal server error. This is a server-side "
              "issue, usually temporary.")
    cut_short = [
        _tool_use_entry(),
        _text_entry(banner),
        _result_entry(subtype="error_during_execution",
                      is_error=True,
                      result=banner),
    ]
    healthy_resp = [_text_entry("done"), _result_entry()]
    assert sb.query_fatal_error(cut_short) is None
    assert sb.transient_turn_error(cut_short) == banner
    assert sb.transient_turn_error(healthy_resp) is None
    # A turn whose LAST assistant block is a tool call ended on the tool,
    # not on a banner, even if an earlier text mentioned one.
    assert sb.transient_turn_error(
        [_text_entry(banner),
         _tool_use_entry(),
         _result_entry()]) is None
    responses = [list(cut_short), list(healthy_resp)]
    sent = []

    async def _fake_stream(_client, message, **kwargs):
        sent.append(message)
        response = responses.pop(0)
        kwargs["flush"](list(response))
        return response

    sleeps = []

    async def _fake_sleep(secs):
        sleeps.append(secs)

    paused = []
    flushed = []
    monkeypatch.setattr(sb, "stream_agent_response", _fake_stream)
    monkeypatch.setattr(sb.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(mgr, "_pause_attempt_clock", paused.append)
    monkeypatch.setattr(mgr, "_flush_log",
                        lambda _path, response: flushed.append(list(response)))
    mgr._client = object()
    collected = asyncio.run(
        mgr._run_streamed_query("solve", log_path="log.md", kind="test"))
    assert sent == ["solve", sb._CONTINUE_AFTER_TRANSIENT_ERROR]
    assert collected == cut_short + healthy_resp
    assert flushed[-1] == cut_short + healthy_resp
    assert flushed[1] == cut_short + healthy_resp
    assert sleeps == [sb._OVERLOAD_POLL_SECS]
    assert paused == sleeps
    assert mgr._conversation_log[-1]["response"] == cut_short + healthy_resp
    assert BaseAgentSessionManager._consecutive_fatal_queries == 0
