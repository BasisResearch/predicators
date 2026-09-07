"""Tests for the solve journal and the wall-clock exploration budgets.

Covers the journal module (entry caps, prompt-injection trimming, the
harness-owned attempt log), the cooperative probe deadline
(:class:`ProbeBudgetExceeded`), and ``run_python``'s budget handling
(refusal after the attempt deadline, per-call timeout with partial
output, ``[budget]`` footer).
"""
# pylint: disable=protected-access
import asyncio
import os
import time
from typing import Any

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.belief_probe import BeliefProbe, ProbeBudgetExceeded
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, Type

_A = journal_mod.ATTEMPTS_FILENAME  # the harness writer's file

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)
_ReachedHi = Predicate("ReachedHi", [_block_type],
                       lambda s, o: s.get(o[0], "x") >= 0.9)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)


class _Model:
    """Fake option model: Move sets block.x to its parameter value."""

    last_execution_failure = None

    def __init__(self):
        self.num_calls = 0
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


def _make_ctx(sandbox_dir=None):
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    task = Task(init, goal)
    return ToolContext(
        types={_block_type},
        predicates={_ReachedHi},
        processes=set(),
        options={_Move},
        train_tasks=[task],
        example_state=init,
        option_model=_Model(),
        current_task=task,
        sandbox_dir=sandbox_dir,
    )


def _call(handler, args) -> str:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result: Any = loop.run_until_complete(handler(args))
    return result["content"][0]["text"]


def _get_tool(ctx, name):
    tools = {t.name: t.handler for t in create_mcp_tools(ctx, [name])}
    assert name in tools, f"{name} not built"
    return tools[name]


# ---------------------------------------------------------------------------
# journal module
# ---------------------------------------------------------------------------


def test_journal_append_and_read(tmp_path):
    """Entries append under headers and read back verbatim."""
    sandbox = str(tmp_path)
    assert journal_mod.read_journal(sandbox) == ""
    assert journal_mod.append_entry(sandbox, "task 0 attempt 1 (auto)",
                                    "- outcome: no capture") is None
    content = journal_mod.read_journal(sandbox,
                                       filename=journal_mod.ATTEMPTS_FILENAME)
    assert "### task 0 attempt 1 (auto)" in content
    assert "- outcome: no capture" in content


def test_journal_entry_truncated_at_cap(tmp_path):
    """Oversize entries are truncated with a notice."""
    sandbox = str(tmp_path)
    note = journal_mod.append_entry(sandbox, "big", "x" * 10000)
    assert note is not None and "truncated" in note
    content = journal_mod.read_journal(sandbox, max_chars=10**6, filename=_A)
    assert "[entry truncated at the per-entry size cap]" in content
    assert len(content) < 5000


def test_journal_read_trims_head_at_entry_boundary(tmp_path):
    """Prompt injection keeps the most recent entries intact."""
    sandbox = str(tmp_path)
    for i in range(20):
        journal_mod.append_entry(sandbox, f"entry {i}",
                                 f"body {i} " + "y" * 500)
    content = journal_mod.read_journal(sandbox, max_chars=2000, filename=_A)
    assert content.startswith("[journal truncated")
    assert "### entry 19" in content
    assert "### entry 0" not in content
    # The kept tail starts at an entry boundary, not mid-entry.
    after_marker = content.split("]\n", 1)[1]
    assert after_marker.startswith("### ")


def test_journal_read_no_sandbox():
    """No sandbox dir reads as empty."""
    assert journal_mod.read_journal(None) == ""


def test_journal_read_raw_and_restore(tmp_path):
    """read_raw snapshots faithfully and restore rolls entries back."""
    sandbox = str(tmp_path)
    assert journal_mod.read_raw(None) is None
    assert journal_mod.read_raw(sandbox, filename=_A) is None
    journal_mod.append_entry(sandbox, "Agent notes (pre-test phase)",
                             "- learning fact")
    snapshot = journal_mod.read_raw(sandbox, filename=_A)
    assert snapshot is not None and "- learning fact" in snapshot
    journal_mod.append_entry(sandbox, "Agent notes (test task 0)",
                             "- test-phase fact")
    journal_mod.restore(sandbox, snapshot, filename=_A)
    assert journal_mod.read_raw(sandbox, filename=_A) == snapshot
    # A None snapshot means no journal file existed: restore deletes.
    journal_mod.restore(sandbox, None, filename=_A)
    assert journal_mod.read_raw(sandbox, filename=_A) is None
    # Deleting an already-absent journal is a no-op, not an error.
    journal_mod.restore(sandbox, None, filename=_A)


# ---------------------------------------------------------------------------
# attempt log (harness-owned file next to the agent's journal)
# ---------------------------------------------------------------------------


def test_attempt_log_is_a_separate_file(tmp_path):
    """Harness entries land in attempts.md; the agent's journal.md is a plain
    file the harness never writes, and each is read, snapshotted and restored
    on its own."""
    sandbox = str(tmp_path)
    assert journal_mod.append_entry(sandbox, "task 0 attempt 1/1 (auto)",
                                    "- outcome: no capture") is None
    assert not os.path.isfile(journal_mod.journal_path(sandbox))
    assert os.path.isfile(journal_mod.attempts_path(sandbox))
    assert journal_mod.read_journal(sandbox) == ""
    attempts = journal_mod.read_journal(sandbox,
                                        filename=journal_mod.ATTEMPTS_FILENAME)
    assert "### task 0 attempt 1/1 (auto)" in attempts
    # The agent writes its journal with the file tools.
    with open(journal_mod.journal_path(sandbox), "w", encoding="utf-8") as f:
        f.write("### task 0 attempt 1\n- tried x=0.5: stopped 3 cm short\n")
    assert "stopped 3 cm short" in journal_mod.read_journal(sandbox)
    snapshot = journal_mod.read_raw(sandbox,
                                    filename=journal_mod.ATTEMPTS_FILENAME)
    journal_mod.append_entry(sandbox, "task 1 attempt 1/1 (auto)",
                             "- outcome: captured")
    journal_mod.restore(sandbox,
                        snapshot,
                        filename=journal_mod.ATTEMPTS_FILENAME)
    assert "task 1" not in journal_mod.read_journal(
        sandbox, filename=journal_mod.ATTEMPTS_FILENAME)
    assert "stopped 3 cm short" in journal_mod.read_journal(sandbox)


# ---------------------------------------------------------------------------
# probe deadline
# ---------------------------------------------------------------------------


def test_probe_raises_after_attempt_deadline():
    """Past the attempt deadline every probe sim call raises."""
    utils.reset_config({})
    ctx = _make_ctx()
    ctx.attempt_deadline = time.monotonic() - 1.0
    sim = BeliefProbe(ctx)
    try:
        sim.reset()
        assert False, "expected ProbeBudgetExceeded"
    except ProbeBudgetExceeded as e:
        assert "submit your single best plan" in str(e)


def test_probe_deadline_skipped_during_best_effort_nudge():
    """The final-submission nudge is never blocked by the spent budget."""
    utils.reset_config({})
    ctx = _make_ctx()
    ctx.attempt_deadline = time.monotonic() - 1.0
    ctx.capture_best_effort_plan = True
    sim = BeliefProbe(ctx)
    sim.reset()  # must not raise


def test_probe_trials_returns_partial_on_mid_loop_budget_expiry():
    """A budget stop mid-trials returns the completed trials (they are minutes
    of sim time living in the return value, not stdout) instead of discarding
    them."""
    utils.reset_config({})
    ctx = _make_ctx()
    ctx.attempt_deadline = time.monotonic() + 60.0
    model = ctx.option_model
    orig = model.get_next_state_and_num_actions

    def _expire_after_rollout(state, option):
        result = orig(state, option)
        ctx.attempt_deadline = time.monotonic() - 1.0
        return result

    model.get_next_state_and_num_actions = _expire_after_rollout
    sim = BeliefProbe(ctx)
    sim.reset()
    res = sim.run("Move(block0:block)[0.95]", render=False, trials=5)
    assert len(res.trials) == 1
    assert res.successes == 1
    assert any("time budget expired after 1/5 trials" in n for n in res.notes)


def test_probe_trials_reraises_when_nothing_completed():
    """With zero completed trials there is nothing to salvage."""
    utils.reset_config({})
    ctx = _make_ctx()
    sim = BeliefProbe(ctx)
    sim.reset()
    ctx.attempt_deadline = time.monotonic() - 1.0
    try:
        sim.run("Move(block0:block)[0.95]", render=False, trials=3)
        assert False, "expected ProbeBudgetExceeded"
    except ProbeBudgetExceeded:
        pass


def test_probe_counts_rollouts():
    """run() meters full-plan rollouts (single and trials)."""
    utils.reset_config({})
    ctx = _make_ctx()
    sim = BeliefProbe(ctx)
    sim.reset()
    sim.run("Move(block0:block)[0.95]", render=False)
    assert ctx.attempt_rollout_count == 1
    sim.reset()
    sim.run("Move(block0:block)[0.95]", render=False, trials=3)
    assert ctx.attempt_rollout_count == 4


# ---------------------------------------------------------------------------
# run_python budgets
# ---------------------------------------------------------------------------


def test_run_python_refuses_after_attempt_deadline(tmp_path):
    """A call arriving past the attempt deadline is refused unrun."""
    utils.reset_config({})
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    ctx.attempt_start = time.monotonic() - 10.0
    ctx.attempt_deadline = time.monotonic() - 1.0
    text = _call(_get_tool(ctx, "run_python"),
                 {"code": "print('should not run')"})
    assert "wall-clock exploration budget" in text
    assert "should not run" not in text
    assert "[budget]" in text


def test_python_call_timeout_returns_partial_output(tmp_path):
    """A per-call timeout stops the sweep and returns printed output."""
    utils.reset_config({
        "agent_sdk_python_call_timeout": 1e-9,
    })
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    ctx.attempt_start = time.monotonic()
    text = _call(_get_tool(ctx, "run_python"),
                 {"code": "print('partial results'); sim.reset()"})
    assert "partial results" in text
    assert "TIME BUDGET" in text
    assert "exceeded its" in text


def test_run_python_budget_footer(tmp_path):
    """Results carry the [budget] footer with rollout deltas."""
    utils.reset_config({
        "agent_sdk_python_call_timeout": 0,
    })
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    ctx.attempt_start = time.monotonic()
    ctx.attempt_deadline = ctx.attempt_start + 2700
    code = "sim.reset(); print(sim.run('Move(block0:block)[0.95]', " \
           "render=False).goal_reached)"
    text = _call(_get_tool(ctx, "run_python"), {"code": code})
    assert "[budget] attempt time" in text
    assert "/45 min" in text
    assert "sim rollouts this attempt: 1 (+1 this call)" in text


def test_run_python_watchdog_stops_sim_free_code(tmp_path):
    """Pure-Python code that never touches the probe is hard-stopped.

    exec() blocks the event loop, so cooperative checks and the sandbox
    interrupt cannot fire - the async-exception watchdog is the only
    preemption that reaches a sim-free loop.
    """
    utils.reset_config({
        "agent_sdk_python_call_timeout": 0.3,
    })
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    code = ("import time\n"
            "print('start')\n"
            "t0 = time.monotonic()\n"
            "while time.monotonic() - t0 < 10:\n"
            "    pass\n"
            "print('done')\n")
    t0 = time.monotonic()
    text = _call(_get_tool(ctx, "run_python"), {"code": code})
    assert time.monotonic() - t0 < 5.0
    assert "start" in text
    assert "TIME BUDGET" in text
    assert "done" not in text


def test_python_call_timeout_exempts_synthesis_sessions(tmp_path):
    """Synthesis probes (candidate simulator; slower rollouts, refits) are
    exempt from the per-call cap."""
    utils.reset_config({
        "agent_sdk_python_call_timeout": 0.1,
    })
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    ctx.probe_option_model_provider = lambda: ctx.option_model
    code = ("import time\n"
            "t0 = time.monotonic()\n"
            "while time.monotonic() - t0 < 0.3:\n"
            "    pass\n"
            "print('done')\n")
    text = _call(_get_tool(ctx, "run_python"), {"code": code})
    assert "done" in text
    assert "TIME BUDGET" not in text


def test_run_python_no_footer_outside_attempt(tmp_path):
    """No attempt in flight (e.g. exploration phase): no footer noise."""
    utils.reset_config({
        "agent_sdk_python_call_timeout": 0,
    })
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    text = _call(_get_tool(ctx, "run_python"), {"code": "print('hi')"})
    assert "[budget]" not in text


# ---------------------------------------------------------------------------
# prompt injection
# ---------------------------------------------------------------------------


def test_solve_prompt_includes_journal_section():
    """build_solve_prompt renders the journal and attempt log contents."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.sketch_prompts import build_solve_prompt
    utils.reset_config({})
    ctx = _make_ctx()
    task = ctx.train_tasks[0]
    journal_text = ("### task 0 attempt 1/3 (auto)\n"
                    "- outcome: no capture")
    prompt = build_solve_prompt(task,
                                all_predicates={_ReachedHi},
                                all_options={_Move},
                                journal="### notes\n- tried x=0.5",
                                attempts=journal_text)
    assert "## Attempt Log" in prompt
    assert "- outcome: no capture" in prompt
    assert "## Solve Journal" in prompt
    assert "- tried x=0.5" in prompt
    assert "./journal.md" in prompt
    assert "record_journal" not in prompt
    # Without journal content the section is absent entirely.
    prompt_no_journal = build_solve_prompt(task,
                                           all_predicates={_ReachedHi},
                                           all_options={_Move})
    assert "## Solve Journal" not in prompt_no_journal
    assert "## Attempt Log" not in prompt_no_journal


def test_read_strategy_absent_present_and_truncated(tmp_path):
    """read_strategy: "" when absent, verbatim when small, head-kept cap."""
    sandbox = str(tmp_path)
    assert journal_mod.read_strategy(sandbox) == ""
    assert journal_mod.read_strategy(None) == ""
    with open(journal_mod.strategy_path(sandbox), "w", encoding="utf-8") as f:
        f.write("## Approach\n- glue both faces\n")
    assert "- glue both faces" in journal_mod.read_strategy(sandbox)
    with open(journal_mod.strategy_path(sandbox), "w", encoding="utf-8") as f:
        f.write("HEADLINE\n" + "x" * 10000)
    content = journal_mod.read_strategy(sandbox)
    assert content.startswith("HEADLINE")
    assert "[strategy truncated at the prompt cap" in content
    assert len(content) < 4300


# ---------------------------------------------------------------------------
# run_python path argument
# ---------------------------------------------------------------------------


def test_run_python_path_runs_a_sandbox_file_in_the_namespace(tmp_path):
    """``path`` executes a sandbox .py file in the same persistent namespace as
    inline ``code``, so helpers developed as files are reusable."""
    utils.reset_config({"agent_sdk_python_call_timeout": 0})
    (tmp_path / "helpers.py").write_text(
        "def double(v):\n    return 2 * v\n\nprint('loaded')\n",
        encoding="utf-8")
    ctx = _make_ctx(sandbox_dir=str(tmp_path))
    tool = _get_tool(ctx, "run_python")
    text = _call(tool, {"path": "helpers.py"})
    assert "loaded" in text
    text = _call(tool, {"code": "print(double(21))"})
    assert "42" in text


def test_run_python_path_stays_inside_the_sandbox(tmp_path):
    """A ``path`` that resolves outside the sandbox is refused unrun, as is a
    call that passes neither or both arguments."""
    utils.reset_config({"agent_sdk_python_call_timeout": 0})
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (tmp_path / "outside.py").write_text("print('escaped')\n",
                                         encoding="utf-8")
    ctx = _make_ctx(sandbox_dir=str(sandbox))
    tool = _get_tool(ctx, "run_python")
    text = _call(tool, {"path": "../outside.py"})
    assert "must stay inside the sandbox" in text
    assert "escaped" not in text
    text = _call(tool, {"path": "missing.py"})
    assert "not a file" in text
    text = _call(tool, {})
    assert "exactly one of" in text
    text = _call(tool, {"code": "print(1)", "path": "missing.py"})
    assert "exactly one of" in text
