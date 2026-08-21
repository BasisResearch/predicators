"""Capture-gating tests for the ``evaluate_policy`` tool (policy mode).

Mirrors test_evaluate_option_plan_capture.py's fixtures: drives the real
MCP handler with a fake option model, covering the policy-mode gates in
front of ``ctx.solved_policy_source`` - multi-rollout validation with
fresh policy memory per rollout, source snapshotting, the
recovered-option-failure semantics, and the mode gate itself.
"""
import asyncio
import os
from typing import Any

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, Type

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

_GOAL_POLICY = '''
def get_option(state, memory):
    for obj in state:
        if state.get(obj, "x") >= 0.9:
            return None
    return "Move(block0:block)[0.95]"
'''


class _Model:
    """Move sets block.x to its parameter; flaky after N calls."""

    last_execution_failure = None

    def __init__(self, succeed_first_n=10**9):
        self.num_calls = 0
        self._succeed_first_n = succeed_first_n
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        self.num_calls += 1
        nxt = state.copy()
        if self.num_calls <= self._succeed_first_n and len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


def _make_ctx(model, sandbox_dir, best_effort=False):
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    task = Task(init, goal)
    ctx = ToolContext(
        types={_block_type},
        predicates={_ReachedHi},
        processes=set(),
        options={_Move},
        train_tasks=[task],
        example_state=init,
        option_model=model,
        current_task=task,
        sandbox_dir=sandbox_dir,
        log_dir=sandbox_dir,
    )
    ctx.capture_goal_reaching_plans = True
    ctx.capture_best_effort_plan = best_effort
    ctx.policy_capture_mode = True
    return ctx


def _write_policy(sandbox_dir, source):
    path = os.path.join(sandbox_dir, "policy.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    return path


def _call_tool(ctx, extra_args=None):
    tools = {
        t.name: t.handler
        for t in create_mcp_tools(ctx, tool_names=["evaluate_policy"])
    }
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result: Any = loop.run_until_complete(
        tools["evaluate_policy"](extra_args or {}))
    return result["content"][0]["text"]


def _run_tool(model, tmp_path, source=_GOAL_POLICY, rollouts=3,
              best_effort=False, extra_args=None):
    # The local-sandbox path resolution reads <log_dir>/sandbox.
    sandbox = os.path.join(str(tmp_path), "sandbox")
    os.makedirs(sandbox, exist_ok=True)
    utils.reset_config({
        "agent_plan_validation_rollouts": rollouts,
        "agent_solve_policy_mode": True,
        "agent_sdk_use_local_sandbox": True,
    })
    _write_policy(sandbox, source)
    ctx = _make_ctx(model, str(tmp_path), best_effort=best_effort)
    return _call_tool(ctx, extra_args=extra_args), ctx, sandbox


def test_robust_policy_is_captured_with_validation_note(tmp_path):
    model = _Model()
    text, ctx, _ = _run_tool(model, tmp_path, rollouts=3)
    assert "Captured policy.py as the current answer" in text
    assert "Validated 3/3 rollouts" in text
    assert ctx.solved_policy_source is not None
    assert "get_option" in ctx.solved_policy_source
    assert ctx.solved_plan is None
    assert ctx.solved_plan_reached_goal is True
    assert ctx.solved_plan_validation_summary == \
        "validation: 3/3 rollouts ok"


def test_flaky_policy_is_not_captured(tmp_path):
    model = _Model(succeed_first_n=1)
    text, ctx, _ = _run_tool(model, tmp_path, rollouts=3)
    assert "FLAKY (policy NOT captured)" in text
    assert ctx.solved_policy_source is None


def test_flaky_policy_best_effort_captured(tmp_path):
    model = _Model(succeed_first_n=1)
    text, ctx, _ = _run_tool(model, tmp_path, rollouts=3, best_effort=True)
    assert "best-effort" in text
    assert ctx.solved_policy_source is not None
    assert ctx.solved_plan_reached_goal is False


def test_source_snapshot_not_rereading_file(tmp_path):
    """Editing policy.py after the call cannot swap unvalidated code."""
    model = _Model()
    _, ctx, sandbox = _run_tool(model, tmp_path, rollouts=1)
    assert ctx.solved_policy_source is not None
    _write_policy(sandbox, "def get_option(state, memory):\n    return None\n")
    assert "Move(block0:block)" in ctx.solved_policy_source


def test_diagnostic_seed_never_captures(tmp_path):
    model = _Model()
    text, ctx, _ = _run_tool(model, tmp_path, rollouts=1,
                             extra_args={"rollout_seed": 999})
    assert "DIAGNOSTIC rollout at planner seed 999" in text
    assert ctx.solved_policy_source is None


def test_recovered_option_failure_still_captures(tmp_path):
    """Closed-loop: a surfaced-and-recovered failure does not disqualify."""
    model = _Model()
    orig = _Model.get_next_state_and_num_actions

    def _first_call_fails(self, state, option):
        if self.num_calls == 0:
            self.num_calls += 1
            self.last_execution_failure = "simulated failure"
            return state.copy(), 0
        return orig(self, state, option)

    model.get_next_state_and_num_actions = _first_call_fails.__get__(model)
    text, ctx, _ = _run_tool(model, tmp_path, rollouts=1)
    assert "OPTION FAILURE (surfaced to the policy" in text
    assert "Captured policy.py as the current answer" in text
    assert ctx.solved_policy_source is not None


def test_mode_gate_refuses_outside_policy_mode(tmp_path):
    model = _Model()
    sandbox = os.path.join(str(tmp_path), "sandbox")
    os.makedirs(sandbox, exist_ok=True)
    utils.reset_config({
        "agent_solve_policy_mode": False,
        "agent_sdk_use_local_sandbox": True,
    })
    _write_policy(sandbox, _GOAL_POLICY)
    ctx = _make_ctx(model, str(tmp_path))
    ctx.policy_capture_mode = False
    text = _call_tool(ctx)
    assert "only available in policy mode" in text


def test_missing_policy_file_is_instructive(tmp_path):
    model = _Model()
    utils.reset_config({
        "agent_solve_policy_mode": True,
        "agent_sdk_use_local_sandbox": True,
    })
    ctx = _make_ctx(model, str(tmp_path))
    text = _call_tool(ctx)
    assert "No ./policy.py found" in text
