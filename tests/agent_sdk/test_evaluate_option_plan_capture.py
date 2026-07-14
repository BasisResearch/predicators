"""Capture-gating tests for the ``evaluate_option_plan`` tool.

Drives the real MCP tool handler with a fake option model (no PyBullet),
covering the two gates in front of ``ctx.solved_plan``:

* multi-rollout validation - the shared sim env is nondeterministic
  across repeats, so a goal-reaching plan is captured only after every
  one of ``CFG.agent_plan_validation_rollouts`` rollouts succeeds; a
  flaky plan is reported to the agent instead of captured;
* task-evaluator legitimacy - a goal-reaching but ``legitimate=False``
  rollout is refused as a reward hack (the real evaluator applies the
  same certificate), while an honest best-effort shortfall
  (``terminated=False``) is captured despite also being
  ``legitimate=False``. The refusal is internal: the agent-facing
  report speaks only in (terminated, reward) terms and never leaks the
  certificate's legitimacy bool or reason string.
"""

import asyncio
from typing import Any

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, TaskEvaluator, Type

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

_PLAN_TEXT = "Move(block0:block)[0.95] -> {ReachedHi(block0:block)}"
# A plan that lands short of the goal (x=0.5 < 0.9), so ReachedHi never
# holds: an honest shortfall, not a reward hack.
_SHORTFALL_PLAN_TEXT = "Move(block0:block)[0.5]"


class _Model:
    """Fake option model: Move sets block.x to its parameter value.

    ``succeed_first_n`` bounds how many calls apply the parameter; later
    calls leave the state unchanged, emulating a flaky plan whose repeat
    rollout misses the goal. Exposes ``last_trajectory`` so evaluator
    verdicts are NON-coarse (a coarse verdict never blocks capture).
    """

    last_execution_failure = None

    def __init__(self, succeed_first_n=10**9):
        self.num_calls = 0
        self._succeed_first_n = succeed_first_n
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if self.num_calls <= self._succeed_first_n and len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


class _StubEvaluator(TaskEvaluator):
    """Deterministic legitimacy verdict."""

    def __init__(self, goal, legit):
        super().__init__(goal)
        self._legit = legit

    def _certify(self, states, step_options):
        if self._legit:
            return True, ""
        return False, "stub: the cascade was staged, not pushed"


def _run_tool(model,
              evaluator=None,
              rollouts=3,
              plan_text=_PLAN_TEXT,
              best_effort=False):
    utils.reset_config({"agent_plan_validation_rollouts": rollouts})
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    task = Task(init, goal, evaluator=evaluator)
    ctx = ToolContext(
        types={_block_type},
        predicates={_ReachedHi},
        processes=set(),
        options={_Move},
        train_tasks=[task],
        example_state=init,
        option_model=model,
        current_task=task,
    )
    ctx.capture_goal_reaching_plans = True
    ctx.capture_best_effort_plan = best_effort
    tools = {
        t.name: t.handler
        for t in create_mcp_tools(ctx, tool_names=["evaluate_option_plan"])
    }
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result: Any = loop.run_until_complete(tools["evaluate_option_plan"]({
        "plan":
        plan_text
    }))
    return result["content"][0]["text"], ctx


def test_robust_plan_is_captured_with_validation_note():
    """All rollouts succeed: captured, with the K/K validation note."""
    model = _Model()
    text, ctx = _run_tool(model, rollouts=3)
    assert "Captured as the current answer" in text
    assert "Validated 3/3 rollouts" in text
    assert ctx.solved_plan is not None
    # One reported rollout + two validation repeats.
    assert model.num_calls == 3


def test_flaky_plan_is_not_captured():
    """A repeat rollout that misses the goal blocks capture, loudly."""
    model = _Model(succeed_first_n=1)
    text, ctx = _run_tool(model, rollouts=3)
    assert "FLAKY (plan NOT captured)" in text
    assert "rollout 2/3 FAILED" in text
    assert "goal not reached" in text
    assert "ReachedHi" in text
    assert "Captured as the current answer" not in text
    assert ctx.solved_plan is None
    # The first rollout DID reach the goal - that is what makes it flaky.
    assert "Goal achieved: True" in text


def test_single_rollout_config_disables_repeats():
    """``agent_plan_validation_rollouts=1`` restores single-rollout capture."""
    model = _Model()
    text, ctx = _run_tool(model, rollouts=1)
    assert "Captured as the current answer" in text
    assert "Validated" not in text
    assert ctx.solved_plan is not None
    assert model.num_calls == 1


def test_illegitimate_plan_is_not_captured_and_skips_repeats():
    """A non-coarse ``legitimate=False`` verdict refuses capture before any
    validation repeats are spent, reported in reward terms only: the
    certificate's reason string never reaches the agent."""
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=False),
                          rollouts=3)
    assert "NOT CAPTURED" in text
    assert "scores it as a non-solve" in text
    assert "stub: the cascade was staged" not in text
    assert "legitimate" not in text
    assert "Captured as the current answer" not in text
    assert ctx.solved_plan is None
    assert model.num_calls == 1


def test_legitimate_plan_passes_both_gates():
    """A legitimate goal-reaching plan validates and captures normally."""
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=True),
                          rollouts=3)
    assert "Captured as the current answer" in text
    assert "Validated 3/3 rollouts" in text
    assert "Task evaluator" in text and "reward=" in text
    assert "legitimate" not in text
    assert ctx.solved_plan is not None
    assert model.num_calls == 3


def test_best_effort_honest_shortfall_is_captured():
    """An honest best-effort shortfall is captured, not refused.

    The plan does not reach the goal (terminated=False), so the
    evaluator marks it legitimate=False - there is no genuine cascade to
    certify. But it is not a reward hack, so under a best-effort
    submission it is captured and executes for its honest reward instead
    of being forfeited.
    """
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=False),
                          rollouts=3,
                          plan_text=_SHORTFALL_PLAN_TEXT,
                          best_effort=True)
    assert "Captured as the current answer" in text
    assert "best-effort: goal NOT reached" in text
    assert "will not count as a solve" in text
    assert "NOT CAPTURED" not in text
    assert ctx.solved_plan is not None
    assert ctx.solved_plan_reached_goal is False
    assert "Goal achieved: False" in text


def test_best_effort_reward_hack_is_still_refused():
    """A best-effort submission does NOT rescue a reward hack: the plan reaches
    the goal atoms (terminated=True) but via an illegitimate route, so it is
    refused even with capture_best_effort_plan set."""
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=False),
                          rollouts=3,
                          best_effort=True)
    assert "NOT CAPTURED" in text
    assert "scores it as a non-solve" in text
    assert "stub: the cascade was staged" not in text
    assert "Captured as the current answer" not in text
    assert ctx.solved_plan is None
