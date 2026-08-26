"""Closed-loop execution of agent-written per-task policies.

``agent_solve_policy_mode`` replaces the captured fixed option plan with
a program the solve agent writes to the sandbox (``policy.py``): a
``get_option(state, memory)`` function that returns the NEXT plan line
(same grammar as sketches) from the actual current state, or ``None``
when finished. This module holds the two shared pieces:

* :func:`build_policy_option_fn` - execs the policy source once and
  wraps ``get_option`` into a ``(state, last_failure) -> Optional[
  _Option]`` callable that parses/grounds each returned line against
  the CURRENT state. Owns the per-episode ``memory`` dict, so callers
  build a fresh instance per episode/rollout.
* :func:`execute_policy_forward` - the closed-loop sibling of
  ``plan_execution.execute_plan_forward``, used for belief-model
  validation. The real executor
  (``AgentModelBasedApproach._policy_to_execution_policy``) mirrors its
  semantics step for step, so validation and real execution share one
  behavioral contract:

  - OPTION failures (not initiable, env failure, 0 actions) do NOT end
    the episode: the failure text is surfaced to the policy via
    ``memory["last_failure"]`` and ``get_option`` is asked again from
    the current state - closed-loop recovery (re-place a drifted block,
    re-aim after a motion-planning refusal) is the policy's whole
    point. The total-options cap bounds retry loops, and re-issuing the
    SAME command (option + objects + params) that just failed, for
    ``CFG.agent_policy_max_repeated_failures`` consecutive failures, is
    fatal: an unchanged command fails the same way, so the loop is a
    policy bug, not recovery (the 2026-08-22 policy-arm tests burned
    20+ of their 50 options on one identical colliding PickBlock). The
    no-effect twin is fatal too: re-issuing one identical command that
    keeps COMPLETING with no observable state change, for
    ``CFG.agent_policy_max_repeated_noops`` consecutive completions, is
    a livelock (the 2026-08-26 cycle-6 test spun 10+ options on one
    completed MoveTo to a pose the robot already held).
  - POLICY-code failures (``get_option`` raises, returns a line that
    does not parse or ground) are fatal: they are bugs in the agent's
    program, not recoverable world events.
  - ``None`` ends the episode; whatever progress was made is scored
    honestly by the goal check.
"""
import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators import utils
from predicators.agent_sdk.plan_execution import ForwardResult, StepOutcome
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import ParameterizedOption, Predicate, State, Task, \
    Type, _Option

logger = logging.getLogger(__name__)

# ``(state, last_failure) -> next option or None``. ``last_failure`` is
# the previous step's failure text (None on a clean step); the composer
# delivers it to the policy as ``memory["last_failure"]``.
PolicyOptionFn = Callable[[State, Optional[str]], Optional[_Option]]


class PolicyError(Exception):
    """Fatal error in agent-written policy code (bug, not a world event)."""


def option_repeat_key(option: _Option) -> Tuple[str, Tuple[str, ...], bytes]:
    """Identity of an issued option for stuck-loop detection.

    Two options are "the same command" when they share the option, the
    ground objects, and (to 1e-6) the continuous parameters. Used by
    both executors to detect a policy that keeps re-issuing an
    identical failing command instead of adapting.
    """
    return (option.name, tuple(o.name for o in option.objects),
            np.round(np.asarray(option.params, dtype=float), 6).tobytes())


def repeated_failure_message(option: _Option, count: int) -> str:
    """Shared fatal-loop message for both executors.

    Worded for the policy's author: the guard exists because a command
    that just failed, re-issued unchanged, almost always fails the same
    way, and a policy that does so K times in a row is burning its
    option budget rather than recovering.
    """
    return (f"policy re-issued the same failing option {count} consecutive "
            f"times ({option.simple_str()}) - a policy must adapt after a "
            "surfaced failure: change the parameters or target, try a "
            "different action, or return None (DONE)")


def states_features_allclose(a: State, b: State) -> bool:
    """Feature-level state equality, ignoring any simulator_state.

    ``State.allclose`` refuses states that carry a simulator_state; for
    the no-effect stuck-loop guard only the observable features matter,
    because an option that changed nothing observable gave the policy
    nothing new to react to.
    """
    if sorted(a.data) != sorted(b.data):
        return False
    return all(
        np.allclose(a.data[obj], b.data[obj], atol=1e-3) for obj in a.data)


def repeated_noop_message(option: _Option, count: int) -> str:
    """Shared fatal message for the no-effect twin of the stuck loop.

    A command that completes cleanly but changes nothing observable is a
    no-op from this state (e.g. a MoveTo to a pose the robot already
    holds); a policy re-issuing it unchanged K times is livelocked - the
    2026-08-26 policy-arm cycle-6 test burned 10+ of its 50 options on
    one identical completed MoveTo this way.
    """
    return (f"policy re-issued the same completed option {count} consecutive "
            f"times with no observable state change ({option.simple_str()}) "
            "- the command is a no-op from this state and repeating it "
            "cannot make progress: change the parameters or target, try a "
            "different action, or return None (DONE)")


def build_policy_option_fn(
    source_text: str,
    task: Task,
    *,
    predicates: Set[Predicate],
    options: Set[ParameterizedOption],
    types: Set[Type],
) -> Tuple[Optional[PolicyOptionFn], Optional[str]]:
    """Compose a policy source into a ``PolicyOptionFn``.

    Returns ``(option_fn, None)`` on success or ``(None, error)`` when
    the source fails to load (exec error, missing/uncallable
    ``get_option``) - load errors are agent-facing messages, runtime
    errors raise :class:`PolicyError` from the returned callable.

    The composed callable owns a fresh per-episode ``memory`` dict, so
    build one instance per episode/rollout. Each call passes
    ``state.copy()`` to ``get_option`` (a buggy policy must not corrupt
    the executor's state), parses the returned line strictly against a
    task rooted at the CURRENT state, and grounds it. Helpers available
    inside the policy source: ``np`` and ``atoms(state)`` (the set of
    ground-atom strings under the session's predicates).
    """
    # pylint: disable=import-outside-toplevel
    from predicators.agent_sdk.proposal_exec import build_exec_context, \
        exec_code_safely
    from predicators.agent_sdk.sketch_parsing import parse_sketch_from_text

    # pylint: enable=import-outside-toplevel

    all_types = set(types)
    for opt in options:
        all_types.update(opt.types)
    for pred in predicates:
        all_types.update(pred.types)
    all_types.update(o.type for o in task.init)

    def _atoms(state: State) -> Set[str]:
        return {str(a) for a in utils.abstract(state, predicates)}

    exec_ctx = build_exec_context(types=all_types,
                                  predicates=predicates,
                                  options=options,
                                  extra_context={
                                      "np": np,
                                      "atoms": _atoms,
                                  })
    get_option, err = exec_code_safely(source_text, exec_ctx, "get_option")
    if err is not None:
        return None, f"policy.py failed to load: {err}"
    if not callable(get_option):
        return None, "policy.py defines `get_option` but it is not callable."

    memory: Dict[str, Any] = {}

    def _option_fn(state: State, last_failure: Optional[str]) -> \
            Optional[_Option]:
        memory["last_failure"] = last_failure
        try:
            line = get_option(state.copy(), memory)
        except Exception as e:  # pylint: disable=broad-except
            raise PolicyError(f"get_option raised {type(e).__name__}: "
                              f"{e}") from e
        if line is None:
            return None
        if not isinstance(line, str):
            raise PolicyError(
                f"get_option must return a plan-line string or None, got "
                f"{type(line).__name__}.")
        step_task = dataclasses.replace(task, init=state, evaluator=None)
        try:
            steps = parse_sketch_from_text(line,
                                           step_task,
                                           predicates=predicates,
                                           options=options,
                                           types=all_types,
                                           parse_continuous_params=True,
                                           strict=True,
                                           parse_ground_samplers=False)
        except ValueError as e:
            raise PolicyError(
                f"get_option returned an unparsable line {line!r}: "
                f"{e}") from e
        if len(steps) != 1:
            raise PolicyError(
                f"get_option must return exactly ONE plan line, got "
                f"{len(steps)} in {line!r}.")
        st = steps[0]
        params = (st.initial_params if st.initial_params is not None else
                  np.array([], dtype=np.float32))
        try:
            return st.option.ground(list(st.objects),
                                    np.asarray(params, dtype=np.float32))
        except (AssertionError, ValueError) as e:
            raise PolicyError(
                f"get_option's line {line!r} failed to ground: {e}") from e

    return _option_fn, None


def execute_policy_forward(
    task: Task,
    option_fn: PolicyOptionFn,
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    max_policy_options: int,
    on_step: Optional[Callable[[int, StepOutcome], None]] = None,
) -> ForwardResult:
    """Run a closed-loop policy through the option model.

    Sibling of ``execute_plan_forward`` with the option pulled from
    ``option_fn(state, last_failure)`` each step. See the module
    docstring for the failure-surfacing contract. The returned
    ``ForwardResult`` sets ``policy_error`` when a fatal policy-code
    error ended the episode; option failures land in the per-step
    outcomes without ending it.
    """
    state = task.init
    steps: List[StepOutcome] = []
    first_failure_idx: Optional[int] = None
    total_actions = 0
    goal_step_idx: Optional[int] = None
    actions_to_goal: Optional[int] = None
    policy_error: Optional[str] = None
    last_failure: Optional[str] = None
    repeat_key: Optional[Tuple[str, Tuple[str, ...], bytes]] = None
    repeat_count = 0
    noop_key: Optional[Tuple[str, Tuple[str, ...], bytes]] = None
    noop_count = 0

    for i in range(max_policy_options):
        try:
            option = option_fn(state, last_failure)
        except PolicyError as e:
            policy_error = str(e)
            break
        if option is None:
            break
        pre = state
        initiable = option.initiable(pre)
        post: Optional[State] = None
        num_actions = 0
        failure_reason: Optional[str] = None
        if not initiable:
            failure_reason = "not initiable"
        else:
            try:
                post, num_actions = \
                    option_model.get_next_state_and_num_actions(pre, option)
            except utils.EnvironmentFailure as e:
                failure_reason = f"env failure: {e}"
                post = None
            except Exception as e:  # pylint: disable=broad-except
                failure_reason = f"execution error: {type(e).__name__}: {e}"
                post = None
            else:
                if num_actions == 0:
                    failure_reason = (getattr(option_model,
                                              "last_execution_failure", None)
                                      or "0 actions")

        outcome = StepOutcome(option=option,
                              pre_state=pre,
                              post_state=post,
                              num_actions=num_actions,
                              initiable=initiable,
                              failure_reason=failure_reason,
                              subgoal_missing=None)
        steps.append(outcome)
        if on_step is not None:
            on_step(i, outcome)

        if failure_reason is not None:
            if first_failure_idx is None:
                first_failure_idx = i
            # Surface the failure to the policy and continue from the
            # best available state (a failed option may leave no
            # post-state; the world is then wherever it already was).
            last_failure = f"{option.name}: {failure_reason}"
            key = option_repeat_key(option)
            repeat_count = repeat_count + 1 if key == repeat_key else 1
            repeat_key = key
            if repeat_count >= CFG.agent_policy_max_repeated_failures:
                policy_error = repeated_failure_message(option, repeat_count)
                break
            if post is not None:
                state = post
                total_actions += num_actions
            continue
        last_failure = None
        repeat_key = None
        repeat_count = 0
        state = post  # type: ignore[assignment]  # non-None on clean steps
        total_actions += num_actions
        if goal_step_idx is None and task.goal_holds(state):
            goal_step_idx = i
            actions_to_goal = total_actions
        # No-effect twin of the failure guard: a clean completion that
        # changed nothing observable, re-issued unchanged K times, is a
        # livelock (skipped once the goal holds - lingering there is
        # harmless and DONE is the policy's call).
        if goal_step_idx is None:
            key = option_repeat_key(option)
            if states_features_allclose(pre, state):
                noop_count = noop_count + 1 if key == noop_key else 1
                noop_key = key
                if noop_count >= CFG.agent_policy_max_repeated_noops:
                    policy_error = repeated_noop_message(option, noop_count)
                    break
            else:
                noop_key = None
                noop_count = 0
    else:
        # Loop exhausted without DONE/goal/policy error: the cap is the
        # anti-oscillation bound, so make it an attributable failure.
        if policy_error is None and not task.goal_holds(state):
            policy_error = (
                f"policy exhausted its option budget ({max_policy_options} "
                "options) without signalling DONE or reaching the goal")

    del predicates  # goal check reads the task; kept for API symmetry
    return ForwardResult(
        steps=steps,
        final_state=state,
        goal_reached=task.goal_holds(state),
        first_failure_idx=first_failure_idx,
        first_subgoal_divergence_idx=None,
        total_actions=total_actions,
        goal_step_idx=goal_step_idx,
        actions_to_goal=actions_to_goal,
        policy_error=policy_error,
    )
