"""Tests for the ``## Trajectory Summary`` query section.

The section is the only outcome feedback an agent without a learn phase
gets about its earlier episodes, so it must say what plan ran and how
the env judged it, not only how many steps it took.
"""
from typing import Sequence

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.sketch_prompts import summarize_trajectories
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)
_Far = Predicate("Far", [_block_type], lambda s, o: s.get(o[0], "x") > 0.5)
_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(0.0, 1.0, (1, )),
    policy=lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
    initiable=lambda s, m, o, p: True,
    terminal=lambda s, m, o, p: True,
)
_Wait = ParameterizedOption(
    "Wait",
    types=[],
    params_space=Box(0.0, 1.0, (0, )),
    policy=lambda s, m, o, p: Action(np.zeros(1, dtype=np.float32)),
    initiable=lambda s, m, o, p: True,
    terminal=lambda s, m, o, p: True,
)


def _state(x: float) -> State:
    return State({_block: np.array([x], dtype=np.float32)})


def _traj(xs: Sequence[float], options: Sequence[str],
          **kwargs) -> LowLevelTrajectory:
    states = [_state(x) for x in xs]
    actions = []
    for name in options:
        act = Action(np.zeros(1, dtype=np.float32))
        if name == "Move":
            act.set_option(_Move.ground([_block], np.array([0.5])))
        elif name == "Wait":
            act.set_option(_Wait.ground([], np.array([])))
        actions.append(act)
    return LowLevelTrajectory(states, actions, _train_task_idx=0, **kwargs)


def _setup() -> None:
    utils.reset_config({"agent_sdk_max_trajectories_in_context": 2})


def test_summary_reports_plan_verdict_and_stable_numbers() -> None:
    """Each recent trajectory shows the option plan it executed (repeats
    collapsed), the env's reward and goal verdict, and keeps its global index
    so a number means the same episode in every session."""
    _setup()
    trajs = [
        _traj([0.0, 0.0], ["Move"], _env_reward=0.0, _env_terminated=False),
        _traj([0.0, 0.2, 0.9], ["Move", "Move"],
              _env_reward=0.0,
              _env_terminated=False),
        _traj([0.0, 0.9, 0.9], ["Move", "Wait"],
              _env_reward=1.0,
              _env_terminated=True),
    ]
    text = summarize_trajectories(trajs, {_Far})
    assert "(3 total, showing last 2)" in text
    assert "Trajectory 0:" not in text
    assert "Trajectory 1: 2 steps" in text
    assert "Executed: Move(block0)\n" in text
    assert "Trajectory 2: 2 steps" in text
    assert "Executed: Move(block0) -> Wait()" in text
    assert "Outcome: env reward 0.00, goal NOT reached" in text
    assert "Outcome: env reward 1.00, goal atoms held at the end" in text
    assert "Gained: Far(block0:block)" in text


def test_summary_falls_back_to_the_task_goal_without_a_verdict() -> None:
    """A trajectory the env never evaluated (no reward) still gets a goal
    verdict from the train task's goal atoms when the tasks are given."""
    _setup()
    task = Task(_state(0.0), {GroundAtom(_Far, [_block])})
    solved = _traj([0.0, 0.9], ["Move"])
    unsolved = _traj([0.0, 0.1], ["Move"])
    text = summarize_trajectories([solved, unsolved], {_Far},
                                  train_tasks=[task])
    assert text.count("Outcome: goal atoms held at the end") == 1
    assert text.count("Outcome: goal NOT reached") == 1
    # Without tasks there is nothing to judge against: no outcome line.
    assert "Outcome" not in summarize_trajectories([solved], {_Far})


def test_summary_without_option_tags_omits_the_plan_line() -> None:
    """Raw-action trajectories (no option on the actions) keep the old step-
    count and atom-delta lines and simply have no plan to show."""
    _setup()
    traj = LowLevelTrajectory([_state(0.0), _state(0.9)],
                              [Action(np.zeros(1, dtype=np.float32))])
    text = summarize_trajectories([traj], {_Far})
    assert "Trajectory 0: 1 steps" in text
    assert "Executed" not in text
    assert "Gained: Far(block0:block)" in text
