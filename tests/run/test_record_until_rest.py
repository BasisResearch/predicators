"""Tests for recording an ended episode until its bodies come to rest
(``continual_record_until_rest``)."""
from typing import List
from unittest.mock import MagicMock

import numpy as np

from predicators.agent_sdk.tools.digests import render_trajectory_digest
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.run.episode import EpisodeRunner, EpisodeState, \
    _max_object_change
from predicators.structs import Action, LowLevelTrajectory, Object, State, Type

_ROBOT = Type("robot", ["x"])
_BLOCK = Type("block", ["x", "roll"])
_ARM = Object("robot", _ROBOT)
_BOX = Object("box", _BLOCK)


def _state(x: float, roll: float, arm: float = 0.0) -> State:
    return State({_ARM: np.array([arm]), _BOX: np.array([x, roll])})


class _FallingBox:
    """Steps of a stand-in PyBullet env whose box keeps falling for a few steps
    after the episode ends, then rests; it records the actions it was given."""

    def __init__(self, motion: List[float]) -> None:
        self.motion = list(motion)
        self.received: List[Action] = []
        self.x = 0.0
        self.env = MagicMock(spec=PyBulletEnv)
        self.env.step.side_effect = self.step

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Move the box by the next motion increment."""
        del render_obs
        self.received.append(action)
        self.x += self.motion.pop(0) if self.motion else 0.0
        # The arm moves too; only non-robot bodies decide rest.
        return _state(self.x, 0.1 * self.x, arm=float(len(self.received)))


def _ended_runner(env: MagicMock) -> EpisodeRunner:
    runner = EpisodeRunner(env, horizon=None, predicates=set())
    runner._observations = [_state(0.0, 0.0), _state(0.0, 0.0)]  # pylint: disable=protected-access
    runner._actions = [Action(np.array([0.3, 0.7], dtype=np.float32))]  # pylint: disable=protected-access
    runner._episode_state = EpisodeState.WIN  # pylint: disable=protected-access
    return runner


def test_settle_holds_the_last_command_until_the_bodies_rest() -> None:
    """The robot holds its last command until the box rests; the episode keeps
    its verdict and length."""
    box = _FallingBox([0.05, 0.03, 0.01])
    runner = _ended_runner(box.env)
    states, actions = runner.settle(max_steps=50, window=4)
    # 3 moving steps, then 4 quiet ones.
    assert len(states) == len(actions) == 7
    assert all(np.allclose(a.arr, [0.3, 0.7]) for a in box.received)
    assert abs(states[-1].get(_BOX, "x") - 0.09) < 1e-6
    # The episode itself is untouched.
    obs, acts = runner.trajectory()
    assert len(obs) == 2 and len(acts) == 1
    assert runner.episode_state is EpisodeState.WIN


def test_settle_stops_at_its_cap() -> None:
    """A body that never rests stops the recording at the cap."""
    box = _FallingBox([0.01] * 100)
    states, _ = _ended_runner(box.env).settle(max_steps=12, window=4)
    assert len(states) == 12


def test_max_object_change_ignores_the_robot_and_wraps_angles() -> None:
    """A robot's motion is not a body's, and angles wrap at pi."""
    before = _state(0.0, np.pi - 0.01, arm=0.0)
    after = _state(0.0, -np.pi + 0.01, arm=5.0)
    assert abs(_max_object_change(before, after) - 0.02) < 1e-9


def test_digest_labels_the_frames_after_the_episode_ended() -> None:
    """The digest names the frames recorded after the episode ended."""
    states = [_state(0.01 * k, 0.0) for k in range(5)]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(4)]
    traj = LowLevelTrajectory(states, actions, _after_end=2)
    text = render_trajectory_digest([traj], [], [],
                                    0,
                                    include_states=False,
                                    max_timesteps=5)
    assert "The episode ended after action 1." in text
    assert text.count("Action: hold (after the episode ended)") == 2
    assert text.count("Action: primitive (no skill)") == 2
