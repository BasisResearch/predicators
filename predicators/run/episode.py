"""One episode of real-env interaction, stepped one primitive action at a time,
for the continual protocol.

``EpisodeRunner`` owns exactly what an episode is: a reset into a task,
a sequence of ``env.step`` calls, and the moment the episode ends in a
``WIN`` (the env's evaluator certifies the trajectory) or a
``GAME_OVER`` (horizon, env failure, or a goal that holds through an
illegitimate trajectory). It knows nothing about budgets, scorecards or
recordings; ``run.continual`` layers those on top.

Skill invocations go through :meth:`EpisodeRunner.run_option`, which
reuses ``utils.option_plan_to_policy`` so a skill terminates, fails and
handles ``Wait`` exactly as it does everywhere else in the harness.
"""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.settings import CFG
from predicators.structs import Action, EpisodeEvaluation, GroundAtom, \
    Predicate, State, _Option


class EpisodeState(enum.Enum):
    """The three episode states of the protocol (section 4.3)."""
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


class EpisodeOver(Exception):
    """A step was requested on an episode that has ended."""


@dataclass(frozen=True)
class StepOutcome:
    """What one primitive step produced."""
    observation: State
    state: EpisodeState
    # The game-over reason ("horizon", "env_failure: ...", "rejected: ...")
    # or "" while the episode continues or on a win.
    reason: str


@dataclass(frozen=True)
class InvocationOutcome:
    """What one skill invocation produced (section 5.1)."""
    option: _Option
    # "succeeded": the controller terminated on its own.
    # "failed": the controller raised (non-initiable, timed out, ...).
    # "interrupted": the episode ended (WIN or GAME_OVER) mid-skill.
    status: str
    steps: int
    start_step: int
    end_step: int
    episode_state: EpisodeState
    reason: str
    observation: State

    @property
    def label(self) -> str:
        """The invocation as ``Skill(objs)[params]`` text."""
        return self.option.simple_str()


class EpisodeRunner:
    """Steps one env through episodes and classifies how they end."""

    def __init__(self,
                 env: BaseEnv,
                 horizon: int,
                 max_option_steps: Optional[int] = None,
                 predicates: Optional[Set[Predicate]] = None) -> None:
        self._env = env
        self._horizon = horizon
        self._max_option_steps = max_option_steps
        # Abstraction used by Wait termination and by divergence checks.
        preds = set(env.predicates) if predicates is None else set(predicates)
        self._predicates = preds
        self._observations: List[State] = []
        self._actions: List[Action] = []
        self._episode_state = EpisodeState.GAME_OVER
        self._reason = "not started"
        self._split = ""
        self._task_idx = -1
        self._env_seconds = 0.0
        self._listeners: List[Callable[[Action, StepOutcome], None]] = []

    # -- Accessors -------------------------------------------------------

    @property
    def env(self) -> BaseEnv:
        """The env being stepped."""
        return self._env

    @property
    def predicates(self) -> Set[Predicate]:
        """Predicates used for abstraction during this runner's episodes."""
        return self._predicates

    @predicates.setter
    def predicates(self, preds: Set[Predicate]) -> None:
        self._predicates = set(preds)

    @property
    def episode_state(self) -> EpisodeState:
        """Where the current episode stands."""
        return self._episode_state

    @property
    def reason(self) -> str:
        """The game-over reason, or ""."""
        return self._reason

    @property
    def num_steps(self) -> int:
        """Primitive steps taken in the current episode."""
        return len(self._actions)

    @property
    def env_seconds(self) -> float:
        """Seconds spent inside env calls since construction."""
        return self._env_seconds

    def add_step_listener(
            self, listener: Callable[[Action, StepOutcome], None]) -> None:
        """Call ``listener`` after every applied step (replays included)."""
        self._listeners.append(listener)

    def observation(self) -> State:
        """The latest observation."""
        assert self._observations, "reset() has not been called"
        return self._observations[-1]

    def trajectory(self) -> Tuple[List[State], List[Action]]:
        """The current episode's observations and actions."""
        return list(self._observations), list(self._actions)

    def abstract(self, state: State) -> Set[GroundAtom]:
        """Atoms of ``state`` under this runner's predicates."""
        return utils.abstract(state, self._predicates)

    def evaluate(self) -> EpisodeEvaluation:
        """The env evaluator's verdict on the current episode."""
        return self._env.evaluate_episode(self._observations, self._actions)

    # -- Episode control -------------------------------------------------

    def reset(self, split: str, task_idx: int) -> State:
        """Start an episode on task ``task_idx`` of ``split``."""
        self._split, self._task_idx = split, task_idx
        t0 = time.perf_counter()
        self._env.reset(split, task_idx)
        obs = self._current_observation()
        self._env_seconds += time.perf_counter() - t0
        self._observations = [obs]
        self._actions = []
        self._episode_state = EpisodeState.NOT_FINISHED
        self._reason = ""
        # A goal that already holds at the initial state is a win only
        # if the evaluator certifies the empty trajectory.
        self._check_terminal()
        return obs

    def step(self, action: Action) -> StepOutcome:
        """Apply one primitive action.

        Raises ``EpisodeOver`` after the episode has ended; ``reset``
        opens the next one.
        """
        if self._episode_state is not EpisodeState.NOT_FINISHED:
            raise EpisodeOver(f"episode is {self._episode_state.value} "
                              f"({self._reason}); only reset is valid")
        t0 = time.perf_counter()
        try:
            if isinstance(self._env, PyBulletEnv):
                obs = self._env.step(action, render_obs=CFG.rgb_observation)
            else:
                obs = self._env.step(action)
        except utils.EnvironmentFailure as e:
            # The action is dropped so states stay one longer than
            # actions, as run_episode_and_get_observations does.
            self._env_seconds += time.perf_counter() - t0
            self._end(EpisodeState.GAME_OVER, f"env_failure: {e}")
            return StepOutcome(self.observation(), self._episode_state,
                               self._reason)
        self._env_seconds += time.perf_counter() - t0
        assert isinstance(obs, State)
        self._actions.append(action)
        self._observations.append(obs)
        self._check_terminal()
        outcome = StepOutcome(obs, self._episode_state, self._reason)
        for listener in self._listeners:
            listener(action, outcome)
        return outcome

    def run_option(self, option: _Option) -> InvocationOutcome:
        """One skill invocation: run ``option`` to termination or failure.

        The episode may end mid-skill; the outcome then reports
        ``interrupted`` with the episode state that ended it.
        """
        start = self.num_steps
        if self._episode_state is not EpisodeState.NOT_FINISHED:
            raise EpisodeOver(f"episode is {self._episode_state.value} "
                              f"({self._reason}); only reset is valid")
        policy = utils.option_plan_to_policy(
            [option],
            max_option_steps=self._max_option_steps,
            abstract_function=self.abstract)
        status, reason = "succeeded", ""
        while True:
            obs = self.observation()
            try:
                act = policy(obs)
            except utils.OptionExecutionFailure as e:
                if getattr(e, "info", {}).get("plan_exhausted"):
                    # The single-option plan ran out: the controller
                    # terminated on its own.
                    break
                status, reason = "failed", str(e.args[0]) if e.args else ""
                break
            outcome = self.step(act)
            if outcome.state is not EpisodeState.NOT_FINISHED:
                # The episode ended on this step. If the controller is
                # at its own terminal state the skill still succeeded;
                # otherwise the episode cut it short.
                if not option.terminal(outcome.observation):
                    status, reason = "interrupted", outcome.reason
                break
        end = self.num_steps
        logging.info("[Episode] %s %s after %d steps%s", option.simple_str(),
                     status, end - start, f" ({reason})" if reason else "")
        return InvocationOutcome(option, status, end - start, start,
                                 end, self._episode_state, reason,
                                 self.observation())

    def replay(self, actions: Sequence[Action]) -> StepOutcome:
        """Re-apply ``actions`` after a reset (resume restore)."""
        outcome = StepOutcome(self.observation(), self._episode_state,
                              self._reason)
        for act in actions:
            outcome = self.step(act)
            if outcome.state is not EpisodeState.NOT_FINISHED:
                break
        return outcome

    def finish(self) -> None:
        """Release anything the env holds for an episode that is being
        abandoned (a reset before it ended)."""
        if self._episode_state is EpisodeState.NOT_FINISHED:
            self._env.finish_execution(False)
            self._episode_state = EpisodeState.GAME_OVER
            self._reason = "abandoned"

    # -- Internals -------------------------------------------------------

    def _current_observation(self) -> State:
        if isinstance(self._env, PyBulletEnv):
            obs = self._env.get_observation(render=CFG.rgb_observation)
        else:
            obs = self._env.get_observation()
        assert isinstance(obs, State)
        return obs

    def _check_terminal(self) -> None:
        if self._env.goal_reached():
            ok, why = self._env.check_episode_trajectory(
                self._observations, self._actions)
            if ok:
                self._end(EpisodeState.WIN, "")
            else:
                self._end(EpisodeState.GAME_OVER, f"rejected: {why}")
            return
        if len(self._actions) >= self._horizon:
            self._end(EpisodeState.GAME_OVER, "horizon")

    def _end(self, state: EpisodeState, reason: str) -> None:
        self._episode_state = state
        self._reason = reason
        self._env.finish_execution(state is EpisodeState.WIN)
