"""Scripted controllers for the continual protocol (section 6.7).

A controller plays one level through a ``ProtocolSession`` and returns
when the level is won or when it gives up. The run loop treats a return
without a win as the end of the run (no skipping). Every arm in the
paper is a controller over the same session: the oracle and the random
arms live here, the LLM agent arms implement ``play_level`` themselves.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Collection, Protocol

import numpy as np

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.envs import BaseEnv
from predicators.run.continual import ProtocolSession
from predicators.run.episode import EpisodeState
from predicators.settings import CFG
from predicators.structs import Action, ParameterizedOption


class Controller(Protocol):
    """Anything that can play a level."""

    def play_level(self, session: ProtocolSession) -> None:
        """Play until the level is won, or return to give up."""


class RandomSkillsController:
    """Invoke uniformly random applicable skills until the level is won."""

    def __init__(self, options: Collection[ParameterizedOption],
                 seed: int) -> None:
        self._options = sorted(options, key=lambda o: o.name)
        self._rng = np.random.default_rng(seed)

    def play_level(self, session: ProtocolSession) -> None:
        """Random applicable skills; reset on game over."""
        while True:
            obs = session.observe()
            if obs.state is EpisodeState.WIN:
                return
            if obs.state is EpisodeState.GAME_OVER:
                session.reset("game over")
                continue
            option = utils.sample_applicable_option(self._options, obs.frame,
                                                    self._rng)
            if option is None:
                session.reset("no applicable skill")
                continue
            session.invoke(option)


class RandomPrimitiveController:
    """Step uniformly random primitive actions until the level is won."""

    def __init__(self, env: BaseEnv, seed: int) -> None:
        self._space = env.action_space
        self._space.seed(seed)

    def play_level(self, session: ProtocolSession) -> None:
        """Random primitive actions; reset on game over."""
        while True:
            obs = session.observe()
            if obs.state is EpisodeState.WIN:
                return
            if obs.state is EpisodeState.GAME_OVER:
                session.reset("game over")
                continue
            session.step(Action(self._space.sample()))


class OracleController:
    """Plan with the oracle approach from the current state and execute the
    resulting closed-loop policy; replan on failure, reset on game over."""

    def __init__(self, approach: BaseApproach) -> None:
        self._approach = approach

    def play_level(self, session: ProtocolSession) -> None:
        """Plan from the current state, execute, replan on failure."""
        failures = 0
        while True:
            obs = session.observe()
            if obs.state is EpisodeState.WIN:
                return
            if obs.state is EpisodeState.GAME_OVER:
                session.reset(f"game over: {obs.reason}")
                continue
            task = dataclasses.replace(obs.level.task, init=obs.frame)
            try:
                policy = self._approach.solve(task, timeout=CFG.timeout)
            except (ApproachFailure, ApproachTimeout) as e:
                failures += 1
                logging.info("[Oracle] planning failed (%d/%d): %s", failures,
                             CFG.continual_max_replans_per_level, e)
                if failures >= CFG.continual_max_replans_per_level:
                    return
                session.reset("planning failed")
                continue
            outcome = session.run_policy(policy, note="oracle plan")
            if outcome.status == "failed" or (
                    outcome.status == "succeeded"
                    and outcome.episode_state is EpisodeState.NOT_FINISHED):
                # The plan raised, or ran out without reaching the goal.
                failures += 1
                logging.info(
                    "[Oracle] execution ended without a win "
                    "(%s: %s), replanning (%d/%d)", outcome.status,
                    outcome.reason, failures,
                    CFG.continual_max_replans_per_level)
                if failures >= CFG.continual_max_replans_per_level:
                    return


def create_controller(env: BaseEnv, approach: BaseApproach) -> Any:
    """The controller for ``approach`` under the continual protocol."""
    if hasattr(approach, "play_level"):
        return approach
    name = approach.get_name()
    if name in ("oracle", "oracle_process_planning"):
        return OracleController(approach)
    if name == "random_options":
        options = getattr(approach, "_initial_options", None)
        if not options:
            # pylint: disable-next=import-outside-toplevel
            from predicators.ground_truth_models import get_gt_options
            options = get_gt_options(env.get_name())
        return RandomSkillsController(options, CFG.seed)
    if name == "random_actions":
        return RandomPrimitiveController(env, CFG.seed)
    raise ValueError(f"No continual-protocol controller for approach "
                     f"{name!r}; it must implement play_level().")
