"""Driving a real arm from a simulated env's rollout.

predicators rolls each option out in simulation exactly as it always has. This
module ships the resulting joint trajectory to the real arm, looks at the
scene, and writes what it saw back into the simulated **twin**. The env stays
pure simulation and never learns that a robot exists: it exposes one optional
collaborator (``PyBulletEnv.attach_executor``) and calls it after each reset
and each step.

Why the twin has to be corrected, rather than merely handing the agent a
perceived state: the episode loop is ``obs = env.step(act)``, and
``PyBulletEnv`` advances *its own physics client* and reads the observation
back out of it. Writing perception into the twin is how perception
reaches the agent at all. If we perceive without syncing, the correction is
overwritten by the twin's own simulation on the very next action.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, cast

import numpy as np

from predicators import utils
from predicators.envs.base_env import BaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.real_robot_bridge import execute_chunks, \
    make_real_robot, reset_arm, reset_env
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Observation, State

# The domain conversions the env must provide, and what each is for. Checked
# when an executor is built, so a missing one names itself instead of surfacing
# as an AttributeError several hundred robot-moving actions later.
_REQUIRED_HOOKS = {
    "state_from_observation": "convert a perceived observation into a State",
    "task_from_observation": "build a task from a perceived observation",
}


class _DomainHooks(Protocol):
    """The domain-specific conversions, which no base class can declare.

    Everything else this module calls is on ``PyBulletEnv``. These two
    are not: turning a perceived observation into a ``State`` or a task
    needs to know what the observation *means*, which is the domain
    env's knowledge alone. Python has no intersection type, so the
    requirement is stated here and checked at construction.
    """

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Correct ``prev_state`` with what was just perceived."""

    def task_from_observation(self, obs: Any,
                              train_or_test: str) -> EnvironmentTask:
        """Build a task from what was just perceived."""


class OptionBoundaryBuffer:
    """Collects actions and hands back one option's worth at its boundary.

    Pure: no robot, no env, no config. An action carrying no option is
    dropped rather than buffered -- there is no boundary to attribute it
    to, and shipping it alone would be motion belonging to no skill.
    """

    def __init__(self) -> None:
        self._actions: List[Action] = []

    def __len__(self) -> int:
        return len(self._actions)

    def add(self, action: Action, obs: Observation) -> Optional[List[Action]]:
        """Buffer ``action``; return a chunk when it ends its option.

        ``obs`` is the state the action led to, which is what the
        option's own ``terminal`` is defined on.
        """
        if not action.has_option():
            return None
        self._actions.append(action)
        if not action.get_option().terminal(obs):
            return None
        chunk, self._actions = self._actions, []
        return chunk

    def discard(self) -> int:
        """Drop whatever is buffered; return how many actions were lost."""
        lost = len(self._actions)
        self._actions = []
        return lost


class TwinCorrector:
    """Writes perception into the simulated twin, and reports divergence."""

    def __init__(self, env: PyBulletEnv, divergence_atol: float) -> None:
        self._env = env
        self._divergence_atol = divergence_atol
        # Largest per-object position disagreement between the twin and the
        # real scene at the last look, in metres; None before the first look.
        self.last_divergence: Optional[float] = None

    def absorb(self, observation: Any) -> Observation:
        """Write what the cameras saw into the twin; return its new reading.

        Divergence is measured against the twin's *pre-sync* state --
        the sim's prediction of where the scene would be -- because that
        is the comparison that says reality went somewhere the model did
        not. ``_set_state``'s own reconstruction check cannot answer
        this: it round-trips the state it was asked to write and
        measures whether PyBullet could realize the request. A toppled
        domino is perfectly realizable, so it never fires on the
        interesting case.
        """
        predicted = self._env.get_observation()
        assert isinstance(predicted, State)
        domain = cast(_DomainHooks, self._env)
        perceived = domain.state_from_observation(observation, predicted)
        self.last_divergence = _max_position_divergence(predicted, perceived)
        if self.last_divergence is not None and \
                self.last_divergence > self._divergence_atol:
            logging.warning(
                "real robot: the scene is %.3f m from where the twin "
                "predicted (tolerance %.3f m); the twin is being corrected, "
                "but the current plan was made against the prediction",
                self.last_divergence, self._divergence_atol)
        self._env.sync_to_state(perceived)
        # No need to refresh the env's cached observation by hand:
        # PyBulletEnv.get_observation re-reads the state out of PyBullet, so
        # this picks the corrected world up (and re-caches it).
        return self._env.get_observation()


class RealRobotExecutor:
    """Ships each option's trajectory to the arm and corrects the twin.

    Implements ``PyBulletEnv``'s ``ActionExecutor`` port. Settings are
    constructor arguments rather than reads of the global config, so a
    test configures one by building it.
    """

    def __init__(self,
                 env: PyBulletEnv,
                 robot: Any,
                 observe_at_boundaries: bool = True,
                 settle_s: float = 0.0,
                 divergence_atol: float = 0.02,
                 human_reset: bool = True) -> None:
        missing = sorted(name for name in _REQUIRED_HOOKS
                         if not callable(getattr(env, name, None)))
        if missing:
            raise TypeError(
                f"{type(env).__name__} cannot be driven on real hardware: "
                "it is missing " +
                ", ".join(f"{name}() (to {_REQUIRED_HOOKS[name]})"
                          for name in missing))
        if observe_at_boundaries and not getattr(robot, "has_perception",
                                                 False):
            raise ValueError(
                "asked to look at the scene between options, but the robot "
                "has no perception configured; set real_robot_perception, or "
                "turn the option-boundary look off for a blind open-loop run")
        if human_reset and not getattr(robot, "has_perception", False):
            raise ValueError(
                "human resets rebuild each episode's task from the scene, so "
                "the robot needs perception; set real_robot_perception, or "
                "turn real_robot_human_reset off to keep the captured scene")
        self._env = env
        self._robot = robot
        self._observe = observe_at_boundaries
        self._settle_s = settle_s
        self._human_reset = human_reset
        self._buffer = OptionBoundaryBuffer()
        self._corrector = TwinCorrector(env, divergence_atol)
        # The scene has to be arranged before the first episode, so a reset is
        # owed from the start. Set again by every reset, i.e. once per episode.
        self._reset_pending = True
        # The twin's home arm configuration, captured now: an executor is
        # attached right after the env is built, so the simulated arm is still
        # at home. A scene reset has to send the real arm somewhere before the
        # human reaches in, and this is that somewhere .
        self._home_arm = self._home_arm_joints(env.get_observation())
        # How many times the scene has been (re)perceived for a task. Read by
        # tests and worth logging on a hardware session.
        self.resets_done = 0

    @property
    def last_divergence(self) -> Optional[float]:
        """How far the scene was from the twin at the last look."""
        return self._corrector.last_divergence

    # -- the ActionExecutor port -------------------------------------------
    def tasks_for(self, train_or_test: str) -> Optional[List[EnvironmentTask]]:
        """Rebuild this split's task from the scene, resetting it first.

        **Why this happens here and not in ``reset``.** The task has to
        be rebuilt before anything consumes it, and the two callers
        consume it differently:

        * *Evaluation* solves at reset. ``_solve_task``
          (``main.py:774``) calls ``cogman.reset(env_task)`` with no
          override policy, so ``_reset_policy`` runs
          ``approach.solve(task)`` -- before ``env.reset`` is ever
          called. A human reset performed in ``env.reset`` would
          therefore plan against a scene that no longer exists.
        * *Exploration* does not solve at reset: the online loop sets
          an override policy first (``main.py:551``), so
          ``_reset_policy`` takes that branch. The task still matters,
          because it is what ``env.reset`` initializes the simulated
          twin from -- a stale one starts every episode from the
          captured scene rather than the one just arranged.

        For a real environment "give me the train task" honestly means
        "look at the scene", so that is what it does.

        Returns None -- leaving the env's captured-scene task alone --
        when no reset is owed, or when human resets are off. The latter
        is what keeps a fixed-plan replay reproducible.
        """
        if not self._human_reset or not self._reset_pending:
            return None
        # Homes the arm out of the way, blocks until the human confirms, then
        # perceives.
        observation = reset_env(self._robot, self._home_arm)
        self._reset_pending = False
        self.resets_done += 1
        logging.info(
            "real robot: scene reset #%d; rebuilding the %s task "
            "from what the cameras see", self.resets_done, train_or_test)
        domain = cast(_DomainHooks, self._env)
        # One task, because a physical scene is one scene. The env keeps the
        # list length stable, so task indices already handed out stay valid.
        return [domain.task_from_observation(observation, train_or_test)]

    def after_reset(self, train_or_test: str, task_idx: int,
                    obs: Observation) -> None:
        """Home the real arm to wherever the twin just reset to.

        The twin is reset first because its home joint configuration is
        what the option trajectories are planned from.

        Both train and test splits execute. Real mode is a property of
        an executor being attached, not of the split.

        This is also where the next reset is owed from: an episode has
        just begun, so the *following* task request has to face a
        freshly arranged scene. Marking it here rather than at the end
        of an episode is what makes it exactly one prompt per episode.
        """
        del train_or_test, task_idx  # every episode homes the same way
        self._reset_pending = True
        lost = self._buffer.discard()
        if lost:
            # The previous episode ended mid-option (step limit, or an
            # exception). Half a skill is not worth executing on the arm, so
            # it is dropped rather than shipped late.
            logging.warning(
                "real robot: dropping %d buffered action(s) from an episode "
                "that ended mid-option; they were never shipped", lost)
        reset_arm(self._robot, self._home_arm_joints(obs))

    def after_step(self, action: Action, obs: Observation) -> Observation:
        """Buffer the action, and ship at an option boundary."""
        chunk = self._buffer.add(action, obs)
        if chunk is None:
            return obs
        observations = execute_chunks(self._robot, [chunk],
                                      self._env.gripper_joint_layout(),
                                      observe=self._observe,
                                      settle_s=self._settle_s)
        for observation in observations:
            obs = self._corrector.absorb(observation)
        return obs

    # -- helpers -----------------------------------------------------------
    def _home_arm_joints(self, obs: Observation) -> List[float]:
        """The twin's home arm joints, fingers dropped.

        ``reset_arm`` takes the 7 arm joints; the twin's joint vector
        also carries the two finger joints, and the layout is what says
        which entries those are.
        """
        assert isinstance(obs, utils.PyBulletState), \
            f"the twin must observe joint positions, got {type(obs).__name__}"
        fingers = set(self._env.gripper_joint_layout().finger_joint_idxs)
        return [
            float(v) for i, v in enumerate(obs.joint_positions)
            if i not in fingers
        ]


def _max_position_divergence(predicted: State,
                             perceived: State) -> Optional[float]:
    """Largest ``(x, y, z)`` distance between the same object in two states.

    Objects missing from either state, or without positional features,
    are skipped; ``None`` means there was nothing comparable.
    """
    worst: Optional[float] = None
    for obj in predicted.data:
        if obj not in perceived.data:
            continue
        if not {"x", "y", "z"}.issubset(obj.type.feature_names):
            continue
        delta = np.array(
            [predicted.get(obj, f) - perceived.get(obj, f) for f in "xyz"])
        distance = float(np.linalg.norm(delta))
        worst = distance if worst is None else max(worst, distance)
    return worst


def attach_real_robot(env: BaseEnv,
                      robot: Any = None) -> Optional[RealRobotExecutor]:
    """Attach a real-robot executor to ``env`` when the config asks for it.

    Returns ``None`` (having done nothing) when ``real_robot_execute``
    is off and the executor otherwise.
    """
    if not CFG.real_robot_execute:
        return None
    if not isinstance(env, PyBulletEnv):
        raise TypeError(
            f"real_robot_execute needs a PyBullet-backed env to act as the "
            f"twin, but {CFG.env} is a {type(env).__name__}. The twin is what "
            "turns an option into the joint trajectory the arm executes.")
    if robot is None:
        robot = make_real_robot()
    executor = RealRobotExecutor(
        env,
        robot,
        observe_at_boundaries=CFG.real_robot_observe_at_option_boundary,
        settle_s=CFG.real_robot_settle_s,
        divergence_atol=CFG.real_robot_divergence_atol,
        human_reset=CFG.real_robot_human_reset)
    env.attach_executor(executor)
    return executor
