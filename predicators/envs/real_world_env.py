"""``RealWorldEnv``: execute any ``PyBulletEnv`` on real hardware.

The env being wrapped is the **twin**. predicators rolls each option out in
simulation exactly as it always has; this wrapper ships the resulting joint
trajectory to the real arm, looks at the bench at option boundaries, and writes
what it saw back into the twin. The wrapped env stays pure simulation and never
learns that a robot exists.

Why the twin has to be corrected, rather than merely handing the agent a
perceived state: the episode loop is ``obs = env.step(act)``, and
``PyBulletEnv.step`` advances *the twin's own physics client* and reads the
observation back out of it. An option is hundreds of low-level actions, and we
look at the bench only between options, so between two looks the twin is the
only thing producing the agent's world state. Writing perception into the twin
is therefore not a step layered on top of perceiving -- it is how perception
reaches the agent at all. Perceive without syncing and the correction is
overwritten by the twin's own simulation on the very next action.

The library's observation type never escapes this module: it is converted to a
``State``, written into the twin, and the agent is handed the *twin's*
observation. ``CogMan`` and the perceiver never see a ``DominoObservation``.

What the wrapper needs from the env it wraps, split by who genuinely owns the
knowledge:

* ``sync_to_state(state) -> None`` and ``gripper_joint_layout()`` -- simulator
  knowledge, defined on ``PyBulletEnv`` and shared by every PyBullet-backed
  real env. Requiring a ``PyBulletEnv`` is what makes these (and the render
  arguments to ``reset``/``step``) available by type rather than by hope.
* ``state_from_observation(obs, prev_state) -> State`` and
  ``task_from_observation(obs, train_or_test) -> EnvironmentTask`` -- domain
  knowledge, which only the domain env can have. No base class can promise
  these, so they are stated as a Protocol and checked at construction, and a
  missing one names itself rather than surfacing mid-episode.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Protocol, Sequence, Set, cast

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.base_env import BaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.real_robot_bridge import execute_chunks, \
    make_real_robot, reset_arm
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Observation, \
    Predicate, State, Type, Video

# The domain conversions the wrapped env must provide, and what each is for.
# Checked at construction so a missing one names itself instead of surfacing as
# an AttributeError several hundred robot-moving actions later.
_REQUIRED_HOOKS = {
    "state_from_observation": "convert a perceived observation into a State",
    "task_from_observation": "build a task from a perceived observation",
}


class _DomainHooks(Protocol):
    """The domain-specific conversions, which no base class can declare.

    Everything else the wrapper calls is on ``PyBulletEnv``. These two
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


class RealWorldEnv(BaseEnv):
    """Executes any ``PyBulletEnv`` on real hardware; the env it wraps stays
    pure sim."""

    def __init__(self, inner: PyBulletEnv, robot: Any) -> None:
        missing = sorted(name for name in _REQUIRED_HOOKS
                         if not callable(getattr(inner, name, None)))
        if missing:
            raise TypeError(
                f"{type(inner).__name__} cannot be driven on real hardware: "
                "it is missing " +
                ", ".join(f"{name}() (to {_REQUIRED_HOOKS[name]})"
                          for name in missing))
        if _observing_at_boundaries() and \
                not getattr(robot, "has_perception", False):
            raise ValueError(
                "real_robot_observe_at_option_boundary is on but the robot "
                "has no perception configured; set real_robot_perception, or "
                "turn the option-boundary look off for a blind open-loop run")
        self._inner = inner
        self._robot = robot
        # Actions rolled out in sim but not yet shipped to the arm.
        self._action_buffer: List[Action] = []
        # Largest per-object position disagreement seen between the twin and
        # the bench at the last look, in metres; None before the first look.
        self.last_divergence: Optional[float] = None
        super().__init__(use_gui=inner.using_gui)
        # Instance-level shadow of the classmethod below. ~25 call sites feed
        # env.get_name() to get_gt_options / get_gt_nsrts, and they must get
        # the INNER env's name -- a wrapped run is still a domino run. The
        # class-level classmethod stays intact for the get_all_subclasses
        # scans; see get_name's docstring.
        self.get_name = inner.get_name  # type: ignore[method-assign]

    # -- identity -----------------------------------------------------------
    @classmethod
    def get_name(cls) -> str:
        """A sentinel name no config requests.

        This must stay a working classmethod returning *something*:
        ``create_new_env`` and four other places pick an env by scanning
        ``utils.get_all_subclasses(BaseEnv)`` and calling
        ``cls.get_name()`` on every concrete subclass. Overriding it as
        an ordinary instance method would make those loops raise and
        break env creation for *every* environment. The instance
        attribute assigned in ``__init__`` shadows this for
        ``env.get_name()`` without touching the class.

        The wrapper is also not a ``PyBulletEnv``, which three of those
        five scans additionally require, so this name only has to avoid
        colliding with a real env's.
        """
        return "real_world_env"

    @property
    def _domain(self) -> _DomainHooks:
        """The wrapped env seen through its domain conversions.

        The same object as ``self._inner``, which is a ``PyBulletEnv``
        for everything else; this view is only for the two hooks no base
        class declares, and ``__init__`` has checked they are there.
        """
        return cast(_DomainHooks, self._inner)

    # -- delegation ---------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """Delegate the long tail to the wrapped env.

        ``__getattr__`` fires only for attributes not found normally, so
        everything defined on this class wins -- which is exactly why
        ``get_name`` needs the explicit treatment above.
        """
        # Guard the pre-__init__ window: an attribute lookup before _inner
        # exists would recurse here forever looking for _inner.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    # BaseEnv's abstract methods have to be real methods, not __getattr__
    # delegation, or this class stays abstract and cannot be instantiated.
    @property
    def predicates(self) -> Set[Predicate]:
        return self._inner.predicates

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self._inner.goal_predicates

    @property
    def types(self) -> Set[Type]:
        return self._inner.types

    @property
    def action_space(self) -> Box:
        return self._inner.action_space

    def simulate(self, state: State, action: Action) -> State:
        """Forward-simulate in the twin.

        Planning-time simulation, not execution: the option model and
        the skill simulator call this to search, and nothing here may
        move the arm.
        """
        return self._inner.simulate(state, action)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        return self._inner.render_state_plt(state, task, action, caption)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._inner.get_train_tasks()

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._inner.get_test_tasks()

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        return self._inner.render_state(state, task, action, caption)

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        return self._inner.render(action, caption)

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        return self._inner.get_event_to_action_fn()

    def goal_reached(self) -> bool:
        return self._inner.goal_reached()

    def get_observation(self) -> Observation:
        """Always the twin's observation.

        The perceived observation is written into the twin and never
        handed out, so the agent sees one representation of the world.
        """
        return self._inner.get_observation()

    # -- execution ----------------------------------------------------------
    def reset(self,
              train_or_test: str,
              task_idx: int,
              render: bool = False) -> Observation:
        """Reset the twin, then home the real arm to match it.

        The twin is reset first because its home joint configuration is
        what the option trajectories are planned from: the arm has to
        start where the first option's streamed waypoints begin, or the
        drift guard trips on the opening move.

        Both splits execute. Real mode is now a property of *being
        wrapped*, not of the split, so an exploration episode drives the
        arm exactly like an evaluation one -- which is what makes
        real-world active learning work without a separate flag.
        """
        self._action_buffer = []
        obs = self._inner.reset(train_or_test, task_idx, render=render)
        if CFG.real_robot_execute:
            reset_arm(self._robot, self._home_arm_joints(obs))
        return obs

    def step(self, action: Action, render_obs: bool = False) -> Observation:
        """Roll the twin forward, and ship at option boundaries.

        The twin is always stepped -- it is the thing that turns an
        option into a joint trajectory. What the real robot sees is a
        chunk of that trajectory, shipped when the option ends.
        """
        obs = self._inner.step(action, render_obs=render_obs)
        if not CFG.real_robot_execute or not action.has_option():
            return obs
        self._action_buffer.append(action)
        if CFG.real_robot_ship_whole_episode:
            # One chunk for the whole episode, flushed by the caller. No
            # mid-episode look: this is the degenerate open-loop case.
            return obs
        if action.get_option().terminal(obs):
            self._ship(self._action_buffer, observe=_observing_at_boundaries())
            self._action_buffer = []
            # The look may have corrected the twin, so re-read it.
            return self.get_observation()
        return obs

    def flush_real_execution(self) -> None:
        """Ship whatever is still buffered (no-op if nothing is).

        Whole-episode mode buffers the entire rollout and relies on
        this; per-option mode reaches it only with a partial trailing
        option, e.g. an episode cut short by the step limit.
        """
        actions, self._action_buffer = self._action_buffer, []
        self._ship(actions, observe=False)

    def _ship(self, actions: Sequence[Action], observe: bool) -> None:
        """Execute ``actions`` on the arm, and absorb anything it saw.

        Always ONE chunk. The buffer is emptied at every option
        boundary, so by construction it holds a single option's actions
        (or, in whole-episode mode, one episode shipped open-loop with
        no look) -- there is never a second option in it to separate
        out. Shipping one option per call is also what makes the closed
        loop a loop: the caller gets control back, and the twin gets
        corrected, before the next option is planned.
        """
        if not actions or not CFG.real_robot_execute:
            return
        observations = execute_chunks(self._robot, [list(actions)],
                                      self._inner.gripper_joint_layout(),
                                      observe=observe,
                                      settle_s=CFG.real_robot_settle_s)
        for observation in observations:
            self._absorb(observation)

    def _absorb(self, observation: Any) -> None:
        """Write what the cameras saw into the twin.

        Divergence is measured against the twin's *pre-sync* state --
        the sim's prediction of where the bench would be -- because that
        is the comparison that says reality went somewhere the model did
        not. ``_set_state``'s own reconstruction check cannot answer
        this: it round-trips the state it was asked to write and
        measures whether PyBullet could realize the request. A toppled
        domino is perfectly realizable, so it never fires on the
        interesting case.
        """
        predicted = self._inner.get_observation()
        assert isinstance(predicted, State)
        perceived = self._domain.state_from_observation(observation, predicted)
        self.last_divergence = _max_position_divergence(predicted, perceived)
        if self.last_divergence is not None and \
                self.last_divergence > CFG.real_robot_divergence_atol:
            logging.warning(
                "real_world_env: the bench is %.3f m from where the twin "
                "predicted (tolerance %.3f m); the twin is being corrected, "
                "but the current plan was made against the prediction",
                self.last_divergence, CFG.real_robot_divergence_atol)
        self._inner.sync_to_state(perceived)
        # No need to refresh the inner env's cached observation by hand:
        # PyBulletEnv.get_observation re-reads the state out of PyBullet, so
        # the next read picks the corrected world up (and re-caches it).

    # -- helpers ------------------------------------------------------------
    def _home_arm_joints(self, obs: Observation) -> List[float]:
        """The twin's home arm joints, fingers dropped.

        ``reset_arm`` takes the 7 arm joints; the twin's joint vector
        also carries the two finger joints, and the layout hook is what
        says which entries those are.

        The joints come from the twin rather than from the robot
        because ``StepReply`` carries observations only -- and they are
        the right source anyway: the arm is about to be commanded to
        exactly the configuration the twin just reset to.
        """
        assert isinstance(obs, utils.PyBulletState), \
            f"the twin must observe joint positions, got {type(obs).__name__}"
        joints = obs.joint_positions
        fingers = set(self._inner.gripper_joint_layout().finger_joint_idxs)
        return [float(v) for i, v in enumerate(joints) if i not in fingers]


def _observing_at_boundaries() -> bool:
    """Whether this run looks at the bench between options.

    Whole-episode shipping has no mid-episode boundary to look at, so it
    silences the look rather than fighting it.
    """
    return bool(CFG.real_robot_observe_at_option_boundary
                and not CFG.real_robot_ship_whole_episode)


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
        features = obj.type.feature_names
        if not {"x", "y", "z"}.issubset(features):
            continue
        delta = np.array(
            [predicted.get(obj, f) - perceived.get(obj, f) for f in "xyz"])
        distance = float(np.linalg.norm(delta))
        worst = distance if worst is None else max(worst, distance)
    return worst


def wrap_for_real_robot(env: BaseEnv, robot: Any = None) -> BaseEnv:
    """Wrap ``env`` for real-robot execution when the config asks for it.

    Returns ``env`` untouched when ``real_robot_execute`` is off, so
    call sites read as one unconditional line.

    **Call this only on the env that is actually executed.** The planner
    builds its own envs -- ``create_option_model`` calls
    ``create_new_env(..., do_cache=False)`` for its private simulator,
    and the shared skill simulator builds one per env class. Wrapping
    inside the env factory would hand the *planner* a real arm: it would
    construct a second ``RealRobot`` and could home the robot in the
    middle of a search.
    """
    if not CFG.real_robot_execute:
        return env
    if not isinstance(env, PyBulletEnv):
        raise TypeError(
            f"real_robot_execute needs a PyBullet-backed env to act as the "
            f"twin, but {CFG.env} is a {type(env).__name__}. The twin is what "
            "turns an option into the joint trajectory the arm executes.")
    if robot is None:
        robot = make_real_robot()
    return RealWorldEnv(env, robot)
