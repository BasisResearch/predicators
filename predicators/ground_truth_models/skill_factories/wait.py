"""Wait skill factories: hold the current pose with finger drift resistance.

This module provides ``create_wait_option``, which builds a
``ParameterizedOption`` that holds the robot's current joint positions
while nudging fingers toward their current open/closed state to resist
drift.  The option is always initiable; it never terminates unless the
config sets ``wait_quiescence_eps``, in which case it terminates once
the non-robot scene has stopped moving (see ``SkillConfig``).

``create_timed_hold_option`` holds the same pose but for a *chosen
duration*: it takes the duration in seconds as its single continuous
parameter.  Use it where the wait itself is the decision -- running a
fan for the burst length that lands a ball in a given zone, say --
rather than a settling delay whose length nobody cares about.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_wait_option,
    )

    Wait = create_wait_option("Wait", config, robot_type)
"""

import weakref
from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import SkillConfig
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type, _Option


def note_external_state_change(option: _Option, state: State) -> None:
    """Tell ``option`` that ``state`` was set from outside, not moved into.

    ``Wait`` ends once the scene holds still for several consecutive steps.
    Writing perception into the twin replaces object poses without the
    scene having moved, so counting that jump would zero the tally at
    every look and ``Wait`` would never see the scene settle. This keeps
    the tally and moves the comparison point past the jump, so the jump is
    skipped rather than counted as motion.

    A no-op for options that track no quiescence.
    """
    memory = option.memory
    if "quiescence_prev" not in memory:
        return
    robot_obj = option.objects[0]
    scene_objs = sorted((o for o in state if o != robot_obj), key=str)
    if not scene_objs:
        return
    memory["quiescence_prev"] = state.vec(scene_objs)
    # The cached identity names the pre-resync state, so drop it.
    memory.pop("quiescence_sref", None)


def _hold_pose_action(config: SkillConfig, state: State,
                      robot_obj: Object) -> Action:
    """One action that re-commands the robot's current joint positions.

    The fingers are nudged a hair further in whichever direction they are
    already closest to, so a hold does not drift them open (or closed)
    over a long wait.
    """
    robot = config.robot
    mid_point = (config.open_fingers_joint + config.closed_fingers_joint) / 2

    current_joint = config.fingers_state_to_joint(
        robot, state.get(robot_obj, "fingers"))
    if current_joint > mid_point:  # currently open -- nudge open
        finger_delta = config.finger_action_nudge_magnitude
    else:  # currently closed -- nudge closed
        finger_delta = -config.finger_action_nudge_magnitude

    pb_state = cast(utils.PyBulletState, state)
    joint_positions = pb_state.joint_positions.copy()
    f_action = joint_positions[robot.left_finger_joint_idx] + finger_delta
    joint_positions[robot.left_finger_joint_idx] = f_action
    joint_positions[robot.right_finger_joint_idx] = f_action

    # Pad base-action dims with zeros for mobile robots so the action
    # matches the (arm + base) action space; a no-op for fixed bases.
    action_arr = np.array(joint_positions, dtype=np.float32)
    n_action = robot.action_space.shape[0]
    if action_arr.shape[0] < n_action:
        action_arr = np.concatenate([
            action_arr,
            np.zeros(n_action - action_arr.shape[0], dtype=np.float32)
        ])
    return Action(
        np.clip(action_arr, robot.action_space.low, robot.action_space.high))


def create_timed_hold_option(
    name: str,
    config: SkillConfig,
    types: Sequence[Type],
    action_dt: float,
    max_seconds: float,
    param_description: str = "hold_seconds (how long to hold this pose)",
) -> ParameterizedOption:
    """Create a hold option that runs for a duration given as its parameter.

    The single continuous parameter is a duration in SECONDS; it is
    converted to a step count with ``action_dt``, the wall-clock time one
    env action covers.  Seconds rather than steps because the duration is
    a physical quantity a sampler learns about the world (how long to run
    a fan), not a property of the simulator's step size.

    The elapsed count is incremented in the policy, never in the terminal
    check: ``terminal`` can be consulted more than once for the same
    state (executor loop plus monitors), so counting there would let
    repeated queries stand in for real elapsed time.

    Args:
        name: Option name.
        config: Shared skill configuration.  See ``SkillConfig``.
        types: Object types; the first must be the robot type.
        action_dt: Seconds of simulated time per env action.
        max_seconds: Upper bound of the duration parameter's range.
        param_description: Description of the duration parameter.

    Returns:
        A ``ParameterizedOption`` with a one-dimensional params space.
    """
    assert action_dt > 0.0, "action_dt must be positive"

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects, params
        # A grounded option can be re-run (validation rollouts replay the
        # grounded plan); a stale count would end the new run instantly.
        memory["elapsed_steps"] = 0
        return True

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del state, objects
        target_steps = int(round(float(params[0]) / action_dt))
        return memory.get("elapsed_steps", 0) >= target_steps

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del params
        memory["elapsed_steps"] = memory.get("elapsed_steps", 0) + 1
        return _hold_pose_action(config, state, objects[0])

    return ParameterizedOption(
        name,
        types=list(types),
        params_space=Box(low=np.array([0.0], dtype=np.float64),
                         high=np.array([max_seconds], dtype=np.float64),
                         dtype=np.float64),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
        params_description=(param_description, ),
    )


def create_wait_option(
    name: str,
    config: SkillConfig,
    robot_type: Type,
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a wait (no-op) option that holds the robot's current pose.

    Nudges fingers toward their current open/closed state to resist drift
    and keeps all other joints at their current positions.  With
    ``config.wait_quiescence_eps`` unset the option never terminates
    (the executor's option-rollout cap ends it); when set, it terminates
    once every non-robot object's features have changed by less than the
    eps for ``config.wait_quiescence_steps`` consecutive steps.

    Args:
        name: Option name (e.g. "Wait").
        config: Shared skill configuration.  See ``SkillConfig``.
        robot_type: The robot ``Type`` object.

    Returns:
        A ``ParameterizedOption`` with ``initiable=True`` always.

    Example::

        wait = create_wait_option("Wait", config, robot_type)
    """

    def _scene_objs(state: State,
                    objects: Sequence[Object]) -> Sequence[Object]:
        robot_obj = objects[0]
        return sorted((o for o in state if o != robot_obj), key=str)

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del params
        # A grounded option can be re-run (validation rollouts reuse the
        # grounded plan); stale quiescence tracking from a previous run
        # would terminate the new run instantly.
        memory.pop("quiescence_count", None)
        memory.pop("quiescence_sref", None)
        # Seed the baseline from the state the wait STARTS in, rather than
        # discarding the first observation. Without the seed the first
        # terminal() call has nothing to difference against and must return
        # False, so a wait entered on an already-settled scene cannot end on
        # its first step -- and the option model's stuck detector aborts the
        # rollout on exactly that step, since two consecutive states of a
        # settled scene are allclose. Seeding costs nothing when the scene
        # is still moving and makes "wait for something that already
        # stopped" terminate immediately, which is what it should do.
        scene_objs = _scene_objs(state, objects)
        if scene_objs:
            memory["quiescence_prev"] = state.vec(scene_objs)
        else:
            memory.pop("quiescence_prev", None)
        return True

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del params
        if config.wait_quiescence_eps is None:
            return False
        scene_objs = _scene_objs(state, objects)
        if not scene_objs:
            return False
        # terminal() can be consulted more than once on the same state
        # (executor loop + monitors); recounting a zero delta would let
        # repeated queries stand in for settled physics steps. Identity
        # via weakref, NOT id(): the allocator reuses a freed state's id,
        # which would silently swallow real steps.
        last_ref = memory.get("quiescence_sref")
        if last_ref is not None and last_ref() is state:
            return (memory.get("quiescence_count", 0) >=
                    config.wait_quiescence_steps)
        memory["quiescence_sref"] = weakref.ref(state)
        vec = state.vec(scene_objs)
        prev = memory.get("quiescence_prev")
        memory["quiescence_prev"] = vec
        if prev is None or prev.shape != vec.shape:
            memory["quiescence_count"] = 0
            return False
        if float(np.max(np.abs(vec - prev))) < config.wait_quiescence_eps:
            count = memory.get("quiescence_count", 0) + 1
        else:
            count = 0
        memory["quiescence_count"] = count
        return count >= config.wait_quiescence_steps

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory, params
        return _hold_pose_action(config, state, objects[0])

    return ParameterizedOption(
        name,
        types=[robot_type],
        params_space=Box(0, 1, (0, )),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
        params_description=params_description,
    )
