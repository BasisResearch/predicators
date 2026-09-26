"""Wait skill factory: holds current pose with finger drift resistance.

This module provides ``create_wait_option``, which builds a
``ParameterizedOption`` that holds the robot's current joint positions
while nudging fingers toward their current open/closed state to resist
drift. A positive ``num_steps`` parameter ends the option after that
many actions. Zero or omitted parameters leave stopping to the executor:
an annotated target atom, an atom change, or the Wait step cap (see
``utils.wait_rollout_step_cap``). The option never inspects the physical
scene, so under observation noise it gives an agent no information its
observations do not.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_wait_option,
    )

    Wait = create_wait_option("Wait", config, robot_type)
"""

from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import SkillConfig
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type


def create_wait_option(
    name: str,
    config: SkillConfig,
    robot_type: Type,
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a wait (no-op) option that holds the robot's current pose.

    The optional integer ``num_steps`` gives an action count, not seconds.
    Positive counts terminate after that many policy actions, unless an
    annotated subgoal or executor cap stops execution sooner.
    Zero (also the default for an empty parameter list) leaves stopping
    to the executor: an annotated subgoal, an atom change, or its cap.
    Fingers are nudged toward their open/closed state to resist drift.

    Args:
        name: Option name (e.g. "Wait").
        config: Shared skill configuration.  See ``SkillConfig``.
        robot_type: The robot ``Type`` object.

    Returns:
        A ``ParameterizedOption`` with ``initiable=True`` always.

    Example::

        wait = create_wait_option("Wait", config, robot_type)
    """
    robot = config.robot
    mid_point = (config.open_fingers_joint + config.closed_fingers_joint) / 2

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects
        count = float(params[0])
        if not np.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError(
                "Wait num_steps must be a finite nonnegative integer")
        memory["wait_num_steps"] = int(count)
        memory["wait_steps_taken"] = 0
        return True

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del state, objects, params
        requested = memory.get("wait_num_steps", 0)
        if requested:
            return memory.get("wait_steps_taken", 0) >= requested
        # Unbounded and annotated waits stop only through the executor.
        return False

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del params
        robot_obj = objects[0]

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
        memory["wait_steps_taken"] = memory.get("wait_steps_taken", 0) + 1
        return Action(
            np.clip(action_arr, robot.action_space.low,
                    robot.action_space.high))

    return ParameterizedOption(
        name,
        types=[robot_type],
        params_space=Box(0, np.inf, (1, )),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
        params_description=params_description
        or ("num_steps: integer action count; 0 or [] waits for the "
            "annotated subgoal or the Wait step cap; subgoals and the cap "
            "can stop a counted wait sooner", ),
        default_params=(0.0, ),
    )
