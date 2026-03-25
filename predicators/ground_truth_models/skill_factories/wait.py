"""Wait skill factory: holds current pose with finger drift resistance.

This module provides ``create_wait_option``, which builds a
``ParameterizedOption`` that holds the robot's current joint positions
while nudging fingers toward their current open/closed state to resist
drift.  The option is always initiable and never terminates.

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

    Nudges fingers toward their current open/closed state to resist drift,
    keeps all other joints at their current positions, and never terminates.

    Args:
        name: Option name (e.g. "Wait").
        config: Shared skill configuration.  See ``SkillConfig``.
        robot_type: The robot ``Type`` object.

    Returns:
        A ``ParameterizedOption`` with ``initiable=True`` and
        ``terminal=False`` always.

    Example::

        wait = create_wait_option("Wait", config, robot_type)
    """
    robot = config.robot
    mid_point = (config.open_fingers_joint + config.closed_fingers_joint) / 2

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory, params
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

        return Action(
            np.clip(
                np.array(joint_positions, dtype=np.float32),
                robot.action_space.low,
                robot.action_space.high,
            ))

    return ParameterizedOption(
        name,
        types=[robot_type],
        params_space=Box(0, 1, (0, )),
        policy=_policy,
        initiable=lambda _1, _2, _3, _4: True,
        terminal=lambda _1, _2, _3, _4: False,
        params_description=params_description,
    )
