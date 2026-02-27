"""Wait (NoOp) option: holds current joint positions with a finger nudge."""

from typing import Dict, Sequence, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import SkillConfig
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type


def create_wait_option(
    config: SkillConfig,
    robot_type: Type,
    name: str = "NoOp",
) -> ParameterizedOption:
    """Create a wait (no-op) option that holds the robot's current pose.

    Nudges fingers toward their current open/closed state to resist drift,
    keeps all other joints at their current positions, and never terminates.

    Args:
        config: Skill configuration (robot, nudge magnitude, finger joints).
        robot_type: The robot Type object.
        name: Option name (default "NoOp").

    Returns:
        A ParameterizedOption with initiable=True, terminal=False always.
    """
    robot = config.robot
    mid_point = (config.open_fingers_joint + config.closed_fingers_joint) / 2

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory, params
        robot_obj = objects[0]

        current_joint = config.fingers_state_to_joint(
            robot, state.get(robot_obj, "fingers"))
        if current_joint > mid_point:  # currently open — nudge open
            finger_delta = config.finger_action_nudge_magnitude
        else:  # currently closed — nudge closed
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
    )
