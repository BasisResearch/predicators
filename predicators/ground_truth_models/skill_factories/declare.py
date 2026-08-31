"""Declare-finished: a skill whose whole effect is that it was run.

The robot says "I am done building" and something in the world reacts.
It moves nothing, touches nothing, and its only trace is a flag on the
actions it emits, which the env reads in its own step.

Why an environment would want this: pressing a physical button is easy
in simulation and awkward on real hardware -- the button has to be
reachable from every staging pose, the arm has to approach it without
sweeping through the scene it just built, and a missed press is
indistinguishable from a press that did not take. A declaration has
none of that, and for a LEARNING agent it is the more interesting
signal anyway: there is no contact to credit the effect to, so an agent
that works out "the wind starts after I declare" has found a genuinely
causal relation rather than a contact one.

Modelled on ``wait.py``'s pose-holding policy, since the arm must stay
exactly where it is: a declaration that nudged the arm could knock the
staged chain, and then the topple would have a mechanical explanation
after all.
"""

from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import SkillConfig
from predicators.structs import DECLARE_FINISHED_KEY, Action, Array, \
    Object, ParameterizedOption, State, Type

# Re-exported so a reader of this file finds the marker here too. An
# env that does not care ignores it, so adding this skill to a domain
# cannot change what any existing env does.
__all__ = ["DECLARE_FINISHED_KEY", "create_declare_option"]


def create_declare_option(
    name: str,
    config: SkillConfig,
    robot_type: Type,
    num_steps: int = 2,
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a no-motion option that announces itself to the env.

    Args:
        name: Option name (e.g. "DeclareFinished").
        config: Shared skill configuration. See ``SkillConfig``.
        robot_type: The robot ``Type`` object.
        num_steps: How many actions to emit. More than one because a
            single action can be swallowed by a controller that has not
            settled, and the flag has to survive into a step the env
            actually runs.

    Returns:
        A ``ParameterizedOption`` that is always initiable and holds the
        robot's pose for ``num_steps`` actions.
    """
    robot = config.robot
    mid_point = (config.open_fingers_joint + config.closed_fingers_joint) / 2

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects, params
        # A grounded option can be re-run (validation rollouts reuse the
        # grounded plan), so the step count must not carry over.
        memory["declare_steps"] = 0
        return True

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del state, objects, params
        return memory.get("declare_steps", 0) >= num_steps

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del params
        memory["declare_steps"] = memory.get("declare_steps", 0) + 1
        robot_obj = objects[0]
        # Hold the fingers where they are, exactly as Wait does: this
        # skill must not disturb the scene it is declaring finished.
        current_joint = config.fingers_state_to_joint(
            robot, state.get(robot_obj, "fingers"))
        if current_joint > mid_point:
            finger_delta = config.finger_action_nudge_magnitude
        else:
            finger_delta = -config.finger_action_nudge_magnitude
        pb_state = cast(utils.PyBulletState, state)
        joint_positions = pb_state.joint_positions.copy()
        f_action = joint_positions[robot.left_finger_joint_idx] + finger_delta
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        n_action = robot.action_space.shape[0]
        if action_arr.shape[0] < n_action:
            action_arr = np.concatenate([
                action_arr,
                np.zeros(n_action - action_arr.shape[0], dtype=np.float32)
            ])
        return Action(np.clip(action_arr, robot.action_space.low,
                              robot.action_space.high),
                      extra_info={DECLARE_FINISHED_KEY: True})

    return ParameterizedOption(
        name,
        types=[robot_type],
        params_space=Box(0, 1, (0, )),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
        params_description=params_description or (),
    )
