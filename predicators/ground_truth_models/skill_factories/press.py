"""Press skill factory: a vertical, top-down press on a button, held.

``create_press_skill`` builds a ``ParameterizedOption`` that presses a
momentary button from above and holds it down:

  1. Closing the gripper (the closed fingertips are the press pad).
  2. Moving to an approach point straight above the button, high enough
     that the transit clears whatever stands beside it (a fan, say).
  3. Descending to a hover just above the button top.
  4. The press: descending PAST the button top by ``press_depth`` and
     dwelling there for ``hold_steps`` -- the button is momentary, so the
     dwell is how long it stays on.
  5. Retracting straight up to the approach point.
  6. Opening the gripper.

The press phase carries ``action_extra_info={"segment": "press", ...}`` so
that an executor driving a real arm can recognise it and ship it as a
GUARDED press (force / stall / depth triggered, then a timed hold) instead
of as a plain move: in sim the descent target is where the plunger bottoms
out, on the bench it is a depth cap the guard is expected to beat. See
``real_robot_bridge._split_actions``.

No continuous parameters: where to press is the button's pose, and how
long to hold is a property of the task, not a choice the planner makes.

Example::

    Press = create_press_skill(
        name="Press",
        types=[robot_type, button_type],
        config=config,
        get_button_top_fn=lambda s, o, p, c: (s.get(o[1], "x"),
                                               s.get(o[1], "y"),
                                               s.get(o[1], "z")),
        hold_steps=30,
        hold_seconds=4.0,
    )
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gym.spaces import Box

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# ``(state, objects, params, config) -> (x, y, z_top)`` in world coordinates:
# the centre of the button's face and its height.
ButtonTopFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                       Tuple[float, float, float]]

# The tag the press phase puts on its actions.
PRESS_SEGMENT_TAG = "press"


def create_press_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_button_top_fn: ButtonTopFn,
    hold_steps: int,
    hold_seconds: float,
    approach_above_m: float = 0.12,
    hover_above_m: float = 0.01,
    press_depth_m: float = 0.004,
    pad_below_tcp_m: float = 0.0,
    ee_yaw: float = 0.0,
    extra_tag: Optional[Dict[str, Any]] = None,
) -> ParameterizedOption:
    """Create a press-and-hold skill.

    Args:
        name: Option name.
        types: Ordered object types; the first must be the robot type.
        config: Shared skill configuration.
        get_button_top_fn: Where the button's face is, ``(x, y, z_top)``.
        hold_steps: Policy steps to dwell at the pressed depth (the sim's
            hold; the button is on for this long).
        hold_seconds: The real hold, in seconds, carried on the press
            actions' tag for the executor that ships them to an arm.
        approach_above_m: Height of the approach point above the hover.
            The lateral transit happens at this height; make it clear
            whatever stands beside the button.
        hover_above_m: Height of the pad above the button top before the
            press begins.
        press_depth_m: How far below the button top the press descends
            in sim. For a plunger this is its travel: the descent target
            is where it bottoms out, so the phase terminates by arrival.
        pad_below_tcp_m: How far the pressing surface sits below the
            EE's control point; the targets are raised by this so the PAD
            hovers and presses at the heights above, not the TCP.
        ee_yaw: The EE yaw to press with.
        extra_tag: Extra entries for the press phase's action tag.
    """
    params_space = Box(0, 1, (0,), dtype=np.float32)
    _empty = np.array([], dtype=np.float32)

    def _tcp_z(z_top: float, above: float) -> float:
        return z_top + above + pad_below_tcp_m

    def _make_target(above: float) -> Callable[
            [State, Sequence[Object], Array, SkillConfig],
            Tuple[float, float, float, float]]:

        def _get_target(state: State, objects: Sequence[Object],
                        params: Array, cfg: SkillConfig
                        ) -> Tuple[float, float, float, float]:
            del params
            x, y, z_top = get_button_top_fn(state, objects, _empty, cfg)
            return x, y, _tcp_z(z_top, above), ee_yaw

        return _get_target

    def _close_fingers_target(state: State, objects: Sequence[Object],
                              params: Array,
                              cfg: SkillConfig) -> Tuple[float, float]:
        del params
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(objects[0], "fingers"))
        return current, cfg.closed_fingers_joint - 0.01

    def _open_fingers_target(state: State, objects: Sequence[Object],
                             params: Array,
                             cfg: SkillConfig) -> Tuple[float, float]:
        del params
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(objects[0], "fingers"))
        return current, cfg.open_fingers_joint

    tag: Dict[str, Any] = {
        "segment": PRESS_SEGMENT_TAG,
        "hold_seconds": float(hold_seconds),
    }
    if extra_tag:
        tag.update(extra_tag)

    approach_above = hover_above_m + approach_above_m
    phases: List[Phase] = [
        Phase(name="CloseFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_close_fingers_target,
              finger_direction="close"),
        # Motion-planned: the way to the approach point crosses the scene.
        make_move_to_phase(name="Approach",
                           get_target_pose_fn=_make_target(approach_above),
                           finger_status="closed"),
        # Straight down. These two are strokes, not transits: they must
        # arrive from above, so no planner is allowed to route them.
        make_move_to_phase(name="Hover",
                           get_target_pose_fn=_make_target(hover_above_m),
                           finger_status="closed",
                           use_motion_planning=False),
        make_move_to_phase(name="Press",
                           get_target_pose_fn=_make_target(-press_depth_m),
                           finger_status="closed",
                           expect_contact=True,
                           use_motion_planning=False,
                           dwell_steps=int(hold_steps)),
        make_move_to_phase(name="Retract",
                           get_target_pose_fn=_make_target(approach_above),
                           finger_status="closed",
                           expect_contact=True,
                           use_motion_planning=False),
        Phase(name="OpenFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_open_fingers_target,
              finger_direction="open"),
    ]
    phases[3].action_extra_info = tag

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=(),
                      base_mode="home").build()
