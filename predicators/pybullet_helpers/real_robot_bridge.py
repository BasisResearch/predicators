"""The predicators side of the real-robot interface: a factory that builds the
in-process ``RealRobot``, and the helpers that turn buffered actions into the
segments it executes.

There is no client and no transport. ``RealRobot`` (from the private
``babyrobot`` package) is a plain Python object that predicators constructs and
calls in the same interpreter, so a failure inside it is an ordinary Python
exception with its own traceback.

**babyrobot is optional and must never be imported at module level.**  It ships
as the private git submodule ``submodules/BabyRobotPredicator`` and is
deliberately absent from ``install_requires``, so predicators' CI -- and any
checkout without access to that repo -- has no ``babyrobot`` on the path. Every
import of it here is inside a function body, so ``import predicators.envs``
keeps working without it and a missing submodule surfaces as one clear error at
the moment someone actually asks for real-robot execution.

Scope: open-loop execution. The caller rolls a plan out in sim, buffers the
joint-target actions, and hands them to ``execute_actions``, which splits them
into move / gripper segments and ships them as a single chunk with no
observation requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from predicators.settings import CFG
from predicators.structs import Action, Array

if TYPE_CHECKING:  # pragma: no cover -- typing only; never imported at runtime
    from babyrobot.realrobot.messages import Segment
    from babyrobot.realrobot.real_robot import RealRobot

    from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot

_MISSING_BABYROBOT = (
    "real-robot execution needs the private BabyRobotPredicator package, "
    "which predicators carries as the git submodule "
    "submodules/BabyRobotPredicator. Check it out and install it:\n"
    "    git submodule update --init submodules/BabyRobotPredicator\n"
    "    pip install -e submodules/BabyRobotPredicator")


class MissingBabyRobotError(ImportError):
    """babyrobot is not importable, so no real robot can be constructed."""


@dataclass(frozen=True)
class GripperJointLayout:
    """Where the finger joints sit in an action array, and what open / closed
    finger values look like.

    This is everything the splitting helpers need to read a gripper
    command off a joint-target action, so they depend on four numbers
    rather than on a ``SingleArmPyBulletRobot``.
    """
    left_finger_joint_idx: int
    right_finger_joint_idx: int
    open_fingers: float
    closed_fingers: float

    @property
    def finger_joint_idxs(self) -> Tuple[int, int]:
        """The finger entries, which arm-only waypoints drop."""
        return (self.left_finger_joint_idx, self.right_finger_joint_idx)


def gripper_joint_layout_from_robot(
        robot: "SingleArmPyBulletRobot") -> GripperJointLayout:
    """Read the layout off a pybullet robot."""
    return GripperJointLayout(
        left_finger_joint_idx=robot.left_finger_joint_idx,
        right_finger_joint_idx=robot.right_finger_joint_idx,
        open_fingers=robot.open_fingers,
        closed_fingers=robot.closed_fingers)


def make_real_robot(
        dry: Optional[bool] = None,
        perception: Any = None,
        home_joints: Optional[Sequence[float]] = None) -> "RealRobot":
    """Construct the in-process ``RealRobot``, importing babyrobot lazily.

    ``dry`` defaults to ``CFG.real_robot_dry`` (no arm is built, so arm
    calls are no-ops); ``perception`` defaults to whatever
    ``CFG.real_robot_perception`` names. Raises ``MissingBabyRobotError``
    -- naming the submodule and the install command -- when babyrobot is
    absent, which is the failure a checkout without access hits.
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from babyrobot.realrobot.real_robot import RealRobot as _RealRobot
    except ImportError as e:
        raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
    if dry is None:
        dry = CFG.real_robot_dry
    if perception is None:
        perception = _make_perception()
    return _RealRobot(perception=perception, dry=dry, home_joints=home_joints)


def _make_perception() -> Any:
    """Build the perception source named by ``CFG.real_robot_perception``.

    ``"none"`` (the default) leaves the robot without cameras, which is
    all open-loop execution needs -- it never looks at the scene.
    ``"scene_file"`` replays ``CFG.domino_real_scene``.
    """
    # pylint: disable=import-outside-toplevel,import-error
    kind = CFG.real_robot_perception
    if kind == "none":
        return None
    if kind == "scene_file":
        from babyrobot.realrobot.perception import FileDominoPerception
        return FileDominoPerception(CFG.domino_real_scene)
    raise ValueError(f"unknown real_robot_perception: {kind!r}")


def reset_arm(robot: "RealRobot", joints: Sequence[float]) -> Sequence[float]:
    """Home the arm to ``joints`` and open the gripper (blocking).

    Returns the joint positions the arm reports afterwards.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import ResetArmRequest
    reply = robot.reset_arm(ResetArmRequest(joints=tuple(joints)))
    return reply.joints


def execute_actions(robot: "RealRobot", actions: Sequence[Action],
                    layout: GripperJointLayout) -> None:
    """Split buffered joint-target actions into move / gripper segments and
    execute them on the robot (blocking).

    Open-loop: the whole buffer ships as ONE chunk and no observation is
    requested, so the caller's world state stays the sim's prediction.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import StepRequest
    segments = _split_actions(actions, layout)
    if not segments:
        return
    robot.step(StepRequest(chunks=(tuple(segments), ), observe=False))


class _NoOption:
    """Sentinel for "this action carries no option"; never equal to one."""


_NO_OPTION = _NoOption()


def split_actions_by_option(actions: Sequence[Action]) -> List[List[Action]]:
    """Group a buffered episode into one chunk of actions per option.

    Consecutive actions carrying the same option object form one chunk;
    an action with no option gets a chunk of its own, since there is no
    option boundary to attribute it to. This is the chunking the real-
    world wrapper needs in order to ask for one observation per option
    boundary; open-loop execution ships a single chunk instead.
    """
    chunks: List[List[Action]] = []
    prev: Any = _NO_OPTION
    for action in actions:
        cur: Any = action.get_option() if action.has_option() else _NO_OPTION
        # `is not`, not `!=`: two Pick options with identical parameters are
        # still two separate rollouts, and each is its own boundary.
        if not chunks or cur is _NO_OPTION or cur is not prev:
            chunks.append([])
        chunks[-1].append(action)
        prev = cur
    return chunks


def _split_actions(actions: Sequence[Action],
                   layout: GripperJointLayout) -> List["Segment"]:
    """Joint-target actions -> [Segment(move, waypoints) | Segment(gripper)].

    A finger value nearer ``closed_fingers`` than ``open_fingers`` is a
    ``close``; consecutive same-gripper steps coalesce into one move of
    arm waypoints, with the finger joints removed.

    Stateless: gripper tracking restarts on every call, so a chunk that
    begins already holding an object re-emits its leading ``close``.
    ``RealRobot`` drops that redundant command session-wide, which is what
    makes per-chunk shipping safe.
    """
    # pylint: disable=import-outside-toplevel,import-error
    from babyrobot.realrobot.messages import Segment as _Segment
    gidx = layout.finger_joint_idxs
    closed, opened = layout.closed_fingers, layout.open_fingers

    def grip(arr: Array) -> str:
        v = float(arr[layout.left_finger_joint_idx])
        return "close" if abs(v - closed) < abs(v - opened) else "open"

    def arm_only(arr: Array) -> Tuple[float, ...]:
        return tuple(float(v) for i, v in enumerate(arr) if i not in gidx)

    segments: List["Segment"] = []
    cur_grip: str = ""
    moves: List[Tuple[float, ...]] = []
    for action in actions:
        arr = action.arr
        g = grip(arr)
        if g != cur_grip:
            if moves:
                segments.append(_Segment(type="move", waypoints=tuple(moves)))
                moves = []
            segments.append(_Segment(type="gripper", command=g))
            cur_grip = g
        moves.append(arm_only(arr))
    if moves:
        segments.append(_Segment(type="move", waypoints=tuple(moves)))
    return segments
