"""Domain-general primitive skills: the robot stack and nothing above it.

The composite skill factories (``create_pick_skill``, ``create_push_skill``,
...) bake task knowledge into the controller: where a jug's handle is,
which way a toggle slides, when to let go of a clip. A real arm exposes
none of that. What it exposes is a planned move to a pose, a guarded
straight-line move, a gripper, and time. This module builds exactly that
set, identically in every environment, so the agent supplies the grasp
points, push directions and release moments itself:

- ``MoveTo(robot)[x, y, z, yaw, tilt]`` -- collision-free planned motion
  of the end effector to a world pose, fingers held at their current
  width. Fails (naming the blocker) when the pose is in contact or no
  collision-free path exists.
- ``MoveLinear(robot)[dx, dy, dz, step]`` -- a straight Cartesian stroke
  by a world-frame displacement at ``step`` metres per environment step,
  orientation and fingers held, through contact. Ends at the displaced
  pose, or fails naming the contact when the arm makes no progress.
- ``MoveUntilContact(robot)[dx, dy, dz, step]`` -- the same stroke,
  ending at the FIRST contact of the hand or the held object with any
  other body, or at the displaced pose if nothing is met.
- ``Gripper(robot)[width, force]`` -- open or close the fingers to a
  width in the robot's ``fingers`` feature units under a grip force
  limit in newtons. Ends at the width, or when the fingers stall on an
  object; the simulator's own grasp rule (a pinch while closing)
  attaches the object rigidly, opening detaches it, and the simulated
  grasp ignores the force limit. The parameter is in the signature so
  the same plan lines drive a real gripper, whose command is exactly
  a width and a force.
- ``Wait(robot)[steps]`` -- the shared wait skill.

Poses and displacements are in the world frame, in metres and radians,
so the agent reads them off its (possibly noisy) observations; nothing
here reads a target object's pose out of the true state.

Selected with ``CFG.skill_library = "primitive"``; the default
``"composite"`` keeps each environment's own factory-built skills.
"""

from typing import Any, Dict, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, build_params_space
from predicators.ground_truth_models.skill_factories.place import \
    _held_assembly_in_contact  # pylint: disable=protected-access
from predicators.ground_truth_models.skill_factories.wait import \
    create_wait_option
from predicators.pybullet_helpers.controllers import get_change_fingers_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type

# The recognised values of CFG.skill_library.
SKILL_LIBRARIES = ("composite", "primitive")

# Skill names of the primitive library, in the order the digest lists
# them (alphabetical, as render_options_digest sorts).
PRIMITIVE_SKILL_NAMES = ("Gripper", "MoveLinear", "MoveTo", "MoveUntilContact",
                         "Wait")

# The largest displacement one stroke may ask for, per axis (metres).
_STROKE_MAX_DISPLACEMENT = 0.3
# Per-step Cartesian speed bounds of a stroke (metres per env step). The
# upper bound is the executor's default EE step; the lower bound keeps a
# stroke from taking hundreds of steps for a few centimetres.
_STROKE_STEP_BOUNDS = (0.002, 0.05)
# A contact closer than this (metres, PyBullet contact distance) counts
# as touching; getContactPoints reports near-contacts with small positive
# separations up to the contact processing threshold.
_CONTACT_DIST = 1e-4
# The gripper reports a stall (an object between the pads) after this
# many consecutive policy steps in which the finger joints moved less
# than the tolerance.
_GRIPPER_STALL_STEPS = 3
_GRIPPER_STALL_TOL = 1e-4
# The fingers have arrived when their joint is within this (metres of
# per-finger travel) of the commanded width. Deliberately tighter than
# SkillConfig.grasp_tol, whose ~2 cm linear equivalent is sized for the
# composite grasp phases that over-close past the object and would let a
# width command end a finger's travel short.
_GRIPPER_ARRIVE_TOL = 2e-3
# Grip force limit bounds (newtons): the continuous range of the parallel
# grippers this library targets (a Franka Hand commands up to 70 N).
_GRIPPER_FORCE_BOUNDS = (0.0, 70.0)

_ANGLE_BOUNDS = (-np.pi, np.pi)


def check_skill_library(value: str) -> str:
    """Validate a ``CFG.skill_library`` value and return it."""
    if value not in SKILL_LIBRARIES:
        raise ValueError(f"Unknown skill_library {value!r}; expected one of "
                         f"{SKILL_LIBRARIES}")
    return value


def _current_ee_pose(state: State, robot_obj: Object) -> Pose:
    position = (state.get(robot_obj,
                          "x"), state.get(robot_obj,
                                          "y"), state.get(robot_obj, "z"))
    orientation = p.getQuaternionFromEuler(
        [0, state.get(robot_obj, "tilt"),
         state.get(robot_obj, "wrist")])
    return Pose(position, orientation)


# ---------------------------------------------------------------------------
# MoveTo
# ---------------------------------------------------------------------------


def _move_to_target(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[Pose, Pose, str]:
    del cfg
    robot_obj = objects[0]
    x, y, z, yaw, tilt = (float(v) for v in params[:5])
    current_pose = _current_ee_pose(state, robot_obj)
    target_pose = Pose((x, y, z), p.getQuaternionFromEuler([0, tilt, yaw]))
    return current_pose, target_pose, "hold"


def create_move_to_pose_skill(
    name: str,
    robot_type: Type,
    config: SkillConfig,
    workspace: Sequence[Tuple[float, float]],
) -> ParameterizedOption:
    """A planned, collision-free move of the end effector to a world pose.

    One phase. With ``CFG.skill_phase_use_motion_planning`` the phase
    plans a BiRRT path on the shared simulator and refuses (raising
    ``OptionExecutionFailure`` that names the blocking bodies) a target
    in contact or a target with no collision-free path; without it the
    phase steps incremental IK straight at the target under the stall
    abort. The fingers keep their current width throughout.

    Args:
        name: Option name.
        robot_type: The robot type (the option's only argument).
        config: Shared skill configuration.
        workspace: ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`` bounds
            of the target position, in metres.
    """
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = workspace
    params_space, params_description = build_params_space([
        ("x (world x of the end effector, metres)", x_lo, x_hi),
        ("y (world y of the end effector, metres)", y_lo, y_hi),
        ("z (world z of the end effector, metres)", z_lo, z_hi),
        ("yaw (end-effector yaw about world z, radians; the robot's wrist "
         "feature reads the current value)", *_ANGLE_BOUNDS),
        ("tilt (end-effector pitch, radians; the robot's tilt feature "
         "reads the current value, keep it for a level move)", *_ANGLE_BOUNDS),
    ])
    phase = Phase(
        name="Move",
        action_type=PhaseAction.MOVE_TO_POSE,
        target_fn=_move_to_target,
        # A goal configuration from unvalidated IK can sit a few
        # millimetres into the very body the agent is reaching for and
        # get the whole move refused; validate the goal solve.
        validate_ik=True,
    )
    return PhaseSkill(name, [robot_type],
                      params_space,
                      config, [phase],
                      params_description=params_description).build()


# ---------------------------------------------------------------------------
# MoveLinear / MoveUntilContact
# ---------------------------------------------------------------------------


def _stroke_target(
    state: State,
    objects: Sequence[Object],
    params: Array,
    cfg: SkillConfig,
) -> Tuple[Pose, Pose, str]:
    del cfg
    robot_obj = objects[0]
    dx, dy, dz = (float(v) for v in params[:3])
    current_pose = _current_ee_pose(state, robot_obj)
    cx, cy, cz = current_pose.position
    target_pose = Pose((cx + dx, cy + dy, cz + dz), current_pose.orientation)
    return current_pose, target_pose, "hold"


def _stroke_step(params: Array) -> float:
    return float(params[3])


def _stroke_params_space() -> Tuple[Box, Tuple[str, ...]]:
    lo, hi = -_STROKE_MAX_DISPLACEMENT, _STROKE_MAX_DISPLACEMENT
    return build_params_space([
        ("dx (world x displacement of the end effector, metres)", lo, hi),
        ("dy (world y displacement, metres)", lo, hi),
        ("dz (world z displacement, metres)", lo, hi),
        ("step (metres travelled per environment step; small for a "
         "gentle touch)", *_STROKE_STEP_BOUNDS),
    ])


def _make_stroke_phase() -> Phase:
    return Phase(
        name="Stroke",
        action_type=PhaseAction.MOVE_TO_POSE,
        target_fn=_stroke_target,
        # A stroke steps IK straight along its line: a collision-free
        # planner asked for a pose at or inside a body either refuses
        # or arrives by a detour from the wrong side, and the direction
        # of travel is the whole point of a stroke.
        use_motion_planning=False,
        # Contact is the expected outcome, not a planning defect.
        expect_contact=True,
        # The displacement is relative to where the stroke STARTS, not
        # to wherever the hand is each step.
        freeze_target=True,
        # A gentle stroke: the per-step clamp is the ``step`` parameter,
        # and the executor arms its joint-jump guard and stall abort.
        step_norm_fn=_stroke_step,
    )


def _hand_in_contact(state: State, config: SkillConfig) -> bool:
    """True when a finger or the end-effector link touches a body other than
    the robot itself or an object it is holding."""
    sim_state = getattr(state, "simulator_state", None)
    if not isinstance(sim_state, dict):
        return False
    client = sim_state.get("physics_client_id")
    robot_id = sim_state.get("robot_id")
    if client is None or robot_id is None:
        return False
    held_ids = set()
    for obj in state:
        if "is_held" in obj.type.feature_names and \
                state.get(obj, "is_held") > 0.5:
            body_id = getattr(obj, "id", None)
            if body_id is not None:
                held_ids.add(body_id)
    robot = config.robot
    hand_links = {
        robot.end_effector_id, robot.left_finger_id, robot.right_finger_id
    }
    for cp in p.getContactPoints(bodyA=robot_id, physicsClientId=client):
        if cp[3] not in hand_links:
            continue
        other = cp[2]
        if other == robot_id or other in held_ids:
            continue
        if cp[8] < _CONTACT_DIST:
            return True
    return False


def stroke_in_contact(state: State, config: SkillConfig) -> bool:
    """True when the hand, or the held assembly, touches another body."""
    return _hand_in_contact(state, config) or _held_assembly_in_contact(state)


class _GuardedStrokeSkill(PhaseSkill):
    """A stroke whose phase also ends at the first external contact."""

    def _phase_is_terminal(self, phase: Phase, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        if stroke_in_contact(state, self._config):
            return True
        return super()._phase_is_terminal(phase, state, memory, objects,
                                          params)


def create_move_linear_skill(name: str, robot_type: Type,
                             config: SkillConfig) -> ParameterizedOption:
    """A straight Cartesian stroke by a world displacement, through contact.

    The stroke ends at the displaced pose. When the arm stops making
    progress toward it (a body in the way, a joint limit) the option
    fails with ``OptionExecutionFailure`` naming the contact, after the
    executor's stall window.
    """
    params_space, params_description = _stroke_params_space()
    return PhaseSkill(name, [robot_type],
                      params_space,
                      config, [_make_stroke_phase()],
                      params_description=params_description).build()


def create_move_until_contact_skill(
        name: str, robot_type: Type,
        config: SkillConfig) -> ParameterizedOption:
    """A guarded straight stroke: ends at the first external contact.

    Contact means a finger or the end-effector link touching a body
    other than the robot or the held object, or the held object (with
    anything rigidly attached to it) touching a body outside that
    assembly. If the stroke starts in contact it ends at once. Without
    any contact it ends at the displaced pose like ``MoveLinear``.
    """
    params_space, params_description = _stroke_params_space()
    return _GuardedStrokeSkill(name, [robot_type],
                               params_space,
                               config, [_make_stroke_phase()],
                               params_description=params_description).build()


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------


def create_gripper_skill(
    name: str,
    robot_type: Type,
    config: SkillConfig,
    finger_state_bounds: Tuple[float, float],
) -> ParameterizedOption:
    """Open or close the fingers to a width under a grip force limit.

    ``width`` is in the units of the robot's ``fingers`` state feature,
    between the closed and open values of ``finger_state_bounds``. The
    option ends when the fingers are within ``_GRIPPER_ARRIVE_TOL`` of
    the width, or when they stall (move less than ``_GRIPPER_STALL_TOL``
    for ``_GRIPPER_STALL_STEPS`` consecutive steps), which is what
    closing on an object looks like. The arm joints are held. Whether a
    closing gripper attaches the object is the simulator's grasp rule,
    not this skill's.

    ``force`` is the grip force limit in newtons. A real gripper's
    command is a width and a force, and the parameter is carried here so
    the same plan lines drive one; the simulated grasp is a rigid
    attachment that ignores it, which the parameter description says.
    """
    lo, hi = sorted(finger_state_bounds)
    params_space, params_description = build_params_space([
        (f"width (finger opening in the robot's fingers feature units, "
         f"{lo:.3g} closed to {hi:.3g} open)", lo, hi),
        ("force (grip force limit in newtons; a real gripper honours it, "
         "the simulated rigid grasp ignores it)", *_GRIPPER_FORCE_BOUNDS),
    ])

    def _current_and_target(state: State, objects: Sequence[Object],
                            params: Array) -> Tuple[float, float]:
        robot_obj = objects[0]
        current = config.fingers_state_to_joint(
            config.robot, state.get(robot_obj, "fingers"))
        target = config.fingers_state_to_joint(config.robot, float(params[0]))
        return current, target

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects, params
        memory.clear()
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        pb_state = cast(utils.PyBulletState, state)
        current, target = _current_and_target(state, objects, params)
        last = memory.get("last_fingers")
        if last is not None and abs(current - last) < _GRIPPER_STALL_TOL:
            memory["stall_steps"] = memory.get("stall_steps", 0) + 1
        else:
            memory["stall_steps"] = 0
        memory["last_fingers"] = current
        return get_change_fingers_action(config.robot,
                                         pb_state.joint_positions, current,
                                         target, config.max_vel_norm)

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        current, target = _current_and_target(state, objects, params)
        if abs(current - target) <= _GRIPPER_ARRIVE_TOL:
            return True
        return memory.get("stall_steps", 0) >= _GRIPPER_STALL_STEPS

    return ParameterizedOption(name,
                               types=[robot_type],
                               params_space=params_space,
                               policy=_policy,
                               initiable=_initiable,
                               terminal=_terminal,
                               params_description=params_description)


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------


def create_primitive_skills(
    config: SkillConfig,
    robot_type: Type,
    workspace: Sequence[Tuple[float, float]],
    finger_state_bounds: Tuple[float, float],
) -> Set[ParameterizedOption]:
    """The full primitive library for one environment."""
    return {
        create_move_to_pose_skill("MoveTo", robot_type, config, workspace),
        create_move_linear_skill("MoveLinear", robot_type, config),
        create_move_until_contact_skill("MoveUntilContact", robot_type,
                                        config),
        create_gripper_skill("Gripper", robot_type, config,
                             finger_state_bounds),
        create_wait_option("Wait", config, robot_type),
    }


def primitive_skills_for_env(env_cls: Any, config: SkillConfig,
                             robot_type: Type) -> Set[ParameterizedOption]:
    """The primitive library sized to a PyBullet env class's workspace
    (``x_lb``..``z_ub``) and finger feature range (``closed_fingers``,
    ``open_fingers``)."""
    workspace = _env_workspace(env_cls)
    finger_bounds = (float(env_cls.closed_fingers),
                     float(env_cls.open_fingers))
    return create_primitive_skills(config, robot_type, workspace,
                                   finger_bounds)


def _env_workspace(env_cls: Any) -> Sequence[Tuple[float, float]]:
    bounds = []
    for axis in ("x", "y", "z"):
        lo: Optional[float] = getattr(env_cls, f"{axis}_lb", None)
        hi: Optional[float] = getattr(env_cls, f"{axis}_ub", None)
        if lo is None or hi is None:
            raise ValueError(
                f"{env_cls.__name__} declares no {axis}_lb/{axis}_ub "
                "workspace bounds; the primitive skill library needs them")
        bounds.append((float(lo), float(hi)))
    return bounds
