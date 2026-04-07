"""Core abstractions for reusable parameterized skills."""
# pylint: disable=wrong-import-position,ungrouped-imports

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, \
    Sequence, Tuple, cast

if TYPE_CHECKING:
    from predicators.envs.pybullet_env import PyBulletEnv

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.controllers import \
    get_change_fingers_action, get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type


class PhaseAction(Enum):
    """The type of action a phase executes."""
    MOVE_TO_POSE = auto()
    CHANGE_FINGERS = auto()


@dataclass(frozen=True)
class SkillConfig:
    """Configuration shared across all skill factories for one environment.

    Every skill factory function (``create_pick_skill``, ``create_place_skill``,
    etc.) takes a ``SkillConfig`` as its fourth argument.  Each environment
    options file creates one ``SkillConfig`` and passes it to all its skill
    factory calls.

    Example::

        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=MyEnv._fingers_state_to_joint,
            robot_init_tilt=MyEnv.robot_init_tilt,   # default 0.0
            robot_init_wrist=MyEnv.robot_init_wrist,  # default 0.0
        )

    Attributes:
        robot: The PyBullet robot instance.
        open_fingers_joint: Joint value for fully open fingers.
        closed_fingers_joint: Joint value for fully closed fingers.
        fingers_state_to_joint: Callable that maps the finger *state feature*
            value to the corresponding joint value.  Signature:
            ``(robot, finger_state_value) -> joint_value``.  Typically
            ``MyEnv._fingers_state_to_joint``.
        collision_bodies: PyBullet body IDs to treat as obstacles during
            BiRRT planning.  Defaults to empty (no collision checking).
        move_to_pose_tol: Squared-distance tolerance for move-to-pose
            terminal (used when BiRRT falls back to incremental IK).
        finger_action_nudge_magnitude: Nudge magnitude for finger drift
            resistance in the wait option and during move phases.
        max_vel_norm: Maximum velocity norm for incremental IK EE movement.
        grasp_tol: Squared-distance tolerance for CHANGE_FINGERS terminal.
        ik_validate: Whether to validate IK solutions.
        robot_init_tilt: Default EE tilt (pitch) angle — the second Euler
            angle in ``[roll=0, pitch, yaw]``.
        robot_init_wrist: Default EE wrist (yaw) angle — the third Euler
            angle.  Usually 0.0 or ``-pi``.
        robot_home_pos: ``(x, y, z)`` home position the robot retreats to
            after push skills.  Required by ``create_push_skill``.
        transport_z: Safe Z height for transit above obstacles during
            pick, place, push, and pour skills.  Default ``0.7``.
        extra: Arbitrary dict for environment-specific constants that
            callbacks may need.  Access via ``config.extra["key"]``.
    """
    robot: SingleArmPyBulletRobot
    open_fingers_joint: float
    closed_fingers_joint: float
    fingers_state_to_joint: Callable[[SingleArmPyBulletRobot, float], float]
    collision_bodies: Tuple[int, ...] = ()
    move_to_pose_tol: float = 1e-4
    finger_action_nudge_magnitude: float = 1e-3
    max_vel_norm: float = 0.05
    grasp_tol: float = 5e-4
    ik_validate: bool = True
    robot_init_tilt: float = 0.0
    robot_init_wrist: float = 0.0
    robot_home_pos: Optional[Tuple[float, float, float]] = None
    transport_z: float = 0.7
    simulator: Optional[PyBulletEnv] = None
    collision_skip_types: Tuple[str, ...] = ()
    sim_extra_collision_bodies: Tuple[int, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)


def build_params_space(
    param_defs: Sequence[Tuple[str, float, float]],
) -> Tuple[Box, Tuple[str, ...]]:
    """Build a params_space and description from ``(name, low, high)`` tuples.

    Returns:
        ``(params_space, params_description)``
    """
    names = tuple(name for name, _, _ in param_defs)
    low = np.array([lo for _, lo, _ in param_defs], dtype=np.float64)
    high = np.array([hi for _, _, hi in param_defs], dtype=np.float64)
    return Box(low=low, high=high, dtype=np.float64), names


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

# Callback signature shared by ALL skill factory ``get_target_pose_fn`` args.
# (state, objects, params, config) -> (x, y, z, yaw)
TargetPoseFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                        Tuple[float, float, float, float]]

# ---------------------------------------------------------------------------
# Internal type aliases for Phase target functions
# ---------------------------------------------------------------------------

# For MOVE_TO_POSE: returns (current_pose, target_pose, finger_status)
MoveToPoseTargetFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                              Tuple[Pose, Pose, str]]

# For CHANGE_FINGERS: returns (current_val, target_val)
ChangeFingersTargetFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                                 Tuple[float, float]]

# Memory keys used per phase, keyed by phase object id.
_BIRRT_TRAJ_KEY = "birrt_traj_{}"  # stores List[JointPositions] or None
_BIRRT_STEP_KEY = "birrt_step_{}"  # stores int index into trajectory
_BIRRT_FINGER_KEY = "birrt_finger_{}"  # stores finger_status str


@dataclass
class Phase:
    """A single phase in a multi-phase skill.

    Attributes:
        name: Human-readable phase name (for logging).
        action_type: Whether this phase moves the EE or changes fingers.
        target_fn: Callable that computes targets from state/objects/params.
            For MOVE_TO_POSE: returns (current_pose, target_pose, finger_status)
            For CHANGE_FINGERS: returns (current_val, target_val)
        terminal_fn: Optional custom terminal condition override.
            Signature: (state, objects, params, config) -> bool
        finger_tol: Tolerance for CHANGE_FINGERS terminal (defaults to
            config.grasp_tol if None).
        use_motion_planning: If True (default) and action_type is
            MOVE_TO_POSE, use BiRRT to plan a joint-space trajectory on the
            first call and cache it; subsequent calls pop waypoints from the
            cached plan. Falls back to incremental IK if planning fails.
            If False, always use incremental IK stepping.
    """
    name: str
    action_type: PhaseAction
    # Union[MoveToPoseTargetFn, ChangeFingersTargetFn]; typed as Any to
    # avoid Pylance issues when unpacking return tuples after runtime
    # dispatch on action_type.
    target_fn: Any
    terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None
    finger_tol: Optional[float] = None
    use_motion_planning: bool = field(
        default_factory=lambda: CFG.skill_phase_use_motion_planning)
    expect_contact: bool = False


class PhaseSkill:
    """A multi-phase controller that builds a ParameterizedOption.

    Each phase is executed sequentially. The skill advances to the next
    phase when the current phase's terminal condition is met. The overall
    skill terminates when the last phase is terminal.

    For MOVE_TO_POSE phases with use_motion_planning=True (the default),
    BiRRT plans a collision-free joint-space trajectory on the first call
    and caches it in the option memory dict. Subsequent calls pop waypoints
    from the cached plan one at a time. If BiRRT fails, the phase falls back
    to incremental IK delta-stepping.

    Usage:
        option = PhaseSkill("Pick", types, params_space, config, phases).build()
    """

    def __init__(self,
                 name: str,
                 types: Sequence[Type],
                 params_space: Box,
                 config: SkillConfig,
                 phases: List[Phase],
                 params_description: Optional[Tuple[str, ...]] = None) -> None:
        assert len(phases) > 0
        self._name = name
        self._types = types
        self._params_space = params_space
        self._config = config
        self._phases = phases
        self._params_description = params_description

    def build(self) -> ParameterizedOption:
        """Build and return the ParameterizedOption."""
        return ParameterizedOption(
            self._name,
            types=self._types,
            params_space=self._params_space,
            policy=self._policy,
            initiable=self._initiable,
            terminal=self._terminal,
            params_description=self._params_description,
        )

    def _initiable(self, state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects, params  # unused
        memory["phase_idx"] = 0
        return True

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        phase_idx = memory["phase_idx"]
        phase = self._phases[phase_idx]

        # Check if current phase is terminal → advance.
        if self._phase_is_terminal(phase, state, memory, objects, params):
            phase_idx += 1
            memory["phase_idx"] = phase_idx
            if phase_idx >= len(self._phases):
                # Should not be called after overall terminal, but guard.
                phase_idx = len(self._phases) - 1
                memory["phase_idx"] = phase_idx
            phase = self._phases[phase_idx]
            logging.debug(f"[{self._name}] Advanced to phase {phase_idx}: "
                          f"{phase.name}")

        if phase.action_type == PhaseAction.MOVE_TO_POSE:
            return self._execute_move(phase, state, memory, objects, params)
        assert phase.action_type == PhaseAction.CHANGE_FINGERS
        return self._execute_fingers(phase, state, objects, params)

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        phase_idx = memory["phase_idx"]
        if phase_idx < len(self._phases) - 1:
            return False
        phase = self._phases[phase_idx]
        return self._phase_is_terminal(phase, state, memory, objects, params)

    # ------------------------------------------------------------------
    # Phase terminal conditions
    # ------------------------------------------------------------------

    def _phase_is_terminal(self, phase: Phase, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        """Check if a phase has reached its terminal condition."""
        # Custom terminal override takes priority.
        if phase.terminal_fn is not None:
            return phase.terminal_fn(state, objects, params, self._config)

        if phase.action_type == PhaseAction.CHANGE_FINGERS:
            current_val, target_val = phase.target_fn(state, objects, params,
                                                      self._config)
            tol = phase.finger_tol if phase.finger_tol is not None \
                else self._config.grasp_tol
            return bool((target_val - current_val)**2 < tol)

        # MOVE_TO_POSE
        if phase.use_motion_planning:
            return self._birrt_phase_is_terminal(phase, state, memory, objects,
                                                 params)
        return self._ik_phase_is_terminal(phase, state, objects, params)

    def _birrt_phase_is_terminal(self, phase: Phase, state: State,
                                 memory: Dict, objects: Sequence[Object],
                                 params: Array) -> bool:
        """Terminal for a BiRRT-planned phase.

        Returns True when the cached trajectory is fully consumed, or
        when the fallback IK terminal is satisfied (BiRRT planning
        failed). Returns False if the trajectory hasn't been computed
        yet (first call).
        """
        pid = id(phase)
        traj_key = _BIRRT_TRAJ_KEY.format(pid)
        step_key = _BIRRT_STEP_KEY.format(pid)

        if traj_key not in memory:
            # Trajectory not yet computed — not terminal.
            return False

        traj = memory[traj_key]
        if traj is None:
            # BiRRT failed; use distance-based terminal (IK fallback mode).
            return self._ik_phase_is_terminal(phase, state, objects, params)

        # All waypoints consumed — fall back to position-based terminal so
        # the phase doesn't end until the robot has actually converged to the
        # target (position control may lag behind the commanded trajectory,
        # and IK inaccuracy means the final waypoint may not exactly match
        # the target Cartesian pose).
        if memory[step_key] >= len(traj):
            return self._ik_phase_is_terminal(phase, state, objects, params)
        return False

    def _ik_phase_is_terminal(self, phase: Phase, state: State,
                              objects: Sequence[Object],
                              params: Array) -> bool:
        """Distance-based terminal for incremental IK phases."""
        current_pose, target_pose, _ = phase.target_fn(state, objects, params,
                                                       self._config)
        squared_dist = np.sum(
            np.square(np.subtract(current_pose.position,
                                  target_pose.position)))
        return bool(squared_dist < self._config.move_to_pose_tol)

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _execute_move(self, phase: Phase, state: State, memory: Dict,
                      objects: Sequence[Object], params: Array) -> Action:
        """Dispatch to BiRRT or incremental IK based on phase flag."""
        if phase.use_motion_planning:
            return self._execute_move_birrt(phase, state, memory, objects,
                                            params)
        return self._execute_move_ik(phase, state, objects, params)

    def _execute_move_birrt(self, phase: Phase, state: State, memory: Dict,
                            objects: Sequence[Object],
                            params: Array) -> Action:
        """Execute a MOVE_TO_POSE phase using BiRRT with lazy plan caching.

        On the first call for this phase:
          1. Compute the target joint positions via IK.
          2. Run BiRRT from the current joint positions to the target.
          3. Cache the resulting trajectory (or None on failure).
          4. Cache the finger_status for nudging during trajectory replay.

        On subsequent calls, pop the next waypoint from the cached trajectory
        and return the corresponding joint-position action, applying a small
        finger nudge matching the phase's finger_status (same as incremental
        IK) to prevent drift and allow finger transitions during movement.

        Falls back to incremental IK if BiRRT planning fails.
        """
        pid = id(phase)
        traj_key = _BIRRT_TRAJ_KEY.format(pid)
        step_key = _BIRRT_STEP_KEY.format(pid)
        finger_key = _BIRRT_FINGER_KEY.format(pid)

        pb_state = cast(utils.PyBulletState, state)
        robot = self._config.robot

        if traj_key not in memory:
            # --- First call: plan the trajectory. ---
            _, target_pose, finger_status = phase.target_fn(
                state, objects, params, self._config)
            memory[finger_key] = finger_status

            if self._config.simulator is not None:
                traj = self._plan_with_simulator(pb_state, target_pose,
                                                 phase.name,
                                                 phase.expect_contact)
            else:
                traj = self._plan_without_simulator(pb_state, target_pose,
                                                    phase.name)

            if traj is None:
                if phase.expect_contact:
                    logging.debug(
                        "[%s/%s] BiRRT failed; falling back to "
                        "incremental IK.", self._name, phase.name)
                    memory[traj_key] = None
                else:
                    raise utils.OptionExecutionFailure(
                        f"[{self._name}/{phase.name}] BiRRT collision: "
                        f"motion planning failed (no collision-free path).")
            else:
                # Skip the first waypoint — BiRRT includes the start
                # position (current joints) as traj[0].  Commanding the
                # robot to stay at its current position is a no-op that
                # triggers the option-model "option got stuck" check
                # (option_model_terminate_on_repeat), aborting the option
                # after a single step.
                traj_list = list(traj)
                memory[traj_key] = traj_list[1:] if len(traj_list) > 1 \
                    else traj_list
            memory[step_key] = 0

            # Restore robot joints — run_motion_planning leaves them at an
            # arbitrary configuration used during collision checking.
            robot.set_joints(pb_state.joint_positions)

        traj = memory[traj_key]
        if traj is None:
            # BiRRT failed — fall back to incremental IK.
            return self._execute_move_ik(phase, state, objects, params)

        # --- Pop next waypoint from cached trajectory. ---
        step = memory[step_key]

        if step >= len(traj):
            # Trajectory fully consumed — use incremental IK to converge
            # to the exact target pose (BiRRT's IK solution may be slightly
            # off from the target Cartesian pose).
            return self._execute_move_ik(phase, state, objects, params)

        target_joints = traj[step]
        memory[step_key] = step + 1

        # Apply finger nudge matching the phase's finger_status, identical
        # to what incremental IK does in controllers.py.  This prevents
        # finger drift and allows finger transitions (e.g. open→closed)
        # to happen gradually during BiRRT trajectory replay.
        joint_action = list(target_joints)
        finger_idx_l = robot.left_finger_joint_idx
        finger_idx_r = robot.right_finger_joint_idx
        current_fingers = pb_state.joint_positions[finger_idx_l]
        finger_status = memory[finger_key]
        if finger_status == "open":
            finger_delta = self._config.finger_action_nudge_magnitude
        else:
            finger_delta = -self._config.finger_action_nudge_magnitude
        f_action = current_fingers + finger_delta
        joint_action[finger_idx_l] = f_action
        joint_action[finger_idx_r] = f_action

        action_arr = np.clip(
            np.array(joint_action, dtype=np.float32),
            robot.action_space.low,
            robot.action_space.high,
        )
        return Action(action_arr)

    # ------------------------------------------------------------------
    # BiRRT planning helpers
    # ------------------------------------------------------------------

    # Non-physical types that have no PyBullet body and should be skipped
    # when collecting collision bodies.
    _SKIP_TYPES = frozenset({
        "robot",
        "loc",
        "angle",
        "human",
        "side",
        "direction",
    })

    @staticmethod
    def _collect_sim_objects(sim: PyBulletEnv) -> Dict[str, Object]:
        """Collect all Objects with body IDs from a PyBulletEnv instance."""
        obj_map: Dict[str, Object] = {}
        # Scan instance attributes for Object instances with body IDs.
        for attr_val in sim.__dict__.values():
            if isinstance(attr_val, Object) and attr_val.id is not None:
                obj_map[attr_val.name] = attr_val
            elif isinstance(attr_val, (list, tuple)):
                for item in attr_val:
                    if isinstance(item, Object) and item.id is not None:
                        obj_map[item.name] = item
        # Composed envs: also enumerate component objects.
        for comp in getattr(sim, '_components', []):
            for obj in comp.get_objects():
                obj_map[obj.name] = obj
        # Always include the robot.
        obj_map[sim._robot.name] = sim._robot  # pylint: disable=protected-access
        return obj_map

    def _plan_without_simulator(
        self,
        pb_state: utils.PyBulletState,
        target_pose: Pose,
        phase_name: str,
    ) -> Optional[Sequence[JointPositions]]:
        """Plan using the config robot's physics client (no collision
        bodies)."""
        robot = self._config.robot
        robot.set_joints(pb_state.joint_positions)
        try:
            target_joints: JointPositions = robot.inverse_kinematics(
                target_pose,
                validate=self._config.ik_validate,
                set_joints=True)
        except InverseKinematicsError:
            pos = target_pose.position
            logging.warning(
                f"[{self._name}/{phase_name}] IK failed for BiRRT target "
                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}); "
                "falling back to incremental IK.")
            return None

        return run_motion_planning(
            robot=robot,
            initial_positions=pb_state.joint_positions,
            target_positions=target_joints,
            collision_bodies=self._config.collision_bodies,
            seed=CFG.seed,
            physics_client_id=robot.physics_client_id,
        )

    def _plan_with_simulator(
        self,
        pb_state: utils.PyBulletState,
        target_pose: Pose,
        phase_name: str,
        expect_contact: bool = False,
    ) -> Optional[Sequence[JointPositions]]:
        """Plan using the simulator env for collision-aware motion planning.

        Remaps the current state onto the simulator's objects, resets
        the simulator, collects collision body IDs, and runs IK + BiRRT
        on the simulator's physics client.
        """
        sim = self._config.simulator
        assert sim is not None

        # 1. Build name -> simulator Object mapping
        sim_obj_map = self._collect_sim_objects(sim)

        # 2. Remap state: simulator Objects with original feature values
        new_state_data: Dict[Object, Any] = {}
        for orig_obj, features in pb_state.data.items():
            sim_obj = sim_obj_map.get(orig_obj.name)
            if sim_obj is not None:
                new_state_data[sim_obj] = features.copy()

        remapped_state = utils.PyBulletState(
            new_state_data, simulator_state=pb_state.simulator_state)

        # 3. Reset simulator to current state
        sim._set_state(remapped_state)  # pylint: disable=protected-access

        # 4. Collect collision body IDs (exclude held objects and
        #    non-physical types) and find the held object.
        collision_bodies: set = set()
        held_object: Optional[int] = None
        for orig_obj in pb_state:
            if orig_obj.type.name in self._SKIP_TYPES or \
                    orig_obj.type.name in self._config.collision_skip_types:
                continue
            sim_obj = sim_obj_map.get(orig_obj.name)
            if sim_obj is None or sim_obj.id is None:
                continue
            if "is_held" in orig_obj.type.feature_names and \
                    pb_state.get(orig_obj, "is_held") > 0.5:
                held_object = sim_obj.id
                continue
            collision_bodies.add(sim_obj.id)

        # 4b. Add tables if present.
        if hasattr(sim, '_table_ids'):
            for tid in sim._table_ids:  # pylint: disable=protected-access
                collision_bodies.add(tid)
        elif hasattr(sim, '_table') and sim._table.id is not None:  # pylint: disable=protected-access
            collision_bodies.add(sim._table.id)  # pylint: disable=protected-access

        # 4c. Add extra sim collision bodies (e.g. virtual buffer zones).
        collision_bodies.update(self._config.sim_extra_collision_bodies)

        # 4d. Add environment-specific extra collision bodies (e.g. liquid
        #     blocks in Grow that aren't tracked as state Objects).
        collision_bodies.update(sim.get_extra_collision_ids())

        # 5. IK + motion planning on simulator's robot
        planning_robot = sim._pybullet_robot  # pylint: disable=protected-access
        planning_robot.set_joints(pb_state.joint_positions)
        try:
            target_joints: JointPositions = planning_robot.inverse_kinematics(
                target_pose,
                validate=self._config.ik_validate,
                set_joints=True)
        except InverseKinematicsError:
            pos = target_pose.position
            logging.warning(
                f"[{self._name}/{phase_name}] IK failed for BiRRT target "
                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}); "
                "falling back to incremental IK.")
            return None

        # Compute base_link_to_held_obj if an object is held.
        base_link_to_held_obj = None
        if held_object is not None and sim._held_obj_to_base_link is not None:  # pylint: disable=protected-access
            base_link_to_held_obj = p.invertTransform(
                *sim._held_obj_to_base_link)  # pylint: disable=protected-access

        traj = run_motion_planning(
            robot=planning_robot,
            initial_positions=pb_state.joint_positions,
            target_positions=target_joints,
            collision_bodies=collision_bodies,
            seed=CFG.seed,
            physics_client_id=sim._physics_client_id,  # pylint: disable=protected-access
            held_object=held_object,
            base_link_to_held_obj=base_link_to_held_obj,
        )

        if traj is None and not expect_contact:
            self._log_collision_diagnostics(
                planning_robot,
                sim._physics_client_id,  # pylint: disable=protected-access
                pb_state.joint_positions,
                target_joints,
                collision_bodies,
                held_object,
                base_link_to_held_obj,
                phase_name)

        return traj

    def _log_collision_diagnostics(
        self,
        planning_robot: SingleArmPyBulletRobot,
        physics_client_id: int,
        start_joints: JointPositions,
        goal_joints: JointPositions,
        collision_bodies: set,
        held_object: Optional[int],
        base_link_to_held_obj: Optional[Any],
        phase_name: str,
    ) -> None:
        """Log which collision bodies cause start/goal collisions."""
        from predicators.pybullet_helpers.link import \
            get_link_state  # pylint: disable=import-outside-toplevel

        def _check(joints: JointPositions, label: str) -> None:
            planning_robot.set_joints(joints)
            if held_object is not None and base_link_to_held_obj is not None:
                wt_bl = get_link_state(
                    planning_robot.robot_id,
                    planning_robot.end_effector_id,
                    physics_client_id=physics_client_id).com_pose
                wt_ho = p.multiplyTransforms(wt_bl[0], wt_bl[1],
                                             base_link_to_held_obj[0],
                                             base_link_to_held_obj[1])
                p.resetBasePositionAndOrientation(
                    held_object,
                    wt_ho[0],
                    wt_ho[1],
                    physicsClientId=physics_client_id)
            p.performCollisionDetection(physicsClientId=physics_client_id)
            margin = CFG.pybullet_birrt_contact_margin
            for body in collision_bodies:
                body_name = ""
                try:
                    body_name = p.getBodyInfo(
                        body, physicsClientId=physics_client_id)[1].decode()
                except Exception:  # pylint: disable=broad-except
                    pass
                contacts = p.getContactPoints(
                    planning_robot.robot_id,
                    body,
                    physicsClientId=physics_client_id)
                if any(c[8] < margin for c in contacts):
                    logging.error(f"[{self._name}/{phase_name}] {label} ROBOT "
                                  f"collision with body {body} ({body_name})")
                if held_object is not None:
                    contacts = p.getContactPoints(
                        held_object, body, physicsClientId=physics_client_id)
                    if any(c[8] < margin for c in contacts):
                        logging.error(
                            f"[{self._name}/{phase_name}] {label} HELD "
                            f"collision with body {body} ({body_name})")

        _check(start_joints, "START")
        _check(goal_joints, "GOAL")

    def _execute_move_ik(self, phase: Phase, state: State,
                         objects: Sequence[Object], params: Array) -> Action:
        """Execute a MOVE_TO_POSE phase using incremental IK delta-stepping."""
        pb_state = cast(utils.PyBulletState, state)
        robot = self._config.robot
        robot.set_joints(pb_state.joint_positions)
        current_pose, target_pose, finger_status = phase.target_fn(
            state, objects, params, self._config)
        try:
            return get_move_end_effector_to_pose_action(
                robot=robot,
                current_joint_positions=pb_state.joint_positions,
                current_pose=current_pose,
                target_pose=target_pose,
                finger_status=finger_status,
                max_vel_norm=self._config.max_vel_norm,
                finger_action_nudge_magnitude=(
                    self._config.finger_action_nudge_magnitude),
                validate=self._config.ik_validate,
            )
        except utils.OptionExecutionFailure:
            cur = current_pose.position
            tgt = target_pose.position
            raise utils.OptionExecutionFailure(
                f"[{self._name}/{phase.name}] IK failed. "
                f"current=({cur[0]:.3f}, {cur[1]:.3f}, {cur[2]:.3f}), "
                f"target=({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}), "
                f"params={params.tolist()}")

    def _execute_fingers(self, phase: Phase, state: State,
                         objects: Sequence[Object], params: Array) -> Action:
        """Execute a CHANGE_FINGERS phase."""
        pb_state = cast(utils.PyBulletState, state)
        current_val, target_val = phase.target_fn(state, objects, params,
                                                  self._config)
        return get_change_fingers_action(
            self._config.robot,
            pb_state.joint_positions,
            current_val,
            target_val,
            self._config.max_vel_norm,
        )
