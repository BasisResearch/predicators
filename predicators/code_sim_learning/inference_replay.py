"""Explicit initial states for offline inference, separate from legacy fits.

Replay always owns a fresh world. Object velocities and attachments
travel by name in the portable State; all robot joints travel in URDF
order. A candidate supplies unobserved quantities rather than obtaining
them from a live task. Capture is for simulator predictions and
evaluator-only audits, not an observation channel for the acting agent.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators.code_sim_learning.commands import ApplyForce, ApplyTorque, \
    Attach, PhysicsCommand, SetVelocity
from predicators.code_sim_learning.model_state import has_model_state
from predicators.code_sim_learning.rollout_env import \
    _pin_all_physical_params, add_rollouts_run, dispose_env
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.run.recording import sanitize_state
from predicators.structs import Action, State

# pylint: disable=protected-access
Velocity = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]


@dataclass(frozen=True)
class CommandWeld:
    """A candidate's fixed constraint, by object name and original frames."""

    parent: str
    child: str
    parent_frame: Pose
    child_frame: Pose
    max_force: float
    erp: float


@dataclass(frozen=True)
class ArticulatedBody:
    """Nonrobot joint state tied to an identical native world layout.

    Native identifiers are not portable semantic names. Restoration
    checks body and joint topology and requires the same allocation
    protocol; this record does not remap bodies across different worlds.
    """

    body_id: int
    body_names: Tuple[str, str]
    joint_layout: Tuple[Tuple[str, int, str], ...]
    joints: Tuple[Tuple[float, float], ...]


def _capture_articulated_bodies(
        env: PyBulletEnv) -> Tuple[ArticulatedBody, ...]:
    """Include grouped bodies and fixtures absent from public object keys."""
    pcid = env._physics_client_id
    records = []
    for index in range(p.getNumBodies(physicsClientId=pcid)):
        body_id = p.getBodyUniqueId(index, physicsClientId=pcid)
        count = p.getNumJoints(body_id, physicsClientId=pcid)
        if body_id == env._pybullet_robot.robot_id or count == 0:
            continue
        names = p.getBodyInfo(body_id, physicsClientId=pcid)
        layout, joints = [], []
        for joint in range(count):
            info = p.getJointInfo(body_id, joint, physicsClientId=pcid)
            layout.append((info[1].decode(), int(info[2]), info[12].decode()))
            value = p.getJointState(body_id, joint, physicsClientId=pcid)
            joints.append((float(value[0]), float(value[1])))
        records.append(
            ArticulatedBody(body_id, (names[0].decode(), names[1].decode()),
                            tuple(layout), tuple(joints)))
    return tuple(sorted(records, key=lambda record: record.body_id))


@dataclass(frozen=True)
class ReplayState:
    """A physical/model state and explicit robot motion at an action boundary.

    ``robot_joints`` contains (position, velocity) for EVERY URDF joint,
    including passive joints. Controlled positions must agree with the
    State's joint_positions. Full body orientations, original command-
    weld frames, and pending next-step commands are candidate quantities
    even when absent from public observations. The caller owns the
    nested state; replay copies it before restoration or stepping, so
    siblings cannot mutate one another. No solver warm-start state is
    represented; replay error must be measured.
    """

    state: State
    robot_joints: Tuple[Tuple[float, float], ...]
    robot_base_velocity: Velocity
    body_poses: Mapping[str, Pose]
    pending_commands: Tuple[PhysicsCommand, ...]
    command_welds: Tuple[CommandWeld, ...]
    articulated_bodies: Tuple[ArticulatedBody, ...]


def capture_replay_state(env: PyBulletEnv) -> ReplayState:
    """Snapshot a simulated candidate, or evaluator state for an offline audit.

    This deliberately is not called by recording or observation code. It
    preserves candidate memory and removes privileged feature payloads.
    The fresh factory must use the same body layout as the candidate;
    Object metadata is copied, not a cross-layout body remapping API.
    """
    raw = env._get_state()
    # State.copy()/sanitize_state() retain Object keys. Their sim_data is
    # mutable engine metadata, so candidate ownership needs a full copy.
    state = sanitize_state(copy.deepcopy(raw))
    state.latent = copy.deepcopy(raw.latent)
    pcid = env._physics_client_id
    robot_id = env._pybullet_robot.robot_id
    joints = []
    for j in range(p.getNumJoints(robot_id, physicsClientId=pcid)):
        info = p.getJointState(robot_id, j, physicsClientId=pcid)
        joints.append((float(info[0]), float(info[1])))
    linear, angular = p.getBaseVelocity(robot_id, physicsClientId=pcid)
    bodies = {
        o.id: o.name
        for o in env._objects if o.id is not None and o.type.name != "robot"
        and o.type.name not in env._VIRTUAL_OBJECT_TYPES
    }
    poses = {}
    for body_id, name in bodies.items():
        position, orientation = p.getBasePositionAndOrientation(
            body_id, physicsClientId=pcid)
        poses[name] = (tuple(position), tuple(orientation))
    welds = []
    for cid in env._cmd_weld_constraints.values():
        info = p.getConstraintInfo(cid, physicsClientId=pcid)
        if info[4] != p.JOINT_FIXED or info[1] != -1 or info[3] != -1:
            raise ValueError("Replay command weld must be a fixed base weld")
        welds.append(
            CommandWeld(bodies[info[0]], bodies[info[2]],
                        (tuple(info[6]), tuple(info[8])),
                        (tuple(info[7]), tuple(info[9])), float(info[10]),
                        float(info[14])))
    return ReplayState(state, tuple(joints), (tuple(linear), tuple(angular)),
                       poses,
                       tuple(copy.deepcopy(env._pending_residual_commands)),
                       tuple(welds), _capture_articulated_bodies(env))


def _validate_pose(pose: Pose) -> None:
    """Reject invalid physical poses before touching the candidate world."""
    position, orientation = (np.asarray(v, dtype=float) for v in pose)
    if (position.shape != (3, ) or orientation.shape != (4, )
            or not np.isfinite(position).all()
            or not np.isfinite(orientation).all() or not np.isclose(
                np.linalg.norm(orientation), 1., rtol=0, atol=1e-6)):
        raise ValueError("Replay requires finite poses with unit quaternions")


def _validate_commands(commands: Tuple[PhysicsCommand, ...],
                       names: set[str]) -> None:
    """Do not silently ignore invalid actuation or unknown target objects."""
    for command in commands:
        vectors = []
        if isinstance(command, Attach):
            if (command.obj_a_name == command.obj_b_name
                    or not {command.obj_a_name, command.obj_b_name} <= names):
                raise ValueError(
                    "Replay attachment command has invalid targets")
            continue
        if isinstance(command, (ApplyForce, ApplyTorque, SetVelocity)):
            if command.obj_name not in names:
                raise ValueError("Replay command has an unknown target")
            if isinstance(command, ApplyForce):
                vectors = [command.force]
            elif isinstance(command, ApplyTorque):
                vectors = [command.torque]
            else:
                vectors = [
                    v for v in (command.linear, command.angular)
                    if v is not None
                ]
        else:
            raise ValueError("Unsupported replay physics command")
        for vector in vectors:
            value = np.asarray(vector, dtype=float)
            if value.shape != (3, ) or not np.isfinite(value).all():
                raise ValueError(
                    "Replay command requires finite three-vectors")


def _restore_candidate(env: PyBulletEnv, candidate: ReplayState) -> None:
    """Restore supplied state and reject missing or inconsistent motion."""
    state = copy.deepcopy(candidate.state)
    sim = state.simulator_state
    if state.privileged is not None:
        raise ValueError("Replay candidates must not contain privileged data")
    if not isinstance(sim, dict) or "joint_positions" not in sim:
        raise ValueError("Replay requires explicit joint_positions")
    velocities = sim.get("body_velocities")
    if not isinstance(velocities, dict):
        raise ValueError("Replay requires explicit body_velocities")
    if has_model_state(type(env)) and state.latent is None:
        raise ValueError("Replay requires explicit model memory")
    robot = env._pybullet_robot
    pcid = env._physics_client_id
    robot_id = robot.robot_id
    count = p.getNumJoints(robot_id, physicsClientId=pcid)
    joints = np.asarray(candidate.robot_joints, dtype=float)
    if joints.shape != (count, 2) or not np.isfinite(joints).all():
        raise ValueError("Replay requires finite position/velocity for every "
                         "robot joint in URDF order")
    controlled = np.asarray(sim["joint_positions"], dtype=float)
    if (controlled.shape != (len(robot.arm_joints), ) or not np.allclose(
            controlled, joints[robot.arm_joints, 0], rtol=0, atol=1e-12)):
        raise ValueError("Replay robot joint positions disagree with State")
    base_velocity = np.asarray(candidate.robot_base_velocity, dtype=float)
    if base_velocity.shape != (2, 3) or not np.isfinite(base_velocity).all():
        raise ValueError("Replay requires finite robot base velocity")
    if int(getattr(robot, "base_action_dim",
                   0)) > 0 and "base_pose" not in sim:
        raise ValueError("Mobile replay requires an explicit base_pose")
    # Validate by domain object type. The fresh factory must reproduce
    # the candidate body layout expected by the domain restore hook.
    physical = [
        o for o in state if o.type.name != "robot"
        and o.type.name not in env._VIRTUAL_OBJECT_TYPES
    ]
    for obj in physical:
        value = np.asarray(velocities.get(obj.name), dtype=float)
        if value.shape != (2, 3) or not np.isfinite(value).all():
            raise ValueError(f"Replay requires finite body_velocities for "
                             f"{obj.name}")
    physical_names = {obj.name for obj in physical}
    if set(candidate.body_poses) != physical_names:
        raise ValueError("Replay requires a physical pose for every body")
    for pose in candidate.body_poses.values():
        _validate_pose(pose)
    _validate_commands(candidate.pending_commands, physical_names)
    for pair in sim.get("command_welds", ()):
        if (len(pair) != 2 or pair[0] == pair[1]
                or not set(pair).issubset(physical_names)):
            raise ValueError("Replay command welds must join two known "
                             "physical objects")
    records = [(w.parent, w.child) for w in candidate.command_welds]
    if (len({frozenset(pair)
             for pair in records}) != len(records)
            or sorted(records) != sorted(sim.get("command_welds", ()))):
        raise ValueError("Replay requires the original frame for every weld")
    for weld in candidate.command_welds:
        _validate_pose(weld.parent_frame)
        _validate_pose(weld.child_frame)
        if (not np.isfinite([weld.max_force, weld.erp]).all()
                or weld.max_force < 0 or not 0 <= weld.erp <= 1):
            raise ValueError("Replay requires valid weld force and ERP")
    env._set_state(state)
    actual_bodies = _capture_articulated_bodies(env)
    expected_bodies = candidate.articulated_bodies
    actual_layout = tuple(
        (b.body_id, b.body_names, b.joint_layout) for b in actual_bodies)
    expected_layout = tuple(
        (b.body_id, b.body_names, b.joint_layout) for b in expected_bodies)
    if actual_layout != expected_layout:
        raise ValueError("Replay requires the same complete articulated body "
                         "layout and native allocation order")
    for body in expected_bodies:
        values = np.asarray(body.joints, dtype=float)
        if values.shape != (len(body.joint_layout), 2) or not \
                np.isfinite(values).all():
            raise ValueError("Replay requires finite position/velocity for "
                             "every articulated body joint")
    # _set_state can skip a body whose pose already matches. Restore its
    # supplied motion unconditionally, including passive robot joints.
    for obj in env._objects:
        if obj in physical and obj.id is not None:
            p.resetBasePositionAndOrientation(obj.id,
                                              *candidate.body_poses[obj.name],
                                              physicsClientId=pcid)
            linear, angular = velocities[obj.name]
            p.resetBaseVelocity(obj.id, linear, angular, physicsClientId=pcid)
    for j, (position, velocity) in enumerate(candidate.robot_joints):
        p.resetJointState(robot_id,
                          j,
                          position,
                          targetVelocity=velocity,
                          physicsClientId=pcid)
    p.resetBaseVelocity(robot_id,
                        *candidate.robot_base_velocity,
                        physicsClientId=pcid)
    for body in expected_bodies:
        for joint, (position, velocity) in enumerate(body.joints):
            p.resetJointState(body.body_id,
                              joint,
                              position,
                              targetVelocity=velocity,
                              physicsClientId=pcid)
    # Restoring current poses cannot reconstruct the frame at attachment
    # creation. Preserve that frame, including any constraint deflection.
    env._clear_commanded_attachments()
    ids = {
        obj.name: obj.id
        for obj in env._objects if obj.name in physical_names
    }
    for weld in candidate.command_welds:
        cid = p.createConstraint(ids[weld.parent],
                                 -1,
                                 ids[weld.child],
                                 -1,
                                 p.JOINT_FIXED, (0., 0., 0.),
                                 weld.parent_frame[0],
                                 weld.child_frame[0],
                                 weld.parent_frame[1],
                                 weld.child_frame[1],
                                 physicsClientId=pcid)
        p.changeConstraint(cid,
                           maxForce=weld.max_force,
                           erp=weld.erp,
                           physicsClientId=pcid)
        env._cmd_weld_constraints[frozenset((weld.parent, weld.child))] = cid
    # Commands emitted after the preceding step belong to this boundary.
    # Do not re-run the model here: that would advance its memory twice.
    env.queue_residual_commands(copy.deepcopy(candidate.pending_commands))
    env._current_observation = env._get_state()


def replay_candidate(factory: Callable[[], PyBulletEnv],
                     initial: ReplayState,
                     actions: List[Action],
                     parameters: Mapping[str, float],
                     *,
                     prefix: Sequence[Action] = ()) -> List[ReplayState]:
    """Replay a candidate without the legacy fitter's rest-start assumption.

    With no prefix, return the reconstructed initial state followed by
    every post-action state. With a prefix, replay those actions in the
    SAME world before returning the boundary state and requested suffix.
    This retains engine history, native constraints, and domain-private
    memory without a mid-trajectory restore. It costs the full prefix on
    every call. The root must still be a valid candidate initialization,
    not an observed noisy pose or a claimed exact engine checkpoint.
    Memory and constraints evolve in the candidate subclass; no
    observations are injected during the rollout. This is an offline API
    and changes no production fit or observation path.
    """
    return replay_initialized_candidate(
        factory,
        lambda env: _restore_candidate(env, initial),
        actions,
        parameters,
        prefix=prefix)


def replay_initialized_candidate(
    factory: Callable[[], PyBulletEnv],
    initialize: Callable[[PyBulletEnv], None],
    actions: Sequence[Action],
    parameters: Mapping[str, float],
    *,
    prefix: Sequence[Action] = ()) -> List[ReplayState]:
    """Replay from an explicit, reproducible candidate initialization protocol.

    The initializer receives a fresh world with parameters already applied.
    It must create the candidate root, including any initialization history
    the engine needs. Parameters are reapplied after initialization, as in
    legacy rollout plumbing. Prefix and suffix run without state restoration
    between them. No reset/task selection is implicit in this API.

    An inference caller must initialize from its declared prior and allowed
    conditioned inputs, never from evaluator-private state. An offline
    mechanical audit may instead use the evaluator reset protocol to isolate
    engine reproducibility from initial-state reconstruction. The initializer,
    its inputs, and runtime belong in the experiment's artifact identity.
    """
    env = factory()
    add_rollouts_run(1)
    try:
        _pin_all_physical_params(env, dict(parameters))
        initialize(env)
        _pin_all_physical_params(env, dict(parameters))
        for action in prefix:
            env.step(Action(action.arr.copy()))
        states = [capture_replay_state(env)]
        for action in actions:
            env.step(Action(action.arr.copy()))
            states.append(capture_replay_state(env))
        return states
    finally:
        dispose_env(env)
