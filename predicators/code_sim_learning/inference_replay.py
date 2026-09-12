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
from typing import Callable, List, Mapping, Tuple

import numpy as np
import pybullet as p

from predicators.code_sim_learning.model_state import has_model_state
from predicators.code_sim_learning.rollout_env import \
    _pin_all_physical_params, add_rollouts_run, dispose_env
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.run.recording import sanitize_state
from predicators.structs import Action, State

# pylint: disable=protected-access
Velocity = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


@dataclass(frozen=True)
class ReplayState:
    """A physical/model state and explicit robot motion at an action boundary.

    ``robot_joints`` contains (position, velocity) for EVERY URDF joint,
    including passive joints. Controlled positions must agree with the
    State's joint_positions. The caller owns the nested state; replay
    copies it before restoration or stepping, so siblings cannot mutate
    one another. No solver warm-start state is represented; replay error
    must be measured.
    """

    state: State
    robot_joints: Tuple[Tuple[float, float], ...]
    robot_base_velocity: Velocity


def capture_replay_state(env: PyBulletEnv) -> ReplayState:
    """Snapshot a simulated candidate, or evaluator state for an offline audit.

    This deliberately is not called by recording or observation code. It
    preserves candidate memory while dropping live client/body handles
    and privileged feature payloads from the copied State.
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
    return ReplayState(state, tuple(joints), (tuple(linear), tuple(angular)))


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
    # Validate by domain object type, not process-local body IDs. Fresh
    # worlds rebuild those IDs while restoring the state.
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
    for pair in sim.get("command_welds", ()):
        if (len(pair) != 2 or pair[0] == pair[1]
                or not set(pair).issubset(physical_names)):
            raise ValueError("Replay command welds must join two known "
                             "physical objects")
    env._set_state(state)
    # _set_state can skip a body whose pose already matches. Restore its
    # supplied motion unconditionally, including passive robot joints.
    for obj in env._objects:
        if obj in physical and obj.id is not None:
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
    env._current_observation = env._get_state()


def replay_candidate(factory: Callable[[], PyBulletEnv], initial: ReplayState,
                     actions: List[Action],
                     parameters: Mapping[str, float]) -> List[ReplayState]:
    """Replay a candidate without the legacy fitter's rest-start assumption.

    Return the reconstructed initial state followed by every post-action
    state. The initial return value lets an audit measure restoration
    error separately from transition error. Memory and constraints
    evolve in the candidate subclass; no observations are injected
    during the rollout. This is an offline API and changes no production
    fit or observation path.
    """
    env = factory()
    add_rollouts_run(1)
    try:
        _pin_all_physical_params(env, dict(parameters))
        _restore_candidate(env, initial)
        _pin_all_physical_params(env, dict(parameters))
        states = [capture_replay_state(env)]
        for action in actions:
            env.step(Action(action.arr.copy()))
            states.append(capture_replay_state(env))
        return states
    finally:
        dispose_env(env)
