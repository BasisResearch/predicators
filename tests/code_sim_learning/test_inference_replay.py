"""Offline candidates retain motion and memory without changing legacy fits."""
# pylint: disable=protected-access
from dataclasses import replace
from typing import Any, ClassVar, Dict, List

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce, CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.inference_replay import \
    capture_replay_state, replay_candidate, replay_initialized_candidate
from predicators.code_sim_learning.rollout_env import rollout_states
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models.balloons.gt_simulator_env import \
    BalloonsResidualEnv
from predicators.pybullet_helpers.objects import create_object
from predicators.structs import Action


class _MovingModel(BalloonsResidualEnv):
    """Visible physics with a recurrent counter and no hidden balloon force."""

    AGENT_PARAM_SPECS: ClassVar[List[Any]] = [ParamSpec("rate", .25)]
    MODEL_STATE_INIT: ClassVar[Dict[str, Any]] = {"events": []}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        del observation, action
        model_state["events"].append(params["rate"])

    def _domain_specific_step(self):
        """Let ordinary engine gravity move the box."""


class _MetadataModel(_MovingModel):
    """A candidate environment that writes its own object metadata."""

    def _set_state(self, state):
        super()._set_state(state)
        for obj in state:
            obj.sim_data["replay_marker"] = "candidate"


class _GroupedModel(_MovingModel):
    """A supplementary physical fan has no public object of its own."""

    def __init__(self):
        super().__init__()
        self.rotor_body = create_object(
            "urdf/partnet_mobility/fan/101450/mobility.urdf",
            position=(5., 5., 5.),
            scale=.08,
            use_fixed_base=True,
            physics_client_id=self._physics_client_id)


@pytest.fixture(name="moving_env")
def _moving_env():
    utils.reset_config({"env": "pybullet_balloons", "seed": 0})
    source = PyBulletBalloonsEnv(use_gui=False)
    env = _MovingModel(use_gui=False)
    try:
        initial = source.level_state(0, [0, 1], (.7, .75))
        initial.set(source._box, "z", 1.1)
        env._set_state(initial)
        p.resetBaseVelocity(env._box.id, [0, 0, .5], [0, 0, 0],
                            physicsClientId=env._physics_client_id)
        robot = env._pybullet_robot
        joint = robot.arm_joints[0]
        position = p.getJointState(robot.robot_id,
                                   joint,
                                   physicsClientId=env._physics_client_id)[0]
        p.resetJointState(robot.robot_id,
                          joint,
                          position,
                          targetVelocity=.12,
                          physicsClientId=env._physics_client_id)
        env._current_observation = env._get_state()
        yield env
    finally:
        env.dispose()
        source.dispose()


def test_moving_candidate_replay_and_branching(moving_env):
    """Replay actual engine motion, a contact, and a recurrent memory
    prefix."""
    env = moving_env
    initial = capture_replay_state(env)
    action = Action(
        np.array(initial.state.simulator_state["joint_positions"],
                 dtype=np.float32))
    observed = [initial]
    for _ in range(30):
        env.step(action)
        observed.append(capture_replay_state(env))
    first = replay_candidate(_MovingModel, initial, [action] * 30, {})
    second = replay_candidate(_MovingModel, initial, [action] * 30, {})
    assert len(first) == len(observed)
    assert first[0].robot_joints == initial.robot_joints
    assert first[0].state.simulator_state["body_velocities"] == \
        initial.state.simulator_state["body_velocities"]
    for expected, actual, repeated in zip(observed, first, second):
        assert actual.state.latent == expected.state.latent
        assert actual.state.allclose(repeated.state)
        assert actual.robot_joints == repeated.robot_joints
        assert abs(
            actual.state.get(env._box, "z") -
            expected.state.get(env._box, "z")) < 1e-5
    assert first[-1].state.get(env._box, "z") < 1.0  # Reaches table contact.
    # A state at a moving prefix is a valid branch, including accumulated
    # memory. Mutating a returned branch cannot change its sibling/input.
    resumed = replay_candidate(_MovingModel, first[3], [action] * 6, {})
    assert resumed[-1].state.latent == first[9].state.latent
    assert abs(resumed[-1].state.get(env._box, "z") -
               first[9].state.get(env._box, "z")) < 1e-3
    resumed[0].state.latent["events"].append(99)
    assert first[3].state.latent == {"events": [.25] * 3}
    assert initial.state.latent == {"events": []}
    assert _MovingModel.MODEL_STATE_INIT == {"events": []}
    # The incumbent estimator intentionally still assumes a rest start.
    legacy = rollout_states(_MovingModel, initial.state, [action] * 6, {})
    assert max(
        abs(a.state.get(env._box, "z") - b.get(env._box, "z"))
        for a, b in zip(first[1:], legacy)) > .005


@pytest.fixture(name="grouped_candidate")
def _grouped_candidate(moving_env):
    env = _GroupedModel()
    try:
        env._set_state(moving_env._get_state())
        pcid = env._physics_client_id
        clip = env._clips[0]
        limit = p.getJointInfo(clip.id, clip.joint_id, physicsClientId=pcid)[9]
        p.resetJointState(clip.id,
                          clip.joint_id,
                          .025 * limit,
                          targetVelocity=.02,
                          physicsClientId=pcid)
        for joint in range(p.getNumJoints(env.rotor_body,
                                          physicsClientId=pcid)):
            if p.getJointInfo(env.rotor_body, joint,
                              physicsClientId=pcid)[2] != p.JOINT_FIXED:
                p.resetJointState(env.rotor_body,
                                  joint,
                                  .7,
                                  targetVelocity=.3,
                                  physicsClientId=pcid)
        initial = capture_replay_state(env)
        assert env.rotor_body not in [obj.id for obj in env._objects]
        assert env.rotor_body in [
            b.body_id for b in initial.articulated_bodies
        ]
        yield initial
    finally:
        env.dispose()


def test_nonrobot_joint_replay_including_unobserved_body(grouped_candidate):
    """A moving lever and a grouped rotor survive a fresh candidate replay."""
    initial = grouped_candidate
    action = Action(
        np.asarray(initial.state.simulator_state["joint_positions"],
                   dtype=np.float32))
    first = replay_candidate(_GroupedModel, initial, [action] * 3, {})
    second = replay_candidate(_GroupedModel, initial, [action] * 3, {})
    assert first[0].articulated_bodies == initial.articulated_bodies
    assert any(velocity != 0 for body in initial.articulated_bodies
               for _, velocity in body.joints)
    for left, right in zip(first, second):
        assert left.articulated_bodies == right.articulated_bodies
        assert left.state.allclose(right.state)


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "layout", "count", "nonfinite"])
def test_nonrobot_joint_replay_rejects_incomplete_state(
        grouped_candidate, problem):
    """Malformed topology or motion cannot silently become a reset default."""
    initial = grouped_candidate
    records = initial.articulated_bodies
    if problem == "missing":
        records = records[:-1]
    elif problem == "duplicate":
        records = records + (records[-1], )
    elif problem == "layout":
        records = (replace(records[0], body_names=("other", "asset")),) + \
            records[1:]
    else:
        joints = records[0].joints[:-1] if problem == "count" else \
            ((float("nan"), 0.),) + records[0].joints[1:]
        records = (replace(records[0], joints=joints), ) + records[1:]
    with pytest.raises(ValueError, match="articulated body"):
        replay_candidate(_GroupedModel,
                         replace(initial, articulated_bodies=records), [], {})


@pytest.mark.parametrize("missing", ["velocity", "memory", "joints"])
def test_candidate_requires_explicit_missing_state(moving_env, missing):
    """Unavailable state fails explicitly and the temporary world is freed."""
    initial = capture_replay_state(moving_env)
    if missing == "velocity":
        initial.state.simulator_state["body_velocities"].pop(
            moving_env._box.name)
    elif missing == "memory":
        initial.state.latent = None
    else:
        initial = replace(initial, robot_joints=())
    clients = []

    def factory():
        env = _MovingModel()
        clients.append(env._physics_client_id)
        return env

    with pytest.raises(ValueError, match="Replay requires"):
        replay_candidate(factory, initial, [], {})
    assert clients and not p.isConnected(clients[0])


def test_candidate_joint_state_must_be_consistent(moving_env):
    """Duplicated joint positions cannot silently disagree."""
    initial = capture_replay_state(moving_env)
    initial.state.simulator_state["joint_positions"][0] += .1
    with pytest.raises(ValueError, match="joint positions disagree"):
        replay_candidate(_MovingModel, initial, [], {})


def test_candidate_owns_object_metadata(moving_env):
    """A candidate's restore hook cannot modify its input or source world."""
    initial = capture_replay_state(moving_env)
    output, = replay_candidate(_MetadataModel, initial, [], {})
    assert all(obj.sim_data["replay_marker"] == "candidate"
               for obj in output.state if obj.type.name != "robot")
    assert all("replay_marker" not in obj.sim_data for obj in initial.state)
    assert all("replay_marker" not in obj.sim_data
               for obj in moving_env._objects)


def test_captured_candidate_is_independent_of_later_source_changes(moving_env):
    """A saved candidate cannot change when its source world moves on."""
    initial = capture_replay_state(moving_env)
    box = next(obj for obj in moving_env._objects if obj.type.name == "box")
    box.sim_data["later_epoch"] = 1
    captured_box = next(obj for obj in initial.state if obj == box)
    assert "later_epoch" not in captured_box.sim_data


def test_candidate_rejects_unknown_attachment(moving_env):
    """Inference must not silently drop an impossible commanded attachment."""
    initial = capture_replay_state(moving_env)
    initial.state.simulator_state["command_welds"] = [(moving_env._box.name,
                                                       "missing_object")]
    with pytest.raises(ValueError, match="two known physical objects"):
        replay_candidate(_MovingModel, initial, [], {})


def test_candidate_restores_command_attachments():
    """A moving welded assembly restores its topology and both velocities."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 0
    })

    def factory():
        return PyBulletBridgeEnv(use_gui=False, skip_residual_dynamics=True)

    env = factory()
    try:
        state = env.get_train_tasks()[0].init.copy()
        first, second = env._spans[:2]
        for i, obj in enumerate((first, second)):
            state.set(obj, "x",
                      .65 + i * (2 * env.span_half_extents[0] + .002))
            state.set(obj, "y", 1.30)
            state.set(obj, "z", env.table_height + env.span_half_extents[2])
            state.set(obj, "yaw", 0.)
        env._set_state(state)
        env._current_observation = env._get_state()
        action = Action(
            np.array(env._current_observation.joint_positions,
                     dtype=np.float32))
        for _ in range(6):
            commands = CommandBuffer()
            commands.attach(first, second)
            commands.apply_force(first, (0., 0., 3.))
            env.queue_residual_commands(commands.commands)
            env.step(action)
        initial = capture_replay_state(env)
        assert initial.state.simulator_state["command_welds"]
        restored, = replay_candidate(factory, initial, [], {})
        assert restored.state.simulator_state["command_welds"] == \
            initial.state.simulator_state["command_welds"]
        assert restored.state.simulator_state["body_velocities"] == \
            initial.state.simulator_state["body_velocities"]
        for obj in (first, second):
            for feature in ("x", "y", "z"):
                assert abs(
                    restored.state.get(obj, feature) -
                    initial.state.get(obj, feature)) < 1e-6
    finally:
        env.dispose()


class _QueuedForceModel(_MovingModel):
    """A force and attachment remain queued at each action boundary."""

    def _domain_specific_step(self):
        commands = CommandBuffer()
        commands.apply_force(self._box, (0., 0., 4.))
        commands.attach(self._balloons[0], self._box)
        self.queue_residual_commands(commands.commands)


def test_replay_retains_pending_commands_and_unobserved_orientation():
    """Continue a real tilted, welded assembly with an outstanding force."""
    utils.reset_config({"env": "pybullet_balloons", "seed": 0})
    source = PyBulletBalloonsEnv(use_gui=False)
    env = _QueuedForceModel()
    try:
        state = source.level_state(0, [0, 1], (.7, .75))
        state.set(source._box, "z", 1.1)
        env._set_state(state)
        p.resetBasePositionAndOrientation(
            env._box.id,
            (state.get(source._box, "x"), state.get(source._box, "y"), 1.1),
            p.getQuaternionFromEuler((.1, .2, .3)),
            physicsClientId=env._physics_client_id)
        env._current_observation = env._get_state()
        action = Action(
            np.array(env._current_observation.joint_positions,
                     dtype=np.float32))
        for _ in range(3):
            env.step(action)
        initial = capture_replay_state(env)
        predicted = replay_candidate(_QueuedForceModel, initial, [], {})[0]
        # These quantities are candidate state, never public observations.
        assert predicted.pending_commands == initial.pending_commands
        assert predicted.command_welds == initial.command_welds
        for name, pose in initial.body_poses.items():
            assert np.allclose(predicted.body_poses[name][0],
                               pose[0],
                               rtol=0,
                               atol=1e-12)
            assert np.allclose(predicted.body_poses[name][1],
                               pose[1],
                               rtol=0,
                               atol=1e-12)
    finally:
        env.dispose()
        source.dispose()


def test_prefix_replay_keeps_engine_history_and_memory(moving_env):
    """A continued contact trajectory equals its uninterrupted candidate."""
    initial = capture_replay_state(moving_env)
    hold = Action(
        np.array(initial.state.simulator_state["joint_positions"],
                 dtype=np.float32))
    full = replay_candidate(_MovingModel, initial, [hold] * 30, {})
    branch = replay_candidate(_MovingModel,
                              initial, [hold] * 19, {},
                              prefix=[hold] * 11)
    assert len(branch) == 20
    for expected, actual in zip(full[11:], branch):
        for obj in expected.state:
            np.testing.assert_array_equal(expected.state[obj],
                                          actual.state[obj])
        assert expected.robot_joints == actual.robot_joints
        assert expected.body_poses == actual.body_poses
        assert expected.pending_commands == actual.pending_commands
        assert expected.command_welds == actual.command_welds
        assert expected.state.latent == actual.state.latent
    assert initial.state.latent == {"events": []}


@pytest.mark.parametrize("invalid", ["poses", "command", "weld"])
def test_replay_rejects_incomplete_physical_state(moving_env, invalid):
    """Missing physical quantities must not become implicit zero defaults."""
    initial = capture_replay_state(moving_env)
    if invalid == "poses":
        initial = replace(initial, body_poses={})
    elif invalid == "command":
        initial = replace(initial,
                          pending_commands=(ApplyForce("missing",
                                                       (0., 0., 1.)), ))
    else:
        initial.state.simulator_state["command_welds"] = [
            (moving_env._balloons[0].name, moving_env._box.name)
        ]
    with pytest.raises(ValueError, match="Replay"):
        replay_candidate(_MovingModel, initial, [], {})


def test_explicit_initializer_replays_prefix_under_candidate_parameters(
        moving_env):
    """The initializer and full prefix use the requested candidate
    parameters."""
    initial = capture_replay_state(moving_env)
    hold = Action(
        np.array(initial.state.simulator_state["joint_positions"],
                 dtype=np.float32))
    seen = []

    def initialize(env):
        seen.append(env.agent_param("rate"))
        env._set_state(initial.state)

    first = replay_initialized_candidate(_MovingModel,
                                         initialize, [hold], {"rate": .5},
                                         prefix=[hold] * 3)
    second = replay_initialized_candidate(_MovingModel,
                                          initialize, [hold], {"rate": .75},
                                          prefix=[hold] * 3)
    assert seen == [.5, .75]
    assert first[0].state.latent == {"events": [.5] * 3}
    assert first[1].state.latent == {"events": [.5] * 4}
    assert second[0].state.latent == {"events": [.75] * 3}
    assert second[1].state.latent == {"events": [.75] * 4}
    assert initial.state.latent == {"events": []}


def test_initializer_failure_disposes_fresh_world():
    """A rejected candidate root must release its engine client."""
    utils.reset_config({"env": "pybullet_balloons", "seed": 0})
    clients = []

    def initialize(env):
        clients.append(env._physics_client_id)
        raise ValueError("invalid candidate root")

    with pytest.raises(ValueError, match="invalid candidate root"):
        replay_initialized_candidate(_MovingModel, initialize, [], {})
    assert clients and not p.isConnected(clients[0])
