"""Offline candidates retain motion and memory without changing legacy fits."""
# pylint: disable=protected-access
from dataclasses import replace
from typing import Any, ClassVar, Dict, List

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.inference_replay import \
    capture_replay_state, replay_candidate
from predicators.code_sim_learning.rollout_env import rollout_states
from predicators.envs.pybullet_balloons import PyBulletBalloonsEnv
from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models.balloons.gt_simulator_env import \
    BalloonsResidualEnv
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
