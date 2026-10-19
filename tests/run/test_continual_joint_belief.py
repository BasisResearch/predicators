"""The session's joint draws of the belief (paper Section 3.2).

Each joint draw pairs one draw of the approach's parameter belief with a
draw of the state factor whose memory is the one those parameters imply
over the observed episode prefix; the draws are fixed per decision
point, and monitoring reads atoms on them.
"""
# pylint: disable=protected-access
import os

import numpy as np
import pybullet as p
import pytest

from predicators import observation_noise, utils
from predicators.approaches import create_approach
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.code_sim_learning.parameter_belief import ParameterBelief
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.structs import Action
from tests.code_sim_learning.test_subclass_model_state import _MemoryModel
from tests.run.test_continual import _config, _make, _Preempted

_RATES = (0.25, 0.5, 1.0)


def _belief() -> ParameterBelief:
    return ParameterBelief(names=["rate"],
                           scales=["linear"],
                           map_estimate={"rate": 0.5},
                           lines={},
                           noise_scale=1.0,
                           draws=np.array([[r] for r in _RATES]))


def _install(monkeypatch, approach, deployed):
    """Give the approach a memory model, a deployed point and a belief."""
    monkeypatch.setattr(approach,
                        "model_state_revision",
                        lambda: ("memory", tuple(deployed.items())),
                        raising=False)

    def make_tracker(params=None):
        values = dict(deployed) if params is None else dict(params)
        return make_subclass_latent_tracker(_MemoryModel, lambda: values)

    monkeypatch.setattr(approach, "make_latent_tracker", make_tracker)
    monkeypatch.setattr(approach, "parameter_belief", _belief, raising=False)


def test_joint_draws_carry_each_draws_memory(tmp_path, monkeypatch):
    """Draw i's memory is the prefix replayed under draw i's parameters; the
    draws are fixed per decision point and move with it."""
    monkeypatch.setattr(observation_noise, "POSITION_FEATURES",
                        frozenset({"pose"}))
    _config(tmp_path,
            "oracle",
            continual_obs_noise_position=.01,
            continual_belief_frame=True,
            belief_joint_draws=3,
            num_test_tasks=1)
    env, approach = _make("oracle")
    _install(monkeypatch, approach, {"rate": 0.5})

    class Controller:
        """Check the draws through the protocol session."""

        def play_level(self, session):
            """Two steps, reading the joint draws after each."""
            action = Action(np.zeros(env.action_space.shape, dtype=np.float32))
            session.step(action)
            draws = session.joint_draws(3)
            assert [theta["rate"] for theta, _ in draws] == list(_RATES)
            assert [s.latent["charge"] for _, s in draws] == list(_RATES)
            # The observation still carries the deployed point's memory.
            assert session.observe().frame.latent["charge"] == 0.5
            again = session.joint_draws(3)
            for (_, first), (_, second) in zip(draws, again):
                assert first.allclose(second)
            session.step(action)
            later = session.joint_draws(3)
            assert [s.latent["charge"] for _, s in later] == \
                [2 * r for r in _RATES]
            # Fresh draws come from the belief's lines; this belief has
            # none, so every fresh draw sits at the most likely value.
            fresh = session.joint_draws(3, fresh=True)
            assert [theta["rate"] for theta, _ in fresh] == [0.5] * 3
            assert [s.latent["charge"] for _, s in fresh] == [1.0] * 3
            fractions = session.joint_atom_fractions(3)
            assert fractions and all(0.0 < f <= 1.0
                                     for f in fractions.values())
            raise _Preempted()

    with pytest.raises(_Preempted):
        ContinualRun(env, approach, Controller()).run()


def test_joint_draws_without_a_belief_use_the_deployed_point(
        tmp_path, monkeypatch):
    """An approach without a parameter belief gets state draws at its deployed
    values; an exact channel leaves the observed frame."""
    _config(tmp_path, "oracle", belief_joint_draws=2, num_test_tasks=1)
    env, approach = _make("oracle")

    class Controller:
        """Read the draws with no noise and no belief."""

        def play_level(self, session):
            """The draws repeat the observed frame."""
            draws = session.joint_draws(2)
            frame = session.observe().frame
            assert [theta for theta, _ in draws] == [{}, {}]
            assert all(s.allclose(frame) for _, s in draws)
            prefix, labels = session.episode_prefix()
            assert len(prefix) == 1 and labels == []
            raise _Preempted()

    del monkeypatch
    with pytest.raises(_Preempted):
        ContinualRun(env, approach, Controller()).run()


def test_a_rolling_body_keeps_its_estimated_velocity(tmp_path):
    """Under continual_belief_velocities and a noisy channel, the agent's views
    carry no body velocity, while the observation frame and the joint draws
    carry the belief's estimate of the rolling ball's, so a rollout staged from
    them rolls on."""
    utils.reset_config({
        "env":
        "pybullet_fan",
        "approach":
        "oracle",
        "seed":
        0,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "experiment_protocol":
        "continual",
        "continual_render":
        False,
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
        "experiment_id":
        "test",
        "fan_exposed_transfer":
        True,
        "fan_ramp_transfer":
        False,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
        "continual_obs_noise_position":
        0.002,
        "continual_belief_frame":
        True,
        "belief_joint_draws":
        4,
        "continual_belief_velocities":
        True,
    })
    env = PyBulletFanEnv(use_gui=False)
    approach = create_approach("oracle", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])

    class Controller:
        """Roll the ball, then read the frame and the draws."""

        def play_level(self, session):
            """Push the ball in the level's engine and hold the robot."""
            world = session._run._level_env
            ball = world._ball
            p.resetBaseVelocity(ball.id, (0.08, 0.0, 0.0),
                                physicsClientId=world._physics_client_id)
            for _ in range(16):
                frame = session.observe().frame
                session.step(
                    Action(np.asarray(frame.joint_positions,
                                      dtype=np.float32)))
            true = p.getBaseVelocity(
                ball.id, physicsClientId=world._physics_client_id)[0]
            assert all("body_velocities" not in (view.simulator_state or {})
                       for view in session._run._observed_views.values())
            belief = session._run.belief()
            mean, error = belief.velocities[ball.name]
            assert abs(mean[0] - true[0]) < 4 * error[0]
            frame = session.observe().frame
            staged = frame.simulator_state["body_velocities"][ball.name]
            assert np.allclose(staged[0], mean)
            assert staged[1] == (0.0, 0.0, 0.0)
            draws = [s for _, s in session.joint_draws(4)]
            spread = [
                s.simulator_state["body_velocities"][ball.name][0][0]
                for s in draws
            ]
            assert len(set(spread)) == 4
            assert all(abs(v - mean[0]) < 5 * error[0] for v in spread)
            raise _Preempted()

    try:
        with pytest.raises(_Preempted):
            ContinualRun(env, approach, Controller()).run()
    finally:
        p.disconnect(env._physics_client_id)
