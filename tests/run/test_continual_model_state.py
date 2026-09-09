"""Subclass memory in the real protocol, apart from raw recordings."""
# pylint: disable=protected-access
import numpy as np
import pytest

from predicators import observation_noise
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.run.continual import ContinualRun
from predicators.structs import Action
from tests.code_sim_learning.test_subclass_model_state import _MemoryModel
from tests.run.test_continual import _config, _make, _Preempted


def test_model_memory_replayed_after_fit_and_reset(tmp_path, monkeypatch):
    """Observed history initializes a revised model without changing data."""
    monkeypatch.setattr(observation_noise, "POSITION_FEATURES",
                        frozenset({"pose"}))
    _config(tmp_path,
            "oracle",
            continual_obs_noise_position=.01,
            continual_belief_frame=True,
            num_test_tasks=1)
    env, approach = _make("oracle")
    params = {"rate": .25}
    monkeypatch.setattr(approach,
                        "model_state_revision",
                        lambda: tuple(params.items()),
                        raising=False)
    monkeypatch.setattr(
        approach, "make_latent_tracker",
        lambda: make_subclass_latent_tracker(_MemoryModel, lambda: params))

    class Controller:
        """Exercise free observations, actions, fitting and resets."""

        def play_level(self, session):
            """Leave a resumable episode after checking the real API."""
            assert session.observe().frame.latent["charge"] == 0
            action = Action(np.zeros(env.action_space.shape, dtype=np.float32))
            session.step(action)
            assert session.observe().frame.latent["charge"] == .25
            assert session.observe().frame.latent["charge"] == .25
            assert session.observe_truth().frame.latent is None
            assert all(s.latent is None for ep in session.level_episodes()
                       for s in ep["states"])
            # A new fit changes the inferred history as well as future steps.
            params["rate"] = .5
            obs = session.observe()
            assert obs.frame.latent["charge"] == .5
            assert obs.belief.frame.latent == obs.frame.latent
            session.step(action)
            assert session.observe().frame.latent["charge"] == 1.0
            session.reset("test memory reset")
            assert session.observe().frame.latent["charge"] == 0
            session.step(action)
            assert session.observe().frame.latent["charge"] == .5
            raise _Preempted()

    with pytest.raises(_Preempted):
        ContinualRun(env, approach, Controller()).run()

    _config(tmp_path,
            "oracle",
            continual_obs_noise_position=.01,
            continual_belief_frame=True,
            num_test_tasks=1,
            auto_resume=True)
    env2, approach2 = _make("oracle")
    monkeypatch.setattr(approach2, "model_state_revision",
                        lambda: tuple(params.items()))
    monkeypatch.setattr(
        approach2, "make_latent_tracker",
        lambda: make_subclass_latent_tracker(_MemoryModel, lambda: params))

    class ResumeController:
        """Verify memory is reconstructed from the resumed episode only."""

        def play_level(self, session):
            """Read the resumed observation without advancing the model."""
            assert session.observe().frame.latent["charge"] == .5
            assert all(s.latent is None for ep in session.level_episodes()
                       for s in ep["states"])
            raise _Preempted()

    with pytest.raises(_Preempted):
        ContinualRun(env2, approach2, ResumeController()).run()
