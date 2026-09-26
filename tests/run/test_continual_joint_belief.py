"""The session's joint draws of the belief (paper Section 3.2).

Each joint draw pairs one draw of the approach's parameter belief with a
draw of the state factor whose memory is the one those parameters imply
over the observed episode prefix; the draws are fixed per decision
point, and monitoring reads atoms on them.
"""
# pylint: disable=protected-access
import numpy as np
import pytest

from predicators import observation_noise
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.code_sim_learning.parameter_belief import ParameterBelief
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
