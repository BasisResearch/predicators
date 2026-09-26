"""Bridge model restoration preserves carried assembly geometry."""
# pylint: disable=protected-access
from typing import Any

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.envs import create_new_env
from predicators.observation_noise import ObservationNoise, step_rng
from predicators.run.recording import sanitize_state
from predicators.structs import Action
from tests.code_sim_learning.test_continual_oracle import _load
from tests.envs.test_pybullet_bridge import _stage_flush_pair


@pytest.mark.parametrize("held", [False, True])
def test_tilted_assembly_restore_preserves_geometry(held: bool) -> None:
    """A current-state rehearsal must not flatten an already welded beam."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "partially_observable": True,
        "bridge_test_span_blocks": 4
    })
    real: Any = create_new_env("pybullet_bridge", do_cache=False)
    model = _load()(use_gui=False)
    try:
        state = sanitize_state(real.get_test_tasks()[0].init)
        spans = sorted(o for o in state if o.name.startswith("span"))
        rotation = np.array(
            p.getMatrixFromQuaternion(p.getQuaternionFromEuler(
                (0., .2, .1)))).reshape(3, 3)
        records = {}
        for i, block in enumerate(spans):
            position = np.array([.4, 1.1, .6
                                 ]) + rotation @ np.array([.1 * i, 0., 0.])
            for feature, value in zip(("x", "y", "z", "roll", "pitch", "yaw"),
                                      [*position, 0., .2, .1]):
                state.set(block, feature, value)
            records[block.name] = {
                "attached_end_a": spans[i - 1].name if i else None,
                "attached_end_b":
                spans[i + 1].name if i + 1 < len(spans) else None
            }
        state.latent = {"blocks": records}
        if held:
            state.set(spans[0], "is_held", 1.)
        for _ in range(2):
            model._set_state(state)
            restored = model._get_state()
            assert len(model._weld_constraints) == 3
            assert (model._held_constraint_id is not None) == held
            for block in spans:
                for feature in ("x", "y", "z", "roll", "pitch", "yaw"):
                    assert np.isclose(
                        state.get(block, feature),
                        restored.get(block, feature),
                        atol=1e-5), (block, feature, state.get(block, feature),
                                     restored.get(block, feature))
            model._set_state(sanitize_state(real.get_train_tasks()[0].init))
            assert not model._weld_constraints
            assert model._held_constraint_id is None
        # A prediction snapshot owns its original constraint frames. They
        # must survive a fresh world even when the body poses are deflected.
        model._set_state(state)
        model._domain_specific_step()
        snapshot = model._get_state().copy()
        expected = snapshot.latent["weld_frames"]
        assert len(expected) == 3
        snapshot.set(spans[-1], "x", snapshot.get(spans[-1], "x") + .002)
        fresh = type(model)(use_gui=False)
        try:
            fresh._set_state(snapshot)
            actual = [
                p.getConstraintInfo(c,
                                    physicsClientId=fresh._physics_client_id)
                for c in fresh._weld_constraints.values()
            ]
            assert len(actual) == len(expected)
            for saved, restored_constraint in zip(expected, actual):
                assert np.allclose(saved["parent_frame"][0],
                                   restored_constraint[6])
                assert np.allclose(saved["parent_frame"][1],
                                   restored_constraint[8])
                assert np.allclose(saved["child_frame"][0],
                                   restored_constraint[7])
                assert np.allclose(saved["child_frame"][1],
                                   restored_constraint[9])
        finally:
            fresh.dispose()
    finally:
        real.dispose()
        model.dispose()


def test_glue_consumption_recovers_joint_from_noisy_history() -> None:
    """Visible glue consumption certifies a latch despite noisy contact
    gaps."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "partially_observable": True,
        "continual_obs_noise_position": .005,
        "continual_obs_noise_orientation": .02
    })
    real: Any = create_new_env("pybullet_bridge", do_cache=False)
    try:
        left, right = _stage_flush_pair(
            real,
            real.get_train_tasks()[0])  # type: ignore[no-untyped-call]
        tracker = make_subclass_latent_tracker(_load(), lambda: {})
        assert tracker is not None
        noise = ObservationNoise.from_cfg()
        tracker.attach(noise.perturb(real._get_state(), step_rng(2, 1, 0, 0)),
                       None)
        for index in range(35):
            action = Action(
                np.array(real._pybullet_robot.get_joints(), dtype=np.float32))
            real._step_once(action)
            inferred = tracker.attach(
                noise.perturb(real._get_state(), step_rng(2, 1, 0, index + 1)),
                action)
        assert len(real._weld_constraints) == 1
        assert not tracker.failed
        assert inferred.latent is not None
        assert inferred.latent["blocks"][
            left.name]["attached_end_b"] == right.name
    finally:
        real.dispose()
