"""Learned attachment memory must exist before a controller plans a lift."""
# pylint: disable=protected-access
from typing import Any

import numpy as np

from predicators import utils
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.envs import create_new_env
from predicators.run.recording import sanitize_state
from predicators.structs import Action


def test_learned_attachments_restore_before_first_action(
        monkeypatch: Any) -> None:
    """An observation-only held beam survives fresh reconstruction, no tick."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "bridge_test_span_blocks": 4,
        "partially_observable": True
    })
    base: Any = base_simulator_class("pybullet_bridge")

    class LearnedLinks(base):
        """A candidate owns its links; the harness supplies no glue rules."""
        AGENT_PARAM_SPECS: Any = []
        MODEL_STATE_INIT: Any = {"links": []}

        def restore_model_state(self) -> None:
            """Re-attach the links this candidate recorded."""
            self.restore_model_attachments(self.model_state["links"])

    real: Any = create_new_env("pybullet_bridge", do_cache=False)
    model = LearnedLinks(use_gui=False)
    try:
        state = sanitize_state(real.get_test_tasks()[0].init)
        spans = sorted(o for o in state if o.name.startswith("span"))
        for i, block in enumerate(spans):
            state.set(block, "x", .4 + .10 * i)
            state.set(block, "y", 1.1)
            state.set(block, "z", .6 + .02 * i)
            state.set(block, "pitch", -.2)
        state.set(spans[0], "is_held", 1.)
        state.latent = {
            "links": [[a.name, b.name] for a, b in zip(spans, spans[1:])]
        }
        # Reproduce the old observation-only lift entry point: inferred
        # links do not exist yet without the explicit restoration hook.
        with monkeypatch.context() as patch:
            patch.setattr(LearnedLinks, "restore_model_state",
                          lambda self: None)
            model._set_state(state)
            ids = model._residual_command_body_ids()
            assert not model.get_welded_partner_ids(ids[spans[0].name])
        for _ in range(2):
            model._set_state(state)
            ids = model._residual_command_body_ids()
            assert model.get_welded_partner_ids(
                ids[spans[0].name]) == {ids[o.name]
                                        for o in spans[1:]}
            restored = model._get_state()
            for obj in spans:
                for feature in ("x", "y", "z", "pitch", "is_held"):
                    assert np.isclose(state.get(obj, feature),
                                      restored.get(obj, feature),
                                      atol=1e-5)
            assert restored.latent == state.latent
            model._set_state(sanitize_state(real.get_train_tasks()[0].init))
            assert not model._cmd_weld_constraints
        model._set_state(state)
        snapshot = model._get_state().copy()
        expected_frames = model._command_weld_frame_records()
        snapshot.set(spans[-1], "x", snapshot.get(spans[-1], "x") + .002)
        fresh = LearnedLinks(use_gui=False)
        try:
            fresh._set_state(snapshot)
            assert fresh._command_weld_frame_records() == expected_frames
        finally:
            fresh.dispose()
        # Model-owned links persist across physics ticks even without an
        # Attach residual command being re-emitted on every action.
        model._step_once(
            Action(
                np.array(model._pybullet_robot.get_joints(),
                         dtype=np.float32)))
        assert len(model._cmd_weld_constraints) == 3
        # Invalid memory is surfaced rather than silently dropping a joint.
        state.latent = {"links": [[spans[0].name, "missing"]]}
        try:
            model._set_state(state)
        except ValueError as err:
            assert "missing" in str(err)
        else:
            raise AssertionError("Unknown attachment must not be ignored")
    finally:
        model.dispose()
        real.dispose()
