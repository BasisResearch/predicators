"""Audit the supplied dynamics on a longer, observed-only Bridge chain."""
# pylint: disable=protected-access
import shlex
import sys
from typing import Any

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.envs import create_new_env
from predicators.run.recording import sanitize_state
from predicators.structs import Action
from scripts.cluster_utils import SingleSeedRunConfig, config_to_cmd_flags, \
    generate_run_configs
from tests.code_sim_learning.test_continual_oracle import _load


def test_transfer_comparisons_preserve_arm_contracts(monkeypatch: Any) -> None:
    """All six arms parse with the pilot's task, noise, and interaction
    budget."""
    original = list(
        generate_run_configs(
            "predicatorv3/protocol_continual_comparisons_noisy_r1.yaml",
            False))
    transfer = list(
        generate_run_configs(
            "predicatorv3/protocol_continual_bridge_span_comparisons_r1.yaml",
            False))
    assert len(transfer) == 18
    assert len({cfg.approach for cfg in transfer}) == 6
    for cfg in transfer:
        assert isinstance(cfg, SingleSeedRunConfig)
        assert cfg.env == "pybullet_bridge"
        assert cfg.seed in (0, 1, 2)
        reference = next(c for c in original
                         if c.env == cfg.env and c.approach == cfg.approach)
        assert cfg.flags == {
            **reference.flags, "bridge_train_span_blocks": 3,
            "bridge_test_span_blocks": 4
        }
        monkeypatch.setattr(
            sys, "argv",
            ["predicators/main.py", *shlex.split(config_to_cmd_flags(cfg))])
        parsed = utils.parse_args()
        assert parsed["bridge_test_span_blocks"] == 4
        assert parsed["approach"] == cfg.approach


def test_four_span_oracle_cures_all_three_joints() -> None:
    """Native physics, exported dynamics, and public memory agree on all
    joints."""
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "partially_observable": True,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "bridge_train_span_blocks": 3,
        "bridge_test_span_blocks": 4
    })
    real: Any = create_new_env("pybullet_bridge", do_cache=False)
    model: Any = None
    try:
        initial = real.reset("test", 0).copy()
        spans = sorted(o for o in initial if o.name.startswith("span"))
        assert len(spans) == 4
        table_z = initial.get(spans[0], "z")
        for i, block in enumerate(spans):
            for feature, value in (("x", .4 + i * .1), ("y", 1.14), ("z",
                                                                     table_z),
                                   ("roll", 0.), ("pitch", 0.), ("yaw", 0.)):
                initial.set(block, feature, value)
        for i, block in enumerate(real._legs):
            initial.set(block, "x", 2.0 + i * .2)
            initial.set(block, "y", 2.0)
        real._set_state(initial)
        for block in spans[:-1]:
            real._set_attr(block, "glue_end_b", 1.)
        clean = sanitize_state(real._get_state())
        cls = _load()
        model = cls(use_gui=False, skip_residual_dynamics=False)
        model._set_state(clean)
        tracker = make_subclass_latent_tracker(cls, lambda: {})
        assert tracker is not None
        tracker.attach(clean, None)
        inferred = clean
        position_errors = []
        orientation_errors = []
        for _ in range(35):
            action = Action(
                np.array(real._pybullet_robot.get_joints(), dtype=np.float32))
            real._step_once(action)
            model._step_once(action)
            actual, predicted = real._get_state(), model._get_state()
            inferred = tracker.attach(sanitize_state(actual), action)
            assert not tracker.failed
            for block in spans:
                position_errors.append(
                    float(
                        np.linalg.norm([
                            actual.get(block, f) - predicted.get(block, f)
                            for f in ("x", "y", "z")
                        ])))
                qa = p.getQuaternionFromEuler(
                    [actual.get(block, f) for f in ("roll", "pitch", "yaw")])
                qp = p.getQuaternionFromEuler([
                    predicted.get(block, f) for f in ("roll", "pitch", "yaw")
                ])
                orientation_errors.append(
                    float(2 * np.arccos(np.clip(abs(np.dot(qa, qp)), 0., 1.))))
                for feature in ("glue_top", "glue_end_a", "glue_end_b"):
                    assert actual.get(block, feature) == predicted.get(
                        block, feature)
            assert len(model._weld_constraints) == len(real._weld_constraints)
        print("chain-prediction",
              max(position_errors),
              max(orientation_errors),
              flush=True)
        # Public-state restoration omits contact solver history. Bound the
        # resulting pose drift by the pilot's declared observation scales,
        # with meters and radians checked separately; mechanisms stay exact.
        assert max(position_errors) < .005
        assert max(orientation_errors) < .02
        assert len(real._weld_constraints) == 3
        assert inferred.latent is not None
        records = inferred.latent["blocks"]
        for left, right in zip(spans, spans[1:]):
            assert records[left.name]["attached_end_b"] == right.name
            assert records[right.name]["attached_end_a"] == left.name
        model._set_state(inferred)
        assert len(model._weld_constraints) == 3
        model._set_state(sanitize_state(real.get_train_tasks()[0].init))
        assert not model._weld_constraints
        assert len(model._get_state().get_objects(real._block_type)) == 5
    finally:
        p.disconnect(real._physics_client_id)
        if model is not None:
            p.disconnect(model._physics_client_id)
