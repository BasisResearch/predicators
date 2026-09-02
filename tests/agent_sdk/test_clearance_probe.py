"""Tests for the capture gate's robot-clearance probe (tools/clearance.py).

Regression for the 2026-09-02 bridge seed3 rerun: two belief-certified
explore plans (8/8 validation rollouts each) died on real contacts of
6.5 mm and 9.9 mm against a block every rollout had cleared - the gate
certified against the belief's own variability but never measured how
close the robot came to a bystander, so plans tighter than the
executor's realization slop passed by the luck of the draw.
"""
from types import SimpleNamespace
from typing import cast

import numpy as np
import pybullet as p

from predicators import utils
from predicators.agent_sdk.tools.clearance import RobotClearanceProbe, \
    clearance_lines, phase_skill_of
from predicators.structs import _Option


class _FakeSkill:
    """A stand-in with only the config the verdict reads."""

    def __init__(self, tol: float) -> None:
        self._config = SimpleNamespace(move_to_pose_tol=tol, simulator=None)


def test_verdict_uses_executor_pose_slop() -> None:
    """The bar is sqrt(move_to_pose_tol): 1e-4 -> 10 mm."""
    probe = RobotClearanceProbe(_FakeSkill(1e-4))
    assert abs(probe.threshold - 0.01) < 1e-12
    probe.num_probes = 5
    probe.min_dist, probe.where = 0.0042, "rollout 2, step PickBlock(span2) vs span1"
    ok, summary, detail = probe.verdict()
    assert not ok
    assert "4.2 mm" in summary and "10 mm" in summary
    assert "span1" in detail and "inside the executor's 10 mm" in detail
    probe.min_dist = 0.0123
    ok, summary, detail = probe.verdict()
    assert ok and detail == "" and "12.3 mm" in summary


def test_verdict_without_probes_or_approach_is_ok() -> None:
    probe = RobotClearanceProbe(_FakeSkill(1e-4))
    assert probe.verdict() == (True, "", "")
    assert not clearance_lines(probe)
    assert not clearance_lines(None)
    probe.num_probes = 3  # nothing came within the query distance
    ok, summary, detail = probe.verdict()
    assert ok and detail == "" and summary.startswith("min robot clearance: >")


def test_phase_skill_of_requires_a_planning_simulator() -> None:
    """Only a skill-factory option with a planning simulator qualifies."""
    no_sim = cast(
        _Option,
        SimpleNamespace(parent=SimpleNamespace(policy=SimpleNamespace(
            __self__=_FakeSkill(1e-4)))))
    assert phase_skill_of([no_sim]) is None
    plain = cast(
        _Option,
        SimpleNamespace(parent=SimpleNamespace(policy=lambda s: None)))
    assert phase_skill_of([plain]) is None


def test_bridge_probe_measures_and_exempts(tmp_path) -> None:
    """On the bridge env's own planning simulator: a block moved under the
    gripper reads as a small clearance, the option's argument objects are
    exempt, and a far scene reads as beyond the query distance."""
    del tmp_path
    utils.reset_config({
        "env": "pybullet_bridge",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 0,
        "skill_phase_use_motion_planning": True,
    })
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
    from predicators.ground_truth_models import get_gt_options
    env = PyBulletBridgeEnv(use_gui=False)
    try:
        task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
        options = {o.name: o for o in get_gt_options(env.get_name())}
        robot = env._robot  # pylint: disable=protected-access
        span1 = next(b for b in env._blocks if b.name == "span1")  # pylint: disable=protected-access
        pick = options["PickBlock"].ground([robot, span1],
                                           np.array([0.0], dtype=np.float32))
        skill = phase_skill_of([pick])
        assert skill is not None
        probe = RobotClearanceProbe(skill, stride=1)
        init = task.init
        # Far scene: nothing within the query distance of the home pose.
        far, _ = probe._min_robot_clearance(init, set())  # pylint: disable=protected-access
        assert far >= 0.05
        # Park span1 right under the gripper: a real, small clearance.
        near = init.copy()
        for feat, val in (("x", init.get(robot,
                                         "x")), ("y", init.get(robot, "y")),
                          ("z", init.get(robot, "z") - 0.12)):
            near.set(span1, feat, val)
        dist, body = probe._min_robot_clearance(near, set())  # pylint: disable=protected-access
        assert body == "span1" and dist < 0.05
        # The pick's own target is exempt: the measurement moves on.
        dist_exempt, body_exempt = probe._min_robot_clearance(  # pylint: disable=protected-access
            near, {"span1"})
        assert body_exempt != "span1" and dist_exempt >= dist
        # observe() on the pick's trajectory records the closest approach
        # against non-argument bodies only.
        probe.observe("rollout 1", pick, [near])
        assert probe.num_probes == 1
        assert "span1" not in probe.where
    finally:
        p.disconnect(env._physics_client_id)  # pylint: disable=protected-access
