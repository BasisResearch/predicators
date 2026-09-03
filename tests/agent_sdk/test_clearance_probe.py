"""Tests for the capture gate's robot-clearance probe (tools/clearance.py).

Regression for the 2026-09-02 bridge seed3 rerun: two belief-certified
explore plans (8/8 validation rollouts each) died on real contacts of
6.5 mm and 9.9 mm against a block every rollout had cleared - the gate
certified against the belief's own variability but never measured how
close the robot came to a bystander, so plans tighter than the
executor's realization slop passed by the luck of the draw.
"""
from collections import namedtuple
from types import SimpleNamespace
from typing import List, cast

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
    probe.min_dist = 0.0042
    probe.where = "rollout 2, step PickBlock(span2) vs span1"
    ok, summary, detail = probe.verdict()
    assert not ok
    assert "4.2 mm" in summary and "10 mm" in summary
    assert "span1" in detail and "inside the executor's 10 mm" in detail
    probe.min_dist = 0.0123
    ok, summary, detail = probe.verdict()
    assert ok and detail == "" and "12.3 mm" in summary


def test_verdict_without_probes_or_approach_is_ok() -> None:
    """No probes, or nothing within the query distance, is a pass."""
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


_Obj = namedtuple("_Obj", ["name"])  # hashable stand-in for an Object


class _RecordingProbe(RobotClearanceProbe):
    """Records the exempt set handed to each clearance query."""

    def __init__(self, skill, held: str = "") -> None:
        super().__init__(skill, stride=1)
        self.exempts: List[set] = []
        self._held = held

    def _min_robot_clearance(self, state, exempt):
        self.exempts.append(set(exempt))
        return 0.02, "wall"

    def _held_object_name(self, state):
        return self._held or None


def _grounded(name: str, objects, skill=None) -> _Option:
    parent = SimpleNamespace(policy=SimpleNamespace(__self__=skill))
    return cast(_Option,
                SimpleNamespace(name=name, parent=parent, objects=objects))


def test_observe_exempts_declared_contacts_and_the_held_object() -> None:
    """A push's switch (skill-declared contact) and the object held when the
    option starts join the option's arguments in the exempt set; a skill
    without the hook, or a failing hook, exempts only the arguments and the
    held object."""
    faucet = _Obj("faucet")
    switch = _Obj("faucet_switch")
    push_skill = SimpleNamespace(
        _config=SimpleNamespace(move_to_pose_tol=1e-4, simulator=None),
        contact_objects=lambda state, objects: {switch})
    probe = _RecordingProbe(_FakeSkill(1e-4))
    probe.observe("rollout 1", _grounded("SwitchOn", [faucet], push_skill),
                  ["s0", "s1"])
    assert probe.exempts == [{"faucet", "faucet_switch"}] * 2
    # A Place holds domino_5 when it starts: the released domino is
    # exempt for the whole option (its retreat passes it by design).
    probe = _RecordingProbe(_FakeSkill(1e-4), held="domino_5")
    probe.observe("rollout 1", _grounded("Place", [], _FakeSkill(1e-4)),
                  ["s0"])
    assert probe.exempts == [{"domino_5"}]
    # A failing contact hook is best-effort: only the arguments remain.
    bad_skill = SimpleNamespace(contact_objects=lambda s, o: 1 / 0)
    probe = _RecordingProbe(_FakeSkill(1e-4))
    probe.observe("rollout 1", _grounded("Push", [faucet], bad_skill), ["s0"])
    assert probe.exempts == [{"faucet"}]
    assert probe.min_dist == 0.02 and "vs wall" in probe.where


def test_push_contact_object_is_the_body_at_the_target_pose() -> None:
    """object_at_pose picks the posed object nearest the push target within the
    contact radius, skipping the grounding's own objects and pose-less
    objects."""
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models.skill_factories.push import \
        object_at_pose
    from predicators.structs import Object, State, Type
    posed = Type("posed", ["x", "y", "z"])
    plain = Type("plain", ["is_on"])
    robot = Object("robot", posed)
    faucet = Object("faucet", plain)
    switch = Object("faucet_switch", posed)
    other = Object("burner_switch0", posed)
    state = State({
        robot: np.array([1.0, 1.45, 0.7]),
        faucet: np.array([0.0]),
        switch: np.array([1.0, 1.45, 0.65]),
        other: np.array([0.6, 1.3, 0.65]),
    })
    target = (1.0, 1.452, 0.65)
    assert object_at_pose(state, target, {robot, faucet}) == switch
    # The robot is the nearest posed body but is excluded.
    assert object_at_pose(state, (1.0, 1.45, 0.69), {robot, faucet}) \
        == switch
    # Nothing within the radius: no contact target.
    assert object_at_pose(state, (2.0, 2.0, 0.65), {robot, faucet}) is None


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
        # Park span1 right under the gripper: a real, small clearance
        # (the robot's z is the fingertip point, the block sits 9 cm
        # below it, i.e. ~3 cm from the finger geometry).
        near = init.copy()
        for feat, val in (("x", init.get(robot,
                                         "x")), ("y", init.get(robot, "y")),
                          ("z", init.get(robot, "z") - 0.09)):
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
        assert not probe.where.endswith("vs span1")
    finally:
        p.disconnect(env._physics_client_id)  # pylint: disable=protected-access
