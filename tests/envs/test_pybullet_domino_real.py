"""Tests for PyBulletDominoRealEnv's perception -> State / Task conversions,
and for the PyBulletEnv primitives the real-world wrapper builds on.

These are pure conversions plus one simulator write: no hardware, no robot, and
**no babyrobot**. That is deliberate and is asserted below -- a suite that
silently skipped without the private submodule would hide exactly the
regressions these tests exist to catch.

The scene fixture is synthetic and tiny so the expected world poses can be
worked out by hand. The base -> world transplant is a rigid yaw(+pi/2) plus a
translation, i.e.

    world_x = 0.75 - base_y
    world_y = 0.72 + base_x
    world_z = base_z + z_off        (z_off = 0.4 - domino_real_table_z)

so every coordinate assertion below is checkable without running the code.
"""
import ast
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs.pybullet_domino.real_geometry import Pose6D, \
    domino_upright_yaw, pose_base_to_world
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.pybullet_helpers.real_robot_bridge import \
    gripper_joint_layout_from_robot
from predicators.structs import GroundAtom

_TABLE_Z = -0.041
_Z_OFF = 0.4 - _TABLE_Z  # 0.441
_START_ID = 6
_TARGET_ID = 5
# Identity orientation in the base frame. After the +pi/2 world yaw the body-y
# (width) axis points along world -x, so domino_upright_yaw reads pi.
_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]
_YAW_FROM_IDENTITY = np.pi


def _record(capture_id: int,
            base_xy: Tuple[float, float],
            *,
            quat: Optional[List[float]] = None,
            role: Optional[str] = None) -> Dict[str, Any]:
    """One scene-JSON domino record, in the shape a real capture emits."""
    rec: Dict[str, Any] = {
        "id": capture_id,
        "center_base_m": [base_xy[0], base_xy[1], 0.03],
        "quat_base_xyzw": list(quat if quat is not None else _IDENTITY_QUAT),
        "dims_m": [0.15, 0.07, 0.029],
    }
    if role is not None:
        rec["role"] = role
    return rec


def _write_scene(tmp_path,
                 records,
                 start_push_dir_base=None,
                 name="scene.json"):
    """Write a scene JSON and return its path."""
    scene = {"frame": "robot_base", "units": "m", "dominoes": records}
    if start_push_dir_base is not None:
        scene["start_push_dir_base"] = start_push_dir_base
    path = tmp_path / name
    path.write_text(json.dumps(scene), encoding="utf-8")
    return str(path)


class _StubDominoPose:
    """Stands in for babyrobot's DominoPose.

    The env reads observations structurally (``id`` / ``xyz`` /
    ``quat_xyzw``), which is what lets these tests run with the private
    submodule absent.
    """

    def __init__(self, capture_id, xyz, quat_xyzw=None):
        self.id = capture_id
        self.xyz = tuple(xyz)
        self.quat_xyzw = tuple(
            quat_xyzw if quat_xyzw is not None else _IDENTITY_QUAT)


class _StubDominoObservation:
    """Stands in for babyrobot's DominoObservation."""

    def __init__(self, dominoes, stamp=0.0):
        self.stamp = stamp
        self.dominoes = list(dominoes)


def _config(scene_path):
    """Apply the config this env is actually run with."""
    utils.reset_config({
        "env": "pybullet_domino_real",
        "domino_real_scene": scene_path,
        "domino_real_table_z": _TABLE_Z,
        "domino_real_start_id": _START_ID,
        "domino_real_target_id": _TARGET_ID,
        # The env's goal semantics assume targets are domino blocks; every
        # shipped config for it sets this.
        "domino_use_domino_blocks_as_target": True,
        "domino_use_skill_factories": False,
        "domino_real_decorate": False,
    })


def _make_env(scene_path):
    """Build the env against ``scene_path``."""
    _config(scene_path)
    return PyBulletDominoRealEnv(use_gui=False)


# The default scene: start and target 20cm apart along base +x, plus two
# movable blocks. Slot order follows scene order, so slot i <-> _SCENE_IDS[i].
_SCENE_RECORDS = [
    _record(_START_ID, (0.0, 0.0)),
    _record(11, (0.1, 0.05)),
    _record(12, (0.15, -0.05)),
    _record(_TARGET_ID, (0.2, 0.0)),
]
_SCENE_IDS = [_START_ID, 11, 12, _TARGET_ID]


@pytest.fixture(scope="module", name="scene_path")
def scene_path_fixture(tmp_path_factory):
    """The default scene JSON, written once for the module."""
    return _write_scene(tmp_path_factory.mktemp("domino_real"), _SCENE_RECORDS)


@pytest.fixture(scope="module", name="env")
def env_fixture(scene_path):
    """One env for the module -- building PyBullet per test is slow."""
    return _make_env(scene_path)


@pytest.fixture(autouse=True)
def _reapply_config(scene_path):
    """Re-apply the config before each test, since CFG is global."""
    _config(scene_path)


def test_env_has_no_module_level_babyrobot_import():
    """These conversions are pure predicators code, so this file runs on a
    checkout without the private submodule -- and must never skip.

    Checked on the parse tree rather than on ``sys.modules``, which
    another test in the same session may legitimately have populated.
    """
    source = inspect.getsourcefile(PyBulletDominoRealEnv)
    assert source is not None
    with open(source, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("babyrobot"), \
                f"babyrobot imported at module level: {name}"


# -- scene loading and roles -------------------------------------------------


def test_scene_ids_follow_scene_order(env):
    """Slot i holds capture id scene_ids[i]; that list is the id -> slot
    map."""
    assert env._scene_ids == _SCENE_IDS  # pylint: disable=protected-access


def test_domino_role_prefers_explicit_role_over_id(env):
    """A scene carrying explicit roles is trusted; a raw capture is keyed by
    id."""
    # pylint: disable=protected-access
    # Explicit role wins even when the id says otherwise.
    assert env._domino_role({"id": _START_ID, "role": "movable"}) == "movable"
    assert env._domino_role({"id": 999, "role": "target"}) == "target"
    # No role field: fall back to the configured ids.
    assert env._domino_role({"id": _START_ID}) == "start"
    assert env._domino_role({"id": _TARGET_ID}) == "target"
    assert env._domino_role({"id": 11}) == "movable"


def test_role_counts_from_scene(env):
    """(num_target, num_nontarget) drives the component's slot allocation."""
    del env  # counts are read from CFG's scene, not the instance
    n_target, n_nontarget = PyBulletDominoRealEnv._scene_role_counts()  # pylint: disable=protected-access
    assert (n_target, n_nontarget) == (1, 3)


# -- base -> world transplant ------------------------------------------------


def test_base_to_world_matches_hand_computed_values():
    """The transplant is yaw(+pi/2) then translate; pin it to known numbers."""
    world = pose_base_to_world(Pose6D((0.2, 0.1, 0.03), tuple(_IDENTITY_QUAT)),
                               _Z_OFF)
    assert world.xyz == pytest.approx((0.75 - 0.1, 0.72 + 0.2, 0.03 + _Z_OFF))
    # Identity in base -> body-y along world -x -> yaw = pi.
    assert domino_upright_yaw(world) == pytest.approx(_YAW_FROM_IDENTITY)


# -- task construction: the two sources must agree ---------------------------


def test_scene_and_observation_tasks_agree(env, scene_path):
    """The captured scene and a live observation of the SAME poses build the
    same task.

    This is the anti-drift property the refactor exists for: both route
    through one conversion.
    """
    scene_task = env._build_task_from_scene()  # pylint: disable=protected-access
    records = json.loads(open(scene_path, encoding="utf-8").read())["dominoes"]
    obs = _StubDominoObservation([
        _StubDominoPose(r["id"], r["center_base_m"], r["quat_base_xyzw"])
        for r in records
    ])
    obs_task = env.task_from_observation(obs, "test")

    assert scene_task.init.allclose(obs_task.init)
    assert scene_task.goal == obs_task.goal


def test_goal_is_toppled_on_the_target(env):
    """Goal = Toppled(target), and only the target."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    assert len(task.goal) == 1
    atom = next(iter(task.goal))
    assert isinstance(atom, GroundAtom)
    assert atom.predicate.name == "Toppled"
    # The target is slot 3 -- the scene's 4th record, capture id _TARGET_ID.
    assert atom.objects[0].name == "domino_3"


def test_task_init_places_dominoes_at_transplanted_poses(env):
    """Each domino lands at its own base pose mapped into the world."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    for slot, rec in enumerate(_SCENE_RECORDS):
        bx, by, bz = rec["center_base_m"]
        dom = comp.dominos[slot]
        assert task.init.get(dom, "x") == pytest.approx(0.75 - by)
        assert task.init.get(dom, "y") == pytest.approx(0.72 + bx)
        assert task.init.get(dom, "z") == pytest.approx(bz + _Z_OFF)
        assert task.init.get(dom, "roll") == pytest.approx(0.0)


# -- start-domino yaw canonicalization ---------------------------------------


def _start_yaw(tmp_path, records, push_dir=None):
    """Build a one-off env for ``records`` and read the start domino's yaw."""
    path = _write_scene(tmp_path, records, push_dir, name="canon.json")
    env = _make_env(path)
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    slot = [r["id"] for r in records].index(_START_ID)
    return task.init.get(comp.dominos[slot], "yaw")


def test_start_yaw_flips_to_face_the_target(tmp_path):
    """A domino is 180-degree symmetric, so perception's yaw branch is
    arbitrary; the start must end up facing the way it has to topple.

    Identity orientation reads as yaw=pi, i.e. facing world -y. With the
    target at world +y, that faces away, so it flips to 0.
    """
    records = [
        _record(_START_ID, (0.0, 0.0)),
        _record(_TARGET_ID, (0.2, 0.0)),  # world +y of the start
    ]
    assert _start_yaw(tmp_path, records) == pytest.approx(0.0)


def test_start_yaw_left_alone_when_already_facing_the_target(tmp_path):
    """With the target the other way, the perceived branch already faces it."""
    records = [
        _record(_START_ID, (0.0, 0.0)),
        _record(_TARGET_ID, (-0.2, 0.0)),  # world -y of the start
    ]
    assert _start_yaw(tmp_path, records) == pytest.approx(_YAW_FROM_IDENTITY)


def test_explicit_start_push_dir_overrides_the_target_default(tmp_path):
    """``start_push_dir_base`` wins over "toward the target".

    The target sits at world +y (which alone would flip the yaw to 0),
    but the scene declares a push along base -x = world -y, which the
    perceived branch already faces -- so no flip.
    """
    records = [
        _record(_START_ID, (0.0, 0.0)),
        _record(_TARGET_ID, (0.2, 0.0)),
    ]
    assert _start_yaw(tmp_path, records,
                      push_dir=[-1.0,
                                0.0]) == pytest.approx(_YAW_FROM_IDENTITY)


# -- state_from_observation --------------------------------------------------


def test_state_from_observation_maps_capture_id_to_slot(env):
    """An observation names dominoes by capture id, not by slot."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    # Move ONLY capture id 12, which lives in slot 2.
    obs = _StubDominoObservation([_StubDominoPose(12, (0.3, 0.25, 0.03))])
    state = env.state_from_observation(obs, task.init)

    moved = comp.dominos[2]
    assert state.get(moved, "x") == pytest.approx(0.75 - 0.25)
    assert state.get(moved, "y") == pytest.approx(0.72 + 0.3)
    assert state.get(moved, "z") == pytest.approx(0.03 + _Z_OFF)
    # Every other slot is untouched.
    for slot in (0, 1, 3):
        other = comp.dominos[slot]
        for feat in ("x", "y", "z", "yaw"):
            assert state.get(other,
                             feat) == pytest.approx(task.init.get(other, feat))


def test_state_from_observation_carries_unseen_dominoes_forward(env):
    """Observations carry no visibility flag, so absent means unchanged."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    empty = _StubDominoObservation([])
    state = env.state_from_observation(empty, task.init)
    for slot in range(len(_SCENE_RECORDS)):
        dom = comp.dominos[slot]
        for feat in ("x", "y", "z", "yaw", "roll"):
            assert state.get(dom,
                             feat) == pytest.approx(task.init.get(dom, feat))


def test_state_from_observation_preserves_robot_and_joint_positions(env):
    """The robot's entry -- and the joint positions ``_set_state`` trusts --
    carry forward untouched.

    Without them ``_set_state`` falls back to IK, which drops wrist roll
    and corrupts a recorded grasp.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    robot = env._robot  # pylint: disable=protected-access
    obs = _StubDominoObservation([_StubDominoPose(11, (0.4, 0.1, 0.03))])
    state = env.state_from_observation(obs, task.init)

    assert np.allclose(state[robot], task.init[robot])
    assert isinstance(state, utils.PyBulletState)
    assert state.joint_positions is not None
    assert list(state.joint_positions) == list(task.init.joint_positions)


def test_state_from_observation_ignores_unknown_capture_ids(env):
    """A spurious detection has no slot to write into; dropping it beats
    aborting a run."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    obs = _StubDominoObservation([_StubDominoPose(4242, (0.1, 0.1, 0.03))])
    state = env.state_from_observation(obs, task.init)
    assert state.allclose(task.init)


def test_state_from_observation_does_not_canonicalize_start_yaw(env):
    """The yaw flip orients the OPENING push, so it belongs to task
    construction.

    Mid-episode the start domino may already have been pushed; re-
    canonicalizing would fight what the cameras actually saw.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    # The task build flipped the start's yaw to 0 (target is at world +y).
    assert task.init.get(comp.dominos[0], "yaw") == pytest.approx(0.0)
    # Re-observing the same pose reports the raw branch, unflipped.
    obs = _StubDominoObservation(
        [_StubDominoPose(_START_ID, (0.0, 0.0, 0.03))])
    state = env.state_from_observation(obs, task.init)
    assert state.get(comp.dominos[0],
                     "yaw") == pytest.approx(_YAW_FROM_IDENTITY)


# -- PyBulletEnv primitives --------------------------------------------------


def test_gripper_joint_layout_matches_the_robot(env):
    """The layout is read off the simulated robot, so the env and the action
    splitter cannot disagree."""
    robot = env._pybullet_robot  # pylint: disable=protected-access
    assert env.gripper_joint_layout() == gripper_joint_layout_from_robot(robot)
    layout = env.gripper_joint_layout()
    assert layout.left_finger_joint_idx != layout.right_finger_joint_idx
    assert layout.open_fingers != layout.closed_fingers


def test_sync_to_state_writes_perceived_poses_into_the_twin(env):
    """A perceived pose reaches the simulator's bodies.

    This is what makes perception visible to the agent at all: the next
    ``env.step`` reads its State back out of these bodies.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    pcid = env._physics_client_id  # pylint: disable=protected-access

    obs = _StubDominoObservation([_StubDominoPose(12, (0.3, 0.25, 0.03))])
    env.sync_to_state(env.state_from_observation(obs, task.init))

    pos, _ = p.getBasePositionAndOrientation(comp.dominos[2].id,
                                             physicsClientId=pcid)
    assert pos[0] == pytest.approx(0.75 - 0.25, abs=1e-4)
    assert pos[1] == pytest.approx(0.72 + 0.3, abs=1e-4)


def test_sync_to_state_zeroes_velocities_on_its_own(env, monkeypatch):
    """``sync_to_state`` clears momentum itself, not by luck.

    ``_set_state`` writes poses through
    ``resetBasePositionAndOrientation``, which does not touch
    velocities, so a body keeps whatever momentum the previous rollout
    gave it and drifts on the next ``stepSimulation``.

    The domino component happens to zero its own blocks at the end of
    ``reset_state``, which would mask a broken ``sync_to_state`` here.
    So that domain-specific hook is stubbed out: what remains is
    ``sync_to_state``'s own contribution, which is the part every future
    PyBullet-backed real environment depends on.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    pcid = env._physics_client_id  # pylint: disable=protected-access
    env.sync_to_state(task.init)

    monkeypatch.setattr(type(env), "_set_domain_specific_state",
                        lambda self, state: None)

    ids = [d.id for d in comp.dominos if d.id is not None]
    assert ids, "expected the component's dominoes to have pybullet bodies"
    for body in ids:
        p.resetBaseVelocity(body, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                            physicsClientId=pcid)
    assert all(
        np.linalg.norm(p.getBaseVelocity(b, physicsClientId=pcid)[0]) > 0
        for b in ids), "failed to spin the bodies up; test proves nothing"

    obs = _StubDominoObservation([_StubDominoPose(12, (0.3, 0.25, 0.03))])
    env.sync_to_state(env.state_from_observation(obs, task.init))

    for body in ids:
        linear, angular = p.getBaseVelocity(body, physicsClientId=pcid)
        assert np.allclose(linear, 0.0), f"body {body} kept linear velocity"
        assert np.allclose(angular, 0.0), f"body {body} kept angular velocity"
