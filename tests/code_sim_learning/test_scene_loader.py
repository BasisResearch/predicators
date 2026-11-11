"""The scene loader builds each level's bodies as the deployment builds them,
and a scene base swaps in the scene of the level a state belongs to."""
# pylint: disable=protected-access
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p
import pytest
import yaml

from predicators import utils
from predicators.code_sim_learning.scene_base import scene_base_class
from predicators.code_sim_learning.scene_loader import NOMINAL_DENSITY, \
    SceneLayout, load_scene, read_layouts, resolve_layout, static_signature
from predicators.code_sim_learning.scene_manifest import GeneratedAsset, \
    build_scene_manifest, manifest_asset_files, scene_manifest_text
from predicators.envs import create_new_env
from predicators.pybullet_helpers.objects import engine_joint_limits, \
    is_hard_stop_joint
from predicators.structs import Object, State, Type

_CONFIGS = Path(__file__).resolve().parents[2] / "scripts" / "configs"
_BENCHMARK = ("fan", "balloons", "domino_high_friction_turn", "bridge", "boil")


def _benchmark_env(key: str) -> Any:
    """The deployment world of one benchmark domain, as the runs build it."""
    with open(_CONFIGS / "empiric" / "envs.yaml", encoding="utf-8") as f:
        entry = yaml.safe_load(f)["ENVS"][key]
    flags = dict(entry["FLAGS"])
    flags.update({
        "env": entry["NAME"],
        "seed": 0,
        "num_test_tasks": 1,
        "scene_manifest_loader": True,
    })
    flags.setdefault("num_train_tasks", 1)
    utils.reset_config(flags)
    return create_new_env(entry["NAME"], do_cache=False, use_gui=False)


def _write_assets(root: Path, assets: Dict[str, Any]) -> None:
    """Put a manifest's files under ``root`` as the sandbox holds them."""
    for path, source in assets.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        if isinstance(source, GeneratedAsset):
            target.write_text(source.text, encoding="utf-8")
        else:
            os.symlink(source, target)


def _asset(root: Path) -> Any:
    return lambda relative: str(root / "assets" / relative)


def _link_boxes(client: int, body: int) -> List[Tuple[float, ...]]:
    """The bounding box of every link of a body."""
    return [
        tuple(v for corner in p.getAABB(body, link, physicsClientId=client)
              for v in corner)
        for link in range(-1, p.getNumJoints(body, physicsClientId=client))
    ]


def _copy_pose(source: Tuple[int, int], target: Tuple[int, int]) -> None:
    """Give the body ``target`` the base pose and joint positions of
    ``source``; each is a (client, body)."""
    position, orientation = p.getBasePositionAndOrientation(
        source[1], physicsClientId=source[0])
    p.resetBasePositionAndOrientation(target[1],
                                      position,
                                      orientation,
                                      physicsClientId=target[0])
    joints = min(p.getNumJoints(source[1], physicsClientId=source[0]),
                 p.getNumJoints(target[1], physicsClientId=target[0]))
    for joint in range(joints):
        p.resetJointState(target[1],
                          joint,
                          p.getJointState(source[1],
                                          joint,
                                          physicsClientId=source[0])[0],
                          0.0,
                          physicsClientId=target[0])


@pytest.mark.parametrize("key", _BENCHMARK)
def test_loaded_bodies_match_the_deployment(key: str, tmp_path: Path) -> None:
    """Each observed body the loader builds from a level's manifest has the
    links, the shapes, the joint limits and the static base of the deployment's
    body, and fills the same space at the same pose; each object without a body
    is listed without one."""
    env = _benchmark_env(key)
    client = env._physics_client_id
    probe = p.connect(p.DIRECT)
    try:
        state = env.reset("train", 0)
        manifest, assets = build_scene_manifest(env, state)
        _write_assets(tmp_path, assets)
        loaded = load_scene(probe, SceneLayout("level_01", manifest),
                            _asset(tmp_path))
        assert not loaded.notes
        deployed = {
            obj.name: obj.id
            for obj in env._objects if getattr(obj, "id", None) is not None
        }
        entries = {
            entry["object"]: entry
            for entry in manifest["bodies"] if entry["object"] is not None
        }
        assert set(entries) <= set(loaded.bodies)
        robot_id = env._pybullet_robot.robot_id
        for name, entry in entries.items():
            body = loaded.bodies[name]
            real = deployed[name]
            assert isinstance(body, int) and real != robot_id
            assert p.getNumJoints(body, physicsClientId=probe) == \
                p.getNumJoints(real, physicsClientId=client) or \
                not entry.get("urdf")
            static = p.getDynamicsInfo(body, -1,
                                       physicsClientId=probe)[0] == 0.0
            assert static == entry["static"]
            if entry.get("urdf"):
                for joint in range(p.getNumJoints(real,
                                                  physicsClientId=client)):
                    listed = entry["joints"][joint]
                    info = p.getJointInfo(body, joint, physicsClientId=probe)
                    assert info[1].decode("utf-8") == listed["name"]
                    limits = engine_joint_limits(probe, body,
                                                 joint) or (info[8], info[9])
                    assert limits == pytest.approx(
                        (listed["lower_limit"], listed["upper_limit"]))
                    assert is_hard_stop_joint(probe, body, joint) == bool(
                        listed.get("hard_stop"))
            # Same pose, same space: per link for a file's body, and the
            # base for one built from shapes (its other links hold only
            # marks nothing touches).
            _copy_pose((client, real), (probe, body))
            mine = _link_boxes(probe, body)
            theirs = _link_boxes(client, real)
            if not entry.get("urdf"):
                mine, theirs = mine[:1], theirs[:1]
            for a, b in zip(mine, theirs):
                assert a == pytest.approx(b, abs=1e-5)
        for name, kind in manifest["objects"].items():
            if kind != "robot" and name not in entries:
                assert loaded.bodies[name] is None
        unnamed = [e for e in manifest["bodies"] if e["object"] is None]
        assert len(loaded.unnamed) == len(unnamed)
        for body, entry in zip(loaded.unnamed, unnamed):
            position, _ = p.getBasePositionAndOrientation(
                body, physicsClientId=probe)
            assert position == pytest.approx(entry["base_pose"]["position"],
                                             abs=1e-6)
    finally:
        p.disconnect(probe)
        p.disconnect(client)


def test_a_built_moving_body_gets_its_volume_at_the_nominal_density(
        tmp_path: Path) -> None:
    """The manifest records no mass: Fan's ball, a sphere built from a shape,
    weighs its volume at the nominal density until the model declares its mass,
    and the static platforms weigh nothing."""
    env = _benchmark_env("fan")
    probe = p.connect(p.DIRECT)
    try:
        state = env.reset("train", 0)
        manifest, assets = build_scene_manifest(env, state)
        _write_assets(tmp_path, assets)
        loaded = load_scene(probe, SceneLayout("level_01", manifest),
                            _asset(tmp_path))
        ball, = [o for o in state if o.type.name == "ball"]
        radius = state.get(ball, "radius")
        mass = p.getDynamicsInfo(loaded.bodies[ball.name],
                                 -1,
                                 physicsClientId=probe)[0]
        assert mass == pytest.approx(NOMINAL_DENSITY * 4.0 / 3.0 *
                                     3.141592653589793 * radius**3,
                                     rel=1e-6)
        assert p.getDynamicsInfo(loaded.bodies["platform_0"],
                                 -1,
                                 physicsClientId=probe)[0] == 0.0
    finally:
        p.disconnect(probe)
        p.disconnect(env._physics_client_id)


def _fan_levels(tmp_path: Path) -> Tuple[Any, State, State]:
    """A sandbox ``reference/`` holding the scenes of Fan's training level and
    its test level, and a scene base bound to it."""
    env = _benchmark_env("fan")
    reference = tmp_path / "reference"
    states = [env.reset("train", 0), env.reset("test", 0)]
    levels = reference / "scene" / "levels"
    levels.mkdir(parents=True)
    for index, state in enumerate(states):
        manifest, assets = build_scene_manifest(env, state)
        _write_assets(reference, assets)
        _write_assets(reference, manifest_asset_files(manifest))
        record = {
            "manifest": manifest,
            "signature": static_signature(manifest, state)
        }
        (levels / f"level_{index + 1:02d}.json").write_text(
            scene_manifest_text(record), encoding="utf-8")
        (reference / "scene" / "scene_manifest.json").write_text(
            scene_manifest_text(manifest), encoding="utf-8")
    types = env.types
    p.disconnect(env._physics_client_id)
    base: Any = scene_base_class("pybullet_fan", types,
                                 str(reference / "assets"))
    return base, states[0], states[1]


def _ramp_width(scene: Any) -> float:
    low, high = p.getAABB(scene.body("ramp_0"),
                          physicsClientId=scene._physics_client_id)
    return float(high[1] - low[1])


def test_a_scene_base_holds_the_scene_of_the_state_s_level(
        tmp_path: Path) -> None:
    """Fan's training level and test level build the ramp at other widths and
    name other platforms.

    A scene base without any body code of its own starts in the latest
    level's scene, and a state of the other level brings that level's
    scene: its ramp, its platforms, a body for every object, at the
    observed poses.
    """
    base, train, test = _fan_levels(tmp_path)
    loads: List[str] = []

    class Scene(base):  # type: ignore[misc,valid-type]
        """Physics only: the base loads the bodies."""

        def _on_scene_loaded(self) -> None:
            loads.append(self._loaded_scene.label)

    scene = Scene(use_gui=False)
    try:
        assert loads == ["level_02"]
        widths = {}
        for label, state in (("test", test), ("train", train), ("test", test)):
            scene._set_state(state)
            ramp, = [o for o in state if o.type.name == "ramp"]
            widths[label] = _ramp_width(scene)
            assert widths[label] == pytest.approx(state.get(ramp, "y_len"),
                                                  abs=1e-4)
            for obj in state:
                if obj.type.name == "robot":
                    continue
                body = scene.bodies[obj.name]
                if body is None:
                    continue
                position, _ = p.getBasePositionAndOrientation(
                    body, physicsClientId=scene._physics_client_id)
                assert position[0] == pytest.approx(state.get(obj, "x"),
                                                    abs=1e-4)
            assert not scene._last_unreconstructible_features
        assert loads == ["level_02", "level_01", "level_02"]
        assert widths["train"] != pytest.approx(widths["test"], abs=1e-3)
        # A state of the level the world already holds keeps its bodies.
        ball = scene.body("ball")
        scene._set_state(test)
        assert scene.body("ball") == ball
        assert loads == ["level_02", "level_01", "level_02"]
    finally:
        p.disconnect(scene._physics_client_id)


def test_a_static_object_stands_where_its_level_s_scene_has_it(
        tmp_path: Path) -> None:
    """An observation reads a platform's height with noise.

    The platform's body stands at the height its level's scene has it
    at, and the state the world reports gives that height; a state that
    shows it moved by more than the noise explains moves it. With the
    flag off, each state's own reading places it.
    """
    base, train, _ = _fan_levels(tmp_path)
    utils.update_config({"continual_obs_noise_position": 0.005})
    scene = base(use_gui=False)
    try:
        platform, = [o for o in train if o.name == "platform_0"]
        stands = float(train.get(platform, "z"))

        def height(offset: float) -> float:
            state = train.copy()
            state.set(platform, "z", stands + offset)
            scene._set_state(state)
            position, _ = p.getBasePositionAndOrientation(
                scene.body("platform_0"),
                physicsClientId=scene._physics_client_id)
            return float(position[2])

        assert height(0.008) == pytest.approx(stands, abs=1e-6)
        reported = scene._get_state()
        bound, = [o for o in reported if o.name == "platform_0"]
        assert reported.get(bound, "z") == pytest.approx(stands, abs=1e-6)
        assert height(0.05) == pytest.approx(stands + 0.05, abs=1e-6)
        utils.update_config({"scene_loader_static_poses": False})
        assert height(0.008) == pytest.approx(stands + 0.008, abs=1e-6)
    finally:
        p.disconnect(scene._physics_client_id)


def test_the_layout_of_a_state_is_the_level_it_shows() -> None:
    """A state belongs to the layout that names its objects and whose static
    objects it shows as that level began; when two layouts fit alike, the world
    keeps the one it holds, else it takes the latest."""
    block = Type("block", ["x", "rot"])
    wall = Type("wall", ["x", "rot"])
    a, b, w = Object("a", block), Object("b", block), Object("w", wall)
    wall_body = [{"object": "w", "static": True, "joints": []}]
    manifest_one = {
        "objects": {
            "a": "block",
            "w": "wall"
        },
        "bodies": wall_body
    }
    manifest_two = {
        "objects": {
            "a": "block",
            "b": "block",
            "w": "wall"
        },
        "bodies": wall_body
    }
    near = SceneLayout("near", manifest_one, {"w": {"x": 0.1, "rot": 3.14}})
    far = SceneLayout("far", manifest_one, {"w": {"x": 0.5, "rot": 3.14}})
    twin = SceneLayout("twin", manifest_one, {"w": {"x": 0.5, "rot": 3.14}})
    more = SceneLayout("more", manifest_two, {"w": {"x": 0.1, "rot": 3.14}})
    layouts = [near, far, twin, more]
    # The wall reads a full turn away from the level's start: the same
    # angle.
    state = State({a: np.array([0.0, 0.0]), w: np.array([0.11, -3.14])})
    assert resolve_layout(layouts, state) is near
    state = State({a: np.array([0.0, 0.0]), w: np.array([0.49, 3.14])})
    assert resolve_layout(layouts, state) is twin
    assert resolve_layout(layouts, state, current="far") is far
    state = State({
        a: np.array([0.0, 0.0]),
        b: np.array([1.0, 0.0]),
        w: np.array([0.49, 3.14])
    })
    assert resolve_layout(layouts, state) is more
    assert resolve_layout([], state) is None


def test_layouts_are_read_per_level_else_from_the_single_manifest(
        tmp_path: Path) -> None:
    """``scene/levels/level_NN.json`` gives one layout per level, in level
    order; without them the single manifest is the one layout."""
    scene = tmp_path / "scene"
    scene.mkdir()
    manifest: Dict[str, Any] = {"objects": {}, "bodies": []}
    assert not read_layouts(str(scene))
    (scene / "scene_manifest.json").write_text(scene_manifest_text(manifest),
                                               encoding="utf-8")
    assert [layout.label for layout in read_layouts(str(scene))] == ["scene"]
    (scene / "levels").mkdir()
    for index in (2, 1):
        (scene / "levels" / f"level_{index:02d}.json").write_text(
            scene_manifest_text({
                "manifest": manifest,
                "signature": {}
            }),
            encoding="utf-8")
    assert [layout.label for layout in read_layouts(str(scene))
            ] == ["level_01", "level_02"]
