"""A joint the deployment holds inside its limits is marked hard_stop in the
scene manifest, and an agent's scene that loads the same file holds it too."""
# pylint: disable=protected-access
import json
import os
import shutil
from typing import Any, Dict, Tuple

import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.scene_base import scene_base_class
from predicators.code_sim_learning.scene_manifest import build_scene_manifest
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.pybullet_helpers.objects import HARD_STOP_JOINTS, \
    engine_joint_limits, hold_hard_stop_joints


def _pushed_travel(scene: Any, body: int, joint: int) -> float:
    """Where a strong push along ``joint`` leaves it, from its lower stop, with
    the step loop's hold after each physics step."""
    client = scene._physics_client_id
    p.resetJointState(body, joint, 0.0, physicsClientId=client)
    p.setJointMotorControl2(body,
                            joint,
                            p.VELOCITY_CONTROL,
                            targetVelocity=0.5,
                            force=50.0,
                            physicsClientId=client)
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
        hold_hard_stop_joints(client)
    return float(p.getJointState(body, joint, physicsClientId=client)[0])


def _deployment_package(root: Any) -> Tuple[Dict[str, Any], str, Any]:
    """The Fan level's manifest, written with its asset files into a sandbox
    reference directory, its switch entry, and the env's types."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = PyBulletFanEnv(use_gui=False)
    try:
        state = env.reset("train", 0)
        manifest, assets = build_scene_manifest(env, state)
        switch_names = {s.name for s in env._switches}
        types = env.types
    finally:
        p.disconnect(env._physics_client_id)
    reference = root / "reference"
    os.makedirs(reference / "scene")
    with open(reference / "scene" / "scene_manifest.json",
              "w",
              encoding="utf-8") as f:
        json.dump(manifest, f)
    for path, source in assets.items():
        if isinstance(source, str):
            os.makedirs(os.path.dirname(reference / path), exist_ok=True)
            shutil.copy(source, reference / path)
    switch = next(b for b in manifest["bodies"] if b["object"] in switch_names)
    return switch, str(reference / "assets"), types


def test_the_manifest_marks_only_the_held_joints(tmp_path):
    """Fan's switch sliders are hard stops; its fan joints are not."""
    switch, _, _ = _deployment_package(tmp_path)
    manifest_path = tmp_path / "reference" / "scene" / "scene_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert "hard_stop" in manifest["legend"]
    held = [(b["object"], j["name"]) for b in manifest["bodies"]
            for j in b["joints"] if j.get("hard_stop")]
    assert held
    for body in manifest["bodies"]:
        for joint in body["joints"]:
            assert bool(joint.get("hard_stop")) == (
                body.get("urdf") == switch["urdf"]
                and joint["type"] == "prismatic"), (body["object"], joint)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_scene_holds_the_manifest_s_hard_stops(tmp_path, scale):
    """A scene that loads the switch file, at any scale, keeps the slider at
    the manifest's limit under a strong push, scaled with the body; the file's
    own travel is several times longer."""
    switch, asset_dir, types = _deployment_package(tmp_path)
    slider = next(j for j in switch["joints"] if j["type"] == "prismatic")
    urdf = switch["urdf"][len("assets/"):]
    base: Any = scene_base_class("pybullet_fan", types, asset_dir)

    class Scene(base):  # type: ignore[misc,valid-type]
        """One switch, loaded from the manifest's file."""

        @classmethod
        def initialize_pybullet(cls, using_gui: bool) -> Any:
            client, robot, bodies = super().initialize_pybullet(using_gui)
            bodies["switch_0"] = p.loadURDF(cls.asset(urdf),
                                            basePosition=[2.0, 2.0, 0.0],
                                            useFixedBase=True,
                                            globalScaling=scale,
                                            physicsClientId=client)
            return client, robot, bodies

    scene = Scene(use_gui=False)
    client = scene._physics_client_id
    try:
        body = scene.body("switch_0")
        joint = next(
            j for j in range(p.getNumJoints(body, physicsClientId=client))
            if p.getJointInfo(
                body, j, physicsClientId=client)[1].decode() == slider["name"])
        assert (client, body, joint) in HARD_STOP_JOINTS
        ratio = scale / switch["urdf_scale"]
        upper = slider["upper_limit"] * ratio
        assert engine_joint_limits(client, body, joint) == pytest.approx(
            (slider["lower_limit"] * ratio, upper))
        assert _pushed_travel(scene, body, joint) == pytest.approx(upper,
                                                                   abs=1e-6)
        # The file's own travel, once the hold is gone.
        HARD_STOP_JOINTS.discard((client, body, joint))
        p.changeDynamics(body,
                         joint,
                         jointLowerLimit=slider["urdf_limits"][0] * scale,
                         jointUpperLimit=slider["urdf_limits"][1] * scale,
                         physicsClientId=client)
        assert _pushed_travel(scene, body, joint) > 3 * upper
    finally:
        p.disconnect(client)


def test_a_scene_without_a_manifest_holds_nothing(tmp_path):
    """With no manifest beside its assets a scene registers no hard stop."""
    switch, asset_dir, types = _deployment_package(tmp_path)
    os.remove(tmp_path / "reference" / "scene" / "scene_manifest.json")
    urdf = switch["urdf"][len("assets/"):]
    base: Any = scene_base_class("pybullet_fan", types, asset_dir)

    class Scene(base):  # type: ignore[misc,valid-type]
        """One switch, loaded from the manifest's file."""

        @classmethod
        def initialize_pybullet(cls, using_gui: bool) -> Any:
            client, robot, bodies = super().initialize_pybullet(using_gui)
            bodies["switch_0"] = p.loadURDF(cls.asset(urdf),
                                            basePosition=[2.0, 2.0, 0.0],
                                            useFixedBase=True,
                                            physicsClientId=client)
            return client, robot, bodies

    scene = Scene(use_gui=False)
    client = scene._physics_client_id
    try:
        assert not [k for k in HARD_STOP_JOINTS if k[0] == client]
    finally:
        p.disconnect(client)
