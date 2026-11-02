"""A mesh the deployment builds from vertices in memory reaches the scene
manifest as a generated OBJ file, with how it is loaded and its margin."""
# pylint: disable=protected-access
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.scene_manifest import GeneratedAsset, \
    build_scene_manifest
from predicators.envs.pybullet_fan import PyBulletFanEnv


def test_fan_s_ramp_reaches_the_manifest_as_a_generated_mesh(tmp_path):
    """Fan's ramp is a wedge built in memory, which PyBullet reads back as an
    unnamed mesh with no vertices.

    The manifest names a generated OBJ file for it and says it is a
    concave triangle mesh with no collision margin; loading that file as
    it says puts the wedge's top where the deployment's is.
    """
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": True,
        "fan_ramp_transfer": True,
        "fan_ramp_rise": 0.004,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    env = PyBulletFanEnv(use_gui=False)
    probe = p.connect(p.DIRECT)
    try:
        state = env.reset("train", 0)
        ramp, = state.get_objects(env._ramp_type)
        manifest, assets = build_scene_manifest(env, state)
        entry, = [b for b in manifest["bodies"] if b["object"] == ramp.name]
        shape, = entry["links"][0]["shapes"]
        assert shape["geometry"] == "mesh"
        expected = f"assets/generated/{ramp.name}_link0_shape0.obj"
        assert shape["file"] == expected
        assert shape["concave"] is True
        assert shape["collision_margin"] == 0.0
        assert "generated" in manifest["legend"]
        asset = assets[shape["file"]]
        assert isinstance(asset, GeneratedAsset)
        lines = asset.text.splitlines()
        assert sum(line.startswith("v ") for line in lines) == 8
        assert sum(line.startswith("f ") for line in lines) == 12
        assert not any(
            s.get("file") == "unknown_file" for body in manifest["bodies"]
            for link in body["links"] for s in link["shapes"])
        # Load the file as the manifest says, at the ramp's pose, in a world
        # of its own.
        path = tmp_path / "ramp.obj"
        path.write_text(asset.text, encoding="utf-8")
        collision = p.createCollisionShape(p.GEOM_MESH,
                                           fileName=str(path),
                                           flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                           physicsClientId=probe)
        real_id = env._boundary_named(ramp).id
        client = env._physics_client_id
        position, orientation = p.getBasePositionAndOrientation(
            real_id, physicsClientId=client)
        body = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision,
                                 basePosition=position,
                                 baseOrientation=orientation,
                                 physicsClientId=probe)
        p.changeDynamics(body,
                         -1,
                         collisionMargin=shape["collision_margin"],
                         physicsClientId=probe)
        y = state.get(ramp, "y")
        for local_x in (0.75, 0.90, 1.00):
            x = local_x + env.ramp_scene_x_offset
            rebuilt = p.rayTest((x, y, 0.8), (x, y, 0.1),
                                physicsClientId=probe)[0]
            deployed = p.rayTest((x, y, 0.8), (x, y, 0.1),
                                 physicsClientId=client)[0]
            assert deployed[0] == real_id and rebuilt[0] == body
            assert rebuilt[3][2] == pytest.approx(deployed[3][2], abs=1e-6)
    finally:
        p.disconnect(probe)
        p.disconnect(env._physics_client_id)
