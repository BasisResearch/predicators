"""Every collision shape in the scene manifest lists the margin the deployment
builds its link with."""
# pylint: disable=protected-access
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.scene_manifest import build_scene_manifest
from predicators.envs.pybullet_fan import PyBulletFanEnv


def test_every_collision_shape_lists_its_link_s_margin() -> None:
    """The engine rounds a convex shape's edges by its margin, so a scene built
    from the manifest needs it: Fan's ramp has none, and each platform's is
    listed as the engine holds it."""
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "fan_exposed_transfer": True,
        "fan_inertial_transfer": True,
        "fan_ramp_transfer": True,
        "fan_train_num_walls_per_task": [0],
        "fan_test_num_walls_per_task": [0],
    })
    env = PyBulletFanEnv(use_gui=False)
    client = env._physics_client_id
    try:
        state = env.reset("train", 0)
        manifest, _ = build_scene_manifest(env, state)
        assert "collision_margin" in manifest["legend"]
        for body in manifest["bodies"]:
            for link in body["links"]:
                for shape in link["shapes"]:
                    assert isinstance(shape["collision_margin"], float)
        entries = {b["object"]: b for b in manifest["bodies"]}
        for obj in state:
            if obj.type.name not in ("platform", "ramp"):
                continue
            listed, = entries[obj.name]["links"][0]["shapes"]
            engine = p.getDynamicsInfo(env._boundary_named(obj).id,
                                       -1,
                                       physicsClientId=client)[11]
            assert listed["collision_margin"] == engine, obj.name
            if obj.type.name == "ramp":
                assert listed["collision_margin"] == 0.0
    finally:
        p.disconnect(client)
