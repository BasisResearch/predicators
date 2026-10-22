"""A joint whose travel the deployment set in the engine reaches the scene
manifest with the limits the engine enforces."""
# pylint: disable=protected-access
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.scene_manifest import build_scene_manifest
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.pybullet_helpers.objects import begin_asset_build, \
    drop_stale_assets, engine_joint_limits, forget_client_assets, \
    set_engine_joint_limits

_SWITCH_URDF = "urdf/partnet_mobility/switch/102812/switch.urdf"


def _prismatic_joint(body: int, client: int) -> int:
    return next(j for j in range(p.getNumJoints(body, physicsClientId=client))
                if p.getJointInfo(body, j, physicsClientId=client)[2] ==
                p.JOINT_PRISMATIC)


def _pushed_travel(body: int, joint: int, client: int) -> float:
    """Where a strong push along ``joint`` leaves it, from its lower stop."""
    p.resetJointState(body, joint, 0.0, physicsClientId=client)
    p.setJointMotorControl2(body,
                            joint,
                            p.VELOCITY_CONTROL,
                            targetVelocity=0.5,
                            force=50.0,
                            physicsClientId=client)
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
    return float(p.getJointState(body, joint, physicsClientId=client)[0])


def test_fan_s_switch_travel_reaches_the_manifest_as_the_engine_enforces_it():
    """Fan caps each switch's slider at a tenth of its URDF travel, in the
    engine, where getJointInfo does not show it.

    The manifest lists the capped limits and the file's own, and a
    switch loaded from the file and given the manifest's limits stops
    where the deployment's switch stops.
    """
    utils.reset_config({
        "env": "pybullet_fan",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = PyBulletFanEnv(use_gui=False)
    probe = p.connect(p.DIRECT)
    try:
        state = env.reset("train", 0)
        manifest, assets = build_scene_manifest(env, state)
        assert "joints" in manifest["legend"]
        entries = {
            b["object"]: b
            for b in manifest["bodies"]
            if b["object"] in {s.name
                               for s in env._switches}
        }
        assert len(entries) == len(env._switches) > 0
        for entry in entries.values():
            slider, = [j for j in entry["joints"] if j["type"] == "prismatic"]
            lower, upper = slider["urdf_limits"]
            assert upper > 0.2
            assert slider["lower_limit"] == pytest.approx(lower)
            assert slider["upper_limit"] == pytest.approx(
                env.switch_joint_scale * upper)
        # Joints the deployment left alone list no file limits.
        assert not any("urdf_limits" in j for b in manifest["bodies"]
                       if b["object"] not in entries for j in b["joints"])
        switch = env._switches[0]
        entry = entries[switch.name]
        index = next(k for k, j in enumerate(entry["joints"])
                     if j["type"] == "prismatic")
        slider = entry["joints"][index]
        body = p.loadURDF(assets[entry["urdf"]],
                          useFixedBase=True,
                          globalScaling=entry["urdf_scale"],
                          physicsClientId=probe)
        file_travel = _pushed_travel(body, index, probe)
        p.changeDynamics(body,
                         index,
                         jointLowerLimit=slider["lower_limit"],
                         jointUpperLimit=slider["upper_limit"],
                         physicsClientId=probe)
        built_travel = _pushed_travel(body, index, probe)
        real_travel = _pushed_travel(switch.id, switch.joint_id,
                                     env._physics_client_id)
        assert real_travel == pytest.approx(slider["upper_limit"], abs=2e-3)
        assert built_travel == pytest.approx(real_travel, abs=2e-3)
        assert file_travel > 3 * real_travel
    finally:
        p.disconnect(probe)
        p.disconnect(env._physics_client_id)


def test_engine_joint_limits_belong_to_the_world_that_set_them():
    """A record lasts while the world that set it owns its client: a later
    world built on the same client id and a disconnect both forget it."""
    client = p.connect(p.DIRECT)
    try:
        begin_asset_build()
        body = p.loadURDF(utils.get_env_asset_path(_SWITCH_URDF),
                          useFixedBase=True,
                          physicsClientId=client)
        joint = _prismatic_joint(body, client)
        assert engine_joint_limits(client, body, joint) is None
        set_engine_joint_limits(body, joint, 0.0, 0.01, client)
        assert engine_joint_limits(client, body, joint) == (0.0, 0.01)
        drop_stale_assets(client, begin_asset_build())
        assert engine_joint_limits(client, body, joint) is None
        set_engine_joint_limits(body, joint, 0.0, 0.02, client)
        assert engine_joint_limits(client, body, joint) == (0.0, 0.02)
        forget_client_assets(client)
        assert engine_joint_limits(client, body, joint) is None
    finally:
        p.disconnect(client)
