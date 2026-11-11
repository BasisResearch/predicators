"""A switch slider pushed past the end of its travel is held at the end by the
next step, in every env that caps switch travel."""
# pylint: disable=protected-access
from unittest.mock import patch

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.objects import ENGINE_JOINT_LIMITS, \
    HARD_STOP_JOINTS, forget_client_assets, set_engine_joint_limits
from predicators.structs import Action


@pytest.mark.parametrize("env_name", [
    "pybullet_balloons",
    "pybullet_barrier",
    "pybullet_boil",
    "pybullet_busyboard",
    "pybullet_fan",
    "pybullet_laser",
    "pybullet_magic_bin",
    "pybullet_switch",
])
def test_switch_slider_stays_inside_its_travel(env_name: str) -> None:
    """Every hard-stop joint is put back at the limit it was pushed past, even
    when the env's own per-substep hook does nothing."""
    utils.reset_config({
        "env": env_name,
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    assert isinstance(env, PyBulletEnv)
    client = env._physics_client_id
    try:
        env.reset("train", 0)
        keys = [k for k in HARD_STOP_JOINTS if k[0] == client]
        assert keys, f"{env_name} registers no switch slider"
        still = Action(
            np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
        for _, body, joint in keys:
            low, high = ENGINE_JOINT_LIMITS[(client, body, joint)]
            for start, end in ((high + 0.05, high), (low - 0.05, low)):
                p.resetJointState(body, joint, start, physicsClientId=client)
                with patch.object(type(env), "_after_physics_substep",
                                  lambda self: None):
                    env.step(still)
                j_pos = p.getJointState(body, joint, physicsClientId=client)[0]
                assert j_pos == pytest.approx(end, abs=1e-6)
    finally:
        p.disconnect(client)


def test_hard_stop_ends_with_its_world() -> None:
    """A closed world's hard stops do not carry over to a joint that reuses its
    ids, nor to a joint whose limits are set again without the cap."""
    key = (99, 3, 1)
    set_engine_joint_limits_record(key)
    HARD_STOP_JOINTS.add(key)
    forget_client_assets(99)
    assert key not in HARD_STOP_JOINTS
    set_engine_joint_limits_record(key)
    HARD_STOP_JOINTS.add(key)
    set_engine_joint_limits_record(key)
    assert key not in HARD_STOP_JOINTS
    forget_client_assets(99)


def set_engine_joint_limits_record(key: tuple) -> None:
    """Record limits for ``key`` the way set_engine_joint_limits does, without
    a live body."""
    with patch.object(p, "changeDynamics"):
        set_engine_joint_limits(key[1], key[2], 0.0, 0.01, key[0])
