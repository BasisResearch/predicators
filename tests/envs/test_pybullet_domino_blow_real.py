"""Tests for pybullet_domino_blow_real: the blow task rebuilt from a fan-bench
scene, with the fan, the button and the goal patch where the cameras saw them.

The button proxy URDF comes from the private BabyRobotPredicator
package, so the env tests skip without it; the layout arithmetic does
not need it and runs everywhere.
"""
# pylint: disable=protected-access
import json
import math
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from predicators import utils
from predicators.envs.pybullet_domino.real_geometry import \
    DOMINO_WORLD_ROBOT_XY, DOMINO_WORLD_ROBOT_YAW, domino_world_z_offset
from predicators.envs.pybullet_domino_blow_real import BenchLayout, \
    PyBulletDominoBlowRealEnv
from predicators.settings import CFG
from predicators.structs import GroundAtom

_TABLE_Z = -0.041


def _block(xy: Tuple[float, float],
           yaw: float = -math.pi / 2) -> Dict[str, Any]:
    """A standing block at base-frame ``xy``, in the exporter's format:

    body x vertical, so the quaternion is a +90 deg pitch then the yaw.
    """
    from scipy.spatial.transform import Rotation
    quat = (Rotation.from_euler("z", yaw) *
            Rotation.from_euler("y", math.pi / 2)).as_quat()
    return {
        "id": 0,
        "center_base_m": [xy[0], xy[1], 0.034],
        "quat_base_xyzw": [float(v) for v in quat],
        "yaw_base_rad": yaw,
        "roll_base_rad": 0.0,
        "fall_deg": 0.0,
        "dims_m": [0.15, 0.07, 0.029],
    }


def _fixture(xy: Tuple[float, float],
             dims: Tuple[float, float, float],
             yaw: float = 0.0,
             z: float = 0.0) -> Dict[str, Any]:
    return {
        "center_base_m": [xy[0], xy[1], z],
        "dims_m": list(dims),
        "yaw_base_rad": yaw,
        "top_z_base_m": z + dims[2] / 2.0,
    }


# The 2026-09-12 bench, rounded: the fan blows along base +y, the goal is
# 0.34 m downwind of it, the button beside the fan, the block staged off
# the line.
_SCENE = {
    "frame": "robot_base",
    "units": "meters/radians",
    "dominoes": [_block((0.456, 0.289))],
    "wind_dir_base": [0.0, 1.0],
    "fixtures": {
        "fan":
        _fixture((0.597, -0.202), (0.071, 0.012, 0.068), z=0.089),
        "button":
        _fixture((0.414, -0.246), (0.082, 0.052, 0.067), yaw=-0.7, z=-0.003),
        "goal":
        _fixture((0.584, 0.141), (0.165, 0.121, 0.009), yaw=-1.35, z=-0.035),
    },
}


def _write_scene(tmp_path, scene=_SCENE):
    path = tmp_path / "fan_bench.json"
    path.write_text(json.dumps(scene))
    return str(path)


def _config(scene_path):
    utils.reset_config({
        "env": "pybullet_domino_blow_real",
        "pybullet_robot": "panda",
        "domino_real_scene": scene_path,
        "domino_real_table_z": _TABLE_Z,
        "domino_use_skill_factories": True,
        "domino_use_continuous_place": True,
        "fan_known_controls_relation": True,
        "domino_real_decorate": False,
    })


@pytest.fixture(scope="module", name="scene_path")
def scene_path_fixture(tmp_path_factory):
    return _write_scene(tmp_path_factory.mktemp("blow_real"))


@pytest.fixture(scope="module", name="env")
def env_fixture(scene_path):
    pytest.importorskip("markerless_estimation")
    _config(scene_path)
    return PyBulletDominoBlowRealEnv(use_gui=False)


@pytest.fixture(autouse=True)
def _reapply_config(scene_path):
    _config(scene_path)


def _world_xy(base_xy):
    """The base -> world transplant, by hand: a +90 deg turn then the
    offset."""
    x, y = base_xy
    return (DOMINO_WORLD_ROBOT_XY[0] - y, DOMINO_WORLD_ROBOT_XY[1] + x)


# -- the layout: no PyBullet, no submodule ------------------------------------


def test_layout_transplants_fixtures_and_wind_to_the_world_frame(scene_path):
    z_off = domino_world_z_offset(_TABLE_Z)
    layout = BenchLayout.from_scene(_SCENE, z_off)
    assert layout.fan_xy == pytest.approx(_world_xy((0.597, -0.202)))
    assert layout.button_xy == pytest.approx(_world_xy((0.414, -0.246)))
    assert layout.goal_xy == pytest.approx(_world_xy((0.584, 0.141)))
    # Base +y turns into world -x.
    assert layout.wind_dir == pytest.approx((-1.0, 0.0), abs=1e-9)
    assert layout.wind_yaw == pytest.approx(math.pi / 2 +
                                            DOMINO_WORLD_ROBOT_YAW)
    # The button top, raised by the table offset.
    assert layout.button_top_z == pytest.approx(-0.003 + 0.067 / 2 + z_off)


def test_layout_projects_the_goal_patch_onto_the_wind():
    # A patch turned exactly across the wind swaps its half extents.
    scene = json.loads(json.dumps(_SCENE))
    scene["fixtures"]["goal"][
        "yaw_base_rad"] = math.pi / 2  # long side along +y = the wind
    along = BenchLayout.from_scene(scene, 0.0)
    assert along.goal_half_along == pytest.approx(0.165 / 2)
    assert along.goal_half_across == pytest.approx(0.121 / 2)
    scene["fixtures"]["goal"]["yaw_base_rad"] = 0.0  # long side across
    across = BenchLayout.from_scene(scene, 0.0)
    assert across.goal_half_along == pytest.approx(0.121 / 2)
    assert across.goal_half_across == pytest.approx(0.165 / 2)


def test_layout_refuses_a_scene_without_fixtures():
    scene = {"dominoes": [_block((0.4, 0.2))]}
    with pytest.raises(ValueError, match="fan_scene_export"):
        BenchLayout.from_scene(scene, 0.0)


# -- the env ------------------------------------------------------------------


def test_task_has_one_block_the_fixtures_and_an_in_goal_goal(env):
    task = env.get_test_tasks()[0]
    state = task.init
    names = {o.type.name for o in state}
    assert names == {"robot", "domino", "fan", "switch", "side", "region"}
    blocks = state.get_objects(env._domino_component.domino_type)
    assert len(blocks) == 1
    region = env._goal_region_component.region
    assert task.goal == {
        GroundAtom(env._goal_region_component.InGoal, [blocks[0], region])
    }


def test_block_and_fixtures_land_at_their_transplanted_poses(env):
    state = env.get_test_tasks()[0].init
    block = state.get_objects(env._domino_component.domino_type)[0]
    assert (state.get(block, "x"), state.get(block, "y")) == \
        pytest.approx(_world_xy((0.456, 0.289)), abs=1e-6)
    assert state.get(block, "roll") == pytest.approx(0.0, abs=1e-6)
    fan = env._fan_component.fans[0]
    assert (state.get(fan, "x"), state.get(fan, "y")) == \
        pytest.approx(env.bench.fan_xy)
    assert state.get(fan, "rot") == pytest.approx(env.bench.wind_yaw)
    button = env._fan_component.button
    assert (state.get(button, "x"), state.get(button, "y")) == \
        pytest.approx(env.bench.button_xy)
    assert state.get(button, "z") == pytest.approx(env.bench.button_top_z)
    region = env._goal_region_component.region
    assert (state.get(region, "x"), state.get(region, "y")) == \
        pytest.approx(env.bench.goal_xy)
    assert state.get(region,
                     "half_x") == pytest.approx(env.bench.goal_half_along)
    assert state.get(region,
                     "half_y") == pytest.approx(env.bench.goal_half_across)


def test_fan_starts_off_and_the_button_is_momentary(env):
    env.reset("test", 0)
    fan = env._fan_component
    assert not fan.any_fan_on()
    # Written on: the plunger is down and the fan reads on...
    fan._set_switch_on(fan.button.id, True)
    assert fan.any_fan_on()
    # ...until the spring returns it, which one step of physics does.
    import pybullet as p
    for _ in range(CFG.pybullet_sim_steps_per_action):
        p.stepSimulation(physicsClientId=env._physics_client_id)
    assert not fan.any_fan_on()


def test_wind_blows_the_block_down_the_measured_axis(env):
    """With the button held, the block placed upwind ends up flat and
    downwind -- the direction the scene says, not the generated task's +x."""
    import pybullet as p

    from predicators.structs import Action
    task = env.get_test_tasks()[0]
    state = task.init.copy()
    block = state.get_objects(env._domino_component.domino_type)[0]
    region = env._goal_region_component.region
    dx, dy = env.bench.wind_dir
    gx, gy = state.get(region, "x"), state.get(region, "y")
    state.set(block, "x", gx - 0.12 * dx)
    state.set(block, "y", gy - 0.12 * dy)
    state.set(block, "yaw", env.bench.wind_yaw + math.pi / 2)
    env._set_state(state)
    utils.reset_config({"domino_blow_wind_force": 2.0})
    env._wire_blow_target(env._get_state())
    fan = env._fan_component
    p.setJointMotorControl2(fan.button.id,
                            fan._button_joint_id,
                            p.POSITION_CONTROL,
                            targetPosition=-0.004,
                            force=5.0,
                            physicsClientId=env._physics_client_id)
    hold = Action(np.array(env._pybullet_robot.get_joints(), dtype=np.float32))
    for _ in range(40):
        env.step(hold)
    after = env._get_state()
    along = ((after.get(block, "x") - (gx - 0.12 * dx)) * dx +
             (after.get(block, "y") - (gy - 0.12 * dy)) * dy)
    assert along > 0.08  # it travelled downwind
    roll = (after.get(block, "roll") + np.pi / 2) % np.pi - np.pi / 2
    assert abs(roll) > 1.0  # and it is flat


def test_state_from_observation_moves_only_the_block(env):
    """A live look rewrites the block; the fixtures and the robot's entry are
    carried forward untouched."""

    class _Pose:

        def __init__(self, capture_id, xyz, quat):
            self.id, self.xyz, self.quat_xyzw = capture_id, xyz, quat

    class _Obs:

        def __init__(self, dominoes):
            self.dominoes = dominoes

    prev = env.get_test_tasks()[0].init
    rec = _block((0.6, 0.05))
    obs = _Obs([_Pose(0, rec["center_base_m"], rec["quat_base_xyzw"])])
    new = env.state_from_observation(obs, prev)
    block = new.get_objects(env._domino_component.domino_type)[0]
    assert (new.get(block, "x"), new.get(block, "y")) == \
        pytest.approx(_world_xy((0.6, 0.05)), abs=1e-6)
    for obj in new:
        if obj.type.name == "domino":
            continue
        for feat in obj.type.feature_names:
            assert new.get(obj, feat) == prev.get(obj, feat)


def test_task_from_observation_agrees_with_the_scene(env):

    class _Pose:

        def __init__(self, capture_id, xyz, quat):
            self.id, self.xyz, self.quat_xyzw = capture_id, xyz, quat

    class _Obs:

        def __init__(self, dominoes):
            self.dominoes = dominoes

    rec = _SCENE["dominoes"][0]
    obs = _Obs([_Pose(0, rec["center_base_m"], rec["quat_base_xyzw"])])
    from_obs = env.task_from_observation(obs)
    from_scene = env.get_test_tasks()[0]
    assert from_obs.goal == from_scene.goal
    for obj in from_scene.init:
        for feat in obj.type.feature_names:
            assert from_obs.init.get(obj, feat) == \
                pytest.approx(from_scene.init.get(obj, feat), abs=1e-6)


def test_press_option_exists_and_turn_fan_on_does_not(env):
    from predicators.ground_truth_models import get_gt_options
    names = {o.name for o in get_gt_options(env.get_name())}
    assert "Press" in names
    assert "TurnFanOn" not in names
    assert "Push" not in names  # the wind moves the block, never the arm


def test_scene_with_two_blocks_is_refused(tmp_path):
    pytest.importorskip("markerless_estimation")
    scene = json.loads(json.dumps(_SCENE))
    scene["dominoes"].append(dict(_block((0.5, 0.3)), id=1))
    _config(_write_scene(tmp_path, scene))
    with pytest.raises(ValueError, match="one block"):
        PyBulletDominoBlowRealEnv(use_gui=False)
