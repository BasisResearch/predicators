"""SceneBase gives box and sphere bodies the sizes the observation gives."""
# pylint: disable=protected-access

from typing import Any, List, Tuple

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.scene_base import scene_base_class
from predicators.envs import create_new_env
from predicators.settings import CFG
from predicators.structs import State

_PLATFORM = (0.32, 0.36, 0.403)
_FRAME = (0.0, 0.0, 0.01)
_RED = (1.0, 0.0, 0.0, 1.0)


def _world(
    tmp_path: Any, inertial: Tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> Tuple[Any, State]:
    """A Fan scene whose platform_0 is a box built at level 2's size, offset in
    its own frame, with its centre of mass at ``inertial`` in its link frame,
    and whose ball is a sphere with damping of its own; every other object has
    no body."""
    utils.reset_config({
        "env": "pybullet_fan",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "scene_observed_sizes": True,
    })
    env = create_new_env("pybullet_fan", do_cache=False, use_gui=False)
    init = env.get_train_tasks()[0].task.init
    names = [o.name for o in init if o.type.name != "robot"]
    assert {"platform_0", "ball"} <= set(names)
    base: Any = scene_base_class("pybullet_fan", env.types, str(tmp_path))

    class Scene(base):  # type: ignore[misc,valid-type]
        """A platform and a ball."""
        AGENT_PARAM_SPECS = [
            ParamSpec("platform_lateral_friction", 0.9, lo=0.1, hi=1.5)
        ]

        @classmethod
        def initialize_pybullet(cls, using_gui: bool) -> Any:
            client, robot, bodies = super().initialize_pybullet(using_gui)
            for name in names:
                bodies[name] = None
            half = [s / 2.0 for s in _PLATFORM]
            bodies["platform_0"] = p.createMultiBody(
                0.0,
                p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=half,
                                       collisionFramePosition=_FRAME,
                                       physicsClientId=client),
                p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half,
                                    rgbaColor=_RED,
                                    visualFramePosition=_FRAME,
                                    physicsClientId=client),
                baseInertialFramePosition=inertial,
                physicsClientId=client)
            bodies["ball"] = p.createMultiBody(
                0.05,
                p.createCollisionShape(p.GEOM_SPHERE,
                                       radius=0.04,
                                       physicsClientId=client),
                p.createVisualShape(p.GEOM_SPHERE,
                                    radius=0.04,
                                    physicsClientId=client),
                physicsClientId=client)
            p.changeDynamics(bodies["ball"],
                             -1,
                             linearDamping=0.0,
                             rollingFriction=0.001,
                             physicsClientId=client)
            return client, robot, bodies

    return Scene(use_gui=False, skip_residual_dynamics=False), init


def _lossy(world: Any) -> List[Tuple[str, str]]:
    """The (object name, feature) pairs the world's last reset could not
    reproduce, the robot's aside: its joints come back through inverse
    kinematics."""
    return [(o.name, f) for o, f in world._last_unreconstructible_features
            if o.type.name != "robot"]


def _level(init: State,
           depth: float,
           radius: float,
           rot: float = 0.0,
           length: float = _PLATFORM[0]) -> State:
    """``init`` with platform_0 ``length`` long, ``depth`` deep, turned by
    ``rot`` and centred under the ball, and the ball of ``radius`` resting on
    it."""
    state = init.copy()
    platform = next(o for o in state if o.name == "platform_0")
    ball = next(o for o in state if o.name == "ball")
    state.set(platform, "x_len", length)
    state.set(platform, "y_len", depth)
    state.set(platform, "z_len", _PLATFORM[2])
    state.set(platform, "x", 0.4)
    state.set(platform, "y", 1.65)
    state.set(platform, "z", _PLATFORM[2] / 2.0)
    state.set(platform, "rot", rot)
    state.set(ball, "radius", radius)
    state.set(ball, "x", 0.4)
    state.set(ball, "y", 1.95)
    state.set(ball, "z", _PLATFORM[2] + _FRAME[2] + radius)
    return state


def test_a_box_and_a_sphere_take_each_level_s_observed_size(tmp_path):
    """A state with a deeper platform and a larger ball rebuilds both at the
    observed sizes, keeping each shape's frame, look, mass and materials; a
    ball over the deeper platform's far end then rests on it."""
    world, init = _world(tmp_path)
    client = world._physics_client_id
    world._set_state(_level(init, _PLATFORM[1], 0.04))
    platform, ball = world.body("platform_0"), world.body("ball")
    assert not _lossy(world)
    # The level-2 size the bodies were built with is already right.
    world._set_state(_level(init, _PLATFORM[1], 0.04))
    assert (world.body("platform_0"), world.body("ball")) == (platform, ball)

    world._set_state(_level(init, 0.7, 0.05))
    new_platform, new_ball = world.body("platform_0"), world.body("ball")
    assert new_platform != platform and new_ball != ball
    assert not _lossy(world)
    objects = {o.name: o for o in world._objects}
    assert objects["platform_0"].id == new_platform
    shape = p.getCollisionShapeData(new_platform, -1, physicsClientId=client)
    assert len(shape) == 1 and shape[0][2] == p.GEOM_BOX
    assert np.allclose(shape[0][3][:3], (_PLATFORM[0], 0.7, _PLATFORM[2]))
    assert np.allclose(shape[0][5], _FRAME)
    look = p.getVisualShapeData(new_platform, physicsClientId=client)
    assert np.allclose(look[0][3][:3], (_PLATFORM[0], 0.7, _PLATFORM[2]))
    assert np.allclose(look[0][7], _RED)
    platform_info = p.getDynamicsInfo(new_platform, -1, physicsClientId=client)
    assert platform_info[0] == 0.0
    # The declared friction is set on the rebuilt body too.
    assert platform_info[1] == pytest.approx(0.9)
    sphere = p.getCollisionShapeData(new_ball, -1, physicsClientId=client)
    assert sphere[0][3][0] == pytest.approx(0.05)
    ball_info = p.getDynamicsInfo(new_ball, -1, physicsClientId=client)
    assert ball_info[0] == pytest.approx(0.05)
    assert ball_info[6] == pytest.approx(0.001)
    assert world._read_material(new_ball, "linear_damping") == 0.0
    # The read-back state shows the bodies where the state put them.
    read = world._get_state()
    ball_obj = objects["ball"]
    assert read.get(ball_obj, "y") == pytest.approx(1.95, abs=1e-6)
    # The ball, past the old platform's far edge (y 1.83), rests on the new
    # one.
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
    z = p.getBasePositionAndOrientation(new_ball, physicsClientId=client)[0][2]
    assert z == pytest.approx(_PLATFORM[2] + _FRAME[2] + 0.05, abs=0.005)


def test_without_the_flag_a_body_keeps_its_size(tmp_path):
    """Off, the ball falls past the edge of the platform built for level 2."""
    world, init = _world(tmp_path)
    client = world._physics_client_id
    utils.update_config({"scene_observed_sizes": False})
    assert not CFG.scene_observed_sizes
    platform = world.body("platform_0")
    world._set_state(_level(init, 0.7, 0.04))
    assert world.body("platform_0") == platform
    for _ in range(240):
        p.stepSimulation(physicsClientId=client)
    ball = world.body("ball")
    assert p.getBasePositionAndOrientation(
        ball, physicsClientId=client)[0][2] < _PLATFORM[2]


def test_a_body_a_constraint_holds_keeps_its_size_and_says_so(tmp_path):
    """A constraint ties the ball: it keeps its body, and the reset lists its
    radius as a feature it could not reproduce."""
    world, init = _world(tmp_path)
    client = world._physics_client_id
    world._set_state(_level(init, _PLATFORM[1], 0.04))
    ball = world.body("ball")
    p.createConstraint(ball,
                       -1,
                       -1,
                       -1,
                       p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0.4, 1.95, 0.5],
                       physicsClientId=client)
    world._set_state(_level(init, _PLATFORM[1], 0.05))
    assert world.body("ball") == ball
    assert _lossy(world) == [("ball", "radius")]


def test_a_rebuilt_body_keeps_its_place_with_its_mass_off_centre(tmp_path):
    """A body whose centre of mass sits off its link frame is rebuilt where it
    was: PyBullet reports a shape's frame from the centre of mass and takes a
    new one's from the link frame."""
    world, init = _world(tmp_path, inertial=(0.0, 0.0, 0.03))
    client = world._physics_client_id
    world._set_state(_level(init, _PLATFORM[1], 0.04))
    low, high = p.getAABB(world.body("platform_0"), -1, physicsClientId=client)
    world._set_state(_level(init, 0.7, 0.04))
    new_low, new_high = p.getAABB(world.body("platform_0"),
                                  -1,
                                  physicsClientId=client)
    assert new_low[2] == pytest.approx(low[2], abs=1e-6)
    assert new_high[2] == pytest.approx(high[2], abs=1e-6)
    assert new_low[0] == pytest.approx(low[0], abs=1e-6)
    assert new_high[1] - new_low[1] == pytest.approx(0.7, abs=1e-6)
    assert (new_low[1] + new_high[1]) / 2 == pytest.approx(
        (low[1] + high[1]) / 2, abs=1e-6)


def test_a_turned_box_is_resized_only_when_its_reading_is_plain(tmp_path):
    """A box turned a quarter turn reads its own lengths and its world extents
    differently: sizes that match either reading leave it as built, and sizes
    that match neither are listed as not reconstructed, with no rebuild."""
    world, init = _world(tmp_path)
    turn = np.pi / 2
    world._set_state(_level(init, _PLATFORM[1], 0.04, rot=turn))
    platform = world.body("platform_0")
    # Its own lengths, then its world extents: both match the box built.
    world._set_state(_level(init, _PLATFORM[1], 0.04, rot=turn))
    assert world.body("platform_0") == platform and not _lossy(world)
    world._set_state(
        _level(init, _PLATFORM[0], 0.04, rot=turn, length=_PLATFORM[1]))
    assert world.body("platform_0") == platform and not _lossy(world)
    # A depth neither reading gives.
    world._set_state(_level(init, 0.7, 0.04, rot=turn))
    assert world.body("platform_0") == platform
    assert ("platform_0", "y_len") in _lossy(world)
