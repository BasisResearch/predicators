"""A SceneBase reset builds the world a fresh scene builds for the state."""
# pylint: disable=protected-access

from typing import Any, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.scene_base import scene_base_class
from predicators.envs import create_new_env
from predicators.structs import Action, State

# Where the scene builds the ball: away from everything a level holds.
_PARKED = (3.0, 3.0, 0.5)


def _scene(tmp_path: Any) -> Tuple[Any, State]:
    """A Fan scene whose ball body is built parked, with a plate that slides on
    a prismatic joint, and with no body for every other object; and the level's
    initial state."""
    utils.reset_config({
        "env": "pybullet_fan",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
    })
    env = create_new_env("pybullet_fan", do_cache=False, use_gui=False)
    init = env.get_train_tasks()[0].task.init
    names = [o.name for o in init if o.type.name != "robot"]
    assert "ball" in names
    base: Any = scene_base_class("pybullet_fan", env.types, str(tmp_path))

    class Scene(base):  # type: ignore[misc,valid-type]
        """A ball that carries a sliding plate."""

        @classmethod
        def initialize_pybullet(cls, using_gui: bool) -> Any:
            client, robot, bodies = super().initialize_pybullet(using_gui)
            for name in names:
                bodies[name] = None
            bodies["ball"] = p.createMultiBody(
                0.05,
                p.createCollisionShape(p.GEOM_SPHERE,
                                       radius=0.04,
                                       physicsClientId=client),
                basePosition=_PARKED,
                linkMasses=[0.01],
                linkCollisionShapeIndices=[
                    p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[0.02, 0.02, 0.005],
                                           physicsClientId=client)
                ],
                linkVisualShapeIndices=[-1],
                linkPositions=[[0.0, 0.0, 0.05]],
                linkOrientations=[[0.0, 0.0, 0.0, 1.0]],
                linkInertialFramePositions=[[0.0, 0.0, 0.0]],
                linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]],
                linkParentIndices=[0],
                linkJointTypes=[p.JOINT_PRISMATIC],
                linkJointAxis=[[0.0, 0.0, 1.0]],
                physicsClientId=client)
            return client, robot, bodies

    return Scene(use_gui=False), init


def test_a_body_the_state_does_not_name_goes_back_where_it_was_built(tmp_path):
    """After a rollout in a level that names the ball, a reset to a level
    without it puts the ball's body back where the scene built it, at rest and
    with its plate where it was built, as in a fresh scene."""
    scene, init = _scene(tmp_path)
    try:
        client = scene._physics_client_id
        body = scene.body("ball")
        ball = next(o for o in init if o.name == "ball")
        without = init.copy()
        del without.data[ball]
        action = Action(np.array(init.joint_positions, dtype=np.float32))
        state = scene.simulate(init, action)
        p.resetJointState(body, 0, 0.03, 0.5, physicsClientId=client)
        for _ in range(3):
            state = scene.simulate(state, action)
        position = p.getBasePositionAndOrientation(body,
                                                   physicsClientId=client)[0]
        assert np.linalg.norm(np.subtract(position, _PARKED)) > 1.0
        scene._set_state(without)
        position, orientation = p.getBasePositionAndOrientation(
            body, physicsClientId=client)
        assert np.allclose(position, _PARKED)
        assert np.allclose(orientation, (0.0, 0.0, 0.0, 1.0))
        assert np.allclose(p.getBaseVelocity(body, physicsClientId=client),
                           0.0)
        plate = p.getJointState(body, 0, physicsClientId=client)
        assert np.allclose(plate[:2], 0.0)
        # A state that names the ball still places it.
        scene._set_state(init)
        position = p.getBasePositionAndOrientation(body,
                                                   physicsClientId=client)[0]
        assert np.allclose(position, [init.get(ball, f) for f in "xyz"],
                           atol=1e-4)
    finally:
        scene.dispose()
