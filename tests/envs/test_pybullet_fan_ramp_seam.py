"""A ball resting on the seam between Fan's ramp and its landing platform rolls
off under a fan's push, as it does a few millimetres away."""
# pylint: disable=protected-access
import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.structs import Action, State
from predicators.utils import PyBulletState


def _push(env: PyBulletFanEnv, state: State, x: float, z: float,
          fan_side: int) -> float:
    """How far the ball rolls in 60 steps from ``x`` with only the fan facing
    ``fan_side`` on."""
    staged = state.copy()
    ball = env._ball
    staged.set(ball, "x", x)
    staged.set(ball, "z", z)
    for obj in staged:
        if obj.type.name == "fan":
            on = float(int(round(staged.get(obj, "facing_side"))) == fan_side)
            staged.set(obj, "is_on", on)
        elif obj.type.name == "switch":
            on = float(int(round(staged.get(obj, "controls_fan"))) == fan_side)
            staged.set(obj, "is_on", on)
    env._set_state(staged)
    current = env._get_state()
    for _ in range(60):
        assert isinstance(current, PyBulletState)
        current = env.step(
            Action(np.asarray(current.joint_positions, dtype=np.float32)))
    return current.get(ball, "x") - x


def test_a_ball_on_the_ramp_seam_rolls_off_under_the_east_fan() -> None:
    """The ramp meets its landing platform edge to edge.

    A box's collision margin rounds its edges, and the landing
    platform's rounded edge was a curb about a millimetre high at the
    seam: a ball resting there stayed put under the east fan, which
    rolls the same ball 3 cm away across the platform. Every platform
    is built with no margin, as the ramp is, and the ball on the seam
    rolls too.
    """
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
        state = env.reset("train", 0).copy()
        ramp, = [o for o in state if o.type.name == "ramp"]
        assert abs(state.get(ramp, "rot")) < 1e-6
        seam = state.get(ramp, "x") + state.get(ramp, "x_len") / 2
        landing, = [
            o for o in state if o.type.name == "platform" and
            abs(state.get(o, "x") - state.get(o, "x_len") / 2 - seam) < 1e-6
        ]
        top = state.get(landing, "z") + state.get(landing, "z_len") / 2
        z = top + env.ball_radius
        on_the_seam = _push(env, state, seam + 0.0005, z, fan_side=0)
        beside_it = _push(env, state, seam + 0.03, z, fan_side=0)
        assert beside_it > 0.05
        assert on_the_seam > 0.05
        for obj in state:
            if obj.type.name in ("platform", "ramp"):
                body = env._boundary_named(obj).id
                margin = p.getDynamicsInfo(body, -1,
                                           physicsClientId=client)[11]
                assert margin == 0.0, obj.name
    finally:
        p.disconnect(client)
