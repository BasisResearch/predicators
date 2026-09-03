"""Live particle visualization demo.

Steps the Airport env with zero actions and draws each object's downsampled
point cloud as PyBullet debug points, refreshed every step.

Run from the repo root::

    python -m particle_world_model.run
"""

import time

import numpy as np
import pybullet as p

from particle_world_model.particles import ParticleDebugDraw, extract_particles
from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco


def run_random_actions():
    # Apply parser defaults to predicators' global CFG (only needed when
    # consuming the envs as a library rather than via main.py).
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})

    robodisco.register_all_environments()
    env = robodisco.make("robodisco/Airport-v0", render_mode="human")

    # Access the underlying pybullet environment.
    pybullet_env = env.unwrapped._env  # type: ignore

    env.reset()

    viz = ParticleDebugDraw(pybullet_env)
    for _ in range(200):
        # Zero actions.
        action = np.zeros(env.action_space.shape)
        env.step(action)

        # Re-extract and redraw the particles for the current world state.
        particles = extract_particles(pybullet_env)
        viz.clear()
        viz.display(particles, point_size=10.0, obj_color=False)

    env.render()  # (H, W, 3) uint8 RGB array

    print("Particles displayed. Rotate camera as needed. Close window to exit.")
    while p.isConnected(physicsClientId=pybullet_env._physics_client_id):
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    run_random_actions()
