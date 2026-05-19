from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco
import numpy as np

def run_random_actions():
    # Apply parser defaults to predicators' global CFG (only needed when
    # consuming the envs as a library rather than via main.py).
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})

    robodisco.register_all_environments()
    env = robodisco.make("robodisco/Airport-v0", render_mode="human")

    # Access the underlying pybullet environment.
    pybullet_env = env.unwrapped._env  # type: ignore

    obs, info = env.reset()

    for _ in range(200):
        # action = env.action_space.sample()
        # zero actions
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)

        # Transform the same local-frame particles into current world coords.
        particles = pybullet_env.extract_particles()
        pybullet_env.clear_particles()
        pybullet_env.display_particles(particles, point_size=10.0, obj_color=False)

    frame = env.render()  # (H, W, 3) uint8 RGB array

    import pybullet as p
    import time
    print("Particles displayed. Rotate camera as needed. Close window to exit.")
    while p.isConnected(physicsClientId=pybullet_env._physics_client_id):
        time.sleep(0.1)

    env.close()

if __name__ == "__main__":
    run_random_actions()
