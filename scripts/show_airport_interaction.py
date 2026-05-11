
import os
import sys
import time
import numpy as np
import pybullet as p

# Add the project root to sys.path
sys.path.append(os.getcwd())

from predicators.envs.pybullet_airport import PyBulletAirportEnv
from predicators.settings import CFG
from predicators.structs import Action
from predicators.pybullet_helpers.geometry import Pose, Pose3D

def show_interaction():
    CFG.seed = 0
    CFG.env = "pybullet_airport"
    CFG.num_train_tasks = 1
    CFG.pybullet_control_mode = "position"
    CFG.pybullet_sim_steps_per_action = 100
    
    print("Initializing Airport environment in GUI mode...")
    # use_gui=True to see the simulation
    env = PyBulletAirportEnv(use_gui=True)
    state = env.reset("train", 0)
    
    # Set camera for better view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[1.5, 0.75, 0.4],
        physicsClientId=env._physics_client_id
    )
    
    button = [obj for obj in state.get_objects(env._button_type)][0]
    pusher = [obj for obj in state.get_objects(env._pusher_type)][0]
    
    # Button position
    bx, by, bz = env.button_stand_x, env.button_stand_y, env.button_stand_z + env.button_height / 2.0
    
    def move_to_pose(target_pose_pos: Pose3D, num_steps: int = 50):
        target_pose = Pose(target_pose_pos, env.get_robot_ee_home_orn())
        try:
            joint_positions = env._pybullet_robot.inverse_kinematics(target_pose, validate=False)
            act = Action(np.array(joint_positions, dtype=np.float32))
            for _ in range(num_steps):
                env.step(act)
                time.sleep(0.01)
            return act
        except Exception as e:
            print(f"IK failed for pose {target_pose_pos}: {e}")
            return None

    print("Moving robot to approach button...")
    move_to_pose((bx, by, bz + 0.1))
    
    print("Pressing button...")
    # Move significantly lower to force the button down with physics
    press_act = move_to_pose((bx, by, bz - 0.1), num_steps=100)
    
    if press_act is not None:
        print("Holding button and watching pusher...")
        for i in range(200):
            state = env.step(press_act)
            if i % 20 == 0:
                print(f"Step {i}: Pusher y = {state.get(pusher, 'y'):.4f}, Button is_pressed = {state.get(button, 'is_pressed')}")
            time.sleep(0.01)
        
    print("Releasing button...")
    release_act = move_to_pose((bx, by, bz + 0.1), num_steps=50)
    
    if release_act is not None:
        print("Watching pusher return...")
        for i in range(100):
            state = env.step(release_act)
            if i % 20 == 0:
                print(f"Step {i}: Pusher y = {state.get(pusher, 'y'):.4f}, Button is_pressed = {state.get(button, 'is_pressed')}")
            time.sleep(0.01)

    print("Interaction complete. Closing in 5 seconds...")
    time.sleep(5)

if __name__ == "__main__":
    show_interaction()
