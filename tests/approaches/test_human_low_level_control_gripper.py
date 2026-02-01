"""Test to verify gripper open/close is instant."""

import numpy as np

from predicators import utils
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv

# Configure
utils.reset_config({
    'env': 'pybullet_circuit',
    'approach': 'human_low_level_control',
    'seed': 0,
    'human_control_move_speed': 0.1,
    'human_control_rot_speed': 0.2,
    'pybullet_max_vel_norm': 0.15,
    'pybullet_sim_steps_per_action': 5,
})

print("Creating environment (no GUI)...")
env = PyBulletCircuitEnv(use_gui=False)

print("Resetting environment...")
obs = env.reset("test", 0)
state = obs

print("\n" + "=" * 60)
print("TEST: Gripper toggle should be instant (1 step)")
print("=" * 60)

from predicators.approaches import create_approach

# Create approach
approach = create_approach(
    'human_low_level_control',
    set(),  # predicates
    set(),  # options
    env.types,
    env.action_space,
    [])

# Get task and create policy
task = env.get_task("test", 0)

# Disable terminal setup for testing
approach._setup_terminal = lambda: None
approach._restore_terminal = lambda: None

policy = approach.solve(task, timeout=10)

# Get robot for checking finger positions
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv as CircuitEnv

_, shadow_robot, _ = CircuitEnv.initialize_pybullet(using_gui=False)

print(f"\nFully open position: {shadow_robot.open_fingers}")
print(f"Fully closed position: {shadow_robot.closed_fingers}")

# Test 1: Toggle to closed
print("\n--- Test 1: Toggle gripper to CLOSED (spacebar) ---")
initial_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"Initial finger position: {initial_finger_pos:.4f}")

approach._get_pressed_key = lambda: ' '  # Simulate spacebar
action = policy(state)
state = env.step(action)

final_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"After 1 step: {final_finger_pos:.4f}")

# Check if finger moved significantly toward closed position in just 1 step
if abs(final_finger_pos - shadow_robot.closed_fingers) < 0.001:
    print("✓ PASS: Gripper closed instantly in 1 step")
    test1_pass = True
else:
    print(
        f"✗ FAIL: Gripper not fully closed. Distance from target: {abs(final_finger_pos - shadow_robot.closed_fingers):.4f}"
    )
    test1_pass = False

# Test 2: Toggle back to open
print("\n--- Test 2: Toggle gripper to OPEN (spacebar) ---")
approach._step_count += 10  # Advance step count to avoid debounce
approach._get_pressed_key = lambda: ' '  # Simulate spacebar again
action = policy(state)
state = env.step(action)

final_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"After 1 step: {final_finger_pos:.4f}")

if abs(final_finger_pos - shadow_robot.open_fingers) < 0.001:
    print("✓ PASS: Gripper opened instantly in 1 step")
    test2_pass = True
else:
    print(
        f"✗ FAIL: Gripper not fully open. Distance from target: {abs(final_finger_pos - shadow_robot.open_fingers):.4f}"
    )
    test2_pass = False

# Test 3: Verify no drift when not pressing keys after gripper toggle
print("\n--- Test 3: No drift after gripper toggle ---")
approach._get_pressed_key = lambda: None  # No key pressed
for i in range(5):
    action = policy(state)
    state = env.step(action)

drift_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"Finger position after 5 no-op steps: {drift_finger_pos:.4f}")

if abs(drift_finger_pos - final_finger_pos) < 0.001:
    print("✓ PASS: No finger drift after gripper toggle")
    test3_pass = True
else:
    print(
        f"✗ FAIL: Finger drifted by {abs(drift_finger_pos - final_finger_pos):.4f}"
    )
    test3_pass = False

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
if test1_pass and test2_pass and test3_pass:
    print("✓ ALL TESTS PASSED: Gripper is instant and doesn't drift")
else:
    print(f"✗ SOME TESTS FAILED:")
    print(f"  - Instant close: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  - Instant open: {'PASS' if test2_pass else 'FAIL'}")
    print(f"  - No drift: {'PASS' if test3_pass else 'FAIL'}")

print("\nTest complete!")
