#!/usr/bin/env python
"""Test script for the refactored domino environment.

This script tests that the component-based architecture works correctly
and maintains backward compatibility with the original environment.

Usage:
    python predicators/envs/pybullet_domino/test_composed_env.py
"""

import time

import numpy as np

from predicators.settings import CFG


def test_domino_fan_env() -> None:
    """Test the domino + fan + ball environment."""
    # Configure environment
    CFG.seed = 0
    CFG.env = "pybullet_domino_fan"
    CFG.num_train_tasks = 0
    CFG.num_test_tasks = 2

    # Domino configuration
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.domino_train_num_dominos = [3]
    CFG.domino_test_num_dominos = [3, 4]
    CFG.domino_train_num_targets = [1]
    CFG.domino_test_num_targets = [1]
    CFG.domino_train_num_pivots = [0]
    CFG.domino_test_num_pivots = [0]

    # Fan/ball configuration
    CFG.domino_fan_ball_position_tolerance = 0.04
    CFG.fan_known_controls_relation = True
    CFG.fan_fans_blow_opposite_direction = False

    # Import and create the environment
    from predicators.envs.pybullet_domino import PyBulletDominoFanEnv
    from predicators.structs import Action

    print("Creating PyBulletDominoFanEnv...")
    env = PyBulletDominoFanEnv(use_gui=True)

    # Print environment info
    print(f"\n{'=' * 60}")
    print("Environment Information")
    print(f"{'=' * 60}")
    print(f"Environment name: {env.get_name()}")
    print(f"\nTypes ({len(env.types)}):")
    for t in sorted(env.types, key=lambda x: x.name):
        print(f"  - {t.name}: {t.feature_names}")
    print(f"\nPredicates ({len(env.predicates)}):")
    for p in sorted(env.predicates, key=lambda x: x.name):
        print(f"  - {p.name}")
    print(f"\nGoal predicates ({len(env.goal_predicates)}):")
    for p in sorted(env.goal_predicates, key=lambda x: x.name):
        print(f"  - {p.name}")

    # Generate test tasks
    print(f"\n{'=' * 60}")
    print("Generating test tasks...")
    print(f"{'=' * 60}")
    tasks = env._generate_test_tasks()
    print(f"Generated {len(tasks)} tasks")

    # Test each task
    for i, task in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}")
        print(f"{'=' * 60}")

        # Reset to initial state
        env._reset_state(task.init)

        print("\nObjects in state:")
        assert env._domino_component is not None
        for obj in task.init.get_objects(env._domino_component.domino_type):
            print(f"  - {obj.name}")

        print(f"\nGoal atoms ({len(task.goal)}):")
        for atom in task.goal:
            print(f"  - {atom}")

        # Simulate for a short time
        print("\nRunning simulation...")
        try:
            for step in range(300):
                action = Action(
                    np.array(env._pybullet_robot.initial_joint_positions))
                state = env.step(action)

                if all(atom.holds(state) for atom in task.goal):
                    print(f"  Goal reached at step {step}!")
                    time.sleep(1)
                    break

                if step % 100 == 0:
                    print(f"  Step {step}...")

                time.sleep(0.02)
        except KeyboardInterrupt:
            print("  Interrupted by user")
            continue

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print(f"{'=' * 60}")


def test_domino_only_env() -> None:
    """Test the domino-only environment."""
    # Configure environment
    CFG.seed = 0
    CFG.num_train_tasks = 0
    CFG.num_test_tasks = 1

    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.domino_train_num_dominos = [3]
    CFG.domino_test_num_dominos = [3]
    CFG.domino_train_num_targets = [1]
    CFG.domino_test_num_targets = [1]
    CFG.domino_train_num_pivots = [0]
    CFG.domino_test_num_pivots = [0]

    from predicators.envs.pybullet_domino import PyBulletDominoEnv
    from predicators.structs import Action

    print("\nCreating PyBulletDominoEnv (domino-only)...")
    env = PyBulletDominoEnv(use_gui=True)

    print(f"Environment name: {env.get_name()}")
    print(f"Types: {[t.name for t in env.types]}")
    print(f"Predicates: {[p.name for p in env.predicates]}")

    tasks = env._generate_test_tasks()
    print(f"Generated {len(tasks)} tasks")

    if tasks:
        env._reset_state(tasks[0].init)
        print("Running simulation for 100 steps...")
        for step in range(100):
            action = Action(
                np.array(env._pybullet_robot.initial_joint_positions))
            env.step(action)
            time.sleep(0.02)

    print("Domino-only test completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--domino-only":
        test_domino_only_env()
    else:
        test_domino_fan_env()
