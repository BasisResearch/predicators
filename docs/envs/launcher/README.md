# Launcher

A tabletop catapult range.
A spring launcher at the left end of the table holds a ball in its muzzle; the robot pushes the launcher's handle back a chosen depth and lets go, the handle snaps home, and the ball flies along the barrel's elevation toward a tower of blocks on a stand.
The goal is to topple the red top block of the tower while every block below it stays standing, with a handful of balls.

## What is hidden

- The launch law: the ball leaves at a spring constant times the compression the handle reached.
  On the base simulator the handle snaps home and the ball stays put.
- The mass of each block material (wood, stone).
- A flown ball that stops is put back in the muzzle while spare balls last; running out of balls, or toppling a block the goal wants standing, loses the level.

## What the agent controls

`Cock(robot, launcher)[approach_distance, contact_z_offset, depth]` and `Wait(robot)`.
The observation carries the launcher's compression and spare count, the ball's pose and speed, the stand's pose, and every block's pose, tilt, colour and target mark.

## Train and test

Train towers are two blocks, one of each material, with three spare balls; test towers are three blocks, farther away, with one spare.
Every level has a window of compressions that takes the top block alone, at least three scan points wide, and the generator records the middle of it.
The oracle plans with a `Hittable` helper answered by a launch probe and fires the middle of the window.

## Why it favours a learned model of the physics

Whether the ball meets the top block, the block below it, the stand or nothing is a matter of centimetres of trajectory height at the tower's distance.
An agent that has identified the spring constant from a training shot rolls the flight forward in the base simulator and reads that height off it; with two balls there is no room to bracket the answer by trial.

## Files

- `predicators/envs/pybullet_launcher_base.py`: the visible simulator (surfaced to agents as reference source).
- `predicators/envs/pybullet_launcher.py`: launch law, masses, tasks, evaluator, predicates.
- `predicators/ground_truth_models/launcher/`: options, helper predicates, processes.
- `tests/envs/test_pybullet_launcher.py`.
- `oracle_solve.mp4` in this directory: the oracle playing a train level then a test level under the continual protocol.
