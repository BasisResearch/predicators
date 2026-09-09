# Crane

A wrecking ram and a crate.
A crane's arm hangs a steel ram just above the table at the left end of a lane, a crate stands in the lane a little to the right, and a translucent bin pad lies further along it, near the table's far edge.
The robot draws the ram back along its arc and lets go; the ram swings, strikes the crate, and the crate slides down the lane.
The goal is the crate at rest on the pad.

## What is hidden

- The mass of each crate material and its grip on the table, which together set how far a given blow sends it.
  On the base simulator every crate weighs and grips the same.
- The hinge's drag on the swing.

## What the agent controls

`Pull(robot, ram, crane)[approach, contact_z, pull]` (the shared push skill aimed at the ram's rest pose with a stroke along the chord of the arc, the hand open so the fingertips bracket the head, and a vertical lift before the retreat so the swing is released cleanly) and `Wait(robot)`.
The observation carries the ram's rest pose, swing angle and speed, the anchor and the arm's length, the crate's pose, colour and speed, and the pad's position and width.

## Train and test

Train levels have a foam or iron crate on an arm of 0.5 or 0.55 m; test levels bring a stone crate, heavier than iron, on a longer arm of 0.65 or 0.7 m, which swings slower for the same pull.
Every level's pad is placed where some pull actually sends the crate, and the generator keeps a level only when the working pulls form a window at least three scan steps wide; the recorded solution is the middle of that window.
A crate that slides off the far edge, or stops short where the ram can no longer reach it, loses the level.
The oracle plans with a `Hittable` helper answered by the same swing probe.

## Why it favours a learned model of the physics

The pull is one number, but where the crate lands is a composition of the arm's length (visible), the crate's mass and grip (hidden behind its colour) and the hinge's drag.
An agent that has identified the materials from training swings can set the pull for a heavier crate on a longer arm it has never seen; an agent guessing pays with the level.

## Files

- `predicators/envs/pybullet_crane_base.py`: the visible simulator (surfaced to agents as reference source).
- `predicators/envs/pybullet_crane.py`: materials, hinge drag, tasks, evaluator, predicates, the swing probe.
- `predicators/ground_truth_models/crane/`: options, helper predicates, processes.
- `tests/envs/test_pybullet_crane.py`.
- `oracle_solve.mp4` in this directory: the oracle playing a train level then a test level under the continual protocol.
