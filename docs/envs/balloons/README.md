# Balloons

Balloons and a box.
A box sits at the left of the table, balloons rest in a rack along the back, each held by a clip in front of it, and a translucent band floats beside the box's column.
The robot pushes a clip open; the freed balloon's string pulls it to the box, and it pulls the box up.
The goal is the box hanging at rest with its centre inside the band, with no balloon burst on the ceiling.

## What is hidden

- Each balloon colour's lift, which fades linearly with height, so a box with enough balloons rises to the height where the pull matches its weight and hangs there.
  On the base simulator an opened clip frees nothing: the balloon stays in the rack and the box stays put.
- The release itself (an open clip's balloon flies to the end of its string above the box, later balloons stacking above earlier ones), the mass of each box material, and the air's drag.
- A freed balloon that reaches the ceiling bursts: the level is lost, and a freed balloon cannot be clipped back.

## What the agent controls

`Release(robot, clip)[approach, contact_z]` (the shared push skill on the clip's toggle) and `Wait(robot)`.
The observation carries the box's pose, colour and speed, every balloon's pose, colour, tied and popped flags, every clip's pose and latched state, and the band's heights.

## Train and test

Train levels have a pine or oak box and two or three balloons; test levels bring a teak box, heavier than both, and four balloons.
Every level has exactly one subset of its balloons whose lift hangs the box in the band by the analytic law, and the generator keeps a level only when the oracle's own plan (open that subset's clips, weakest lift first) hangs the box in the band on the simulator's physics.
The oracle plans with `Needed` and `Holds` helpers and a derived `AllNeededTied`.

## Why it favours a learned model of the physics

Which clips to open is a small choice, but the height each choice gives is a quantitative composition of per colour lifts, the fade and the box's mass, and one balloon too many bursts on the ceiling.
An agent that has identified the lifts and the fade from a training level computes the subset for a heavier box it has never seen; an agent guessing pays with the level.

## Files

- `predicators/envs/pybullet_balloons_base.py`: the visible simulator (surfaced to agents as reference source).
- `predicators/envs/pybullet_balloons.py`: release, lift, pop, masses, tasks, evaluator, predicates.
- `predicators/ground_truth_models/balloons/`: options, helper predicates, processes, the oracle's plan.
- `tests/envs/test_pybullet_balloons.py`.
- `oracle_solve.mp4` in this directory: the oracle playing a train level then a test level under the continual protocol.
