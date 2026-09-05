# Magnets

A magnet puzzle on a mat.
The robot holds a wand from the start and never touches a piece with it: hovering the wand's tip near a piece moves the piece, toward the point under the tip for the colours the wand pulls and away from it for the colours it pushes.
The goal is a painted slot per pulled piece, in the piece's colour, with every piece still on the mat.

## What is hidden

- Each piece colour's polarity (pulled or pushed) and its range, and the speed law they share.
  On the base simulator the wand hovers and nothing moves.
- A piece that leaves the mat cannot be recovered: the level is lost.

## What the agent controls

`Hover(robot, wand)[x, y]` slides the tip to a point slowly, so a pulled piece follows; `Jump(robot, wand)[x, y]` moves at full speed, which no piece keeps up with; `Wait(robot)`.
The observation carries the wand's pose, every piece's pose, colour and planar speed, and every slot's pose and colour.

## Train and test

Train mats carry two pieces, test mats three, with distinct colours and at least one pulled colour with a slot.
The generator keeps a level only when the oracle's own plan (jump over a pulled piece, hover to its slot, next) clears it with no piece lost.
The oracle plans with `Pulled` / `Pushed` helpers and samplers that aim at the piece and at the slot.

## Why it favours a learned model of the physics

A hover path moves every piece within range at once, and a pushed piece near the mat's edge goes off it.
Per colour a sign and a distance, identified from a few hovers, let the base simulator roll out a whole path and show which pieces move where; trying the path for real is how a piece gets lost.

## Files

- `predicators/envs/pybullet_magnets_base.py`: the visible simulator (surfaced to agents as reference source).
- `predicators/envs/pybullet_magnets.py`: the field law, tasks, evaluator, predicates.
- `predicators/ground_truth_models/magnets/`: options, helper predicates, processes, the oracle's plan.
- `tests/envs/test_pybullet_magnets.py`.
- `oracle_solve.mp4` in this directory: the oracle playing a train level then a test level under the continual protocol.
