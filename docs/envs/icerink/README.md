# Ice rink

A tabletop shuffleboard with a hidden material physics.
Small square tiles sit on a slab of ice bounded by a wall along the far edge and the right edge; the near edge and the left edge are open.
The robot pushes a tile along one of four directions at a chosen stroke speed, and the tile slides until its friction, a wall or another tile stops it.
The goal is a painted target per tile, in the tile's colour.

## What is hidden

- Each tile colour is a material with its own sliding friction (blue ice, black rubber, green felt, grey steel).
  The base simulator gives every tile the same friction, so on it every colour slides the same distance.
- The dark strip across the slab brakes a tile crossing it with extra friction.
- A tile that slides past an open edge has left the rink for good: the level is lost.

## What the agent controls

`Push(robot, tile, direction)[approach_distance, contact_z_offset, speed]` and `Wait(robot)`.
The speed parameter is the gripper's speed through the stroke, and travel grows with it up to about 0.38 m/s.
The observation carries every tile's pose, colour and planar speed, and every target's pose and colour.

## Train and test

Train rinks carry two tiles, test rinks three, with distinct colours; every target is reachable by a single push from the start, and the generator records the push that made it.
The oracle (`oracle_process_planning`) plans with two helper predicates answered by a physics probe, `Reachable` and `WouldLeave`, and a sampler that bisects the stroke speed on the same probe.

## Why it favours a learned model of the physics

The answer to every push is a stopping point that depends on the tile's friction, the strip, the walls and the other tiles.
Two scalars per colour and one brake coefficient, identified from one slide each, let the base simulator predict it; trying the push for real costs the tile when the guess is wrong.

## Files

- `predicators/envs/pybullet_icerink_base.py`: the visible simulator (surfaced to agents as reference source).
- `predicators/envs/pybullet_icerink.py`: materials, the strip, tasks, evaluator, predicates.
- `predicators/ground_truth_models/icerink/`: options, helper predicates, processes.
- `tests/envs/test_pybullet_icerink.py`.
- `oracle_solve.mp4` in this directory: the oracle playing a train level then a test level under the continual protocol.
