Write the world model document for this environment. There are 2 recorded trajectories (9 skill-level transitions) available: 0 oracle demonstration(s), which reached the goal by construction, and 2 interaction trajectory/ies collected during online learning, some of which may have failed to reach the goal.

  [0] interaction, task 0
  [1] interaction, task 0

Each trajectory carries a `train_task_idx`. `is_goal_state(state, task_idx)` (equivalently `train_tasks[task_idx].goal_holds(state)`) checks a single state for the goal atoms. Use it to confirm which trajectories reached the goal and to treat failed interaction trajectories as counterexamples: places where the environment disagreed with what a skill was expected to do.

## Task goals (natural language)

- Build the bridge.

A `world_model.md` from an earlier cycle exists at `./world_model.md`. Read it first; this cycle's data may confirm, refine, or contradict what it says. Revise it in place.

Data-structure source code is at: ./reference/structs.py

## Available Predicates

- Holding(robot:robot, block:block)

## Object Types

- robot: hand
- block: x, y, held

## Options

- Pick(robot:robot, block:block)[]

## Available Tools

  - run_python

## This session

Read the data-structures file first, then explore the trajectory data with `run_python`. Write your world model to `./world_model.md` under the headings given in the system prompt, and finish with the deliverables listed there.
