Synthesize a world model program for this environment. There are 0 recorded trajectories (0 skill-level transitions) available: 0 oracle demonstration(s), which reached the goal by construction, and 0 interaction trajectory/ies collected during online learning, some of which may have failed to reach the goal.

Each trajectory carries a `train_task_idx`. `is_goal_state(state, task_idx)` (equivalently `train_tasks[task_idx].goal_holds(state)`) checks a single state for the goal atoms. Reaching the goal atoms does not by itself mean an episode is solved; when a task objective is stated below, score full trajectories with `evaluate_trajectory`. Use `is_goal_state` to confirm which trajectories reached the goal atoms and to treat failed interaction trajectories as counterexamples: places where the environment disagreed with what a skill was expected to do.

Data-structure source code is at: ./reference/structs.py

## Available Predicates (for subgoal annotations)

- Holding(robot:robot, block:block)

Subgoal annotations in plans for `sim.refine` / `sim.run` must reference these predicate names with matching arity and types.

## Object Types

- robot: hand
- block: x, y, held

## Options

Plans (for `sim.refine` / `sim.run`) and your `transition` must match these typed signatures and parameter boxes exactly:

- Pick(robot:robot, block:block)[]

## Available Tools

  - run_python

## This session

Read the data-structures file first, then explore the trajectory data with `run_python`. Write your world model to `./world_model.py`, defining `LATENT_FEATURES`, `initial_latent`, and `transition`, and iterate with `Edit` and `sim.score()`. Pass `task_idx` explicitly to `sim.reset`; `sim.task(task_idx)` prints a task digest. Finish with the deliverables listed in the system prompt: a final `sim.score()`, the GO/NO-GO check, the decision record, `./open_questions.md`, and `./strategy.md`.

## Zero-shot synthesis

No trajectory has been recorded and none will be before you finish: this session is the whole learning phase, and what you write here is what the planner uses on the test tasks. The trajectory counts above are zero for that reason, and `sim.score` has no data to score against. Build the world model from the task description, the object types and options, the scene (`sim.task`, `sim.reset`, `sim.render`) and your own knowledge of the mechanisms involved, and validate it with `sim.refine` / `sim.run` rollouts of a full plan. State each mechanism you commit to, and the evidence you would want for it, in the decision record.
