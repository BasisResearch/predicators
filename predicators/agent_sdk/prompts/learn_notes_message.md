# Natural-language world model (learn) first message

Composed by `AgentNotesWorldModelApproach._build_notes_learn_message`
through `learn_prompts.build_notes_learn_message`.

<!-- section: skeleton -->
Write the world model document for this environment. There are
__N_TRAJS__ recorded trajectories (__N_TRANSITIONS__ skill-level
transitions) available: __N_DEMOS__ oracle demonstration(s), which
reached the goal by construction, and __N_INTERACTION__ interaction
trajectory/ies collected during online learning, some of which may
have failed to reach the goal.

__TRAJECTORY_LISTING__

Each trajectory carries a `train_task_idx`. `is_goal_state(state,
task_idx)` (equivalently `train_tasks[task_idx].goal_holds(state)`)
checks a single state for the goal atoms. Use it to confirm which
trajectories reached the goal and to treat failed interaction
trajectories as counterexamples: places where the environment
disagreed with what a skill was expected to do.

__OBJECTIVE_BLOCK__

__GOAL_BLOCK__

__PRIOR_NOTES_BLOCK__

Data-structure source code is at: __STRUCTS_REF__

## Available Predicates

__PREDICATE_LISTING__

## Object Types

__TYPES_DIGEST__

## Options

__OPTIONS_DIGEST__

__TOOLS_BLOCK__

## This session

Read the data-structures file first, then explore the trajectory data
with `run_python`. Write your world model to `__NOTES_FILE__` under the
headings given in the system prompt, and finish with the deliverables
listed there.

<!-- section: goal -->
## Task goals (natural language)

__GOALS__

<!-- section: prior_notes -->
A `world_model.md` from an earlier cycle exists at `__NOTES_FILE__`.
Read it first; this cycle's data may confirm, refine, or contradict
what it says. Revise it in place.

<!-- section: zero_shot -->
## Zero-shot synthesis

No trajectory has been recorded and none will be before you finish:
this session is the whole learning phase, and what you write here is
what the planner reasons with on the test tasks. The trajectory counts
above are zero for that reason. Build the document from the task
description, the object types and options, and your own knowledge of
the mechanisms involved, and label every claim as a hypothesis with
the evidence you would want for it.
