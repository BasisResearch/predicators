# Program world model synthesis (learn) first message

Composed by `AgentProgramWorldModelApproach._build_program_learn_message`
through `learn_prompts.build_program_learn_message`. The message carries
this cycle's data and digests; the rules of the phase live in the system
prompt (`learn_program_system.md`).

<!-- section: skeleton -->
Synthesize a world model program for this environment. There are
__N_TRAJS__ recorded trajectories (__N_TRANSITIONS__ skill-level
transitions) available: __N_DEMOS__ oracle demonstration(s), which
reached the goal by construction, and __N_INTERACTION__ interaction
trajectory/ies collected during online learning, some of which may
have failed to reach the goal.

__TRAJECTORY_LISTING__

Each trajectory carries a `train_task_idx`. `is_goal_state(state,
task_idx)` (equivalently `train_tasks[task_idx].goal_holds(state)`)
checks a single state for the goal atoms. Reaching the goal atoms does
not by itself mean an episode is solved; when a task objective is
stated below, score full trajectories with `evaluate_trajectory`. Use
`is_goal_state` to confirm which trajectories reached the goal atoms
and to treat failed interaction trajectories as counterexamples: places
where the environment disagreed with what a skill was expected to do.

__OBJECTIVE_BLOCK__

__PRIOR_STATE_BLOCK__

Data-structure source code is at: __STRUCTS_REF__

## Available Predicates (for subgoal annotations)

__PREDICATE_LISTING__

Subgoal annotations in plans for `sim.refine` / `sim.run` must
reference these predicate names with matching arity and types.

## Object Types

__TYPES_DIGEST__

## Options

Plans (for `sim.refine` / `sim.run`) and your `transition` must match
these typed signatures and parameter boxes exactly:

__OPTIONS_DIGEST__

__TOOLS_BLOCK__

## This session

Read the data-structures file first, then explore the trajectory data
with `run_python`. Write your world model to `__WORLD_MODEL_FILE__`,
defining `LATENT_FEATURES`, `initial_latent`, and `transition`, and
iterate with `Edit` and `sim.score()`. Pass `task_idx` explicitly to
`sim.reset`; `sim.task(task_idx)` prints a task digest. Finish with the
deliverables listed in the system prompt: a final `sim.score()`, the
GO/NO-GO check, the decision record, `./open_questions.md`, and
`./strategy.md`.

<!-- section: zero_shot -->
## Zero-shot synthesis

No trajectory has been recorded and none will be before you finish:
this session is the whole learning phase, and what you write here is
what the planner uses on the test tasks. The trajectory counts above
are zero for that reason, and `sim.score` has no data to score
against. Build the world model from the task description, the object
types and options, the scene (`sim.task`, `sim.reset`, `sim.render`)
and your own knowledge of the mechanisms involved, and validate it
with `sim.refine` / `sim.run` rollouts of a full plan. State each
mechanism you commit to, and the evidence you would want for it, in
the decision record.
