# Synthesis (learn) first message

Composed by `AgentSimLearningApproach._build_synthesis_learn_message`
through `learn_prompts.build_learn_message`. The message carries this
cycle's data and digests; the rules of the phase live in the system
prompt (`learn_system.md`).

<!-- section: skeleton -->
Synthesize a residual dynamics simulator for this environment. There
are __N_TRAJS__ trajectories (__N_TRANSITIONS__ step transitions)
available: __N_DEMOS__ oracle demonstration(s), which reached the goal
by construction, and __N_INTERACTION__ interaction trajectory/ies
collected during online learning, some of which may have failed to
reach the goal.

__TRAJECTORY_LISTING__

Each trajectory carries a `train_task_idx`. `is_goal_state(state,
task_idx)` (equivalently `train_tasks[task_idx].goal_holds(state)`)
checks a single state for the goal atoms. Reaching the goal atoms does
not by itself mean an episode is solved; when a task objective is
stated below, score full trajectories with `evaluate_trajectory`. Use
`is_goal_state` to confirm which trajectories reached the goal atoms
and to treat failed interaction trajectories as counterexamples: places
where a predicate or rule said "this should work" and the environment
disagreed.

__OBJECTIVE_BLOCK__

__PRIOR_STATE_BLOCK__

__DIVERGENCE_BLOCK__

Data-structure source code is at: __STRUCTS_REF__

__BASE_SIM_BLOCK__

A residual scan between the base simulator's prediction and the
observed next state suggests that these features carry residual
dynamics (a starting hint; it may include base-sim jitter, so refine it
as you go):

__INFERRED_HINT__

## Available Predicates (for subgoal annotations)

__PREDICATE_LISTING__

Subgoal annotations in plans for `sim.refine` / `sim.run` must
reference these predicate names with matching arity and types. Any
threshold or condition you bake into a rule must be consistent with
what the predicate's classifier checks, or refinement rejects
parameter samples that look correct on paper.

## Object Types

__TYPES_DIGEST__

## Options

Plans (for `sim.refine` / `sim.run`) and rules must match these typed
signatures and parameter boxes exactly:

__OPTIONS_DIGEST__

__TOOLS_BLOCK__

## This session

Read the data-structures file first, then explore the trajectory data
with `run_python`. Write your simulator to `__SIMULATOR_FILE__`,
defining `RESIDUAL_RULES`, `PARAM_SPECS`, and `RESIDUAL_FEATURES`, and
iterate with `Edit` and re-scoring. Pass `task_idx` explicitly to
`sim.reset`; `sim.task(task_idx)` prints a task digest. Finish with the
deliverables listed in the system prompt: a final `sim.fit()`, the
GO/NO-GO check, the decision record, `./open_questions.md`, and
`./strategy.md`.

<!-- section: divergence_prior -->
## Where the prior model diverges from the data

Computed just now from the prior cycle's `simulator.py` with its
parameters refit to all trajectories above, so the remaining
mismatches need structural fixes, not tuning. Re-score any edit with
`sim.residuals()` (same report, current file):

__REPORT__

<!-- section: divergence_base -->
## Where the base simulator diverges from the data

No prior model exists yet, so every feature below is an unmodeled
mechanism: this is the map of what your `simulator.py` needs to cover,
each with its worst transition located in the data. Re-score any edit
with `sim.residuals()` (same report, current file):

__REPORT__

<!-- section: base_sim -->
The base simulator's own source code is available (read-only):

__REF_LISTING__

These files are byte-identical to the code your base-sim rollouts
execute: scene geometry and constants, body construction, stepping,
and state read/write. They deliberately omit the environment's hidden
domain-specific step, the residual dynamics you are here to model, and
its task generation and goal semantics. Use them to ground hypotheses
(masses, damping, substeps per action, how switches toggle) instead of
re-measuring those from data.

<!-- section: tools -->
## Available Tools

__TOOL_LISTING__

<!-- section: objective -->
## Task objective (env ground-truth reward)

__DESCRIPTION__

The trajectory roster above shows each interaction episode's
env-computed reward. In `run_python`, `evaluate_trajectory(states,
actions=None, task_idx=0)` scores any state sequence with the same
ground-truth evaluator: a collected trajectory's `states` and
`actions`, or a rollout of your simulator (where the verdict is only
as trustworthy as the simulator). It returns `{reward, solved}`;
`solved` means the episode is scored as a success, and a rollout can
reach the goal atoms and still be `solved=False`.

<!-- section: prior_state -->
Prior cycle state: __PRIOR_FILES__ already exist in the sandbox from a
previous learning cycle. Read them first: they are the previous cycle's
committed result and a reasonable starting point for incremental
refinement, though a fresh rewrite is fine if the prior approach looks
fundamentally wrong. Structural decisions are not binding across
cycles: re-read the decision record at the top of `simulator.py` and
re-decide the architecture itself (what the base sim carries and what
the rules model, which features the rules own, the latent structure,
whether disclosed base-sim parameters should be identified) rather
than only tuning what exists. In particular, if the trajectory roster
shows goal-reaching episodes scored `solved=0`, suspect a structural
modeling error (for example mis-calibrated base physics that the rules
only paper over near the fit data), not only parameter values. Earlier
versions are in `./simulator_versions/` and `./predicates_versions/`
(named `cycle_XXX_vers_YYY_*.py`); cross-reference the roster's
provenance tags against those files to see which rules and predicates
produced each failed plan.
