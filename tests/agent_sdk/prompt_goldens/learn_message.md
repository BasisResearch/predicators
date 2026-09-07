Synthesize a residual dynamics simulator for this environment. There are 3 trajectories (42 step transitions) available: 1 oracle demonstration(s), which reached the goal by construction, and 2 interaction trajectory/ies collected during online learning, some of which may have failed to reach the goal.

[0] demo, task 0
[1] interaction, task 0

Each trajectory carries a `train_task_idx`. `is_goal_state(state, task_idx)` (equivalently `train_tasks[task_idx].goal_holds(state)`) checks a single state for the goal atoms. Reaching the goal atoms does not by itself mean an episode is solved; when a task objective is stated below, score full trajectories with `evaluate_trajectory`. Use `is_goal_state` to confirm which trajectories reached the goal atoms and to treat failed interaction trajectories as counterexamples: places where a predicate or rule said "this should work" and the environment disagreed.

## Task objective (env ground-truth reward)

reward = success - 0.1 x moves used

The trajectory roster above shows each interaction episode's env-computed reward. In `run_python`, `evaluate_trajectory(states, actions=None, task_idx=0)` scores any state sequence with the same ground-truth evaluator: a collected trajectory's `states` and `actions`, or a rollout of your simulator (where the verdict is only as trustworthy as the simulator). It returns `{reward, solved}`; `solved` means the episode is scored as a success, and a rollout can reach the goal atoms and still be `solved=False`.

Prior cycle state: `./simulator.py` and `./predicates.py` already exist in the sandbox from a previous learning cycle. Read them first: they are the previous cycle's committed result and a reasonable starting point for incremental refinement, though a fresh rewrite is fine if the prior approach looks fundamentally wrong. Structural decisions are not binding across cycles: re-read the decision record at the top of `simulator.py` and re-decide the architecture itself (what the base sim carries and what the rules model, which features the rules own, the latent structure, whether disclosed base-sim parameters should be identified) rather than only tuning what exists. In particular, if the trajectory roster shows goal-reaching episodes scored `solved=0`, suspect a structural modeling error (for example mis-calibrated base physics that the rules only paper over near the fit data), not only parameter values. Earlier versions are in `./simulator_versions/` and `./predicates_versions/` (named `cycle_XXX_vers_YYY_*.py`); cross-reference the roster's provenance tags against those files to see which rules and predicates produced each failed plan.

## Where the prior model diverges from the data

Computed just now from the prior cycle's `simulator.py` with its parameters refit to all trajectories above, so the remaining mismatches need structural fixes, not tuning. Re-score any edit with `sim.residuals()` (same report, current file):

fixture.is_on: 4 mismatches

Data-structure source code is at: ./reference/structs.py

The base simulator's own source code is available (read-only):

  - ./reference/base_sim/scene.py

These files are byte-identical to the code your base-sim rollouts execute: scene geometry and constants, body construction, stepping, and state read/write. They deliberately omit the environment's hidden domain-specific step, the residual dynamics you are here to model, and its task generation and goal semantics. Use them to ground hypotheses (masses, damping, substeps per action, how switches toggle) instead of re-measuring those from data.

A residual scan between the base simulator's prediction and the observed next state suggests that these features carry residual dynamics (a starting hint; it may include base-sim jitter, so refine it as you go):

{'fixture': ['is_on']}

## Available Predicates (for subgoal annotations)

  Holding(robot, thing)

Subgoal annotations in plans for `sim.refine` / `sim.run` must reference these predicate names with matching arity and types. Any threshold or condition you bake into a rule must be consistent with what the predicate's classifier checks, or refinement rejects parameter samples that look correct on paper.

## Object Types

thing: x, y
fixture: x, y, is_on

## Options

Plans (for `sim.refine` / `sim.run`) and rules must match these typed signatures and parameter boxes exactly:

MoveTo(thing, fixture)[dx, dy]

## Available Tools

  - run_python
  - Read
  - Write
  - Edit

## This session

Read the data-structures file first, then explore the trajectory data with `run_python`. Write your simulator to `./simulator.py`, defining `RESIDUAL_RULES`, `PARAM_SPECS`, and `RESIDUAL_FEATURES`, and iterate with `Edit` and re-scoring. Pass `task_idx` explicitly to `sim.reset`; `sim.task(task_idx)` prints a task digest. Finish with the deliverables listed in the system prompt: a final `sim.fit()`, the GO/NO-GO check, the decision record, `./open_questions.md`, and `./strategy.md`.

## Predicate Invention

Only the predicates under "Available Predicates" above exist; this approach stripped the environment's symbolic predicates down to that allowlist. Invent every other subgoal predicate in `./predicates.py` as `LEARNED_PREDICATES`, following the system prompt's "Predicate Invention" section.

Goal (natural language): switch the fixture on.

Workflow: edit `predicates.py`, call `sim.predicates()` in `run_python`, then run `sim.refine` / `sim.run` with sketches that reference your invented names. Any predicate a sketch references must exist in `predicates.py` first.

## Partial observability

Some causally important quantities may be absent from the observation entirely (under no name), possibly several, possibly none. Inspect the trajectories first to judge whether any hidden process is at work and which observable features are your window into it; then, if latents are needed, model them in `latent` with Pattern A or Pattern B (or a mix).
