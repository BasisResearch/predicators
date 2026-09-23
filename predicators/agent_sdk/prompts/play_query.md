# Continual-protocol play round: query

Rendered by `play_prompts.build_play_query`.
One message per round of the run's conversation: the run's first message, a new level, a continuation after the agent stopped, or a resume after a preemption.
The system prompt is `play_system.md`.

<!-- section: skeleton -->
__OPENING__

## Level __LEVEL_NUMBER__ of __LEVELS_TOTAL__

Goal: __GOAL_NL__

Goal atoms: __GOAL_ATOMS__

## Ledger

__LEDGER__

__CONTEXT__

## Current observation

__OBSERVATION__

## Model and data

__MODEL__

## Your journal (`./journal.md`)

__JOURNAL__

## Attempts record (`./attempts.md`, written by the harness)

__ATTEMPTS__

## Available vocabulary

### Skills

__SKILLS__

### Predicates

__PREDICATES__

### Types

__TYPES__

## Next action

__INSTRUCTIONS__

<!-- section: skeleton_continue -->
__OPENING__

## Ledger

__LEDGER__

__CONTEXT__

## Current observation

__OBSERVATION__

## Model and data

__MODEL__

## Next action

__INSTRUCTIONS__

<!-- section: opening_first -->
This is the first conversation round of the run.
Use the current task, observation, and available records below to decide what to do next.

<!-- section: opening_level -->
Round __ROUND_NUMBER__ of the run: level __LEVEL_NUMBER__ begins, in the same conversation.
Your journal, your sandbox files and the recorded data carry over; the environment is a new task.

<!-- section: opening_continue -->
Round __ROUND_NUMBER__ of the run: you stopped, and level __LEVEL_NUMBER__ is not settled, so it continues from the observation below.
The level, the skills and the journal are as before.

<!-- section: opening_resumed -->
Round __ROUND_NUMBER__ of the run: level __LEVEL_NUMBER__ resumes after compute preemption.
The conversation and recorded environment state have been restored.
Check the observation, ledger, and recorded actions before retrying an interrupted call; do not assume that it made no progress.

<!-- section: instructions -->
Choose the next action from this state and carry it out with the tools, following the decision workflow.

<!-- section: no_journal -->
(empty: no journal yet)

<!-- section: no_attempts -->
(empty: no round has acted in the environment yet)

<!-- section: learning_model_free -->
This arm has no belief model.
Recorded episodes so far: __N_EPISODES__ (__N_STEPS__ steps), in `./data/trajectories.pkl`.

<!-- section: no_model -->
No model yet: `sim` runs the real skill controllers on the visible base physics with hidden mechanisms disabled, so reach, grasp and collision checks already work.
Build `./simulator.py` (and `./predicates.py` if useful) in `run_python` and call `sim.fit()` before you act on a test level.
Recorded episodes so far: __N_EPISODES__ (__N_STEPS__ steps).

<!-- section: model_status -->
Current files: `simulator.py` __SIMULATOR_VERSION__, `predicates.py` __PREDICATES_VERSION__.
Last fit: __FIT_STATUS__.
Recorded episodes so far: __N_EPISODES__ (__N_STEPS__ steps); __NEW_EPISODES__ episode(s) since the last fit.__REFIT_NOTE__
