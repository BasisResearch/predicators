# Continual-protocol play round: query

Rendered by `play_prompts.build_play_query`. One message per round of
the run's conversation: the run's first message, a new level, a
continuation after the agent stopped, or a resume after a preemption.
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

## Skills

__SKILLS__

## Predicates

__PREDICATES__

## Types

__TYPES__

## Model and data

__MODEL__

## Your journal (`./journal.md`)

__JOURNAL__

## Attempts record (`./attempts.md`, written by the harness)

__ATTEMPTS__

## Instructions

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

## Instructions

__INSTRUCTIONS__

<!-- section: opening_first -->
This is the first round of the run. The environment is new: its
dynamics are hidden, and the recorded data is empty. Decide how to
spend the budget.

<!-- section: opening_level -->
Round __ROUND_NUMBER__ of the run: level __LEVEL_NUMBER__ begins, in
the same conversation. Your journal, your sandbox files and the
recorded data carry over; the environment is a new task.

<!-- section: opening_continue -->
Round __ROUND_NUMBER__ of the run: you stopped, and level
__LEVEL_NUMBER__ is not settled, so it continues from the observation
below. The level, the skills and the journal are as before.

<!-- section: opening_resumed -->
Round __ROUND_NUMBER__ of the run. Your previous turn on level
__LEVEL_NUMBER__ was interrupted by a compute preemption and this
conversation has been restored. The environment has been rebuilt at
the last recorded step; any tool call that had not returned did not
complete and its steps are not counted. Check the observation and
continue.

<!-- section: instructions -->
Decide what to do next and do it with the tools. Read the observation,
the ledger and the context line before you act. Test in the sandbox
what the sandbox can answer. When you act in the environment, annotate
the expected outcome. Write to `./journal.md` as you learn, not only at
the end: it is what survives a compaction. When the level is won, say
so and stop; when it is lost on a level with no resets, write your
notes and stop.

<!-- section: no_journal -->
(empty: no journal yet)

<!-- section: no_attempts -->
(empty: no round has acted in the environment yet)

<!-- section: learning_model_free -->
This arm has no belief model and no learning session. Recorded
episodes so far: __N_EPISODES__ (__N_STEPS__ steps), in
`./data/trajectories.pkl`.
