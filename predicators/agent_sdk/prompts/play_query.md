# Continual-protocol play session: query

Composed by `play_prompts.build_play_query`. Sent once at the start of
every play session; the tools carry everything that changes within a
session.

<!-- section: skeleton -->
__OPENING__

## Level __LEVEL_NUMBER__ of __LEVELS_TOTAL__

Goal: __GOAL_NL__

Goal atoms: __GOAL_ATOMS__

## Ledger

__LEDGER__

## Current observation

__OBSERVATION__

## Skills

__SKILLS__

## Predicates

__PREDICATES__

## Types

__TYPES__

## Learning status

__LEARNING__

## Your journal (`./journal.md`)

__JOURNAL__

## Attempts record (`./attempts.md`, written by the harness)

__ATTEMPTS__

## Handoff from your previous session

__HANDOFF__

## Instructions

__INSTRUCTIONS__

<!-- section: opening_first -->
This is the first session of the run. The environment is new: its
dynamics are hidden, and the recorded data is empty. Decide how to
spend the budget.

<!-- section: opening_next -->
This is session __SESSION_NUMBER__ of the run. Continue from your
journal and the handoff.

<!-- section: opening_resumed -->
Your previous session was interrupted by a compute preemption and its
transcript has been restored. The environment has been rebuilt at the
last recorded step; any tool call that had not returned did not
complete and its steps are not counted. Check the observation and
continue.

<!-- section: instructions -->
Decide what to do next and do it with the tools. Read the observation
and the ledger before you act. Test in the sandbox what the sandbox
can answer. When you act in the environment, annotate the expected
outcome. Update `./journal.md` before you end the session, and end it
with `session_end` and a handoff note. If the level is already won,
write your notes and end the session.

<!-- section: no_journal -->
(empty: no journal yet)

<!-- section: no_attempts -->
(empty: no sessions have acted in the environment yet)

<!-- section: no_handoff -->
(none)

<!-- section: learning_none -->
No learning session has run yet: there is no belief model behind
`sim`, and `sim.run` / `sim.refine` are unavailable until one exists.
Recorded episodes so far: __N_EPISODES__ (__N_STEPS__ steps).

<!-- section: learning_model_free -->
This arm has no belief model and no learning session. Recorded
episodes so far: __N_EPISODES__ (__N_STEPS__ steps), in
`./data/trajectories.pkl`.

<!-- section: learning_some -->
Learning sessions so far: __N_LEARN__. Belief model version:
__SIM_VERSION__; predicates version: __PRED_VERSION__; fit status:
__FIT_STATUS__. Recorded episodes so far: __N_EPISODES__ (__N_STEPS__
steps), of which __N_NEW__ were recorded after the last learning
session.
