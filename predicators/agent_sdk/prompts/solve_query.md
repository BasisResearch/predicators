# Solve / explore query

Composed by `sketch_prompts.build_solve_prompt`. The query carries the
task (goal, scene, vocabulary), the run state (records, scheduled
plans, open questions), and this episode's instructions. Rules that
hold for every episode live in the system prompt (`solve_system.md`).

<!-- section: skeleton -->
__OPENING__

__GOAL_NL_SECTION__

__SCORING_SECTION__

__GOAL_ATOMS_SECTION__

__EXPERIMENT_SECTION__

## Initial State Atoms

__ATOMS__

## Initial State Features

__STATE__

__IMAGE_SECTION__

## Objects

__OBJECTS__

## Available Options

__OPTIONS__

## Available Predicates (for subgoal annotations)

__PREDICATES__

__TRAJECTORY_SUMMARY__

__TOOLS_SECTION__

__STRATEGY_SECTION__

__ATTEMPTS_SECTION__

__JOURNAL_SECTION__

__SCHEDULED_PLANS_SECTION__

## Instructions

__INSTRUCTIONS__

<!-- section: opening_solve -->
Solve the task below: produce a plan that reaches its goal.

<!-- section: opening_explore -->
Design this episode's experiment for the task below. Reaching the goal
is the most informative experiment available, so treat solving the task
as part of information gathering.

<!-- section: goal_nl -->
## Goal Description

__GOAL_NL__

<!-- section: scoring -->
## Scoring (env ground-truth reward)

__OBJECTIVE__

Decode every reward you observe with this rule before hypothesizing any
other mechanism; there are no hidden reward terms.

<!-- section: goal_atoms -->
## Goal Atoms

__GOAL_ATOMS__

<!-- section: experiment_guidance -->
## Experiment Guidance

__GUIDANCE__

<!-- section: no_atoms -->
(none: no atom of the available predicates holds initially)

<!-- section: tools -->
## Available Tools

__TOOL_LIST__

<!-- section: strategy -->
## Domain Strategy (advisory, written during learning)

__STRATEGY__

<!-- section: attempts -->
## Attempt Log (recorded by the harness)

__ATTEMPTS__

<!-- section: journal -->
## Solve Journal (./journal.md)

__JOURNAL__

<!-- section: no_journal -->
(no journal entries yet)

<!-- section: scheduled_plans -->
## Plans Already Scheduled This Cycle

The plan(s) below are already queued to run on this task before any
learning happens, so their data will be collected regardless of what
you propose now.

__PLANS__

Propose a plan whose data is complementary rather than redundant: cover
the open questions, mechanisms, or parameter regions the plan(s) above
leave unmeasured. A goal-reaching plan is still preferred when it can
carry that coverage; when it cannot, a designed experiment that settles
what the scheduled plans will not is the better use of this episode.
Only when the model is believed correct everywhere and no meaningfully
different goal-reaching plan exists, repeat the best
plan.__CERTIFIED_RULE__

<!-- section: certified_rule -->
If a scheduled plan is marked belief-certified, this episode is the
second test of the belief model, and one success of one plan is weak
evidence. In order of preference: (1) a STRUCTURALLY different
goal-reaching plan (a different option sequence, order, grasp, or
contact arrangement) validated through the same `submit_plan` gate; (2)
when no structurally different plan exists for this goal, the same
structure with materially different parameters (a different placement
pose, offset, or timing, not a jitter), validated the same way; (3)
only as a last resort, the certified plan resubmitted unchanged. State
which of the three you chose and why. A certified plan that then fails
for real is the most informative outcome this episode can produce, not
a loss.

<!-- section: instructions_capture -->
Inspect the environment with your tools, then produce the plan and
deliver it through the capture gate as the system prompt's Deliverable
section specifies. When a step does not reach its subgoal,
__STUCK_ADVICE__

<!-- section: instructions_explore -->
Inspect the environment with your tools, design the episode as the
system prompt's Exploration Setting specifies, and output the plan
lines as your final text.

<!-- section: stuck_tune -->
tune that step's parameters from the rendered image and the object
poses (working principles 4 and 5), then re-test it.

<!-- section: stuck_revise -->
revise the plan (different objects, a different ordering, an added
intermediate step, or a corrected annotation), then re-test it.
