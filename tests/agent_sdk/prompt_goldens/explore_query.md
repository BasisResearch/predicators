Design this episode's experiment for the task below. Reaching the goal is the most informative experiment available, so treat solving the task as part of information gathering.

## Goal Description

Put the thing on the fixture and switch it on.

## Goal Atoms

Active(fixture0:fixture)
At(thing0:thing, fixture0:fixture)

## Experiment Guidance

The learning phase left this ranked ledger of open questions:
1. Does Active require At? Experiment: MoveTo then Wait.

## Initial State Atoms

(none: no atom of the available predicates holds initially)

## Initial State Features

  {'fixture0:fixture': {'x': 1.00, 'y': 0.00, 'is_on': 0.00},
   'thing0:thing': {'x': 0.00, 'y': 0.00}}

## Objects

  fixture0: fixture
  thing0: thing

## Available Options

  MoveTo(thing, fixture)  [params: dx, dy, range [-0.5, -0.5] to [0.5, 0.5]]
  Wait()

## Available Predicates (for subgoal annotations)

  Active(fixture)
  At(thing, fixture): thing rests on fixture

## Available Tools

  - submit_plan
  - run_python

## Plans Already Scheduled This Cycle

The plan(s) below are already queued to run on this task before any learning happens, so their data will be collected regardless of what you propose now.

Plan 1:
  0: MoveTo(thing0, fixture0)[0.2000, 0.0000]
  NOTE: belief-certified; executes verbatim as a solve attempt.

Propose a plan whose data is complementary rather than redundant: cover the open questions, mechanisms, or parameter regions the plan(s) above leave unmeasured. A goal-reaching plan is still preferred when it can carry that coverage; when it cannot, a designed experiment that settles what the scheduled plans will not is the better use of this episode. Only when the model is believed correct everywhere and no meaningfully different goal-reaching plan exists, repeat the best plan.

If a scheduled plan is marked belief-certified, this episode is the second test of the belief model, and one success of one plan is weak evidence. In order of preference: (1) a STRUCTURALLY different goal-reaching plan (a different option sequence, order, grasp, or contact arrangement) validated through the same `submit_plan` gate; (2) when no structurally different plan exists for this goal, the same structure with materially different parameters (a different placement pose, offset, or timing, not a jitter), validated the same way; (3) only as a last resort, the certified plan resubmitted unchanged. State which of the three you chose and why. A certified plan that then fails for real is the most informative outcome this episode can produce, not a loss.

## Instructions

Inspect the environment with your tools, design the episode as the system prompt's Exploration Setting specifies, and output the plan lines as your final text.
