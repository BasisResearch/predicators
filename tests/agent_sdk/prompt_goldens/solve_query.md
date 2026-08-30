Solve the task below: produce a plan that reaches its goal.

## Goal Description

Put the thing on the fixture and switch it on.

## Scoring (env ground-truth reward)

reward = (1.0 if success else 0.0) - 0.1 x moves used

Decode every reward you observe with this rule before hypothesizing any other mechanism; there are no hidden reward terms.

## Goal Atoms

Active(fixture0:fixture)
At(thing0:thing, fixture0:fixture)

## Initial State Atoms

(none: no atom of the available predicates holds initially)

## Initial State Features

  {'fixture0:fixture': {'x': 1.00, 'y': 0.00, 'is_on': 0.00},
   'thing0:thing': {'x': 0.00, 'y': 0.00}}

## Initial State Image

A rendering of the initial scene is at `./test_images/task000_initial_state.png`. Read it first.

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
  - Read
  - Write

## Domain Strategy (advisory, written during learning)

## Approach
- move, then activate

## Attempt Log (recorded by the harness)

### task 0 attempt 1/3
- outcome: no capture

## Solve Journal (./journal.md)

### task 0 attempt 1
- MoveTo dx=0.2 reached At

## Instructions

Inspect the environment with your tools, then produce the plan and deliver it through the capture gate as the system prompt's Deliverable section specifies. When a step does not reach its subgoal, tune that step's parameters from the rendered image and the object poses (working principles 4 and 5), then re-test it.
