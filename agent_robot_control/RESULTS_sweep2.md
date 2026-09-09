# Sweep 2 results (2026-09-09)

Claude Code headless on `claude-opus-5`, 300 turns, $20 per run, 3 seeds per
cell, 100k-interaction cap, camera 335x180, no environment resets. Conditions
are cumulative: `no_particles` has move_to and wait; `move_to` adds
pixels_to_particles; `model_free` adds run_rl_on_particles. Domains all changed
from sweep 1 (three-leg plug, push-only discs, delayed pusher), so these
numbers are not comparable with `RESULTS_sweep1.md`.

Twenty-one further launches were destroyed by the account session limit and
re-run; one run spent the whole $20 cap and is reported separately rather than
counted as a failure. 26 valid runs, $201 of API spend.

| env | harness | condition | runs | success | median first success | mean interactions | mean tool calls | RL calls | mean wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| pybullet_airport | claude_code | model_free | 3 | 3/3 | 1122 | 1180 | 74.3 | 0 | 19 |
| pybullet_airport | claude_code | move_to | 3 | 2/3 | 866 | 2629 | 71.0 | 0 | 21 |
| pybullet_airport | claude_code | no_particles | 3 | 0/3 | nan | 3490 | 153.3 | 0 | 45 |
| pybullet_donut | claude_code | model_free | 3 | 3/3 | 694 | 2743 | 84.0 | 1 | 31 |
| pybullet_donut | claude_code | move_to | 3 | 2/3 | 8730 | 8017 | 84.3 | 0 | 29 |
| pybullet_donut | claude_code | no_particles | 2 | 1/2 | 724 | 1098 | 105.0 | 0 | 35 |
| pybullet_plug_outlet | claude_code | model_free | 3 | 3/3 | 66 | 101 | 28.3 | 0 | 7 |
| pybullet_plug_outlet | claude_code | move_to | 3 | 3/3 | 67 | 110 | 31.0 | 0 | 7 |
| pybullet_plug_outlet | claude_code | no_particles | 3 | 2/3 | 272 | 367 | 109.3 | 0 | 39 |

## By condition

| condition | solved | note |
|---|---|---|
| `no_particles` | 3/8 | move_to and wait only |
| `move_to` | 7/9 | adds the particle tool |
| `model_free` | 9/9 | adds the RL tool, called once in nine runs |

## Reading

- **The perception tool is worth more than the RL tool.** Removing
  `pixels_to_particles`, which is free and costs no interactions, drops success
  from 7/9 to 3/8. Those agents substitute probing motions for measurement:
  130 to 135 move_to calls against 11 to 30 for agents that could ask.
- **Perception buys speed as well as success.** Three legs at 4 mm clearance
  took 66 interactions with particles and 272 without (one of three seeds
  never got there).
- **`model_free` going 9/9 is not evidence for RL.** Only one genuine RL call
  happened in the sweep: donut seed 0 spent 5,201 interactions, 75% of that
  run's total, stagnated with zero reward successes, and the agent then solved
  the push by hand 183 interactions later. An unused tool costs nothing.
- **Airport 0/3 without particles** is the sharpest single cell: tracking a
  moving item and timing a delayed press needs the depth and labels the tool
  provides.
- **The delay change worked; the plug change did not.** The pusher lag moved
  the working press from 0.40 m to 0.60-0.70 m of lead and made the task
  solvable but unreliable (4/6 with particles). The three-leg plug, at the
  4 mm clearance its oracle gate forced, is the easiest task in either sweep:
  every run seated it in 64 to 70 interactions with no yaw command at all,
  because plug and outlet both start square. Randomising the plug's starting
  yaw is the fix.

## Caveats

The donut `move_to` median of 8,730 is one slow seed (17,318 interactions)
inside a cell of three. Every RL call in this sweep initially died on a bug in
the episode recorder (odd camera width, DEBUG_LOG 25); the two affected runs
were re-run with it fixed, and the six `model_free` runs that never called RL
are untouched by it.
