# Sweep 1 results (2026-09-07/08)

Claude Code (`claude-sonnet-5`) headless, 3 seeds per cell, 100k-interaction
cap, 120-turn cap, `model_based` not run (backend is a stub). Success is the
env's goal predicate, scored at the first interaction it holds. Eight runs
that the Claude account's usage cap cut short were discarded and re-run
(DEBUG_LOG 19); the Airport cells used the `wait` tool (added after the first
Airport attempts, DEBUG_LOG 15). Curves:
`~/arc_outputs/analysis/success_vs_interactions.png`.

| env | harness | condition | runs | success | median first success | mean interactions | mean tool calls | RL calls | mean wall (min) |
|---|---|---|---|---|---|---|---|---|---|
| pybullet_airport | claude_code | model_free | 3 | 1/3 | 1129 | 4337 | 69.0 | 3 | 28 |
| pybullet_airport | claude_code | move_to | 3 | 2/3 | 2073 | 2120 | 80.0 | 0 | 17 |
| pybullet_donut | claude_code | model_free | 3 | 3/3 | 103 | 135 | 12.7 | 0 | 1 |
| pybullet_donut | claude_code | move_to | 3 | 3/3 | 110 | 132 | 14.3 | 0 | 1 |
| pybullet_plug_outlethard | claude_code | model_free | 3 | 3/3 | 2660 | 13555 | 69.7 | 11 | 59 |
| pybullet_plug_outlethard | claude_code | move_to | 3 | 2/3 | 124 | 146 | 62.0 | 0 | 12 |
| pybullet_plug_outletmedium | claude_code | model_free | 3 | 3/3 | 1651 | 11047 | 49.0 | 7 | 48 |
| pybullet_plug_outletmedium | claude_code | move_to | 3 | 1/3 | 388 | 2645 | 107.3 | 0 | 21 |

## Reading

- **Donut**: coarse control suffices (6/6 at about 100 to 150 interactions);
  the RL tool is never invoked. A pure baseline task.
- **Plug-outlet medium (2 mm)**: `move_to` alone 1/3; with the RL tool 3/3,
  first success at 1,265 to 2,612 interactions. In all three RL successes the
  env goal fired *during* RL exploration while the agent's own particle
  reward never registered success (the inserted prong is occluded, DEBUG_LOG
  16), so RL ran on to its budget and the agent did not know it had won.
- **Plug-outlet hard (1 mm)**: `move_to` alone 2/3 at 87 and 162
  interactions (the agent centres the plug better than the scripted oracle by
  grasping across its narrow side); with the RL tool 3/3, first success at
  1,574 / 2,660 / 11,422. Seed 0 is the cleanest RL result: three RL calls
  whose agent-written reward registered 16, 6 and 7 successful episodes, and
  the env goal held 43 interactions after the last call returned.
- **Airport (timing task)**: `move_to` + `wait` 2/3 (1,128 and 3,018
  interactions); with the RL tool 1/3, and the two failures used RL calls
  that stagnated. The button route needs a press timed to a 5 cm window of
  belt travel (DEBUG_LOG 13, 18); agents that read the belt speed from two
  particle snapshots got it, others were consistently late.
- **Cost of RL on the interaction axis**: the RL condition raised success
  from 4/9 to 7/9 on the two precision tiers (medium + hard), but its mean
  interaction count is 11k to 14k versus 150 to 2,600 for coarse control, and
  the success-versus-interactions curves cross only after about 1,000
  interactions. On Airport the RL tool hurt (1/3 vs 2/3): agents reached for
  it when the real problem was timing.

## Hand-written-reward pilots (no harness)

Plug insertion from an anchor 5 cm above the socket with a 1 cm offset, 20k
budget: SAC and PPO both stagnated (mean return improving slowly, best
per-step reward flat at about -0.13), no insertion. Donut push pilots
stagnated after the donut left the workspace box. The agent-written rewards
in the sweep did better than these, mostly by choosing tighter anchors and
smaller boxes.

## Caveats

Single fixed camera (occlusion), object names from segmentation, perfect
depth, 3 seeds, one harness and one model. `model_based` condition pending.
