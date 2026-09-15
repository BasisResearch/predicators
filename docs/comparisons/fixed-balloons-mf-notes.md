# Fixed Balloons: MF results and seed-0 analysis

All three intended MF seeds solved every level with the sustained-hover goal.
The mean charged cost is 718 steps over three whole-run successful seeds, and mean resets is 1.33.
There is no matched MB rerun under this changed goal.
The current cohort table is [Bridge/Balloons integrity results](bridge-balloons-integrity-results.md).
Its frozen runtime is `cad1000f92c5e5e94714ae613426b89761823ee0`, with the original task distribution, 1 cm position noise, 0.02 rad orientation noise, and zero scalar-reading noise.
The goal requires 25 complete consecutive environment intervals in the band below 0.01 m/s.
Earlier instantaneous-goal Balloons and all hatch runs remain separate comparisons.

| Seed | Wins | Steps | Resets |
|---|---:|---:|---:|
| 0 | 3/3 | 771 | 1 |
| 1 | 3/3 | 554 | 1 |
| 2 | 3/3 | 829 | 2 |

All three scorecards record `all_levels_won`, a completion time, and the expected frozen runtime.
Their totals agree with their per-level records.
The detailed behavior analysis below concerns seed 0.

| Level | Role | Wins | Steps | Resets |
|---|---|---:|---:|---:|
| 1 | Training | 1/1 | 452 | 1 |
| 2 | Training | 1/1 | 217 | 0 |
| 3 | Test | 1/1 | 102 | 0 |

The [final scorecard](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_dwell_r1/seed0/run_20260912_180112/scorecard.json) records `all_levels_won` and a completion time.
Totals agree with the per-level records.

## How it solved the tasks

The direct coding agent measured resting heights and oscillation decay after real balloon releases, used one training reset, and retained the measurements in its journal.
It wrote an empirical equilibrium helper, `lift_model.py`, from those measurements.
The inspected helper contains scalar arithmetic and calibrated constants, without engine imports or calls.
Its fitted power-law relation is the agent's approximation, not a claim that it recovered the environment's exact dynamics.

On the test task, it recognized an oak box and used the gold balloon's measured height from the first training task.
Before the irreversible release, it spent 12 Wait steps averaging noisy resting-height observations.
It then released gold and won after another 49 Wait steps; the goal interrupted the Wait when the dwell completed.
This behavior is recorded in the [test-level agent log](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_dwell_r1/seed0/run_20260912_180112/agent/003_play_20260912_182006.md).
The helper's source is recorded in the [second training-level log](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_dwell_r1/seed0/run_20260912_180112/agent/002_play_20260912_181548.md).

The baseline has no supplied simulator, but its coding and journal capabilities still let it construct small empirical models from public observations.
Calling it MF does not imply that its reasoning must avoid physical hypotheses or numerical analysis.
The visible box-speed reading has zero scalar noise in this configuration, so using it to detect rest is an allowed observation.
These inspected actions explain the successful transfer; they do not constitute an exhaustive sandbox security audit or establish an MB/MF gap under the changed goal.
