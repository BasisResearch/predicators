# Original noisy balloons: latest subclass MB versus existing MF

The user authorized two new runs of the latest native-subclass MB agent on the original non-hatch balloons task.
The comparison reuses the two completed `balloons-agent_continual_model_free_cross_noise` runs; no new MF runs are needed.

## Configurable task distribution

The user confirmed that tasks may differ slightly if they come from the same distribution.
The pilot therefore uses normal task generation, with no recorded-state fixtures or custom experiment entrypoint.
`balloons_scene` selects `chute` or `hatch` geometry.
The new `balloons_task_generation` flag selects `original` or `validated` acceptance rules; `validated` remains the default.
The original setting is restricted to chute geometry and preserves the baseline sampler from `92d37ec9723b`.
Its historical simultaneous-release screening and probe controller are sampling criteria, not certificates of every public Release sequence.
The actual agent, public skills, physics, observation noise and evaluator retain the latest subclass runtime's implementations.
No baseline actions, outcomes or learned models are supplied to MB.

| Setting combination | Meaning |
|---|---|
| `balloons_scene=chute`, `balloons_task_generation=original` | Original non-hatch distribution used by the existing MF baseline |
| `balloons_scene=chute`, `balloons_task_generation=validated` | Revised non-hatch distribution with public-controller validation |
| `balloons_scene=hatch`, `balloons_task_generation=validated` | Current hatch v4 distribution |

This MB comparison uses the first combination.
The separate rendering refinement and aligned-row preview are not part of this experiment.

| Setting | Value |
|---|---|
| New MB seeds | 0, 1 |
| Levels per seed | 2 training, then 1 test |
| Position noise | 0.01 m |
| Orientation noise | 0.02 radians |
| Scalar noise | 0 |
| Observation access | Partially observable for both arms |
| MB features | Native simulator subclass, all uncertainty features and generic model repair |
| Real-step pool | 15,000 per seed |
| Continual active-time cap | 48 hours |
| Partition | `mit_preemptable` |
| Resources | 8 CPUs, 16 GB per seed |
| Allocation | 12 hours with automatic requeue and resume |

The runtime builds on the checked latest subclass/hatch implementation at `86391e7ac`, with the explicit original-distribution flag committed as `2db66d5a2` and the shared control interface updated before launch.
The checkout is `/home/ycliang/predicators-balloons-original-subclass-r1` on branch `balloons-original-distribution-r1`.
Config: `scripts/configs/predicatorv3/protocol_continual_balloons_original_subclass.yaml`.
Entrypoint: the standard `predicators/main.py`.

## Shared control interface

Both MB and MF implementations now expose current `joint_positions` and the action-space joint order in the observation's `[control]` JSON.
The values come from the current observation, including before the first action and immediately after reset.
Object observations remain noisy and partially observable.

`Wait(robot:robot)[1]` advances one environment step while holding the arm, with the existing finger stabilization behavior.
A positive integer count stops at the requested number of actions, an annotated subgoal, or the execution cap, whichever comes first.
Unrelated atom changes and quiescence do not truncate an explicitly timed wait.
An annotated subgoal takes precedence over quiescence for default-length waits.
`Wait(robot:robot)[]`, `[0]`, and omitted brackets retain default stopping behavior.
The same timing applies to simulator rollouts, including stationary scenes.
Reaching the cap ends a Wait normally so the next skill can execute; missing annotated subgoals are still reported as divergence.

The reused MF scorecards predate this shared interface update.
This is a comparison against the historical MF baseline, not an isolated ablation of simulator representation or uncertainty features.

## Validation and launch

Setup job `22410363` generated six fresh levels through normal environment setup and passed checks of their task version, sizes, goals and fresh resets.
For seeds 0 and 1, the resulting layouts and goal descriptions also match the existing MF baseline without using recorded-state fixtures.
Job `22410361` passed 17 focused tests covering the new selector, existing probe validation, hatch behavior and native subclass replay.
Its initial lint findings were corrected in the final check pass.
Job `22410638` passed the distribution implementation's final formatting, lint and whole-repository type checking.
The missing joint observation was reproduced through both real continual tool surfaces before the fix.
Job `22412058` then reproduced the cap-edge failure through real noisy PyBullet sessions: a bounded Wait was marked failed at the generic cap.
The fix lets capped waits terminate normally and preserves the following skill.

Wait validation passed 103 functional checks: 23 focused Wait/control tests and 55 continual/option/struct/prompt tests in `22412125`, plus 25 balloons/parser/sketch integration tests in `22412176`.
Initial test-helper typing errors were corrected.
Final job `22412264` passed both real MB/MF tool tests, lint, and whole-repository mypy over 889 files.
Changed-file formatting and lint passed; full-repository functional pytest was not run, and no PR or merge was made.

The frozen runtime is commit `41e4334270a23c9d63306c51f35c417537f8dc45`.
MB array `22412376` was submitted at 21:14 UTC on September 9, 2026, with seeds 0 and 1, normal MB priority, and automatic requeue/resume.
Both tasks were verified RUNNING at 21:15 UTC on `mit_preemptable`, with 8 CPUs and 16 GB per seed.
The existing result watcher was extended and restarted with its notification history preserved; both new task IDs appear in its status snapshot.
The separate hatch MB/MF pilot remains on its existing frozen runtime.
Both original-task MB jobs subsequently completed with Slurm state `COMPLETED`, exit code `0:0`, and final `all_levels_won` scorecards, verified September 10, 2026.

The [launch manifest](../../logs/balloons_original_subclass_20260909/launch-manifest.json) records the frozen source, exact flags, validation jobs and job IDs.
The [distribution preflight report](../../logs/balloons_original_subclass_20260909/distribution-preflight.json) records the six freshly sampled levels.
Run outputs use the standard continual viewer layout under `logs/agent_continual/balloons-agent_continual_original_subclass_r1/seed<N>/run_<stamp>/`.

## Results

| Arm | Seed | Levels won | Steps | Resets | Status |
|---|---:|---:|---:|---:|---|
| Existing MF | 0 | 2/3 | 513 | 0 | Agent ended |
| Existing MF | 1 | 3/3 | 461 | 1 | All levels won |
| Latest subclass MB | 0 | 3/3 | 466 | 0 | All levels won |
| Latest subclass MB | 1 | 3/3 | 338 | 1 | All levels won |

MF seed 0: [scorecard](../../logs/agent_continual_model_free/balloons-agent_continual_model_free_cross_noise/seed0/run_20260909_075624/scorecard.json).
MF seed 1: [scorecard](../../logs/agent_continual_model_free/balloons-agent_continual_model_free_cross_noise/seed1/run_20260909_075654/scorecard.json).
MB seed 0: [scorecard](../../logs/agent_continual/balloons-agent_continual_original_subclass_r1/seed0/run_20260909_171523/scorecard.json).
MB seed 1: [scorecard](../../logs/agent_continual/balloons-agent_continual_original_subclass_r1/seed1/run_20260909_171520/scorecard.json).

| Arm | Levels won | Whole-run success | Mean steps, successful runs only | Qualifying seeds | Mean resets, all runs |
|---|---:|---:|---:|---|---:|
| Latest subclass MB | 6/6 (100%) | 2/2 (100%) | 402 | 0, 1 (n=2) | 0.5 |
| Historical MF | 5/6 (83.3%) | 1/2 (50%) | 461 | 1 (n=1) | 0.5 |

MB seed 0 won its two training levels in 235 and 48 steps, then the test level in 183 steps.
MB seed 1 won its two training levels in 81 and 209 steps, then the test level in 48 steps.
The reset occurred during seed 1's second training level.
The successful-only step means use different qualifying seed sets because MF seed 0 did not finish all levels.
The comparison is now complete for these two development seeds; it does not isolate which MB changes caused the observed outcomes.

Keep this comparison separate from the earlier repair-agent runs and all hatch agents or oracles.
These two seeds have already informed development, so this is a paired development check rather than an independent held-out estimate.
Report per-seed wins, steps and resets; average steps only over whole-run successful seeds and include the qualifying count.
Infrastructure or setup failures do not count as agent failures, and unfinished comparisons do not establish an improvement.
