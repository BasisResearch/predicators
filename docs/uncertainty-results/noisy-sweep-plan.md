# Three-seed noisy sweep across five domains

The user authorized a fresh MB/MF sweep on September 10, 2026, with three seeds per domain and MB prioritized.
The matrix contains 30 seed results: 28 new runs and the two completed original non-hatch balloons MB runs.
All new runs use seeds 0, 1, and 2, except balloons MB, which adds only seed 2.
All 15 MF runs are fresh, including balloons, so the shared control interface matches MB.

## Frozen agent and environments

The validated agent runtime is `41e4334270a23c9d63306c51f35c417537f8dc45`, the version used by the two reused balloons MB seeds.
The isolated checkout is `/home/ycliang/predicators-noisy-sweep-20260910`, branch `noisy-five-domain-20260910`, at config-only commit `b09217bb38f2c3082136994ae43fbef2eb590e82`.
Runtime source is identical to `41e433427`; the new commit adds only the sweep YAML.
This choice keeps all three balloons MB seeds on one agent version.
Later prompt streamlining and the uncommitted explicit-fit deployment change in the main checkout are outside this frozen sweep.
The current learned-model artifacts from completed runs are not supplied to new seeds.

MB uses the native simulator subclass, interval belief, noise-aware probing, fit-side noise filtering, carried posterior, fit evidence, execution belief, and generic model repair.
MF uses the plain code agent with all six uncertainty switches disabled and no simulator-planning interface.
Both arms use partial observations, declared observation noise, current joint observations, and the shared bounded `Wait` interface.

| Domain | Position sigma | Orientation sigma | Reading sigma | Train + test | Real-step pool per seed |
|---|---:|---:|---:|---:|---:|
| Bridge | 5 mm | 0.02 rad | 0 | 1 + 1 | 20,000 |
| Fan | 5 mm | 0.02 rad | 0 | 1 + 1 | 10,000 |
| Domino | 10 mm | 0.04 rad | 0 | 1 + 1 | 10,000 |
| Boil | 12.5 mm | 0.05 rad | 0.07 | 1 + 1 | 10,000 |
| Balloons | 10 mm | 0.02 rad | 0 | 2 + 1 | 15,000 |

Domino retains the high-friction task with turning test layouts.
Balloons uses `balloons_scene=chute` and `balloons_task_generation=original` with the original jam-decoy setting.
Hatch tasks, oracle validation, earlier repair comparisons, historical MF baselines, and noiseless controls remain separate.

## Scheduling and validation

All agent allocations use `mit_preemptable`, eight CPUs, 16 GB, and 12-hour allocations with automatic requeue and resume.
The continual active-time cap remains 48 hours per run.
Only one array is eligible at a time, with up to three seeds concurrently.
The order is balloons MB seed 2, then bridge, fan, domino, and boil MB, followed by the five MF arrays.
Dependencies enforce MB-first execution, and MF also receives `nice=1000` versus MB's `nice=0`.
Account selection uses `a,c`, with the existing limit-aware selection at startup; account `b` had an active limit marker during preflight.

| Order | Domain | Arm | New seeds | Slurm array |
|---:|---|---|---|---:|
| 1 | Balloons | MB | 2 | 22456622 |
| 2 | Bridge | MB | 0, 1, 2 | 22456623 |
| 3 | Fan | MB | 0, 1, 2 | 22456624 |
| 4 | Domino | MB | 0, 1, 2 | 22456625 |
| 5 | Boil | MB | 0, 1, 2 | 22456626 |
| 6 | Bridge | MF | 0, 1, 2 | 22456627 |
| 7 | Fan | MF | 0, 1, 2 | 22456628 |
| 8 | Domino | MF | 0, 1, 2 | 22456629 |
| 9 | Boil | MF | 0, 1, 2 | 22456631 |
| 10 | Balloons | MF | 0, 1, 2 | 22456632 |

Balloons MB seed 2 reached the normal continual play loop on compute node `node2904`, with frozen-source verification passing and a fresh 15,000-step scorecard.
All nine later arrays were verified pending on their recorded dependencies, and every MF array had lower priority than MB.

Compute job `22456127` passed all 10 config resolutions, exact balloons MB flag parity, reuse scorecard validation, and 36 focused functional tests.
The tests cover all five native simulator bases, model memory, real MB/MF joint and Wait tools, continual play, fit noise filtering, and observation belief.
Initial setup job `22456022` stopped at a path assertion because compute nodes resolve `/home` to the physical shared-filesystem path.
The preflight now compares resolved source paths; this was a diagnostic setup issue before tests or agents, not an agent failure.
The experiment source was unchanged.
Each job verifies the frozen source hash and commit before starting or resuming.
No full repository suite or PR validation was required for this config-only experiment checkout, and no PR was opened.

## Results and notifications

The [launch manifest](../../logs/noisy_sweep_20260910/launch-manifest.json) records exact flags, job IDs, dependencies, resources, source commits, and reused scorecard paths.
Operational scripts and preflight evidence are in `logs/noisy_sweep_20260910/`.
New scorecards use the standard continual viewer directories, with experiment suffix `noise_sweep_r1`.
The reused balloons MB seeds remain in their original experiment group, `balloons-agent_continual_original_subclass_r1`.

The [live results table](noisy-sweep-table.md) combines these explicitly identified sources.
Regenerate it with `python docs/uncertainty-results/make_noisy_sweep_table.py`.
The result watcher refreshes the table and reports newly finished results in the authorized conversation.
It does not launch, resume, cancel, or otherwise change experiments.

Whole-run success means winning every training and test level.
Level solve rate is reported separately.
Step averages include only whole-run successful seeds and always show the qualifying count.
Reset averages and solve rates use finished agent seeds, with their count shown; partial tables remain provisional.
Missing or unfinished scorecards and infrastructure failures are excluded from agent performance averages.
The two reused balloons MB seeds were used during development, so this three-seed sweep is a small replication study rather than an untouched statistical evaluation.
