# MB model repair and balloons validation

The user approved spending more simulator computation to improve real-environment sample efficiency on 2026-09-09.
The agent improvement and the balloons task correction are separate changes and separate comparisons.

## Agent and harness

A rejected canonical fit remains `UNVALIDATED` in subsequent probes and after checkpoint restoration.
A fit that excluded recordings reports `PARTIAL FIT` and the accepted/total motion-segment count.
If a rejected refit retains an earlier valid fit of the same file, the status describes both the rejection and the retained parameters.
These labels never gate real actions.

`sim.validate(traj_idxs=None, params=None)` replays available recorded actions at the current candidate's deployed rule and physical parameters.
It uses full recordings without fit-side segment rejection, reports each recording's normalized error, and withholds the aggregate if a replay fails.
Candidate comparisons use the same recording pool, motion scope, normalization and settings.
Explicit parameter overrides are diagnostic and are never deployed.
This supports checking estimates from exploratory training-subset fits against other available training recordings.
A recording counts as held out only when it was excluded from both fitting and model design.
Replay reconstructs initial state and starts velocities at rest, so reconstruction error remains a limitation.
The report includes rollout count and elapsed simulator time.

`agent_model_repair=True` enables an advisory workflow in the continual MB prompt.
Its default is `False`, and changing it leaves the MF prompt unchanged.
The agent compares alternative model structures, explicitly fits candidates, checks their recorded-action predictions, and compares plans across data-consistent models.
It chooses real probes only when their predicted observable outcomes distinguish hypotheses that recommend different actions.
Initial data collection and useful actions with imperfect models remain possible.
Repeated work without new evidence or a new hypothesis is discouraged.
The fit rejection advice in this mode does not infer chaotic data solely from a large residual.
No domain-specific lift law, task answer, future observation, or test-outcome lookup is added to the agent.

## Validation evidence

The original recorded-action reproduction completed in Slurm job 22371942 on `mit_preemptable`.
The red-after-green recovery and the two winning choices on the saved balloons task were reproduced before runtime changes.

Job 22372358 passed 51 targeted tests, full mypy on 880 source files, and changed-file pylint and formatting checks.
Job 22372675 passed another 28 approach/checkpoint and trimming-advice tests, full mypy, and the changed test file's lint and formatting checks.
This is targeted validation, not a claim that full repository pytest or merge checks have passed.

Job 22372371 exercised the public fit and validation tools on the failed balloons agent's saved first model and training recording.
The fit still rejected its only segment at normalized RMS 0.154058, while downstream status correctly remained `UNVALIDATED`, with 0/1 accepted segments.
The new validation check retained all 82 recorded actions and reported RMS 0.325827 at the deployed values.
These two RMS values use different scopes and recording preparation and must not be compared as an improvement percentage.
The validation cost one simulator rollout, approximately 1.24 seconds, and no real steps.
Its report is `logs/mb_repair_20260909/workbench-replay/report.txt` in the main checkout.

## Experimental comparisons

First compare the repair flag on the original balloons generator and original noise settings, with all six uncertainty flags on in both MB arms.
Use both seeds 0 and 1, with fresh run directories and a frozen runtime.
The new MB control shares the status corrections and replay tool, so the paired difference tests the advisory repair workflow.
Treat these already investigated seeds as development validation, not an unbiased generalization estimate.

Then check MB with repair in noisy bridge, fan, domino, and boil using the previously evaluated settings.
Reuse existing noiseless references and MF results; do not launch new noiseless controls.
Report historical comparisons with their code versions.
If a regression appears, isolate it with a matched MB control before attributing it to repair.

Keep corrected balloons tasks in a separate dataset/version and evaluate the comparison arms on identical new tasks.
Do not pool their scores with the old generator's scores or select tasks by which arm wins.
The generator audit and agent results answer different questions.

Solve rate and resets average over every completed seed in an arm.
Steps average only over completed seeds that won every level, with the qualifying seed count shown.
Report simulator work separately and interpret steps alongside solve rate.

## Submitted original-task runs

The implementation is committed on `bridge-learning` as `61e44d4ce`.
The experiment checkout is frozen at the equivalent commit `df53ca75f` in `/home/ycliang/predicators-mb-repair-r1`.
Preflight job 22372885 verified the diagnostic successes, frozen source hashes, and unchanged balloons environment/base/skill files relative to the original sweep.

| Experiment | Seeds | Slurm array |
|---|---|---|
| Balloons, MB with repair | 0, 1 | 22372894 |
| Balloons, MB control | 0, 1 | 22372896 |
| Bridge, MB with repair | 0, 1 | 22372897 |
| Fan, MB with repair | 0, 1 | 22372898 |
| Domino, MB with repair | 0, 1 | 22372900 |
| Boil, MB with repair | 0, 1 | 22372901 |

All arrays use `mit_preemptable`, the existing account-selection launcher, and requeue/resume support.
They run in two-seed batches after the original sweep's balloons MF array 22319435, keeping at most two agent runs from this sequence active at once.
Every array independently requires the successful preflight gate.
No new noiseless controls or MF runs were submitted in this original-task phase.
The resolved configuration and source hashes are in `logs/mb_repair_20260909/experiments/launch-manifest.json` and `frozen-source.json`.

Collection job 22373247 will update [the performance table](model-repair-table.md), snapshot and TSV after all six arrays terminate.
Missing or unfinished seeds withhold the arm's averages and fail collection rather than changing its denominator.
This schedules report generation on disk, not a chat notification.
