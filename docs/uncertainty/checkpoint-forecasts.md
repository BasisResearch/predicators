# Forecasting complete offline checkpoints

September 13, 2026.
This follow-up evaluates the numerical exploration comparison and the [initial-state ablation](initial-state-ablation.md) on their common 97-action Domino suffix.
It is an offline diagnostic of unassessed populations, not posterior publication or an agent solve-rate experiment.

The worker snapshots the selected complete fit report and checksum-protected checkpoint into its own artifact directory.
It reconstructs the typed inference request and resumes the completed checkpoint with a target callback that must never be called.
The resulting complete population, weights, counters and diagnostics must exactly match the fit report.
A missing or incomplete source cannot produce forecasts, and a report/checkpoint mismatch fails validation.
The explicit zero refresh configuration field is the only compatibility adjustment for earlier reports that predate that field.

Each positive-weight row reconstructs its physical state from the saved proposal coordinates and verifies exact equality with the retained physical parameters and state coordinates.
This avoids inverting physical parameters back into unit coordinates, which could introduce a rounding error before replay.
Point-start rows combine their five saved parameter coordinates with the frozen constant state coordinates.
Zero-weight rows are recorded as such and contribute no forecast mass; positive-weight histories are never dropped because their future predictions are poor.

Every physical continuation replays all 161 recorded actions uninterrupted in its own fresh world.
The first positive-weight history is repeated exactly, and each row must reproduce its saved fitting-prefix factor and the declared output-model identity.
Joint and point-start targets check their different base-factor conventions explicitly.
Forecast means and variances use only the fitting-prefix observations to condition the scalar output-error process.
Future observations enter only after the physical continuations and prefix-conditioned moments have been constructed, for density and error evaluation.
Posterior weights remain the fitting weights; future likelihoods do not reweight the plotted forecasts.

The output preserves complete compressed histories, hashes, weighted Cartesian moments, native toppling curves and whole-future mixture scores.
Work is counted in native actions, including the repeated reference history.
The source program was historically synthesized from training experience, so this suffix is not established as unseen during program synthesis.
It remains a fixed-program, fixed-recorded-action prediction diagnostic.

Before applying the worker to pending fits, compute job `22673647_0` checks the entire saved population from the completed original numerical seed 100.
It must exactly reproduce the earlier scalar forecast's means, variances, toppling curves, errors and mixture score using the new checkpoint-driven, four-process path.
The validation and prospective source mapping are frozen in `logs/uncertainty_domino_checkpoint_forecasts_20260913`.
New-source forecast jobs will require both that validation and their own source fit to complete successfully.

## Completed validation and scheduled follow-ups

The first attempt, `22673647_0`, failed because its frozen observation module predated `log_future_likelihood`.
That setup failure and its allocation are retained; the failed worker did not report a complete native-action count.
The corrected bundle uses the same observation module as the previously validated scalar forecast and checks for the scoring method before dispatching histories.
The running fits were not changed.

Corrected job `22673693_0` completed successfully in an 83-second allocation, with 77.38 worker seconds and 10,465 native actions including the repeated first row.
The reconstructed checkpoint, all aggregate metrics and all 64 complete decoded history artifacts match the previous forecast exactly.
The conditional Cartesian RMSE is 0.01158500918 m and the whole-future mixture log score is 14105.63277779, reproducing the earlier diagnostic rather than providing a new estimator result.
Artifact hashes and decoded-history equality are independently verified in `validation-verification.json`.

| Forecast job | Source fit | Arm | Numerical seed |
| --- | --- | --- | --- |
| `22673718_1` | `22673150_0` | Joint state, mixed proposals | 100 |
| `22673719_2` | `22673150_1` | Joint state, mixed proposals | 101 |
| `22673720_3` | `22673150_2` | Joint state, local proposals | 100 |
| `22673721_4` | `22673150_3` | Joint state, local proposals | 101 |
| `22673722_5` | `22673316_0` | Point start, mixed proposals | 100 |
| `22673723_6` | `22673316_1` | Point start, mixed proposals | 101 |

Each submitted follow-up requires both successful validation and successful completion of its source fit.
The corrected source, source mapping and submission manifest are in `logs/uncertainty_domino_checkpoint_forecasts_v2_20260913`.
These compute dependencies generate diagnostics only; they do not restore the disabled MB/MF task monitor or send result notifications to other tasks.
