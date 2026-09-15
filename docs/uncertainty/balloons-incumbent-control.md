# Balloons: incumbent fitter control on the reserved suffix

September 13, 2026.
This supplies the existing-fitter comparison for the [prefix posterior forecasts](balloons-prefix-forecasts.md).
The experiment is frozen under `logs/uncertainty_balloons_incumbent_prefix_20260913/`.
It does not change the production agent.

## Shared inputs and retained differences

Both arms receive the same first 64 actions and 65 observations from the original noisy Balloons development recording, with 171 actions reserved for future prediction.
Both use the same selected simulator subclass, its ten declared physical parameters and the identified historical runtime.
The program was previously selected using training data, so the comparison is conditional on that fixed development program, not an independent program-synthesis evaluation.
No local parameter-override file is present in either isolated runtime.

The incumbent uses its full existing fitting pipeline, with interval belief, fit-side noise handling and fit evidence enabled.
Its optimizer, trimming, parameter-selection rules and observed-state initialization remain intact.
The posterior arm uses the separately declared joint initial-state and transition/output model.
This is a comparison of the complete configurations, not an estimator-only ablation.

The incumbent preparation produces three overlapping action windows, using zero-based half-open indices:

| Action window | Actions | Initial observation treatment |
| --- | --- | --- |
| `[0, 21)` | 21 | Original initial frame. |
| `[15, 64)` | 49 | Average noisy features over observation frames 8 through 15. |
| `[63, 64)` | 1 | Average noisy features over observation frames 56 through 63. |

The eight-frame averages are arithmetic for ordinary noisy features and circular for angular features.
Exact features retain their original values.
The driver verifies every following frame and action against its original recording window, and independently reconstructs each averaged initial feature from the identified source observations.
The posterior likelihood uses the complete 64-action history once; the incumbent's overlapping windows and averaging are preserved for comparison.

## Native validation and current execution

Preflight `22693164` completed in 30 seconds, executing 1,410 native actions.
It compares three complete 235-action trajectories at the declared default, low-gold-lift and high-gold-lift parameter settings.
The normal rollout helper and an independent manual reset/pin/zero-velocity/step lifecycle agree exactly in every case, and the lift changes alter predictions.
Future observations are not used in these trajectories.

The initial fitting attempt `22693556` stopped before fitting because the new comparison driver incorrectly required averaged segment starts to equal raw observations.
Its dependent verifier and comparison jobs were cancelled by the scheduler.
This was a comparison-harness failure, not an agent outcome.
The corrected driver verifies raw successor frames and actions separately from the intended rest-window averages.
Those checks pass, and fitting job `22693806` completed in 39:11 of allocation time.
The driver recorded 157,664 native actions, including fitting and repeated predictions.
All three prepared segments survived.
The full incumbent's publication rules retained the original applied parameters: several were insensitive or anchored, and the fitted drag value at its upper bound was not applied.
This is the verified outcome of the complete incumbent fitter, not a shortcut that skipped fitting.

| Job | Purpose |
| --- | --- |
| `22693806` | Full incumbent fit, followed by two repeated causal forecasts. |
| `22693822` | After fitting: independently reproduce the selected forecast and score the reserved 171 actions. |
| `22693842` | After incumbent verification and the posterior pair: compare the verified predictions. |

The future metrics match the posterior reports: thirteen position/speed features and fourteen event indicators.
The control reports selected-point predictions, not the incumbent's complete interval or planning ensemble.
Final comparisons must retain numerical-stability warnings and cannot be treated as agent solve-rate or sample-efficiency results.

Independent verifier `22693822` completed in 17 allocation seconds and reproduced every frame of the complete 235-action selected forecast using the manual native lifecycle.
The selected-point forecast reproduces all reserved tie, burst and clip indicators, but misses the final `InBand`/evaluator-win event.
Each of those two indicators has one error across the 171 reserved frames; the rest indicator differs on 67 frames.
These are prediction errors on recorded actions, not a failed agent seed.
Both posterior fits have completed and their population forecasts are now running; comparison `22693842` remains dependent on their independently verified paired report.
