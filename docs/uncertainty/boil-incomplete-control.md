# Boil: incomplete-model control

September 12, 2026.
This is a fixed-program diagnostic within the [simplification plan](simplification-proposal.md), not a new agent run or a completed physical-state posterior.

## What the archived program can predict

The selected first training level contains 264 actions and 265 public observation frames.
The fixed `cycle_000_vers_001_simulator.py` artifact has SHA256 `aeeb0f7cb16a5062e561ed5771f27a0b607c0bc1f1ebe435174a7054615e4ce5`.
It declares no parameters or model memory and overrides `_domain_specific_step` with an optional geometry dump followed by `return None`.
The diagnostic explicitly disables that dump with `BOIL_DUMP_GEOM=0`.

The source review establishes a specific limitation of uninterrupted replay with this program:

- Water volume is read from a liquid body's visual shape dimensions, or as zero when no liquid body exists.
  The omitted filling hook is what recreates that shape as water rises.
  Rigid-body motion alone does not change the dimensions.
- Bubbling is derived from the per-environment heat dictionary.
  The omitted heating hook is what updates heat during stepping.
  The observation-only initialization has no privileged heat and initializes it to zero.
- Spillage is read from a stored faucet attribute.
  Initialization sets its observable value to zero, and the omitted faucet hook is what increments it.

Consequently these three scalar predictions remain constant within a replay episode.
Their initialization may limit which constants are physically achievable, but allowing any real constant gives a more favorable fit than those restrictions can.
This review concerns the frozen model without an external executor, repeated state injection, program edits or resets inside the episode.
The relevant environment, base-simulator and observation-noise sources are unchanged between the comparison runtime `b09217bb3` and this review.
Their hashes are retained in the diagnostic report.

## Error that initial-state uncertainty cannot remove

All three readings use the declared additive, unclipped Gaussian sensor standard deviation of 0.07.
For each channel, minimizing squared error over all possible constants gives the observed mean.
The resulting scalar likelihood is an upper bound for every constant prediction under that sensor channel, including a mixture over uncertain constant initial values.
This bound concerns the scalar factor only, not the complete observation likelihood or model evidence.

| Reading | Best constant using all 265 frames | Minimum RMSE | Minimum RMSE / sensor sigma | Future RMSE using the first 65 frames' mean |
| --- | --- | --- | --- | --- |
| Spilled level | 0.00294 | 0.07084 | 1.01 | 0.07202 |
| Bubbling level | 0.17956 | 0.38548 | 5.51 | 0.48986 |
| Water volume | 0.63501 | 0.41490 | 5.93 | 0.86762 |

The future column fits a constant from the initial frame and first 64 actions, then evaluates the remaining 200 observations without refitting.
This is a causal split for this scalar calculation; it is not a claim that the suffix was unseen during historical program synthesis.
The all-frame optimum is deliberately optimistic and is not used to forecast the suffix.

The constant-mean Gaussian reference gives centered chi-squared statistics of 271.42, 8,036.15 and 9,309.70 respectively, with 264 degrees of freedom.
Spill variation is close to the declared noise scale, whereas filling and bubbling require dynamics absent from the frozen program.
The reference statistics are diagnostics, not posterior-predictive checks or new agent decision thresholds.
Gaussian readings retain positive likelihood even at large residuals, so this is predictive inadequacy, not the exact model inconsistency established for the [Bridge glue control](experiments-20260912.md#a-structural-exact-output-contradiction-in-the-frozen-bridge-model).

## Verification and next comparison

Compute job `22650461` completed successfully on `mit_preemptable`.
It verified the recording and program hashes, reconstructed the declared public noise, and checked the constant optimum against an independent linear least-squares solution and shifted-constant likelihoods.
A separate 10,000-dataset stationary Gaussian reference had mean centered chi-squared statistic 263.694 against expectation 264 and coverage 0.9906 at the nominal 0.99 reference quantile.
Those generated scalar datasets validate the diagnostic calculation; they are not Boil tasks or agent seeds.
The report is [job-22650461.json](../../logs/uncertainty_boil_scalar_control_20260912/job-22650461.json), with source, configuration and scripts beside it.

The original array `22650411` was cancelled while all four tasks were still pending.
Its worker incorrectly bypassed the public fit dispatch and called the optimizer with an empty parameter schema.
The frozen agent's `sim.fit()` explicitly returns without fitting or publication for these parameter-free subclasses.
This was an experiment-harness error; the production agent already handles the case correctly.

Replacement array `22651160` exercises the real `BeliefProbe.fit()` and frozen synthesis backend, verifies that return, and then saves uninterrupted legacy predictions from the public noisy initial frame at the program's declared dynamics.
It checks the native parameter registry and declared schema before interpreting the result.
The first-64-action and complete-recording arms only partition prediction diagnostics; neither estimates parameters.
The tasks use one CPU each because the optimizer and its parallel validation workers are never invoked.
The frozen runtime, public observations, program and CPU model remain the same.
Artifacts are in `logs/uncertainty_incomplete_legacy_v2_20260912`; the original bundle remains preserved.
All four replacement tasks completed with exit code zero: each Boil task replayed 264 actions and each Bridge task replayed 1,186 actions.
Both domain pairs produced exactly identical complete predictions, despite their different diagnostic prefix lengths, and reported zero fitting likelihood evaluations.
The [verification report](../../logs/uncertainty_incomplete_legacy_v2_20260912/verification.json) records the public fit response, empty native registry, paired prediction equality and feature errors.

| Boil scalar | Legacy suffix RMSE, 200 actions |
| --- | ---: |
| Spilled level | 0.06909 |
| Bubbling level | 0.48782 |
| Water volume | 0.87211 |

These are actual legacy predictions initialized from the public noisy frame, distinct from the optimistic constant-model calculation above.
The Bridge suffix retains mismatches in four exact glue attributes: `span0.glue_end_a`, `span0.glue_end_b`, `span1.glue_end_b`, and `span2.glue_end_a`.
Neither result is an agent solve-rate trial or an assessed replacement posterior.

These controls preserve failed predictions in the comparison instead of attempting to repair missing dynamics with wider initial-state uncertainty or larger sensor variance.
They do not supply an adequate posterior, finish the five-domain comparison, or authorize live use of the replacement estimator.
