# Boil heating-aware forecasts

September 14, 2026.
This supplies the 224/40-action continuation and population adapters for the [heating-aware joint fits](boil-heating-joint-target.md).
The native continuation fixture, its independent reader and the preliminary population checks have passed.
The fitting and forecast pipelines remain development experiments, and the production estimator is unchanged.

## Continuation contract

Every forecast starts from the candidate initialization and replays all 224 fitting actions before continuing for 40 actions.
The candidate's physical parameters and initial scene remain fixed for that entire trajectory.
The simulator's inferred heating memory at action 224 is carried forward.
No independent thermal-prior draw or substitution of conditional parameter means is used.

During the fitting prefix, the physical replay conditions on the recorded robot-joint observations under the existing transition law.
After action 224, generation samples one variance per controlled joint from its conditional inverse-gamma distribution and uses it throughout the future trajectory.
This preserves the established shared-variance model.
Output-discrepancy moments are filtered through all 225 prefix observations and propagated into the future under the same law used for density evaluation.
The declared sensor noise is unchanged.

Generation has access to observations at steps 0 through 224 only.
Density evaluation receives the future observations separately and combines their conditional output density with the integrated robot-transition factors.
The future boundary is step 225, and variance conditioning includes all 224 fitting transitions.

## Native fixture

The four cases are source rows 0, 3, 6 and 9 of the verified thermal-guide preflight.
This fixed selection covers broad and local proposals for each of its two selected scenes.
These are mechanical cases, not independent fitted populations or agent seeds.

Native job `22759308` completed in 2:23 with 5,120 simulator actions.
Each case checks exact repeated generation, a generation-to-density round trip, agreement with the complete fitted-prefix score, and preservation of the 224-action conditioning boundary.
The fixture rejects a future observation with the wrong step index.
It also verifies that contradicting an exact burner observation produces zero future density.

Independent reader `22759309` checks every generated, density and contradiction artifact.
Its separate calculations include Student predictive factors, the closed-form shared-variance integral, conditional Gaussian draws, output moments and native task predicates.
It checks corruptions representing the old 132-action boundary, the old variance-conditioning count, a wrong future step and missing heating memory.
Two complete histories are replayed in fresh worlds.
The reader completed in 4:10, checking twelve artifacts, 28,512 joint-transition factors and all sixteen deliberately corrupted records.
Its fresh native work used 1,424 simulator actions.
The verified native-fixture checksum is `3e6bde4e6d5313889b5700b90e7e61a23ba6dd50ba3cc513dea2727ccf6d0bae`.
Source and script checksums and both frozen input manifests were checked after completion.

## Weighted population adapter

The adapter requires a completed fitting checkpoint and its successful numerical and native reader.
It reconstructs the completed sampler result without permitting a refit and verifies the arm, numerical seed, prior, data, runtime and configuration identities.
Every positive-weight sample contributes with its original weight.
Both arms retain all 84 physical joint coordinates; the guided arm's 85-coordinate proposal is decoded with the checked mixture mapping.
Its density correction is checked against the fitting score, but is not applied a second time to already-corrected posterior weights or to future likelihoods.

Each full forecast uses two banks of four generated futures per positive-weight particle, plus one recorded-future density evaluation per particle.
The short forecast fixtures use one generated future per bank.
The aggregate reports predictive means and variances, clean-state errors, goal and event probabilities, future mixture density and disagreement between the two sampling banks.
Zero-density samples retain their mass in the mixture denominator.

The independent reader checks every artifact, decodes the guided proposals with a separate normal-CDF calculation, reconstructs weighted central moments and densities, and repeats two full histories.
The preliminary metrics check uses deliberately nonuniform weights and includes missing and duplicate histories, incorrect weights, a wrong horizon, negative variance, invalid events, duplicate density rows and a nonfinite density.
It also checks uniform and guided decoding on all twelve original guide cases.
Job `22759907` completed these preliminary checks in fifteen seconds after the native reader succeeded.
It verifies 24 proposal mappings and rejects all eight malformed population summaries.
The nonuniform-weight reference and zero-density denominator checks pass.
Actual completed-checkpoint recovery and every population history still require the queued forecast fixtures and their readers.

## Queued pipelines

All jobs use `mit_preemptable` compute nodes.
The native adapter and its reader request one-hour allocations, which fit before scheduled maintenance.
The short fitting fixtures and their readers now have two-hour scheduler allocations, based on verified timing references; their scientific inputs and numerical caps are unchanged.
Three fixtures have started, while the fourth is waiting for resources.
The full fitting jobs retain their eight-hour allocation limits.
The forecast jobs below wait for the required fitting readers and preliminary checks; full forecasts also require their corresponding forecast fixture reader.
The same requirements are checked again inside each forecast process.

| Arm | Numerical seed | Forecast fixture | Fixture reader | Full forecast | Full reader |
|---|---:|---:|---:|---:|---:|
| Uniform | 410 | 22760108 | 22760109 | 22760110 | 22760111 |
| Uniform | 411 | 22760112 | 22760113 | 22760114 | 22760115 |
| Guided | 410 | 22760116 | 22760117 | 22760118 | 22760119 |
| Guided | 411 | 22760120 | 22760121 | 22760122 | 22760123 |

The assessment suffix is extracted from steps 225 through 264 of the previously inspected recording.
Its overlap with the fitting prefix is checked before extraction, and its clean event labels are checked against native predicates.
Clean observations are used for assessment only.
This development comparison cannot substitute for fresh untouched predictive validation or live-agent non-regression evidence.

Frozen artifacts are in `logs/uncertainty_boil_heating_forecast_adapter_20260914` and `logs/uncertainty_boil_heating_forecasts_20260914`.
Numerical replica agreement, budget stability and matched forecast comparisons remain open.
The [matched comparison collector and longer-prefix incumbent control](boil-heating-comparison.md) are now queued with their validation dependencies.
