# Balloons transition and output composition

September 12, 2026.
This joins the previously tested [conditional velocity transition](transition-discrepancy.md) and [coupled robot-output model](orientation-discrepancy.md) for the original non-hatch Balloons development recording.
It is an explicit stochastic model extension beside the deterministic reference and the production agent.
It changes neither the sensor variances nor the exact tie, pop and clip observations.

## Accounting for each observation

After each native action, the existing transition diagnostic conditions Gaussian joint-position corrections on the nine exact current-joint readings.
It conditions the rest/Gaussian box-velocity correction on exact speed, retaining the radial density and sampling the conditional direction.
Both transition factors remain in the composed path factor.
The observation uses the native predicted cached robot pose before correction, as established by the cache-phase audit.

The remaining robot xyz readings receive an explicit AR(1) output discrepancy with persistence 0.9.
The robot Euler triple receives the previously tested quaternion-output discrepancy with scale 0.001.
All other continuous measurements retain their original sensor law.
The checked finger readout remains a deterministic reduction from its measured joint coordinate.
Every remaining exact event still has an indicator likelihood.

The speed field is partitioned into its conditional transition factor; it is not relabeled as an external input or given a tolerance-band likelihood.
The exact joint readbacks are verified, and their transition densities are retained once.
A blank initial observation starts the output-error process at zero error without scoring the analytically conditioned root observation again.
Its placeholder prediction is unobserved and supplies no data term.
The initial-root proposal and its normalizing factors remain separate from these conditional path scores.

For each 32-action path, the accounting covers 1,824 remaining measured scalars and 32 conditionally represented speed readings, totaling all 1,856 public readings.
Floating-point reconstruction bounds for speed remain diagnostics of the conditional sphere construction, not likelihood tolerances.

## Saved-path composition check

Job `22643404_0` completed 48 scoring cases using all twelve archived conditional paths.
It evaluated no robot-output discrepancy and xyz innovation scales 0.001, 0.005 and 0.02, retaining the three previously declared velocity scales and two direction draws for each root.
The zero setting preserves the earlier exact-output reference; the positive settings include the quaternion factor.

| Robot-output setting | Finite paths from root 0 | Finite paths from root 1 |
| --- | --- | --- |
| Original exact outputs | 0/6 | 0/6 |
| XYZ scale 0.001 and quaternion scale 0.001 | 6/6 | 0/6 |
| XYZ scale 0.005 and quaternion scale 0.001 | 6/6 | 0/6 |
| XYZ scale 0.02 and quaternion scale 0.001 | 6/6 | 0/6 |

Root 1 still predicts balloon 0 tied and clip 0 on one action early, at action 22.
The additional continuous discrepancy does not erase these two event disagreements.
The [report](../../logs/uncertainty_balloons_composed_output_20260912/pilot-22643404_0.json) retains every model identity, transition factor, remaining likelihood and field count.
These are conditional path-support results, not model evidence, calibrated uncertainty or posterior samples.

## Complete first training episode

Array `22643477` extends the same conditional transition protocol through all 235 actions of the first training reset episode.
It uses robot xyz innovation scale 0.001 and quaternion scale 0.001 while retaining all three velocity scales, both direction draws and both root-seed cases.
The initial-scene law and fixed learned program remain those of the earlier free-pose reference.
Future observations are used throughout this full-recording support diagnostic, so none of it is a held-out forecast.

Both array tasks completed successfully.
All twelve complete trajectories repeat exactly in fresh worlds.
All six paths from root seed 0 have finite complete conditional path factors; all six from root seed 1 retain the same two action-22 event disagreements.
Each path accounts for all 13,630 public measured scalars: 13,395 remaining measurements and 235 speed readings represented by the transition law.

The reports save each sampled physical root, joint state, body motion case and complete path rather than relying only on RNG seeds to reproduce a candidate.
They capture actual node1380 and node1381 runtimes and do not assume identical RNG-to-physical mappings across CPUs.
See the [root 0 report](../../logs/uncertainty_balloons_full_transition_20260912/pilot-22643477_0.json) and [root 1 report](../../logs/uncertainty_balloons_full_transition_20260912/pilot-22643477_1.json).

This establishes complete-recording support for the declared stochastic extension, not for the deterministic sensor-only model.
The path factors vary substantially across velocity scales and direction draws.
Simply normalizing these few selected paths would not establish an adequate posterior over parameters, initial states and intermediate velocity directions.
The next inference construction must include the original parameter prior, root conditioning factors, conditional direction proposal and every retained observation factor, followed by numerical and predictive checks.
Joint corrections can also introduce physical inconsistencies; the transition model's limitations and previously measured forecast errors remain relevant to acceptance.

## Fixed parameter-prior provenance

The latest saved program has ten parameters and several data-narrowed optimizer bounds.
The version history retains the original [0.05, 8] support for four lift coefficients, but narrows lift height from [0.9, 1.4] and three mass supports from [0.02, 0.4].
It also replaces native `air_drag` with an explicit force-based `drag_rate`, then adds a multiplicative `lift_scale` calibration coefficient.
These last two roles cannot silently inherit the old native-damping law or data-derived narrow bounds as an independent prior.

The next fixed-program experiment declares independent uniform laws on [0.05, 8] for the lift coefficients and [0.9, 1.4] for lift height, with log-uniform laws on [0.02, 0.4] for the three color-indexed mass coefficients.
It separately declares `drag_rate ~ Uniform(0, 40)` and `lift_scale ~ LogUniform(0.5, 2)`.
These are explicit engineering priors for development, not recovered historical probability distributions or a claim of prior specification before observing any data.
The optimizer's coordinate scale alone does not establish a probability law.
Program selection from training data and unchanged-role support provenance remain visible in the comparison.

The [prior audit manifest](../../logs/uncertainty_balloons_parameter_prior_20260912/plan.json) hashes all five saved declarations, the proposed laws and the fixed program.
Its compute probe checks whether all thirty lower/middle/upper overrides are actually returned unchanged by the program, including values outside its latest optimizer bounds.
Job `22644425_0` completed with all thirty probes passing and all other parameter values unchanged.
The [readback report](../../logs/uncertainty_balloons_parameter_prior_20260912/pilot-22644425_0.json) records the actual runtime and every requested value.
This interface check does not establish parameter identifiability or favorable physical trajectories across the prior.

With these ten parameters, the complete stochastic target has 522-553 active continuous dimensions across the sixteen initial motion cases.
This comprises ten parameters, the existing 42-73 initial-state coordinates and two conditional velocity-direction coordinates for each of the 235 positive observed speeds.
Conditioned joint-position corrections contribute transition densities but no remaining free position coordinates.
The Gaussian and quaternion output discrepancies are integrated in their observation factors rather than added as sampled coordinates.
The fixed-parameter paths from root seed 0 have a resting robot and three moving balloons; that root component contributes 60 initial coordinates, or 540 total when parameters and all directions are inferred.
This high-dimensional construction needs an explicit proposal and numerical validation; the few support paths do not justify choosing a particle budget by themselves.
