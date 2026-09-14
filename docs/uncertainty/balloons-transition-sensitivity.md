# Balloons: sensitivity to future dynamics noise

September 14, 2026.
This diagnostic follows the failed [prefix posterior forecasts](balloons-prefix-forecasts.md).
It tests conditional continuations, not a refitted probability model or agent performance.

## Controlled comparison

The current discrepancy model perturbs arm-joint positions with standard deviation 0.001 and samples a box-velocity transition after every native action.
That velocity transition has a 0.1 probability of setting all three linear velocity components to zero; otherwise it adds Gaussian noise with standard deviation 0.01 per component.
These physical interventions affect subsequent contacts and can therefore change irreversible balloon bursts.

We select the highest-weight particle from each completed prefix population, independently of future prediction error.
These are particle 48 of fit seed 620, with weight 0.05047, and particle 18 of fit seed 621, with weight 0.05974.
Each particle retains its complete 206-coordinate parameter, initial-state and conditional-direction vector.
Four archived future random seeds per particle are crossed with four interventions: both noise effects, joints only, velocity only, and neither.
All 32 histories reconstruct the same first 64 actions and then continue through the remaining 171 actions in one fresh world.
Suppressed interventions still consume their random draws, preserving the paired random schedules.
Future observations are absent from generation and enter only the subsequent assessment.

The frozen bundle is `logs/uncertainty_balloons_transition_sensitivity_20260914`.
Native job `22713764` completed in 2:32 on a compute node with the same CPU model as the archived runs.
It executed 9,400 native actions, including eight complete repeated histories.
All eight original-treatment paths reproduce their archived complete histories exactly, and all interventions preserve their original fitting-prefix trajectories and scores exactly.
Independent reader `22713765` completed in 12 seconds.
It checks all 5,472 future joint vectors and velocity draws, recomputes event labels and feature errors, and rejects four deliberately corrupted prefix, joint, velocity and event records.

## Results

Each row uses four simulated continuations from one selected fitted state.
Errors are averages of the four individual future-trajectory RMSEs against the reserved clean development trajectory.
These draws are not agent seeds, and their goal counts are not solve rates.

| Fitting seed | Active future noise | Box-height RMSE (m) | Box-speed RMSE (m/s) | Final burst count | Final predicted goal count |
|---|---|---:|---:|---:|---:|
| 620 | Joint and velocity | 0.18992 | 0.22886 | 0/4 | 0/4 |
| 620 | Joint only | 0.20728 | 0.26137 | 0/4 | 0/4 |
| 620 | Velocity only | 0.20678 | 0.25179 | 0/4 | 0/4 |
| 620 | Neither | 0.22072 | 0.23738 | 0/4 | 0/4 |
| 621 | Joint and velocity | 0.22680 | 0.65371 | 4/4 | 0/4 |
| 621 | Joint only | 0.10548 | 0.41266 | 1/4 | 0/4 |
| 621 | Velocity only | 0.36394 | 0.88007 | 4/4 | 0/4 |
| 621 | Neither | 0.11345 | 0.37631 | 0/4 | 0/4 |

At the selected seed-621 state, the velocity intervention contributes to spurious bursts and large motion errors.
Removing both interventions eliminates bursts in these four continuations, but still does not recover the goal.
At the selected seed-620 state, removing noise does not improve either reported error.
Thus future transition noise explains part of the regression, but does not explain all of the fitted-state and dynamics error.
There is no basis for promoting a simple noise-disable change to production.

The next model investigation should separate the velocity reset atom from continuous velocity noise and inspect errors already present in the conditioned prefix.
Any proposed replacement must use a coherent law during both fitting and forecasting, then be refitted and assessed across complete weighted populations.
These two selected states cannot establish posterior-wide improvement, numerical convergence, or the Stage B acceptance gate.

## Separating reset events from continuous velocity noise

The follow-up bundle `logs/uncertainty_balloons_velocity_components_20260914` crosses joint noise on/off with either the velocity reset branch or the Gaussian branch.
Suppressing a reset retains the native velocity on that branch; it does not replace the skipped reset with a new Gaussian draw.
Suppressing a Gaussian branch still draws its noise and then retains native velocity.
This preserves the original paired random schedules and isolates interventions; it does not define or fit a new posterior.

Native job `22714229` completed in 2:45 with 11,280 actions, including eight original-treatment baseline histories and eight complete repeats.
All original-treatment histories and all fitting prefixes reproduce exactly.
Reader `22714231` completed in 10 seconds, checking the 32 new histories, 5,472 future joint vectors and velocity draws, and four corruption controls.

| Fitting seed | Active future noise | Box-height RMSE (m) | Box-speed RMSE (m/s) | Final burst count | Final predicted goal count |
|---|---|---:|---:|---:|---:|
| 620 | Joint and Gaussian velocity | 0.20953 | 0.23888 | 0/4 | 0/4 |
| 620 | Gaussian velocity only | 0.19419 | 0.26363 | 0/4 | 0/4 |
| 620 | Joint and velocity reset | 0.40998 | 1.27533 | 2/4 | 0/4 |
| 620 | Velocity reset only | 0.23846 | 0.63505 | 1/4 | 0/4 |
| 621 | Joint and Gaussian velocity | 0.18292 | 0.34631 | 4/4 | 0/4 |
| 621 | Gaussian velocity only | 0.20205 | 0.35242 | 4/4 | 0/4 |
| 621 | Joint and velocity reset | 0.10594 | 0.42441 | 2/4 | 0/4 |
| 621 | Velocity reset only | 0.11305 | 0.38959 | 0/4 | 0/4 |

At the selected seed-621 state, continuous velocity noise alone suffices to cause the four observed forecast bursts, while resets alone do not.
At the selected seed-620 state, reset interventions can also cause bursts.
The effects are nonlinear and depend on the fitted state; the reset branch is not the sole cause.
Neither intervention recovers the goal.

The same reader finds substantial discrepancies during the conditioned fitting prefix.
The two selected histories require speed corrections with RMS 0.26270 and 0.06844 m/s, compared with the declared 0.01 m/s continuous velocity scale.
All 64 fitted speed observations in each history are positive, so none uses the zero-speed atom during conditioning.
The nominal discrepancy scale should not be interpreted as a bound on these conditioned corrections.
These errors motivate investigation of the sampled parameters and initial states in addition to future noise.

## Conditional parameter profiles

The frozen program's own decision record describes an earlier overdamped optimizer solution that matched mean heights but lost the observed oscillation.
The selected posterior states have drag values 23.20 and 21.20, compared with the program's default 1.979.
This motivates a diagnostic rather than a conclusion that changing drag alone will fix the problem.

Bundle `logs/uncertainty_balloons_parameter_profiles_20260914` evaluates eight predefined parameter profiles at each selected initial state and latent direction path.
Profiles include the selected parameters, all existing program defaults, selected single-parameter changes and three interpolations in the original prior's unit coordinates.
All 196 non-parameter coordinates remain fixed within each profile.
The parameters remain inside the same original prior bounds.
The program defaults already contain historical development-data choices and are not independent evidence or a newly asserted prior.

Native job `22714315` completed in 1:41 with 5,808 actions.
Every fitting prefix repeats exactly and matches its separately generated full-history prefix.
Independent reader `22714316` completed in 19 seconds, verifying all 1,024 radial speed factors, joint factors, output-score composition, parameter edits and future intervention semantics.
Future evaluation uses native continuation with both physical noise interventions disabled, separately from the unchanged conditioned-prefix score.

| Fitting seed | Parameter profile | Prefix log score | Prefix height RMSE against noisy observations (m) | Future height RMSE against clean observations (m) | Future speed RMSE (m/s) |
|---|---|---:|---:|---:|---:|
| 620 | Selected | -14,534.08 | 0.02552 | 0.22072 | 0.23738 |
| 620 | Program defaults | 4,920.01 | 0.01378 | 0.01279 | 0.05321 |
| 621 | Selected | 7,411.31 | 0.02291 | 0.11345 | 0.37631 |
| 621 | Program defaults | -1,477.20 | 0.02682 | 0.01383 | 0.05748 |

The prefix scores are conditional likelihoods at fixed initial states and latent paths, not marginal parameter evidence or posterior weights.
At the selected seed-620 state, defaults improve the score by about 19,454 and greatly improve the native future prediction.
This exposes a much better conditional fitting point than that particular retained sample.
At the seed-621 state, defaults improve future motion prediction but reduce the fitted output score enough to lower the total prefix score.
Other profiles have large discontinuities and can produce violent trajectories; the complete report retains those failures.
Changing drag alone barely changes seed 621's prefix score yet severely worsens its future prediction.
Thus both numerical exploration and the relationship between the conditional scoring model and future dynamics remain concerns.
None of these profiles predicts the final goal correctly.

The next proposal audit uses prefix observations before the first exact clip/tie change to estimate initial-location proposal centers, and tests those centers with existing program defaults and selected parameters.
It retains the original prior, feasibility rules and likelihood, including their treatment of motion.
The use of a quiet-prefix mean is a proposal heuristic, not an assumption that those measurements are independent initial-state observations in the likelihood.
The audit must verify normalized proposal densities and native target scores before any new fitting run uses it.

## Verified proposal audit and matched fits

The audit in `logs/uncertainty_balloons_prefix_guidance_20260914` uses observations 0 through 22, before the first exact clip/tie change at step 23, for its optional initial-location means.
It compares four proposal centers with the same twelve random inputs per guide, plus one evaluation of each center.
Native job `22714432` completed in 1:46 with 4,864 actions, including repeated evaluations of all 52 points.
Geometry or exact-output failures remain explicit unsupported candidates.

| Proposal center | Finite random draws | Best random-draw prefix log score | Center prefix log score |
|---|---:|---:|---:|
| Original guide | 3/12 | -1,252,294.47 | -268,373.62 |
| Original scene, program-default parameters | 8/12 | 6,559.01 | -651.49 |
| Prefix location means, program-default parameters | 9/12 | 4,516.61 | Unsupported |
| Prefix location means, selected seed-621 parameters | 5/12 | 5,960.21 | Unsupported |

These small counts describe proposal support and are not posterior-quality estimates.
The independent reader verifies all 52 mixture proposal densities and 38 evaluated conditional likelihoods, including the original correction for the full mixture density.
The first reader, `22714433`, encountered a 2.78e-17 inverse-CDF coordinate difference between AMD and Intel processors while checking exact proposal provenance.
Reader `22714522` passes the unchanged checks on the same Intel CPU model as generation.
This is a reader/runtime precision issue, not an agent or model failure.

The next comparison changes only the original guide's ten parameter-center coordinates to the frozen program defaults.
It retains the original scene center; the additional mean-location change is not included in these fits.
The original broad proposal component and normalized mixture correction remain intact, so this changes how the target is explored rather than changing the declared prior or likelihood.

The frozen fitting bundle is `logs/uncertainty_balloons_default_guided_fits_20260914`.
Native target validation `22714576` has passed: archived prefix histories reproduce, future-data corruption leaves the fitting data unchanged, 14 of 16 new proposal cases have finite support, and serial/parallel target evaluations agree exactly.
The maximum checked factorization error is 2.27e-13.
Array `22714577` runs numerical seeds 620 and 621 with the original 64 particles, 32 temperatures, eight moves, four workers and 16,448-evaluation budget.
Fit readers `22714628` and `22714629` will recover each complete checkpoint and freshly repeat every retained native target.

The two new fits remain separate from the earlier populations with the same numerical seed labels.
Complete weighted reserved-future forecasts and numerical-stability checks are still required before assessing the changed guide.
In particular, adequate inference must retain prior uncertainty for parameters that the recorded prefix does not inform; a better conditional fitting score alone is insufficient.
There is no new agent-performance result or production-estimator change.


## Weighted comparison for the new guide

Bundle `logs/uncertainty_balloons_default_guided_forecasts_20260914` binds the forecast adapter to the new guide and fitting identities.
It preserves every retained positive-weight particle and its full joint history, and separates future generation from conditioning on reserved observations for density assessment.
The frozen forecasting budget remains two banks of four generation draws and eight density draws per positive-weight particle.
Source preparation waits for independent complete-checkpoint and final-target verification, then publishes checksummed immutable copies of the reports and checkpoints.
It resolves the verified final attempt rather than copying live files or assuming attempt zero will finish.

Native adapter job `22715007` completed in 1:18 with 3,012 actions.
Independent reader `22715008` completed in 29 seconds with 192 additional prefix actions.
The checks cover three distinct physical candidates across six generation/density histories, including two candidates drawn through local components of the new guide.
They also retain twelve malformed-result rejections and verify that the old guide cannot decode a new local particle as the stored physical state.
The reader's source checksum matches the completed adapter report.
These results establish the tested adapter mechanics, not the quality of the pending fitted posterior.

Full forecast jobs `22715115` and `22715117` wait for fit readers `22714628`/`22714629` and the adapter reader.
Independent forecast readers `22715116` and `22715118` and final comparison `22715120` are dependency-queued.
Comparison fixture `22715119` completes successfully, verifying the original results and incumbent while retaining missing new forecasts as pending.
The comparison preserves the original physical prior, recording, simulator program, likelihood, numerical seeds and fitting budget across the old and new guides.
It reports replica agreement, predictive errors, future density, ancestry and fit/forecast cost without treating a higher fitting score as acceptance.
Cost records include available interrupted-attempt reports and explicitly exclude native work that an interrupted process never returned.

## Checking parameter dependence

The fixed program selects box mass by material and reads a balloon's color-specific lift coefficient when that balloon is tied and live.
This suggests that the fitting prefix may leave several parameters unobserved.
Bundle `logs/uncertainty_balloons_parameter_dependence_20260914` tests this using the same two supported fitted scenes and latent direction paths as the conditional profiles.
At each scene it evaluates the original point, each of the ten parameter coordinates at prior-unit values 0.05 and 0.95, and two simultaneous interventions on the five suspected inactive parameters.
All non-parameter coordinates remain fixed, and every evaluation repeats in a fresh native trajectory.
The report preserves both changed and unchanged predictions, likelihoods and support outcomes rather than assuming the suspected independence is true.
It also compares the original completed populations' weighted parameter marginals with the declared prior in unit coordinates.
These finite interventions are local dependence evidence; they do not by themselves prove a global likelihood factorization.
Native job `22715205` and independent reader `22715206` have completed on the matching Intel compute node, with 5,888 and 1,792 native actions respectively.
All five suspected inactive parameters preserve both evaluated histories and scores exactly, while the other parameters have detectable effects.
The [selective-guidance follow-up](balloons-selective-guidance.md) records the unexpectedly narrow fitted marginals and a target-preserving numerical intervention.
