# Fan prefix comparison

This is a Stage B development comparison under the [simplification proposal](simplification-proposal.md).
The existing full-recording Fan inference pilots use all 132 actions, so they cannot provide an unused suffix from that recording.
The separate legacy fit `22638308_1` uses 64 actions and already supplies predictions for the remaining 68 actions.

## Prefix-only support check

Job `22674026` completed successfully on `mit_preemptable` on September 13, 2026, with one CPU on node1412.
It used 106 allocation seconds, 90.47 worker seconds and 4,160 native actions under its 20-minute allocation limit.
Its frozen bundle is `logs/uncertainty_fan_prefix_support_20260913`.
The submission manifest records worker and configuration hashes.

The worker retains only the initial observation and first 64 action/observation transitions before constructing any candidate scene or scoring predictions.
It checks that the sensor schema and every retained observation use exactly the first observation's feature keys.
The full recording hash remains provenance, while the inference data identity describes only the retained prefix.
It uses the same historical program, physical runtime, original scene prior and output discrepancy model as the full-recording Fan pilots.
The historical program may have seen later training experience during synthesis, so this is an estimator comparison on a reserved suffix, not an unseen-program evaluation.

The search does not import the full-recording supported candidate or its guided proposal.
It checks 16 stratified original-prior fan speeds at the first-observation proposal median scene, followed by 48 independently sampled scenes from the original first-observation proposal restricted to its robot/ball rest components.
Rest-component restriction is a declared support-search choice; these candidates are neither joint-prior samples nor posterior samples.
Unknown rotor state and fixture locations retain the declared chart semantics.
No omitted velocity is claimed to be an exact observed zero.

Every candidate is simulated for 64 native actions in a fresh world, with geometry witnesses, full-prefix likelihood, initial likelihood and complete prediction digest retained.
The first feasible finite candidate is repeated exactly; if none is found, the first median candidate is repeated instead.
A finite search with no supported candidate cannot prove that the target has no support.

## Acceptance and follow-up

The completed native audit found the following support:

| Candidate construction | Probes | Valid initial geometry | Finite likelihood and valid geometry |
| --- | ---: | ---: | ---: |
| First-observation proposal median, stratified speed | 16 | 0 | 0 |
| Sampled first-observation proposal, robot/ball rest cases | 48 | 12 | 8 |

The first supported candidate, index 26, repeated with exactly matching likelihoods, geometry witnesses and complete predicted-observation digest.
The median scene has a 5.11 mm initial penetration and is unsuitable as a fixed-state baseline without a separately declared feasible selection policy.
The independent verification checks all 64 result records, native action accounting and all submitted script hashes.
Reports and verification are retained in the frozen bundle.

This establishes prefix-only support for constructing a prefix-specific inference proposal; it does not establish numerical adequacy.
Any selected guide must depend only on the prefix, and its proposal density must be included in the inference correction.
Keep a component with full original support if adding local guidance.
Independent numerical replicas and budget sensitivity remain required before treating fitted samples as usable posterior estimates.
Only then compare the frozen-weight predictions on the remaining 68 actions against the existing legacy predictions, including ball trajectory, event timing, goal outcomes and computation cost.
This support audit produces no agent solve-rate result and does not change the production fitter.

## Prefix inference protocol

The follow-up bundle is `logs/uncertainty_fan_prefix_inference_20260913`.
It uses the original 120-coordinate proposal conditioned on the first reading, with the prior/proposal correction retained.
It adds no local mixture around a selected supported candidate.
Both resting and moving robot/ball cases remain available with their original probabilities, unlike the restricted support search above.
The parameter prior remains `Uniform[0, 1]` for fan speed.

The base factor contains the initial observation likelihood and the original-prior/proposal correction, subject to initial geometry and complete-prefix exact constraints.
Tempering applies to the remaining prefix log likelihood, so each observation enters exactly once.
The inference data identity includes only the first 64 actions and 65 observations.
The fixed source, offline modules, worker scripts, actual numerical runtime and isolated working directory enter the runtime identity.

Two numerical replicas are specified with seeds 302 and 303, 64 particles, 32 cubic temperature stages, eight moves per stage and a maximum of 16,448 target evaluations each.
Their proposal kernel mixes local moves with full-range block refreshes at probability 0.5.
The parameter has its own block; other blocks group coupled robot, fixture, ball, switch and rotor coordinates.
Four isolated processes evaluate complete candidate maps and likelihoods in order.
Complete-stage checkpoints preserve the fixed numerical budget across preemption; interrupted work remains additional actual cost.

The compute preflight checks retained support-audit factors and identical initial sampler states across serial and four-process execution before the replicas may launch.
A completed fit remains numerically unassessed until independent prediction comparisons and budget sensitivity have been evaluated.
The two replicas are inference diagnostics on the same recording, not two agent seeds or evidence of solve-rate improvement.

The corrected preflight `22674192` completed successfully in 159 allocation seconds and 152.93 worker seconds, with 9,728 native actions.
All 12 checked native target rows and the entire 64-particle initial sampler state match exactly between serial and four-process evaluation.
Initialization found 11 finite particles with weight effective sample size 9.10; this describes initialization only.
Serial initialization took 84.68 seconds and parallel initialization took 21.64 seconds, excluding process startup and the separate fixed-row checks.
The preceding attempt `22674140` failed in report construction because checkpoint log weights are serialized strings; its native comparisons had reached equality assertions, and its allocation and report remain retained.
The corrected attempt changes only summary decoding and adds the initial weight effective sample size.
The launch manifest pins the successful preflight, target scripts and configuration before any fit starts.
Array `22674242` submits task 0 for numerical seed 302 and task 1 for numerical seed 303, each with four CPUs, 20 GB and a four-hour allocation limit on `mit_preemptable`.
Both fits and their reserved-suffix forecasts have now completed; numerical adequacy remains unestablished, as detailed below.

## Reserved-suffix forecast validation

The separate bundle `logs/uncertainty_fan_prefix_forecast_20260913` prepares comparisons on the 68 actions reserved from fitting.
Its source guard requires a complete fit report and a matching, complete checksum-verified checkpoint.
Resuming that checkpoint must reproduce the entire saved sampler result without requesting any new target evaluation.
The source prior, inference identity and configuration must also match the validated fitting protocol.

Prediction uses a separately identified observation module with conditional future scoring; the running fit snapshots remain unchanged.
Every positive-weight particle reconstructs its saved native scene directly from its original proposal coordinates.
Its physical fan speed, model digest and both prefix target factors must match the fitting checkpoint exactly.
Zero-weight rows are identified explicitly, and poor future likelihood never removes a positive-weight row or changes its weight.
Each trajectory runs all 132 actions from its initial scene, preserving the physical state through the fit/prediction boundary.
The first positive-weight trajectory is repeated, and compressed complete histories are retained with checksums.

Ball-position predictions include both the native simulator mean and the output-error mean conditioned only on the prefix.
Reported events include switch/fan activation and the learned target-hit readout.
The native geometric goal predicate is evaluated separately using the frozen environment's axis tolerances, so a learned target-hit field cannot silently substitute for goal geometry.
The comparison reports goal probability, first goal occurrence within the suffix, event Brier errors and native computation cost alongside position error.
The legacy report must use the identical prefix data and candidate program, with predictions aligned by their primitive step indices.

A short native fixture, `22674287`, completed one temperature stage and one move using the original fitting runtime, with 7,872 native actions and 66.08 worker seconds.
It has one surviving initial ancestor and is explicitly unassessed.
It exists only to exercise complete-checkpoint recovery and forecast comparison before the two full fits finish.

Forecast validation `22674541_0` completed successfully in 71 allocation seconds and 64.73 worker seconds, with 8,580 native actions.
It recovered the complete checkpoint exactly and verified the saved prefix factors for every positive-weight particle under the forecast runtime.
Independent artifact checks verify all 64 complete 133-frame histories, their hashes, original physical samples and unchanged fitting weights.
All 64 fixture particles assign zero density to the observed future; they remain included in the trajectory, event and goal summaries, and the empirical mixture density is reported as zero.
This is useful coverage of the evaluator's failure handling, not evidence of a reliable posterior or an estimator advantage.
The two full-fit forecast jobs are submitted with dependencies on this successful validation and their individual source fits.

| Numerical seed | Prefix fit | Dependent forecast |
| --- | --- | --- |
| 302 | `22674242_0` | `22674579_1` |
| 303 | `22674242_1` | `22674580_2` |

## Paired forecast report

The report in `logs/uncertainty_fan_prefix_summary_20260913` verifies the two full prefix-fit forecasts before comparing them.
It checks their frozen script and plan hashes, complete source fits, identical prior and inference identities, numerical configurations and expected numerical seeds.
It reconstructs native ball-position means, fan/switch/target event probabilities, geometric goal probabilities and cumulative goal probabilities from every positive-weight complete history.
Original samples, weights, first-goal timing and zero-weight exclusions must match the stored artifacts exactly.
Future mixture density is reconstructed from the original weighted per-history scores, retaining zero-density predictions explicitly.

The two rows report position errors, event and goal Brier errors, parameter quantiles, ancestry, target evaluations, native actions and allocation/attempt costs alongside the matched legacy prediction.
The pair reports differences between conditional and native position means, event curves, geometric goal curves and probability of reaching the goal within the suffix.
These are descriptive comparisons on one recording, not a new acceptance threshold, calibrated posterior claim or agent solve-rate comparison.
The learned target-hit readout and native geometric goal remain separate predictions.
The real environment's recorded target-hit feature uses that geometric goal condition; the source implementation is an axis-wise distance check with tolerance `pos_gap / 2`.

Compute validation `22676293` completed with one CPU and 22 allocation seconds, performing no native simulation.
It verifies all 64 complete fixture histories and exactly reconstructs positions, events, geometric goals and timing while retaining all 64 zero-future-density particles.
It rejects dropped particles, altered weights, changed native means, changed event probabilities, changed goal probabilities and an altered zero-density mixture.
The initial summary is explicitly incomplete because the full inference forecasts are still pending.
Finite summary `22676382` depends on successful validation and termination of both full forecast jobs.
This does not re-enable the MB/MF notification monitor.

## Completed 64-particle prefix comparison

Both fits complete all 32 stages: seed 302 uses 15,466 target evaluations and retains one initial ancestor; seed 303 uses 15,609 evaluations and retains two.
They take 5,521 and 5,568 allocation seconds on four CPUs, with 989,824 and 998,976 completed native fitting actions.
These are numerical replicas on the same development recording, not independent agent seeds.
The two forecasts complete in 73 and 79 allocation seconds and retain every positive-weight history.

| Forecast | Conditional ball-position RMSE to noisy readings | Native ball-position RMSE | Native goal Brier error | Probability of reaching geometric goal in suffix |
| --- | ---: | ---: | ---: | ---: |
| Legacy point forecast | 7.072 mm | 7.072 mm | 0.044118 | 1.0 |
| Numerical seed 302 | 6.311 mm | 6.442 mm | 0.042419 | 0.78125 |
| Numerical seed 303 | 6.130 mm | 6.172 mm | 0.020787 | 0.90209 |

The recorded goal first occurs at action 116, within the reserved suffix.
The new replicas have lower position error on this recording, but their conditional mean positions differ by 2.866 mm and their native geometric goal curves differ by as much as 0.33212.
Their probability of ever reaching the goal within the suffix differs by 0.12084.
Only one positive-weight particle in each population assigns nonzero density to the complete observed future; all other particles still contribute to the position, event and goal forecasts at their unchanged weights.
Their similar mixture log densities, 18,398.66 and 18,398.31, therefore do not establish a stable density estimate.
No domain-specific numerical acceptance threshold is introduced after these outcomes.

The fan-speed empirical 5th, 50th and 95th percentiles are `[0.06622, 0.08353, 0.91372]` and `[0.07732, 0.10989, 0.93241]`.
Their largest masses at an exact retained speed value are 0.328125 and 0.283687.
These empirical distributions, ancestry and decision-relevant prediction differences require further numerical assessment.
The cost is also substantial: the matched legacy fit and replay used 30.88 worker seconds and 1,777 native actions.
This comparison does not demonstrate an agent efficiency or solve-rate improvement.

The first completed summary `22676382` preserved the prediction metrics but omitted parameter quantiles because it read a generic field instead of `fan_speed_quantiles`.
The corrected `compare-v2.py` independently reconstructs those weighted quantiles and checks the saved values.
Validation `22676681` reproduces the history checks, validates both completed parameter summaries and rejects altered quantiles; it completes in ten allocation seconds with zero native simulation.
The corrected complete report is `summary-22676681.json` in the paired-report bundle.
The first report and reader remain preserved as the reporting defect's reproduction.

## Larger-population comparison

The next numerical comparison doubles particles from 64 to 128 and the evaluation limit from 16,448 to 32,896 while preserving the program, observations, original prior, output model, proposal kernel, temperature schedule and moves.
It uses the same two numerical seeds, 302 and 303, without reusing a completed population as initialization.
The frozen bundle is `logs/uncertainty_fan_prefix_budget_20260913`.
Native preflight `22676726` must reproduce fixed target factors and match the entire 128-particle initial population between serial and four-process execution.
Its target identity and prior must exactly equal the validated 64-particle protocol, allowing only the two declared numerical configuration changes.
The preflight action count scales with the actual population size.

Finite gate `22676770` validates that evidence and pins the run inputs before array `22676775` can start.
The two full fits request four CPUs, 20 GB and six hours each on node1412 in `mit_preemptable`, with complete-stage checkpoints and at most two simultaneous tasks.
Their same-suffix forecasts are prepared as dependent jobs under the validated larger configuration, as detailed below.
Compare both larger replicas and all four cross-budget replica pairs, preserving zero-density outcomes and actual computation costs.
These jobs do not change the production estimator or restore result notifications.

## Cross-budget forecast pipeline

The frozen forecast bundle is `logs/uncertainty_fan_budget_forecast_20260913`.
Its simulator mapping, model, prediction worker, checkpoint recovery driver and runtime preparation are byte-for-byte copies of the validated 64-particle forecast implementation.
The forecast plan is finalized only after the successful larger-fit preflight is available, with exact checks on the original target identity, prior and permitted numerical differences.
Contract validation rejects changed identities, priors, proposal configurations, incomplete native equality checks and a mismatched initial population size.
This is a manifest validation, not a new native inference result.

Finite preparation job `22677008` depends on successful fit gate `22676770`.
Forecast `22677013_0` depends on that preparation and fit `22676775_0`; forecast `22677014_1` depends on that preparation and fit `22676775_1`.
Each forecast requests four CPUs, 20 GB and 30 minutes on node1412 in `mit_preemptable`.
Every positive-weight particle retains its original weight, complete 133-frame history and future-density outcome.
The 68 forecast actions remain excluded from scene construction and parameter fitting.

The budget report in `logs/uncertainty_fan_budget_summary_20260913` compares all four populations and all six pairs: one within each budget and four across budgets.
It imports the checksum-pinned complete-history verifier used by the corrected 64-particle report.
Each row must match its own validated configuration, expected numerical seed and source artifact hashes; pairs must have identical target identities and priors, allowing only particle count and the proportional evaluation limit to differ.
It reports the original weighted position, event and goal predictions, zero future densities, parameter quantiles, retained-value concentration, ancestry and computation costs.
No acceptance threshold or posterior publication is introduced by this report.

Compute validation `22677052` completed on one CPU in eight allocation seconds with zero native simulation.
It exactly reproduced both completed baseline rows and their pair comparison from 128 saved histories.
It rejected seven mismatches: target identity, prior, proposal, mislabeled particle count, dropped history, altered weight and discarded zero-density outcome.
The first summary correctly contains two completed rows, one completed pair and five incomplete pairs.
Finite summary `22677060` depends on successful report validation and termination of both larger forecasts.
These finite analysis jobs do not send notifications or re-enable the disabled MB/MF monitor.

The larger native preflight `22676726` has now completed in 268 allocation seconds, using four CPUs and 17,920 native actions.
All retained target factors and the complete 128-particle initial state match exactly across serial and four-process execution.
Independent comparison confirms the original prior and target identity are unchanged and only the two declared numerical configuration fields differ.
The initial population contains 24 finite particles with weight effective sample size 20.19; these are initialization diagnostics, not a completed posterior.
Serial initialization takes 170.43 seconds and parallel initialization 44.72 seconds, excluding the separate fixed-row comparisons and startup.
The validated preflight checksum is `1d90e36965fff29d925e87fe93717f50586720cc598175836c80593c31090b11`.

## Completed population-size comparison

Both 128-particle fits, their forecasts and summary `22677060` have completed under the same original target as the 64-particle pair.
The summary verifies all 384 positive-weight histories across four populations, their unchanged weights and all six pairwise comparisons.
Its plan and comparison-script hashes match the report, and all four forecast hashes were checked against their current files.

| Particles | Numerical seed | Conditional position RMSE (m) | Goal Brier score | Surviving initial ancestors | Fit allocation seconds (4 CPUs) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 302 | 0.00631118 | 0.0424194 | 1 | 5,521 |
| 64 | 303 | 0.00612967 | 0.0207869 | 2 | 5,568 |
| 128 | 302 | 0.00685677 | 0.0493406 | 2 | 10,854 |
| 128 | 303 | 0.00546470 | 0.0399988 | 1 | 11,087 |

The 128-particle pair differs by 5.043 mm RMS in conditional position means and by up to 0.298481 in its native goal-probability curves.
The corresponding 64-particle disagreements are 2.866 mm and 0.332124.
Across budgets, position-mean disagreements range from 2.077 to 7.317 mm and maximum goal-curve gaps range from 0.269680 to 0.549676.
Doubling the population therefore does not establish stable, budget-insensitive predictions.

The 128-particle seed-302 empirical fan-speed distribution places 50.77% of its weight at one exact value, 0.9168676.
Its 50% and 95% quantiles consequently coincide; this is not evidence of precise identification.
The two larger populations retain five and two particles with positive complete-future density; the remaining 123 and 126 particles contribute zero without having their population weights discarded.
The larger forecasts each use 17,028 native actions, taking 110 and 114 allocation seconds on four CPUs.
The final summary takes nine seconds on one CPU without native actions.
These remain offline numerical experiments, not new agent seeds or an accepted posterior replacement.
