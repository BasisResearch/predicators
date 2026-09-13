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
Neither has a completed posterior or prediction comparison yet.

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
