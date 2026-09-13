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

## Initial balloon-orientation check

Job `22646175_0` holds the saved root, world-frame velocities, parameters, actions and conditional direction draws fixed while changing initial balloon orientations.
It tests the original orientations, an independent rotation of each of the three balloons, and independent rotations of all three together.
All five 235-action paths repeat exactly in fresh worlds.
Each rotated case also has exactly the same complete path, public readings and composed likelihood factor as the baseline.
The [report](../../logs/uncertainty_balloons_orientation_gauge_20260912/pilot-22646175_0.json) retains the actual intervened physical roots.

This supports investigating whether the nine initial balloon-orientation coordinates can be integrated out as rotational gauges under this fixed program.
The program resets a released balloon's orientation and motion before attaching it, but any reduction must also account for the pre-release contact dynamics and native inertia.
The test is specific to the sampled root and declared transition model; it does not establish invariance for arbitrary future simulator programs.
No initial-state dimensions or prior factors have been removed by this diagnostic.

### Native justification and quotient representation

The follow-up `22647466` checks both saved roots, with the original world-frame angular velocities and a separate larger-angular-velocity control.
For each combination it compares original, identity and random initial balloon rotations over all 235 actions, repeating every trajectory in a fresh world.
All twelve trajectories repeat exactly, and all eight orientation comparisons have exactly equal public paths and composed likelihood factors.
Changing angular velocity changes root 0's trajectory and score, with a maximum public-coordinate difference of approximately `4.03e-5`; root 1's tested spin change has no effect.
Angular velocity therefore remains in the initial-state model.

The native readbacks establish that each balloon has a centered sphere collision shape of radius 0.03 m, mass 0.005 kg and isotropic inertia `diag(1.8e-6, 1.8e-6, 1.8e-6)`.
The collision and inertial origins coincide, and there are no articulated joints.
The frozen constructor sets scalar lateral, rolling and spinning friction and no anisotropic friction.
The learned program does not observe the initial balloon rotations and resets each rotation to identity, with zero motion, before creating an attachment.
Thus rotating an unattached sphere's body frame changes neither its geometry nor its world inertia or scalar contact law, and the rotation is discarded before it could define an attachment frame.
Initial collision-feasibility tests and public readings are also independent of this rotation.

Under this fixed program and prior, write the initial balloon rotation as `R` and all retained variables as `z`.
The original law factorizes as `p(z) dHaar(R)`, independently for each balloon, and the likelihood and feasibility indicator depend only on `z`.
Integrating each normalized Haar measure contributes exactly one.
The new coordinate map therefore uses identity rotations as representatives of these equivalence classes, without treating the initial rotations as known or narrowing their original priors.
Their posterior marginals remain the independent original Haar laws if full states need to be reconstructed later.
This reduction is tied to the audited program and runtime; it is not applied to arbitrary future simulator programs.
The quotient uses the physical rotational symmetry; the reported native tests do not establish bitwise equivalence for every possible floating-point state.

The resulting initial-state target has 33-64 active continuous coordinates across the same sixteen motion cases.
Including ten parameters and 470 conditional velocity directions gives 513-544 active coordinates; the previously studied root 0 component has 531.
The original unreduced counts above remain the provenance of the earlier experiments.

The [root 0 native report](../../logs/uncertainty_balloons_gauge_native_v2_20260912/pilot-22647466_0.json) and [root 1 native report](../../logs/uncertainty_balloons_gauge_native_v2_20260912/pilot-22647466_1.json) preserve the body properties, source definitions, physical roots and complete paths.
The first audit attempt, `22647401`, failed while reporting source for a dynamically defined class, before evaluating any trajectory.
The corrected audit extracts the definitions from the frozen source files; the earlier setup failure remains archived and is not a model failure.

### Joint coordinate preflight

The next [coordinate-map manifest](../../logs/uncertainty_balloons_joint_map_20260912/plan.json) combines the fixed parameter priors, reduced scene law and every conditional transition direction in one deterministic unit chart.
It retains all sixteen robot/balloon rest-motion alternatives, all world-frame velocities, full box rotation, uncertain fixture placements and the unobserved robot joints.
The chart uses 548 unit coordinates: ten parameter coordinates, 68 scene slots and 470 direction coordinates.
The scene slots include four mixture selectors and auxiliary coordinates unused in rest components; integrating those unused uniform coordinates contributes one.
Uniform and log-uniform parameter quantiles preserve the declared original laws, and the initial Gaussian/truncated-Gaussian scene transforms preserve their first-observation conditioning factors.

Array `22647822` checks the mapped saved roots, all sixteen motion cases and parameter perturbations on the negative-support root.
It verifies initial exact observations, geometric feasibility and repeated complete trajectories, while retaining subsequent event contradictions.
The coordinate inverse is a numerical proposal initializer: a tiny floating-point change in the reconstructed root is not claimed to reproduce the earlier saved path exactly.
Each mapped point must instead pass its own repeated native replay.
This preflight is not a posterior sampler or a numerical-adequacy certificate.

Both tasks completed successfully.
All 24 mapped cases pass initial geometry, preserve every exact initial observation and repeat their complete trajectories exactly.
The mapped positive root and all sixteen motion alternatives have finite complete path factors.
The mapped negative root and its six parameter perturbations retain later event contradictions.
Reconstructing the positive root through quantile coordinates changes its public features by at most `1.05e-17` and joint coordinates by at most `6.67e-16`.
Its path factor changes slightly from the earlier physical-root experiment, as expected for a numerically reconstructed proposal point; its own repeated runs match exactly.

### A proposal that accounts for table clearance

The first broad/local mixture preflight, `22648086`, found only two geometrically valid and finite points among 24 draws.
The diagnostic `22648389` reproduces the saved points on another checked runtime and identifies the rejected contacts.
Most local rejections are box/table penetration: independently perturbing height and full box orientation can lower a corner through the table.
Other rejected points put a balloon below the table surface.
These failures are retained as geometry rejections, rather than being repaired after sampling.

The revised proposal conditions the local height distribution on the proposed body's vertical support clearing the visible table surface.
For the box, the lower height bound uses its half-width times the sum of absolute entries in the world-vertical row of its proposed rotation matrix.
For a balloon, it uses the sphere radius.
The additional `1e-6 m` clearance belongs only to this proposal component, not to the prior support, collision criterion or observation likelihood.
Each height is drawn from a normalized truncated Gaussian in the original root's unit coordinate; its normalizer depends on the proposed box orientation and remains in the proposal density.
The entire candidate still passes the native full-scene collision check.

The mixture retains weights `0.1 / 0.45 / 0.45` for broad, local and wider-local components.
The broad component retains the complete original unit support, including feasible configurations omitted by the local clearance restriction.
The importance correction uses the full mixture density, not just the selected component's density.
All motion selectors, conditional direction coordinates and unused rest-case auxiliary coordinates remain uniform in every component.
Thus this is a change in how candidates are proposed, not a new parameter or physical-state prior.

The revised preflight `22648872` completed all 24 draws.
The broad component retains eight geometry rejections; all sixteen local draws now pass geometry and repeat their full trajectories exactly.
Four of eight local and all eight wider-local points have finite full-recording likelihoods, while four local points retain exact-event contradictions.
An independent coupled-height quadrature reference, `22648893`, verifies normalization of each of the three proposal densities to `1e-7` and checks 300 transported points.
These checks establish proposal accounting and supported candidates, not posterior coverage or an agent advantage.

### Full joint sampler pilots

Array `22649168` submits two full-recording inference pilots with numerical seeds 300 and 301.
Each uses 128 particles, 32 cubic-spaced temperatures and eight Metropolis moves per stage, with a maximum of 32,896 target evaluations.
Proposal blocks cover the mixture selector, individual parameters, coupled scene groups and sixteen-action direction windows.
The base factor retains the initial conditioning terms, inverse mixture density, exact joint/speed transition densities and full geometry/event support.
The remaining sensor and marginalized output-discrepancy likelihood is tempered to its complete value.
The original parameter priors remain fixed.

Each worker checks the mapped support witness twice before fitting, freezes its actual runtime identity, and saves complete [sampler continuation records](sampler-checkpoints.md) on shared storage.
Both are restricted to node1391's Intel Xeon Gold 6230 runtime so resumed fits and replica comparisons do not silently change CPU-dependent numerical paths.
The eight-hour jobs use `mit_preemptable`; unfinished stages can be repeated from the last completed checkpoint.
Attempt telemetry separately counts reconstructed primitive actions, including the startup replay checks.

The [pilot manifest](../../logs/uncertainty_balloons_joint_pilot_20260912/plan.json) identifies the full target, proposal, source, runtime controls and validation inputs.
The original geometry normalizer and fixed initial robot-output factors cancel from these posterior comparisons; the pilots do not estimate model evidence.
Their numerical availability remains unevaluated until the separate assessment is complete.
No resulting parameter distribution is routed to the acting agent by this experiment.

## Checkpoint continuation, September 13

Both original array tasks `22649168_0` and `22649168_1` reached their eight-hour Slurm limits with terminal `TIMEOUT` states.
Numerical seeds 300 and 301 retained complete-stage checkpoints at stages 14 and 13, after 7,922 and 7,601 evaluations respectively.
Their old progress reports still say running because the scheduler terminated the process; those reports do not override the terminal job states.

Array `22671041` continues those same numerical seeds from the saved sampler populations and random states, using the identical frozen worker, target, proposal and 32,896-evaluation cap.
Each continuation has another eight-hour allocation on node1391; the additional allocation is part of the total inference cost and is not a new experiment seed.
The original checkpoints were copied and hashed before continuation in `logs/uncertainty_balloons_joint_pilot_20260912/continuation-20260913`.
The first continuation has confirmed `resumed=true` with 7,922 saved evaluations; the other task is queued for resources at this update.
No completed Balloons posterior or numerical adequacy result is available from these fits yet.

## Unconditional future-generation audit

Compute job `22652531` validates the generation half of the stochastic forecast using the same mapped support witness and frozen learned program as the joint sampler preflight.
It first reproduces the complete 235-action conditional path factor exactly as `-33314.92152710342`.
It then holds the first 64 conditional actions and their direction coordinates fixed and generates eight different 32-action physical futures.
The witness was previously selected using the full training recording; this is a mechanical integration check, not a prefix-fitted posterior or an independent held-out estimator comparison.

During future generation, the nine joint-position corrections are unconditional Gaussian draws with standard deviation 0.001.
Box linear velocity is drawn through `VelocityDiscrepancy.sample` with rest probability 0.1 and moving scale 0.01, retaining angular velocity.
The predicted link-cache observation phase, attachments, discrete events and simulator memory retain the existing transition protocol.
The generator's observation lookup table contains only the initial frame and 64 prefix readings, so attempting to retrieve a future reading would fail.
Recorded future speeds and joints never determine the future corrections.

All eight physical paths repeat exactly in fresh worlds, and all eight retain exactly the same conditional prefix as the full-recording reference.
Their future paths differ across process-noise seeds.
For two seeds, the complete generated paths also match the earlier inline velocity-draw implementation exactly.
The audit performs 1,963 native actions, including the full reference, repeated paths and inline-draw comparisons.
It runs on `node1381`, with the same Intel Xeon Gold 6230 CPU model as the original `node1391` preflight, and records its actual runtime.

Each physical path then generates a complete 32-frame observation future with all 58 public fields.
Robot xyz output errors use only the fitted prefix; Euler and sensor draws follow their declared laws, and the finger readout follows its sampled source coordinate.
Exact fields without an assigned output-discrepancy factor retain their generated physical values.
The observation draws repeat with the same output seed and never feed back into physical simulation.
Prefix speed is omitted only from the output-error conditioning input because its density is already represented by the physical transition factor; generated future speed remains present.

Across the 256 generated future transitions, the rest branch occurs 25 times.
The 2,304 joint-position innovations have empirical mean `1.0623e-6` and standard deviation `0.0010270`.
These are descriptive generation diagnostics, not posterior predictive calibration or task outcomes.
The separate component tests validate the velocity mixture's moments, rest probability and speed CDF against independent probability references.

The original audit `22652483` reproduced the full conditional score but failed an assertion that incorrectly required robot xyz output-discrepancy draws to equal their native values.
The corrected assertion exempts all assigned output factors; the transition law and sampled paths were unchanged.
That failed diagnostic remains archived.
The successful report, source hashes, candidate coordinates and per-seed physical/observation artifacts are in `logs/uncertainty_balloons_future_generation_v2_20260912`.

This audit supplies unconditional physical and observation generation, not the complete stochastic posterior forecast interface.
Numerically assessed joint posterior weights and integration over future transitions remain required.
In particular, future-density evaluation must use Gaussian joint and radial speed factors with conditional-history integration, rather than treating finitely many unconditional paths as exact-output equality components.
