# Domino joint-inference integration

September 12, 2026.

This experiment connects the offline sampler to a complete candidate physical scene and the [composed observation likelihood](orientation-discrepancy.md).
It is an integration pilot for the simplification plan, not a validated replacement fitter or an agent solve-rate experiment.
Production parameter fitting and execution estimation remain on the incumbent implementation.

## Target and initialization

The frozen program is Domino training seed 0's `cycle_000_vers_002_simulator.py`, running against historical source `b09217bb3` with the saved offline-module overlays.
The fit uses the initial observation and the next 64 recorded actions and observations from L01.
It conditions on the observed unheld initial case and immutable task descriptors.
It does not claim a distribution over other initial attachment cases or use recorded private velocities.

The five parameter declarations have identical bounds and scales in both archived program versions.
Changed fitted initial values are not new prior centers.
The check establishes declaration stability across these versions, not that the program was specified independently of its training observations.

| Parameter | Fixed prior |
| --- | --- |
| Lateral friction | Log-uniform [0.01, 2] |
| Restitution | Uniform [0, 0.9] |
| Rolling friction | Uniform [0, 0.1] |
| Spinning friction | Log-uniform [0.01, 2] |
| Mass | Log-uniform [0.005, 1] |

Before geometric conditioning, a declared mixture assigns probability 0.8 to all six dominoes resting upright on the support plane and 0.2 to free orientations and motion.
This shared rest/moving case is an engineering prior, not a guarantee inferred from missing recording fields.
Body placement cells come from the visible component workspace, eroded by enclosing body radii.
The rest component uses uniform horizontal position and yaw, with fixed support height and zero motion.
The moving component uses uniform position, Haar orientation, independent linear velocity coordinates in [-0.1, 0.1] m/s and angular coordinates in [-0.2, 0.2] rad/s.

Movable robot joint positions have zero-centered Gaussian priors with standard deviation pi radians for angular joints and 0.1 m for prismatic joints.
Nine exact initial controlled positions are conditioned with their original density retained; four other movable positions remain uncertain.
All movable joint velocities are zero in the rest case and uniform within plus or minus 0.1 in the moving case.
Including the five parameters, the active continuous dimensions are 27 at rest and 94 when moving, plus the discrete case.

Geometric conditioning rejects the entire scene for prohibited queried penetrations below -1e-7 m.
The only explicit fixture exceptions are the two fixed wheel/plane contacts at -0.011075 m under this full-reset initializer.
This check does not add robot self-collision to the engine model.
One whole-scene support normalizer applies across both cases; feasibility is not normalized separately within each case.
The five contact/mass parameters do not alter this geometry, so that common normalizer cancels within the parameter/state posterior.
No model-evidence estimate is claimed without evaluating the normalizer.

The proposal concentrates horizontal positions and supported yaw around the first noisy readings using truncated Gaussian distributions.
The original physical priors remain those above, and the prior/proposal density ratios are retained.
The complete initial observation is scored once, with the conditioned exact joint factor retained once.
The remaining observation factors use the explicitly declared scalar and coupled-orientation output discrepancies; this is not a sensor-only posterior.

Each candidate initializes a fresh world using its sampled poses and joints before semantic state restoration, then installs full body orientations, velocities and joint motion.
Refreshing `get_observation()` updates the simulator's backing observation before uninterrupted action replay.
This avoids allowing noisy pose values to determine initializer side effects before replacing them with sampled poses.

## Sampler implementation

`SamplerConfig.proposal_blocks` optionally partitions the proposal coordinates into disjoint groups.
Each move selects a group uniformly and applies a symmetric Gaussian proposal within it, retaining the existing bounds rejection and conditional density correction.
State-independent selection preserves the same tempered target.
An empty block declaration keeps the original full-vector algorithm and random stream.
Blocks do not increase the evaluation budget or establish that a high-dimensional target has been explored adequately.

Validation passed 17 functional tests, including conditional and unconditional correlated-grid references, uninformed marginals, malformed partitions and budget exhaustion.
Two-file type checking and lint passed in job `22636679`; its only remaining failure was docstring wrapping, subsequently corrected.
Job `22636896` passed final pinned formatting and exact old/new default-result comparisons across eight RNG seeds, excluding only the added empty configuration field.
The final executable syntax trees match the functional/type/lint snapshot after removing docstrings.

The pilot uses 95 unit proposal coordinates with auxiliary unused coordinates in the rest case.
Those auxiliary uniforms integrate to one; they are not additional uncertain physical quantities.
Separate blocks update the five parameters, the mixture selector, robot state, and each domino's state.
Each of the two runs uses 32 particles, eight temperatures, four moves, proposal scale 0.05, and at most 1,056 candidate evaluations.
These intentionally small budgets test integration before larger numerical comparisons.

## Validation and artifacts

The corrected scene preflight, job `22636219`, accepted eight of eight proposed scenes and reproduced every 64-action trajectory exactly in a fresh second world.
It covered six rest and two moving candidates.
Two candidates had finite complete-output likelihood; the other six contradicted exact outputs.
Those candidate rejections are neither infrastructure failures nor proofs that the whole model is inconsistent.
The earlier `22636183` setup attempt failed by assigning the read-only `_current_state` property; its artifact is retained separately.

Artifacts are in `logs/uncertainty_domino_joint_scene_v3_20260912` and `logs/uncertainty_domino_joint_fit_20260912`.
The latter freezes source, overlays, program/data hashes, prior policy, worker and compute configuration.
Pilot jobs `22636393_100` and `22636393_101` completed on `mit_preemptable` and wrote individual final reports.
Completion, final particle ESS, or finite likelihood alone cannot establish numerical adequacy.
Independent-run agreement, budget sensitivity and predictive assessment remain required before posterior use.

Both pilots completed the temperature schedule but collapsed to one surviving initial ancestor.
Their parameter estimates disagree substantially: the lateral-friction medians are approximately 0.112 and 0.0206, and the mass medians are 0.212 and 0.0343.
These are diagnostic outputs of inadequate small-budget approximations, not reportable parameter estimates or evidence of identification.
The subsequent [process reproducibility audit](sampling-reproducibility.md) also found uncontrolled initialization order and small cross-node differences in generated poses.
Their disagreement therefore cannot be attributed solely to sampler RNG variation under an identical numerical runtime.

| Sampler RNG seed | Candidate evaluations | Finite initial particles / 32 | First-stage ESS | Surviving ancestors | Distinct final parameter vectors | Worker seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 612 | 11 | 1.00002 | 1 | 1 | 561.2 |
| 101 | 769 | 15 | 1.00000 | 1 | 3 | 651.4 |

The total accepted-move counts, 371 and 420, include updates to initial state and auxiliary coordinates and therefore do not demonstrate exploration of the parameters.
The next sampling comparison must address concentration at the first temperature and poor physical-parameter movement, retaining the same target and reporting independent-run agreement.
Increasing the budget or changing proposal groups is a numerical experiment, not permission to publish the current collapsed samples.

The runtime identity is still a development identity rather than complete capture of installed native dependencies and assets.
The program was synthesized from historical training experience, so later recording suffixes are not established as unseen during program synthesis.
At that pilot stage, the full legacy-fitter comparison, all-five-domain validation and matched planning experiments remained open.
The later [legacy comparison](offline-fitter-comparison.md) supplies completed incumbent fits and predictions; matched replacement-posterior and planning comparisons still remain open.
Fan also initially lacked an explicit original prior because its latest program narrowed bounds using the fitting data.
The subsequent [Fan joint-inference experiment](fan-joint-inference.md) addresses that prior declaration without adopting the data-narrowed bounds as independent prior information.

## Conditioned-base continuation and compute recovery

The subsequent [reproducible initialization experiment](sampling-reproducibility.md) uses 64 particles, 32 cubic-spaced temperatures and eight moves per stage, retaining the initial-observation factor in the conditional base.
It fits the same 64-action development prefix under fixed priors and an identified CPU/runtime.
Run `22637359_100` reached its four-hour Slurm limit before producing a completed sampler result.
Its last saved progress report recorded 9,536 target calls; the scheduler reports `TIMEOUT`, not an agent outcome or a completed posterior.
No sampler checkpoint existed in that frozen worker, so its best candidate cannot serve as a continuation state.
The other numerical seed, `22637359_101`, used an eight-hour allocation and subsequently completed.

Replacement `22649657_100` restarts numerical seed 100 from the original prior with the same data, priors, temperature schedule and evaluation budget.
It adds the tested scalar likelihood optimization, [stage checkpoints](sampler-checkpoints.md) and an eight-hour allocation on the same AMD EPYC 7542 worker node.
The runtime identity changes to identify those source changes; the statistical model and sampler configuration do not change.
Its startup checks compare complete old/new likelihoods on two replayed candidates before fitting.
One pair retains zero support, and the finite pair matches exactly at `8141.576094195281` on this AMD runtime.
The retry saved its initialized population after 64 evaluations and subsequently completed all 32 stages.
All sixty recorded finite-initial-base entries also match the timed-out run's entries exactly.

The [retry manifest](../../logs/uncertainty_domino_conditioned_checkpoint_20260912/plan.json) retains the timeout reason and hashes the old likelihood source used for the paired check.
Attempt files retain per-attempt counters; the sampler result and checkpoint retain the cumulative numerical evaluation count.
This recovery does not establish posterior adequacy, convergence or an improvement in agent performance.

### Completed numerical pilots, September 13

Both conditioned-base fits now have terminal `COMPLETED` job states and complete, structurally valid 64-row weighted sampler results at temperature 1.
Numerical seed 100 used 14,318 target evaluations in its replacement attempt, and seed 101 used 14,038.
Their completed attempts took approximately 5 hours 8 minutes and 5 hours 32 minutes; seed 100 also incurred the earlier four-hour timed-out attempt.
Both retained only one original ancestor after 11 resampling events.
The minimum recorded effective sample sizes were 18.32 and 20.33, respectively.

Their empirical parameter summaries disagree substantially:

| Parameter | Seed 100: 5th / 50th / 95th percentile | Seed 101: 5th / 50th / 95th percentile |
| --- | --- | --- |
| Lateral friction | 0.11496 / 0.17337 / 0.21346 | 0.42817 / 0.65932 / 0.87769 |
| Restitution | 0.25945 / 0.41405 / 0.55124 | 0.72363 / 0.87345 / 0.89240 |
| Rolling friction | 0.003084 / 0.003496 / 0.006734 | 0.000129 / 0.000129 / 0.002509 |
| Spinning friction | 0.01137 / 0.02051 / 0.02760 | 0.03731 / 0.04631 / 0.06521 |
| Mass | 0.30506 / 0.40014 / 0.53333 | 0.49608 / 0.64470 / 0.78762 |

These are summaries of the numerical populations, not validated credible intervals.
Shared data, sensor, program, prior and sampler configuration were verified, along with complete finite samples and normalized nonnegative weights.
The runtime-source difference and its earlier exact likelihood parity checks remain part of the comparison's provenance.
Completion does not resolve the disagreement or establish trustworthy posterior coverage.
Predictive stability and a defensible exploration of the joint parameter/initial-state distribution remain required before any posterior publication or legacy comparison claim.
The source hashes and completion checks are retained in `logs/uncertainty_domino_conditioned_checkpoint_20260912/completion-comparison-20260913.json`.

### Future predictions from the completed populations

Array `22671823` replays every complete weighted row of each unassessed population through the 97-action suffix following the fitted 64-action prefix.
It uses the retained physical parameter values directly, avoiding a lossy inverse transform back into prior coordinates, while preserving each row's state coordinates and correlations.
Before evaluating the population, each worker reconstructs the archived best candidate exactly, repeats its full 161-action history, and reproduces its original fitting log likelihood exactly.
The first retained row is also repeated exactly, every fitting prefix has finite likelihood, and no row is dropped or reweighted using future observations.

For Cartesian readings, conditional output means and variances follow the declared scalar error process filtered on the fitting prefix, with the original future sensor variance retained.
Native toppling indicators use the frozen domain threshold; they describe model predictions, not achieved task outcomes.
These diagnostics consume unassessed populations to investigate their disagreement and do not bypass the production posterior-assessment boundary.

Both jobs completed and all 128 compressed history artifacts passed their hash checks.
Each job performed 10,787 native actions, including its reconstruction checks, and took approximately 3.5 minutes.

| Quantity | Numerical seed 100 | Numerical seed 101 |
| --- | ---: | ---: |
| Future Cartesian conditional-mean RMSE | 0.011585 m | 0.012130 m |
| Future native-mean RMSE | 0.011688 m | 0.012354 m |
| Whole-future mixture log density | 14105.63278 | 14434.93358 |
| Final predicted toppling probability, domino_1 | 0.96875 | 0.32628 |

Across all future Cartesian readings, the two conditional means differ by 0.006934 m RMS, with a maximum difference of 0.052394 m.
Their predicted standard deviations differ by 0.001694 m RMS.
These apparently similar averaged errors do not imply equivalent decisions.
At primitive step 158, the two populations predict `domino_3` toppling with probabilities 0.90625 and 0, respectively.
Their final `domino_1` predictions also differ by more than 0.64.

The full future log-density estimates differ by 329.30080, and each is dominated by roughly one weighted contribution among the 64 rows.
An informative future can concentrate those contributions even under a valid prefix posterior, so this alone is not proof that the probability model is wrong.
Together with the replica differences, it leaves the present finite-population predictive calculation unsuitable for numerical acceptance.
A low average position error on one suffix cannot establish posterior coverage, toppling/timing stability, or unchanged agent performance.

The source populations, reconstruction code, full histories, per-row scores, complete forecast curves and comparison checks are retained in `logs/uncertainty_domino_population_forecast_20260913`.
This is a population-stability diagnostic; a completed matched comparison against the incumbent predictions and the broader validation gates remain required.
