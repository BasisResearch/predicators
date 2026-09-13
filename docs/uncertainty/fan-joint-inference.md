# Fan joint-inference integration

September 12, 2026.
This extends the [full Fan scene law](fan-initial-scene.md) into the offline batch sampler required by the [simplification proposal](simplification-proposal.md).
The production agent remains on the incumbent fitter.
These are fixed-program development experiments on the original 132-action training recording, not agent runs or held-out predictions.

## A fixed coordinate map for every initial case

The initializer now accepts a 120-dimensional unit vector instead of consuming a variable-length random stream.
Its physical prior, observation-informed proposal, initial observation factors and whole-scene geometric support policy remain the declared Fan model.
Each coordinate has a fixed role across rest/moving cases:

| Slots, inclusive | Role |
| --- | --- |
| 0 | Airflow speed under the original Uniform[0, 1] prior |
| 1 | Robot rest/motion case |
| 2-18 | Free robot positions and potentially moving velocities |
| 19-58 | Ten fixtures, four coordinates each: xyz and yaw/categorical orientation |
| 59 | Ball supported-rest/free-motion case |
| 60-71 | Ball pose and potentially moving twist |
| 72-79 | Four switch conditional position-mixture and velocity pairs |
| 80-119 | Twenty rotor positions and velocities |

Coordinates unused in a particular case remain normalized auxiliary uniforms.
The box dimension is therefore not the number of physical degrees of freedom in each scene.
The existing 83-113 active continuous dimensions, including airflow speed, and 65,536 discrete cases remain represented.
Exact controlled initial positions are still conditioned with their density retained.
No later observation is assigned directly into a simulator state.

The inverse helper only constructs a proposal center from the previously found supported-rest-ball witness.
It does not restrict the forward map to that case or claim an inverse for every free-ball quaternion chart.

Compute job `22643080_0` completed the native map audit.
The saved physical witness and its mapped reconstruction each have full-recording log likelihood 34241.55615373634.
Both repeat exactly in fresh worlds, including public observations, robot joints and nonrobot articulated state.
Four fresh unit draws also repeat exactly; three fail geometry and the fourth fails the full observation target.
Those outcomes are retained as failures to reach support, not evidence that the full target is impossible.
The [report](../../logs/uncertainty_fan_coordinate_map_20260912/pilot-22643080_0.json) records the complete coordinates, physical candidates and actual node1377 runtime.

## Informed proposals without changing the prior

Let `u` denote the original unit chart and `r0(u)` its existing original-prior/proposal correction, including exact initial joint and switch evidence.
The new proposal draws `u` from a mixture with weight 0.1 on the entire original unit cube and weights 0.45 each on nearby and wider components around the support witness.
The wider component has ten times the nearby standard deviations.
These proposal choices use the complete training recording; they are not new prior information or held-out evidence.

Each local component uses normalized truncated Gaussian laws on 57 active chart coordinates.
Matching discrete-case selectors use uniform laws over their corresponding chart intervals.
Unused auxiliaries, supported-ball yaw and all rotor states retain uniform proposal draws.
The broad component retains support for every original continuous region and discrete case, including other motion cases and fixture orientations.
It does not guarantee that a finite run actually visits those alternatives.

The mixture density `q(u)` is evaluated with log-sum-exp across all three components, including components other than the one that generated the draw.
The corrected factor is `log r0(u) - log q(u)`.
Using only the selected component's density would define different weights and is not done here.
The original parameter and scene prior remains fixed.
The whole-scene support normalizer is still common and independent of airflow speed; no model-evidence estimate is claimed.

Compute job `22643174_0` completed a stratified proposal-overlap audit with eight draws per component.
The component sample counts are diagnostic allocations, not mixture samples to pool as an unweighted posterior.

| Component | Geometry-feasible draws | Finite complete targets |
| --- | --- | --- |
| Broad unit cube | 2/8 | 0/8 |
| Nearby | 8/8 | 5/8 |
| Wider | 6/8 | 3/8 |

All three repeated first-draw trajectories match exactly, including the failing broad case.
Checks cover truncated-law CDF/quantile round trips, positive interval lengths and the broad component's density lower bound.
The largest CDF round-trip error is below 2.2e-13.
The [report](../../logs/uncertainty_fan_joint_proposal_20260912/pilot-22643174_0.json) retains every candidate and its full mixture-density correction.
This establishes usable proposal overlap for an integration experiment; it does not establish posterior adequacy.

## Submitted full-recording inference

Array `22643258` runs independent seeds 200 and 201 on `mit_preemptable`, pinned to node1412 with the same CPU and Python hash seed.
Each uses 64 particles, 32 cubic temperatures, eight Metropolis moves per temperature and a maximum of 16,448 target evaluations.
The eight-hour wall limit is an external compute cap, not a declaration of convergence.
Fresh-world 132-action replay occurs for each candidate, including candidates that change initial geometry or motion.

The sampler's proposal space has one additional mixture-selector coordinate.
Its returned joint rows contain airflow speed and the original 119 scene-chart coordinates, so downstream parameter projection uses `theta.fan_speed` explicitly.
Symmetric scalar-coordinate moves act in the 121-dimensional proposal space and include the complete density correction in their acceptance ratio.

At temperature zero the target includes the original-prior/proposal correction, initial-output likelihood, geometric support and all complete-recording exact constraints.
The tempered term is the complete-output log likelihood minus the initial-output log likelihood.
At temperature one their sum is the declared complete target, with every observation entering once.
No exact event is softened by tempering or by a tolerance window.

The [frozen manifest](../../logs/uncertainty_fan_joint_pilot_20260912/plan.json) identifies the historical runtime, offline overlays, program, source scripts, proposal, full training data and hashed prerequisite report.
Each worker first verifies a finite, exactly repeated mapped witness on its own runtime.
Sampler completion leaves numerical availability unevaluated.
Independent agreement, weight and ancestor concentration, parameter movement, budget/proposal sensitivity and predictive investigations remain necessary before returning a usable physical-domain posterior to an acting agent.

## Rollout cost audit

Both sampler tasks subsequently started on the declared node1412 and passed their own finite, exactly repeated witness checks.
Compute audit `22644679_0` compared full replay snapshots against direct public-observation collection on five saved physical candidates, including both finite and event-incompatible histories.
Every public prediction and complete likelihood matched exactly.
The [timing report](../../logs/uncertainty_fan_rollout_cost_20260912/pilot-22644679_0.json) alternates method order and clears the quaternion cache before each measurement.

Snapshot collection took 1.13-1.70 seconds per 132-action history, versus 0.95-1.14 seconds for direct public observations.
Initialization took approximately 0.60 seconds, while cold likelihood evaluation took 3.62-3.71 seconds in either method.
Removing full snapshots therefore addresses only a small part of this measured cost; likelihood evaluation is the larger target for profiling.
These are cold-cache diagnostic measurements, not the amortized cost of the active samplers, which reuse their observation-factor cache.
The running frozen experiments were not modified by this audit.

## Checkpoint recovery preparation

After roughly two hours, each original worker had completed approximately 3,500 target evaluations.
That measured throughput projects beyond the eight-hour allocation for its fixed 16,448-evaluation budget.
An attempt to extend each running allocation to twelve hours was denied by Slurm, and both original jobs remain running with their original limits.
The request, denial and verified unchanged job state are recorded in `logs/uncertainty_fan_joint_pilot_20260912/scheduler-budget-extension.json`.
No sampler result or failure is inferred from that projection.

A recovery worker is prepared in `logs/uncertainty_fan_checkpoint_recovery_20260912` with the existing checked stage-checkpoint implementation and the exact-density optimization.
It retains the original prior, data, numerical seeds, proposal, temperature schedule and evaluation budget.
Its future scheduler allocation is twelve hours; that larger external allowance must be reported separately from the original eight-hour attempts.
The frozen old workers cannot acquire checkpoints retroactively, so a recovery starts from the original prior unless a compatible new-worker checkpoint exists.
It never substitutes an old best candidate for the initial particle population.

Array `22650616` checks the new worker's complete 64-particle initialization against each original run, including proposal component, geometry, full likelihood and conditional-base factor.
It also requires the full mapped-witness likelihood to match the original exactly and to repeat on the worker.
The checks save the complete initialized sampler state with its RNG, weights, counters and identity, then stop before tempering.
The worker identity includes the frozen source plus actual Python, NumPy, SciPy, PyBullet binary, CPU features, hash seed and node, so an incompatible checkpoint is rejected.
Both check tasks completed successfully.
Each matched all 64 original initial-particle records exactly, reproduced the original full witness likelihood `34241.55615373634`, and saved a stage-zero checkpoint after 64 evaluations.
The measured worker times were 158.62 and 158.50 seconds.
This establishes initialization parity; it does not establish posterior adequacy.

Conditional recovery jobs `22650786_0` and `22650787_1` are queued behind failure dependencies on original tasks `22643258_0` and `22643258_1` respectively.
They resume the corresponding verified new-worker initialization checkpoints and retain numerical seeds 200 and 201.
A separate startup gate requires a successful current queue query showing the original absent and scheduler accounting affirmatively showing terminal failure.
It refuses an active or requeued original, successful completion, missing evidence or a failed query.
Nine controlled gate cases passed, and an end-to-end check against the actually running original correctly refused recovery.
Thus a scheduler dependency alone is not used as proof that replacement is safe to start.
Unused dependency jobs must be cancelled if their originals complete successfully.
The initially queued unguarded recovery submissions `22650769` and `22650770` were cancelled while pending and superseded by the guarded jobs before any sampler started.

The recovery manifest, gate checks and authoritative scheduler evidence are in the same bundle.
Original jobs remain live and unchanged; no full replacement is running concurrently.
Original attempts, initialization checks and any eventual recovery all belong to the same two numerical seeds, not additional agent outcomes or independent posterior replications.

### Recovery activation, September 13

Slurm now confirms both original tasks `22643258_0` and `22643258_1` timed out at their eight-hour limits.
Their guarded recovery jobs `22650786_0` and `22650787_1` activated after those terminal failures and are running on node1412.
They restart the same numerical seeds from the verified stage-zero populations after 64 evaluations; the original eight-hour workers did not save their later sampler populations.
The recovery allocations are twelve hours each, and original-attempt costs remain part of the total.
They are not additional independent fits or agent outcomes.
The earlier descriptions of live original jobs and pending recoveries above record the state when the safeguards were implemented.
No completed or numerically assessed Fan posterior is claimed by this update.

## Complete-population report preparation

The two full-recording recovery runs have reached saved stages 30 and 31 of 32, with their target-evaluation counters advancing.
Their configuration has 64 particles and uses all 132 recorded actions.
They are distinct from the newer prefix-only fits and the 128-particle prefix budget comparison.

The report bundle `logs/uncertainty_fan_full_fit_summary_20260913` pins the recovery plan, original runtime overlay and worker scripts.
A completed source report must have a complete sampler at temperature one, the expected seed and numerical configuration, and matching prior and inference identities.
The reporter restores the complete sampler checkpoint under the frozen implementation and requires the entire recovered result to match the source report without requesting another target evaluation.
Weighted fan-speed quantiles are independently reconstructed and must match the saved values.
The report also retains mean, standard deviation, distinct positive-weight values, largest exact-value mass, ancestry and evaluation counts.

The paired comparison requires identical complete target identities, priors, numerical configurations and recorded runtimes.
It reports the maximum difference between weighted empirical CDFs and their one-dimensional Wasserstein distance.
These are descriptive distances between dependent numerical populations, not an independent-sample hypothesis test or an automatic adequacy decision.
There are no reserved observations in this full-recording study, so marginal agreement cannot substitute for the separate prefix forecast comparison.

Original timeout and recovery allocation costs are reported separately and combined only when both accounting records are available.
Latest-attempt worker time and native actions remain separate from sampler evaluations carried in the checkpoint.
Missing or noncomplete fits remain explicit; neither is interpreted as an agent outcome.

Compute validation `22678043` completed in eleven allocation seconds on one CPU and performed no native simulation.
Known discrete distributions verify the weighted quantile convention, zero-weight exclusions, CDF distance and Wasserstein distance; invalid weights, lost mass and nonfinite values are rejected.
The initial report correctly remains incomplete while both source fits are running.
This validates the marginal calculations and incomplete-report path; completed-checkpoint recovery will be exercised when completed source fits exist.
Finite report `22678058` depends on successful validation and termination of both recovery runs.
It requests one CPU, 4 GB and ten minutes on `mit_preemptable`, without sending notifications or modifying the fits.

The full-recording seed-201 recovery `22650787_1` has completed at 15,092 target evaluations, using 30,790 allocation seconds and 30,769.80 worker seconds in this recovery attempt.
It records 1,983,960 native actions in the recovery and retains one initial ancestor.
Report `22678127` now exercises complete-checkpoint recovery successfully and independently verifies its fan-speed quantiles: [0.09034013, 0.09036351, 0.09053017].
There are eight distinct positive-weight speeds, with 0.75 of empirical mass at one exact speed value.
This concentration does not establish precise identification; the paired result remains incomplete until seed 200 finishes.
Including the original 28,812-second timeout, this seed has consumed 59,602 allocation CPU-seconds across the two attempts.
