# Isolating carried prior centers

The [Stage B plan](simplification-proposal.md) requires separating the effects of fixed-prior fitting and uncertain initial-state inference.
The existing cold legacy comparisons contain no previous fit, so they cannot exercise carrying.
This experiment isolates carrying within the legacy fitter, separately from the new joint sampler and the initial-state ablation.

The implemented legacy policy carries accepted fitted parameter values into subsequent prior centers.
It does not carry a full posterior density or its covariance.
An anchored fallback does not update the stored center, and an earlier accepted center can remain stored across later fits that do not replace it.
The experiment calls the actual `fit_prior_anchors` and `note_carried_posterior` methods to preserve these semantics.

## Paired protocol

Use the same frozen Fan and Domino programs, latest parameter declarations, physical runtime, noise settings, trajectory preparation and legacy width/selection rules as the completed cold comparisons.
One arm retains registry prior centers; the other enables `code_sim_learning_carry_posterior`.
Both begin without carried history.
The fitting prefix grows from 64 to 96 actions, repeats the identical 96-action fit, then uses the complete recording.
Full recordings contain 132 Fan actions and 161 Domino actions.
Predictions after each fit use its selected values, and actions beyond that fit's prefix remain unused by fitting.
The final full-recording fit has no reserved suffix.

Both arms use fixed declared optimizer initial values and pass their own previously selected values to the same hold policy, with no parameter declaration edits.
Neither uses a fit cache or explainability cache, so the repeated-data stage actually reruns fitting.
Neither adds a cross-cycle consistency adjustment.
These choices isolate the carried-center policy; they are not a reconstruction of a complete agent conversation or every production publication path.
The parameter prior family and initial-state treatment remain those of the legacy fitter.
In particular, the fixed-center arm is not the new fixed-prior Bayesian estimator.

The first 64-action fit must exactly reproduce the archived cold fit's fitted values, selected values and diagnostic report before later stages proceed.
The original observed frames must remain unchanged across preparation, fitting and prediction.
Each stage records its data identity, entering anchors, carried values before and after fitting, selected parameters, legacy widths, full predictions and native computation cost.
The repeated 96-action stages share an identical data identity.

## Interpretation and execution

The comparison checks whether carrying changes applied parameters, diagnostic widths or predictions, including after a repeated fit with no new evidence.
Legacy diagnostic widths are not newly asserted to be credible intervals.
If neither domain carries an accepted center into a later fit, retain that result as an inactive-policy control; it cannot establish that removing active carrying is harmless.
Active-policy coverage and the remaining five-domain inference comparisons would still be needed.

The frozen bundle is `logs/uncertainty_carried_prior_comparison_20260913`.
Array `22674821` is submitted to `mit_preemptable`, with six CPUs, 16 GB and a two-hour allocation limit per task on node1412, allowing two concurrent tasks.

| Task | Domain | Carry accepted centers |
| --- | --- | --- |
| `22674821_0` | Fan | Off |
| `22674821_1` | Fan | On |
| `22674821_2` | Domino | Off |
| `22674821_3` | Domino | On |

The preceding array `22674643` failed its configuration guard before fitting because the guard compared runtime tuples with JSON lists.
The serialized configurations were verified identical; the corrected guard normalizes representation before comparison.
The failed reports and original driver are retained, and those setup outcomes are not agent failures.
Both Fan and Domino pairs have completed successfully.

## Completed Fan control

Both Fan arms reproduce the archived cold fit exactly and produce identical selected parameters, diagnostic widths and complete prediction histories at every stage.

| Fitted actions | Selected fan speed in both arms | Legacy diagnostic width | Carry active before fit |
| --- | ---: | ---: | --- |
| 64 | 0.0846 | 0.1 | No |
| 96 | 0.0846 | 0.1 | No |
| 96, repeated data | 0.0846 | 0.1 | No |
| 132 | 0.0846 | 0.1 | No |

Every verdict is anchored, so no accepted fitted center enters a subsequent prior.
The repeated 96-action fits change neither the selected value nor the prediction history in either arm.
Each arm uses 11,332 recorded native simulator actions across the four fits and their predictions.
The paired comparison and source hashes are retained in `fan-comparison.json` in the bundle.
This is an inactive-policy control and does not establish that removing active carrying preserves performance.

## Completed Domino control

Both Domino arms reproduce their archived cold fit exactly and produce identical selected values, diagnostic widths and complete prediction histories at all four stages.
No accepted center is carried into a later fit.
Selected lateral friction remains 0.674, restitution 0.02, rolling friction 0.006, spinning friction 0.5 and mass 0.1.
The repeated 96-action stage changes neither selected values, widths nor predictions in either arm.
Each arm records 59,252 native actions across the four fits and predictions.
The paired report and source hashes are retained in `domino-comparison.json` in the frozen bundle.

Together, these two recordings cover only inactive carrying.
They do not establish whether removing accepted-center feedback changes fitting or repeated-data confidence when the policy is active.
That comparison requires additional development experience with an accepted fitted center, retaining this inactive result instead of replacing it.

## Earlier-program repeated-data comparison

A targeted inspection of the existing Domino seed-0 training log identifies an accepted fit followed by carrying: the first full 161-action fit moves lateral friction from its declared 0.3 starting value, and the subsequent fit carries 0.674.
The saved `cycle_000_vers_001_simulator.py` has declared friction 0.3, whereas the latest `cycle_000_vers_002_simulator.py` used above already declares 0.674.
An AST comparison confirms that the only executable difference between those two saved artifacts is this declared initial value; documentation strings also differ.
The saved files are used unchanged, including their original generated declarations.
This selects a candidate active-policy case from training experience, not from test-level outcomes.
It does not guarantee that every historical fit detail will reproduce on the controlled runtime.

The earlier-program comparison uses the same complete 161-action training recording three times, starting both arms with empty carried history.
Both arms retain the fixed earlier program, same data, parameter bounds, optimizer configuration, preparation, ordinary held-value policy and prediction path.
Only accepted-center carrying is toggled.
The first full-data fits must match exactly across arms in entering anchors, fitted and selected values, diagnostic reports and complete predictions before a difference in later fits can be attributed to carrying.
All three stages must have identical data identities.
There are no held-out actions in this repeated-data diagnostic, so its predictions are in-sample replay and cannot establish predictive improvement.
Legacy widths remain diagnostic quantities, not newly validated credible intervals.

| Arm | Compute task | Initial declared friction | Fitting actions by stage |
| --- | --- | ---: | --- |
| Fixed registry centers | `22675807_0` | 0.3 | 161, 161, 161 |
| Carry accepted centers | `22675807_1` | 0.3 | 161, 161, 161 |

The array requests six CPUs, 16 GB and two hours per task on node1412 in `mit_preemptable`.
A finite report job, `22675822`, depends on both fits completing successfully.
The paired validator passes a repetition fixture derived from the previous inactive reports and rejects altered first-fit values, an altered first-fit verdict and changed repeated data.
That fixture validates report guards only; it is not another physical fit or evidence of active carrying.
The run, source-selection rationale, AST comparison, validation report and submission hashes are frozen in `logs/uncertainty_carried_prior_active_20260913`.
The initial fit now reproduces an accepted friction estimate, but the first comparison exposed an additional reference-lifetime effect described below.
The earlier inactive controls are retained separately, and the production estimator is unchanged.

## Mutable reference defaults and the corrected isolated comparison

Task `22675807_0` completes its first fit and selects friction 0.6739569455671225 with a weakly identified verdict.
It then fails the entering-anchor guard before the second fit.
This is a diagnostic harness failure, not an agent failure or a completed repeated-data comparison.
The completed first-stage report is retained; the paired summary `22675822` was cancelled because its required off-arm success can no longer occur.
Task `22675807_1` retains the original reference-lifetime protocol and remains a separate diagnostic, not the on arm of the corrected pair.

The failure is caused by a concrete behavior of the frozen subclass interface.
`PyBulletEnv._agent_param_info()` returns current `_agent_param_values` as the registry defaults.
Applying selected parameters to the reused reference changes those defaults, and `physical_param_anchors()` reads them on the next fit.
Explicit carrying can therefore be disabled while this reference-lifetime path still feeds fitted values into later prior centers.
The prior inactive controls did not expose the distinction because their selected values stayed at their initial defaults.

Direct compute audit `22676108` reproduces this through the actual frozen subclass, `physical_param_anchors()` and `fit_prior_anchors()` methods.
With the carry flag off, the anchor changes from 0.3 to 0.6739569455671225 after applying the first fit to that reference.
A separate unfitted reference retains exactly the original anchors.
The audit uses zero native actions and completes in 26 allocation seconds on one CPU.
Its preceding attempt `22676089` failed during setup because a local list reused the factory's counter name; the original script and error remain preserved separately.

The corrected comparison keeps the reference used to obtain registry centers at its original declared values.
Predictions still use fresh rollout worlds with explicitly selected parameters, and both arms retain their normal previously applied held values.
Only the carrying-enabled arm can replace a prior center through the actual accepted-center policy.
This isolates explicit carrying; it does not assert that simply turning off the historical flag reproduces fixed-prior fitting in every production path.
It also leaves the historical production registry behavior unchanged while the replacement remains under evaluation.

| Corrected arm | Task | Reference policy |
| --- | --- | --- |
| Fixed centers | `22676127_0` | Unfitted reference, no carried centers |
| Explicit carrying | `22676127_1` | Unfitted reference, accepted centers overlaid by the existing carry method |

Both tasks repeat the same 161-action recording three times with the earlier frozen program.
Each must reproduce the saved completed first fit exactly before advancing, including its data, anchors, fitted values, applied values and diagnostic report.
The paired report additionally requires exact first-fit agreement between arms and unchanged data across every repetition.
Its validator accepts the declared repetition fixture and rejects changed reference policies, failed first-fit reference checks and changed first-fit values.
These report checks are not physical inference results.

Array `22676127` depends on successful native metadata validation and requests six CPUs, 16 GB and two hours per task on node1412 in `mit_preemptable`.
Finite report `22676128` depends on successful completion of both tasks.
The corrected frozen bundle is `logs/uncertainty_carried_prior_isolated_20260913`; source hashes, setup failures and the original-reference diagnostic remain separate.
The probability model for the new inference method already uses an explicit immutable prior identity; this experiment checks the legacy feedback mechanisms it is meant to replace.

## Completed original reused-reference carrying arm

The original carrying task `22675807_1` completed all three identical-data fits in 1,211 allocation seconds and 1,187.65 worker seconds, recording 100,803 native simulator actions.
These are simulation actions used by fitting and replay, not new environment interactions or agent solve-rate results.
All selected parameter values and complete prediction histories are exactly unchanged across the three stages.
Its initial fit matches the completed first stage of the interrupted off arm exactly.

| Repeated-data fit | Entering friction anchor | Selected friction | Legacy width | Verdict |
| --- | ---: | ---: | ---: | --- |
| First | 0.3 | 0.6739569 | 0.3449425 | Weakly identified |
| Second | 0.6739569 | 0.6739569 | 0.4926973 | Anchored |
| Third | 0.6739569 | 0.6739569 | 0.4926973 | Anchored |

The width changes after refitting identical evidence and then remains stable; this case does not show progressive narrowing.
It does not isolate explicit carrying from mutable reference defaults and cannot establish a change in decisions or performance.
The verification and source hashes are retained in `reused-reference-diagnostic.json` in the original active-comparison bundle.
The corrected isolated pair remains necessary.

Both corrected tasks have now completed their first fit and passed the exact saved-reference check, with entering friction anchor 0.3 and width 0.3449425.
Their repeated-data stages are running; the isolated paired outcome is not yet complete.
