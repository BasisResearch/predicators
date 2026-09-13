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
Actual accepted-center coverage and any effects remain pending these runs.
The earlier inactive controls are retained separately, and the production estimator is unchanged.
