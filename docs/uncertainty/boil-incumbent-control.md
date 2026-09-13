# Boil incumbent fitter control

September 13, 2026.
This adds the incumbent-fitter comparison needed alongside the [canonical Boil posterior forecasts](boil-canonical-forecasts.md).
The work remains in Stage B, with production behavior unchanged.

## Fixed model and comparison scope

The original noisy-sweep Boil artifacts declare no learnable parameters.
They remain incomplete-model controls, but cannot compare parameter estimators.
The canonical posterior experiment instead freezes an earlier filling/heating program with eight parameters, as described in the [historical model audit](historical-model-controls.md).
This control uses that same literal program, source recording and first 132 actions with 133 observations.
The remaining 132 actions are reserved for prediction assessment.

The archived program uses recurrent rules, while the current MB fitting path treats a simulator subclass's declared parameters as physical parameters.
Calling the old rule dispatcher would therefore change which fitter and parameter-selection policy were exercised.
The experiment represents the selected historical program through the current subclass interface before invoking the incumbent fitter.
This is an offline conversion of this particular fixed program, not a new production simulator format.

The subclass calls the original two rule functions without rewriting their dynamics.
Its declared model state retains the original latent dictionary and observable feature updates.
Only water volume, bubbling and spilled volume may be updated; physics commands are forbidden.
The original rules do not read their history argument, which the converter checks explicitly.
The generated feature values are exposed through the state readout without changing native body properties or adding physical forces.

## Native parity check

Job `22691098` completed in 31 seconds on a compute node.
It compares four complete 264-action trajectories against the archived literal-rule control: historical defaults, their independent repeat, and the lower and upper fill-rate bounds.
All observed predictions and every carried model-memory value agree exactly across the conversion.
The check executes 1,056 native actions.
It establishes parity for these complete test trajectories; a separate reader also checks the final fitted values before the control's predictions can enter the comparison.

## Incumbent fit

Job `22691175` runs the frozen `b09217bb3` incumbent fitting implementation through the same full rollout orchestrator used by subclass models.
It receives the model's eight declared parameter specifications and registry anchors, with the original noisy-sweep fit configuration.
Interval belief, fit-side noise handling and fit evidence are enabled.
Because the subclass declares model memory, the existing trajectory-preparation path retains the complete 132-action prefix.
The experiment verifies that preparation neither drops actions nor changes the supplied observation frames.
It retains the incumbent's residual scaling, robust objective, explainability trimming, optimization, identifiability reporting and parameter selection.
Parameters not selected by that policy retain their declared or registry values; a fit with no survivors is reported explicitly.

The fit requests four CPUs, 20 GB and at most eight hours on `mit_preemptable`.
Per-process completed-world telemetry records native steps, including work performed in child processes.
The main report refreshes those totals at phase boundaries, so an in-progress report's counter is not a current total.
Allocation time and unfinished work remain separate costs.

After fitting, the selected parameter values generate two fresh full-episode predictions from the public initial observation.
Those predictions must agree exactly.
Neither fitting nor generation reads the reserved future observations.

## Verification and comparison

Reader `22691200` waits for the fit to complete successfully.
It replays the selected parameters using the original native no-op simulator and literal post-step rules, independently of the converted subclass.
Every predicted observation and carried memory value must match the control's saved full trajectory.
Only after that check does it read the assessment-only future observations and calculate six feature errors and five native goal-predicate scores.

Comparison `22691239` waits for this reader and the two-posterior report `22690864`.
It requires identical fitting-data and literal-program identities, the same held-out assessment artifact, and successful independent verification of every input.
It compares the incumbent's selected-point forecast with each joint posterior's predictive mean, retaining per-feature errors and goal-relevant Brier scores.
The incumbent's interval or ensemble planning policy is not represented by that selected-point row.

This comparison does not isolate the numerical estimator alone.
The posterior arm also changes initial-state treatment and introduces explicit transition/output discrepancy; the incumbent retains its existing initialization and objective.
Those differences remain named experimental changes, not hidden adjustments to sensor noise.
The comparison does not establish calibration, planning equivalence or live-agent non-regression.
The matched ablations and later planning experiments remain requirements of the full simplification plan.

Artifacts and frozen scripts are in `logs/uncertainty_boil_legacy_subclass_control_20260913/`.
At submission, the incumbent fit and both canonical posterior fits were running; no final comparison result was available.
