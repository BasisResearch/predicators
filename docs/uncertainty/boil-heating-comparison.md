# Boil: matched heating-aware comparison

September 14, 2026.
This collects the [heating-aware fits](boil-heating-joint-target.md) and [224/40-action forecasts](boil-heating-forecasts.md) into a matched numerical comparison.
The fits and forecasts remain pending; there is no new posterior or agent-performance result.

## Matched sampler comparison

The collector requires complete forecast reports, their independent readers, and the corresponding verified fitting checkpoints.
It checks that uniform and guided treatments use the same physical prior, program, observations, noise model, parameter names and numerical budget.
Different proposal and runtime identities are expected because the guided treatment uses its corrected thermal proposal.
Numerical seeds 410 and 411 remain separate within each arm.

The report recomputes feature errors and event scores from the complete forecast curves rather than trusting only headline summaries.
It reports final-event squared error separately from average Brier error: missing a goal that holds only at the final frame should not be obscured by the preceding 39 frames.
It preserves weighted parameter means, standard deviations and covariance, numerical ancestry, effective sample sizes, future-density failures, and disagreement both between sampler replicas and between forecast banks.
Those parameter summaries are numerical approximations, not calibrated confidence intervals.

Fitting, forecast generation, verification and development setup costs are reported separately.
Scheduler allocation seconds and allocated CPU seconds supplement the instrumented simulator counts and reported wall times.
Allocated CPU seconds are resource allocations, not measured CPU utilization.
The report makes no amortized production-cost claim.

Missing reports stay incomplete, and unsuccessful forecast reports are identified separately from missing results.
Neither is counted as an agent failure.
Completion of this collector does not make the posterior assessment available.
Cross-budget stability, a matched legacy comparison, fresh untouched predictive evaluation, adequate evidence across all five domains and subsequent live validation remain outstanding requirements.

## Collector validation

The initial check `22761047` failed when a gap calculation subtracted Boolean native goal indicators.
The failure happened in the report checker before any fitting results were compared.
Its dependent collector `22761048` was cancelled.
The original inputs are retained unchanged.

The separately frozen v2 converts event indicators to numeric probabilities before computing differences.
Check `22761384` exercises known feature errors, a rare final-step goal miss, nonuniform weighted covariance, mismatched treatment identities and budgets, malformed metrics and the four actually pending forecast sources.
Full collector `22761385` depends on that check and all four complete forecast readers.
The check completed in seventeen seconds and passes all ten rejection controls, the known-error and covariance references, and explicit handling of the four pending forecasts.
The full collector remains dependency-queued.

## Matched incumbent prefix

A separate control runs the full frozen incumbent subclass fitter on all 225 observations and 224 actions.
It uses the same literal historical program as the new joint target.
Its original fitting objective, observation-based initial state, parameter-selection rules and uncertainty settings remain in place.
Before fitting, the control requires exact equality of its configuration and stamped parameter specifications with the earlier verified incumbent control.
This prevents the longer data prefix from concealing another fitter configuration change.

The subclass conversion preflight `22761362` completed in 32 seconds with 1,056 simulator actions.
All four complete archived trajectories and their hidden feature memory reproduce exactly.
Fit `22761363` and independent reader `22761364` follow that preflight.
The fit is running, and its configuration and parameter-specification equality checks have passed.
The independent reader remains dependency-queued.

The control replays the selected parameters from the initial observation over the entire action sequence, repeats that prediction in a fresh world, and evaluates only steps 225 through 264 afterward.
The reader reconstructs the prediction and memory using the literal program independently of the subclass wrapper.
Future observations do not enter fitting or prediction generation.

This is a matched-data selected-point prediction control.
It does not evaluate the incumbent's complete interval or ensemble planning policy.
The new joint inference also changes initial-state treatment and declares discrepancy laws, so a difference between these methods is not a sampler-only effect.
The old 132-action incumbent result remains a separate historical control.
The [combined incumbent/posterior report](boil-heating-legacy-comparison.md) is queued behind the complete independent readers and the matched posterior collector.
It will retain all four posterior rows and omit performance differences while either side is incomplete.

Frozen bundles are `logs/uncertainty_boil_heating_comparison_v2_20260914` and `logs/uncertainty_boil_heating_legacy_control_20260914`.
The initial failed collector inputs remain in `logs/uncertainty_boil_heating_comparison_20260914`.
