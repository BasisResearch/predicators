# Boil: incumbent and posterior forecast comparison contract

September 14, 2026.
This combines the [matched posterior comparison](boil-heating-comparison.md) with the incumbent's selected-point prediction on the same 224-action fitting prefix.
The incumbent fit and independent reader are complete, while replacement posterior fitting and forecasting pipelines are underway.
No performance difference is available yet.

## Required source evidence

The incumbent row requires the completed full-fitter report and its independent literal-program replay reader.
The collector verifies the program and data identities, the complete 224-action segment, unchanged fitter configuration and parameter specifications, exact repeated prediction, and all 264 carried-memory entries.
Applied parameters must match the incumbent's reported selection or its original defaults when no fitted segment survives.
The initial observation must match the shared recorded prefix.

The four posterior rows require the completed uniform/guided comparison and its complete chain of fitting, checkpoint and forecast verification.
The collector reconstructs that comparison from its verified sources before incorporating it.
It retains both numerical seeds in both arms, the original replica disagreements and all source hashes.
Missing or unsuccessful inputs remain explicitly incomplete.
Performance differences are omitted until both the verified incumbent and all four verified posterior rows are available.

## Common metrics and interpretation

Predictions are scored on steps 225 through 264, which are excluded from this fit but belong to a previously inspected development recording.
The collector recomputes the incumbent's clean and noisy feature errors, event Brier errors and final-event errors from its prediction curve.
It compares those with the corresponding posterior forecast metrics.
Differences are posterior error minus incumbent error, with every numerical run shown separately.
A negative difference means lower error on this suffix.
It does not by itself establish a causal effect of the sampler or a live-agent improvement.

The incumbent uses its existing initial-state treatment and fitting objective.
The posterior treatment integrates initial-scene uncertainty and uses the explicitly declared discrepancy model.
The control runs the full incumbent fitter but scores its selected point; it does not test the incumbent's complete interval or ensemble planning policy.
The report preserves these distinctions rather than treating all methods as interchangeable distributions.

Fitting, prediction generation, independent reading and scheduler allocation costs remain separate.
The report retains the original pending or unknown states rather than assigning zero cost to missing work.
Its posterior assessment remains unavailable after report completion until the separate numerical, predictive and live-validation requirements pass.

## Validation and jobs

The report checker exercises a perfect prediction, a miss confined to the final goal frame, known signed error differences, incorrect prefix and future boundaries, missing memory, incorrect applied parameters, changed configuration and premature completion.
The metadata fixtures test validation rules only; they are not represented as fitted or accepted posteriors.
The checker also reads the actual pending sources and confirms that it cannot report a completed comparison or performance differences.

Check `22761575` completed in sixteen seconds on `mit_preemptable`.
All eleven rejection controls pass, along with the known-error and signed-difference references.
At check time it retained the running incumbent and absent complete posterior comparison as pending, with no performance differences.
Original report job `22761576` was subsequently cancelled by the failed fitting-reader dependency before starting.
Replacement report `22762519` depends on recovered posterior collector `22762518` and verifies the already-completed incumbent reader `22761364`.
The [recovery note](boil-heating-reader-recovery.md) records the unchanged metric code, transport adapter and replacement accounting.
The same requirements are checked inside the final report process.
Frozen scripts and reports are in `logs/uncertainty_boil_heating_legacy_comparison_20260914`.
