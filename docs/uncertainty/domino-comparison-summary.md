# Combined Domino prediction assessment

The [Stage B comparison](simplification-proposal.md) now has a validated report builder for the three matched Domino protocols.
It compares local proposals, mixed local/full-range proposals, and the explicit fixed-initial-state approximation on the same 64 fitted actions and 97 reserved actions.
Each protocol has two independent numerical replicas, seeds 100 and 101.
These are inference replicas on one recording, not agent solve-rate seeds.

The report requires a completed forecast and its pinned completed source fit before accepting a result row.
It verifies the program, data, output model, numerical budget and proposal setting against the declared protocol.
It checks every positive-weight complete history, its physical sample and weight against the fit snapshot, then reconstructs the reported toppling curve from the saved histories.
Dropping a positive-weight particle, changing its physical sample or altering the aggregate event curve fails validation.
Missing or failed inputs remain explicit, and a pair is not compared until both members are available.

The report includes position error, frame-averaged and per-object toppling Brier errors, final toppling probabilities, event timing mass and inference cost.
Stored truth enters this evaluator only; it does not select parameters, particles, proposals or predictions.
Frame averages are descriptive and are not treated as independent experimental samples.
The legacy point forecast remains a separate baseline, with its own initialization and fitting policy.

## Predeclared screening checks

| Difference between replicas | Maximum for the initial screen |
| --- | ---: |
| RMS difference across conditional mean xyz coordinates | 2.5 mm |
| Largest toppling probability difference across objects and reserved frames | 0.20 |
| Largest final-frame toppling probability difference | 0.15 |

These thresholds were recorded in the mixed-proposal plan before its fits completed.
Passing this screen does not establish numerical adequacy, acceptable cost or safe use in planning.
Budget sensitivity, broader recordings and the remaining stages are still required.
Allocation seconds, allocated CPUs and allocated CPU-seconds are reported separately from target-evaluation counts and recorded forecast native actions.
Interrupted or missing accounting is not silently reported as zero total cost.

## Validation and scheduled reports

Compute validation `22674965` reproduces both earlier population diagnostics exactly, verifies 64 complete saved histories and rejects three deliberately corrupted inputs.
The earlier populations fail all three screening checks: their coordinate-mean RMS difference is 6.93 mm, largest toppling probability gap is 0.90625, and final probability gap is 0.64247.
This reproduces the previously known disagreement; it is not a new result for the current fits.
The first generated current summary is explicitly incomplete because all six forecast inputs were still missing.

The frozen bundle is `logs/uncertainty_domino_comparison_summary_20260913`.
Three lightweight summary jobs are queued after their respective forecast pairs reach terminal states, with successful report-builder validation also required.
Each snapshot includes whichever other protocols have completed by then and marks the remainder incomplete.

| Protocol triggering the snapshot | Summary job |
| --- | --- |
| Mixed proposals | `22675142` |
| Local proposals | `22675143` |
| Fixed initial state | `22675144` |

This is a finite dependency chain for offline analysis, not a restored MB/MF notification monitor.

## Completed mixed-proposal prediction pair

Both mixed-proposal forecasts completed, and summary `22675142` verified all 128 positive-weight histories across the two replicas.
The local-proposal and fixed-initial-state comparisons remain incomplete in this snapshot.

| Forecast | Position RMSE to noisy readings | Toppling Brier error | Final probability for domino 1 to topple |
| --- | ---: | ---: | ---: |
| Legacy point forecast | 11.062 mm | 0.0034364 | 0 |
| Mixed proposals, numerical seed 100 | 10.906 mm | 0.0001380 | 0.9375 |
| Mixed proposals, numerical seed 101 | 10.854 mm | 0.0000596 | 0.9539 |

The recorded final outcome for domino 1 is toppled.
Position errors for the particle forecasts use their prefix-conditioned output means; the reports also retain native simulator mean errors separately.
These are descriptive results for one reserved action sequence, not agent performance or independent-dataset calibration.

| Replica agreement check | Observed difference | Threshold | Result |
| --- | ---: | ---: | --- |
| RMS across mean xyz coordinates | 3.038 mm | 2.5 mm | Fails |
| Largest toppling probability gap | 0.18385 | 0.20 | Passes |
| Largest final toppling probability gap | 0.01645 | 0.15 | Passes |

The pair still fails the complete predeclared screen.
The improved agreement relative to the earlier populations does not isolate the proposal change because the matched local controls have not completed.
The concentrated parameter values, budget sensitivity and substantial inference cost also remain unresolved.
