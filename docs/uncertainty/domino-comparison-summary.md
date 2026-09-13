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

## Completed fixed-initial-state pair

Summary `22675144` additionally verifies the two complete point-start forecasts, including every positive-weight history.
Their 0.880 mm position-mean difference, 0.01215 maximum toppling-curve gap and 0.01215 final gap pass all three exploratory screens.
Their position RMSEs are 11.441 and 11.399 mm, and their toppling Brier errors are 0.0000028704 and 0.0000038637.
See the [initial-state ablation](initial-state-ablation.md) for its approximation, cost and source reports.
The local-only pair remains incomplete in this snapshot.
The next [budget comparison](domino-budget-sensitivity.md) tests each completed target at twice the particle count; no production use is approved by this initial screen.

## Completed matched local-proposal controls

Both local fits and their reserved-action forecasts have completed, so summary `22675143` now contains all six populations and all three within-protocol pairs.
The two local fits use 14,365 and 14,429 target evaluations and retain one initial ancestor each.
Their four-CPU allocation times are 5,075 and 4,279 seconds; forecasts require 86 and 70 allocation seconds and 10,465 native actions each.
The completed report verifies all 384 positive-weight complete histories across the six populations.

For each numerical seed, an additional source check confirms that the local and mixed fits have identical complete inference identities, priors and seeds.
Their numerical configurations differ only in full-range block-refresh probability: zero for the local arm and 0.5 for the mixed arm.
This closes the earlier matched-control gap; the evidence comes from two numerical replicas of one target, not repeated datasets or agent seeds.

| Forecast | Conditional position RMSE | Toppling Brier error | Final probability for domino 1 to topple |
| --- | ---: | ---: | ---: |
| Legacy point forecast | 11.062 mm | 0.0034364 | 0 |
| Local proposals, numerical seed 100 | 11.848 mm | 0.0032677 | 0.95746 |
| Local proposals, numerical seed 101 | 12.320 mm | 0.0189003 | 0 |
| Mixed proposals, numerical seed 100 | 10.906 mm | 0.0001380 | 0.93750 |
| Mixed proposals, numerical seed 101 | 10.854 mm | 0.0000596 | 0.95395 |

The local seed-101 population predicts no final toppling for any of the six objects, while the recorded final state has four toppled objects.
It assigns much larger rolling friction than the other local replica; its empirical median is 0.06193 versus 0.00326.
This is an observed association, not an isolated causal diagnosis of the prediction error.

| Replica agreement check | Local proposals | Mixed proposals | Screen limit |
| --- | ---: | ---: | ---: |
| RMS difference across conditional position means | 7.954 mm | 3.038 mm | 2.5 mm |
| Largest toppling probability gap | 1.0 | 0.18385 | 0.20 |
| Largest final-frame toppling probability gap | 1.0 | 0.01645 | 0.15 |

The local pair fails all three predeclared exploratory checks; the mixed pair passes the two toppling checks but still fails the position-mean check.
The proposal change reduces disagreement and prediction errors on this matched recording, yet does not establish numerical adequacy or justify deployment.
The ongoing 128-particle mixed and fixed-initial-state comparisons remain the next budget-sensitivity evidence.
The matching checks and source summary hash are retained in `completed-local-matching-verification.json` in the summary bundle.
