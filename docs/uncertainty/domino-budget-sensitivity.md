# Domino population-size sensitivity

September 13, 2026.
This is a Stage B numerical comparison under the [simplification proposal](simplification-proposal.md), using the same development recording and frozen simulator as the completed [64-particle forecasts](domino-comparison-summary.md).
These are offline inference replicas, not agent experiments or additional independent task datasets.

## Why this comparison

The 64-particle joint fits retain one or two initial ancestors and disagree on some decision-relevant predictions.
The point-start approximation retains five or six ancestors and passes the initial prediction-agreement screens, but that does not establish convergence or trustworthy parameter uncertainty.
The completed local-scale audit shows irregular target changes near retained scenes, with no uniformly better smaller proposal scale.
The next controlled numerical change increases population size instead of changing the probability model or selecting a new proposal heuristic.

## Controlled protocol

Run both the complete joint target and the separately labeled first-observation point-start approximation with 128 particles, using numerical seeds 100 and 101 for each.
Each run starts from its original declared prior/proposal, with no reuse of a completed 64-particle population as an initialization.
The only numerical configuration changes are 64 to 128 particles and a maximum evaluation budget of 16,448 to 32,896.
Retain 32 cubic-spaced temperatures, eight moves per stage, local scale 0.05, the 50/50 local/full-range mixture and the original proposal blocks.
The joint representation has 25 blocks; the point-start representation has five.
These are independent numerical replicas within each setting; using the same seed numbers across population sizes does not make their later random schedules identical.

The data, physical program, parameter prior, initial-state policy, output model and native runtime remain fixed within each state treatment.
Worker, setup, preparation and driver files are copied byte-for-byte from their corresponding 64-particle bundles.
The new run directories contain no previous inference checkpoints.
Each new fit repeats its archived native preflight before sampling and then saves complete-stage checkpoints under the changed numerical configuration.
All four tasks have passed those preflights and started sampling on compute nodes.

| State treatment | Fit tasks | Dependent forecast tasks |
| --- | --- | --- |
| Joint uncertain initial state | `22675729_0`, `22675729_1` | `22675741_0`, `22675742_1` |
| Fixed first-observation state | `22675730_0`, `22675730_1` | `22675743_2`, `22675744_3` |

Each fit requests four CPUs, 20 GB and six hours on node1412 in `mit_preemptable`.
Each two-task array permits at most two concurrent replicas.
Forecasts retain the existing four-CPU, 20-GB, 30-minute allocation and begin only after their corresponding fit succeeds.
Runtime accounting must include actual allocation cost and repeated work after interruption, separately from the sampler's cumulative evaluation count.

## Evaluation fixed before outcomes

Use the original 64-action fitting prefix and 97-action causal suffix.
The forecast driver remains unchanged and restores the complete checkpoint, verifies the retained physical samples and prefix factors, then propagates every positive-weight particle through the full recorded history.
It repeats the first complete history and records all weights and explicit zero-weight rows.
No forecast or future observation alters the fitted population.

For each state treatment, compare both 128-particle replicas with each other and all four 64-versus-128 replica pairs.
Retain the existing thresholds of 2.5 mm for RMS differences in conditional position means, 0.20 for the largest toppling-curve gap and 0.15 for the largest final toppling gap.
Report every pair rather than selecting the best agreement.
Also inspect parameter quantiles, mass at repeated values, ancestry, position errors, toppling errors, event timing and compute cost.
Do not pool the joint and fixed-state populations or treat them as estimates of the same target.

These thresholds remain exploratory screens, not proof of calibrated uncertainty or completed Stage B acceptance.
A stable but poorly predictive result still needs its prediction failures reported.
A failure at the larger population size is evidence against relying on apparent stability at the smaller size.
Broader recordings, active carried-center coverage, the other domains and later planning gates remain necessary.
The production agent is unchanged.

Frozen bundles and submission hashes are in `logs/uncertainty_domino_budget_joint_20260913`, `logs/uncertainty_domino_budget_point_20260913` and `logs/uncertainty_domino_budget_forecast_20260913`.
These finite inference and prediction jobs do not re-enable the MB/MF notification monitor.

## Report validation and follow-up

The comparison report is prepared in `logs/uncertainty_domino_budget_summary_20260913`.
It enumerates all six pairs among the two 64-particle and two 128-particle replicas for each state treatment, producing twelve pairs total, including eight cross-budget pairs.
Missing or unsuccessful forecasts remain explicit and cannot produce a completed pair or a passed screen.
The paired checks require the same complete inference identity and prior, allowing only the declared particle-count and evaluation-budget differences in sampler configuration.
Joint-state and point-state targets are never compared as estimates of the same distribution.

Each completed row verifies its source fit, numerical configuration, numerical seed, complete checkpoint provenance, unchanged weights and all saved positive-weight histories using the previously validated history checker.
Its parameter report independently reconstructs the saved empirical 5th, 50th and 95th percentiles and records the largest mass at any exact retained value.
This prevents coincident quantiles from silently being described as precise parameter identification.
Cost fields retain scheduler accounting, latest-attempt fitting seconds and forecast native actions separately, with interrupted-work limitations explicit.

Compute job `22675912` tests reproduction of the four completed 64-particle forecasts and both existing replica screens before the new results arrive.
It also exercises rejection of mismatched priors, runtimes, proposal settings, state treatments, actual fit data identities and a 64-particle population merely relabeled as 128 particles.
These are analysis checks, not new physical fits or agent results.

The validation completed successfully in job `22675912`, using one CPU and nine allocation seconds with no native simulation.
All four reference forecasts, both existing replica comparisons and 256 complete saved histories reproduce exactly; all six deliberately incomparable inputs are rejected.
The initial summary is explicitly incomplete, with four completed rows and two completed pairs; ten pairs still require the larger fits and forecasts.
Finite snapshot jobs `22675954` and `22675955` depend on validated reporting and termination of the joint and point-state forecast pairs, respectively.
Each snapshot reports every available result while retaining missing or failed inputs explicitly.
These analysis dependencies do not create result notifications in this task.

The baseline parameter diagnostics also show why prediction agreement alone is insufficient.
For the point-start replicas, median lateral friction is 0.2734 versus 0.4649 and median rolling friction is 0.003994 versus 0.001571, despite passing the reserved-sequence prediction screens.
Their largest exact-value masses reach 33.84% and 56.02%, respectively, across the five reported parameters.
This does not prove that the parameter distributions are wrong, but it prevents interpreting agreement on one action sequence as established uncertainty over other plans.
The comparison retains all empirical quantiles and concentration measures for the larger-population check.


## Completed 128-particle results

All four fits and their forecasts have completed.
The final report `logs/uncertainty_domino_budget_summary_20260913/summary-22675955.json` retains all eight populations and all twelve within-target pairs.
Its existing history verifier checks all 768 positive-weight saved histories across the 64- and 128-particle settings.
Independent scalar verification `22680987` recomputes every reported parameter quantile, exact-value concentration and pairwise prediction difference from the saved populations and forecasts.
It confirms the report checksum `74b778eae972534e83bb9397951f8ef82dc25fa84b7604516f5a7f9937a67612`.
The initial joint-only snapshot and verification remain archived separately.

| State treatment | Numerical seed | Conditional position RMSE | Toppling Brier error | Final domino 1 toppling probability | Surviving initial ancestors | Target evaluations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint, 128 particles | 100 | 11.534 mm | 0.001360 | 0.8944 | 1 | 30,889 |
| Joint, 128 particles | 101 | 10.953 mm | 0.001072 | 0.8899 | 1 | 30,760 |
| Point start, 128 particles | 100 | 11.401 mm | 0.0002245 | 0.6981 | 10 | 31,907 |
| Point start, 128 particles | 101 | 11.427 mm | 0.00003125 | 0.8672 | 14 | 31,910 |

These are predictions on the fixed 97-action suffix, not solve rates or additional agent seeds.
The reference suffix records domino 1 as toppled at its final frame.
Individual forecast errors on this one suffix do not determine which approximation is closer to the full posterior.

| Replica pair | Position-mean RMS difference | Maximum toppling-curve gap | Maximum final toppling gap | Passed screens |
| --- | ---: | ---: | ---: | --- |
| Joint, 64 particles | 3.038 mm | 0.18385 | 0.01645 | Curve and final |
| Joint, 128 particles | 4.580 mm | 0.20429 | 0.02236 | Final only |
| Point start, 64 particles | 0.880 mm | 0.01215 | 0.01215 | All three |
| Point start, 128 particles | 0.353 mm | 0.17530 | 0.16910 | Position and curve |

None of the six joint-state replica pairs passes all three screens.
Of the four joint cross-budget pairs, position-mean differences range from 1.631 to 5.102 mm and toppling-curve gaps range from 0.19493 to 0.37369.
Every joint pair passes the final-toppling screen, but agreement at the last frame does not remove the disagreement in preceding predictions.

Three of the six point-state pairs pass all screens: the original 64-particle pair and its two comparisons with 128-particle seed 101.
Both comparisons with 128-particle seed 100 fail the toppling-curve and final-toppling screens, with final gaps of 0.27208 and 0.25993.
The 128-particle point-state pair itself fails the final-toppling screen.
The smaller-budget point-state agreement therefore does not justify treating its uncertainty as numerically established.
Removing uncertain initial states is not a validated shortcut around the sampling problem.

Parameter concentration remains material in the joint fits.
Their median lateral frictions are 0.34490 and 0.08197, and the second fit places 97.72% of rolling-friction mass on one retained value.
Its rolling-friction 5th, 50th and 95th percentiles all coincide at 0.00343717.
These empirical quantiles must not be interpreted as precise identification without numerical validation.
The point-state fits retain more ancestry and distinct parameter values, but their decision-relevant probabilities still change across numerical budgets.

The joint fits used 3:01:04 and 3:00:02 of allocation time on four CPUs; the point-state fits used 3:06:44 and 3:08:55 on four CPUs.
Each forecast evaluated 20,769 native actions, including the repeated first history, and used 129-133 allocation seconds on four CPUs.
The final existing report took 31 seconds on one CPU; the independent scalar check took two seconds and zero native actions.
Accounting includes the fit allocation rather than only the last sampling stage.

This closes the planned 64-versus-128 comparison, not Stage B.
Doubling population size did not resolve the declared stability failures, and the evidence does not support publishing either state treatment as an assessed physical posterior.
The next numerical investigation must address exploration of parameter and state tradeoffs, rather than assume initial-state uncertainty alone explains the problem or that another population increase will suffice.
The incumbent agent remains unchanged.
