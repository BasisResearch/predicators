# Fan speed ambiguity and reflection diagnostic

This is Stage B development work under the [simplification proposal](simplification-proposal.md).
It investigates disagreement between the two completed [tempered Fan populations](guided-tempering.md), without changing the program, prior, data or production estimator.
All results here are offline inference diagnostics, not agent solve-rate seeds.

## Conditional speed profiles

The diagnostic selects the highest-weight particle from each population before inspecting the speed profiles: seed 302 particle 53 and seed 303 particle 61.
For each selected scene it varies only `fan_speed`, holding all other 119 proposal coordinates fixed.
The 29 distinct speed settings per scene include its original speed, and each original trajectory is repeated once.
This produces 60 histories with 3,840 native actions.

Native job `22710907` and independent reader `22710919` pass.
Both original 65-frame prefixes reproduce exactly, all other scene coordinates and initial observations remain unchanged, and the independent scalar likelihood reference agrees within `2.8422e-14`.
The frozen reports are in `logs/uncertainty_fan_speed_profiles_v2_20260913`.
An earlier tuple/list comparison failure is preserved separately; canonicalized saved trajectories match exactly.

At both fixed scenes, every tested pair `v` and `1-v` has exactly the same prefix log likelihood.
Ten reflected pairs are checked per scene, spanning speeds 0.025 through 0.975.
For example, 0.0725 and 0.9275 yield the same score, as do 0.12 and 0.88.
The original high-speed candidate, 0.913752, scores only 0.0565 log units below the best low-speed grid point, 0.085, in its own scene.
Thus the high-speed branch is a plausible explanation of this prefix under the declared broad prior, rather than an obviously bad numerical point.
These are conditional profiles at two scenes, not marginalized posterior probabilities or a proof of universal symmetry.
The physical cause of the reflection relation has not yet been independently established.

The independent reader also recomputes complete-future likelihoods and exact-event mismatches for all 128 original saved histories.
All 64 histories of seed 302 and 63 of seed 303 have zero complete-future density, attributable to disagreements in exact fan, switch or target indicators.
This explains the zero scores; it does not establish that changing the parameter sampler alone will fix the forecasts.

## Full-population diagnostic

The next audit replays both original and reflected speeds for every positive-weight particle in both populations, holding the remaining coordinates fixed.
It compares all 133 original frames against their saved histories before interpreting the reflected predictions.
The bounded workload is 128 pairs and 33,792 native actions, on four CPUs in `mit_preemptable`.

For each original joint proposal coordinate `u`, define the reflection `R(u)` by replacing `u[0]` with `1-u[0]`.
The full fitting target in these coordinates is the prefix log likelihood plus the original-prior/proposal log correction.
The diagnostic computes `a = min(1, exp(log_target(R(u)) - log_target(u)))` from that target alone.
It then assigns weights `w * (1 - a/2)` and `w * a/2` to the original and reflected histories.
This is the expectation of one lazy symmetric Metropolis transition, with reflection proposed half the time.
The transformation preserves the target as a stationary distribution, including when the two target values differ.
It does not imply that the original approximation or the transformed population has converged.
Future observations do not enter these weights.

The independent reader checks source identities, full original histories, unchanged scene coordinates, prefix and future likelihoods, geometric goal curves, normalized transition weights and an asymmetric two-state detailed-balance reference.
It reports changes in replica disagreement for position, event and goal predictions, while retaining all zero-future-density weight.
The frozen corrected bundle is `logs/uncertainty_fan_reflection_v2_20260913`, with native job `22711267` and dependent reader `22711276`.
The earlier job `22711251` stopped on a source-report field mismatch after matching original native histories; its failed bundle is retained separately.
The correction checks the stored first goal step and independently reconstructs the complete goal curve.

## Completed full-population result

Native job `22711267` completes in 3:06 and independent reader `22711276` completes in 1:25.
All 128 original complete histories reproduce exactly.
All 128 reflected candidates have exactly the same fitting-target score as their originals, so the transition splits each original weight equally between its two alternatives.
Ninety-four pairs have exactly equal observed prefixes, and 23 have exactly equal observed futures.
Bitwise trajectory differences in the remaining pairs do not establish a practically different forecast.

The follow-up saved-history audit `22711446` checks those differences directly, without additional simulation.
Every pair has exactly the same future event indicators and geometric goal curve.
The maximum future ball-coordinate difference is `1.1098e-6` m, far below the 0.005 m position-noise scale.
Some robot readouts differ more, with the largest joint-coordinate difference approximately `2.4921e-4`.
Thus this test establishes speed ambiguity but does not identify it as the cause of the population-level prediction disagreement.

| Difference between numerical replicas | Original populations | After reflection transition |
|---|---:|---:|
| Native mean-position RMS gap | 0.00669314753 m | 0.00669314772 m |
| Maximum goal-probability gap | 0.3710034 | 0.3710034 |
| Maximum event-probability gap | 0.4991637 | 0.4991637 |

The independently reconstructed original mean-position, event and goal curves agree with the previously published source reports within `8.8818e-16`.
The final goal probabilities remain 0.8531 and 0.5456 for seeds 302 and 303, respectively.
The zero-future-density masses remain one and approximately 0.9781.
These probabilities concern predictions of the recorded suffix, not agent solve rates.

This rules out a one-step correction of speed-branch representation as a remedy for the measured forecast disagreement in these two populations.
It does not establish adequate scene exploration, a universally prediction-equivalent parameterization, or a converged posterior.
The follow-up [fixed-parameter scene exchanges](fan-static-fixtures.md) locate the remaining event and goal sensitivity in fixture poses and evaluate a separate static-fixture discrepancy law.
No reflection option is added to the production agent or sampler on the strength of this negative result.
