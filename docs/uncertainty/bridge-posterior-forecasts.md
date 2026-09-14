# Bridge forecasts from complete joint candidates

September 13, 2026.
This prepares the running [Bridge joint fits](bridge-joint-inference.md) for held-out prediction checks.
The production estimator remains unchanged, and the full posterior forecast comparison is not complete.

## Candidate preparation and causal continuation

The adapter accepts the complete 147-coordinate guided proposal, preserving its six physical parameters and 140 original scene auxiliary coordinates through the checked map.
It reevaluates the candidate's original fitting target and checks its complete joint coordinates and score against the supplied fitting record.
A candidate with unsupported fitting observations is rejected rather than removed from a weighted population and renormalized away.

The generalized generator retains the checked model and numerical operation sequence, replacing only the earlier single-witness assertion with the candidate's own complete fitted prefix.
It reconstructs all 600 fitting actions and requires exact agreement in the root, predictions, learned memory, joint-variance statistics and conditional event factors before continuing.
Future joint variances, step residuals, reach offsets and output errors use the same causal laws as the [original generator](bridge-causal-futures.md).
Future observations are available only to the separate [density evaluator](bridge-future-density.md).
The density evaluator must reconstruct that same fitting prefix and scores only the future factors.

Fixture `22707346` applies generation and density evaluation to two different supported candidates from the checked initial populations.
Each candidate is prepared from its original complete target, then its 586-action generated future and corresponding density replay are each repeated in fresh worlds.
Both candidates pass exact prefix, target, generated-history and density checks.
The fixture completed 10,688 native actions in 7:44 of measured script time.
Independent reader `22707374` completed in 1:45, verifying shared-variance draws, joint and reach factors, literal memory updates, output sampling and future-density accounting for both candidates.
These selected candidates are test inputs, not an assessed posterior population.

## Reserved assessment inputs

Preparation `22707523` completed the 586-action clean and noisy assessment suffixes, with all data kept outside fitting and generation.
It reproduces the incumbent's fifteen block-position RMSE entries against the earlier verified assessment.
The public `Bridged` geometry predicate is evaluated on complete reconstructed public poses.
It is distinct from hidden attachment state; glue occupancy and consumption are observed-feature events rather than labels for actual welds.

The clean recording satisfies `Bridged` only at action 1186, the final reserved step.
The noisy observations do not satisfy it at any reserved step.
Task-outcome predictions must therefore be assessed against the clean reference, while noisy observation errors are reported separately.
An earlier preparation attempt rejected equivalent tuple/list feature-key representations; the corrected comparison normalizes the key containers without altering measurements.

## Weighted population driver and checks

The complete-population driver retains every positive-weight fitted candidate with its original weight and complete joint state throughout each generated future.
It checks that the fit and checkpoint are complete, that their weights and samples agree, and that data, program and probability-model identities match the reserved assessment.
Its source guard rejects ten malformed or unfinished input cases before generating a future.
These forecasts assess raw completed numerical pilots; they do not bypass the production interface's requirement for an assessed posterior.

Each positive-weight candidate receives two independent banks of four complete generated futures and one separate conditional-density evaluation of the recorded suffix.
Generated tasks receive no reserved observations.
All candidate preparation and future simulation steps are counted.
Zero-density contributions remain in the weighted density mixture, with no removal or renormalization of their source mass.
The summaries include block-position errors, glue predicate probabilities, geometric goal probabilities, forecast-bank differences and complete recorded-future density.

A finite-mixture reference verifies unequal weights and a zero-density contribution; eight corruption cases reject missing, duplicated, relabeled or reweighted histories.
Native fixture `22707973` completed in 2:47 with 13,060 simulator actions, using two fixed support points with weights 0.25 and 0.75 and two draws per bank.
These chosen weights are test inputs, not posterior estimates.
Independent readers `22708064` and `22708157` completed in 3:01 and 3:02, checking every history's factors and weighted sums and repeating a generated and recorded-future path in fresh simulators.
The second reader additionally checks original checkpoint weights for full-population inputs, all reported glue metrics and a roundoff-tolerant final probability comparison.
Separate check `22708224` reconstructs the bank-difference metrics through an independent dense weighted calculation.

## Queued full comparison

Forecast jobs `22708283` and `22708284` are queued behind the corresponding full-fit verifiers `22707128` and `22707130`.
They will be followed by forecast readers `22708285` and `22708286`, including bank checks, then comparison `22708287` against the existing incumbent forecast.
Each forecast and reader has a 16-CPU, 64-GB, three-hour allocation on `node1412` in `mit_preemptable`.
The final comparison also reports fitting cost, forecast cost, lineage loss and differences between the numerical replicas.

The full fitted-population forecasts are not yet complete.
Neither completed simulation nor finite future density is sufficient for predictive acceptance.

Adapter artifacts are in `logs/uncertainty_bridge_forecast_adapter_20260913/`, its reader in `logs/uncertainty_bridge_forecast_adapter_verification_20260913/`, and reserved assessment inputs in `logs/uncertainty_bridge_forecast_assessment_20260913/`.

The weighted driver is in `logs/uncertainty_bridge_population_forecast_20260913/`; full verification and comparison use `logs/uncertainty_bridge_population_forecast_verification_v2_20260913/`.
