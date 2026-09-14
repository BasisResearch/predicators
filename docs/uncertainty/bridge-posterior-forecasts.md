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

## Remaining population comparison

After the adapter reader passes, the complete-population driver must retain every positive-weight fitted candidate, its original weight and one complete joint state throughout each generated future.
Repeated future draws should be separated into independent banks so forecast sampling variability can be distinguished from differences between the two fitted populations.
The recorded-future density must retain zero-support contributions and combine candidates using their original weights.
Final assessment must include block-position errors, glue-state events, geometric goal probabilities, numerical replica differences and actual inference/forecast cost, alongside the existing incumbent comparison.
Neither completed simulation nor a finite future density is sufficient for predictive acceptance.

Adapter artifacts are in `logs/uncertainty_bridge_forecast_adapter_20260913/`, its reader in `logs/uncertainty_bridge_forecast_adapter_verification_20260913/`, and reserved assessment inputs in `logs/uncertainty_bridge_forecast_assessment_20260913/`.
