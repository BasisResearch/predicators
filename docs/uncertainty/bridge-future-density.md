# Bridge conditional future-density validation

September 13, 2026.
This extends the [causal future generator](bridge-causal-futures.md) with density evaluation under the same joint-motion, glue-transition and output laws.
It uses the same fixed training-prefix support point and retains all 600 fitting actions before evaluating the 586-action suffix.
This is a fixed-candidate development diagnostic, not a posterior or agent result.

## Conditioning and factorization

The generator never reads future observations.
The separate density evaluator conditions joint corrections on future joint readings and integrates the shared per-joint variance across the whole history.
Each future glue reading is conditioned through normalized Gaussian reach-interval masses; incompatible observations have zero density, and compatible intervals must give identical complete next memory and commands.
The remaining observation factors retain the output-error state conditioned on the fitting prefix.
Only suffix factors enter the reported conditional future score; fitting-prefix likelihood is not counted again.

The density adapter changes the checked conditional replay's observation horizon and records its variance state at the fitting boundary.
The permitted source edits are enumerated and checked before replay.
Every case reconstructs the original prefix exactly, including geometry, all predicted features, learned memory, event factors and joint-variance state.

## Native and independent checks

Job `22706700` completed in 4:40 with 8,318 native actions.
Two generated suffixes, the original recorded suffix and an impossible exact glue reading are each repeated in fresh worlds.
Both generated cases reproduce the generator's full physical predictions, learned memory and pre-update observations exactly and have finite density.
The impossible glue reading is rejected at action 601.

Reader `22706738` completed in 1:15.
It checks the full literal glue updates and memory, independently integrates reach probabilities, and compares the accumulated joint factors with a closed-form inverse-gamma integral conditioned on the prefix residuals.
It separately accumulates suffix scalar innovations, sensor factors and checked readouts; orientation uses its previously validated kernel with an independent rotation conversion.
The maximum individual joint-factor discrepancy is 1.367e-12 and maximum reach-interval discrepancy is 5.372e-11.

| Future observations | Joint log density | Glue log probability | Remaining output log density | Total conditional log density |
| --- | ---: | ---: | ---: | ---: |
| Generated, numerical seed 710 | 34,157.56 | 0 | 80,760.62 | 114,918.18 |
| Generated, numerical seed 711 | 34,023.62 | 0 | 80,832.95 | 114,856.57 |
| Recorded suffix | 25,671.59 | 0 | -1,821,608.70 | -1,795,937.10 |
| Impossible glue at action 601 | 57.76 before rejection | Negative infinity | Not evaluated | Negative infinity |

These log densities are tied to the declared coordinates and measurement units and are not probabilities or solve rates.
The fixed candidate assigns very low density to the recorded suffix, dominated by continuous output disagreement.
Exact glue compatibility therefore does not establish good predictions.
The paired full fits must be assessed through weighted posterior forecasts and decision-relevant errors before any production change.

Source and native artifacts are in `logs/uncertainty_bridge_future_density_20260913/`.
The independent reader is in `logs/uncertainty_bridge_future_density_verification_20260913/`.
