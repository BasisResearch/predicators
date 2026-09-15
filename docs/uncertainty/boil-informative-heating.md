# Boil heating: information and conditional numerical accuracy

September 14, 2026.
This diagnostic follows the [completed reduced-target comparison](validation-status-20260914.md#boil-reduced-target-outcome).
It asks whether informative heating observations constrain the thermal parameters under the existing simulator program and declared sensor noise.
It does not change the production agent or establish an adequate joint posterior over the scene and all parameters.

## Fixed conditional problem

The physical histories come from the independently verified density trajectories of the two completed reduced fits, numerical seeds 410 and 411.
Selection uses the highest fitting weight with the smallest particle index as a tie-break, independently of future prediction errors.
All nonthermal parameters and each candidate physical history remain fixed.
Only burner radius, heating onset and heating width are inferred.

The original independent uniform priors remain radius `[0.04, 0.25]`, onset `[5, 80]`, and width `[1, 40]`.
The bubbling sensor standard deviation remains 0.07.
The existing output model has no discrepancy process on bubbling, and none is added here.
The likelihood multiplies the Gaussian bubbling-reading factors through the designated prefix.
The initial reading supplies a parameter-independent constant and is explicitly omitted from the reported conditional normalizer.

The two prefixes contain 132 and 224 actions.
The first never turns the burner on and retains the exact prior-retention control.
The second includes heating observations and is a separate development case.
Clean evaluator observations and noisy observations after action 224 do not enter this conditional inference.
The source density histories condition on observed robot motion causally; the native audit changes later joint observations and verifies that every prediction through action 224 remains unchanged.
This diagnostic conditions on a fixed physical prefix, and therefore does not integrate uncertainty about that prefix or update the other fitted quantities with the additional observations.

## Integration and verification

For a fixed physical prefix, distances to the burner partition the radius prior into intervals with identical heating decisions.
The radius integral is exact on this partition.
Onset and width use product Gauss-Legendre quadrature under their original normalized priors.
The producer groups reading residuals by accumulated integer heat, while the independent reader sums each Gaussian observation residual directly.
Both methods preserve the joint onset-width likelihood; marginal summaries do not imply independent posterior parameters.

Native job `22755839` completed in 1:20, with 1,320 simulator actions.
It reproduces both complete archived native histories, predictions and carried memory in fresh worlds, and verifies prefix independence from altered later observations.
Reader `22755840` completed in six seconds, checking both prefixes for both selected candidates, the original observations, radius partitions, Gaussian likelihood probes and numerical moments.
The verified result checksum is `43edab5af557698d42eb65ff3a653484ab6b2dc4a543056e38966e4bf38ee810`.

The initial 128- and 256-node rules disagree by up to 0.174 in parameter means on the informative prefix.
That motivates a separate precision experiment at 512, 1,024 and 2,048 nodes, with absolute tolerances of 0.005 for parameter means, standard deviations and the log normalizer.
It reuses the same verified conditional targets and performs no additional native simulation.
Jobs `22756314` and `22756315` completed in 15 and 12 seconds, respectively.
The independent reader confirms the higher-order integrals through direct residual sums.
Its verified result checksum is `94e57f245aa7d86cb49ee5c071d63b6afbe0a405f833fd256137dd32d2bb79a1`.

| Comparison of quadrature orders | Maximum mean difference | Maximum standard-deviation difference | Log-normalizer difference | Declared precision screen |
|---|---:|---:|---:|---|
| 512 vs 1,024 | 0.001524 | 0.000992 | 0.008052 | Fails log-normalizer tolerance |
| 1,024 vs 2,048 | 0.000229 | 0.000163 | 0.001142 | Passes |

This is empirical quadrature stability on this conditional problem, not a certified global integration bound or acceptance of the full-scene sampler.

## Conditional outcome

The two selected candidates induce the same heating-count history and scalar conditional likelihood.
They therefore produce identical conditional results; they are not two independent posterior-replication successes.
The higher-precision worker recognizes and reuses that identical calculation explicitly.

| Parameter | 132-action mean | 132-action standard deviation | 224-action mean | 224-action standard deviation |
|---|---:|---:|---:|---:|
| Burner radius, metres | 0.14500 | 0.06062 | 0.14500 | 0.06062 |
| Heating onset, action steps | 42.50000 | 21.65064 | 30.12237 | 0.29841 |
| Heating ramp width, action steps | 20.50000 | 11.25833 | 5.71475 | 0.51485 |

The all-off prefix retains all three normalized priors.
The longer prefix strongly constrains onset and width under the unchanged program and sensor likelihood.
In these selected histories, every radius in the original support induces the same heating decisions, so radius remains uniform even after heating.
This conditional invariance does not certify radius independence across all possible scenes.

The result separates lack of informative observations from failure to compute a low-dimensional conditional distribution.
It supports testing heating-aware fitting without introducing a new bubbling-discrepancy term merely to match historical defaults.
It does not demonstrate held-out forecasting accuracy: no such score was computed here.

## Next integration requirement

A complete comparison on the longer prefix must define a new data identity and update the joint scene and parameter inference using all observations through that prefix.
The all-off independence certificate cannot be carried forward for onset and width.
Conditional thermal draws must preserve their dependence on each other and on the candidate scene and nonthermal parameters.
Do not substitute the conditional means in this table into every particle or independently sample these marginal standard deviations.
The original 132-action benchmark remains an extrapolation and prior-retention case.
Future prediction and live-agent acceptance still require their separate comparisons.

Frozen scripts and reports are in `logs/uncertainty_boil_informative_heating_20260914` and `logs/uncertainty_boil_heating_quadrature_20260914`.
