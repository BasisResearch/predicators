# Reduced Boil inference and forecasts with independent heating priors

September 14, 2026.
This follows the [independent-prior representation](independent-prior-factors.md) and its exact reduction of the all-burners-off Boil fitting target.
The work remains an offline Stage B comparison; the production estimator is unchanged.

## Fitting validation

Both 81-coordinate fitting fixtures completed successfully, using 219 and 180 target evaluations for numerical seeds 410 and 411.
Independent readers `22717219` and `22717221` replay the complete numerical ledgers and checkpoints and freshly verify all 32 final native targets per fixture, using 4,224 native actions each.
The verified source checksums are `8490860cad3bb895298296e523cb171ac3f737b56d931679107306bb322fc23f` and `e4b1360238f326cb6c990ceab3f3ac1c79d665a770da0d6739310b0ad6c826b7`.
These deliberately small two-temperature fixtures establish implementation consistency, not posterior adequacy.

Full fits `22717256` and `22717258` completed on compute node `node1411` in `mit_preemptable`.
They retain the original 32 particles, 64 temperatures, eight moves and 20,000-evaluation cap, with the three independent singleton blocks removed.
Their independent readers `22717257` and `22717259` completed successfully.
The reports retain an unavailable numerical assessment until replication and budget-stability requirements are met.

## Restoring the independent priors in forecasts

The frozen adapter in `logs/uncertainty_boil_reduced_forecasts_20260914` first reconstructs each complete reduced checkpoint without performing any additional fitting.
It verifies the expected target, configuration, seed, complete retained coordinate schema, factorization declaration and matching fit reader.
Each positive-weight retained joint row keeps its original weight and correlations.

For each generated trajectory, the adapter draws burner radius, onset and width independently from their original normalized uniform priors.
A separate deterministic random seed identifies that thermal draw.
The resulting complete parameter vector stays fixed throughout the trajectory, including its fitting prefix and future continuation.
The remaining parameter and scene coordinates are copied from one retained joint row.
No internal midpoint used during reduced fitting is interpreted as an inferred heating parameter.
The original physical continuation law and future simulator random-seed convention are retained.

Full forecasts use two generation banks with four continuations per positive-weight particle in each bank.
Each continuation therefore includes both the retained uncertainty and a thermal prior draw.
Output means, total variances and event probabilities include their combined variability.
Generation does not receive reserved future observations.

Future-data density is assessed separately, with eight independent thermal prior draws per retained particle.
The adapter averages densities over those draws before mixing particles according to their original weights.
Zero-density contributions stay in the denominator.
The reported log mixture is the logarithm of a Monte Carlo mean density; it is neither an exact integral nor an unbiased log-density estimate.
A particle whose sampled density draws are all zero is not thereby proved to have zero density under its full continuous prior.
The fixture uses two density draws per particle and one continuation per generation bank solely to test the implementation.

## Independent checks

Compute-node guard job `22717364` passes twelve source-corruption checks and ten malformed-history checks.
These include changed factor bounds, stale factorization scope, incomplete checkpoints, altered weights, missing or duplicated density draws, and negative conditional variances.
A separate nonuniform-weight reference verifies the total-variance calculation and averaging denominator when three of four prior-density draws contribute zero.

Native forecast fixture `22717373` completed all 128 histories, using 40,656 native simulator actions.
Independent reader `22717375` also passes: all 128 histories, 304,128 joint factors, independent thermal-prior density and weighted-summary checks, and two fresh complete histories.
It uses 6,864 additional native actions.
The verified forecast source checksum is `bbbb8324640aaf3a348a6888e4e21e6de44a539c8698477b2e489c023192bd35`.
The reader reconstructs the 81-to-84 coordinate mapping separately, regenerates thermal draws from their recorded seeds, checks every native history and readout, and repeats complete generation and density histories.
It independently computes the thermal density average and weighted prediction summaries.
Full forecasts `22717437` and `22717439`, independent readers `22717438` and `22717440`, and final comparison `22717441` completed successfully.
The [completed comparison](validation-status-20260914.md#boil-reduced-target-outcome) records the results and remaining numerical and predictive failures.

## Comparison and remaining acceptance

The comparison bundle `logs/uncertainty_boil_reduced_comparison_20260914` retains all existing Boil comparison rows, including the original full supported fits and the incumbent.
New reduced-target rows must match the original program, fitting observations and sensor model, with the declared full prior preserved by factorization.
It checks that the only fitting configuration difference from the full supported fits is removal and remapping of the three singleton proposal blocks.
The dimension change also changes random streams and is not described as identical stochastic execution.

The comparison reports fitting and forecast costs, between-fit prediction differences, within-forecast bank differences, and the explicitly approximate density calculation.
Comparison fixture `22717399` completed its checks and retained all seven original rows while correctly reporting both new full comparisons as pending.
Missing results remain pending and cannot support an improvement claim.
A completed mechanical comparison would still require numerical and predictive adequacy before live planning integration or retirement of the incumbent.
A separate [conditional diagnostic](boil-informative-heating.md) using a 224-action prefix containing heating completed as `22755839`, with independent reader `22755840` and higher-precision follow-up `22756314`/`22756315`.
It holds selected physical histories and nonthermal parameters fixed and does not replace either the full-scene inference comparison or the original all-off extrapolation benchmark.
