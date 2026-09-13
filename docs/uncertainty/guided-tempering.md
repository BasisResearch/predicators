# Guided proposals and the tempering sequence

The first Fan fixture guide preserves the original physical posterior but has an inefficient initial numerical distribution.
Its native initial population contains 15 supported candidates out of 64, with weight effective sample size only 2.010.
The first completed resampling stage of seed 302 retains two original ancestors.
These observations motivate changing the numerical bridge while retaining the original prior, likelihood, proposal and final target.
They do not establish that the first guided fit will fail its final prediction checks.

## Target and intermediate distributions

Let `u` denote the original canonical scene coordinates, `b(u)` their existing initial log factor, and `l(u)` the remaining prefix log likelihood.
Let `q(u)` be the complete mixture density induced by the augmented guided proposal in the original coordinate representation.
The mixture retains 10% original proposal mass, so its density is strictly positive throughout the original support.
Both constructions below retain all existing exact-observation and geometry support checks.

```text
Current bridge:      b(u) - log q(u) + beta * l(u)
Alternative bridge:  b(u) + beta * (l(u) - log q(u))
Final target:        b(u) + l(u) - log q(u), at beta = 1
```

Both final augmented-coordinate targets induce the same original physical posterior after accounting for the proposal transformation.
The alternative initial distribution is a guided numerical reference, not the original prior or the posterior after only the first observation.
It avoids immediately undoing the guide through the initial importance weights.
No guide likelihood is added as extra observation evidence, and the original physical prior is not narrowed.
The stored annealed factor now includes the finite mixture correction; a forecast reader must reconstruct the actual prefix likelihood explicitly rather than interpret that stored factor as a raw likelihood.
Intermediate distributions are numerical choices; their final density correction and exact support remain mandatory.

## Exact reference

The reference has independent standard normal priors, Gaussian observations, and analytically available Gaussian posteriors.
One additional coordinate is unobserved and must retain its original prior.
The guide equals the exact posterior for the observed coordinates and retains the same defensive mixture with the original proposal.
Each cell uses four sampler seeds, 64 particles, 32 temperatures, four moves per temperature and the same numerical budget.
The two bridges use identical original priors, observations, proposals and final target factors.

| Observed coordinates | Correction placement | Mean marginal CDF error, four fits | Mean posterior-mean RMSE, four fits | Surviving original ancestors |
| --- | --- | ---: | ---: | ---: |
| 1 | Initial base | 0.03503 | 0.07824 | 29-64 |
| 1 | Annealed factor | 0.02798 | 0.04664 | 64 |
| 30 | Initial base | 0.14675 | 0.38403 | 3-4 |
| 30 | Annealed factor | 0.03633 | 0.08177 | 64 |

CDF error averages absolute discrepancies at reference probabilities 0.05, 0.25, 0.5, 0.75 and 0.95 over every coordinate, including the uninformed coordinate.
This is an exploratory numerical comparison on exact references, not a physical-domain calibration or agent-performance result.
It does not prove that the alternative bridge is uniformly better for approximate or misleading guides.
Array `22679038` completed all 16 fits in 11 and 8 allocation seconds on one CPU per shard.
Independent verification `22679339` completed in 14 seconds, deriving posterior parameters from prior and observation precisions and recomputing every saved population's summaries.
The reference uses zero native simulator actions.
Artifacts are in `logs/uncertainty_guided_tempering_reference_20260913`.

## Native Fan preflight and comparison

Preflight `22679317` completed in 176 allocation seconds on four compute CPUs, evaluating 10,240 native actions.
It verifies the changed factorization at the same supported and unsupported physical candidates and preserves the complete final target.
The complete serial and parallel initial sampler states agree exactly.
All 15 supported initial candidates now receive equal weights, giving effective sample size 15.0 instead of 2.010 under the first bridge.
That initialization result does not establish final parameter or prediction accuracy.

Array `22679393` runs the alternative bridge with sampler seeds 302 and 303, matching the existing guided pair's proposal, 64 particles, 32 temperatures, eight moves, 16,448-evaluation limit and four-hour allocation limit.
Each fit uses four CPUs on `mit_preemptable`.
The original guided fits remain frozen as the comparison arm.
The native bundle is `logs/uncertainty_fan_tempered_guide_20260913`, with complete-stage checkpoints and a launch manifest that pins the passed preflight and exact-reference verification.
The reserved-future adapter below accounts for the correction in the annealed factor.
Neither bridge is used by the acting agent.


## Reserved-future forecast validation

The forecast bundle is `logs/uncertainty_fan_tempered_forecast_20260913`.
It uses the verified forecast-only observation extension, preserving the original density methods and fitting data.
The short numerical fixture `22680058` completed in 79 allocation seconds on four CPUs, evaluating 7,808 native actions.
Its full-trajectory forecast `22680068_0` completed in 72 allocation seconds, evaluating 8,580 native actions across all 64 positive-weight scenes and one repeated history.
This fixture validates the adapter and is not an assessed posterior or agent seed.

For every scene, the native replay checks the original initial factor against the stored base and the remaining prefix likelihood minus the complete mixture correction against the stored annealed factor.
The report retains the raw prefix likelihood, initial likelihood and canonical prior/proposal ratio separately.
The independent reader reconstructs these factors from the saved report and checkpoint while also checking original weights, complete histories, native events and goals.
Its first attempt `22680091` failed because it compared numeric factors directly with their checkpoint string encoding.
The corrected reader decodes those values and reuses the completed native artifacts; no simulator work was repeated to fix this reporting error.

Final comparison validation `22680173` passed in 19 allocation seconds on one CPU, verifying 128 original histories and rejecting eight incompatible comparison contracts.
Final native-artifact validation `22680174` passed in 20 seconds on the source node, verifying all 64 tempered histories and rejecting nine corruptions, including changed raw likelihood and initial density factors.
The corrected source hashes match both validation reports.
Incomplete-report validation `22680175` passed in nine seconds, retaining all six planned populations while only the two original forecasts are available.

Full tempered forecasts `22680182_1` and `22680183_2` depend on the passed checks and their respective completed fits.
Summary `22680197` also waits for the original guided forecasts `22679493_1` and `22679494_2`.
It compares two original, two base-corrected guided and two tempered-correction populations, retaining all 15 pairwise comparisons when available.
Original physical prior, data, program and output law remain fixed; the reader explicitly verifies the algebraic factorization change instead of requiring the numerical bridge identities to be identical.
The reporting bundle is `logs/uncertainty_fan_tempered_summary_20260913`.
No completed tempered posterior forecast or live-agent acceptance result is available yet.
