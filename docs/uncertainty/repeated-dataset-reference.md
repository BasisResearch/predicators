# Repeated-dataset posterior reference

This study addresses the repeated-dataset parameter-coverage requirement in the [simplification proposal](simplification-proposal.md).
Earlier reference studies checked repeated numerical runs on fixed datasets.
This study instead generates independent datasets from a declared prior and observation law, then compares the candidate sampler with the exact posterior for each dataset.
It is a statistical implementation reference, not evidence of physical-domain calibration or an agent performance result.

## Fixed protocol

The frozen bundle is `logs/uncertainty_repeated_dataset_reference_20260913`, using source commit `748ca519b` without an offline overlay.
There are 128 independently generated datasets for each of two cases.
The generating parameter, observation noise and numerical sampler use separate random streams.
The truth enters only evaluation; the likelihood receives the design matrix, observations and declared noise scale.
Both numerical budgets use the same datasets, original priors and likelihoods.

| Case | Unknown coordinates | Observations | Exact reference |
| --- | --- | --- | --- |
| Stationary object | Constant position | Eight independent Gaussian readings, noise standard deviation 0.2 | Univariate Gaussian posterior |
| Linear motion with uncertain start | Initial position and velocity | Eight positions at times 0.9 through 1.1, noise standard deviation 0.2 | Correlated bivariate Gaussian posterior |

Every unknown has an independent standard normal original prior.
The numerical proposal uses uniform coordinates transformed through the inverse Gaussian CDF, preserving the normalized prior with a unit base-density ratio.
No fitted posterior becomes a later prior.
The second case deliberately makes initial position and velocity hard to distinguish over the short observed time span.

Each dataset is fitted at 256 and 2,048 particles, giving 512 planned fits.
The sampler uses 32 temperature stages, eight moves per stage, local proposal scale 0.05 and block-refresh probability 0.5.
The evaluation limit is the particle count multiplied by 257.
The study retains complete weighted populations, data identities, numerical outcomes, ancestry, target evaluations and elapsed times.
These particles are numerical integration states, not environment or agent seeds.

## Reference and reporting checks

For each coordinate, the report checks the posterior CDF at the generating truth, central 90% interval coverage, and errors against the exact posterior CDF.
The predeclared numerical screen requires mean absolute CDF error at most 0.025 and its 95th percentile at most 0.075.
Coverage is prior-predictive: its reference expectation averages over parameters drawn from the declared prior as well as over observation noise.
It is not a guarantee of 90% frequentist coverage for every fixed parameter value.

Exact binomial intervals and diagnostics retain dataset counts.
A familywise diagnostic level of 0.01 is divided across the six candidate-coordinate comparisons and three distinct exact-reference comparisons.
Posterior-CDF uniformity p-values are descriptive; no separate automatic rejection rule is attached to them.
Passing these screens does not approve a posterior for physical planning.

A coordinate's full coverage comparison is produced only when all 128 planned fits for that case and budget are available and numerically complete.
Missing shards, infrastructure errors and noncomplete numerical results remain explicit.
Partial successful fits cannot silently replace the planned denominator.
The reporter checks every population's checksum, sampler configuration, data identity, original weights, empirical quantiles and CDF values.
It regenerates the independent dataset and solves the Gaussian reference again before accepting the reported truth or reference values.
It also checks identical datasets and exact reference distributions across the two budgets.

## Validation and compute schedule

Preflight `22677340` completed in ten allocation seconds on one compute CPU, with 3.28 worker seconds and zero native simulator actions.
Independent dense-grid integration agrees with the closed-form posterior means to at most 1.08e-14 and covariance entries to at most 2.34e-15 on the checked datasets.
Repeated atomic target evaluations match exactly and agree with direct Gaussian likelihood calculations.
A complete 256-particle correlated-case trial also finishes; this single dataset is harness validation, not a coverage result.

Report validation `22677390` completed in ten allocation seconds on one CPU, checking a full weighted population and rejecting altered truth, readings, exact CDF, empirical CDF, quantiles, coverage indicators and particle count.
The full array `22677408` has started after successful report validation; its first four shards are running and the remaining four wait for the concurrency limit.
It has eight shards with at most four simultaneous jobs, each requesting one CPU, 4 GB and two hours on `mit_preemptable`.
The scheduler rejected a dependency on the earlier completed preflight; its successful accounting and report were verified directly before submission, and the current validation dependency remains in place.
No array was created by the rejected submission.

Finite summary `22677409` waits for report validation and termination of the array, retaining any incomplete outcomes.
These jobs perform no native physics simulation and send no MB/MF notifications.
The production estimator and the ongoing physical-domain comparisons remain unchanged.

## Partial-report execution check

Compute job `22677945` completed in six allocation seconds on one CPU, performing no new inference trials or native simulation.
It ran the complete report path on 221 actual fitted populations from the running study, verifying their saved weights, samples, quantiles, data identities and exact references.
The four case/budget groups contained 57, 55, 55 and 54 available datasets respectively.
Every group correctly retained its planned denominator of 128 and withheld coverage estimates while incomplete.
The report remained incomplete, and the guard verified that partial successful fits cannot produce a full-group coverage result.
The frozen report and validation are `partial-summary-22677945.json` and `partial-validation-22677945.json` in the study bundle.
This checks the reporting path on real sampler outputs; it does not establish the study's eventual coverage or numerical accuracy.
