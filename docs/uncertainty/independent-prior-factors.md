# Independent prior factors in the inference result

September 14, 2026.
This extends the offline parameter-inference representation used by the [simplification proposal](simplification-proposal.md).
The production estimator remains unchanged.

## Motivation and scope

The [Boil heating diagnostic](boil-unobserved-heating.md) establishes that the pinned program's all-burners-off fitting prefix cannot constrain burner radius, heating onset or heating width.
A finite resampled population can nevertheless appear concentrated in those coordinates.
Representing their exact independent priors explicitly avoids that approximation error without choosing a narrower prior or using future observations to select parameter values.

The result has the form `p(retained | data) * p(independent)`.
The retained block includes correlated parameters and uncertain initial-state coordinates, and stays a joint sampled distribution.
The independent block currently supports normalized uniform box priors in explicit physical coordinates.
Log-uniform laws, correlated factors, and parameters whose independence is only observed at a few fitted states are outside this interface's declared scope.

## Representation and checks

`inference_factorization.py` introduces a factorization declaration, a scoped evidence check, and a parameter consumer for the resulting product posterior.
The declaration identifies the full and reduced targets, the complete retained coordinate schema, and the independent prior law.
Both targets must refer to the same model program, fitting data and observation model.
The evidence scope changes when either identity, the retained schema, or the independent prior changes.
A matching digest binds the evidence to the declaration; it does not establish mathematical independence by itself.
The supporting argument must apply throughout the target's support.

The reduced posterior must independently pass its declared numerical assessment, and the factorization check must pass before any parameter draws or intervals are available.
Even a request containing only independent coordinates cannot bypass an unavailable reduced posterior.
Predictive failures remain visible, with distinct full-target and reduced-target scopes, and do not silently erase a numerically adequate posterior.
None of these checks approves an action or publishes a production fit.

The consumer returns exact uniform marginal quantiles for independent coordinates and weighted empirical quantiles for retained coordinates.
Joint draws resample whole retained rows according to their weights, then independently draw the analytic block.
Aliases of one coordinate share the same draw.
The resulting existing `ParameterEnsemble` carries equal Monte Carlo weights, the sampling seed, the original retained-particle indices, the full target identity and the scoped checks.
There is no finite exact `weighted_samples()` table for a continuous analytic factor; callers explicitly choose a resampling count and seed.
The existing sampled-only parameter consumer remains unchanged.

## Reduced Boil experiment

The target adapter in `logs/uncertainty_boil_reduced_target_20260914` removes proposal coordinates 79, 81 and 82 and output-joint coordinates 3, 5 and 6.
This leaves 81 jointly sampled coordinates, while retaining the original heating priors separately.
The frozen native evaluator uses midpoint values internally for the removed parameters because its unchanged fitting score is independent of them.
Those midpoints never appear in the reduced sampled joint and are not claimed as posterior samples.
The adapter retains the full original target score and all scene proposal corrections; each removed normalized uniform prior integrates to one.

The structural argument is tied to the exact reviewed program, empty initial memory, strictly positive onset and width bounds, and exact burner-off observations throughout the 132-action prefix.
Changes to that program, noisy burner-state observations, nonzero initial memory, or a prefix containing burner activation require a new argument or restoration of joint inference.
Native preflight compares seven parameter assignments at each selected source state, including finite and rejected targets, and checks unchanged scores and projected coordinates.
An independent reader uses a separate coordinate mapping, verifies physical prior bounds, rejects a retained midpoint coordinate and repeats native histories.
These are mechanical checks, not numerical mixing or prediction acceptance.

The fresh fitting bundle `logs/uncertainty_boil_reduced_fits_20260914` retains the original 32 particles, 64 geometric temperatures, eight moves, proposal scale 0.05, refresh probability 0.5 and 20,000-evaluation cap.
Removing the three independent singleton blocks leaves ten proposal blocks, with uniform selection among those remaining blocks.
Numerical seeds 410 and 411 are retained, but changing dimension changes the random streams.
No 84-dimensional checkpoint is resumed as an 81-dimensional fit.
Full fits depend on completed native preflight and a verified small fitting fixture for each seed.
Completed sampler artifacts remain unavailable as assessed posteriors until independent numerical replication and budget checks establish adequacy.
The original extrapolation comparison and incumbent results remain separate and preserved.

## Acceptance still required

Compute-node check `22717200` passes 27 focused tests, mypy, repository pytest-pylint checks, and pinned formatting checks.
Earlier check attempts exposed test typing and lint issues, which were corrected before this passing frozen run.
Native preflight `22717213` has completed: 56 parameter assignments at eight source states preserve the expected reduced target, with 792 native actions and four rejected changes to the factorization assumptions.
Independent reader `22717214` also passes all 56 assignments, rejects a retained placeholder coordinate and performs 264 fresh native actions.
Its verified source checksum is `7ec3f083e8a866ccad24db8242e53b6860593a73c60e010c82b5b38d7f6884de`.
Small fitting fixtures `22717216` and `22717220` and readers `22717219` and `22717221` have completed successfully.
Full fits `22717256` and `22717258` are now running, with full-fit readers `22717257` and `22717259` dependent on completion.
The subsequent [forecast validation](reduced-boil-forecasts.md) restores the three thermal priors explicitly.
All jobs use compute nodes on `mit_preemptable`.
The Boil adapter and each fitting fixture require independent verification before full fitting.
Future comparisons must restore the full heating priors when drawing parameter sets or integrate them explicitly, while preserving correlations and weights in the retained block.
An informative prefix containing heating remains a separate learning test; it does not replace the all-off extrapolation case.
Stable five-domain inference, live planning integration and final non-regression evidence remain required before retiring existing uncertainty mechanisms.
