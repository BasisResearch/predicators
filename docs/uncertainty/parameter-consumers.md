# Parameter consumers of assessed inference

September 12, 2026.
The [simplification proposal](simplification-proposal.md#3-standardize-the-inference-result-evaluate-the-approximation) requires credible intervals and planning ensembles to come from the same joint posterior.
[ParameterPosterior](../../predicators/code_sim_learning/inference_parameters.py) now implements that consumer boundary beside the incumbent adapter.
It does not select an estimator, publish a parameter update, change a prompt or approve a plan.

## One approximation, explicit projections

Construct a view from `AssessedInference` and an explicit parameter schema.
Simulator parameter names may differ from joint inference coordinates:

```python
view = ParameterPosterior(
    assessed,
    names=("mass", "friction"),
    coordinates=("theta.mass", "theta.friction"),
)
quantiles = view.marginal_quantiles((0.05, 0.5, 0.95))
weighted = view.weighted_samples()
parameter_maps = weighted.as_dicts()
```

If coordinates are omitted, they equal the supplied names.
No prefix stripping, marginal refitting or parameter-name inference occurs.
The mapping may explicitly tie two simulator parameters to the same inferred coordinate.

`marginal_quantiles` uses the source approximation's weighted empirical CDF and its existing inverse-CDF convention.
`weighted_samples` projects complete rows onto the requested parameters, removing only zero-mass rows and retaining the remaining original weights.
It preserves dependence across parameters, including separated modes and nonlinear relationships.
Dropping episode initial-state coordinates performs a parameter marginal projection; it does not independently recombine parameter values.
This operation also does not provide a joint belief over parameters and the current execution state.
The proposal's initial planning comparison still retains the existing execution estimator, while any conditional execution-state extension remains separate.

Every returned ensemble records the inference identity, assessment-protocol identity, simulator names, source coordinate names, source particle indices, weights and predictive checks.
Parameter dictionaries returned to a simulator are owned copies.
Empty parameter schemas yield explicit empty maps without inventing a fitted parameter.

For consumers that require equal weights, `resample(count, seed)` draws complete rows multinomially with an explicit local RNG seed.
The output records that seed, equal weights and the original particle indices, including repeated selections.
This adds Monte Carlo error to the represented approximation and does not increase its information or discover missing modes.
`expectation(outcomes)` computes the weighted expectation of one finite outcome per row.
Boolean outcomes give success probability under that represented ensemble, not a guarantee or an action decision.
Stress-test candidates and their outcomes must remain separately labeled; these weights do not assign them posterior probability.

## Information seeking from the same weights

`ParameterEnsemble.atom_information(read_probabilities)` applies the existing information-seeking criterion to the ensemble's own weights.
Rows must follow `as_dicts()` order, columns identify atoms, and each entry gives the probability of reading that atom as true under the declared observation channel.
Binary entries represent exact reads.
The caller remains responsible for evaluating the same atoms and observation channel under each parameter map.

For each atom, the score is `H(sum_k w_k p_k) - sum_k w_k H(p_k)`, averaged over atoms.
Both terms use the same posterior mass, avoiding an implicit conversion of unequal posterior weights into equal member counts.
This is the existing mean of per-atom information scores, not the joint information in reading all atoms together.
It does not define a new probe threshold, approve an action, or remove predictive failures attached to the ensemble.

The existing `mean_bernoulli_entropy` and `noisy_read_information` helpers accept optional normalized member weights.
The weighted path checks probabilities, dimensions and finite nonnegative masses, and refuses unnormalized inputs rather than repairing them.
A valid ensemble with no atom columns scores zero; an empty or malformed weighted ensemble is rejected.
Omitting weights retains the incumbent arithmetic and call sites.
Acting-agent routing remains unchanged until the offline inference and saved-decision gates pass.

## Numerical and predictive status

Construction checks assessment/posterior identity, original-prior identity, source coordinates and the declared numerical assessment protocol.
It also validates the completed source sample structure through the existing assessment boundary.
An unavailable or unevaluated result retains its diagnostics but raises `UnavailableParameterPosterior` if a consumer requests samples or quantiles.
Sampler completion alone does not authorize access through this view.
The boundary does not choose adequate numerical criteria or verify the scientific sufficiency of a caller's protocol.

A predictive failure remains attached to a numerically available posterior and every derived ensemble.
It does not silently remove that posterior, turn uncertainty into a deployment verdict or excuse a contradictory exact constraint in the fitted target.
Decision thresholds, stress tests, model revision and parameter publication remain separate responsibilities.
Malformed weights, sample dimensions, source indices and requested outcomes fail explicitly.

## Validation and integration gate

The tests use an exact weighted multimodal reference with a nonlinear parameter relation, a zero-mass outlier and a nuisance initial-state coordinate.
They verify source quantiles, weights, row provenance, explicit name mapping, shared coordinates, resampling frequencies and preservation of parameter dependence.
Unavailable numerical results and predictive failures exercise distinct paths.

An end-to-end numerical test fits a noisy sum observation under a three-coordinate prior, checks analytic mean references, passes that result through assessment and evaluates the resulting joint parameter ensemble.
The ensemble retains the learned relation between the two parameters, while the unused coordinate retains its prior mean.
This is a small numerical consumer test, not a protocol sufficient to certify a physical-domain posterior.

Compute job `22642559` passed all 19 functional tests, focused mypy and pylint, and pinned formatter checks.
The checked source hashes match the committed module and test files; the frozen check manifest and output are in `logs/uncertainty_parameter_view_checks_v6_20260912`.

The weighted information extension passed 61 functional tests, four-file mypy and pylint, and pinned formatter checks in compute job `22650348`.
Its independent reference enumerates the joint distribution of parameter member and binary read, including unequal masses, irrelevant zero-mass rows and identical-member splitting.
The assessed-posterior integration test retains a predictive failure while producing the analytically expected information score through the shared ensemble.
All 768 default-path comparisons with the frozen incumbent matched exactly, including empty atom sets and unanimous or uncertain reads.
Artifacts and checked source hashes are in `logs/uncertainty_weighted_information_checks_20260912`.
These checks establish scoring and interface behavior, not physical-domain posterior adequacy or unchanged solve rates.

These APIs are ready for offline and saved-decision comparisons once the source inference passes its numerical and predictive investigations.
No acting agent has been routed through them, and no legacy mechanism has been retired.
