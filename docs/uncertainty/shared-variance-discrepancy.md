# Learning a shared discrepancy variance

September 13, 2026.
This is an offline alternative to a fixed joint-transition discrepancy scale, motivated by the [Boil intervention audit](boil-joint-noise-attribution.md).
It does not change sensor noise or production agent behavior.
Learning the scale is a model change that still requires new fitting and predictive validation.

## Probability model

For one selected channel, declare a single variance `v` shared across its residual history:

```
v ~ InverseGamma(alpha, beta)
r_t | v ~ Normal(0, v)
```

The inverse-gamma density is `beta**alpha / Gamma(alpha) * v**(-alpha-1) * exp(-beta/v)` on positive variance.
The scale `beta` has squared channel units.
The residuals may describe an explicitly declared transition or output discrepancy; that choice and the sharing boundary belong in the caller's probability-model identity.
This component does not establish that independent zero-mean residuals are adequate for a particular domain.

After `n` residuals with squared sum `S`, the conditional variance has shape `a = alpha + n/2` and scale `b = beta + S/2`.
The complete normalized residual-history log density is:

```
log Gamma(a) - log Gamma(alpha)
+ alpha * log(beta) - a * log(b)
- n/2 * log(2*pi)
```

The next residual has a Student-t predictive density with `2*a` degrees of freedom and scale `sqrt(b/a)`.
Sequentially conditioning and multiplying those predictive factors gives the same complete marginal as integrating the shared variance once.
Independent Student-t draws with a fixed scale at every step would instead lose the dependence induced by the shared variance.

For a physical future, draw one variance from the conditional inverse-gamma law and retain that draw throughout the entire continuation.
The simulator then draws its per-step Gaussian residuals conditionally on that shared value.
When evaluating the density of an observed future, update the variance sufficient statistics causally and retain each normalized Student-t factor.
Generating a future and evaluating its marginal density are different operations, even though they use the same declared model.

The original prior is fixed before the fitting history.
Refitting the same complete data uses that original prior again; it does not treat the previous posterior as a new prior.
The component has no sensor-noise parameter and never writes a simulator state.
It raises explicit errors for invalid inputs and unrepresentable arithmetic rather than substituting a tiny variance or clipping observations.

## Implementation and checks

[inference_variance.py](../../predicators/code_sim_learning/inference_variance.py) supplies immutable original-prior and conditional sufficient-statistic objects, complete-history density, causal predictive factors and shared-variance draws.
The implementation is not routed into the production fitter or execution estimator.

Compute check `22699354` passed thirteen functional tests, focused two-file type and lint checks, and the pinned formatters.
The tests independently integrate the original variance density by numerical quadrature, check Student-t predictions against a separate implementation, verify density changes under physical-unit conversion, and check dependence between squared future residuals under a common variance.
They also check repeated batch fitting and explicit rejection of invalid inputs and arithmetic overflow.

## Boil integration

The first physical comparison retains the previous Boil program, original scene and parameter priors, observations, 132-action prefix, proposal map, sampler settings and numerical seeds 410/411.
It replaces only the fixed 0.001 joint-transition scale with one independent shared variance per controlled joint for this episode.
Each original variance prior has `alpha=2` and `beta=1e-6`, giving the same prior mean variance as the former fixed value squared.
These are native squared units: radians squared for the seven arm joints and metres squared for the two fingers.
The prior specification is recorded explicitly; data-derived scale estimates are not substituted for the prior.

During fitting, the nine variances are analytically integrated out, so they add no sampler proposal coordinates.
Every exact observed joint correction retains its normalized predictive density, and the output model does not add a second joint-error density for the same correction.
The candidate physical histories are unchanged under the existing exact-joint conditioning protocol; their statistical weights change.
The initial-state and model-parameter priors remain fixed.

Native preflight `22699445` completed eight candidate checks and their independent repeats, totaling 2,112 native actions.
All native observations and literal model outputs match the archived fixed-scale prefixes exactly.
The new complete likelihood agrees with an independent gamma-integral calculation after replacing the old joint factors, with maximum difference 3.638e-12.
Repeated fresh replay and complete target evaluation are exact.
The resulting conditional root-mean-square finger scales range from approximately 0.127 to 0.288 mm on these selected candidates, below the former fixed 1 mm.
Those candidates were selected before this diagnostic; they are not samples from a newly fitted posterior.

Fresh matched 32-particle fits `22699492_0` and `_1` are running on compute nodes.
Their complete model identity includes the numerical variance prior as well as the transition law.
Both retain the original fitting-data and physical-prior definitions, while recording the changed discrepancy model separately.
They remain unassessed until independent replicas, held-out predictions and computational cost are checked.

Future fixture `22699602` tests repeated generated continuations, one variance draw per joint per history, and density round trips using generated observations on two fixed candidates.
It completed in 2:04 with 1,848 native actions.
Both generated histories repeat exactly and reproduce their full physical trajectories and marginal density when their generated future observations are supplied to the density evaluator.
Independent Student-t calculations differ from the recorded per-step factors by at most 2.354e-13, and complete future joint densities agree with direct gamma-integral calculations.
Separate artifact reader `22699757` completed in 20 seconds, checking four saved histories and 9,504 joint factors, future joint-output alignment, native predicates, scalar moments and literal memory.
It rejects altered shared variances, future joint readings and density factors.
Full-population forecasts still need the completed replacement fits and the corresponding verified adapter.
Reducing perturbation scale alone does not resolve the remaining heating errors or establish agent non-regression.

Frozen component checks are in `logs/uncertainty_shared_variance_20260913/`, fitting/preflight artifacts in `logs/uncertainty_boil_shared_variance_20260913/`, and future checks in `logs/uncertainty_boil_shared_variance_forecast_20260913/`.
