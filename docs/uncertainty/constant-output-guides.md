# Constant-output likelihood and proposal guides

The Fan proposal currently uses only the first noisy reading to place each fixture coordinate.
The frozen output model then scores every prefix reading, with correlated Gaussian output discrepancy retained.
A more informed proposal can use those prefix readings while preserving the original prior and complete likelihood.
This addresses proposal efficiency; it does not declare an object stationary from noisy observations or change the acting agent's execution estimator.

## Implemented calculation

`constant_output_likelihood()` in `inference_output_error.py` integrates the existing scalar AR(1) discrepancy model for a constant simulator output.
Its affine Kalman innovations collect a quadratic log likelihood in linear time, including missing readings at their original primitive-step positions.
The returned `ConstantOutputLikelihood` stores the center, scale and log density at the peak:

```text
log p(readings | location) = log_peak - 0.5 * ((location - center) / sigma)^2
```

This is a reading likelihood, not a normalized parameter posterior.
A caller must still supply its original prior and justify the output model.
The helper retains the model's initial discrepancy variance, temporal correlation, innovations and declared sensor noise.
Only when the discrepancy vanishes does it reduce to ordinary independent Gaussian averaging.
Exact observations require the existing explicit-conditioning machinery; zero sensor noise, missing evidence and unrepresentable numerical scales are rejected rather than assigned a floor.

A static-fixture proposal can use the Gaussian center and scale, truncated to its original support, while retaining its explicit prior/proposal density correction.
The original simulator likelihood must still score each reading exactly once.
Using a data-informed proposal does not authorize adding the guide likelihood as additional evidence.
Quantized or otherwise transformed physical readouts can make the constant-location Gaussian an approximate guide; they remain governed by the original target evaluator.
The supplied history must exclude any reserved future when constructing a fitting proposal.

## Verification

Compute job `22678241` passed 33 functional tests, two-file mypy and pylint, and pinned formatting checks.
Independent dense Gaussian calculations verify centers, scales and full log densities for positive, negative, zero and unit persistence, missing readings and nonzero initial error.
Existing filtering and forecast tests also pass.
Separate tests check independent averaging, translation stability, exact-observation rejection and explicit numerical failures.
The final formatted source hashes are retained in `logs/uncertainty_constant_output_checks_20260913/checked-sources.json` and match the committed files.

The subsequent Fan audit `22678319` completed in 34 allocation seconds on one compute CPU, with zero native simulator actions.
It reconstructs the exact 64-action fitting-data identity before extracting the 65 prefix readings.
Across 128 saved positive-weight prefix histories, all 30 fixture x/y/z coordinates remain exactly constant.
This verifies those histories, not arbitrary new candidate programs or every possible initial scene.
For each coordinate, the new quadratic likelihood matches the original scalar filtering likelihood at four locations to at most 1.14e-13 log-density error.

| Fixture-location guide | Scale |
| --- | ---: |
| First reading alone | 5.000 mm |
| All 65 prefix readings with the existing correlated output-error law | 3.312 mm |
| Treating those readings as independent sensor noise only | 0.620 mm |

The correlated guide's center differs from the first reading by as much as 6.067 mm across the fixture coordinates.
Using the independent-reading formula would substantially overstate the information under the current output model.
The new calculation retains the correlation instead of introducing a new noise assumption.

## Guided proposal validation

The audit bundle is `logs/uncertainty_fan_constant_guide_audit_20260913`, with report checksum `5e0298336461e9ea09639a7828abbb20dd91ae4d7c17b06db3151051388b10a0`.
Before launching a guided fit, verify its complete proposal density, retain original support and demonstrate target equality at identical physical candidates.
A mixture with the original proposal can preserve support while the guide focuses fixture locations.
Keep the original canonical scene coordinates in saved joint samples, so proposal changes do not silently change their meanings.
Any new fit must retain the same fitting prefix, original scene prior, physical program and output model, and remain separate from the ongoing particle-count comparison.
The guided comparison below is offline; no production fitter or planning rule uses the helper.

The isolated implementation is in `logs/uncertainty_fan_fixture_guide_20260913`.
It samples a mixture with 10% original proposal mass and 90% fixture-guided mass, changing only the 30 fixture position coordinates.
Both components map into the same original 120-coordinate scene representation; an additional proposal-only coordinate selects the component.
The target subtracts the log density of the complete mixture in the original coordinate representation, regardless of the selected component.
The physical simulator program, original scene prior, sensor/output law and 64-action fitting prefix remain fixed.
The additional 68 actions remain reserved for future prediction assessment.

Compute reference `22678659` verified proposal normalization, corrected zeroth/first/second moments and component-independent correction for identical represented points.
The normalization integral was 1.0000000000006977, and the corrected moments matched 1, 1/2 and 1/3 within 3.2e-10.
These are analytic proposal checks with zero native simulator actions, not a physical posterior result.
Native preflight `22678746` completed in 198 allocation seconds on four compute CPUs, evaluating 10,240 native actions.
It preserved all 12 original support/target witnesses and checked four guided candidates, including two supported and two unsupported cases.
Their original physical target factors were unchanged apart from the declared full-mixture correction.
The complete 64-particle serial and parallel initial sampler states were exactly equal.
Serial initialization took 99.88 seconds and parallel initialization took 24.61 seconds within this same allocation.

Initialization is a caution, not a success claim: 15 of 64 candidates had finite support, but their weight effective sample size was only 2.010, compared with approximately 9.1 for the original proposal.
A proposal informed by the entire prefix can concentrate on locations that receive little mass at the initial tempered stage.
The correction preserves the target mathematically but does not ensure efficient initialization or later exploration.
Do not use supported-candidate counts alone to claim improvement.

Array `22678810` launches two exploratory guided fits with sampler seeds 302 and 303, matching the original 64-particle Fan prefix pair.
Each uses four compute CPUs on `mit_preemptable`, with 32 temperatures, eight moves, the original 16,448-evaluation limit and a four-hour allocation limit.
The launch manifest pins the tested preflight, scripts, model and data/proposal plan.
These runs are separate from the unchanged 128-particle budget comparison.
Final population diversity, independent-run agreement, reserved-future predictions and total inference cost remain required before judging this guide.
The forecast adapter must replay saved canonical scene samples while verifying the proposal correction against the augmented sampler coordinates; the original forecast driver assumes these two coordinate arrays are identical and cannot be reused unchanged.


The adapted forecast validation bundle is `logs/uncertainty_fan_guided_forecast_20260913`.
Short sampler fixture `22678838` is queued, followed on successful completion by forecast validation `22678841_0`.
The adapter first restores the completed augmented-coordinate checkpoint without additional target evaluations.
It transforms each positive-weight proposal into the saved original scene coordinates, then checks the original native prefix likelihood and the fully corrected base density during full-trajectory replay.
The first positive-weight history is repeated, and every history is saved with its unchanged population weight.
These jobs validate the adapter only; the full forecasts below depend on the additional saved-artifact verification passing.


## Matched prediction comparison

The comparison reader is in `logs/uncertainty_fan_guided_summary_20260913`.
It permits the intended proposal/conditioning identity change while requiring identical data, observation model, physical program, original scene prior and sampler settings apart from the additional proposal coordinate.
The original simulator and scene mapping are byte-identical between arms.
The only changed inference overlay adds the two constant-output helper definitions; removing those additions leaves the entire original module syntax tree unchanged.
The original target evaluator is preserved byte-for-byte in the guided wrapper.

Validation `22678900` completed in 11 allocation seconds on one compute CPU, independently checking all 128 positive-weight original forecast histories and their unchanged weights.
It rejected eight incompatible comparisons covering data, sensor, program, prior, sampler configuration, coordinate meanings, source runtime and target factorization.
This establishes comparison checks, not guided prediction quality.
The guided saved-artifact verifier `22678924` remains dependent on the native forecast fixture.
It checks every canonical scene and full-mixture correction, then tests rejection of altered or missing coordinate checks, corrections, weights, predictions, particles and checkpoint identities.

Full guided forecasts `22678925_1` and `22678926_2` are queued behind that verifier and their respective completed fits.
Summary `22678943` waits for the validation jobs and both forecast terminal states, retaining missing or unavailable outcomes explicitly.
The summary reports both original and both guided populations, all available pairwise prediction disagreements, parameter mass concentration and inference/forecast costs.
No guided prediction result or adequacy conclusion is available yet.
