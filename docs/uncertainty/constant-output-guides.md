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

## Next comparison

The audit bundle is `logs/uncertainty_fan_constant_guide_audit_20260913`, with report checksum `5e0298336461e9ea09639a7828abbb20dd91ae4d7c17b06db3151051388b10a0`.
Before launching a guided fit, verify its complete proposal density, retain original support and demonstrate target equality at identical physical candidates.
A mixture with the original proposal can preserve support while the guide focuses fixture locations.
Keep the original canonical scene coordinates in saved joint samples, so proposal changes do not silently change their meanings.
Any new fit must retain the same fitting prefix, original scene prior, physical program and output model, and remain separate from the ongoing particle-count comparison.
No guided physical inference run has been launched by this change, and no production fitter or planning rule uses the helper yet.
