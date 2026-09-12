# Coupled orientation output discrepancy

September 12, 2026.
This is an offline model extension for the [uncertainty simplification plan](simplification-proposal.md), followed by integration with the complete output likelihood.
It changes neither the declared sensor variance nor production agent behavior.

## Probability model

The [native readout audit](observation-reductions.md#native-euler-readout-support) found that independent ordinary angle densities cannot describe the robot's exact Euler observations.
The [PyBullet conversion implementation](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c) branches on the raw quaternion product `s = 2*(w*y-x*z)` at thresholds +/-0.99999.
Outside the ordinary branch it sets roll to zero, sets pitch to the corresponding pole, and computes a half-angle yaw that can extend beyond [-pi, pi].
The implementation does not normalize its quaternion input before this calculation.
Native conversion probes and Gaussian draws validate these behaviors against the installed runtime; the linked upstream file alone does not pin that runtime.

For a native predicted readout, the declared extension introduces a latent quaternion input:

$$
Q_t\sim \tfrac12\mathcal N(\mu_t,\sigma_q^2 I_4)
       +\tfrac12\mathcal N(-\mu_t,\sigma_q^2 I_4),
\qquad o_t=g(Q_t).
$$

Here `g` is the specified raw quaternion-to-Euler map, and the observation is exact conditional on the latent input.
The Gaussian scale is a discrepancy parameter with a fixed value or an original prior; it is not sensor variance.
The sign mixture describes unmodeled quaternion sign while retaining the original yaw branch in the observation.
Errors are independent over primitive steps conditional on the predictions and scale.
The scalar Cartesian discrepancy may separately retain its temporal dependence.

This is a statistical model of the readout input, not isotropic angular noise on SO(3), a physical force correction, or an exact finite-precision quantization model.
In particular, Gaussian samples are not normalized to unit length.
Normalizing them would change the pole probabilities and require a different likelihood.
These assumptions remain candidates for predictive evaluation, not established properties of real model error.

The recorded diagnostic constructs `mu_t` from the native predicted Euler triple using `getQuaternionFromEuler` and includes both signs.
This defines a representative center using only the forecast; it does not recover information lost by the earlier Euler readout or copy a recorded hidden quaternion.
Its conversion implementation belongs in the inference runtime identity.

## Marginalization and numerical checks

`QuaternionOutputError` in `inference_orientation.py` integrates the latent quaternion input.
The observation measure has a three-dimensional ordinary component and two one-dimensional pole components.
Their densities must be combined with their respective measures, rather than compared as three independent scalar angle densities.

For an ordinary observed pitch `p`, write quaternion radius as `r` and unit-quaternion pitch as `beta`.
The exact constraint is `r^2*sin(beta)=sin(p)`.
The likelihood integrates both quaternion lifts and all radii from `sqrt(abs(sin(p)))` upward, retaining the resulting `r*cos(p)/8` Jacobian.
For stable quadrature near the endpoint, the implementation instead uses `r^2=abs(sin(p))+u^2`, retaining `u` throughout the inverse map and using `r dr = u du`.
Reconstructing the small difference by subtracting rounded squared radii caused the recorded Balloons action-23 convergence failure; its regression test now passes at both requested tolerances.

At either pole, the observed raw yaw fixes the x/y polar angle.
The remaining x/y radius is integrated numerically, while the Gaussian z/w half-space probability is integrated analytically.
The yaw change of variable contributes a factor one-half, separately from the antipodal mixture weights.
Yaw retains its native [-2*pi, 2*pi] support.
Wrapping away its branch would combine distinct observations and require summing their densities explicitly.

Adaptive quadrature operates on scaled log integrands and bounds the omitted radial tail by an integrated Gaussian envelope.
Nonconvergence is an inference computation error, not a zero-likelihood declaration against the physical model.
Quadrature's reported errors do not prove discovery of every possible mode; independent checks and sensitivity to numerical settings remain necessary.

For a zero-centered four-dimensional Gaussian, an independent analytic reference gives ordinary density

$$
\frac{\cos(p)}{16\pi^2\sigma_q^2}
\exp\left(-\frac{|\sin(p)|}{2\sigma_q^2}\right),
$$

and density `exp(-c/(2*sigma_q^2))/(8*pi)` on each pole's raw yaw interval, where `c` is the pole threshold.
The ordinary mass and the two pole masses sum to one.
Tests check this reference at three scales, compare noncentral event probabilities with independent Gaussian samples and native conversion calls, preserve antipodal branch mass, and exercise extremely small log densities and invalid support.
The sampling checks use generated numerical cases, not agent seeds.

## Recorded orientation diagnostic

Job `22635163` completed on `mit_preemptable` using the five previously saved native 64-action forecasts.
The original fixed-state, noisy-first-frame initialization ablation is unchanged.
The first frame supplies that initializer and is excluded from this conditional forecast diagnostic's likelihood.
There is no inferred physical initial state or fitted simulator parameter in this experiment.

The original discrepancy-scale prior assigns equal mass to 0.0001, 0.001, 0.01 and 0.1.
Thirty-two actions inform its weights; the following 32 supply the joint future score.
The future suffix is excluded from discrepancy-scale fitting, but it is not established as unseen during historical simulator-program synthesis.
Every prior component is retained; a numerical failure makes the diagnostic unavailable rather than authorizing renormalization over successful components.

| Domain | Orientation readings scored | Dominant scale after prefix | Future joint log score | Largest per-reading change at tighter quadrature |
| --- | ---: | ---: | ---: | ---: |
| Bridge | 64 | 0.0001 | 331.1062 | 3.20e-13 |
| Fan | 64 | 0.0001 | 345.8598 | 3.06e-13 |
| Domino | 64 | 0.0001 | 91.1091 | 3.55e-13 |
| Boil | 64 | 0.001 | 124.8735 | 5.69e-13 |
| Balloons | 64 | 0.001 | 114.1498 | 7.96e-13 |

Scores are log densities under the declared mixed observation measure, not solve rates or evidence of improvement over the incumbent.
Tolerances were 1e-7 and 1e-9 for every scale and frame.
All 2,560 density evaluations completed; each tolerance covered 1,280 evaluations.
Artifacts are in [the frozen diagnostic bundle](../../logs/uncertainty_orientation_domains_v3_20260912/plan.json) and [its report](../../logs/uncertainty_orientation_domains_v3_20260912/reference-22635163.json).
Earlier bundles retain the robot-field selection setup failure and the unresolved near-pole quadrature attempts.

## Complete output composition

`OutputObservationModel` in `inference_observation.py` combines scalar discrepancy factors, coupled Euler factors, and checked deterministic readouts.
It enforces disjoint measurement assignments, preserves the source likelihood after readout verification, and scores every unassigned measurement under the original sensor model.
Unknown observations, missing predictions, partial Euler readings, duplicate factors, and unsupported noisy Euler readings are explicit errors.
Complete reset episodes retain their initial observation unless a caller explicitly supplies an empty reading for a separately declared conditional diagnostic.
Repeated full-data fits restart the original discrepancy laws.
The original data and the model's complete factor identity belong in the joint inference identity.

Integration job `22635183` applies this composition to all public output fields in the five 64-action forecasts, conditional on the initial frame.
It fixes scalar discrepancy persistence to 0.9, initial discrepancy to zero, Cartesian innovation scale to 0.005 and joint innovation scale to 0.001.
The orientation discrepancy scale is fixed at 0.001.
These settings are an explicit integration reference, not selected production settings or a fit against the legacy estimator.

| Domain | Complete conditional output likelihood | Remaining exact witnesses |
| --- | --- | --- |
| Fan | Finite | None in this window |
| Domino | Finite | None in this window |
| Bridge | Zero | One glue-field discrepancy |
| Boil | Zero | Two switch/faucet discrepancies |
| Balloons | Zero | Box speed discrepancies at all 64 actions |

Every finger reduction is checked against its observed source joint; orientation fields retain their coupled density.
The [integration report](../../logs/uncertainty_observation_domains_20260912/reference-22635183.json) records the full observed-field counts, factor identities and exact witnesses.
No event or speed observation is dropped to make a candidate compatible.
The three zero-likelihood outcomes concern these supplied forecasts; they are not proofs against every parameter and initial state.
The separately reviewed Bridge constant-glue invariant remains the stronger full-recording inconsistency control.

Fan and Domino now provide positive complete-output cases for the next joint parameter/initial-state comparison.
This does not complete that comparison: feasible uncertain initial-state priors, full runtime capture, the incumbent fitter, prediction assessment and the later planning/retirement gates remain required.
The current production agent remains unchanged.

## Validation provenance

Final compute job `22635342` passed 27 functional tests, four-file mypy and lint, and pinned isort, yapf and docformatter checks.
The working implementation and test hashes match that frozen validation snapshot.
The functional suite covers the new orientation and composition code together with the existing scalar-output and checked-readout components; the count is not added to older suites as unique tests.
Earlier attempts preserve the near-pole numerical failure, test type-annotation errors, and static-check issues that were fixed before this result.
The recorded diagnostics used the same corrected likelihood arithmetic before the final callback binding, import annotation, and schema-validation refinements.
Those refinements preserve scores for their valid complete recordings; the final tests additionally ensure that an unknown field is rejected before a contradictory readout could short-circuit validation.
Artifacts for the final checks are in `logs/uncertainty_orientation_v7_20260912`.
