# Bridge joint inference target

September 13, 2026.
This implements a complete augmented parameter/initial-scene map for the revised Bridge program and its explicitly declared joint/reach discrepancy model.
It builds on the [original scene law](bridge-initial-scene.md), [conditional glue support](bridge-glue-support.md) and [causal future generator](bridge-causal-futures.md).
The acting agent still uses the incumbent estimator.
This is a development comparison with a changed discrepancy model, not an estimator-only ablation or a production replacement.

## Original prior and conditioning

The original six-parameter prior uses the public program's declared bounds.
`glue_latch`, `drip_r`, `drip_h`, `glue_rate` and `bond_dist` have independent uniform priors on their declared continuous intervals.
`bond_dwell` has a discrete uniform prior on the integers 1 through 120, inclusive.
The initial scene keeps the original rest/moving cases, six equiprobable resting faces, pose and velocity laws, robot-joint law, fixture support and unheld/unbonded reset-memory contract.
The whole-scene geometry normalizer is common and parameter-independent for this reset model; its unknown value prevents absolute-evidence claims.

The first exact partial glue increment from zero fixes `glue_rate = 0.2`, retaining its original density `1 / 0.95` once.
A consecutive `0.2, 0.4, 1` progression requires `0.4 < glue_latch <= 0.2 + 0.2 + 0.2` under the literal floating-point update.
The latch coordinate is sampled uniformly within that interval, retaining its mass under the original uniform `(0.4, 1)` prior.
Subsequent exact glue outputs still require compatible causal transitions; parameter conditioning does not remove those checks or their probabilities.

## Complete coordinate map

| Proposal coordinates | Meaning |
| --- | --- |
| 0 through 139 | Existing scene coordinates, with the checked prior-preserving resting-face proposal. |
| 140 | Latch value within the supported interval. |
| 141 through 143 | Radius, reach height and bond distance under a defensive Gaussian-box proposal. |
| 144 | The continuous-parameter proposal's mixture coordinate. |
| 145 | Discrete bond dwell under a defensive categorical proposal. |

The continuous proposal retains a 25% original-prior component and has standard deviation 0.01 in each physical parameter coordinate.
The dwell guide centers on the training-prefix support value 25 with scale 5, while retaining a 25% uniform component over all 120 original cases.
Every proposal supplies its original-prior/proposal correction.
Guide likelihoods and centers do not replace the original prior or remove observation factors.

The joint result carries six physical parameters and 140 scene auxiliary coordinates for the pinned original scene decoder.
Those auxiliary coordinates must be reconstructed with the identified decoder and data; they are not standalone Cartesian state coordinates.
Original prior assumptions, conditioning/guidance, data, output/discrepancy model, program and runtime receive separate identities.
Only the 600 fitting actions and their 601 observations enter the inference data identity.

## Target factors

The base factor contains the original root correction, resting-face and parameter proposal corrections, exact-rate/latch conditioning factors and complete initial output likelihood.
The remaining likelihood contains all 600 conditional joint-transition factors, normalized exact glue-event masses and remaining output factors.
The output law retains correlated scalar errors, coupled robot-orientation errors, the checked finger readout, original noisy channels and all other exact constraints.
Joint variances are integrated across the entire prefix; reach offsets are marginalized only when compatible intervals have identical complete next memory and commands.

Forbidden initial geometry has zero base and target support.
A later incompatible exact glue event retains a finite initial base where appropriate but has zero complete likelihood.
Setup failures and unsupported numerical operations raise errors; they are not silently converted into model likelihoods.

## Native preflight and next gate

Preflight `22704066` completed in 6:56 with 8,632 native actions.
The existing supported point is recovered exactly through the new coordinate map and agrees with its independently checked complete composition after adding the new proposal and latch factors.
Six nearby cases change radius, height, bond distance, latch, span position or bottle position; all have finite complete targets and repeat exactly.
Four broad draws are retained: two fail geometry and two contradict the first deposition at action 58.
The preflight also checks 32 independent parameter-proposal and conditional-factor calculations.
An earlier attempt failed before native replay because its frozen compute overlay omitted the target-evaluation module; the corrected allocation retains the same calculations and candidates.

Independent reader `22704257` completed in 2:21, checking parameter corrections, literal conditional updates, joint factors and complete output sums for all eleven cases.
Its maximum complete-target factor discrepancy is 2.9104e-11.

## Initial-population support

Screen `22704578` evaluated 32 unit-uniform candidates for each of numerical seeds 810 and 811, before any resampling or rejuvenation.
It completed in 56 seconds with 1,865 native actions.
The populations respectively contain nine and twelve geometrically feasible candidates, but neither contains a finite complete target.
Most feasible candidates contradict the first deposition at action 58; later failures occur at actions 87, 122 and 200.
The existing proposal is therefore insufficient to initialize either full sampler, despite the independently verified support points.

A separate proposal experiment preserves the same target and mixes a 25% full unit-uniform component with a normalized local product guide around the supported training-prefix point.
The local component selects its rest, face, mixture and dwell cases through restricted uniform intervals and guides 27 active continuous unit coordinates with standard deviation 0.02.
All other coordinates retain their uniform distributions.
These restrictions apply only to the local proposal; the broad component retains the entire original proposal support.
The additional factor is the negative log of the complete mixture density, not the density of the selected component.
Independent one-dimensional integration checks all 27 Gaussian normalizers, and both proposal branches are checked against separate density calculations.

Local screen `22704778` completed in 3:12 with 14,100 native actions.
It finds three finite candidates in seed 810 and twelve in seed 811, out of 32 draws each.
However, their effective sample sizes at the original first temperature, `1/32`, are both approximately one.
On the finite-support subsets, base-only effective sample sizes are approximately 1.69 and 1.04, so choosing a smaller first temperature alone cannot resolve initialization concentration.
Independent proposal-inverse, weighting and serial/parallel native verification completed as `22705833`, including 1,200 native actions and exact repeated target calculations.
The maximum inverse-coordinate discrepancy is 1.111e-15.
Its earlier attempt compared tuple-valued in-memory metadata directly with JSON lists; the corrected reader canonicalizes serialization before comparing physical results.

These are initialization screens, not posterior fits or agent experiments.
The next numerical work must improve supported-population initialization and weight balance before a full fitting run can be useful.
Any change must retain joint original-prior corrections and count rejected evaluations; conditioning proposals separately for each fixed parameter without the corresponding parameter-dependent acceptance factors would change the target.
Full sampling remains separate from numerical and predictive acceptance.

Artifacts are in `logs/uncertainty_bridge_joint_inference_20260913/`.
The corrected local proposal and its diagnostics are in `logs/uncertainty_bridge_local_proposal_20260913/`.
