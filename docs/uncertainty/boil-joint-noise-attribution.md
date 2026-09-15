# Boil forecast regression: joint-perturbation attribution

September 13, 2026.
The [completed Boil comparison](boil-canonical-forecasts.md#completed-forecasts-and-incumbent-comparison) showed much worse jug motion and heating forecasts under the experimental joint-transition model than under the incumbent selected point.
This diagnostic isolates future joint perturbations while keeping fitted candidates and their conditioning histories fixed.
It does not fit a replacement posterior or run an agent.

## Matched intervention

The plan selects particle indices 0, 8, 16 and 24 from each completed numerical population, 410 and 411, before inspecting their intervention outcomes.
Each uses the original bank-0/draw-0 random seed.
These eight points are diagnostic cases, not a representative posterior sample or a posterior-weighted forecast estimate.
Their original weights are retained as provenance, without renormalizing or averaging the selected subset.

Every case receives four complete continuations from its original root:

- All nine future joint perturbations, reproducing the archived forecast exactly.
- No future joint perturbations.
- Perturbations on the seven revolute arm joints only.
- Perturbations on the two prismatic gripper joints only.

The entire 132-action fitting prefix, candidate parameters, initial state, physical predictions, model memory and conditioning factors remain identical.
The intervention starts only in the reserved future.
All variants consume the same Gaussian random stream, including discarded draws, so the masks do not shift later random numbers.
The future generator receives no observed future states.
The no-perturbation variant is also repeated independently in a fresh world.

Suppressed-perturbation paths are explicitly marked as counterfactuals.
They do not retain the original Gaussian future-density fields as if those were the normalized densities of the intervened model.
A new probability model and fresh fitting are required before any variant can be assessed as a replacement posterior.

## Verified result

Compute job `22699007` completed in 3:34, covering all 32 intervention histories plus the eight no-perturbation repeats.
Independent reader `22699056` completed in 34 seconds.
It verifies every mask and Gaussian draw, unchanged fitting prefixes, literal model updates, native goal predicates, output moments and future-error calculations.
It also rejects deliberately altered suppressed-finger coordinates and altered prefix memory.

| Numerical population / particle | Jug x RMSE, all perturbations (m) | None | Arm only | Fingers only |
| --- | ---: | ---: | ---: | ---: |
| 410 / 0 | 0.1552 | 0.0073 | 0.0127 | 0.0111 |
| 410 / 8 | 0.1444 | 0.0073 | 0.2300 | 0.2262 |
| 410 / 16 | 0.1508 | 0.0073 | 0.0077 | 0.1458 |
| 410 / 24 | 0.1996 | 0.0073 | 0.0077 | 0.1978 |
| 411 / 0 | 0.3092 | 0.0036 | 0.0019 | 0.0040 |
| 411 / 8 | 0.2674 | 0.0067 | 0.0054 | 0.1883 |
| 411 / 16 | 0.2742 | 0.0034 | 0.2131 | 0.2646 |
| 411 / 24 | 0.0018 | 0.0013 | 0.0019 | 0.1575 |

Suppressing all future perturbations consistently brings these jug-x errors into the millimetre range.
The no-perturbation paths have twelve or thirteen held future frames, whereas most original variants have only one to seven.
Finger perturbations are frequently harmful, but arm-only perturbations also produce large errors in two cases.
The effect is nonlinear: the combined perturbations need not be worse than each individual mask on every candidate.
This identifies stochastic joint perturbations as a major contributor to the motion regression in these selected cases.
It does not justify removing their density factors from fitting or claiming that one mask is a validated inference model.

Heating remains a separate problem.
Even without future joint perturbations, bubbling errors range from approximately 0.107 to 0.592 on these candidates, and some final task-goal predictions still fail.
The new law's state and parameter uncertainty, numerical exploration and heating information in the fitting prefix still need separate assessment.

The saved fitting residuals also show that most per-joint root-mean-square deviations are below the fixed 0.001 scale, particularly for the fingers.
The next [shared-variance model](shared-variance-discrepancy.md) therefore estimates a declared variance from those residuals, preserving its uncertainty and complete likelihood, rather than choosing a smaller fixed scale from future performance.
Its posterior must be refitted under the changed law.
The production estimator remains unchanged.

The frozen plan, source hashes, full trajectories and verification report are in `logs/uncertainty_boil_joint_noise_attribution_20260913/`.
