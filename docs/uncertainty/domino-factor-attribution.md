# Domino likelihood-factor attribution

This Stage B diagnostic follows the [coupled-direction comparisons](domino-coupled-directions.md).
It identifies which existing likelihood factors cause large score changes without modifying the inference target, priors, program, observations or acting agent.

## Controlled replay

For each of the same 16 weighted anchor rows, select the finite joint-direction pair with the largest fitting-score difference between the small jittered and unjittered audits.
Replay its anchor and both candidates, giving 48 cases, plus one repeated complete replay.
Selection uses only the existing 64-action fitting prefix and its scores.
These deliberately selected contrasts are diagnostic examples, not a representative calibration sample or agent seeds.

The original native initialization, first-observation state, action sequence and output likelihood are retained.
The instrumentation records the 65 predicted and observed frames, per-factor likelihood contributions, and native contact records at primitive-step boundaries.
Every remaining likelihood exactly reproduces its previously saved target value.
The repeated case also reproduces its complete predictions, contacts and factor decomposition exactly.

Compute job `22681870` completed in 42 allocation seconds on four CPUs in `mit_preemptable`.
The 49 replays use 3,136 simulated actions.
Artifacts, frozen input identities and scripts are in `logs/uncertainty_domino_factor_attribution_20260913/`.
The summary verifies the complete factors, initial-observation removal and all 48 score contrasts, including identical cases with zero change.
An independent variance-form Kalman calculation verifies all 93,600 scalar time factors, with maximum absolute log-factor discrepancy 1.422e-14.
The independently checked scalar terms account for the dominant contributions below.

## What changes the scores

Robot proprioception and robot pose account for 59.9% to 99.7% of the sum of absolute channel contributions, with median 96.8%, for the 16 jittered-versus-unjittered contrasts.
Controlled joint-position channel 6 is the largest individual contribution in 15 contrasts; channel 5 is largest in the remaining contrast.
Its maximum predicted difference between matched candidates ranges from 0.0007623 to 0.012082 radians.
The existing proprioceptive discrepancy process has persistence 0.9 and innovation scale 0.001 per primitive step.
Large likelihood ratios from these transient joint differences therefore need not involve a different final object outcome.

| Anchor | Numerical seed | Jittered minus unjittered log score | Robot share of absolute channel changes | Largest channel | Peak step |
| ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 100 | -21.756 | 70.8% | Joint 6 | 51 |
| 1 | 100 | 44.075 | 95.4% | Joint 6 | 50 |
| 2 | 100 | 1.343 | 83.3% | Joint 5 | 57 |
| 3 | 100 | -83.775 | 99.5% | Joint 6 | 53 |
| 4 | 100 | -36.089 | 99.2% | Joint 6 | 52 |
| 5 | 100 | -53.496 | 94.8% | Joint 6 | 50 |
| 6 | 100 | 36.524 | 98.1% | Joint 6 | 49 |
| 7 | 100 | 98.293 | 98.8% | Joint 6 | 52 |
| 8 | 101 | -15.327 | 74.0% | Joint 6 | 52 |
| 9 | 101 | -9.783 | 59.9% | Joint 6 | 53 |
| 10 | 101 | 55.613 | 98.1% | Joint 6 | 54 |
| 11 | 101 | 129.646 | 99.7% | Joint 6 | 49 |
| 12 | 101 | -117.307 | 99.2% | Joint 6 | 48 |
| 13 | 101 | -26.525 | 98.1% | Joint 6 | 52 |
| 14 | 101 | -22.103 | 84.9% | Joint 6 | 52 |
| 15 | 101 | 14.113 | 67.1% | Joint 6 | 52 |

All 16 matched pairs have the same sets of contacting body/link pairs at the recorded primitive-step boundaries.
This does not establish identical contact forces, locations, multiplicities or contact behavior inside an environment step.
Most dominant score changes occur during steps 48-54; the remaining channel-5 case peaks at step 57.
This narrows the investigation to the robot's transient response rather than establishing a categorical contact-event mismatch.

The attribution is not evidence that joint observations should be dropped or their error scale increased.
Those observations may legitimately constrain contact parameters through the robot's response.
The joint and Cartesian channels also cannot be merged by assuming fresh forward kinematics reproduces the recorded Cartesian pose: the [earlier observation-phase audit](observation-reductions.md#cartesian-robot-pose-is-different) already disproved that shortcut.
The likelihood still needs a principled account of their actual timing and dependencies.

## Consequence for the plan

Neither broad nor smaller donor-direction moves provide a consistent numerical repair across the retained populations.
The new attribution explains why apparent object-level similarity does not imply similar fitting probability: the score changes mainly come from robot joint transients.
It does not establish that the current probability model is calibrated or that a particular alternative sampler will explore it adequately.

The next numerical investigation should test local sensitivity of those transient joint predictions and distinguish a narrow, reproducible parameter constraint from irregular numerical or model behavior.
Any sensitivity-informed proposal must retain the original prior and complete likelihood and pass independent numerical reference checks before a new native inference comparison.
Keep the original fitter in control until the Stage A/B evidence requirements and later planning comparisons are met.

## Completed local-sensitivity test

Job `22682046` evaluates anchors 0, 7, 8 and 15 at symmetric changes of 1e-6, 1e-5 and 1e-4 in each of the five unit-prior parameters.
The 124 cases plus an exact repeated replay use 8,000 native actions and complete in 81 allocation seconds on four CPUs.
Every original anchor score reproduces exactly, and a separate scalar check verifies all 241,800 time factors with maximum absolute error 1.422e-14.
The complete results are in `logs/uncertainty_domino_transient_sensitivity_20260913/`.

For each parameter and scale, the diagnostic estimates the central-difference vector of joint-6 predictions across the fitting prefix.
The relative vector difference is the norm of the difference divided by the larger vector norm.
For the 16 non-restitution anchor/parameter combinations, changing the scale from 1e-6 to 1e-5 gives relative derivative differences of approximately 0.847 to 1.408.
Changing from 1e-6 to 1e-4 gives differences of approximately 0.965 to 1.024.
These are large discrepancies, not stable local sensitivities suitable for an unvalidated gradient or curvature proposal.
All tested restitution derivatives remain exactly zero.
Exact repetition rules out nondeterministic evaluation as the explanation for these particular differences; it does not establish a smooth simulator map.

The evidence does not justify simply increasing particles again, choosing a gradient proposal, or deleting the informative joint channels.
A possible next model comparison is to place explicitly modeled joint discrepancy in physical transitions, condition those transitions on exact joint readings and retain their densities once, using the existing transition-conditioning machinery.
That would be a different probability model from the current output-error law and must be labeled and validated separately, including its observation phase and unconditional future generation.
It must not be presented as a numerical sampler improvement under the unchanged target.
