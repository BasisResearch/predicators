# Fan fixture sensitivity and static observation model

This is Stage B development work under the [simplification proposal](simplification-proposal.md).
The [reflection diagnostic](fan-reflection-diagnostic.md) ruled out speed-branch representation as a remedy for the measured event and goal disagreement between two fitted populations.
The following conditional experiments locate another source of that disagreement and assess an explicit alternative discrepancy model.
They are not agent results or evidence of an accepted posterior replacement.

## Conditional scene exchanges

The experiment retains the same two highest-weight anchors selected before the reflection profiles: seed 302 particle 53 and seed 303 particle 61.
Each anchor retains its original speed while receiving selected initial-scene coordinates from the other anchor.
These exchanged candidates receive no posterior weights.
Both fitting-prefix scores and reserved-future predictions are reported, with geometry rejections retained explicitly.

The broad exchange audit `22712070` and independent reader `22712072` pass.
Twelve complete histories use 1,584 native actions, and two other candidates are rejected by the initial geometry check.
The groups are robot joints, fixture poses, ball state, switch articulations and fan rotors.
Exchanging robot joints, switch articulations or fan rotors leaves both anchors' ball, event and goal predictions unchanged in this test.
Exchanging all fixture poses in the seed 303 anchor changes 19 future goal labels and changes its final goal from false to true.
The reverse fixture exchange is geometry-rejected, as is the opposite isolated ball exchange, exposing coupling between fixture and ball positions.

The individual-fixture audit `22712128` and independent reader `22712131` also pass.
Twenty-five complete histories use 3,300 native actions, and one candidate is geometry-rejected.
Only exchanging the target fixture changes the anchors' final goal outcomes.
It changes 18 and 19 future goal labels, respectively, and changes ball positions by approximately 0.0140 and 0.0151 m RMS.
Exchanging switch 2's fixture pose changes one activation-time label for the switch and its fan, without changing the ball or goal curve.
Jointly exchanging fixtures and ball state is geometrically supported in both directions and transfers the final-goal difference.

| Target-pose exchange | Fitting-target log-score change | Changed future goal labels | Final predicted goal |
|---|---:|---:|---|
| Seed 302 anchor receives seed 303 target | -1.0727 | 18 | True to false |
| Seed 303 anchor receives seed 302 target | +1.0727 | 19 | False to true |

The target z coordinates in these two scenes are 0.3965203 and 0.4062214 m, a difference of approximately 9.7 mm.
The target is a physical pad, so uncertainty in its pose can change contact dynamics as well as the geometric goal test.
A follow-up coordinate audit separates x, y, z and yaw for the target and switch 2: native job `22712378` and reader `22712380` both pass.
All 20 histories complete, using 2,640 native actions.
Exchanging only target z changes the final goal in both directions, with 18 and 20 changed future goal labels and ball-position RMS changes of approximately 0.0133 and 0.0151 m.
Exchanging only switch 2 y transfers its one-step activation-time difference in both directions.
Target x changes one goal label in the seed 302 anchor without changing the ball trajectory or final goal; the other coordinate exchanges do not change the checked event or goal curves.
All of these are conditional comparisons at two scenes; they do not establish a population-wide causal decomposition.

## Static fixtures and the discrepancy law

The frozen public environment source gives the walls and target zero mass and creates switches with fixed bases.
The frozen candidate program writes only the ball's velocity.
The fixture xyz observation fields therefore describe fixed base positions in this comparison.
The existing model nevertheless assigns an additive AR discrepancy process to these 30 fixture coordinates, in addition to their original 0.005 m sensor noise and sampled initial pose.
That is a modeling choice, rather than a requirement imposed by their dynamics.

The alternative removes only those 30 discrepancy factors, allowing the unchanged sensor model to score their measurements directly.
It retains all dynamic output factors, angular factors, exact events, physical priors and candidate programs.
This is explicitly a change to the observation/discrepancy model, not a numerical sampler correction.
It is justified by the fixed-fixture source contract and evaluated separately from changes to initialization or sampling budgets.

The saved-history audit `22712247` and independent reader `22712284` pass all 128 original histories and 3,840 fixture factors without new simulation.
Every checked fixture coordinate is exactly constant throughout its complete native history.
An independent scalar Kalman calculation and Gaussian sensor likelihood reproduce the composite likelihood change within `9.0949e-13`.
The unchanged noise injector gives independent sensor measurements at these recorded steps.

For one fixture coordinate and 65 fitting observations, the sensor-only constant-location likelihood has standard deviation `0.005 / sqrt(65)`, approximately 0.0006202 m.
The previous discrepancy law gives a corresponding constant-location likelihood width of approximately 0.0033115 m.
These are likelihood widths, not an assertion about the final marginal posterior after all geometry and trajectory constraints.

Reweighting the existing approximate populations with the fitting-likelihood change gives effective sample sizes approximately 1.0000 and 1.0123 out of 64.
Those weights never use future measurements.
Their concentration prevents treating this reweighting as a reliable posterior comparison and motivates fresh inference under the alternative law.

The old initialization also does not reproduce Boil's severe base-weight concentration.
The checked seed 302 initial state has 15 supported particles with essentially equal weights and effective sample size 15.
Seed 303's recorded initial effective sample size is 23.0957 among 26 supported particles.
No additional support-initialization or finite-factor-tempering treatment is introduced in this matched model comparison.

## Fresh matched inference

The new frozen fit bundle is `logs/uncertainty_fan_static_fits_20260914`.
It retains the original 64 particles, 32 temperatures, eight moves per stage, proposal blocks, broad speed prior and 16,448-evaluation budget, with numerical replicas 302 and 303.
Only the static-fixture discrepancy factors and their corresponding normalized defensive proposal change.
The proposal uses the sensor-only constant-location likelihood while retaining the original proposal-mixture component and original-prior/proposal correction.
Future observations are unavailable to the fitting target.

Native preflight `22712355` passes all 64 fresh target evaluations across serial and parallel execution and compares complete initial sampler checkpoints, using 8,192 native actions.
Its initial population has 28 finite candidates and weight effective sample size 24.0957, which is an initialization diagnostic rather than evidence of final exploration.
Independent reader `22712364` also passes, reconstructing each physical candidate, rerunning its fitting prefix, checking its complete target factors and verifying the fixed-fixture observation partition with another 4,096 native actions.
Array `22712476` submits the full paired fits for seeds 302 and 303 on `mit_preemptable`, with 16 CPUs and 64 GB per task on the matching-CPU node1411.
The serial/parallel native preflight establishes unchanged evaluation results across the tested execution modes; worker count changes allocation throughput rather than the ordered sampler target or random stream.
Full fits preserve complete-stage checkpoints for preemption recovery and retain the original 16,448-evaluation budget.
The weighted future adapter and independent prediction checks are now implemented and validated as described below.
Prediction stability, comparison with the incumbent and adequate numerical exploration remain unproven.
The production estimator remains unchanged.

## Weighted future pipeline

The frozen adapter is in `logs/uncertainty_fan_static_forecasts_20260914`.
It recovers a completed source checkpoint without another fitting evaluation and verifies the new proposal-to-canonical-coordinate mapping for every positive-weight particle.
Each particle retains its complete sampled scene, original weight and complete 132-action trajectory.
The changed static-fixture likelihood is used consistently in prefix checks and future scoring.
The full history is generated before future observations are scored, and all zero-density contributions remain in the mixture.

The short fitting fixture `22712635` completes with 7,808 native actions.
Its full forecast fixture `22712670` completes 64 weighted histories plus one exact repeat, using 8,580 native actions.
Independent reader `22712683` verifies all complete histories and their weights, the completed checkpoint, every prefix and future likelihood, and the ball's conditional output moments.
It independently reconstructs mixture variance using within-component variance plus deviations from the weighted mean, checks a nonuniform two-component reference, and recomputes feature errors, event scores and goal scores.
All eight corruption checks pass, rejecting a dropped particle, changed weights, native predictions, events, goals, density, conditional means and conditional variances.
The deliberately short fixture is numerically unassessed, and its prediction metrics are not evidence for the alternative model.

Full forecasts `22712813` and `22712819` are queued behind the respective running fits and the passed fixture reader.
Their independent readers are `22712820` and `22712825`.
Comparison `22712836` depends on both readers and on the comparison fixture `22712826`.
The comparison fixture has passed, reproducing both earlier tempered populations and correctly preserving the two new rows as pending.
It runs on the matching-CPU node1412 after only its pending resource request was changed; no frozen code or running fit was modified.
The full comparison retains both original and new numerical replicas and the unchanged incumbent forecast, checking identical data, program, original physical prior and sampling budget while recording the changed discrepancy model.
Both fits, full forecasts, independent readers and comparison `22712836` have now completed.
The following results supersede the pending status above.
All new forecast hashes match their independent verification reports.

## Completed static-fixture comparison

| Configuration | Fitting seed | Ball-position RMSE (m) | Goal Brier score | Final goal probability | Zero-density histories |
|---|---|---:|---:|---:|---:|
| Incumbent selected point | N/A | 0.007072 | 0.044118 | 1.0000 | N/A |
| Original discrepancy | 302 | 0.005348 | 0.017019 | 0.8531 | 64/64 |
| Original discrepancy | 303 | 0.008733 | 0.064487 | 0.5456 | 63/64 |
| Static-fixture sensor model | 302 | 0.005858 | 0.018993 | 0.9844 | 57/64 |
| Static-fixture sensor model | 303 | 0.006059 | 0.020533 | 0.8645 | 62/64 |

The new model reduces between-fit native position RMS disagreement from 0.006693 m to 0.000619 m.
Both new goal Brier scores are better than the incumbent's on this development recording, and the formerly weaker numerical replica improves.
However, maximum goal-probability disagreement over the future remains 0.36726, compared with 0.37100 previously.
Most positive-weight histories still contradict at least one exact future observation, and the two fits retain one and two original lineages.
The model change therefore improves some forecasts without closing the numerical or predictive gate.
These are offline predictions, not agent solve-rate results.

The new fits use 15,453 and 15,492 target evaluations, close to the original 15,455 and 15,550.
Their shorter elapsed times also reflect the increase from four to sixteen compute workers and must not be attributed solely to the observation-model change.
Each forecast uses 8,580 native actions including its repeated history, separate from fitting and reader costs.
