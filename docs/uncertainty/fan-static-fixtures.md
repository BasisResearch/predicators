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
The weighted future adapter and independent prediction comparison for this new observation model are still required.
Prediction stability, comparison with the incumbent and adequate numerical exploration remain unproven.
The production estimator remains unchanged.
