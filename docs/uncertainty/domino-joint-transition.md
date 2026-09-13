# Domino joint-transition discrepancy comparison

This is a different probability model under the [simplification proposal](simplification-proposal.md), not a sampler change under the previous target.
The [transient diagnostics](domino-factor-attribution.md) found that robot joints dominate selected score changes and that their local sensitivities are irregular.
This comparison tests physical joint-transition discrepancy instead of the previous joint output-error process.
The old model and all of its results remain the controls.
No acting agent uses this alternative.

## Declared model

After each native primitive action, let the simulator predict controlled joint positions `q_native`.
The alternative draws each next physical joint position independently from `Normal(q_native, 0.001**2)` in that joint's coordinate units.
For these Fetch configurations, that means radians for the seven revolute joints and meters for the two finger joints.
This is a declared transition-error scale, not sensor noise or a numerical tolerance.
Joint velocities are retained, and positions are neither wrapped nor clipped.
The Gaussian law does not itself certify mechanical bounds, collision support or an accurate contact-force model.

During fitting, the exact observed joint positions analytically determine the transition innovations.
Their Gaussian densities remain in the likelihood once per joint per primitive step.
The corrected physical positions propagate into the following native step.
Every other body state, event, attachment and program memory continues through the simulator.

The nine former Gaussian AR output factors for joint positions are removed in this model.
Their exact sensor checks remain and must agree with the corrected joint readings.
All other output factors, sensor measurements and the checked finger readout remain unchanged.
Adding the former joint output factors after conditioning would count another density at zero residual; the audit explicitly detects that extra normalization factor.

The robot Cartesian fields `x`, `y`, `z`, `roll`, `tilt` and `wrist` retain their native predicted cached-link values from before the correction.
They are never replaced by observed Cartesian values.
This follows the [verified native observation phase](transition-discrepancy.md#preserve-the-native-robot-observation-phase), rather than assuming fresh forward kinematics reproduces historical observations.

The original parameter prior, simulator program, initial point state and 64-action fitting prefix are fixed.
The first-observation factor is unchanged; the parameter-independent initial factor removed by the point-start comparison remains parameter-independent here.
The inference identity includes both the new transition law and the remaining output model.
These point-start fits do not establish an uncertain-initial-state posterior.

## Conditional scoring and future generation

Future generation draws the joint innovations without access to future observations.
The existing normalized output model then generates readings, including temporally correlated discrepancy and exact derived displays.

Future-density evaluation is a separate computation.
Exact future joint readings determine the corresponding innovations and physical continuation, and their densities are retained.
Remaining future observations enter the output likelihood, not the physical correction.
A density-evaluation trajectory conditioned on future joints must never be reused as an unconditional goal prediction.
The joint-only transition law introduces no unobserved direction integral when every corrected joint coordinate is observed exactly.
Other latent initial-state uncertainty still belongs in the posterior population.

## Native validation

Job `22683003` completed in 91 allocation seconds on four CPUs in `mit_preemptable`, using 9,152 native actions.
Its bundle is `logs/uncertainty_domino_joint_transition_20260913/`.
The two observation-module extensions used for future generation preserve the old target exactly at four archived anchors, each checked twice.

All 124 fixed sensitivity cases have finite scores under the alternative model.
One complete case repeats exactly, including predictions, corrected joints and transition factors.
The independent reader checks 74,304 Gaussian joint factors to maximum absolute log-density error 3.553e-15.
It verifies complete joint coverage and the sum of transition and remaining-output likelihoods.
The audit also rejects a deliberately inconsistent joint reading and verifies the extra constant that would arise from incorrectly retaining the old joint output factors.

Four generated 16-action futures reproduce exactly when their generated readings are passed back through conditional density evaluation.
This includes their complete physical predictions and transition/output densities, not just final states.
These are fixed-candidate component checks, not calibrated forecasts or agent seeds.
The largest angular correction in these records is 0.014080 radians at the wrist-roll joint; the largest finger correction is 0.002393 meters.

The native pre-correction joint-response derivatives remain irregular across perturbation scales 1e-6, 1e-5 and 1e-4.
Corrected joint positions equal their observations by construction and are not used to claim zero parameter sensitivity.
The component checks therefore justify a labeled inference experiment, not a claim that the model fixes the numerical problem.

## Bounded inference comparison

Array `22683118` runs two new replicas, seeds 100 and 101, from the original prior.
Both startup checks pass and both runs have initialized their 64-particle populations.
Each uses the same 32 cubic-spaced temperatures, eight moves per temperature, five single-coordinate blocks, scale 0.05, 50/50 local/full-range proposal mixture and 16,448-evaluation budget as the earlier 64-particle point-start comparison.
The only intended statistical change is the explicitly identified joint-transition model.
Each allocation requests four CPUs, 20 GB and four hours on node1412 in `mit_preemptable`.
Complete-stage checkpoints preserve weights, random state, ancestry and cumulative evaluation counts.
The bundle is `logs/uncertainty_domino_joint_transition_fit_20260913/`.

The full-suffix fixture tests generation and density evaluation over all 97 reserved actions before posterior forecasts are trusted.
It also checks that changing future object-position readings while retaining future joint readings does not change the density evaluator's physical history.
The separate bundle is `logs/uncertainty_domino_joint_transition_future_20260913/`.
The first job, `22683253`, completed its native work but failed a reporting assertion that compared in-memory tuples with lists loaded from JSON.
The corrected check canonicalizes that representation while retaining exact numeric equality and saves completed native records before subsequent assertions.
Replacement `22683284` passes in 37 allocation seconds on four CPUs, using 1,771 native actions; the earlier 38-second allocation and its 1,771 actions remain additional compute cost.

All four generated complete futures replay exactly when their generated readings are used for conditional scoring.
Their fitting prefixes and both recorded-suffix density cases match the archived preflight exactly.
Changing future object-position readings while retaining future joints leaves the evaluator's physical predictions and transitions unchanged.
The independent reader verifies all six retained complete histories and 8,694 joint factors to maximum absolute log-density error 8.882e-16, and rejects altered prefix values.
Both fixed-candidate recorded-suffix densities are finite.
This establishes a supported, reproducible full-suffix calculation for those candidates, not a comparison of posterior predictions or model evidence.

After the fits complete, their forecasts must preserve original posterior weights, separate unconditional predictions from conditioned density trajectories, and assess numerical replication and simulator cost.
No fit or forecast in this experiment has been approved for planning or deployment.
