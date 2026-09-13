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

Array `22683118` completed both new replicas, seeds 100 and 101, from the original prior.
Their allocations took 1:50:43 and 1:50:29 respectively on four CPUs.
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

## Checkpoint-driven posterior forecasts

The posterior forecast adapter is implemented in `logs/uncertainty_domino_transition_posterior_forecast_20260913/`.
A short native sampler fixture, `22683423_0`, completed with 16 particles, two temperatures and at most 48 evaluations.
It is a checkpoint and forecast integration fixture, not an assessed posterior or a new agent seed.
Restoring its completed checkpoint reproduces the complete saved sampler result without another target evaluation.

For each positive-weight fitted particle, the adapter generates four unconditional futures in each of two independent random-number banks.
It retains the original particle weight, divided only by the number of draws within each bank.
Position means and variances integrate the scalar output-error process conditional on the fitting prefix; native toppling events come from the generated physical trajectories.
A separate history conditions on the actual future joint readings solely to evaluate the recorded suffix density.
Those future-conditioned histories never contribute to unconditional position or toppling predictions, and suffix observations never reweight the fitted particles.
Zero particle weights and zero predictive densities remain explicit.

Forecast fixture `22684064_0` completed in 150 allocation seconds on four CPUs, with 23,506 native actions including two exact complete-history repeats.
Independent verification `22684436` completed in 31 allocation seconds on one CPU.
It reads all 144 saved histories, checks the complete checkpoint population and original weights, reconstructs the scalar Gaussian filtering and forecast moments, and recomputes all weighted summaries.
It independently checks 208,656 Gaussian joint factors, with maximum absolute log-factor discrepancy 8.882e-16.
Deliberately changed weights, generation/density roles, prefix values, forecast moments, future joints and joint-density factors are all rejected.
These checks establish artifact consistency and arithmetic for the fixture, not statistical adequacy of its short fit.

The two full forecasts, `22684522_1` and `22684523_2`, completed together with their independent verifiers in 11:50 and 11:49 respectively on four CPUs.
Each checks the frozen fixture-validation gate before generating predictions and runs the independent artifact verifier after finishing.
The declared forecast budget is two banks of four draws per positive-weight particle plus one separate density history per particle, using four CPUs for at most 45 minutes.
With all 64 weights positive, this requires 93,058 native actions including the two complete-history repeats.
Between-bank disagreement measures future-simulation Monte Carlo variability; it does not measure posterior uncertainty or establish convergence of parameter inference.

The paired report retains the earlier exploratory screens: 2.5 mm RMS disagreement in conditional position means, 0.20 maximum toppling-curve gap and 0.15 maximum final toppling gap.
It compares the new replicas against the original 64-particle point-start replicas while requiring identical prior, fitting data, program and sampler settings.
The joint-discrepancy law is explicitly required to differ.
Missing forecasts or verification reports leave the comparison incomplete.
Uncertain initial-state inference, wider development coverage, acceptable inference cost and closed-loop acceptance remain separate requirements.

Comparison-reader validation `22684684` reproduces identical fixture summaries, detects injected position and event regressions, rejects misaligned coordinates and preserves an incomplete status when either new forecast is missing.
Finite report job `22684725` completed after both full forecasts and verifiers; it is not a notification monitor.
The bundle's `verified-inputs.json` pins the tested scripts, fixture reports and job mapping.

## Completed comparison and larger-budget follow-up

The new replicas agree to 0.215 mm in position means, but their maximum toppling-curve and final-toppling gaps are both 0.166830.
The final gap exceeds the predeclared 0.15 limit, so the alternative still fails its numerical replication screen.
The older joint-output model's matched 64-particle pair has 0.880 mm position disagreement and a 0.012149 toppling gap; its separate larger-budget failures remain recorded in the earlier comparison.
Agreement at this single budget does not approve either model.

| Model and numerical seed | Position RMSE (m) | Toppling Brier score | Final-toppling Brier score | Forecast native actions |
| --- | ---: | ---: | ---: | ---: |
| Joint output error, 100 | 0.0114410 | 0.000002870 | 0.000148375 | 10,465 |
| Joint output error, 101 | 0.0113989 | 0.000003864 | 0.000293803 | 10,465 |
| Joint transition, 100 | 0.0113551 | 0.000119152 | 0.011549145 | 93,058 |
| Joint transition, 101 | 0.0113573 | 0.000018461 | 0.001548140 | 93,058 |

These are held-out suffix predictions on one development recording, not solve rates or agent seeds.
The new model's lower position error accompanies worse toppling scores on this recording.
Its two future banks differ by 0.084 and 0.158 mm in position means and by 0.013326 and 0.006608 in toppling curves, substantially less than the disagreement between fitted replicas.
The two full verifiers each check 834,624 joint factors with maximum error 8.882e-16, preserve original weights and reject all six corruption classes.

The new fits retain only six and five initial ancestors.
Mass medians differ, approximately 0.176 versus 0.113, as do spinning-friction medians, approximately 1.263 versus 0.614.
The next experiment tests budget sensitivity rather than assuming that more accepted proposals imply reliable uncertainty.

Array `22688869`, in `logs/uncertainty_domino_joint_transition_128_20260913/`, runs matched seeds 100 and 101 with 128 particles and a 32,896-evaluation cap.
The original prior, law, program, data, temperature schedule and proposal settings are retained.
Forecast jobs `22689058_0` and `22689059_1` depend on their respective successful fits and reuse the exact verified forecast and reader implementations.
Each requests four CPUs for at most one hour and checks all generated artifacts after completion.
The expected cost with 128 positive weights is 185,794 native actions per forecast, including two repeats.

The budget report `22689095` retains all six within-budget and cross-budget comparisons among the four new-model populations.
Its reader checks matching prior, data, sensor and program identities and all sampler settings except particle/evaluation budgets.
The runtime identities legitimately differ because they include the budget-specific plan; each is independently checked against its frozen plan, scripts and preflight rather than requiring the two runtime hashes to match.
Follow-up reader check `22689407` passes, and both live 128-particle reports match their independently reconstructed expected runtime identity.
Reader test `22689104` passes identical-summary, injected-regression and misaligned-coordinate controls and correctly reports the current two-population comparison as incomplete.
The follow-up forecast/report bundle is `logs/uncertainty_domino_transition_128_forecast_20260913/`.

Both larger fits have now completed all 32 temperatures.
Seed 100 used 31,921 evaluations in a 3:39:43 allocation and retained 15 initial ancestors; seed 101 used 31,959 evaluations in 3:40:32 and retained 10.
Each recorded five resampling events.
Their complete sample populations and weights match the checksummed checkpoints exactly.
The dependent native forecast/verifier jobs `22689058_0` and `22689059_1` are running, with the budget comparison `22689095` still queued behind them.
Completion and ancestor counts do not establish that the previously observed toppling-prediction disagreement has been resolved.
All results remain a fixed-initial-state ablation and do not close Stage B.
