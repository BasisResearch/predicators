# Initial-state ablation with a fixed parameter prior

September 13, 2026.
This addresses the Stage B requirement to separate uncertain recording starts from parameter-inference changes in the [simplification proposal](simplification-proposal.md).
The new arm is an explicitly labeled point-state approximation for offline comparison, not a claim that the initial noisy observation reveals the true physical state.

## Controlled comparison

The joint arm uses the complete declared initial-scene law and the five original parameter priors from [Domino joint inference](domino-joint-inference.md).
The point-start arm retains those parameter priors, the same fixed program, the same 64-action observation ledger, and the same scalar/coupled-orientation output model.
It replaces uncertain initial-state inference with one deterministic scene chosen using only the first observation and the previously declared initialization policy.
No later fitting frame, held-out suffix, archived fitted scene or private evaluator state selects that point.

The selection uses the higher-mass rest case, exact observed controlled joints, and coordinate-wise medians of the original first-observation proposal for remaining quantities.
Horizontal body positions and yaw therefore come from the truncated first-reading proposals, support heights follow the declared upright rest geometry, and unobserved Gaussian joint positions take their prior medians.
Rest velocities remain zero as a deliberate point approximation, not a conclusion drawn from missing recording metadata.
The selector coordinate is 0.4 within the original rest interval [0, 0.8), and other initial-state proposal coordinates are 0.5.
If that scene fails geometry or initial-observation support, the protocol reports it unsupported rather than searching for a replacement using later observations.

Writing this point as `x_hat(o_0)`, the ablation targets the original parameter prior times the conditional remaining-recording likelihood at that fixed state.
The output-error process is still conditioned on the first observation, preserving its declared temporal dependence.
The initial-observation density and fixed-state/proposal factor are constant in the parameters at this scene and cancel from their normalized posterior approximation.
The live target checks that the removed factor remains equal to the frozen preflight value for every evaluation.
This is not the full joint posterior with state uncertainty integrated out, and its apparent parameter confidence must be interpreted accordingly.

## Native preflight

Compute job `22673275` completed on `mit_preemptable` with four CPUs.
The median scene passes the declared geometric checks and has finite initial-observation likelihood, with log density 88.31077418.
The median-parameter reference repeats exactly.
All 64 separately sampled parameter settings retain the same initial state and initial-base factor; 39 have finite likelihood over the complete 64-action prefix.
The remaining zero-likelihood settings stay in the report and are not discarded as infrastructure failures.
The preflight performs 4,224 native actions in 42.11 worker seconds.
A finite search failing to find support would not, by itself, prove that the fixed-state model is inconsistent.

The report, selected scene, first-observation identity, parameter proposals and complete outcomes are in `logs/uncertainty_domino_point_start_preflight_20260913`.
The state-selection rule was fixed before those subsequent-action likelihoods were inspected.

## Submitted fits

Array `22673316` runs numerical seeds 100 and 101 under the point-state approximation.
Both use the tested 50/50 local/full-range proposal mixture, 64 particles, 32 cubic-spaced temperatures, eight moves, local scale 0.05 and a maximum of 16,448 evaluations.
Each requests four CPUs, 20 GB and four hours on the same declared AMD worker node, with complete-stage checkpoints and at most two simultaneous tasks.
The worker first reproduces the frozen preflight target and verifies the output-model identity.
Both tasks completed all 32 stages and passed their preflights.

There are five active proposal coordinates, one for each physical parameter, rather than the joint arm's 95-coordinate representation.
The complete output rows retain the constant initial-state coordinates for unambiguous replay.
Blocks partition the active coordinates, so the point-start arm gives each parameter more proposed updates within the same total evaluation budget.
That reduced numerical difficulty is recorded explicitly; prediction differences cannot be interpreted before assessing each approximation's stability.
The ablation does not isolate the legacy carried-prior policy, whose sequential comparison remains required separately.

Frozen source, runtime identities, submission hashes, stage summaries and checkpoints are in `logs/uncertainty_domino_point_start_fit_20260913`.
The production agent remains unchanged, and incomplete fits are not usable posterior results.

## Completed point-start comparison

Fits `22673316_0` and `_1` completed in 5,751 and 5,580 allocation seconds on four CPUs, using 15,927 and 15,952 target evaluations.
They retained five and six initial ancestors, respectively.
Forecasts `22673722_5` and `22673723_6` completed, and summary `22675144` verified all 128 positive-weight complete histories and unchanged source weights.

| Numerical seed | Conditional position RMSE to noisy readings | Toppling Brier error | Final toppling probability for domino 1 |
| --- | ---: | ---: | ---: |
| 100 | 11.441 mm | 0.0000028704 | 0.97016 |
| 101 | 11.399 mm | 0.0000038637 | 0.95801 |

The two replicas differ by 0.880 mm in their conditional position means and by at most 0.01215 in their toppling probabilities, including at the final frame.
All three predeclared exploratory agreement screens pass.
The point-state forecasts have greater position error than the legacy forecast (11.062 mm) and the two mixed joint forecasts (10.906 and 10.854 mm), while their toppling Brier errors are smaller on this reserved sequence.
Neither result establishes an agent advantage or adequate parameter uncertainty.
The comparison is one recording with different numerical difficulty between the state treatments.
A [population-size comparison](domino-budget-sensitivity.md) now tests both targets at 128 particles before interpreting agreement at 64 particles as stability.
