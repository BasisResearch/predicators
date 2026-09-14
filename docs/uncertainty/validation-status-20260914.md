# Uncertainty replacement: validation status on September 14

The replacement remains in offline Stage B validation.
The production MB parameter fitter, execution estimator and decision rules are unchanged.
These experiments use fixed development programs and recorded interactions; numerical fitting seeds are not agent seeds.
No result below establishes a new solve rate or permission to retire the current estimator.

## Agreed implementation scope

Replay each candidate from initialization using the recorded low-level actions.
Arbitrary mid-trajectory restoration is an optional optimization and is not a migration requirement.
Initial-state uncertainty and candidate-specific simulator memory remain part of the inference problem.
Fix reproducible replay defects and missing program mechanisms before attributing residual mismatch to an explicitly declared discrepancy law.
Keep sensor noise fixed and test any discrepancy extension consistently in fitting and future generation.
The [proposal](simplification-proposal.md) records the full implementation decisions.

## Completed comparisons

| Domain | Latest evidence | Remaining difficulty |
|---|---|---|
| Domino | Both 64- and 128-particle comparisons are complete, including the fixed-state joint-transition alternative. | That alternative passes its 128-particle replica screens but retains a 0.1928 cross-budget final-toppling gap; uncertain-state composition is now queued for validation. |
| Fan | Fixed-fixture treatment reduces between-fit positional disagreement from 6.69 mm to 0.619 mm, with improved goal Brier scores on this recording. | Maximum future goal-probability disagreement remains 0.367; most sampled histories assign zero density to the recorded future. |
| Boil | Full reduced fits, forecasts, independent readers and comparison are complete. | The thermal prior is now represented correctly, but both fits retain one initial lineage and predictions remain worse than the incumbent. |
| Bridge | Both resumed fits, complete forecasts and independent readers are complete. | Both fits retain one initial lineage; all methods miss the final clean geometric goal, and the two posterior fits disagree on glue events. |
| Balloons | Original, default-guided and selective comparisons are complete and independently verified. | Selective proposals broaden some marginals but worsen height error and predict 97-99% final burst probability; the short fitting prefix also excludes later color releases. |

The earlier source-program audit found no learnable parameters in the saved Boil and Bridge programs from the original noisy sweep.
Those remain incomplete-model controls.
The positive Boil and Bridge comparisons instead use separately identified historical programs with learnable dynamics, including their learned defaults.
Their selected-point incumbent rows therefore carry historical information that an uninformative fitting prefix cannot recreate from a broad parameter prior.
This is a known comparison difference, not evidence that sensor noise should be enlarged or priors narrowed after seeing future scores.
See the [historical model controls](historical-model-controls.md) and [Boil incumbent control](boil-incumbent-control.md).

The new [Domino composition check](domino-uncertain-transition.md) preserves the original initial-scene prior and combines it with the already declared joint-transition law.
Fourteen cases include archived joint proposals, fixed-state controls and unselected proposal vectors.
Native `22766058` and independent reader `22766059` are pending; preparation checks alone do not establish native parity or an adequate uncertain-state posterior.

## Boil reduced-target outcome

The completed source is `logs/uncertainty_boil_reduced_comparison_20260914/full.json`.
All 42 referenced source hashes were checked after completion.
Both full fitting readers and both complete forecast readers passed.

| Method | Numerical seed | Bubbling RMSE against clean readings | Final goal probability | Surviving initial lineages |
|---|---:|---:|---:|---:|
| Incumbent selected point | N/A | 0.01915 | 1.0000 | N/A |
| Original supported full target | 410 | 0.33856 | 0.5637 | 1 |
| Original supported full target | 411 | 0.31636 | 0.7691 | 1 |
| Reduced target with independent thermal priors | 410 | 0.30445 | 0.6836 | 1 |
| Reduced target with independent thermal priors | 411 | 0.37039 | 0.5156 | 1 |

The dimension reduction preserves the original probability model but does not solve exploration of the retained scene and parameters.
It improves bubbling error in one fit and worsens it in the other.
The maximum future goal-probability gap between reduced fits is 0.1680.
One initial lineage is a warning about exploration, not by itself proof that every reported posterior moment is wrong.
The independent-replica and budget evidence remains necessary.

The original 132-action prefix never activates the burner.
The three heating parameters should retain their prior under that prefix, even when a historically learned default predicts the future better.
New study `logs/uncertainty_boil_informative_heating_20260914` compares conditional heating inference at 132 and 224 actions using the same declared sensor variance and original thermal priors.
It selects two candidate scenes using fitting weight and a fixed index tie-break, holds nonthermal quantities fixed, and checks two quadrature budgets.
The longer prefix is a new development case; observations after action 224 and clean evaluator readings are excluded from its inference.
This study can establish conditional learnability, but cannot establish an adequate full-scene posterior or a live-agent improvement.
Native job `22755839` and independent reader `22755840` completed successfully on `mit_preemptable`.
The follow-up precision check `22756314` and independent reader `22756315` also completed; the 1,024/2,048-node comparison passes its declared moment and log-normalizer tolerances.
Conditional onset and width become concentrated at means 30.12237 and 5.71475, while radius remains uniform in these two selected histories.
The two histories give the same conditional heating target, not independent evidence of full-scene posterior agreement.
See the [informative-heating diagnostic](boil-informative-heating.md) for the original failed precision comparison, prior-retention control and remaining integration requirements.

The subsequent [complete 224-action joint target](boil-heating-joint-target.md) passes native preflight `22757366` and independent reader `22757367`.
It retains all 84 joint coordinates and includes every observation channel under the original priors and noise laws.
The corrected thermal proposal also passes native preflight `22757771` and independent density and mapping reader `22757772`.
All four matched uniform and guided fitting fixtures and their recovered readers now pass on `mit_preemptable`.
The [reader recovery](boil-heating-reader-recovery.md) corrects only a tuple/list transport mismatch, preserving all numerical checks and reusing the completed fits.
The dependency-cancelled downstream jobs are replaced with identical scientific inputs and budgets; verified runtime references support six-hour full-fit and two-hour reader allocations.
All four full fits are running, and all four forecast fixtures now pass independent verification of their complete histories and weighted summaries.
The [40-action forecast adapter](boil-heating-forecasts.md) passes native fixture `22759308` with 5,120 simulator actions.
Independent reader `22759309` verifies all twelve artifacts and sixteen corruption controls, with another 1,424 native actions.
Weighted population check `22759907` also passes its proposal mapping, nonuniform moment and zero-density controls.
Remaining forecast work follows its fitting and fixture dependencies.
Predictive and numerical comparisons remain unfinished.
The [matched collector](boil-heating-comparison.md) passes its known-error, weighted-covariance, identity and incomplete-source checks in `22761384`; replacement collection `22762518` waits on all four complete forecast readers.
The matched-prefix incumbent fit `22761363` and independent reader `22761364` have completed.
Its point prediction matches all five clean event curves and the final goal; water-volume RMSE is 0.008744 and bubbling error is zero on the already-saturated suffix.
Its configuration and parameter specifications are verified unchanged from the earlier control.
The combined incumbent/posterior report check `22761575` passes all eleven rejection controls and retains the real inputs as incomplete.
Replacement final collector `22762519` remains queued behind the verified incumbent and complete posterior comparison, with recovery provenance and allocation costs retained.

## Bridge completed recovery

The completed source is `logs/uncertainty_bridge_resumed_forecasts_v2_20260914/paired-comparison.json`.
All nine directly referenced sources match their recorded hashes.
The original 64 GiB allocations failed with an out-of-memory condition; the checked 128 GiB continuations completed without changing the numerical target or resetting its budget.

Numerical seeds 810 and 811 used 10,834 and 10,825 fitting evaluations, respectively, and each ends with one initial lineage.
All three methods, including the incumbent selected point, predict final goal probability zero.
The clean assessment contains the goal at the final frame, while evaluating the same geometric predicate on noisy poses misses it.
Their identical goal Brier score of 1/586 therefore conceals a consequential final-frame error rather than demonstrating success.
The maximum between-fit glue-probability gap is 0.1512, and maximum per-coordinate positional RMS disagreement is 8.49 mm.
The future-density calculation has 26/32 zero-density particles in seed 810 and none in seed 811.
These findings keep predictive and numerical acceptance open.

The final report's fitting time describes the resumed allocation.
Compute accounting must also retain each original 2:51:51 allocation, recovery checks, initialization, forecast generation and independent verification.
Do not interpret resumed elapsed time as total fitting cost.

## Balloons invocation recovery

Selective forecast jobs `22715569` and `22715571` completed, but readers `22715570` and `22715572` failed before their Python verifier started.
Their launcher requires both the report path and numerical index; its submission omitted the second argument and shell expansion failed with `$2: unbound variable`.
The dependent comparison `22715574` was cancelled.
This is a verification-launch failure, not a failed fit, prediction, or agent seed.

The exact failing expansion was reproduced in `logs/uncertainty_balloons_selective_reader_recovery_20260914/reproduction.json`.
Recovery array `22755276` explicitly supplies report and index to the unchanged frozen launcher and verifier.
Comparison `22755277` depends on successful completion of both readers.
Both recovered readers and comparison `22755277` have now completed successfully.
No completed fit or forecast was regenerated, and old frozen source files were not edited.
The recovery manifest pins the original reports, scripts and invocation wrapper.
The [completed selective comparison](balloons-selective-guidance.md#completed-comparison-and-information-limitation) worsens height error in both fits despite broader marginals.
It also identifies a separate information limitation: the 64-action prefix ends before the red and green releases assessed in the continuation.

## Next acceptance work

For Balloons, distinguish prediction after an observed release from extrapolation to an unseen color using a separately identified longer-prefix diagnostic.
For Boil, finish the active heating-aware joint fits and their forecast comparisons, retaining the original all-off control.
Use their results to choose a numerical or model change rather than repeat the same unstable fits at a larger budget without a specific hypothesis.
Existing conditional scalar-discrepancy, joint-variance and physical-transition experiments remain distinct model choices.
Any new law motivated by these already-inspected future recordings needs a new untouched evaluation before acceptance.
Live posterior integration, matched five-domain continual evaluation, and retirement of the current parameter fitter remain required later stages.
