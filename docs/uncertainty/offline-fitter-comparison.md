# Offline fitter comparison setup

September 12, 2026.

This comparison connects the incumbent's complete fitting pipeline to the same public observation ledger used by the posterior experiments.
It is a cold-fit development comparison on fixed programs, not a reconstruction of historical agent conversations or evidence of closed-loop improvement.
The [simplification proposal](simplification-proposal.md) still requires independent posterior validation, prediction comparisons and matched live runs before retirement.

## Shared evidence and legacy path

`RecordingProjection.to_state()` reconstructs object features and public joint/base observations using object handles owned by the visible model.
It preserves the supplied floating-point values, creates new feature arrays and uses canonical object insertion order.
It rejects missing features, unmatched objects, duplicate handles, unknown channels, discontinuous joint indices and incomplete base poses.
It does not read private recording velocities, attachment frames, inferred memory or evaluator state.
Its result is an observation frame for the incumbent fitter, not a feasible physical initial state for joint inference.

Both paths use the same declared step-keyed noise reconstructed by `load_recorded_level()`.
The adapter verifies an exact projection round trip before a frame reaches the legacy fitter.
The compute preflight checked all 162 Domino frames and 133 Fan frames from the selected first training levels, with no changed or dropped public values.
These recordings contain 161 and 132 actions respectively, each in one reset episode.
Seven functional tests and two-file type/lint/format checks passed in job `22637746`.

The comparison invokes the frozen approach's actual `_rollout_fit_trajectories()` preparation, followed by `run_rollout_sysid()`.
It retains the configured settled-tail truncation, rest segmentation, noise filter, residual scaling, trimming, identifiability reports, interval handling, evidence calculation and trust selection.
It does not replace the incumbent with a least-squares proxy.
Registry scale stamping and prior anchors are obtained from the same visible model.
The comparison starts with no carried fit history or cache, making the meaning of a cold fit explicit even though the experiment configuration enables carrying for successive fits.
Sequential carried-prior comparisons remain a separate required comparison.

Reports retain fitted and applied parameters separately.
Subsequent predictions use the applied values merged with the program's declared initial values, which exposes refusals to apply a fit rather than silently evaluating an unpublished optimum.
Predictions replay each complete recorded action sequence from its initial observation using the incumbent's existing rest-start assumption.
The original observation ledger is checked again after fitting to detect unintended mutation.
Worker time, constructed worlds, rollout calls and primitive simulation steps are recorded alongside per-feature prediction errors.
The experiment retains six configured rollout workers and requests six CPUs.
Step and world telemetry is collected at disconnection in each process, so completed child rollouts are included instead of reporting only parent-process work.
Running reports exclude unfinished child worlds until their disconnection; final reports include them.
Jobs `22638339_0` and `_1` verified this accounting for Domino and Fan using four three-step rollouts across two forked workers: each report contained exactly 12 steps and five worlds including the unstepped reference world.

Canonical object ordering is a declared runtime control shared with the new initialization path.
It is not a claim of bitwise reproduction of historical fit calls with their original insertion order and carried history.
Keep runtime-control changes separate from estimator effects in the final matched agent experiments.
No acting-agent code or production defaults were changed by this comparison setup.

## Fan prior provenance

The saved Fan programs have identical dynamics-method syntax after excluding docstrings, but their parameter declarations changed:

| Version | Initial speed | Declared bounds | Optimizer scale |
| --- | --- | --- | --- |
| 1 | 0.05 | [0, 1] | Log, invalid with zero lower bound |
| 2 | 0.05 | [0.001, 2] | Log |
| 3 | 0.0846 | [0.077, 0.092] | Linear |

Version 3 explicitly derives its narrow interval from the recorded trajectory.
Reusing that interval as an independent original Bayesian prior would reuse information from the fitting data.
The earliest optimizer declaration also cannot define a proper log-uniform prior on its support: the integral diverges at zero, and the repository rejects the declaration.
The invalid original artifact is retained rather than silently changing its bound or importing it as if executable.

For the planned posterior comparison, declare `fan_speed ~ Uniform(0, 1)` using the earliest saved support and an explicit normalized density.
This is a new, fixed experimental prior declaration, not a claim that the incumbent had this probability distribution or that optimizer scale determines prior density.
It does not reuse the fitted value or narrow fitted interval as its center or support.
The program itself was synthesized from training data, so this is a fixed-program development comparison, not evidence of prior specification before all observation of the domain.

The latest executable program remains fixed.
A compute-node audit confirmed that its parameter override accepts 0, 0.5 and 1 exactly without clipping to its current fitting bounds.
The method-syntax check confirmed the same dynamics across all three saved versions.
The legacy arm retains its current declarations and fitting bounds; its prior and deployment policy are part of the incumbent algorithm being compared.
A future ablation must separate the effect of the fixed original prior from uncertain-state inference.
Fan's [full initial-state prior](fan-initial-scene.md), including static layout and articulated switch state, now has a complete-recording support witness.
A numerically adequate joint-posterior approximation remains required before comparing estimators.

## Submitted comparison jobs

Array `22638308` is submitted to `mit_preemptable`, pinned to the same `node1412` CPU and explicit Python hash seed as the controlled Domino posterior runs.

| Array task | Domain | Fitting data | Prediction use |
| --- | --- | --- | --- |
| 0 | Domino | Initial frame and first 64 actions | Remaining 97 actions held out from this fit |
| 1 | Fan | Initial frame and first 64 actions | Remaining 68 actions held out from this fit |
| 2 | Domino | All 161 training actions | Reconstruction; no unused suffix in this recording |
| 3 | Fan | All 132 training actions | Reconstruction; no unused suffix in this recording |

The suffixes are not established as unseen during historical program synthesis.
All four tasks have now completed and supply incumbent prediction baselines; this does not establish posterior numerical adequacy or an agent advantage.
The frozen worker uses historical runtime `b09217bb3` plus the identified offline modules, the same physics source as the current Domino posterior pilots.
The eight-hour job limit is an external compute cap, not a statistical stopping criterion or evidence that an interrupted fit completed.

The first preflight failed only on Fan because it assumed every visible environment had `_components`.
The corrected object lookup supports direct environment-owned objects as well as optional components.
The still-pending first comparison array `22637634` was cancelled before execution after that setup failure.
Its still-pending replacement `22637957` was superseded by `22638308` to allocate the configured six workers correctly and include child-process telemetry.
Corrected preflight jobs `22637747_0` and `_1` both completed successfully.
Infrastructure/setup outcomes are not agent seeds.

Artifacts are in `logs/uncertainty_legacy_preflight_v2_20260912`, `logs/uncertainty_observation_state_v3_20260912`, `logs/uncertainty_legacy_telemetry_20260912` and `logs/uncertainty_legacy_comparison_v3_20260912`.

## Completed incumbent comparisons

All four tasks completed successfully on their declared node1412 with their original frozen configuration.
The retained segments are the incumbent's prepared fitting units; segmentation can overlap, so their action counts must not be interpreted as distinct observations.

| Task | Domain | Supplied actions | Retained segments | Worker seconds | Native steps | Worlds |
| --- | --- | --- | --- | --- | --- | --- |
| 22638308_0 | Domino | 64 | 3 | 214.39 | 13,625 | 563 |
| 22638308_1 | Fan | 64 | 1 | 30.88 | 1,777 | 37 |
| 22638308_2 | Domino | 161 | 6 | 279.03 | 20,753 | 704 |
| 22638308_3 | Fan | 132 | 3 | 52.38 | 4,461 | 119 |

Both Domino fits predict using the original anchor values: lateral friction 0.674, restitution 0.02, rolling friction 0.006, spinning friction 0.5 and mass 0.1.
The 64-action fit reports four parameters as anchored and restitution as insensitive.
The full-recording fit reports four as anchored and mass as not identified, retaining its anchor for prediction.
Both Fan fits predict with speed 0.0846 and report it as anchored, with legacy belief interval [0.077, 0.092].
These values describe the incumbent publication policy, not a posterior inference result or proof that the recordings contain no parameter information.

Each Domino report stores 161 predicted frames and each Fan report stores 132.
The 64-action arms leave suffixes of 97 and 68 actions unused by those fits respectively; the full-recording arms leave none.
These outputs are ready for a matched prediction comparison when replacement inference passes its numerical checks.

Fan's 64-action cold legacy fit, task `22638308_1`, completed on the originally declared node1412.
Its preprocessing produces one 47-action segment and retains that segment through fitting.
The internal fitted move to approximately 0.085223 is rejected by the incumbent anchor test as data-equivalent; the final applied `fan_speed` remains 0.0846.
The legacy belief interval remains [0.077, 0.092], with its original `anchored` verdict.
Those are legacy quantities, not posterior credible intervals.

The [completed report](../../logs/uncertainty_legacy_comparison_v3_20260912/pilot-22638308_1.json) includes predictions for all 132 training actions, including the 68-action suffix unused by this particular fit.
That suffix is not certified as unseen during historical program synthesis.
The worker records 30.88 seconds, 1,777 native steps and 37 worlds across parent and child processes.
No adequate replacement posterior is yet available for a matched estimator conclusion.

An [eligibility review](../../logs/uncertainty_legacy_comparison_v3_20260912/eligibility-review.json) verified a second matching CPU node, but jobs began on their original node before eligibility was changed.
No source, prior or resource changes were applied, and all temporary scheduling holds were released and checked.
