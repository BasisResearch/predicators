# Bridge glue model: structural and geometry errors

This Stage B development diagnostic separates a wrong glue-update law from accumulated native trajectory error.
It extends the [historical model audit](historical-model-controls.md), retaining both the original no-op and transferred-program controls.
It changes neither the production agent nor the running inference experiments.

## A parameter change cannot reproduce the recorded glue sequence

Four faces in the frozen Bridge training recording show consecutive readings `0 → 0.2 → 0.4 → 1`.
Glue is an exact observed channel in this recording.
The transferred program implements deposition as `min(1, previous_level + glue_rate)`.
Its other glue mutations drain a partial level to zero, consume glue when a bond forms, or retain an existing level.
The rule memory persists between actions and is not reset from later observations.

Starting from zero, the first positive reading forces `glue_rate = 0.2`.
That rate reproduces the second reading but gives `0.6`, not `1`, at the third deposition.
Drainage, retention and bond consumption cannot supply the missing increase.
This contradiction holds even if geometry is allowed to choose deposition or the other transitions arbitrarily on each step.
It therefore rules out fitting this exact sequence by changing any of the transferred program's parameters or initial physical states while preserving its update law and memory contract.
It does not establish that every possible learned Bridge program is inconsistent.

| Face | Four consecutive observation steps |
| --- | --- |
| `span0.glue_end_a` | 86, 87, 88, 89 |
| `span1.glue_end_b` | 121, 122, 123, 124 |
| `span0.glue_end_b` | 155, 156, 157, 158 |
| `span2.glue_end_a` | 199, 200, 201, 202 |

An independent check executes the actual historical deposition function with continuously eligible geometry.
Its default rate produces `0, 0.35, 0.7, 1`; rate `0.2` produces `0, 0.2, 0.4, 0.6000000000000001`.
The floating-point result has the same contradiction as the exact arithmetic argument.
The source's own earlier decision record describes the observed `0.2, 0.4, 1` progression, so the implemented constant-increment law also differs from that recorded modeling intent.

## Correcting saturation alone is insufficient

A diagnostic one-line revision uses rate `0.2` and latches to `1` when accumulated progress reaches `0.6`.
That revision reproduces the isolated progression.
It is chosen after inspecting development evidence, is not a fitted posterior model, and has not been accepted as a replacement.

To separate geometry drift from glue-law errors, each program is also evaluated on all 1,186 recorded post-action public geometries.
The six cases use either the noisy public geometry or its clean public projection, with the historical defaults, rate `0.2` alone, or the saturation revision.
Only the initial glue readings enter the model memory.
Subsequent glue readings are replaced by the model's own carried levels before invoking its rules; dwell counters and bonds also persist.
Attachment commands are counted but not executed, because geometry is supplied from the recording.
No private simulator state enters these rule evaluations.

| Rule variant | Mismatched glue readings with noisy geometry | Mismatched glue readings with clean geometry |
| --- | ---: | ---: |
| Historical defaults | 4,444 | 3,816 |
| Historical law, rate `0.2` | 3,382 | 4,442 |
| Rate `0.2`, latch at `0.6` | 4,442 | 3,814 |

Each column compares 17,790 face readings, including unchanged readings between events.
These counts are descriptive errors, not independent statistical trials.
They are also not causal forecast scores: later recorded geometry is explicitly supplied.
Lower counts from an incorrect law can reflect accidental compensation between errors and do not justify selecting that law.

The clean-geometry cases still show incorrect face selection and event timing.
At steps 87-89, the program deposits on `span0.glue_top`, while the recorded change is on `span0.glue_end_a`.
At steps 122-124, it similarly maintains glue on `span1.glue_top` instead of the observed `span1.glue_end_b`.
Its first deposition on `span0.glue_end_a` starts at step 57, one step before the recorded change, even with clean geometry.
Consequently, neither denoising the geometry nor fixing the saturation progression is sufficient for this transferred program.
The diagnosed face-selection and timing errors do not, by themselves, prove that all geometric parameter settings fail.

## Separating face choice from deposition eligibility

A follow-up tests choosing the face whose center is nearest to the bottle tip, followed by a single distance threshold for deposition.
It uses the historical bottle-tip offset and block-face geometry, without reading private domain mechanics.
The threshold is chosen by exhaustive search over prediction change points within 10 cm, using only actions 1-124 and minimizing incorrect deposition labels.
Already saturated faces are excluded from that label loss because another deposition would be invisible in their glue reading.
No later glue levels enter threshold selection.
Each geometry source is evaluated separately using either its post-action or pre-action frame.

On clean post-action geometry, the nearest face matches the observed face at all thirteen positive glue transitions across the recording.
This resolves the face-choice ambiguity at those positive examples, but does not establish a correct deposition trigger.
The one-distance trigger cannot reproduce the fitting-prefix labels in any of the four cases.

| Supplied geometry | Frame phase | Selected radius | Prefix TP / FP / FN | Suffix TP / FP / FN |
| --- | --- | ---: | --- | --- |
| Noisy | Post-action | 0.040166 m | 7 / 6 / 0 | 3 / 1 / 3 |
| Noisy | Pre-action | 0 m | 0 / 0 / 7 | 0 / 0 / 6 |
| Clean | Post-action | 0 m | 0 / 0 / 7 | 0 / 0 / 6 |
| Clean | Pre-action | 0 m | 0 / 0 / 7 | 0 / 0 / 6 |

TP counts correctly predicted positive transitions, FP counts spurious deposition labels, and FN counts missed positive transitions.
The prefix has 1,825 informative face readings and seven positive transitions; the suffix has 13,702 informative readings and six positive transitions.
The zero-radius solutions predict no deposition and miss every positive transition; they are failures, not acceptable low-error models.
All later geometry remains supplied from the recording, and the suffix had already been inspected during prior development diagnostics, so this is neither a causal prediction test nor an untouched validation set.
The failures do not prove that uncertain latent geometry or a richer physical deposition model has no support.

Compute job `22695397` completed in 16 allocation seconds, followed by independent verifier `22695437` in 17 seconds, both on `node1412` under `mit_preemptable`.
The verifier uses an independent rotation implementation to check 16,320 finite face distances, agreeing within `2.23e-16` m.
It independently scores all 148 tested thresholds, verifies that only prefix labels select them, and checks all four candidates' predicted labels and confusion counts.
The frozen bundle is `logs/uncertainty_bridge_deposition_geometry_20260913/`.

The next geometry hypothesis should preserve the distinction between choosing a face and determining whether a drip reaches it, including height and side-of-plane eligibility.
A scalar distance alone is insufficient on these supplied trajectories.
Any revised program still requires causal native replay and a separately frozen estimator comparison.

## Verification and next work

The frozen bundle is `logs/uncertainty_bridge_glue_attribution_20260913/`.
Job `22695084` completed the six cases in a 35-second allocation on `node1412` under `mit_preemptable`.
Independent verifier `22695164` completed in 25 seconds.
It reconstructs public feature views without a native world, calls the two rules directly instead of using the rule-dispatch helper, and reproduces every glue prediction and attachment-command count across 7,116 rule steps.
It independently checks error totals, all four recorded witnesses, and the actual deposition-function traces.
These jobs perform no recorded native actions and are neither agent seeds nor parameter-inference runs.
The plan and verification manifest pin source programs, recordings, scripts and completed artifacts.

The next supported positive Bridge case requires a revised glue model, including deposition geometry and observation timing, before fitting can become meaningful.
Validate any revision on causal native trajectories and freeze it for both estimator arms; the geometry-conditioned diagnostic cannot establish attachment physics or held-out prediction quality.
Keep this historical program as an explicit inconsistent-model case rather than concealing its exact-output errors with additional sensor noise or a larger sampler.
Stage A/B acceptance remains open, and production continues to use the incumbent estimator.
