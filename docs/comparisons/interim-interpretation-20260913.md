# Continual comparison interpretation, 2026-09-13

Snapshot checked at 08:17 UTC against final scorecards and terminal experiment logs.

**The user approved retaining this arm as No harness fitting, disclosing agent-written numerical dynamics fitting.**
The harness fitting API is disabled, but this does not remove all numerical estimation.
Agent-written dynamics fitting is confirmed in Boil seeds 0, 1, 2 and Balloons seeds 1, 2; the other seeds have not been fully audited.
The frozen prompt originally discouraged custom fitting; the narrower interpretation was adopted after observing those calls.
See the [protocol audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/no-fitting-protocol-audit.json).
This note interprets the completed groups; the [live comparison table](/home/ycliang/predicators/docs/comparisons/continual-results.md) remains authoritative for later outcomes, per-seed steps, resets, and source paths.
There are 50 finished non-Bridge comparison seeds, including 44 whole-run successes, and three finished Bridge MF follow-up seeds.
The experiments are still incomplete.

## Completed comparison groups

Each cell gives whole-run successes out of three seeds, followed by mean steps over those successes and its qualifying count.
Whole-run success requires every training and test level to be won.

| Method | Fan | Domino | Boil | Original Balloons |
|---|---|---|---|---|
| Oracle dynamics | 3/3; 408.7 (n=3) | 2/3; 345 (n=2) | 3/3; 549.3 (n=3) | 3/3; 379 (n=3) |
| Oracle scene | 3/3; 513.7 (n=3) | 3/3; 319 (n=3) | 3/3; 945.7 (n=3) | 1/3; 335 (n=1) |
| Zero-shot model | 3/3; 451 (n=3) | 3/3; 425.7 (n=3) | 3/3; 724 (n=3) | 2/3; 388.5 (n=2) |
| No harness fitting | 3/3; 344 (n=3) | 2/3; 455.5 (n=2) | 3/3; 791.3 (n=3) | 2/3; 609.5 (n=2) |

The standalone-program and no-explicit-uncertainty groups are incomplete and excluded from this completed-group table.
The full table also reports resets; infrastructure interruptions do not enter any agent average.

## What these results support so far

Zero-shot models solve all three seeds in Fan, Domino, and Boil.
Consequently, these samples do not support claiming that learning dynamics from interaction is necessary for success in every domain.
The zero-shot arm still adapts its actions and journal during continual interaction; only its dynamics and parameter values are sealed before the first real action.

The no-harness-fitting arm solves every Fan and Boil seed, but this cannot establish that numerical fitting is unnecessary.
Executed tool logs show least-squares estimates of filling dynamics in all three Boil seeds, and numerical fitting of Balloons dynamics in seeds 1 and 2.
The harness fitting API was disabled; arbitrary agent code still performed fitting.
The observed outcomes are retained under the user-approved no-harness-fitting interpretation; no rerun is requested.

Original Balloons is more discriminating among the completed methods: oracle dynamics wins 3/3, oracle scene 1/3, zero-shot 2/3, and no harness fitting 2/3.
These are small-sample descriptive differences, not established effect sizes.
The no-harness-fitting results have the narrower interpretation described above.
In particular, oracle scene's 335-step mean includes only its one successful seed; it is not evidence that oracle scene is more efficient overall than a method that solves all seeds.

Correct supplied dynamics are not a guaranteed successful controller: the Domino oracle-dynamics arm loses one test level to a recorded game-over.
Do not replace that outcome with an oracle mechanical audit or interpret this arm as a guaranteed performance ceiling.
The terminal logs establish the recorded outcome; agents' explanations of their own modeling or physical failures remain hypotheses unless independently reproduced.

## Cohorts that must stay separate

The [original MB/MF sweep](/home/ycliang/predicators/docs/uncertainty-results/noisy-sweep-table.md) used an earlier runtime.
The comparison runtime restored the original task selection and action capabilities, but is not byte-identical to that historical runtime.
These are useful reference results, not proof that every observed difference is caused solely by the named ablation.

The [new Bridge follow-up](/home/ycliang/predicators/docs/comparisons/bridge-span-followup-results.md) changes training-to-test span size from three to four blocks.
Its three fresh MF seeds each win training and fail test, with total steps 3131, 4654, and 4507 and resets 0, 3, and 0.
There are no successful whole runs, so mean successful steps is unavailable (n=0).
The earlier pilot has one successful MB run and one successful MF run; it stays separate from these fresh seeds.
That pilot is insufficient to establish a reliable MB advantage, and the 18 Bridge comparison seeds are still pending.

The [fixed Balloons MF cohort](/home/ycliang/predicators/docs/comparisons/bridge-balloons-integrity-results.md) requires sustained hovering and is also separate.
Its three successes cannot be substituted for an MF control under the original instantaneous goal used in the table above.
No MB run under that changed goal was requested.

## Remaining evidence

Finish the standalone-program and no-explicit-uncertainty groups in all four original non-Bridge domains and all six comparison methods on the new Bridge variant.
Keep the no-harness-fitting interpretation and agent-written-fitting disclosure in the final report.
Only then can the full six-method sweep be reported as complete.
The current partial uncertainty-ablation results do not establish whether explicit uncertainty improves success or sample efficiency.

Scorecard and terminal-log verification for this snapshot is saved in the [outcome audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/outcome-audit-20260913T081745Z.json).


## Uncertainty protocol audit

A later audit confirmed that original Balloons seed1 in the no-explicit-uncertainty arm executed an agent-written parameter sweep.
It evaluated release choices over 4001 red-lift values at each of three possible mass ratios and printed intervals that would reach the goal.
The supplied uncertainty tools were disabled, but the run does not establish strictly point-estimate reasoning.
The user has been asked whether to retain a narrower no-harness-uncertainty-tools interpretation or pursue a stricter implementation and rerun.
Point fitting, observation averaging, and controller-parameter searches are not themselves evidence of a violation.
See the [executed-code evidence](/home/ycliang/predicators/logs/continual_comparisons_20260912/no-uncertainty-protocol-audit.json).
