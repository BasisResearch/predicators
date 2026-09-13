# Continual comparison interpretation, 2026-09-13

Results are checked against final scorecards and terminal experiment logs; the live table tracks subsequent outcomes and scheduler state.

**The user approved retaining this arm as No harness fitting, disclosing agent-written numerical dynamics fitting.**
The harness fitting API is disabled, but this does not remove all numerical estimation.
Agent-written dynamics fitting is confirmed in Boil seeds 0, 1, 2 and Balloons seeds 1, 2; the other seeds have not been fully audited.
The frozen prompt originally discouraged custom fitting; the narrower interpretation was adopted after observing those calls.
See the [protocol audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/no-fitting-protocol-audit.json).
This note interprets the completed groups; the [live comparison table](/home/ycliang/predicators/docs/comparisons/continual-results.md) remains authoritative for later outcomes, per-seed steps, resets, and source paths.
All 72 non-Bridge comparison seeds are finalized, including 60 whole-run successes.
Their source identities, terminal outcomes, and step/reset totals are verified in the [complete non-Bridge audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/non-bridge-final-audit-20260913.json).
Bridge oracle-scene seed 2 is also finalized at 2/2 levels, 2,916 steps and zero resets, bringing the full sweep to 73/90 finalized seeds.
The separate primary Bridge MB/MF comparison is complete across three distinct seeds per arm; one replacement Bridge baseline seed is complete and the other 17 are held pending the domain-variant decision.
The full five-domain sweep remains incomplete because 17 Bridge comparisons are held pending the variant decision.

## Completed comparison groups

Each cell gives whole-run successes out of three seeds, followed by mean steps over those successes and its qualifying count.
Whole-run success requires every training and test level to be won.

| Method | Fan | Domino | Boil | Original Balloons |
|---|---|---|---|---|
| Standalone program (protocol caveats) | 3/3; 724 (n=3) | 0/3; unavailable (n=0) | 3/3; 1,838 (n=3) | 1/3; 403 (n=1) |
| Oracle dynamics | 3/3; 408.7 (n=3) | 2/3; 345 (n=2) | 3/3; 549.3 (n=3) | 3/3; 379 (n=3) |
| Oracle scene | 3/3; 513.7 (n=3) | 3/3; 319 (n=3) | 3/3; 945.7 (n=3) | 1/3; 335 (n=1) |
| Zero-shot model | 3/3; 451 (n=3) | 3/3; 425.7 (n=3) | 3/3; 724 (n=3) | 2/3; 388.5 (n=2) |
| No harness fitting | 3/3; 344 (n=3) | 2/3; 455.5 (n=2) | 3/3; 791.3 (n=3) | 2/3; 609.5 (n=2) |
| No explicit uncertainty (protocol caveats) | 3/3; 312.3 (n=3) | 2/3; 328 (n=2) | 3/3; 735 (n=3) | 3/3; 260.7 (n=3) |

The table above contains methods with all four non-Bridge domains complete.
Fan standalone is now complete at 3/3 whole-run successes, 724 mean steps (n=3), and zero resets; per-seed steps are 735, 890, and 547.
Fan seeds 0 and 2 and Boil seeds 1 and 2 saved no world_model.py, and their play transcripts contain no sim.run/refine/score call mentions, despite the prompt requesting executable modeling.
Real actions were not gated on writing or using a model, so these are assigned-arm outcomes rather than evidence that those seeds used the standalone prediction interface.
These four seeds all succeeded, so this qualification affects both the Fan and Boil standalone success rates.
The completed-run audit covers all twelve finalized standalone seeds.
File presence and interface-call mentions alone do not establish predictive accuracy or whether decisions used predictions.
A zero sim_rollouts counter also does not exclude scoring calls or direct analytical code.
See the [standalone model-use audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/standalone-model-use-audit-20260913.json).
Domino standalone is complete with 0/3 whole-run successes: each seed won training and gave up on test.
Seeds 0, 1, and 2 used 702, 484, and 459 total steps respectively, all with zero resets; mean successful steps is unavailable (n=0).
The [final Domino standalone audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/domino-standalone-final-20260913.json) separates the verified failures from the agents' unverified mechanical explanations.
Balloons standalone is complete at 1/3 whole-run successes: seed 0 won 3/3 levels in 403 steps with one reset, seed 1 won 2/3 in 558 steps with one reset, and seed 2 won 2/3 in 788 steps with zero resets.
Both failed seeds explicitly gave up on the test level after winning training; neither is an infrastructure failure.
Mean successful steps is 403 (n=1), and mean resets over all three finalized seeds is 0.667.
The [final Balloons standalone audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/balloons-standalone-final-20260913.json) verifies each outcome and treats the agents' physical explanations as unconfirmed.
Standalone results use the original stricter prompt that prohibited physics-engine imports; the engine-permitted replacement has not been launched.
The full table reports per-seed outcomes and resets; infrastructure interruptions do not enter any agent average.

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
The primary distinct-seed comparison reuses the original successful pilot seed 0 for both arms and adds unique seeds 1 and 2.
MB and MF each solve 1/3 whole runs, so this variant shows no observed solve-rate advantage for MB.
Mean successful steps are 2,939 for MB and 2,634 for MF, each based on only one qualifying seed; mean resets across all three finalized seeds are zero and one respectively.
The unnecessary additional MF seed-0 failure (3,131 steps, zero resets) is disclosed separately and does not replace its successful pilot or count as another independent seed.
See the [primary per-seed table](bridge-span-mb-mf-results.md) and [comparison conclusion](bridge-span-comparison-conclusion.md).
Bridge oracle-scene seed 2 has finished successfully; all 17 other Bridge baseline/ablation seeds are now held at the user's request until the domain variant is decided.

The [fixed Balloons MF cohort](/home/ycliang/predicators/docs/comparisons/bridge-balloons-integrity-results.md) requires sustained hovering and is also separate.
Its three successes cannot be substituted for an MF control under the original instantaneous goal used in the table above.
No MB run under that changed goal was requested.

## Remaining evidence

All original non-Bridge comparison outcomes are complete.
No further non-Bridge experiment is needed to finish this frozen r1 cohort.
Keep the 17 unfinished Bridge comparison seeds held until the user decides which domain variant to use.
Their previous automatic dependency-release plan is superseded; three interrupted runs retain saved checkpoints for later replay-verified resumption.
The no-explicit-uncertainty arm is complete in the four non-Bridge domains, but its interpretation remains unresolved because three seeds used custom uncertainty checks, including both successful Domino seeds.
Keep the no-harness-fitting interpretation and agent-written-fitting disclosure in the final report.
Only then can the full six-method sweep be reported as complete.
The observed uncertainty-ablation results do not isolate whether explicit uncertainty improves success or sample efficiency.

Scorecard and terminal-log verification for this snapshot is saved in the [outcome audit](/home/ycliang/predicators/logs/continual_comparisons_20260912/outcome-audit-20260913T081745Z.json).


## Uncertainty protocol audit

A later audit confirmed that original Balloons seed1 in the no-explicit-uncertainty arm executed an agent-written parameter sweep.
It evaluated release choices over 4001 red-lift values at each of three possible mass ratios and printed intervals that would reach the goal.
A subsequent audit confirmed another sweep in still-running Domino seed 1.
Before observing a real push, it evaluated future cascades at five friction values and compared one-blue and two-blue layout thresholds; its journal explicitly described having no real dynamics data and making a prior-based decision.
It later varied friction jointly with push controls to compare future outcomes.
These are sensitivity checks for future decisions, rather than parameter fitting against recorded transitions.
The supplied uncertainty tools were disabled, but these runs do not establish strictly point-estimate reasoning.
Domino seed 1 later finished at 2/2 levels, 272 steps and zero resets; the parameter-sweep caveat applies to this successful result.
Domino seed 2 subsequently finished successfully at 2/2 levels, 384 steps and zero resets, but its executed code also tested the same push across 14 nominal/perturbed layout states and later scored placement candidates across 11 pose perturbations.
Those pose perturbations and all-variants success checks form explicit state-uncertainty robustness validation, so this seed also does not establish strictly point-estimate performance.
Its separate friction search against recorded transitions is point fitting and is not the reason for this finding.
The user has been asked whether to retain a narrower no-harness-uncertainty-tools interpretation or pursue a stricter implementation and rerun.
Point fitting, observation averaging, and controller-parameter searches at fixed dynamics are not themselves evidence of a violation.
See the [executed-code evidence](/home/ycliang/predicators/logs/continual_comparisons_20260912/no-uncertainty-protocol-audit.json).
