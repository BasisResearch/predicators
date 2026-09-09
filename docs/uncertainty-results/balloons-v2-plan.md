# Corrected balloons task validation

This is generation version 2, separate from the original noisy balloons benchmark.
The environment's lift, attachment, burst dynamics and evaluator success condition are unchanged.
The task-selection and oracle helpers now validate executable release sequences.

## What the investigation changed

The old helper stopped at the first low-speed point after motion and could label an oscillation turning point as a jam.
On the saved oak-box task, red plus green passed that point and subsequently won; gold alone also won.
The corrected helper checks for success and burst at every simulation step.
It retains the evaluator's existing one-frame speed threshold for a win; it does not add a dwell requirement or require asymptotic equilibrium inside the band.
A failure requires 20 consecutive low-speed frames with stable box position, including an angular-speed check.
A rollout that runs out of time remains unresolved.

Wall contact alone does not establish a jam.
A jam label requires sustained wall support during off-target rest and a successful replay of the same release sequence with box-wall collisions disabled.
That diagnostic restores the collisions afterward and does not change task physics.

The audit also found that release order changes outcomes.
For a pine box and band approximately [0.8044, 0.8544], red then gold bursts, while gold then red wins.
Blue plus green wins in either immediate release order.
These are distinct executable plans even when their final balloon subsets have similar analytic equilibria.
A subset cannot be called a losing decoy without specifying its release sequence.

The initial correction that required a unique winning subset and decoys losing in every immediate release order exhausted 80 sampled candidates.
The implementation therefore keeps physics and scoring, selects a reference subset that wins in every tested immediate release order, and requires a witnessed losing alternative sequence for a test task.
It permits other winning subsets and reports their witnessed count; it makes no uniqueness claim.
The old ambiguous oak task is rejected as a test challenge because its in-band candidates supply no verified losing sequence.
The optional strict jam challenge remains available through `balloons_require_jam_decoy=True` and fails explicitly if the requested property cannot be generated.
Default generation permits verified burst decoys.

## Scope of the checks

Candidate subsets are proposed by their analytic in-band equilibria.
Every permutation of each candidate is executed with the real Release skill, with a final hold and evaluator checks throughout.
This covers immediate release orders, not every possible release timing, intermediate wait, or subset outside the candidate pool.
The reference is deterministic: fewest releases, then lexicographic order, among candidates whose tested orders all win.
The oracle uses this reference and is checked through executable skills.
Offline task metrics include `task_generation_version=2` and `witnessed_winning_candidate_subsets`.
They are evaluator metadata, not agent guidance.

The task generator never selects levels by MB or MF outcomes.
A verified transient failure does not establish that MB will outperform MF.
New results must be reported separately from the frozen original-task results.

## Evidence and planned comparison

Job 22371942 reproduced the original recorded-action counterfactuals before edits.
Job 22372163 audited 120 longer simultaneous-release traces; job 22372720 checked the release-order counterexample through skills.
Job 22373207 generated both training levels and the test level for seeds 0 and 1, and all six passed executable oracle checks.
Job 22373439 passed 18 targeted environment, simulator and regression tests, full mypy on 877 source files, and changed-file formatting and pylint checks.
These are targeted checks, not full repository pytest or merge validation.

The separate noisy comparison uses fresh seeds 2 and 3, position sigma 0.01 m, orientation sigma 0.02 rad and no scalar-reading noise.
Its four arms are MF, MB without the six uncertainty features, MB with them, and MB with them plus advisory repair.
All arms use the same corrected generator and frozen runtime.
The last two MB arms differ only in the repair flag.
No new noiseless controls are needed.
Fresh noisy protocol oracle checks must pass before the comparison is submitted.
Seeds 0 and 1 were used to diagnose and develop the changes, so seeds 2 and 3 provide a small additional check beyond those examples.
The original-task repair comparison retains seeds 0 and 1 for direct before/after evaluation.
Two new seeds are still a small sample, not a precise generalization estimate.

Fresh noisy protocol oracle array 22373440 won all three levels for both seeds: seed 2 used 250 real steps and seed 3 used 211, with zero resets in both.
This exercises the actual continual runner and configured skills, in addition to the generator's own probes.

Combined MB-repair and generator checks passed in job 22373693, including full mypy on 881 source files.
The source changes are committed on `bridge-learning` as `61e44d4ce` (MB repair) and `06a5c131e` (task validation).
The original-task experiment checkout remains unchanged at `df53ca75f`.
The corrected benchmark configuration is `scripts/configs/predicatorv3/protocol_continual_balloons_validated_v2.yaml`.

## Submitted corrected-task runs

The corrected runtime is frozen at `24e23125c` in `/home/ycliang/predicators-balloons-v2-r1`.
Preflight job 22374025 passed: it verified the completed diagnostics and noisy oracle scorecards, unchanged physics and evaluator methods, and frozen source hashes.

| Arm | Seeds | Slurm array |
|---|---|---|
| MB with uncertainty features and repair | 2, 3 | 22374077 |
| MB with uncertainty features | 2, 3 | 22374078 |
| MB without uncertainty features | 2, 3 | 22374080 |
| MF | 2, 3 | 22374081 |

All arrays use `mit_preemptable`, the existing account-selection launcher, and requeue/resume support.
They follow the original-task sequence's final array 22372901 and run in two-seed batches, with every array requiring the preflight gate.
The resolved settings and source hashes are in `logs/mb_repair_20260909/balloons-v2/launch-manifest.json` and `frozen-source.json`.
No corrected-task agent outcomes were available at submission.

Collector 22374089 will update [the corrected-task table](balloons-v2-table.md), snapshot and TSV after the four arrays terminate.
Steps average only whole-run successful seeds; solve rate and resets use all seeds, and incomplete arms withhold averages.
The original-task results retain their [separate table](model-repair-table.md).
