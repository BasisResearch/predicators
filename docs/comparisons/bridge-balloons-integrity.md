# Bridge transfer and sustained-hover Balloons

This cohort addresses the MF integrity audit of 2026-09-12.
Historical scores and the already frozen continual comparison sweep remain separate.
Only the eighteen Bridge baseline/ablation seeds are held while the transfer pilot is evaluated.
Non-Bridge jobs were released and their dependencies bypass the held Bridge arrays.

## Shared observation boundary

Both coding agents receive skill API documentation and the runtime list of typed signatures, parameter meanings, and bounds.
The shared controller implementation and its task-specific comments are no longer copied into their reference directories.
MB retains its separately permitted visible simulator reference.
Actual controller failures return an outcome category rather than hidden attachment identities, internal collision distances, or simulator body IDs.
The full diagnostic remains in the experimenter's private log.
Agent observations serialize public object schemas and features, without simulator metadata.
Simulator restoration resolves those objects by name against the receiving world's own body roster.
Ordinary Python analysis, fitted analytical models, and journals remain permitted for MF.

## Bridge pilot

Configuration: `scripts/configs/predicatorv3/protocol_continual_bridge_span_transfer_r1.yaml`.
One MB seed and one MF seed, both seed 0, each encounter one training task with three span blocks and one test task with four.
The support separation grows from 0.25 m to 0.35 m, keeping each outer span supported by a leg and the interior dependent on the joined structure.
The goal remains independent of object ordering and uses all task blocks except the two legs.
The environment allocates enough bodies for either split and parks inactive ones out of view.
Three-block staging retains its original sampling.
Four-block staging uses finite search over a grid with 14 cm row spacing, avoiding the packing failures of the smaller task's sampler.
Both arms have 5 mm position noise, 0.02 rad orientation noise, a 10,000-step budget per level, and the same controller interface.
MB retains its existing uncertainty features.
No new uncertainty simplification changes from the parallel task are included.

The pilot is exploratory.
Report per-level and whole-run steps, resets, and success for both arms, then compare the efficiency ratio with the historical three-block cohort.
A single paired seed does not establish a statistically reliable performance gap.
The user's subsequent sweep decision depends on this pilot: if the gap widens, rerun MB, MF, and the comparison arms on the variant; otherwise rerun MF and the comparison arms on the variant.
Keep cohort labels explicit whenever old MB scores are shown alongside a changed task.

## Balloons rerun

Configuration: `scripts/configs/predicatorv3/protocol_continual_balloons_dwell_mf_r1.yaml`.
Run MF seeds 0, 1, and 2 only; a new MB Balloons run is deferred at the user's request.
The box must stay inside the band below the configured speed threshold for 25 complete environment-step intervals, requiring 26 consecutive endpoint observations.
A temporary slowdown keeps the episode open rather than winning or losing it.
A burst remains immediately terminal and unsuccessful.
The evaluator uses the episode history, so repeated observation calls do not accumulate dwell and replay reconstructs the same decision.
Each new episode begins with an empty dwell history.
The instantaneous InBand predicate is a local subgoal; episode success additionally requires the explicitly described dwell.
Continual termination, episode rewards, and simulation trajectory verdicts use the same temporal evaluator.

The original reference planner can select a transient turning-point solution that fails the sustained criterion.
That does not establish task infeasibility: other release sequences can succeed on the same task.
Audits `22648749_0`, `22648749_1`, and `22648749_2` found sustained-hover witnesses for every one of the nine original tasks, with task descriptions matching the archived MF seeds.
The new MF cohort therefore keeps `balloons_task_generation: original` and `balloons_require_jam_decoy: true`, exactly as in the original MF sampling configuration.
The goal and shared observation interface are corrected; the task sampling distribution is preserved.
Position noise remains 1 cm, orientation noise 0.02 rad, and the real interaction budget remains 5,000 steps per level.
Each seed has two training levels and one test level.
The separate validated-generator search was cancelled after these witnesses established that it was unnecessary.

## Validation and execution records

Compute-node logs, validation outcomes, launch manifests, and scheduler IDs are stored in `/home/ycliang/predicators/logs/bridge_balloons_integrity_20260912/`.
The initial regression reproduces the public-record metadata leak and the one-frame continual Balloons win.
A controller-failure regression passes a hidden-weld diagnostic through the actual invocation API and checks that its returned message contains no privileged details.
Bridge mechanics check both roster sizes, restoration of public observations, collapse without welds, stability with welds, and the public controller's ability to pick every staged object in the four-block pilot.
Native task audits are feasibility checks, not agent seeds or solve-rate observations.
Mean steps are computed only over whole-run successful agent seeds, with the qualifying count shown.
Infrastructure failures are excluded from agent solve/reset averages.

## Pilot launch and follow-up preparation

Bridge MB seed 0 is job `22646173_0` and Bridge MF seed 0 is `22646174_0`, both on `mit_preemptable` with account b.
Their frozen runtime is commit `3e95b1798c6fc7b6e24dccf0fd0b6702c8801d94` in `/home/ycliang/predicators-bridge-balloons-frozen-bridge-20260912`.
The prelaunch suite passed 82 checks, type checking passed for 17 files, and lint passed.
Both runs have current scorecards; their results are generated in `bridge-balloons-integrity-results.md` in the primary checkout.
The eighteen older Bridge comparison seeds remain held, while non-Bridge dependency chains bypass them.

`scripts/configs/predicatorv3/protocol_continual_bridge_span_comparisons_r1.yaml` prepares the six Section 4 comparison arms with seeds 0, 1, and 2 for the changed span distribution.
This follow-up configuration has not been launched; the paired pilot still determines whether to include new MB seeds.
The native four-span oracle audit verifies all three curing joints, inferred reciprocal attachment identities, model restoration, and return to the smaller training roster.
Its measured prediction error was at most 0.834 mm in position and 0.01299 rad in orientation over the 35-step audit, below the pilot's 5 mm and 0.02 rad observation scales.
Glue transitions and weld counts agree exactly; this is bounded prediction accuracy after public-state restoration, not exact reproduction of contact solver history.
The configuration and public-boundary audit passed eight checks in job `22646683`.
Type checking passed for all three changed Python files, and all three lint checks passed in job `22646775`.

The first Balloons reference-only checks rejected plans, not all possible solutions.
A subsequent validated-generator search exhausted its initial allocation and was superseded by the successful original-task alternative-sequence audits.
These are mechanical validation outcomes, not agent scores.
No Balloons MB rerun is authorized for this cohort.

The final original-task and observation-boundary suite passed nine checks in job `22648911`.
Type checking and both lint checks passed in job `22648912`.
