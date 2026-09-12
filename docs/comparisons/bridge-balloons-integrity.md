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

The original task screen accepts some trajectories that only reach the band at a turning point.
Mechanical checks of its reference plans reproduced eventual rest outside the band.
The new MF cohort therefore uses the validated non-hatch task generator with the sustained criterion and public Release controller.
These are new task draws under a changed acceptance rule, not reused outcomes or a paired replacement for historical MF.
Position noise remains 1 cm, orientation noise 0.02 rad, and the real interaction budget remains 5,000 steps per level.
Each seed has two training levels and one test level.

## Validation and execution records

Compute-node logs, validation outcomes, launch manifests, and scheduler IDs are stored in `/home/ycliang/predicators/logs/bridge_balloons_integrity_20260912/`.
The initial regression reproduces the public-record metadata leak and the one-frame continual Balloons win.
A controller-failure regression passes a hidden-weld diagnostic through the actual invocation API and checks that its returned message contains no privileged details.
Bridge mechanics check both roster sizes, restoration of public observations, collapse without welds, stability with welds, and the public controller's ability to pick every staged object in the four-block pilot.
Native task audits are feasibility checks, not agent seeds or solve-rate observations.
Mean steps are computed only over whole-run successful agent seeds, with the qualifying count shown.
Infrastructure failures are excluded from agent solve/reset averages.
