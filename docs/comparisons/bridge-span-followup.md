# Bridge span-transfer follow-up

The paired pilot did not show a larger advantage for MB.
Both seed-0 agents won the three-block training level and four-block test level without resets.

| Pilot arm | Training steps | Test steps | Total steps | Resets |
|---|---:|---:|---:|---:|
| MB with uncertainty | 1,631 | 1,308 | 2,939 | 0 |
| MF direct coding agent | 1,372 | 1,262 | 2,634 | 0 |

This is an exploratory paired pilot, not a statistically reliable estimate of the difference between agents.
The user's conditional instruction therefore selects MF and the six Section 4 comparison arms for the follow-up, without new MB runs.

## Cohort

The follow-up contains twenty-one fresh agent runs: seeds 0, 1, and 2 for MF, standalone executable model, oracle dynamics, oracle scene reconstruction, frozen zero-shot synthesis, no numerical fitting, and no explicit decision uncertainty.
The pilot runs remain separate, including the pilot's seed 0.
Each new run encounters one three-block training span and one four-block test span.
The observation noise remains 5 mm in position and 0.02 rad in orientation, with a 10,000-step interaction budget per level.
The production agent and shared controller interface match the integrity-fixed pilot; the parallel uncertainty-simplification changes are not included.

Configurations:

- `scripts/configs/predicatorv3/protocol_continual_bridge_span_mf_sweep_r1.yaml`
- `scripts/configs/predicatorv3/protocol_continual_bridge_span_comparisons_r1.yaml`

The eighteen previously held Bridge comparison seeds are superseded by the eighteen comparison seeds in this cohort once submission succeeds.
The seventy-two non-Bridge comparison seeds continue on their original frozen runtime.
The new comparison arrays extend the existing experiment order, preserving the concurrency limit of six comparison seeds.
Each queued seed follows the corresponding seed in the preceding array, making six independent comparison chains.
The three new MF runs can start independently.
All runs use account b and `mit_preemptable` compute nodes.

## Validation

All three seeded test layouts passed public-controller pickability checks for every staged block and bottle.
The native mechanics and oracle tests passed eleven checks in job `22648954`.
The four-span oracle reproduces all three glue-to-weld transitions and inferred reciprocal attachments, with bounded pose differences after public-state restoration.
Type checking and all three lint checks passed in job `22648955`.
The full twenty-five-test comparison-interface suite passed in job `22648953`, including a separate-process standalone-model resume.
It completed in 34 minutes with peak allocation memory of approximately 2.4 GB, including the resume subprocess.

The earlier combined suite exceeded its allocation and then exhausted 8 GB of memory because its cases left native physics clients connected.
A test-only fixture now tracks and disconnects clients created by each case, including evicted skill simulators, and removes their cached handles.
It does not change agent execution or experiment outcomes.
The launch controller verified all required test and static-check summaries before submission.

## Submitted cohort

Controller `22650115` submitted and released all twenty-one seeds from immutable runtime `73b5e517bf18fea0ec79eec5c6d1a18964f8a4c3`.
The frozen checkout is `/home/ycliang/predicators-bridge-transfer-frozen-20260912`.
The three MF seeds were verified running; the comparison seeds were pending on their declared dependencies at submission.
Each array below contains seeds 0, 1, and 2.
The predecessor column names the preceding array; seed `s` now depends on its predecessor's seed `s`.

| Arm | Array | Predecessor |
|---|---|---|
| MF direct coding agent | 22650415 | None |
| Oracle dynamics | 22650416 | 22642729 |
| Oracle scene reconstruction | 22650417 | 22642732 |
| Zero-shot synthesis | 22650418 | 22650416 |
| No numerical fitting | 22650419 | 22650417 |
| No explicit uncertainty | 22650420 | 22650418 |
| Standalone program | 22650421 | 22650419 |

The eighteen old Bridge jobs in arrays `22642706`, `22642711`, `22642716`, `22642721`, `22642726`, and `22642731` were cancelled only after the replacement arrays were successfully submitted.
They have no agent outcomes and are marked superseded in the historical table.
The non-Bridge comparison arrays were not changed.

After submission, the dependencies of all sixty-nine pending comparison seeds were updated to operate independently per seed.
This lets a completed predecessor release the next seed while the other two predecessor seeds continue.
The update preserves method order within each chain and permits at most six comparison jobs at once.
The separate MF arrays retain their original scheduling.
The original submission dependencies and every verified update are recorded in `/home/ycliang/predicators/logs/continual_comparisons_20260912/seed-chain-update-20260912.json`.

## Reporting

Operational scripts, validation logs, immutable-source hashes, and submission journals are in `/home/ycliang/predicators/logs/bridge_span_followup_20260912/`.
The reporter writes [Bridge follow-up results](/home/ycliang/predicators/docs/comparisons/bridge-span-followup-results.md) in the primary checkout and is called when each experiment exits.
Report wins, charged steps, and resets for each seed.
Only completed agent outcomes enter solve-rate and reset averages; only whole-run successes enter mean steps, with the qualifying count shown.
Infrastructure interruptions remain separate.
Historical three-block MB scores and the single transfer pilot must not be presented as a matched three-seed MB cohort for this changed task.
