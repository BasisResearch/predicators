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
The new comparison arrays extend the two existing dependency chains, preserving their concurrency limit of six comparison seeds.
The three new MF runs can start independently.
All runs use account b and `mit_preemptable` compute nodes.

## Validation

All three seeded test layouts passed public-controller pickability checks for every staged block and bottle.
The native mechanics and oracle tests passed eleven checks in job `22648954`.
The four-span oracle reproduces all three glue-to-weld transitions and inferred reciprocal attachments, with bounded pose differences after public-state restoration.
Type checking and all three lint checks passed in job `22648955`.
The full twenty-five-test comparison-interface suite is running in job `22648953`.

The earlier combined suite exceeded its allocation and then exhausted 8 GB of memory because its cases left native physics clients connected.
A test-only fixture now tracks and disconnects clients created by each case, including evicted skill simulators, and removes their cached handles.
It does not change agent execution or experiment outcomes.
The entire suite must pass before the follow-up is submitted.

## Reporting

Operational scripts, validation logs, immutable-source hashes, and submission journals are in `/home/ycliang/predicators/logs/bridge_span_followup_20260912/`.
The reporter writes `docs/comparisons/bridge-span-followup-results.md` in the primary checkout after submission.
Report wins, charged steps, and resets for each seed.
Only completed agent outcomes enter solve-rate and reset averages; only whole-run successes enter mean steps, with the qualifying count shown.
Infrastructure interruptions remain separate.
Historical three-block MB scores and the single transfer pilot must not be presented as a matched three-seed MB cohort for this changed task.
