# Noisy hatch v4 MB/MF pilot

The user authorized two seeds per arm after the offline original-balloons modeling diagnostic exceeded its job limit.
The native simulator subclass can represent the hatch mechanics: the existing replay tests cover successful passage, contact failure and payload rotation with position error below 5 mm and pitch error below 0.05 radians.
Those tests establish representational capacity, while this pilot tests whether MB learns useful dynamics from its own experience.
The separate v4 oracle validation finished both seeds successfully; it is reported in [the oracle results](hatch-oracle-replay.md).

## Fixed experiment setup

| Setting | Value |
|---|---|
| Task | Balloons hatch, generation version 4 |
| Paired seeds | 4 and 5 |
| Levels per seed | 2 training, then 1 test |
| Position observation noise | 0.01 m |
| Orientation observation noise | 0.02 radians |
| Scalar observation noise | 0 |
| Real-step pool | 15,000 per seed, from 3 levels at 5,000 each |
| MB | Native simulator subclass, uncertainty features and generic model repair |
| MF | Existing model-free code agent |
| Observation access | Partially observable for both arms |
| Partition | `mit_preemptable` |
| Resources | 8 CPUs and 16 GB per seed |
| Slurm allocation | 12 hours, with preemption and wall-time requeue enabled |
| Continual active-time cap | 48 hours |
| Account pool | `a,c`, with the existing account picker |

The account picker reported account `b` as limited, so it is outside this launch's account pool.
Inference usage could not be queried for the selected accounts; the existing launcher therefore uses its normal fallback assignment and rate-limit handling.
Both arms start fresh, with no saved offline model or oracle dynamics supplied to the agents.
The new config was checked to generate exactly four runs, with identical task/noise settings and the intended MB-only uncertainty flags.
No noiseless controls or other-domain experiments are part of this launch.

## Source and launch

Runtime checkout: `/home/ycliang/predicators-balloons-hatch-fix-r1`.
Frozen commit: `86391e7ac12c9ae7bef2e5bd1678790f9b2f180c`.
Config: `scripts/configs/predicatorv3/protocol_continual_balloons_hatch_v4.yaml` in that checkout.
This commit adds the pilot config to the previously checked v4 Release correction, `a35937c33`.
The wrapper exports the checkout as `PYTHONPATH`, and started runs report the expected commit.

| Arm | Slurm array | Seeds | Experiment name | Queue nice value |
|---|---:|---|---|---:|
| MB | 22403398 | 4, 5 | `balloons-agent_continual_hatch_v4_mb` | 0 |
| MF | 22403399 | 4, 5 | `balloons-agent_continual_model_free_hatch_v4` | 1000 |

MB was submitted first and has higher queue priority.
All four tasks initially started; MB seed 5 was preempted during startup and automatically requeued by Slurm.
Preemption is an infrastructure event, not an agent result.

The [launch manifest](../../logs/balloons_hatch_v4_20260909/launch-manifest.json) records the exact flags, source commit, accounts and job IDs.
Slurm output is under `logs/balloons_hatch_v4_20260909/`.
Agent logs and scorecards use the main `logs/<approach>/<experiment>/seed<N>/run_<stamp>/` layout for the continual viewer.
Scorecards appear after task generation, which took tens of minutes in the oracle checks.

## Results

There are no completed agent results at launch.
Keep this pilot separate from original balloons, corrected balloons v2 and the hatch v3/v4 oracle checks.
Report each seed's levels won, primitive steps and resets.
Compute mean steps only over whole-run successful seeds and include their qualifying count.
Exclude infrastructure failures from solved/failed agent-seed totals and do not infer an MB advantage from unfinished comparisons.

The existing result watcher now covers all four agent jobs and preserves its notification history.
It reports a completed scorecard as soon as one appears, including when the job is still rendering a video, and also reports terminal job states without a final scorecard.
