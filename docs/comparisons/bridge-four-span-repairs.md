# Four-span Bridge infrastructure repairs

Separate Opus MB and MF cohort on the four-span transfer (three-span training row, four-span test row), configuration `scripts/configs/predicatorv3/continual_bridge_span_transfer_r2.yaml` (env `bridge_span_transfer` in `envs/continual.yaml`; the original per-arm file `protocol_continual_bridge_span_transfer_r2.yaml` was folded into the menus on Sept 17).
It keeps the span_transfer_r1 noise (5 mm position, 0.02 rad orientation), partial observability, uncertainty features and the 60-substep settle budget.
It is not comparable with the r1 cohort: the carry, the certificate and the completion protocol changed.

## What failed in the r1 cohort

Opus MB won 1 of 3 seeds and MF 1 of 3.
Seed 1 grasped the welded row near an end; the row tilted 15.9 degrees at the grasp with the far block 55 mm low, clipped a standing leg during the transit and was flung off the table.
Seed 2 placed the row but the settle certificate rejected it; that rejection was never reproduced because the certificate recorded nothing and the run kept the pre-settle scene.
Both winning seeds grasped the block nearest the row's centre and lift-tested the weld before seating.
The symbolic oracle had only a three-block seat operator, so no repair could be validated on a four-span level.

## Repairs

`pybullet_grasp_max_force` sets the PyBullet maxForce of the gripper-to-object constraint (None keeps the default of 500).
The welds are enforced at 10000 and the pinned partners follow the held root rigidly, so the default-strength grasp was the one compliant link of the carry.
The cohort sets it to 10000; a four-span row carried by its end block now hangs level (the regression test bounds the tilt to 1 degree and the far block's drop to 5 mm).

`bridge_lift_before_transit` prepends a straight vertical lift to Place before the transit at transport height, ported from commit 76f2257e8.
A row that starts its transit from the pick height sweeps across the standing legs.

The certificate re-anchors resting welds once per action's worth of substeps while it settles, the same anti-creep step ordinary play applies, so it never punishes the idle skate of a welded row; an unwelded span still drops out.
It logs a before/after snapshot (poses, velocities, joints, weld frames, contacts, atoms) and names the missing and lost relations on rejection.
The episode runner records and reports the settled scene, so a rejection can be diagnosed from the run's own record.

`bridge_goal_robot_clearance` makes the episode certify only once every robot link is that far from every block, so the settle never runs against a gripper still on the row; the task description states the condition.
The cohort sets 0.01 m.

The oracle gains `RowComplete(first, last)`, a derived predicate that holds for the two ends of a cured Attached path through exactly the task's span count of lying blocks.
`PickRow3`, `PickRow4`, `SeatSpan3` and `SeatSpan4` require it of their outer spans, so a three-span chain can never be seated as the bridge of a four-span task.
The grasped span is the middle of an odd row and the one just right of centre of an even row.

## Validation

All runs on compute nodes (mit_quicktest), Sept 16, 2026, from the pilot worktree.
The new regression tests pass: the lift-first phase, the recorded post-certificate scene, the clearance gate, RowComplete on three- and four-span chains, the abstraction of a four-span task, and the rigid carry (a four-span row held by its end block stays within 1 degree of level with the far block within 5 mm of the held block's height).
The remaining Bridge environment tests pass unchanged (16 tests).
The oracle end-to-end test passes on three spans with the original flags and on four spans with the cohort flags, where the plan uses PickRow4 and SeatSpan4, the goal is reached after 1526 steps and the settle certificate accepts the bridge.
Four-span oracle probes with the cohort flags on seeds 0, 1 and 2 build and certify the bridge.
Seed 1 first looped through 300 replans: the oracle had parked the glue bottle at a cell the parking sampler judged clear while the legs were still staged, then stood a leg at the adjacent site, and the palm of every later bottle grasp clipped that leg by 8 mm.
The parking sampler now applies the initial staging's rule (no mid- or back-row cell within 7 cm of a site's column) and keeps a palm-wide berth from standing blocks; this touches only the oracle's escape-hatch samplers, never the agent.
Two lessons from the validation: a derived predicate used in planning must be monotone in the atoms, because delete-relaxed reachability evaluates it on a superset where every pair is Attached (a first RowComplete that forbade extra edges made every goal unreachable); and the four-span oracle needs the cohort's weighted skeleton search and release preload (without them the same task failed a Place descent and gave up after eight minutes).
The first launch of the cohort exposed one more bug, in the certificate itself: its diagnostic snapshot read every block of the body pool, and in a transfer run the pool holds the four-span test row while the three-span training task's state omits the fourth span, so the step that completed the training bridge raised KeyError inside the harness on every later environment call.
The Opus MB agent had built the three-span bridge with the robot withdrawn 0.21 m when this hit; both runs were cancelled and relaunched on the fix.
The certificate, its snapshot and the clearance gate now read only the current task's blocks, a regression test covers the three-span task from the four-span pool, and the oracle end-to-end test gained that case (train row from the transfer pool) since neither of the earlier cases had exercised it.
None of this is an agent result; it certifies the runtime the Opus cohort runs on.

## Launch record

Seed 0 of the cohort ran on Sept 16, 2026 from this worktree at eb0095c7d, accounts b and c, jobs 22857352 (MB) and 22857353 (MF), after a first launch (22851897/98) was cancelled on the pooled-span certificate bug and its run directories set aside as `run_*_cancelled_keyerror`.

| Arm | Training (3 spans) | Test (4 spans) | Resets | Failed skill calls |
|---|---|---|---|---|
| Opus MB, gated on a fitted model | won, 1362 steps | won, 1940 steps | 0 | 0 |
| Opus MF | won, 1960 steps | won, 2332 steps | 0 | 19 |

Both test bridges passed the settle certificate.
The MB agent wrote a 230-line executable glue, cure and weld model using the simulator's Attach commands and rehearsed the whole four-span build in it before its first test-level action; the r1 MB models were no-ops.
On the r1 runtime Opus MB and MF each won 1 of 3 seeds; this is one seed on a changed runtime and completion protocol, so it is not a like-for-like comparison, and seeds 1 and 2 remain to be run.
