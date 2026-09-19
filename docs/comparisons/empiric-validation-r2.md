# EMPIRIC r2: prospective robustness checks

The requested cohort is two additional seeds per domain, seeds 3 and 4, across Boil, Domino, Balloons, Bridge, and Fan.
The launcher is [continual_empiric_benchmark_r2.yaml](../../scripts/configs/predicatorv3/continual_empiric_benchmark_r2.yaml).
Its experiment key is `<domain>-mb_opus_benchmark_r2`, under `agent_continual`.
The main figures retain historical EMPIRIC and show this cohort separately as **EMPIRIC r2**.
All finished outcomes count, including failures; an unfinished run is not a failed run.

## Validation policy

Legacy blocking skill preflight remains off.
The new audit is nonblocking: it does not reject actions or send predictions to the agent.
It rehearses skill requests and individual raw actions in an isolated candidate world, then records the real execution outcome under the same identifier.
It uses one point-estimate rollout, with no automatic belief draws, parameter sweeps, or planner-seed repetitions.
Terminal predictions receive the stronger task certificate using the observed episode prefix and predicted suffix, with the same candidate physics.
Intermediate requests receive controller and annotated-outcome checks; these are not task-success certificates.
Restoration mismatches, missing models, missing per-step recordings, and exceptions are explicitly distinguished from passes.
Closed-loop raw policies are recorded as unavailable, not silently certified.
The audit records model hash, deployed parameters, diagnostic duration, simulated steps, predicted poses and grasps, and actual public execution outcomes.
Logs are in each run's `agent/validation_audit.jsonl`.

The diagnostic allowance is 600 seconds per level, retained across workbench rounds and resumptions, with a 30-second watchdog per rehearsal.
It has no additional real-action allowance and does not consume the agent's explicit simulator-rollout counter.
Diagnostic time still consumes wall-clock time; skipped checks after the allowance is exhausted remain unavailable.
The watchdog is cooperative at Python boundaries, so a native engine call may delay interruption.
This cohort is not a randomized enforced-versus-disabled preflight comparison and does not by itself establish the causal benefit of gating.

## Learned-model restoration

Candidates now have an explicit `restore_model_state()` hook, invoked before skill planning, and a `restore_model_attachments()` helper for model-inferred rigid links.
Links are registered with held-assembly collision checks, persist across physics steps, and retain their local joint frames in prediction snapshots.
The new `sim.check_restore()` diagnostic checks pose, memory, attachment, and frame round trips in fresh worlds.
The candidate still owns the inference of which links exist; the harness supplies neither Oracle glue rules nor hidden execution state.
The shared simulator instructions explain the contract to newly written models.
These instructions and runtime changes distinguish r2 from the historical cohort.

## Historical Bridge diagnostics

At Oracle seed 2, test step 1267, the historical lift failed while the repaired model's three fresh rehearsals completed.
An additional offline replay with exact recorded observations also completed all three rehearsals ([output](../../logs/bridge-replay-exact-23105528.out)).
Therefore observation noise alone does not explain this mismatch; the original execution's engine/controller state is not fully reproduced.
No exact state is supplied to agents in r2.

A saved EMPIRIC Bridge seed-2 model, replaying that same observed history at its declared default parameters, reconstructed zero registered joints and also predicted the lift would complete ([output](../../logs/bridge-replay-learned-23105529.out)).
This is a cross-trajectory diagnostic of the saved model, not a replay of its original fitted deployment or evidence that its historical run failed.
Its code creates constraints during the dynamics hook, rather than restoring registered attachments before controller planning.
The new contract addresses that failure mode for candidates that implement it; it does not automatically repair every old simulator artifact or establish that the historical lift mismatch is solved.

## Launch receipt

Compute-node checks passed: 59 runtime/tool/configuration tests plus 31 prompt checks (`23105755`), a 53-test follow-up (`23105891`), five final audit/restoration tests (`23105996`), and 12 attachment regressions (`23106006`).
Targeted mypy checks passed, and the final source lint check passed after correcting report-generator line wrapping (`23106124`).
Pinned formatter checks also passed.
These are overlapping targeted regression batches, not the full repository CI suite.
The frozen runtime is `f2ed37aef`, at `/home/ycliang/predicators-empiric-r2-frozen-20260919`.
The same checkout passed all six frozen smoke tests (`23106203`) before submission.
All five arrays use indices `3-4`, `mit_preemptable`, eight CPUs and 16 GB per task, automatic checkpoint resume, and requeue.
Slurm allocations are 12 hours with the launcher's pre-timeout requeue; the run's active wall-clock allowance remains 48 hours.

| Domain | Array job | Seeds |
|---|---|---|
| Balloons | `23106296` | 3, 4 |
| Bridge | `23106297` | 3, 4 |
| Boil | `23106299` | 3, 4 |
| Fan | `23106301` | 3, 4 |
| Domino | `23106302` | 3, 4 |

Figure/report monitor `23106204` replaces `23104366` to load the new EMPIRIC r2 entry.
It checks every 60 seconds on a compute node and updates the main figure and Markdown report when finished scorecards change.
Its requested allocation is 48 hours; it does not launch additional experiments or change running code.
The historical runs and the already-running Oracle repair cohort retain their original frozen runtimes.
