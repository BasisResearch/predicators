# Hatch oracle: release validation mismatch

The first noisy hatch oracle stopped after repeated Wait plans because its box was trapped below the hatch.
The task generator had certified a different execution of the same release order.
This is a task-validation and controller-integration issue; it is separate from the original balloons agent's missing learned model.

## Finished noisy continual oracles

| Seed | Levels won | Whole-run steps | Resets | End reason |
|---|---:|---:|---:|---|
| 4 | 0/3 | 279 | 0 | Controller returned without winning the first training level |
| 5 | 3/3 | 274 | 0 | All levels won |

Seed 4 did not attempt the remaining two levels.
The mean steps over whole-run successful oracle seeds is 274, with n=1.
These are oracle checks, not MB or MF results.
Both job states were verified completed with exit code zero.
Scorecards are under `logs/balloons_followup_20260909/oracle-runs/oracle_process_planning/balloons-hatch-v3-deterministic-oracle/`.

## Saved-action reproduction

Compute job `22393666` reconstructed seed 4's first training task and replayed all 279 recorded primitive actions.
It reproduced no win, with the box at 0.544009 m and supported against the hatch.
An otherwise identical replay with box-panel collisions disabled first won at step 81 and finished at 0.720000 m.
The original execution was therefore blocked by contact; increasing the Wait retry limit would not address the cause demonstrated here.
This replay is a diagnostic of an existing oracle run, not an additional solve-rate seed.

## Controlled Release comparison

The generator's `probe_release_option()` explicitly disables motion planning.
The normal oracle uses the public Release option with motion planning enabled.
Job `22393934` compared these two configurations on the reconstructed initial task, using the same clip parameters and release orders.
Its initial robot features were restored to the saved task's realized values rather than the nominal home pose.

| Release order | Motion planning | Outcome | Primitive actions | Final box height |
|---|---|---|---:|---:|
| Gold then blue | Off | Won | 66 | 0.739081 m |
| Blue then gold | Off | Won | 77 | 0.710024 m |
| Gold then blue | On | Unresolved below hatch | 559 | 0.543958 m |
| Blue then gold | On | Won | 88 | 0.710023 m |

The gold-then-blue normal-option replay did not satisfy the diagnostic's sustained-contact classifier, so its outcome remains unresolved rather than being relabeled a certified jam.
The separate exact-action replay above supplies the causal contact evidence for the recorded oracle failure.
Together, these checks show that the generator's guarantee over probe release orders does not carry over to the normal Release implementation.
The supplied reference subset alone is insufficient to certify the oracle's executed plan.

These diagnostics established the need to align task certification with actual Release execution and ensure that the oracle executes a certified order.
Full noisy continual checks are required before launching hatch agent comparisons.
The diagnostics themselves contain no new MB/MF solve-rate run.

Reports and scripts: `logs/balloons_followup_20260909/wait-diagnostic/{report,transit-report}.json`, `diagnose_wait.py`, and `diagnose_transit.py`.
Earlier failed diagnostic jobs were setup errors and are excluded from results.
All simulations ran on `mit_preemptable`.

## Visual explanation

[The overview slides](../envs/balloons/overview_slides.html) now contain the chute/hatch comparison, collision rules, fixed-sequence replay videos, and measured height curves.
Their videos reproduce the earlier mechanical audit on seed 5 with motion planning disabled, as explicitly described in the [asset notes](../envs/balloons/visuals/README.md).
They demonstrate the physical mechanism and do not certify the normal oracle controller.
The deck and assets are committed as `c621f3be3`.
All 17 slides passed browser playback, navigation and overflow checks; the PDF passed page-boundary and comparison-image checks.

## Public Release correction

Commit `a35937c33` on `balloons-hatch-release-parity` changes certification to construct the same Release controller as the public option set.
The isolated checkout is `/home/ycliang/predicators-balloons-hatch-fix-r1`, based on integrated source `71f0dbe20`.
Candidate cache keys now distinguish motion-planning mode, seed and release parameters, and probe options no longer survive configuration changes in a global cache.
The generator retains its requirement that all tested immediate orders of its reference subset win.
The physics and evaluator are unchanged.

The new regression reproduces the failed task through public `env.step()` calls, including the saved initial robot pose.
Job `22394867` failed before the fix because public Release did not win while certification reported a win in 66 actions.
Job `22394904` passed all 17 focused hatch, probe-validation and balloons environment tests after the fix, including the regression with motion planning both enabled and disabled.
Job `22394942` passed changed-file lint, formatting with the CI isort version, and mypy on all 886 source files.

Task-generation metadata advances to hatch version 4 and chute version 3 because the correction can reject previously accepted tasks.
Earlier scorecards and the failed task remain unchanged.
Regenerated tasks can differ, so later successful oracle checks must not be described as solving the identical old seed 4 task.

Noisy continual validation array `22395404`, seeds 4 and 5, was submitted on `mit_preemptable` from the committed fix.
It uses the same noise settings and oracle configuration as the earlier validation, with a separate experiment name `balloons-hatch-v4-public-release-oracle`.
Scorecards will be written under the main `logs/oracle_process_planning/` directory so the continual viewer can discover them.
There is no new oracle verdict at submission and no MB/MF comparison was launched.
The existing result watcher now includes both tasks, preserving its prior notification state.

### Hatch v4 results as of 17:47 UTC

| Seed | Status | Levels won | Whole-run steps | Resets |
|---|---|---:|---:|---:|
| 4 | Completed | 3/3 | 281 | 0 |
| 5 | Running, no scorecard yet | Pending | Pending | Pending |

Seed 4's training levels took 92 and 72 steps, and its test level took 117 steps, all with zero resets.
Its scorecard and completed Slurm job with exit code zero agree.
The mean over whole-run successful v4 oracle seeds so far is 281 steps with n=1; seed 5 is unfinished and excluded.
This is a v4 oracle result on regenerated tasks, not an MB result or a replay of the identical v3 seed 4 task.

Neither workload's delay was mainly queueing.
The offline synthesis job waited 42 seconds to start, and both v4 oracle tasks waited 40 seconds.
The offline diagnostic spent roughly 19 minutes before entering the long `sim.fit()` call and about 101 minutes inside that call before its two-hour job limit.
Its log contains 33 LM fit summaries, additional bracket-search results and repeated refits that pin parameters back to their declared baselines.

The hatch generator simulates candidate balloon subsets in every release order, now using the public motion-planned Release controller.
Unresolved probes can consume 500 wait steps, and qualifying contact failures require another rollout with payload-panel collisions disabled.
It retries candidate tasks until the required winning reference and losing-sequence conditions are established.
Seed 4's total job duration was 34 minutes 21 seconds, but its scorecard records only 59 seconds of active continual play.
Roughly 33 minutes were spent before play, primarily on setup and task generation/validation.
Seed 5 was still consuming CPU with no scorecard at the last check.

## Offline synthesis logs

The separate original-balloons modeling diagnostic remains on its frozen source at `a8ab7d48a`.
Its conversation log is [the training-only synthesis transcript](../../logs/balloons_followup_20260909/offline-learning-v2/agent/001_learn_20260909_114132.md), and detailed fit progress is in [the compute-job log](../../logs/balloons_followup_20260909/offline-learning-22389016.out).
Job `22389016` reached its two-hour Slurm time limit at 17:41 UTC on September 9 while turn 70 was still waiting for `sim.fit()`.
The scheduler records `TIMEOUT`, and the compute log confirms termination due to the time limit.
The agent had declared 11 model parameters; the fitter repeatedly replays trajectories and refits the remaining parameters while testing whether each changed parameter can return to its baseline.
This explains why the fit takes much longer than one replay of the 422 recorded training actions.
The provisional `agent/sandbox/simulator.py` was preserved, but neither `replay.txt` nor `outcome.json` was produced.
There is no completed fit return or final model replay score from this attempt.
This is an incomplete offline diagnostic caused by the job time limit, not a solved or failed agent seed.
The offline diagnostic has no continual scorecard, so it is not listed as a continual run in the viewer.

At the same verification, hatch v4 oracle seeds 4 and 5 were still running after roughly 30 minutes, with no scorecards yet.
Their task generation and validation had not yielded new oracle outcomes.
No experiment was relaunched or changed in response to the timeout notification.
