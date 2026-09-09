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

The next correction must align task certification with the actual Release execution and ensure that the oracle executes a certified order.
Then repeat the full noisy continual checks before launching hatch agent comparisons.
No production fix or new MB/MF solve-rate run is included in these diagnostics.

Reports and scripts: `logs/balloons_followup_20260909/wait-diagnostic/{report,transit-report}.json`, `diagnose_wait.py`, and `diagnose_transit.py`.
Earlier failed diagnostic jobs were setup errors and are excluded from results.
All simulations ran on `mit_preemptable`.

## Visual explanation

[The overview slides](../envs/balloons/overview_slides.html) now contain the chute/hatch comparison, collision rules, fixed-sequence replay videos, and measured height curves.
Their videos reproduce the earlier mechanical audit on seed 5 with motion planning disabled, as explicitly described in the [asset notes](../envs/balloons/visuals/README.md).
They demonstrate the physical mechanism and do not certify the normal oracle controller.
The deck and assets are committed as `c621f3be3`.
All 17 slides passed browser playback, navigation and overflow checks; the PDF passed page-boundary and comparison-image checks.
