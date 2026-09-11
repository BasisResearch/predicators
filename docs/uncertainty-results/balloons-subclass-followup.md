# Balloons model and hatch follow-up

The original repaired balloons seed 0 still has no executable learned lift or attachment mechanism.
Its three saved simulator versions are identical no-op artifacts.
This follow-up checks model construction and recorded-action prediction before another MB solve-rate comparison.
The original task, corrected generator and hatch prototype remain separate experiments.

## Training-only audit

The source is `logs/agent_continual/balloons-agent_continual_repair_on_r1/seed0/run_20260909_075741`.
Only the two training levels enter the model audit; the test level and the final journal, which contains test experience, are excluded.
The three training episodes contain 166, 108 and 148 actions, respectively.
Their exported noisy object features are retained, with portable robot joint state restored from the corresponding harness recordings so the simulator can replay the actions.
Object velocities and other privileged simulator state are excluded from the learning input.

At its deployed defaults, the saved no-op model has the following full-recording errors:

| Training task | Episode | Actions | Normalized replay RMS |
|---|---:|---:|---:|
| 0 | 0 | 166 | 0.262381 |
| 1 | 0 | 108 | 0.204216 |
| 1 | 1 | 148 | 0.250453 |

The aggregate RMS is 0.244410 over the same three recordings.
These are normalized predictive errors, not metres, solve rates or a held-out generalization estimate.
All recordings remain in validation even if fitting rejects them.
The public replay audit took three simulator rollouts and approximately seven seconds with portable robot state in compute job `22388751`.

The diagnostic in job `22389016` uses the normal subclass synthesis instructions and the same noisy training experience, with no real-action tools and no test input.
It runs from frozen checkout `/home/ycliang/predicators-balloons-model-audit-r1` at `a8ab7d48a`.
Its sandbox has all three training recordings, verified after correcting an audit-script data-sync error.
The aborted setup job `22388799` is excluded from agent results.
It must produce an executable artifact and a complete replay report at its deployed values.
Its result is an offline modeling diagnostic, not a new successful agent seed or a replay of the original continual conversation.
Any candidate selected on these recordings has used them for model development.

## Visible simulator base

A loaded balloons artifact could call inherited hidden helpers such as `true_box_mass` on the newly supplied `BaseSimulator`.
The artifact-loading regression reproduced that exposure in compute job `22388286`.
The balloons adapter now derives from the separate visible physics core, with empty task and predicate definitions.
Hidden lift laws, true masses and task-generation helpers are no longer inherited.
Visible mass and damping setters live in the shared base, so the real environment and the learned model use the same parameterization and state-restoration code.
The other four domains retain their previous adapter implementation in this change.

The focused subclass, memory and compatibility tests passed in job `22388415` (17 tests).
Formatting, whole-repository mypy on 885 files and 16 prompt goldens passed in job `22388512`.
An unused import found by lint was removed; final changed-file lint and CI-pinned import ordering passed in job `22388684`.
Environment, public validation and continual-loop regressions passed in job `22388671` (18 tests).
The code correction is committed as `a8ab7d48a`.
This is local validation; no PR, push or merge is included.

## Separate hatch validation

Deterministic contact ordering is being integrated only for the opt-in hatch scene.
Replay now compares a fresh real environment and the native subclass against the original continuous sequence of actual Release actions.
The native test must use the configured ground-truth constants, not its deliberately approximate parameter initializers.

The previous fixed witness changes behavior with deterministic ordering and becomes contact-blocked.
Its old success assertion is therefore invalid under the proposed repeatable physics configuration.
A bounded audit of all 24 placements of the four visible colors checked the same two equal-count subsets and both release orders in job `22388615`.
Several placements have a reference that wins in both orders and a causal jam that disappears when the other pair is released in reverse order.
None meets the earlier all-orders-losing decoy criterion.
The generator now certifies a specific losing sequence, preserving the equal-count comparison without claiming that every order loses.
The hatch prototype is committed as `3be863516`.
Its five focused tests, changed-file lint and whole-repository mypy passed in job `22388885`.
A separate integration checkout at `71f0dbe20` combines it with the unified MB contract and visible-base correction.
Its 21 functional checks, changed-file lint and whole-repository mypy on 886 files passed in job `22389101`.
Generated-task audits are job `22389017`, seeds 0, 4 and 5.
Noisy continual oracle checks are job `22389260`, seeds 4 and 5, dependent on successful task audits and integration checks.
The generator and oracle must pass before a hatch agent pilot can launch.
Mechanical validation is not evidence of an MB advantage.

All expensive checks use compute nodes on `mit_preemptable`.
Logs and diagnostic scripts are under `logs/balloons_followup_20260909/` in the main checkout.

## Result notifications

A local watcher checks the diagnostic jobs every 120 seconds and queues verified-result requests into this conversation.
Its state and events live under `logs/balloons_followup_20260909/`; it also refreshes `docs/uncertainty-results/balloons-followup-status.json`.
Offline modeling, mechanical task audits and oracle results are reported separately from MB/MF agent performance.
The watcher never launches or modifies experiments.
