# Continual comparisons: Section 4 items 3-8

The user selected only the six additional comparisons; the existing full MB and MF cohorts remain separate.
The requested sweep is five noisy domains and paired seeds 0, 1, 2, using the distributions in `protocol_continual_noisy_sweep_r1.yaml`.
The combined configuration is `scripts/configs/predicatorv3/protocol_continual_comparisons_noisy_r1.yaml`.
It must not be launched until all six arm contracts and their end-to-end tests pass.

| Item | Arm | Continual adaptation | Required validation |
| --- | --- | --- | --- |
| 3 | Standalone learned simulator | Reuse the executable option transition model, with live recorded data and model edits inside the play conversation; no engine predictions | Model edits and recorded skill transitions reach planning, memory and resume; no supplied engine bypass |
| 4 | Oracle dynamics | Supply fixed correct mechanisms and parameters, with noisy observations and inferred memory | Recorded-action parity in each domain; model remains fixed across action, reset and resume |
| 5 | Oracle scene reconstruction | Supply true geometry, articulation and base physics, omit mechanisms, freeze model | Disclose calibration; missing mechanisms stay absent and true current poses remain hidden |
| 6 | Zero-shot synthesis | Synthesize before the first real interaction, then freeze dynamics and parameter values | Refuse first action without a valid model; seal before charging it; prevent later edits/refits including after resume |
| 7 | No numerical fitting | Agent revises declared values and ranges from recordings; numerical optimizer disabled | Live fit and fitted residual/sweep routes refuse; declared values deploy without optimization |
| 8 | No explicit uncertainty | Keep noise-aware fitting, smoothed point observations and inferred memory; remove distributional decision tools | No belief draws, parameter sweeps, disagreement probes or probabilistic predicate monitoring; ordinary rehearsals remain |

## Progress

An isolated checkout at `/home/ycliang/predicators-continual-comparisons` protects concurrent uncertainty simplification edits.
The six-arm, ninety-run configuration is drafted.
Items 3, 7 and 8 have initial implementations; their scripted continual integration tests and the existing MB/MF regression tests passed together (17 tests, job 22638423).
Item 3 still needs complete execution-memory and engine-isolation auditing before launch.
Item 6 seals a valid model before the first charged request, refuses later dynamics edits and fitting, and restores the seal across resume; its integration test passed in job 22638481 (18 tests with regressions).
Item 5 supplies a sealed bare-engine subclass with fixed known Domino friction and Balloons material masses/drag while leaving task-generation flags unchanged.
Its Boil play integration passed with the other tests in job 22638564 (19 tests); five-domain physical calibration auditing remains pending.
The strengthened standalone test executes a model rollout and checks that engine diagnostic modes refuse; all 19 focused tests passed again in job 22638794.
Mypy and the repository's pytest-pylint checks passed for the original 10 touched implementation/test modules in job 22638902.
The standalone current-state memory regression was reproduced through real play tools in job 22639299: the first completed skill left the probe without inferred memory.
Current-state probes now replay the current episode's skill history under the latest candidate, using observed pre-states and a private deterministic replay stream.
The same candidate also supplies predicate-memory materialization and planning particles.
The edit, checkpoint-reload, reset, and unsupported-primitive-action checks passed in job 22639440.
The other 19 continual regression checks passed in job 22639368; its one failure was the new test's expected ledger total omitting the charged reset, which was corrected before job 22639440.
Raw primitive actions remain executable, but the option-level program cannot reconstruct their effects on hidden memory; current-state probes report this limitation explicitly.
Inferred program memory is available at skill boundaries, not as a primitive-step observer for interrupting a running skill.
Checkpoint testing so far reloads approach artifacts in a live continual session; a fresh-process recording-resume audit remains required.
Replay copies each recorded skill before calling model code so a prediction cannot mutate the recording's option memory; the stronger regression passed in job 22639480.
Item 4 now has a frozen oracle approach and source generation for Domino, Fan, and original Balloons.
Bridge still rejects setup until its observation-memory contract is implemented and audited.
The old Balloons oracle rule module documents momentum loss when a release updates the state during motion, so copying that oracle unchanged would not satisfy the requested ground-truth dynamics comparison.
The supplied Balloons subclass preserves momentum and pins the true lifts, masses, drag, and fade height.
Nine mechanical checks passed in job 22639456: three source/calibration checks, two Balloons release schedules including a release during flight, and four Fan wind/contact cases.
Balloons pose errors stayed below 5 mm and tied/popped flags agreed; Fan ball coordinates stayed within 10 micrometers with explicit wind-motion witnesses.
These checks do not establish agent outcomes or complete the five-domain oracle audit.
The old Boil oracle omits the one-step consecutive-on delay and uses a different water-volume heating condition; its source cannot be copied unchanged into the ground-truth arm.
The new Boil subclass uses the native effect ordering and infers hidden heat, switch history, and the spill countdown from public observations.
Its first mechanical audit exposed shared burner `prev_on` records across the live and model worlds, shifting modeled boiling by one action.
Oracle state restoration now deep-copies objects so native mutable mechanism records remain private to that world.
The corrected Boil audit passed five checks in job 22640531, covering heating, insufficient water, filling/overflow, uncovered-faucet spilling, and continuation from restored inferred memory.
Contradictory privileged heat inserted into a restoration input is ignored.
The continual integration and existing MB/MF regressions passed together (21 tests, job 22640532), including a complete scripted oracle-agent play round.
The 13-module type/lint check passed in job 22640592.
The final 12-module type/lint check, including the new oracle files and recording-isolation assertion, passed in job 22639507 after fixing one overlong source string.
No comparison experiments have been submitted.

## Runtime compatibility still required

The isolated checkout inherited a version without the original Balloons distribution selector and bounded-Wait/current-joint interface present in the frozen MB/MF runtime `b09217bb3` / `41e433427`.
The missing pieces are documented by source comparison and the original commits `2db66d5a2` and `41e433427`.
Restore those behaviors before launch rather than dropping the configuration keys or comparing changed action interfaces.
An audit of the actual CLI parser for all ninety comparison configurations is submitted as job 22640624.
The concurrent uncertainty checkout and the running frozen MF source must remain untouched.

## Reporting and launch requirements

Freeze the verified runtime before launching; do not allow requeues to import a changing development checkout.
Run expensive validation and experiments on compute nodes using `mit_preemptable`.
Record configuration, runtime commit, account, Slurm job and scorecard paths in a manifest.
Report per-seed wins, total charged steps and resets, and per-arm per-domain averages.
Average steps only over whole-run successful seeds and state the qualifying count.
Infrastructure failures and unfinished runs are not failed agent seeds.
An oracle replay audit is mechanical validation, not an agent result.
Preserve the existing MB/MF results and do not relaunch them as part of this sweep.
