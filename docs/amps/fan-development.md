# Fan domain development record

Goal: find a physically valid Fan setting where predictive simulation improves robust completion relative to direct control.
All entries here are development experiments, not held-out paper results.
Keep every configuration and failed attempt rather than selecting seeds after seeing outcomes.
Agents must share layouts, observation access, real-step budgets, reset rules, and task certification.
Simulator access and its associated harness remain the intended method difference.
Expensive checks and agent runs use compute nodes.
Before launching agents on any new setting, show the user visualizations of its actual training and test tasks and wait for review.
Physical diagnostics may establish feasibility before that review, but are not agent experiments.

## Decision rule

Screen each physically validated configuration on matched seeds 0 and 1 for both agents.
Inspect both successes and failures for interface errors, privilege leaks, simulator mismatch, and actual strategy before changing the domain.
A promising candidate has EMPIRIC solving both screening seeds while the direct agent fails at least one for task-related reasons, not account limits or infrastructure failures.
Freeze that candidate and evaluate both agents on fresh seeds 2 through 4 before making any robustness claim, following the user's revised request for five total seeds per agent including the pilot.
Those three fresh seeds are confirmation data only if their results are not used to revise the candidate; report them separately from the two screening seeds when assessing confirmation.
If confirmation fails, retain the negative result and label subsequent experiments as further development.
This is a screening rule, not a statistical significance claim or a guaranteed outcome.
For the inertial candidate, `scripts/configs/predicatorv3/continual_fan_inertial_confirmation_r1.yaml` now prepares separate confirmation run keys for seeds 2 through 4.
Configuration resolution produced exactly ten runs with environment, arguments, and flags identical to the frozen inertial pilot; equality was rechecked immediately before submission.
After both EMPIRIC pilot seeds passed, confirmation arrays `23252966` (EMPIRIC) and `23252967` (direct), seeds 2 through 6, were submitted on 2026-09-20 to `mit_preemptable` from frozen runtime `8d12ae07d89a994889b03f5cfe2488dacbdf8140`.
At the user's subsequent request for only three additional seeds per agent, tasks `23252966_5`, `23252966_6`, `23252967_5`, and `23252967_6` were cancelled without inspecting their outcomes for selection.
Their partial logs are preserved, are not task failures, and must not be automatically resumed; seeds 2 through 4 continue unchanged.
Each task has 8 CPUs, 16 GB, and 12 hours, with automatic requeue and account selection among a through d; dat remains backup.
No domain or agent settings changed for confirmation.
The first finished confirmation run is direct seed 3: `all_levels_won`, 1,262 total steps, with both training and test accepted by the evaluator.
Its scorecard is `logs/agent_continual_model_free/fan_inertial-mf_opus_inertial_confirmation_r1/seed3/run_20260920_111039/scorecard.json`.
EMPIRIC confirmation seed 2 also has accepted evaluator wins on both levels, at 1,328 total steps.
Its scorecard is `logs/agent_continual/fan_inertial-mb_opus_inertial_confirmation_r1/seed2/run_20260920_111039/scorecard.json`.
EMPIRIC confirmation seed 4 has accepted wins on both levels at 1,305 total steps, and direct confirmation seed 2 has accepted wins on both levels at 2,614 total steps.
Their scorecards are `logs/agent_continual/fan_inertial-mb_opus_inertial_confirmation_r1/seed4/run_20260920_111039/scorecard.json` and `logs/agent_continual_model_free/fan_inertial-mf_opus_inertial_confirmation_r1/seed2/run_20260920_111039/scorecard.json`.
Direct confirmation seed 4 finished with accepted wins on both levels at 2,067 total steps (1,507 training and 560 test), with no resets.
Its scorecard is `logs/agent_continual_model_free/fan_inertial-mf_opus_inertial_confirmation_r1/seed4/run_20260920_111039/scorecard.json`.
Direct control therefore solved all three fresh confirmation seeds, so the pilot solve-rate gap did not replicate on this confirmation batch.
EMPIRIC confirmation seed 3 also has evaluator-accepted wins on both levels: 311 training steps and 1,660 test steps, 1,971 total with zero resets.
Its scorecard is `logs/agent_continual/fan_inertial-mb_opus_inertial_confirmation_r1/seed3/run_20260920_111039/scorecard.json`; postmortem processing was still active when the accepted episodes were checked.
Fresh inertial confirmation therefore finished 3/3 for each arm: the pilot solve-rate gap did not replicate.
Across pilot and confirmation, EMPIRIC is 5/5 and direct control is 4/5, but the pooled difference must not be presented as an independently confirmed advantage.
The successful direct seed 4 journal records four rest-to-rest test pulses: two along x and two along y, without an opposing braking pulse, completing the test in 560 steps.
It calibrated a scalar pulse-duration-to-displacement relationship and remeasured stopped positions between pulses.
Direct seed 3 instead records a small fitted dynamics model, test-time refitting, and an opposing-pulse correction after overshooting its desired x position, completing the test in 797 steps.
These are agent-reported strategies in the corresponding `agent/sandbox/journal.md` files, consistent with their accepted scorecards; the fitted parameter interpretations have not been independently replayed.
The direct baseline can write its own prediction code, so absence of the EMPIRIC harness does not imply absence of modeling or informal uncertainty reasoning.
The inertial task still permits stopping, observing, recalibrating, and axis-by-axis progress; its confirmation results do not establish a need for the full harness.
The already-launched ramp pilot tests an additional physical effect, but must earn its own evidence rather than inherit the inertial screening result.

## Candidates

### Exposed transfer r1

Protected calibration tray followed by an exposed L-shaped route, unchanged original pilot.
The direct agent learned and refined a simple pulse-duration model and solved seed 1 in 724 steps without resets.
EMPIRIC seed 1 also solved, in 1,095 steps.
At the user's request, unfinished seed-0 jobs `23232486_0` and `23232487_0` were cancelled on 2026-09-20; their partial logs and both completed seed-1 results are preserved.
The cancelled seeds are not task failures and must not be automatically resumed.
This does not establish the desired completion advantage.
See [the original pilot](fan-exposed-transfer.md).

### Inertial transfer r1

Same geometry, lower drag, and weaker wind, with persistent momentum and optional active braking.
All 27 environment checks passed; privileged plans replayed successfully through actual switches on both screening seeds.
Arrays `23239940` (EMPIRIC) and `23239941` (direct) run matched seeds 0 and 1 from frozen commit `8d12ae07d89a994889b03f5cfe2488dacbdf8140`.
Direct seed 0 finished with `all_levels_won`: 924 training steps plus 653 test steps, 1,577 total, zero resets.
Its scorecard is `logs/agent_continual_model_free/fan_inertial-mf_opus_inertial_pilot_r1/seed0/run_20260920_092144/scorecard.json`.
EMPIRIC seed 0 also finished with `all_levels_won`: 406 training steps plus 999 test steps, 1,405 total, zero resets.
Its scorecard is `logs/agent_continual/fan_inertial-mb_opus_inertial_pilot_r1/seed0/run_20260920_092143/scorecard.json`.
Direct seed 1 ended with `agent_ended` after explicitly calling `give_up`: training won in 1,453 steps, test unsolved after 1,914 steps, 3,367 total, zero resets.
Its scorecard is `logs/agent_continual_model_free/fan_inertial-mf_opus_inertial_pilot_r1/seed1/run_20260920_092143/scorecard.json`.
The test episode itself did not trigger a terminal fall or evaluator rejection; the agent forfeited after an overshoot and unsuccessful recovery attempts.
Its trace places the ball beyond the platform edge, about 3 cm lower, and attributes the resulting wedged position to contact with the adjacent fan bank.
The postmortem attributes the overshoot to extrapolating a three-parameter position-dependent force model from two pulses, choosing a 76-step pulse despite a simpler model predicting 53 steps.
That causal interpretation and the claim that recovery was impossible have not been independently replayed.
The same postmortem calls switch states in `trajectories.pkl` hidden, but a frozen-runtime source audit does not support that characterization of the on/off flag.
`pybullet_fan_base.py:913-923` derives observable `fan.is_on` and `switch.is_on` from the same controlling switch, and both agents' initial test observations show all four fan flags.
The shared exporter in `agent_sdk/sandbox_setup.py:688` copies only `State.data`, dropping privileged state and simulator bookkeeping; both agents are explicitly given recorded observable trajectories.
Switch objects omitted from the compact observation text are present in those recordings, but the reported on/off access is redundant with an already observable feature, not evidence of hidden dynamics leakage.
This source-level check addresses that specific concern, not every possible sandbox information channel.
EMPIRIC seed 1 finished with `all_levels_won`: 1,003 training steps plus 1,269 test steps, 2,272 total, zero resets; both episodes have accepted evaluator wins.
Its scorecard is `logs/agent_continual/fan_inertial-mb_opus_inertial_pilot_r1/seed1/run_20260920_092153/scorecard.json`.
The completed pilot is EMPIRIC 2/2 and direct control 1/2, meeting the screening rule but not establishing a robust advantage without fresh-seed confirmation.
On the completed matched seed, EMPIRIC used 172 fewer total steps, but 346 more test steps; both methods solved, so this is not a completion-rate gap.
The direct agent's journal fits a pulse-response rule, while EMPIRIC uses fitted simulation; a training-efficiency difference alone does not meet the completion-rate screening rule.
During seed-0 test planning, EMPIRIC's trace reports a simulated 3.5 mm step between the nominally level platforms, caused by noisy geometry reconstruction, and tests a coplanar hypothesis.
Direct seed 0 has already crossed that join in real execution.
This is a relevant method robustness issue under the shared observation noise, not an infrastructure failure or a justification for dropping the seed.
Direct seed 0 used four forward pulses on test, settling and recalibrating between them, without opposing-fan braking.
Its final observed mean errors were approximately +3.5 cm in x and -3.2 cm in y, close to but within the evaluator's 4 cm tolerance; the physical settling certificate accepted the run.
The success is retained and does not support a completion-rate advantage on this seed.
EMPIRIC seed 0's postmortem reports additional mismatch from simulated versus executed switch durations and target-pad contact, with its final observed position only about 3 mm inside the tolerance boundary.
These are the agent's diagnoses, not independently replayed causal findings; the scorecard independently records an accepted test win.
See [physics and validation](fan-inertial-transfer.md).

### Downhill ramp candidate

User-approved pilot launched on 2026-09-20 after visual review.
Adds an observed convex ramp that descends 4 mm over 40 cm into a lower landing before the turn.
The grade is deliberately shallow so gravity changes speed without creating a large uncontrolled drop relative to slow physical switches.
Training has the same grade with a wide, fenced landing; test uses an exposed narrower path.
Ramp dimensions and rise are visible to both agents and reconstructed in the simulator base.
Geometry, gravity, restoration, and real-controller feasibility must pass before agent launch.
Geometry, gravity, and restoration passed with the full 28-check environment suite in job `23240367`.
The initial single-forward-pulse feasibility search failed to find an accurate safe landing on either seed (array `23240475`); feasibility is not yet established.
An expanded search with opposing-fan braking also failed (`23240884`), with the ball stopping near the ramp entrance rather than falling.
This exposed a collision-geometry issue: the default convex-mesh margin creates an unintended entrance lip relative to the neighboring box support.
The ramp now uses an exact static triangle mesh with zero collision margin; its visible dimensions and intended slope are unchanged.
Strict surface-height, downhill-motion, and simulator-restoration checks passed in job `23241516` (28 checks).
Actual-controller replays on the corrected mesh are array `23241689`; the additional seam-crossing regression is job `23241698`.
The seam-crossing regression passed with all 28 checks: a ball driven from the actual training initial state crossed the ramp and reached x=1.126 m instead of sticking at the entrance.
The corrected-mesh controller search (`23241689`) still failed to reach the target column.
A contact probe (`23242039`) identified physical contact with the original right fan bank at x=1.126 m, inside the enlarged ramp landing.
The ramp variant now moves that bank 40 cm outward in both observed task state and simulator reconstruction, without changing original or inertial-pilot fan positions.
The clearance regression now requires crossing beyond x=1.25 m, not merely reaching the ramp end.
Updated validation, controller replay, and visual-preview jobs are `23242172`, `23242175`, and `23242188` respectively.
The clearance regression passed all 28 checks, reaching x=1.409 m on the protected training landing.
With the accidental obstructions removed, the bounded controller search now produces genuine exposed-edge failures (46/68 and 47/68 candidate trajectories), but still no sufficiently accurate safe landing.
The closest safe plan uses the maximum tested braking duration, so array `23242490` extends that duration from 60 to 120 steps without changing the domain.
This is privileged physical feasibility testing, not evidence about either agent's solve rate.
The updated preview explicitly marks the fan bank outside the deck (`23242406`).
The extended braking search succeeded on both seeds (`23242490`), using opposing-fan braking before the turn.
Seed 0 selected x pulse wait 8, reverse-fan wait 120, and y pulse wait 40; seed 1 selected waits 8, 80, and 42 respectively.
These waits are controller-plan parameters, not total fan-on times or charged physical steps.
The mesh face winding was corrected to outward normals so its top is visible in the actual simulator render; the 28-check suite still passed (`23242634`).
Independent fresh-process replays on the final geometry also succeeded through actual switch controllers: seed 0 in 588 steps (`23242828`) and seed 1 in 553 steps (`23242834`).
These replay counts include conservative settling waits and exclude privileged plan-search cost; they must not be compared as agent sample-efficiency results.
The launch configuration is `scripts/configs/predicatorv3/continual_fan_ramp_pilot_r1.yaml`, with distinct run keys and matched seeds 0 and 1.
Job `23242898` passed all 32 checks, including the resolved launcher and physical regressions.
The reviewed candidate is frozen at `logs/fan-ramp-runtime-20260920`, commit `1510ebdc4b0f5dde005bc72665bd6a0dbe7886ce`.
Following the user's explicit request for two ramp seeds per agent, arrays `23254928` (EMPIRIC) and `23254929` (direct) were submitted from that frozen runtime on `mit_preemptable`.
Each task has 8 CPUs, 16 GB, and 12 hours, with automatic requeue and accounts a through d; dat remains backup.
These four pilot runs are separate from the six remaining inertial confirmation runs and do not alter paper results.
Direct ramp seed 0 finished with `level_lost`: training won in 747 steps, then the evaluator rejected the test because the ball fell off the platform after 1,172 test steps (1,919 total, zero resets).
Its authoritative scorecard is `logs/agent_continual_model_free/fan_ramp-mf_opus_ramp_pilot_r1/seed0/run_20260920_112733/scorecard.json`.
The agent's postmortem attributes the failure to delayed brake activation: `SwitchOn(fan_1)` allegedly took 83 steps to toggle versus 30 in calibration, leaving the ball unbraked during descent.
Independent inspection of the saved observation trajectory confirms fan 0 turned off at state index 1,087 and fan 1 turned on at 1,169, an 82-step interval with both off.
The agent's 83-step figure used an off-by-one off index and is not the duration of the `SwitchOn` invocation itself; that invocation was interrupted after 78 steps according to the execution log.
At brake activation the observed ball x was already 1.3746 m, near the landing edge, consistent with braking too late.
This verifies the delayed activation sequence, not the counterfactual claim that a different sequence would certainly succeed; the terminal fall itself is evaluator-confirmed.
Its claim that its physics model was otherwise accurate is not an independent validation of that model.
EMPIRIC ramp seed 1 finished with `all_levels_won`: accepted training and test wins in 707 and 1,369 steps respectively (2,076 total, zero resets).
Its scorecard is `logs/agent_continual/fan_ramp-mb_opus_ramp_pilot_r1/seed1/run_20260920_112733/scorecard.json`.
The agent reports using repeated static observations to resolve surface heights and adapting the overlapping-fan sequence to observed switch latency; these strategy claims await trajectory-level analysis.
It also reports a narrow final goal margin, so the accepted success should not be described as demonstrated robustness on its own.
EMPIRIC ramp seed 0 also finished with `all_levels_won`: accepted training and test wins in 1,004 and 1,622 steps respectively (2,626 total, zero resets).
Its scorecard is `logs/agent_continual/fan_ramp-mb_opus_ramp_pilot_r1/seed0/run_20260920_112730/scorecard.json`.
Both EMPIRIC screening seeds succeeded and direct seed 0 has an evaluator-confirmed task failure, satisfying the numerical pilot screening rule.
Direct seed 1 remains unfinished and must be retained regardless of outcome.
The next stage is to finish the strategy/interface audit and run matched fresh seeds 2 through 4 on the unchanged frozen ramp runtime; no robustness claim is established by this pilot alone.
After inspecting the successful strategy logs, the direct failure trajectory, and the shared observable-only trajectory exporter, confirmation arrays `23266180` (EMPIRIC) and `23266181` (direct) were submitted for seeds 2 through 4.
The configuration is `scripts/configs/predicatorv3/continual_fan_ramp_confirmation_r1.yaml`.
Immediately before submission, all 1,158 source blobs under the frozen runtime's `predicators`, `scripts`, `prompts`, and `main.py` were checked against commit `1510ebdc4b0f5dde005bc72665bd6a0dbe7886ce`, accounting for symlink contents, and matched.
Resolved environment, flags, arguments, and accelerator settings matched the pilot per arm; only seed range and experiment identifiers changed.
These six compute-node jobs use 8 CPUs, 16 GB, 12 hours, requeue, and primary accounts a through d; dat remains backup.
Keep the three fresh seeds separate from screening results when assessing confirmation, and retain the unfinished direct pilot seed regardless of its eventual outcome.
During its unfinished test run, direct ramp seed 0 implemented Monte Carlo checks over correlated drag and ramp-acceleration estimates and compared delayed-braking survival predictions.
The evidence is its `agent/002_play_20260920_114411.md` log under `logs/agent_continual_model_free/fan_ramp-mf_opus_ramp_pilot_r1/seed0/run_20260920_112733`.
These are the agent's own model predictions, not measured success probabilities or completed task outcomes.
This further demonstrates that the direct baseline can implement uncertainty reasoning without the supplied harness; interpret eventual differences as differences in the complete workflows, not exclusive access to modeling or uncertainty.
The intermediate zero-margin convex-hull attempt (`23241380`, `23241392`) did not establish feasibility and failed a ray-contact check, so it is not the accepted geometry implementation.
No agent experiment is authorized for launch before the user's visual review and successful physical validation.

![Ramp training and test layouts](figures/fan-ramp-overview.png)

![Actual ramp simulation scenes](figures/fan-ramp-scenes.png)

![Ramp side profile with exaggerated vertical scale](figures/fan-ramp-profile.png)

## Reporting

Do not replace the original Fan maze or paper results during development.
Do not count diagnostic reference-policy successes as EMPIRIC results.
Do not count unfinished, account-limited, or infrastructure-failed runs as task failures.
