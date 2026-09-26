# Fifteen-seed MF log integrity review

Reviewed September 15, 2026.
Selection: all fifteen MF runs linked in [the eight-agent sweep](eight-agent-five-domain-sweep.md), using [the archived noisy sweep snapshot](../uncertainty-results/noisy-sweep-snapshot.json).

## Assessment

The saved logs do not support an unconditional claim that all MF runs are free of information leakage.
No explicit tool call was found loading hidden environment dynamics, querying a live physics client, or reading another experiment's answers.
However, Bridge exposes hidden attachment information, multiple domains expose precise controller collision geometry, and Balloons seed 1 attempted to inspect internal configuration.
The configuration attempt was blocked before execution.
These are observation-boundary and benchmark-integrity findings; they do not establish how much each issue changed solve rates or costs.

The baseline is a direct coding agent with parameterized controllers, supplied controller source, public observations and renders, recorded trajectories, ordinary Python analysis, and persistent text memory.
It has no supplied predictive simulator or MB fitting interface.
It may construct its own empirical models and fit them to its recorded observations.
That permitted self-modeling should not be classified as cheating.

## Coverage and limits

The review enumerated all 57 saved play Markdown files and screened all 1,571 recorded tool-call blocks, including commands and file contents printed outside their JSON arguments.
Screening covered hidden environment imports, configuration inspection, simulator-state queries, privileged metadata, physics-client queries, external paths, and supplied simulator tools.
Flagged calls were checked against their saved results, and relevant behavioral explanations and journals were inspected.
An additional scan of agent-authored Python files in the main sandbox excluded supplied references and internal worktree directories.
The existing detailed six-run Bridge/Balloons recording audit remains applicable; it checked 8,075 public frames.
This review did not independently decode every trajectory frame in the nine Boil/Fan/Domino runs.
It did not reconstruct every subprocess or every delegated agent's separate transcript, and oversized tool outputs may be absent from the saved play Markdown.
The result is a review of available experiment logs, not a proof of isolation against arbitrary code.

All fifteen selected scorecards record runtime b09217bb38f2.
The original MF cohort and its outcomes remain unchanged.

## Findings by domain

| Domain | Seeds reviewed | Finding |
|---|---|---|
| Bridge | 0, 1, 2 | Supplied source describes welded assemblies and solution-specific handling; seed 1 explicitly uses a hidden attachment named in a controller failure. |
| Fan | 0, 1, 2 | Observed speed and switch-latency calibration explains transfer; seed 2 records exact collision penetrations as controller calibration measurements. |
| Domino | 0, 1, 2 | Agents design analytical cascade geometry; seeds 0 and 1 explicitly infer object dimensions from controller collision reports. All three still fail the test level. |
| Boil | 0, 1, 2 | Agents average noisy trajectories, calibrate filling/heating, and use timed Wait; seed 1 deliberately probes zero-step planning failures and uses collision diagnostics for geometry. |
| Balloons | 0, 1, 2 | Agents fit their own dynamics; instantaneous goal is used. Seed 1 attempts internal configuration inspection, but the tool call fails before execution. |

## Specific evidence

### Bridge attachment disclosure

[Seed 1, play 003](/home/ycliang/predicators/logs/agent_continual_model_free/bridge-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_144727/agent/003_play_20260910_151152.md:1019) explicitly says the failure message naming a welded span establishes that the bond formed.
The [prior detailed audit](mf-integrity-audit-20260912.md) records the returned attachment label and exact penetration depth.
This is a confirmed leak through shared controller feedback, not evidence of an agent secretly importing the environment.
MB receives the same shared controller interface.

### Fan collision geometry

[Seed 2 journal](/home/ycliang/predicators/logs/agent_continual_model_free/fan-agent_continual_model_free_noise_sweep_r1/seed2/run_20260910_164315/agent/sandbox/journal.md:75) lists switch penetrations of -0.0354, -0.0045, and -0.0088 m and derives how penetration changes with approach distance and height.
The agent also estimates ball speed and switching latency from its own recorded trajectory.
The latter is allowed learning; precise internal collision diagnostics provide an additional measurement channel beyond noisy object features.

### Domino collision metrology

[Seed 0, play 003](/home/ycliang/predicators/logs/agent_continual_model_free/domino_high_friction_turn-agent_continual_model_free_noise_sweep_r1/seed0/run_20260910_174127/agent/003_play_20260912_080522.md:2079) returns a zero-step collision refusal with a -0.0033 m object separation.
[The following explanation](/home/ycliang/predicators/logs/agent_continual_model_free/domino_high_friction_turn-agent_continual_model_free_noise_sweep_r1/seed0/run_20260910_174127/agent/003_play_20260912_080522.md:2099) explicitly proposes fitting the true domino footprint from collision reports.
Seed 1 similarly uses collision refusals to constrain width.
The simulator-state/PyBullet code seen in displayed Place references is supplied controller source, not an agent-issued live-state query.

### Boil free planning probes

[Seed 1, play 003](/home/ycliang/predicators/logs/agent_continual_model_free/boil-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_192202/agent/003_play_20260912_151728.md:2251) explicitly treats zero-step collision checking as a free geometry probe and tests competing interpretations of the faucet position.
Its [journal](/home/ycliang/predicators/logs/agent_continual_model_free/boil-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_192202/agent/sandbox/journal.md:339) describes controller diagnostics as a metric ruler for locating the grasp window.
The successful behavior also uses empirical fill-rate estimation, noisy-pose averaging, and ordinary Python/image analysis.

### Balloons internal-settings attempt

[Seed 1, play 002](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_201533/agent/002_play_20260912_093930.md:579) issues a Python command importing CFG and GlobalSettings and printing fields whose names contain balloon.
The saved Tool Error states that the PreToolUse validator could not be located after working-directory drift.
The command therefore did not execute and no configuration values were returned by that call.
This is an attempted internal-configuration probe, not a confirmed acquisition of hidden parameters.
No subsequent explicit configuration-inspection call was found in the screened tool blocks.

### Sandbox recovery

Several runs attempt worktree, Monitor, relative-path, or validator-shim repairs after the tool working directory drifts into the reference directory.
Bridge seed 1 requests a Bash call with dangerouslyDisableSandbox enabled, but the saved hook error blocks it.
Domino seed 2 writes a shim that delegates to the original validator.
The reviewed recovery commands concern the same run's files; no cross-run answer read was found.
These events nevertheless limit claims that the sandbox boundary is robust.

## Reporting implications

Keep the recorded outcomes as outcomes under the implemented historical protocol.
Describe MF as a direct coding agent without supplied simulator/model-learning tools.
Disclose supplied controller source and diagnostic access.
Do not attribute the latest performance gap entirely to learning hidden dynamics under the declared observation noise.
For a clean comparison, the shared controller observation boundary needs evaluation for both MB and MF, followed by a separate cohort if that boundary changes.
Changing only MF would not isolate the effect fairly.

No experiments were launched, restarted, or changed for this review.

