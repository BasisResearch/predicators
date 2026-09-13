# Bridge MB seed 2: four-block transfer failure

The run won the three-block training level in 1,125 steps and lost the four-block test after 1,267 steps, for 1/2 levels, 2,392 total steps, and zero resets.
The final [scorecard](/home/ycliang/predicators/logs/agent_continual/bridge-agent_continual_span_transfer_r1/seed2/run_20260913_052324/scorecard.json) records `level_lost`, not an infrastructure or usage-limit failure.
This failed seed does not enter mean steps over whole-run successes.

## Executable model

The [final simulator](/home/ycliang/predicators/logs/agent_continual/bridge-agent_continual_span_transfer_r1/seed2/run_20260913_052324/agent/sandbox/simulator.py) and both saved level versions are byte-identical, with SHA-256 `a825eab80d3df153319e650a0a9cf6bb31274b09ce1d8b617ee19906251ea5ea`.
The model has no learnable parameters and its domain-specific step immediately returns without adding dynamics.
The agent documented glue application and bonding in prose, but never implemented glue or weld dynamics in its executable residual.
Its test-level journal explicitly says the supplied simulator has glue disabled and is used for geometry and controller checks.
Thus, assignment to the MB arm and availability of uncertainty features did not ensure that this seed learned an executable model of the mechanism that mattered for final transfer.

## What the failure establishes

The [settling check](/home/ycliang/predicators-bridge-balloons-frozen-bridge-20260912/predicators/envs/pybullet_bridge.py:552) advances the live physics client for 60 unactuated substeps and checks the geometric goal again.
The build failed that check after placement.
The rejection text about an unwelded row is generic; it does not establish that weld constraints were absent or broke.
The agent's explanation involving grasp tilt, stored constraint energy, and snapping welds is a hypothesis rather than an independently reproduced diagnosis.
The missing executable mechanism prevents this run from demonstrating successful model-based prediction of the welded transfer, but does not prove that adding such a model alone would prevent the failure.

The [audit artifact](/home/ycliang/predicators/logs/bridge_mb_extension_20260913/seed2-final-20260913.json) records source identity, model hashes, outcome evidence, and these causal limits.
Keep this result in the original cohort; any future agent correction should use a separately identified comparison.

## Matched seed-2 scene evidence

A later audit compared MB seed 2 with the completed oracle-scene seed 2.
Every recorded initial object feature matches exactly on both training and test levels.
Their Bridge environment, base environment, and continual runner files have identical SHA-256 hashes.
Of 53 checked task, noise, and controller configuration values, 52 match literally; the remaining displayed value differs only in a process-specific function address.
The oracle-scene run solved both levels in 2,916 steps with zero resets, including 1,606 test steps.
This is strong evidence that this four-block layout admits a successful execution under the same task mechanics.
It does not prove robust solvability across all layouts or explain why MB's particular construction failed.
The oracle-scene arm is an agent with fixed scene-only predictions, not an oracle action policy, and its Bridge simulator also omits curing and weld dynamics.
Its success therefore also prevents attributing MB's failure solely to the absent residual mechanism.
The recorded states omit privileged state, and this audit did not replay either trajectory.
The full comparison is saved in `/home/ycliang/predicators/logs/bridge_mb_extension_20260913/seed2-task-match-20260913.json`.

A separate code review found that the final 60-substep certificate bypasses ordinary domain updates, including weld relaxation intended to suppress numerical creep.
Testing that difference against normal waiting remains necessary before treating it as a cause or changing the success rule.

## Recorded-action replay gate, 2026-09-13

Compute arrays `22678198` and `22678239` replayed the recorded MB and oracle-scene seed-2 test actions before comparing the existing certificate with three ordinary hold-position steps.
These are mechanical diagnostics, not agent runs or new solve-rate seeds.
Both replays begin with exactly matching recorded observations but diverge at action 38 of the initial PickBottle invocation, affecting bottle and robot features.
The first robot position discrepancy reaches approximately 0.289 mm and later differences grow substantially.
Matching the original environment cache setup in the second array did not eliminate this divergence.
The replayed MB layout already fails the geometric goal before certification, whereas the recorded run reached the certificate with a candidate goal layout.
Consequently, neither replay can establish the cause of the original settling rejection.
The replayed oracle-scene layout survives both checks, but this does not validate the MB counterfactual.
The next reproduction should include the public skill-controller calls and their simulator side effects, rather than assuming recorded joint commands alone reconstruct the original execution.
No experiment runtime or success rule was changed, and all unfinished Bridge baselines remain held.
The assessment and per-mode artifacts are under `/home/ycliang/predicators/logs/bridge_mb_extension_20260913/settling-audit-assessment.json`.
