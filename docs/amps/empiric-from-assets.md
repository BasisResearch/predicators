# EMPIRIC from assets

## Scope

This development arm removes the supplied domain-specific simulator scaffold while retaining EMPIRIC's fitting and uncertainty machinery.
It is registered as `agent_continual_from_assets`.
It does not replace the existing EMPIRIC arm, paper results, or running Fan ramp comparisons.

The agent receives the generic PyBullet wrapper, a robot-bound `SceneBase`, the observation schema, a reconstructed scene manifest, and referenced assets.
It writes a `SceneBase` subclass that constructs the scene and implements domain-specific readout mappings, mechanisms, parameters, and inferred memory.
The loader rejects rule-only artifacts and classes that do not inherit `SceneBase`.
Before a model exists, the planning substrate contains only generic robot infrastructure; simulation rollouts and rendering are unavailable.
Once a model loads, fitting and rehearsal construct the agent's scene class, including independent trial worlds.

The existing `agent_continual_real_to_sim` comparison shares this scene-building implementation but continues to disable fitting and uncertainty.
The new arm instead inherits EMPIRIC's fitting tools, belief handling, parameter estimation and uncertainty-aware rehearsal.
The prompt explicitly distinguishes missing mechanisms from wrong parameter values and asks for observation round trips, independent replay and model-state restoration checks.
It does not require a source-code replica of the benchmark environment.

## Remaining assumptions

The manifest is generated from the benchmark simulator's initial scene, not inferred from images.
Its geometry, articulation and observed-object associations are supplied reconstruction.
The manifest omits masses, friction, restitution and damping, but raw asset files may contain nominal physical constants.
The agent is instructed to treat these as assumptions to validate, not calibrated ground truth.
The generic wrapper still supplies the robot controller, grasp conventions, generic body-pose synchronization and a passive scalar feature store.
The scalar store does not implement domain dynamics or joint/readout mappings.
This is a reduction in supplied domain implementation, not reconstruction from raw vision or a bare-engine-only benchmark.

## Pilot

The configuration is [continual_from_assets_pilot_r1.yaml](https://github.com/BasisResearch/predicators/blob/iclr-empiric-submission/scripts/configs/predicatorv3/continual_from_assets_pilot_r1.yaml).
It runs Opus 5 on seeds 0 and 1 of Fan + ramp, four-span Bridge, Domino, Balloons and Boil.
There are ten intended runs total, each with its training and test levels.
Fan uses the previously reviewed 3 mm ramp with the 10 cm landing extension and the repaired shared skills, not Fan maze.
The environments, observation noise and real-step budgets are inherited from the benchmark menus.
Mandatory skill preflight and validation auditing remain disabled.
The experiment keys use `from_assets_opus_pilot_r1` with the prefixes `fan_ramp`, `bridge`, `domino_high_friction_turn`, `balloons` and `boil`.
Results belong to this separate development cohort and must not be merged into the main EMPIRIC row.

The first questions are whether the agent constructs a usable model, invokes fitting successfully, transfers its model across levels, and preserves inferred state during rehearsal.
Inspect model quality and infrastructure failures separately from task success.
Two seeds per domain are not enough to establish a robust solve-rate difference.

### Launch, September 21, 2026

All four tasks were verified running on compute nodes at 18:50 UTC.
Bridge seeds 0-1 are array `23407427`; Fan maze seeds 0-1 are array `23407428`.
The user immediately corrected Fan maze to Fan + ramp; both tasks of array `23407428` were cancelled with their logs preserved.
The abandoned maze runs are not part of the intended pilot and must not be resumed or counted as task failures.
Bridge array `23407427` remains unchanged.
The [expansion-only configuration](https://github.com/BasisResearch/predicators/blob/iclr-empiric-submission/scripts/configs/predicatorv3/continual_from_assets_expansion_r1.yaml) submits only the eight new tasks, with Bridge and Fan maze disabled to prevent duplicates.
Its runtime menus are pinned to the frozen implementation.
The resolved Fan + ramp flags match the repaired EMPIRIC ramp cohort, apart from making the default fitting flag explicit.
The eight replacement/additional tasks were submitted on account d as the following two-seed arrays:

- Fan + ramp: `23407658`.
- Domino: `23407657`.
- Balloons: `23407655`.
- Boil: `23407656`.

The corrected full pilot configuration passed its capability and domain-selection test before submission.
The frozen agent implementation remains at `485a39d62`; only the submitted environment configuration changed.
Each task requests 8 CPUs, 16 GB and 12 hours on `mit_preemptable`, with checkpoint resume and requeue enabled.
They use account d, which passed a live availability probe before submission.
Account b was session-limited; dat passed the probe and remains backup.

The frozen runtime is `logs/empiric-from-assets-runtime-20260921`, commit `485a39d62`, on its own `codex/empiric-from-assets-pilot-r1` branch.
Its preceding commit `4f9f0480a` preserves the already-existing environment and shared-skill changes from the working checkout.
The main checkout's unrelated changes were left intact, and existing running cohorts were not restarted.
Scorecards and agent artifacts are under `logs/agent_continual_from_assets/<experiment>/seed<N>/run_*/`.
Scheduler logs are under the frozen runtime's `logs/` directory.

## Validation

The end-to-end test exercises the actual continual-session tools with an agent-written scene, rather than a model API mock.
It checks that no domain simulator is injected, a missing model refuses simulation, a rule-only artifact is rejected, a loaded scene supports skill rehearsal, independent trials use fresh worlds, parameter values are isolated across fitting worlds, model memory survives restoration, and an explicit harness fit produces a fit result.
The scene fixture now names the actual observable `water_volume` feature; its former `water_level` declaration had never been exercised by fitting in the no-fitting test.
Existing prompt goldens remain unchanged.

Validation passed 83 distinct targeted tests across scene construction, continual execution, ablations, subclass loading, prompts, candidate isolation and restoration.
The exact frozen runtime passed 41 of these checks and then repeated its six scene-built-agent checks using the launcher's environment after building the robot IK solver.
Seven changed Python files passed type checking and the project's pytest-based lint workflow.
Formatting checks passed.
Two stale regression tests were updated to use the existing instance-based loader API and current no-fitting prompt wording.
The full repository test suite was not run.
