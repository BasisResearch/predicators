# One simulator contract

New MB simulator artifacts export `RESIDUAL_ENV`, a subclass of the injected `BaseSimulator` or a supplied domain base simulator.
The class declares `AGENT_PARAM_SPECS` and `RESIDUAL_FEATURES`, implements its mechanism in `_domain_specific_step`, and may use ordinary Python functions and the existing physics-command helpers.
The agent no longer chooses between subclass and rule artifact formats.
Historical rule artifacts remain loadable for replay and checkpoint compatibility.
MF instructions and observations remain unchanged.

`BaseSimulator` is concrete in balloons, boil, bridge, domino and fan.
It keeps the environment in visible-physics mode and replaces the hidden dynamics hook with a no-op that the model overrides.
The agent's hook still runs, without enabling the real environment's hidden mass, drag or process defaults.
Declared names from the base's physical parameter menu are forwarded to its existing setters; new constants are read through `agent_param`.
Existing predicate and sampler parameter views share the deployed subclass values without retaining a reference to the entire approach.

## State and observations

Optional `MODEL_STATE_INIT` declares the model's persistent hidden state.
Each rollout and real execution tracker owns a deep copy.
The class's `update_model_state(observation, model_state, params, action)` class method advances it using a sanitized observation and the preceding action.
The same update runs once per simulated or observed transition.
The first observation initializes state without advancing it.
Longer history belongs in the declared state, so branching and checkpoint restoration can reconstruct it.
Physical effects remain in `_domain_specific_step`; execution tracking never steps or mutates the real physics environment.

Model state travels in `State.latent` during planning and is restored even when two branches have identical observable poses.
Parameters are supplied at their currently deployed values.
Changing a model's code creates a new model instance; resets and fresh fit rollouts cannot inherit another episode's model state.
Real recordings remain raw observations without inferred model state.
Continual observations, belief frames and predicate abstraction attach inferred memory separately.
Repeated free observations do not advance it.
A change in model code or deployed parameters replays the observed episode prefix into a fresh tracker, and resume reconstructs it from the recording.
`sim.reset(current=True)` loads a present candidate without fitting and refreshes that inferred memory before copying the current observation.
Editing or fitting mid-episode therefore does not require another real action or a separate `env_observe` call before a current-state rollout.

## Fitting and compatibility

Subclass fitting and validation use complete simulator rollouts.
Fitting remains explicit through `sim.fit`; diagnostic probes never fit or deploy parameters implicitly.
The existing noise filter, interval observations, carried posterior, evidence and parameter ensembles remain shared infrastructure.
Parameter-free subclasses are valid models and must still support replay validation.
The migration retains the existing rule loader and its historical behavior behind the compatibility path.

## Validation

The first regression reproduces missing execution-time latent tracking for a loaded subclass and checks prediction, branching, restoration, parameter updates and reset isolation.
Additional regressions cover the public validation and residual tools, parameter-free models, all five domain bases, live predicate parameters and the continual observation lifecycle.
The current-state probe regression first reproduced a stale estimate after a parameter change through the real continual play loop.
Its companion edits a subclass artifact during an episode and checks reconstructed memory at carried parameters.
Prompt goldens are regenerated from their builders and verified together with the migration.
Twelve rendered normal-MF and minimal-knowledge MB/MF prompt combinations are byte-identical to the pre-migration commit `e8d7b6180`, including noise and repair-flag combinations.
Expensive tests, formatting and type checks run on compute nodes on `mit_preemptable`.
No agent experiment is authorized by this refactor alone; the previous cancellations remain in force.

Validation completed on 2026-09-09:

- Whole-repository mypy passed for 885 source files.
- The final whole-repository pylint command passed, with 13 files checked and 901 unchanged files accepted from its successful cache entries.
- Formatting passed with yapf 0.32.0 and docformatter 1.4; a separate whole-repository import check passed with CI's isort 5.10.1.
- The SDK/code-learning and integration sweeps covered 864 tests and exposed six prompt/report regressions.
  Those were corrected, and the affected suites passed on rerun (45 tests).
- The final continual model-edit and generated-prompt checks passed (24 tests).

The check logs are in `/home/ycliang/predicators/logs/simulator_subclass_20260909/`.
Final job IDs are `22386961` (format/type/goldens), `22386595` (continual/goldens), `22387004` (CI-pinned imports and changed-file lint), and `22387211` (whole-repository lint).
This was targeted functional validation, not a full-repository pytest run or GitHub CI run.

## Balloons follow-up

This migration does not establish that MB now solves the failed balloons seed.
That original-task run retained a no-op simulator, so fixing the artifact interface alone does not supply the missing causal model.
The shared learning instructions retain the distinction between an optional unsupported mechanism and a goal-required mechanism that must be implemented as a labelled hypothesis with a confirming experiment.
The next balloons check must inspect the learned model and its full recorded-action replay before drawing conclusions about real solve rate.
The separate hatch prototype remains a domain change and must keep its results separate from original-task results.
