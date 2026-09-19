# Simulator probe API

[Documentation index](../README.md) | [Learned models](README.md)

Status: checked against the working-tree implementation on September 19, 2026, with HEAD `1ed5f7e7d`.
This documents the `sim` object exposed to agents through `run_python`, not the simulator subclass interface used to implement dynamics.
The source of truth is [BeliefProbe and its result types](../../predicators/agent_sdk/belief_probe.py).
Historical experiments run from frozen worktrees can expose different interfaces.

## Predictions are not environment wins

`sim` runs the session's candidate or deployed dynamics model without taking real environment actions.
For supplied-model arms, it queries the supplied simulator instead.
Model predictions can be wrong, and reaching goal predicates is not necessarily an evaluator-accepted solve.
Only the live environment's `WIN` establishes actual completion.

There are three distinct quantities:

| Quantity | Meaning |
|---|---|
| `goal_reached` | The rollout reaches the task's goal predicates. |
| Per-trial `solved` | The task evaluator accepts the simulated trajectory, including its additional rules. |
| Live environment `WIN` | The executed trajectory is accepted in the environment. |

For example, a Domino target can topple through unintended robot-body contact while failing the evaluator's fingertip-only counterfactual replay.
Correct dynamics alone do not make a goal-predicate check equivalent to that evaluator check.
Even evaluator acceptance in simulation is conditional on the model used for physical replay.

## `sim.run`

```python
sim.run(
    plan_text,
    render=True,
    trials=1,
    solved=False,
    contacts=False,
    physics_sweep=False,
    seed=None,
    fresh=False,
    belief_draws=0,
)
```

Plans contain one skill invocation per line, using `Skill(obj:type, ...)[parameters]`.
Optional `-> {atoms}` annotations are checked against each step's post-state.
Use the skills and object names available in the current task.

| Mode | Variation between rollouts | Full task evaluator? | Advances probe state? |
|---|---|---|---|
| `run(plan)` | One rollout at the current state and model | No | Yes |
| `run(plan, trials=N)` | Motion-planner seeds; same nominal start and dynamics | No | No, for `N > 1` |
| `run(plan, trials=N, solved=True)` | Same as trials | Yes | No |
| `run(plan, belief_draws=K)` | Plausible starting states sampled from the observation belief; fixed planner seed | No | No |
| `run(plan, physics_sweep=True)` | A grid across physical-parameter uncertainty ranges, plus fitted values | No | No |
| `run(plan, contacts=True)` | One rollout with contact tracing | No | Yes |
| `run(plan, seed=S, fresh=True)` | One rollout at the requested seed on a fresh environment, when available | No | No |

Repeated trials use fresh physics environments when the session provides the required scope.
Inspect the result's notes and `fresh_env_per_trial` rather than assuming all sessions support independent fresh rollouts.
Replaying a snapshot on a shared physics environment is not equivalent to fresh trials because solver state can affect the result.

### State uncertainty, planner randomness, and parameter uncertainty

`belief_draws=K` samples where objects may actually be given the noisy observation.
It uses the current observation belief when the probe starts at that observation; otherwise it constructs a belief around the probe state using the declared noise.
It requires enabled and declared observation noise.

`trials=N` varies motion-planner random seeds, not starting-state samples or physical parameters.
The `seed` argument sets the base planner seed, and trials report their individual seeds.
Supplied and learned engine-backed candidate models now have candidate-aware fresh-world support for trials, state draws, physical sweeps, and explicit `fresh=True` single runs.
It preserves the deployed model and parameters without refitting or adding rollouts; sessions without that support still report the limitation.
Fresh worlds remove shared-engine history, but do not establish that inferred hidden state or noisy geometry is correct.

`physics_sweep=True` uses a grid spanning the model's physical-parameter uncertainty ranges, described by the implementation as the plus/minus-one-sigma range.
It is not random sampling from a parameter posterior, and its passing fraction is not a posterior success probability.
It requires the relevant uncertainty machinery and fresh-environment support.
Read the notices if no identified parameter interval is available.

### Evaluator-scored trials

```python
# full_plan includes the complete sequence from the level's initial state,
# not just the final push or a suffix after placements.
sim.reset()
result = sim.run(full_plan, trials=3, solved=True)
for trial in result.trials:
    print(trial["goal_reached"], trial["solved"],
          trial["reward"], trial["note"])
```

`solved=True` requires `trials >= 2`, an available task evaluator, and the task's unmodified initial state.
Call plain `reset()` before this check; `reset(current=True)`, state modifications, or a state-advancing rollout invalidate the required starting condition.
The evaluator can depend on the whole trajectory, so replay the executed prefix together with the proposed suffix.
This reconstructs a model-predicted trajectory; it does not certify that the current real state matches that replay.

`result.successes` still counts goal-reaching trials, not evaluator-accepted trials.
Inspect each trial's `solved`, `reward`, `note`, and `evaluation_status` instead.
`evaluation_status` distinguishes `accepted`, `rejected`, `unavailable`, `error`, and `not_requested`.
A missing verdict (`solved is None`) is never acceptance.
Missing per-step trajectories make certification unavailable; evaluator exceptions report `error` with an exception-type diagnostic in `evaluation_error`.
Neither condition is counted as evaluator success, including during `refine(require_solved=True)`.
Domino counterfactual checks replay the deployment's Push controller on the candidate's own physics, including its deployed parameters, for supplied, learned, and scene-built models.
They do not substitute ground-truth dynamics for a learned model.

### Mode restrictions

- `belief_draws > 0` cannot be combined with multiple trials, `solved`, `contacts`, `physics_sweep`, or `fresh`.
- `physics_sweep=True` cannot be combined with multiple trials, `solved`, `contacts`, or `fresh`.
- `contacts=True` is single-run only.
- `fresh=True` is single-run only.
- Explicit-uncertainty-disabled sessions reject belief draws and physics sweeps.

In particular, the current API cannot evaluate task acceptance for each belief draw through a combined `belief_draws`/`solved` call.
Passing separate uncertainty and evaluator checks is not equivalent to evaluator acceptance across every uncertain state and parameter combination.

### Result objects

All result objects have a printable summary and a `.text` representation.

| Result | Main fields |
|---|---|
| `ProbeResult` | `steps`, `goal_reached`, `final_atoms`, `final_state`, `states`, `notes` |
| `ProbeTrialsResult` | `trials`, `successes`, `fresh_env_per_trial`, `notes` |
| `ProbeBeliefResult` | `draws`, `successes`, `from_belief`, `notes` |
| `ProbeSweepResult` | `points`, `successes`, `notes` |

Single-run `states` contains the recorded predicted trajectory, with low-level states where the model supports them and otherwise option-boundary states.
`contacts=True` adds contact-pair spans to step reports, useful for identifying what actually caused an object to move.
Do not discard notices when deciding whether a check is trustworthy.

## State and inspection

| Call | Purpose |
|---|---|
| `reset()` | Start from the current task's initial state. |
| `reset(task_idx=0)` | Start from an indexed training task; required when synthesis has no current solve task. |
| `reset(current=True)` | Start from the latest recorded real observation. |
| `reset(mods=...)` | Apply object-feature modifications to a copy of the starting state. |
| `snapshot()` / `restore(id)` | Save and restore probe state for branching experiments. |
| `drop(id)` / `clear_snapshots()` | Release saved snapshots. |
| `task()` | Describe the selected task. |
| `state()` / `state("object_name")` | Inspect full-precision state features. |
| `atoms()` | Inspect predicates holding in the probe state. |
| `render(label, annotations=None)` | Save a scene image, optionally with annotations. |
| `belief(draws=None)` | Inspect the observation belief and estimated atom frequencies. |

These operations do not reset, modify, or execute actions in the live environment.

## Search and parallel rollouts

`refine(sketch_text, timeout=60, require_goal=False, require_solved=False)` searches continuous skill parameters from the current probe state.
By default, success refers to the sketch's subgoal requirements, not task completion.
`require_goal=True` additionally requires the goal, while `require_solved=True` also requires evaluator acceptance and the same pristine initial-state condition as evaluator-scored trials.
Inspect the returned verdict and replay the resulting plan continuously before relying on it.

`suggest_probes(sketch_text, max_draws=20, top_k=3)` ranks feasible parameter alternatives by model-ensemble disagreement on annotated subgoals.
It provides advice and does not execute or replace the proposed action.

`run_policy(trials=1, seed=None, source=None)` tests a closed-loop `policy.py` from the current probe state.
It has no `solved=True` argument and is not an evaluator-acceptance certificate.

```python
sim.reset()
handles = [sim.run_async(plan, trials=3, solved=True)
           for plan in candidate_plans]
print(sim.gather(handles, timeout=30))
for handle in handles:
    if handle.done():
        print(handle.error if handle.error else handle.result)
```

`run_async` launches the same run modes without advancing the parent probe state and disables rendering.
Results refer to the session and model as of launch.
`gather` reports completed, pending, and stale results; pending handles may be gathered again.
Its stale check tracks the simulator file and should not be interpreted as comprehensive validation of every dependency change.

## Model-learning diagnostics

These methods depend on the session's providers and the agent arm's permitted tools.

| Call | Purpose |
|---|---|
| `fit(...)` | Fit candidate model parameters; availability and publication depend on the learning session. |
| `validate(...)` | Replay recorded actions and report prediction errors, not future task acceptance. |
| `residuals(...)` | Inspect model-versus-recording discrepancies and supported fitting/sweep diagnostics. |
| `score(...)` | Score a program world model against recorded trajectories, not a plan's task success. |
| `predicates(...)` | Reload and inspect learned predicates in an authoring session. |
| `samplers()` | Reload and inspect learned skill-parameter samplers. |

Frozen supplied models disable fitting and model edits.
Standalone program models and no-uncertainty arms expose restricted surfaces; a method listed here is not a promise that every agent can call it.
See [exploration tool surfaces](../../predicators/agent_sdk/tools/exploration.py) for session-facing capability restrictions.

The solve-time namespace deliberately does not expose a general `evaluate_trajectory` function; evaluator access is gated through `run(solved=True)` and `refine(require_solved=True)`.
Other learning namespaces or historical prompts may expose additional helpers, so check the actual session rather than copying a dated prompt example.

## Further reading

- [Simulator subclass interface](subclass-simulator.md): implementing a dynamics model.
- [Uncertainty explained](../uncertainty/explained.md): state beliefs and parameter uncertainty.
- [Continual protocol](../protocol/overview.md): real interaction and task evaluation.
