# Continual agents with minimal built-in knowledge

The minimal arms share the continual protocol's observations, goal descriptions, persistent sandbox, journal, trajectory recording, evaluator and pooled step budget.
They receive no primitive skills, supplied predicates, helper objects or domain-specific simulator.
This boundary is enforced by their approach constructors, tool rosters and reference-file selection, including when configuration flags request additional scaffolding.

| Approach | Supplied simulator | Supplied skills |
| --- | --- | --- |
| `agent_continual` | Domain-specific base simulator | Yes |
| `agent_continual_model_free` | None | Yes |
| `agent_continual_model_based_minimal` | Generic `PyBulletEnv` class only | None |
| `agent_continual_model_free_minimal` | None | None |

The minimal model-based arm receives sandbox copies of `PyBulletEnv` and its abstract `BaseEnv` dependency.
The copy imports the copied dependency, so importing it does not load the repository's environment registry.
The agent implements scene construction, domain dynamics, fitting and rollout evaluation in its own Python files.
It can run these programs with `python3` and use their predictions to choose real actions.
It receives no prebuilt `sim` probe, residual-fitting pipeline or evaluator for hypothetical trajectories.

The minimal model-free arm receives no simulator source.
It can analyze recorded experience and write its own action-selection policy and reusable control routines.

## Low-level control

Both minimal arms expose `env_observe`, `env_step`, `env_reset`, `give_up` and `env_run_policy`.
Their control observation includes object features, the goal, step counts, action bounds, and robot joint positions and actuator order where available.
Joint commands are absolute position targets; a null lower or upper bound means that side is unbounded.
The policy receives only a JSON observation, including the configured observation noise, with no live environment, evaluator or skill objects.

An agent can write a sandbox file defining `get_action(observation, memory)` and invoke it with `env_run_policy(path="policy.py", max_steps=100)`.
The function returns one numeric action vector, or `None` to return control to the agent.
Its module globals and memory dictionary persist within that invocation.
The agent must save anything needed across invocations or preemptions in its sandbox files.

Every returned action passes validation and then goes through `ProtocolSession.step`, updating the recording and sandbox data after each step.
Execution stops at the requested step count, `None`, a terminal episode, a run cap, a timeout or a policy error.
Already executed actions remain charged after an error, and the environment is never automatically reset.
Policy execution runs in a separate interpreter with the existing sandbox Python guard and blocks concurrent environment commands until it finishes.
Worker diagnostics are saved under `tool_outputs/policies/`.

Domino trajectories whose actions carry no option labels use the evaluator's existing label-free certificate.
Staging checks and the counterfactual cascade probe still apply; the absence of a supplied `Push` skill no longer forces rejection of a physical push.

## Sweep

`scripts/configs/predicatorv3/protocol_continual_minimal_knowledge_sweep.yaml` inherits the uniform sweep and adds the two minimal arms.
It expands to four approaches across balloons, boil, bridge, domino and fan, preserving the original per-domain budgets and level configuration.
The inherited approach entries retain their existing experiment IDs and resume behavior.

```bash
python scripts/engaging/launch.py \
    -c predicatorv3/protocol_continual_minimal_knowledge_sweep.yaml \
    --partition mit_preemptable --accounts a,b
```
