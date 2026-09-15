# Continual interaction driver

The MB and MF tool adapters submit typed execution requests to the same run-owned `InteractionExecutor`.
The driver applies real actions through `ProtocolSession`, which retains observation noise, execution belief, data refresh, recording, reset rules, and step and time budgets.
Model learning and simulated rollouts remain agent computation outside this executor.

## Control flow

The conversation chooses an execution request and awaits its outcome.
The driver executes that request, including any inner loop, and the tool adapter formats the result for the existing conversation.
The LLM-facing tool names, argument schemas, and prompt text are preserved.

| Agent request | Driver behavior |
|---|---|
| `PrimitiveAction` | Apply one validated action through the protocol. |
| `RequestReset` | Apply the protocol's existing reset eligibility and charges. |
| `ExecuteSkill` | Run one grounded skill and retain its observed outcome. |
| `ExecutePlan` | Run skills in order, stopping on failure, requested divergence, WIN, or GAME_OVER. |
| `ExecutePolicy` | Open the controller, supply each current observation, request an action, and step until the controller yields or a limit is reached. |
| `GiveUp` | End the run after the conversation has saved its notes and the approach has checkpointed. |

The policy loop now lives in the driver:

```python
async with request.open_controller() as get_action:
    for _ in range(request.max_steps):
        observation = session.observe()
        action = await get_action(observation)
        if action is None:
            break
        outcome = session.step(action)
        if outcome.state is not EpisodeState.NOT_FINISHED:
            break
```

The sandbox adapter owns loading and cleaning up the agent's Python controller and translating observations and action vectors.
The driver owns when that controller receives an observation, when its returned action is applied, and when physical execution stops.
Skills and generated policies can take multiple physics steps without another LLM turn.
The conversation, its journal, and its model workbench persist across requests as before.

This is an in-process request boundary.
It does not introduce an RPC service, a new thread, a serialized request queue, or a new conversation boundary.
The existing synchronous `ProtocolSession` methods remain available to reference controllers and preserve their contracts.

## Partial execution and stopping

Each request has an `ExecutionProgress` object that retains completed invocation results and actual protocol charges even if execution raises.
A step that reaches a cap remains charged and recorded before the exception propagates.
The adapter can therefore report completed plan actions or policy steps when a later action fails.
The existing data callback still runs at the original protocol operation boundaries, including after each primitive policy action.
Adapter observation and reporting hooks also run at their original points before and after each skill, so a predicate or reporting error prevents later plan actions from executing.

One executor belongs to the run's session, so tool adapters share its busy state.
A concurrent execution request is refused while a controller owns the environment.
Cancellation exits the controller context and clears that state while retaining already applied actions.
Internal computation, observation reads, and a policy returning `None` do not add environment steps.

The scorecard and recording formats and the existing replay and conversation-resume machinery are unchanged.
This refactor does not add an exactly-once execution guarantee across process crashes.

## Validation

Tests compare direct protocol execution with typed requests on a real cover environment, including successful completion, divergence, episode horizon, and a run cap reached during execution.
They compare observations, recorded action sequences, wins, steps, resets, and invocation counts.
Additional tests exercise controller cancellation, refusal of concurrent requests, cleanup, data notifications, and yielding without a real step.
The existing continual protocol, tool, MB, MF, and primitive-controller tests exercise the retained public behavior.

A separate compute-node comparison runs the same tool calls against an archived pre-refactor tree and this tree in bridge, fan, domino, boil, and original non-hatch balloons with observation noise, plus a cover task with a known solution.
It compares tool replies, observations, action histories, and scorecard metrics exactly, excluding only elapsed wall-clock text.
Its script and before/after artifacts are under `/home/ycliang/predicators/logs/interaction_driver_20260911/`.
These are deterministic interaction checks, not new agent solve-rate seeds.
The frozen noisy-sweep runtime and its experiment records are separate from this change.

Validated on September 11, 2026:

- Compute job `22575199`: all 64 functional tests passed, focused lint passed, and mypy reported no issues in the five changed source files.
- The same job applied the repository's pinned isort, yapf, and docformatter checks to the changed Python files.
- Compute job `22575200`: exact before/after equality for all six scripted environments on the current checkout's runtime.
- Compute job `22575264`: exact before/after equality for all six scripted environments on the frozen sweep runtime, with the refactor applied to a separate validation checkout.

The latter comparison reads the frozen source without changing its experiment configuration or records.
The validation checkout retains the sweep's original domain and agent code alongside the interaction refactor.
Neither comparison calls an LLM or supplies new solve-rate evidence.
The baseline test run also exposed an obsolete assertion requiring the MF prompt to mention a learned model; the test was corrected to retain the existing checks that MB-only tools and model descriptions are absent.
