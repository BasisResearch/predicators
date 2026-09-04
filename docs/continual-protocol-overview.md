# The continual protocol: an overview for collaborators

Draft of 2026-09-04.
This is the short version.
The full design, with the reasoning behind each choice, is `docs/continual-protocol.md`.

## In one paragraph

The paper's main table moves from the phased explore, learn, test loop to a continual protocol modelled on ARC-AGI-3.
An agent is dropped into one environment and plays its tasks in order, as levels, in one continuous run.
The protocol charges for exactly one thing: a low-level environment step.
A reset is charged `continual_reset_cost` steps (1000 by default) and is counted separately.
Everything the agent does off-line is free: rollouts in its learned model, parameter fits, predicate and operator synthesis, code, and its own reasoning.
Each run produces a scorecard of base metrics: levels won, steps and resets per level, and the steps before the first win.
We record everything and choose how to aggregate later.

## Why change

In the phased loop the schedule was ours: fixed explore episodes, then a learning phase, then test episodes, with fixed budgets.
The agent never chose when to gather data, when to learn, and when to act.
In the continual protocol the agent decides.
Data is costly, resets are costly, and what it learns carries across tasks.
An agent that learns a model and tests its hypotheses in the model should spend fewer real steps than one that probes the real environment, and the scorecard makes that visible.

The reference is ARC-AGI-3 (docs.arcprize.org): environments with levels, one primitive action, `RESET` counted, episode states `NOT_FINISHED`, `WIN` and `GAME_OVER`, a scorecard and a recording per run.
We do not adopt its score formula, its cap, or its human baseline.

## The protocol

Levels.
An environment's levels are its random train tasks followed by its test tasks, in the order the environment generates them.
There is no difficulty ordering and no weighting by position.
Levels are sequential: the next level starts after a win, and the transition is free.
There is no train and test split any more; every level is recorded.

Actions.
The unit of account is one `env.step` call.
A reset is charged `continual_reset_cost` steps (1000 by default, so a reset is a last resort) and counted as one reset.
Skills (`Pick`, `Place`, `Push`, `Wait`, `MoveTo` and the rest of the option library) are not part of the protocol.
They are an agent-side library that runs `env.step` until the controller terminates, and their steps are charged like any other.
Our agent is offered the library; some baselines get only the primitive action space.
The vocabulary follows Kaelbling and Lozano-Pérez (2025): a skill is a parameterised closed-loop controller, a plan is a sequence of skill invocations with parameter values, a skeleton is the same with the parameters unbound.

Winning and losing.
`WIN` is the environment evaluator's certified success, not the goal atoms alone.
In domino, for example, the goal atoms can hold after an illegitimate topple, and the evaluator rejects the episode.
`GAME_OVER` comes from the horizon, an environment failure, a rejected episode, or an irrecoverable state, and only a reset is valid after it on a level with resets.
Test levels have no resets by default (`continual_allow_test_resets`), so a `GAME_OVER` there loses the level and ends the run.
A failed skill is not `GAME_OVER`; its steps count and the episode continues.

Caps.
A run has a pooled step allowance so that early learning is amortised across levels: 5000 steps per level for boil, fan, domino and busyboard, 10000 for bridge, plus a 48 hour wall-clock cap.
These are guards against runaway runs, not scoring terms.
No skipping: a level that is never won ends the run, and later levels are recorded as not attempted.

## What is recorded

Per level: whether it was won and at which step, steps, resets, skill invocations and how many failed, game overs with their reasons, divergences between the agent's annotated expectations and what happened, wall clock split into environment and sandbox time, sandbox counts (rollouts, fits, sessions, turns, LLM cost), and the steps and resets before the first win.
Recovery bookkeeping (preemptions, resumes, downtime, harness resets) is kept apart from the agent's own counts.
Per run: levels completed, the totals, and the end reason (all levels won, step cap, wall-clock cap, the agent ended it, or a crash).

On disk, one scorecard per run at `scorecards/<run_id>.json`, rewritten after every skill invocation and reset.
One recording per level at `recordings/<run_id>/L<k>/`: every primitive step (`actions.jsonl`), an index with one line per skill invocation, reset, resume, win and game over (`index.jsonl`), the episodes, a checkpoint, and a render per event.
The agent's own material lives in `recordings/<run_id>/agent/`: the system prompt, one transcript per session, and its sandbox (journal, attempts record, data, images).
Any aggregate can be recomputed from these files.
The figure we expect to show is the cumulative steps versus levels won curve.

## The arms

- `oracle`: the ground-truth planner under the same protocol; the reference column, not a normaliser.
- `random_primitives`: uniformly random low-level actions.
- `random_skills`: random skill invocations with random parameters.
- `agent_continual`: our agent, using the C1 learner (hybrid simulator synthesis, parameter fit, predicate invention).
  It plays through sessions, each a fresh context over its journal, with tools `env_observe`, `env_step`, `env_reset`, `env_end_run`, `skills_list`, `skills_invoke`, `skills_execute_plan`, `learn_run`, `session_end` and `run_python`.
  Every observation carries the object features, the atoms in its predicate vocabulary, and a render it can look at.
  A learning session, requested with `learn_run`, fits the belief model over every recorded episode and is free in steps.
- `agent_continual_model_free`: the model-free baseline, the same agent with the env and skill tools, the sandbox and the journal only: no belief model, no `sim`, no `run_python` and no learning session.
  Its own code in the sandbox reads the recorded data; what it cannot read off the data it learns from the environment at the price of steps.
- Planned: a primitive-only agent and a fixed-schedule controller that reproduces the phased loop inside the protocol.

Preemption is handled: a requeued job replays the recorded actions to verify the environment state, reopens the agent's chat session by id, and books any lost progress separately from the agent's counts.

## First results

Boil, seed 0, two levels (one train task, then one test task), 2026-09-04.

| arm | levels won | steps per level | resets | active time | LLM cost |
|---|---|---|---|---|---|
| oracle | 2 / 2 | 217, 217 | 0 | 33 s | 0 |
| random primitives | 0 / 2 | 10000 (cap) | 19 | 2 min | 0 |
| random skills | 0 / 2 | 10000 (cap) | 19 | 10 min | 0 |
| agent | 2 / 2 | 2127, 374 | 5 | 55 min | $24.66 |

The agent won level 1 after 2127 steps and 5 resets, in one session of 129 turns and 85 skill invocations.
It never asked for a learning session and ran no rollout in a model.
Instead it probed the real environment: it measured the fill rate, found the volume cap, found that any spill is fatal, and wrote an exact recipe into its journal.
On level 2 it read the recorded data, confirmed the same recipe applied, and replayed it in 374 steps with no reset.
That is a legitimate strategy under the protocol, and the numbers show its cost.
Whether the incentives should push harder towards learning a model is the first open question below.

## How to run and view

Launch (un-skip the arms you want in the yaml; each job requeues and resumes itself):

```bash
PYTHONPATH=. python scripts/engaging/launch.py -c predicatorv3/protocol_continual.yaml --partition mit_preemptable
```

View:

```bash
python scripts/continual_viewer.py --port 25152
```

The index at `http://127.0.0.1:25152/` lists every run, grouped by agent or by environment, with a filter.
Each table is one experiment id and each row one run, named by start stamp and seed, with buttons to copy the recording path, pause the run (cancel its job; a relaunch with `--auto_resume` continues it) and delete it.
A run page has a left menu: overview, the run's replay, each level's entry into it and its event list, the agent's sessions, and every file of the recording (the system prompt, the transcripts, the sandbox's journal and data, each level's index and renders).
The replay plays the whole run as one continuous sequence, one frame per recorded event: the render, the action, the agent's reasoning that led to it, the tool call and its result, and the atoms that changed, with keyboard control.
Levels are bands above the slider and resets, resumes, wins and game overs are ticks below it; clicking either jumps there.

Aggregate a set of scorecards into `runs.csv`, `levels.csv` and `summary.md`:

```bash
python scripts/aggregate_scorecards.py
```

## Open questions

1. Aggregation: which number goes in the table (levels won, steps before the first win, the curve, a reset-weighted cost).
2. Cost pressure: in the first runs a reset cost one step, observation is free, and a failed skill costs only its steps, so on boil probing the real environment was cheaper than learning a model.
   The reset price was raised to 1000 steps on 2026-09-04 (`continual_reset_cost`); tighter caps derived from the oracle's step counts remain an option.
3. Level lists: every environment currently has one train and one test level; the paper needs the full lists.
4. The remaining arms and the four other environments.
5. The transcripts carry the agent's text but not its thinking blocks; the viewer shows thinking when the SDK returns it.
