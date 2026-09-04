# Continual protocol: one agent, one environment, a scorecard

A short overview for collaborators is `docs/continual-protocol-overview.md`.

Status: design settled 2026-09-04; step 1 of the build (the protocol core, no LLM) is implemented and verified on cover and on PyBullet boil the same day.
Decision: the paper's main table moves to this protocol.
The phased explore/learn/test loop in `predicators/run/online_learning.py` stays as a legacy protocol and is not relaunched for the main table.

## 1. Why

The phased loop makes the harness guess what the agent meant.
That guessing is where the last week's bugs came from: the `mental_model_solved` discount, `require_all_attempts`, `skip_redundant_test`, the certified-plan-versus-experiment prompt conflict, and the stale-results catch-up test.
In the continual protocol the harness does not judge intent.
The agent spends low-level steps in a real environment and the scorecard records what it spent and what it won.
How the recorded metrics are aggregated into a headline number is decided later, from the data, and is not part of the protocol.
Whether to explore, fit a model, invent predicates, or attempt the goal is the agent's decision.

## 2. Reference: ARC-AGI-3

We borrow the ARC-AGI-3 pipeline design (https://docs.arcprize.org/).
The facts we rely on, quoted from the docs:

- Action: "An action is a discrete interaction with the environment. Each turn where the agent submits a command, move, or input that affects the game state counts as an action." Tool calls and reasoning steps do not count.
- Actions are `RESET` plus a small fixed set of primitives; `RESET` "initializes or restarts the game or level state".
- States: `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`. After `GAME_OVER` "the only valid action is `RESET`".
- Scorecard fields: per-environment runs, action counts, completed levels and reset counts, `total_levels_completed`, `total_actions`, `total_environments_completed`.
- Competition mode: "only Level Resets are permitted", the agent may "only interact (call `make`) a single time for each environment", a single scorecard, and "scoring is against all available environments, even if you choose not to interact with them".
- Agent loop: `while not is_done() and action_counter <= MAX_ACTIONS: choose_action(); take_action(); append_frame()`, with `MAX_ACTIONS` as a guard.
- Recordings: one JSONL per play with `timestamp`, `game_id`, `state`, `levels_completed`, `action_input {id, data, reasoning}`, `available_actions`, `frame`.
- Swarm: one agent per game, run concurrently, scorecard lifecycle managed by the harness, replay links at the end.

We adopt the environment, level, primitive-action, reset, state, scorecard, recording, and swarm structure.
We do not adopt the human-baseline normalisation, the 1.15 cap, or the level-index weighting.
There is no human oracle for these envs, our levels are not ordered by difficulty, and the aggregation is deferred until the base metrics exist (section 4.4).

## 3. Mapping

| ARC-AGI-3 | This protocol |
|---|---|
| Game / environment | An env family with hidden dynamics: bridge, domino, fan, boil, busyboard |
| Level 1..N, sequential | A task in the env's task list: its random train tasks, then its test tasks, in that order |
| Play (one `RESET` to `WIN` or `GAME_OVER`) | An episode |
| `RESET` | `env.reset()`: restart the current level; counted |
| `ACTION1..7`, the primitives | One low-level env action, the protocol's only primitive |
| An agent's own macro over primitives | A skill: a parameterised closed-loop controller such as `Pick` or `Push` that runs `env.step` until it terminates with success or failure; agent-side, not part of the protocol |
| Frame (64x64 grid) | The observation digest: object features, observable atoms, render |
| `available_actions` | The low-level action space, and for skill-using agents the applicable skills with parameter ranges |
| `WIN` | The env's evaluator certifies the episode as a legitimate success |
| `GAME_OVER` | Horizon exhausted, env failure, a rejected episode, or an irrecoverable state |
| Human baseline (upper median actions) | None. Raw counts are recorded; the oracle is a reference arm, not a normaliser |
| Reasoning and tool calls are free | The sandbox is free: sim rollouts, fits, predicate invention, code |
| Scorecard | `scorecard.json` per run, aggregated across runs |
| Recording JSONL and replay viewer | Per-level recording plus the existing trajectory viewer |
| Swarm | One Slurm job per env and seed, plus an aggregator |
| Benchmarking harness (model configs x games, tags) | The launcher configs in `scripts/configs/predicatorv3/` |

## 4. Protocol definition

### 4.1 Environments and levels

Each env exposes an ordered list of levels.
A level is a task: an initial state, goal atoms, an NL goal, and the env's evaluator for that task.
The list is the env's random train tasks followed by its test tasks, in the order the env generates them.
There is no difficulty ordering and no weighting by position.
Levels are sequential.
The agent starts level k+1 only after winning level k.
The transition to the next level is free: the harness resets into the new level and returns its first observation without counting a reset.
There is no train/test split in this protocol.
Every level is recorded.
The "test tasks are eval-only" rule of the phased loop does not apply.

### 4.2 Actions

The protocol's only primitive is the low-level env action: one `env.step(action)` call, one env step.
The unit of account is the low-level step.
`env.reset()` is counted as one step and separately as one reset.

Skills are not part of the protocol.
We use the vocabulary of Kaelbling and Lozano-Pérez, "Rationally Engineering Rational Robots" (2025), section 1.4.2:

- A skill is one of "a set of low-level parameterized 'skills,' which are closed-loop sensorimotor policies (controllers) that will run for some time and then terminate, generally with some indication of success or failure."
- A skill invocation is one call to a skill controller with values for its continuous parameters.
- A plan "is a sequence of calls to skill controllers, including values for their continuous parameters."
- A skeleton is a "sequence of skill invocations" whose continuous parameters are still unbound; binding them is the agent's job, in the sandbox.
- A primitive action is one low-level step, the thing `env.step` takes.

`Pick`, `Place`, `Push`, `Wait`, `MoveTo` and the other parameterised options in the code are the skills.
They are an agent-side library that runs `env.step` until the controller terminates.
Our agent is offered the library and is encouraged to use it.
Some baseline arms are given only the primitive action space and no skills.
A skill invocation is recorded as a secondary count with the step range it covered and its termination status.

Nothing that happens inside the sandbox is a step: belief-model rollouts, system-identification fits, predicate and operator synthesis, code execution, and the agent's own reasoning are all free.
The only limits on free work are operational wall-clock caps so that runs finish (section 6.5).

### 4.3 Episode states and the win condition

- `NOT_FINISHED`: the level is in progress.
- `WIN`: the env's evaluator certifies the episode. The harness advances to the next level.
- `GAME_OVER`: the episode cannot continue. Only `env.reset()` is valid.

The win condition is the evaluator's verdict, not the goal atoms alone.
`BaseEnv.evaluate_episode` returns the task evaluator's reward and terminated flags, and a success is certified only when the episode also satisfied the task's legitimacy rule.
In domino the goal atoms can hold after an illegitimate topple, and the evaluator "rejects episodes where the robot toppled anything other than the green start block".
Such an episode is terminated but rejected: the goal atoms hold and no continuation can make it legitimate, so it is `GAME_OVER` with reason `rejected`.
The agent sees the state and the boolean, never the rule that fired, matching the current evaluator contract.
An env without an evaluator falls back to the goal atoms.

`GAME_OVER` is also raised by the horizon, by an `EnvironmentFailure`, and by the env declaring the state irrecoverable.
A failed skill (`OptionExecutionFailure`) is not `GAME_OVER`: its steps are counted and the episode continues from the resulting state.

### 4.4 Recorded metrics

The protocol records base metrics and defers aggregation.
Nothing below is normalised, capped, or weighted.

Per level, for one run:

- `won`: whether the level was won, and the step index at which it was won.
- `steps`: low-level env steps on the level, including reset steps.
- `resets`: agent-initiated resets on the level.
- `skill_invocations`: calls to skill controllers, with how many terminated in failure, for arms that use the library.
- `game_overs`: episodes that ended in `GAME_OVER`, with the reason for each (`horizon`, `env_failure`, `rejected`, `irrecoverable`).
- `divergences`: skill invocations whose observed outcome differed from the agent's annotated expected outcome.
- `wall_clock`: seconds on the level, split into env time and sandbox time.
- `sandbox`: sim rollouts, fits, learn sub-sessions, agent sessions, turns, and LLM cost in USD.
- `evaluation`: the env's reward and terminated verdicts for each episode.
- `steps_before_first_win` and `resets_before_first_win`, the two counts most likely to enter any later aggregate.
- `preemptions`, `resumes`, `downtime`, `harness_resets`, `interrupted_invocations`: the recovery bookkeeping of section 6.6, kept apart from the agent's own counts.

Per env run:

- `levels_completed` and `levels_total`.
- `total_steps`, `total_resets`, `total_skill_invocations`, `total_wall_clock`, `total_llm_cost`.
- `end_reason`: all levels won, step cap, wall-clock cap, agent ended the run, or crash.

The per-step detail lives in the recording (section 4.7), so any aggregate can be recomputed from disk.
Candidate aggregates to evaluate once data exists: levels completed, steps before the first win per level, the cumulative steps-versus-levels-won curve, and a reset-weighted cost.
The curve is the figure that shows skill acquisition over time.

### 4.5 Reference arms

There is no human baseline and the oracle is not a normaliser.
The oracle and the random-actions arm run under the same protocol and are recorded with the same metrics.
The oracle's step counts per level become a reference column in the table, alongside the agent arms.

### 4.6 Constraints (the competition-mode analogue)

- One run per env and seed. The env is instantiated once and the agent plays its levels in order.
- Level resets only. There is no way to return to a previous level.
- No skipping, for now. A level that is never won ends the run for that env, and later levels are recorded as not attempted.
- The run ends when the last level is won, when the step cap is hit, when the agent ends it, or when the wall-clock cap expires.
- The table covers all envs in the suite, whether or not the agent was run on them.
- The step cap is a pooled per-run allowance set per env from the data in section 4.8. It is a guard against runaway runs, not a scoring term.

### 4.7 Scorecard and recordings

The harness writes `scorecard.json` after every skill invocation and every reset, and at least every N steps for primitive-only arms, so a crashed run still has a valid partial scorecard.
The file holds the per-level and per-run metrics of section 4.4 and the run metadata (env, seed, arm, git SHA, config).
Recordings are one directory per level at `recordings/<run_id>/<env>-L<k>/`: the low-level trajectory (states and actions, the same `LowLevelTrajectory` shape that `data/trajectories.pkl` uses) written incrementally, plus a JSONL index with one line per skill invocation or reset: timestamp, step range, skill and parameter values, termination status, the agent's expected outcome and reasoning note, the observed atoms afterwards, the episode state, and the render path.
The existing trajectory viewer reads the index and pairs each entry with its render.
A repository-level aggregator turns a set of scorecards into tables and curves once the aggregation is chosen.

### 4.8 Empirical step counts and the cap

Low-level step counts from the existing runs, gathered on 2026-09-04 from `results/*.pkl` (test episodes that were solved) and from the sandbox `data/trajectories.pkl` of the tier-1 C1 runs (explore episodes).

| env | horizon | oracle solve, median (range) | agent solve, median (range) | explore episode range | explore steps before the first solve |
|---|---|---|---|---|---|
| boil | 500 | 174 | 237 (233-280) | 16-420 | 829 (2 episodes) |
| fan | 500 | no oracle result | 192 (88-378 across arms) | 367-500 | 867 (2 episodes) |
| domino | 500 | no oracle result | 210 (177-500) | 150-300 | 411-725 (2 to 4 episodes) |
| busyboard | 2000 | 109 (106-136) | 188 (166-188) | 175-500 | 1000 (2 episodes) |
| bridge | 3000 | 682 (581-736) | 812 (606-1577) | 157-2000 | seed 0: 2754 (2 episodes); seed 3: not solved after 7117 (6 episodes) |

For comparison, the phased loop allowed 5 cycles of 2 explore episodes at 500 steps each, 5000 explore steps per train task (bridge: 2000 per episode, 20000), plus up to six test episodes at the horizon.
The agents that solved used between 400 and 2800 explore steps, and bridge seed 3 is the only run to exhaust more than 7000 without a win.

Proposed cap, pooled over the run so that early learning is amortised across levels:

| env | allowance per level | run cap with 5 levels |
|---|---|---|
| boil, fan, domino | 5000 | 25000 |
| busyboard | 5000 | 25000 |
| bridge | 10000 | 50000 |

This is the phased loop's per-task explore budget for the 500-horizon envs, about twenty solve-lengths per level for the small envs and twelve for bridge, and above the steps bridge seed 3 has consumed without a win.
The values are launcher constants and are the first thing to revisit once the first continual runs exist.

## 5. The agent

### 5.1 Tool surface

The sandbox and its tools carry over unchanged: `run_python` with the belief probe, `sim.fit`, `sim.predicates()`, `sim.samplers()`, `sim.refine`, `sim.run`, the journal, and `data/trajectories.pkl`.
The phase-specific tools `submit_plan`, `submit_policy`, and the capture and clearance gates are replaced by two namespaces.

The protocol namespace, offered to every arm:

- `env.observe()`: the current observation. Free.
- `env.step(action)`: one low-level action. One step.
- `env.reset(note=None)`: restart the current level. One step plus one reset.
- `env.end_run(note)`: end the run for this env.

The skill library, offered to our agent and to the skill-using baselines.
The names follow the paper's usage: a skill controller is invoked, a plan is executed by invoking its skills in order, and a primitive action is stepped.

- `skills.list()`: the skills, with their parameter spaces and which are applicable now.
- `skills.invoke(skill, params, expected=None, note=None)`: one skill invocation. Runs the controller to termination and returns its indication of success or failure, the steps it took, and the observation. `expected` is the agent's expected outcome as atoms; the harness reports the divergence but does not block.
- `skills.execute_plan(plan, stop_on_divergence=True)`: execute a plan, a sequence of skill invocations with parameter values, by invoking each skill in order. Stops on a failed invocation, on a divergence from an expected outcome, on `WIN`, or on `GAME_OVER`, and returns the per-invocation results.
- `skills.run_policy(policy)`: step a closed-loop policy until it ends or the episode does. Skill invocations are detected from the option each action carries, so a policy built from a plan is recorded exactly as the same plan sent through `execute_plan`. This is how the oracle arm and a submitted policy execute.

A skeleton with unbound parameters is never sent to the env.
The agent binds it in the sandbox, with `sim.refine` or its own code, and executes the resulting plan.

Learning, offered to the learning arms:

- `learn.run(kind)`: launch a learning sub-session of an existing kind (simulator synthesis, predicate invention, sampler synthesis) in-process and return its summary. Free.

Every tool result carries the ledger footer: level, steps and resets on this level and in the run, and the remaining step cap.
The footer is the pacing signal, in the same spirit as the current `[budget]` footer.

### 5.2 Observation

The observation is the analogue of ARC's frame:

- `state`: `NOT_FINISHED`, `WIN`, or `GAME_OVER`, and the reason for `GAME_OVER`.
- `level`: index, goal atoms, NL goal, the evaluator's objective description, levels completed, total levels.
- `frame`: the object feature table and the env's observable atoms, plus a render path.
- `action_space`: the low-level action vector's dimension, bounds, and semantics.
- `skills`: for skill-using arms, the applicable skills with their parameter spaces.
- `evaluation`: reward and terminated from `evaluate_episode`.
- `ledger`: as in the footer.

The atoms are evaluated under the arm's own predicate vocabulary: an arm that hides env predicates from its agent (C1 keeps `Holding` and invents the rest) sees only its kept and invented predicates, and goal atoms it cannot express are replaced by the goal description.
The env's full atom set is recorded in the level index for analysis, never shown to such an arm.
Kept env predicates and invented predicates are listed separately.
An invented predicate that reads latent state is always false on a real observation, and the separate listing keeps that visible instead of letting it masquerade as env truth.

### 5.3 Loop and context management

Continuous does not mean one LLM context.
The run is a sequence of SDK sessions on one agent-owned loop.
The default session unit is one level: a session starts at the level's first observation from the journal and the scorecard, and ends when the level is won, when the agent ends it, or when the session turn cap is hit, in which case the next session resumes the same level from the journal.
Within a session the SDK's own context compaction applies, as it does for any long session.
Nothing the harness does between sessions changes the env state.
This is the journal handoff that already exists, with the harness no longer choosing the session kind.

Knowledge carries across levels through the sandbox: `predicates.py`, the fitted simulator, the journal, and the trajectory data all persist for the whole env run.
Levels are the same env family, so a learned model transfers directly and only the grounding changes.

### 5.4 Prompt

One system prompt states the rules: what a step is, that a reset is a step and is counted separately, that the sandbox is free, what is recorded, and the loop.
One per-session query carries the observation, the ledger, and the journal.
The ARC template's objective sentence is the model: "Your objective is to WIN and avoid GAME_OVER while minimizing actions."
The agent decides when a belief-model rollout is worth more than a real step.
The prompt gives no schedule and no certification rule.

## 6. Implementation plan

### 6.1 New modules

- `predicators/run/continual.py`: the protocol loop. Owns the level list, the episode runner, the scorecard, the recordings, checkpoints, and the run-end conditions.
- `predicators/run/scorecard.py`: the `Level`, `EnvCard`, and `Scorecard` dataclasses holding the section 4.4 metrics, with incremental JSON writes. No formulas.
- `predicators/run/episode.py`: an `EpisodeRunner` with `reset(task)`, `step(action)`, `run_option(option)`, `observe()`, and `evaluate()`. It is factored out of `run/testing.py::_execute_task` and `run/online_learning.py::generate_interaction_results`, and both protocols use it.
- `predicators/agent_sdk/tools/env_tools.py`: the `env.*` namespace, the observation formatter, the ledger footer, and the recording writer.
- `predicators/agent_sdk/tools/skill_tools.py`: the `skills.*` namespace over the existing option grounding, `option_plan_to_policy`, and the divergence check on expected outcomes. In code the skills stay `_Option` objects; the agent-facing name is skill.
- `predicators/agent_sdk/prompts/play_system.md` and `play_query.md`: the continual prompts, with golden renders like the existing templates.
- `predicators/approaches/agent_continual_approach.py`: `AgentSessionMixin` plus the tool context, the session loop, session resume, and `learn.run` as a sub-session launcher over the existing synthesis code.
- `scripts/aggregate_scorecards.py`: scorecards to tables and curves, parameterised by the aggregation chosen later.
- `scripts/configs/predicatorv3/protocol_continual.yaml`: the launcher, one job per env and seed and arm.

### 6.2 Entry point

`run_pipeline` in `predicators/run/online_learning.py` dispatches on a new `experiment_protocol` flag: `phased` (today's loop, unchanged) or `continual`.
The continual branch does not construct interaction requests, an explorer, early stopping, or a test round.

### 6.3 Reused unchanged

Sandbox setup and the pyguard, session managers, the journal, `budget.py`'s watchdog, plan parsing and grounding, `option_plan_to_policy`, the subgoal annotation parser and divergence detection, `export_trajectories`, the learn sessions and their prompts, predicate synthesis, and the viewer.

### 6.4 Retired for this protocol

Explorers, `InteractionRequest` and `InteractionResult`, `mental_model_solved`, early stopping and its flags, the certified-plan replay, the capture and clearance gates, the test-phase sandbox rollback, and the train/test split semantics.
They remain in the tree for the phased protocol.

### 6.5 Limits

Today's phased loop has four limits on interaction: explore episodes stop at `max_num_steps_interaction_request` (500, bridge 2000), test episodes stop at the env horizon (500, busyboard 2000, bridge 3000), each skill stops at `max_num_steps_option_rollout`, and a solve attempt has a wall clock (`agent_solve_attempt_wall_clock`, 2700 s, one attempt).

In the continual protocol there is no per-task limit.
The limits are:

- The env horizon per episode. Exhausting it is `GAME_OVER`, and the agent resets and continues.
- The pooled step cap per run (section 4.8).
- A wall-clock cap per env run, with requeue: 48 h proposed.
- The per-session turn cap and the existing per-call timeouts. The sysid fit budget stays as it is; a fit is not a step but it is bounded in time.

With no skipping, a level the agent cannot win consumes the remaining pool, which is the same outcome a per-level cap would produce.

### 6.6 Checkpoints, preemption, and chat resume

The recording is the env checkpoint.
The harness records the low-level action sequence of the current episode since its last reset.
On requeue it reconstructs the episode by re-instantiating the env with the same seed and replaying that sequence.
PyBullet stepping is expected to be deterministic for an identical low-level sequence, and this must be verified per env before the protocol relies on it.
If the replay diverges from the recorded observations, the harness counts one reset and restarts the level, and marks the scorecard.
The sandbox directory and `scorecard.json` are already durable.

The chat context can be resumed too.
The installed `claude_agent_sdk` (0.2.128) exposes `resume`, `continue_conversation`, `fork_session`, and `session_store` on `ClaudeAgentOptions`, and the CLI stores every session transcript under `~/.claude/projects/<cwd-slug>/` on the shared NFS home, where 358 sandbox session directories from past runs already sit.
None of this is wired today: `agent_sdk_resume_session` in `settings.py` is read nowhere, the `session_id` setter on the session manager is never called, and `save_session_info` is never invoked.
Wiring it means capturing the session id from the SDK's init message, persisting it with the level checkpoint, and passing `resume=` on the relaunch with the sandbox at the same path, spelled the same way.
The slug is derived from `cwd`, and the projects directory already shows both the `/home/ycliang` and the `/orcd/home/002/ycliang` spellings of this repo, so a requeue that spells the path differently would start a fresh transcript.
Resume is the mid-level recovery.
Level boundaries still start a fresh session from the journal by design, and the journal remains the fallback when a transcript is missing or the resume fails.

What a resume preserves, and what it may lose:

- Preserved: every completed turn and tool result up to the preemption. The CLI appends the transcript per message, so the restored context is the conversation as it was, compactions included.
- Preserved: the env episode up to the last recorded step, and the scorecard up to its last write.
- Lost, and accepted: the turn in flight, meaning its partial reasoning and any tool call whose result never arrived, plus the env steps executed after the last recording flush.

The harness keeps the two sides consistent.
It restores the env to the last recorded step, counts only recorded steps, and opens the resumed session with a message that states the restore point, the ledger, and that the interrupted call did not complete.
The recording appends the low-level action of every step as it is taken, so the env side loses nothing: the action log is a few floats per step and the states are replayed from it, with full states written only at skill boundaries.
The only loss is therefore the LLM turn in flight.

Preemption must not move the metrics.
The rules that keep the recorded counts and the agent's performance independent of when and how often a job is requeued:

- Steps and resets are only ever counted for actions the agent took. A restart the harness performs because a replay diverged is recorded as a `harness_reset`, not as an agent reset, and the steps of the abandoned episode stay attributed to the level with a `harness_reset` marker so an aggregate can include or exclude them.
- Wall-clock metrics count active time only. Queue time between preemption and resume is recorded as `downtime` and never enters `wall_clock`, in the same way the phased loop pauses the attempt clock for refused queries.
- Per level the scorecard records `preemptions`, `resumes`, `downtime`, `harness_resets`, and `interrupted_invocations`, so any run can be checked for a preemption effect after the fact, and preempted and unpreempted runs can be compared.
- Long sandbox work publishes on completion. A fit or a synthesis that finished before the kill is on disk in the sandbox and the resumed agent finds it; one that was in progress is lost and is redone, which costs wall-clock only, never steps.
- A resumed session re-reads its transcript, which costs prompt tokens. That lands in `llm_cost` and is explained by `resumes`; it is not a key metric.
- Replay is verified, never assumed. After replaying the action log the harness compares the reconstructed state with the last recorded state feature by feature, within a tolerance, and only then hands the env back to the agent. A silent divergence would corrupt the agent's picture of the world and is the one failure that could hurt performance rather than bookkeeping.

The check that this holds is part of build step 1: run the oracle and random controllers with forced kills at random points and requeue, and diff the scorecards against unpreempted runs of the same seed.
Steps, resets, and levels won must be identical, and active wall-clock within tolerance.
For the LLM agent the same forced-kill run on boil gives an estimate of the effect on its counts, which cannot be exact because the agent is not deterministic.
Jobs on this cluster are requeued every 12 h by the wall-time trap in any case, so a bridge level will see several resumes and this test is not optional.

### 6.7 Baseline arms

Every arm in the paper is an agent playing the same env API, so the comparison is apples to apples:

- Oracle: a reference arm with the ground-truth model, recorded like every other arm.
- Random low-level actions: the lower reference.
- Low-level-only agents: the LLM agent with `env.*` only, no skill library.
- Full agent: the skill library and the sandbox with simulator learning, predicate invention, and samplers.
- Model-free agent: the skill library, the env tools, and `run_python` over the data only, no belief model.
- Tool and prompt ablations: the same agent with a tool removed or a prompt section removed.
- Fixed-schedule agents: scripted controllers that explore for K episodes, learn once, then attempt, expressed with the same tools. These are the phased-loop baselines re-expressed.

The current arm list (C1 to C8, U1, A1 to A8) is re-mapped onto these categories once the protocol is fixed.
Results from the phased runs are not reused in the main table.

## 7. Open decisions

Settled on 2026-09-04: record base metrics only, no score cap, no oracle normalisation, aggregation deferred; levels are random train tasks then test tasks with no ordering; the protocol primitive is the low-level action and skills are agent-side; the win is the evaluator's certified success; no skipping for now.

1. Step cap: the per-env allowances in section 4.8, pooled over the run.
2. Preemption: replay-restore for the env plus SDK session resume for the chat, with a counted reset and a journal restart as the fallbacks.
3. Session unit: one SDK session per level by default.
4. Wall-clock cap per env run: 48 h.

## 8. Build order

1. Protocol core with no LLM: `scorecard.py`, `episode.py`, `env_tools.py`, `skill_tools.py`, `continual.py`, plus the random and oracle controllers. This validates the metrics, the recordings, and the preemption restore end to end.
2. The agent: the play prompts, `agent_continual_approach.py`, `learn.run`, session resume. First runs on boil and fan.
3. Level lists for all five envs.
4. The remaining arms, the launcher, the aggregator, and the viewer changes.

## 9. Implementation status and how to run

Step 1 of the build order landed on 2026-09-04 in `predicators/run/`:

- `scorecard.py`: `RunCard`, `LevelCard`, `EpisodeRecord` and the atomic JSON writes.
- `episode.py`: `EpisodeRunner`, the primitive step, the win and game-over classification, `run_option` for one skill invocation.
- `recording.py`: `LevelRecording`, the per-step action log, the index, the episodes pickle, the checkpoint and the renders.
- `continual.py`: `ContinualRun` (the loop, the caps, the resume with replay verification) and `ProtocolSession` (the API of section 5.1, with `run_policy`).
- `controllers.py`: the oracle, random-skills and random-primitives controllers.
- `run_pipeline` dispatches on `experiment_protocol`; the phased loop is unchanged.

Tests: `tests/run/test_continual.py` pins the counts, the recordings, the preemption resume (lossless replay and the harness-reset fallback), the protocol errors and the divergence check on the cover env.
`tests/test_continual_viewer.py` renders the viewer pages for a real run.

Launching:

```bash
python scripts/engaging/launch.py -c predicatorv3/protocol_continual.yaml --partition mit_preemptable
```

The launcher passes `--auto_resume`, so a requeue resumes from the scorecard and the level recording.
Outputs land in `scorecards/<run_id>.json` and `recordings/<run_id>/L<k>/`.

Viewing:

```bash
python scripts/continual_viewer.py --port 25152
```

The index lists every run in one table per agent and env pair, nested under agent-name or env-name headers with a toggle and a filter, as in the phased log viewer.
A run page is a left menu plus a content pane filled from the hash route: the overview (metadata, the cumulative steps-versus-levels-won curve, the per-level metrics of section 4.4), each level's replay and event list, the agent's sessions, and a tree of the recording's files (the system prompt, the transcripts, the sandbox's journal, attempts, data and images, each level's index, actions and renders).
The replay follows the ARC-AGI-3 replay viewer: one frame per recorded event with the render, the action, the agent's thinking and text that led to it (paired from the session transcripts), the tool call and its result, and the atoms that changed, with keyboard playback and a JSON export, and it plays inside the run page.
A session renders as a conversation with thinking, assistant text, tool calls and results, and renders inline.

First PyBullet result, the oracle arm on boil (train task then test task), job 21960691: both levels won, 217 steps each, 9 skill invocations each, no resets, 33 s active.

Step 2, the agent arm, landed the same day:

- `predicators/approaches/agent_continual_approach.py`: `AgentContinualApproach`, C1's learner (hybrid simulator, parameter fit, predicate invention) playing through play sessions. It implements `play_level`: each session is a fresh context over the journal; after a session the arm services what the session asked for (a learning session, the run's end), records the session in `attempts.md`, rebuilds the learning data from the recorded episodes, and checkpoints.
- `predicators/agent_sdk/tools/continual_tools.py`: the play tools `env_observe`, `env_step`, `env_reset`, `env_end_run`, `skills_list`, `skills_invoke`, `skills_execute_plan`, `learn_run`, `session_end`, thin text adapters over `ProtocolSession`. Every result ends with the ledger line. Ending the session and running a learning session cannot happen inside a running SDK session, so `learn_run`, `env_end_run` and `session_end` record requests the arm acts on after the query returns.
- `predicators/agent_sdk/prompts/play_system.md` and `play_query.md`, rendered by `agent_sdk/play_prompts.py`: the rules of section 4, the tools, the skill grammar, the sandbox, learning, the journal protocol and the session protocol; the query carries the level, the ledger, the observation with a render, the skills, the predicates, the learning status, the journal, the attempts record and the handoff note.
- The agent's sandbox and CLI transcripts live in a stable directory per run, `recordings/<run_id>/agent/`, not per launch, so a requeue finds them. SDK session resume is wired: the session manager records the CLI session id into `session_info.json` as soon as a session opens, and an in-flight session at checkpoint time is reopened with `resume` on the next launch (section 6.6).
- The observation the agent gets is both numbers and pixels: the object feature table and the atoms in text, and a render saved into the sandbox's `test_images/` at every observation, invocation, plan and reset, named in the tool result and readable with `Read`.

Tests: `tests/agent_sdk/test_continual_tools.py` drives the tools over a real session on cover (a win through `skills_execute_plan`, divergences on positive and `NOT` expectations, parse errors, game over then reset, queued learning and run end, the cap hit inside a tool); `tests/approaches/test_agent_continual_approach.py` runs the play loop on boil with a scripted agent in place of the LLM (a session that acts and queues learning, the learning service, the attempts record, the checkpoint, the resume of an in-flight session id, the idle guard).

Launching the agent arm: un-skip `agent_continual` in `protocol_continual.yaml`.

First agent result, boil seed 0, job 21964274 (2026-09-04): both levels won, 2127 steps and 5 resets on level 1 (one session, 129 turns, 85 skill invocations, 44 failed, one horizon game over), 374 steps and no reset on level 2, 55 min active, $24.66.
The agent asked for no learning session and ran no model rollout: it measured the dynamics by probing the real environment, wrote a recipe into its journal, and replayed it on level 2.
Under the current prices (a reset is one step, observation is free, a failed skill costs its steps) real probing was the cheaper policy; the cost pressure is open question 2 of the overview.

Next: the remaining arms (model-free, primitive-only, fixed-schedule controllers), the level lists for all five envs, and the aggregation choice.
