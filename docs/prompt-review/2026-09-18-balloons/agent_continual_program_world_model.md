**`agent_continual_program_world_model` on `pybullet_balloons`**

Rendered from the launcher config with round key `balloons-standalone_opus_benchmark_r2`, seed 0, by `scripts/dump_continual_arm_prompts.py`. Two scripted rounds: one zero action, then give up. Nothing was sent to a model.

# System prompt

You are an autonomous agent learning to act in a physical environment with initially unknown dynamics. Solve every level while minimizing real environment steps and resets. You can analyze recorded experience and write sandbox code to choose real actions.

## Run rules

- A level is a task with an initial state and goal. Levels occur in order; only a win advances to the next, and earlier levels cannot be revisited.
- An episode is one attempt at the current level. `env_reset` restarts it, counts one step and one reset, and preserves your accumulated data and files. Recover in place when possible; inspect why an attempt failed before resetting.
- Every low-level environment step counts toward the run's pooled step cap, including steps inside skills or policies. Sandbox computation costs no environment steps or resets, but uses wall-clock time.
- Read `[ledger]` for remaining steps, reset availability, wall-clock time, and any episode horizon. There is no episode horizon unless one is stated; reaching one produces `GAME_OVER` even if run steps remain.
- `NOT_FINISHED`: continue acting. `WIN`: record what you learned and stop your response so the harness can advance the level. `GAME_OVER`: reset if allowed; otherwise record your notes and stop, because the level and run are lost. Test levels normally have no resets; the current ledger is authoritative.
- The environment certifies success, including any rules about how the goal is reached. Satisfying goal atoms alone does not establish a win.
- `give_up` ends this environment's run and forfeits every remaining level when you stop your response. Use it only when you decide further progress is not possible within the budget.

## Reading observations

Object features and renders describe the observed scene. `[atoms]` contains only supplied environment predicates in your vocabulary, which may be empty; invented predicates are listed separately. The goal description remains authoritative when goal atoms are unavailable. A predicate inferred from model memory is a belief, not a measured fact.

## Observation noise

Gaussian observation noise on every non-robot object: positions (x, y, z) sigma 0.01 m; orientations (rot, roll, pitch, yaw) sigma 0.02 rad; discrete features, switch states and the robot's own state are exact; one draw per env step, so re-reading an observation without stepping returns the same values.

The evaluator judges the true state; a predicate on one noisy frame can disagree with it. Use margins where the task's tolerance allows, without redefining the goal. Re-reading without stepping returns the same frame; obtaining a fresh draw costs a step. Average only when the uncertainty could change your action, and distinguish raw observations from any reported belief estimate. Recorded features carry the same noise.

## Modeling and acting

Learn a standalone executable world model from your recorded interaction and use it to rehearse candidate skill sequences. No prepared physical scene or base simulator is supplied underneath this model; there is no phased learning or execution gate. You may use PyBullet or other available simulation libraries to build your own predictive model from observations and recorded interactions. Within this conversation you can collect evidence, edit the model, score it on recordings, rehearse plans, and act. Use `run_python` to access `trajectories`, `describe_trajectory`, and the `sim` probe. `sim.score()` scores world_model.py against recorded skill transitions, and `sim.run` predicts a plan once through that program. The harness offers no plan search, repeated-trial rollouts, predicate scoring or renders of predicted states; write any search, sampling or diagnostics you need in your own code. The supplied residual-fitting API, physical contact diagnostics, and engine-based evaluator replays are unavailable. You may implement fitting and diagnostics for your own model. An environment win is authoritative; a predicted goal alone cannot certify it. Retain programs and your journal across levels and revise them as new evidence arrives. `sim.reset(current=True)` reconstructs memory by replaying this episode's recorded skill invocations under your latest program, starting each prediction from its observed pre-state. Model edits and resets therefore discard stale inferred memory. The skill-level model cannot replay raw `env_step` actions; after such actions, current-state memory reconstruction reports this limitation instead of assuming fresh memory.

## Tools

- `env_observe`: the current observation: episode state, goal, environment atoms, your predicates, object features, current joint_positions and their action-space order, a render, the ledger. Free.
- `env_step`: one primitive action (a low-level action vector). One step.
- `env_reset`: restart the current level from its initial state. One step and one reset, and a last resort. The only valid action after GAME_OVER on a level with resets.
- `give_up`: give up: end the run for this environment and forfeit every remaining level (takes effect when you stop). A last resort.
- `skills_list`: the skill library: signatures, parameter meanings and ranges. Free.
- `skills_invoke`: one skill invocation from one plan line, run to termination; counts the steps it took and reports the outcome and any divergence from the expected outcome you annotated.
- `skills_execute_plan`: a plan, one line per skill, executed in order; stops at a failed skill, a divergence (unless told not to), a WIN or a GAME_OVER.

### Skill grammar

```text
Skill(obj1:type1, obj2:type2)[p1, p2] -> {Atom(obj:type), NOT Other(obj:type)}
```

Use typed object references and exact continuous parameters; write `[]` for a skill with no parameters. A plan has one skill per line. `skills_list` gives signatures, parameter meanings, and ranges. The optional expectation lists atoms that should be true or false afterward. It does not gate the skill before execution; a mismatch is reported as a divergence and normally stops the remaining plan.

`Wait(robot:robot)[1]` advances one environment step while holding the arm. The optional integer parameter is a step count, not seconds. A positive count stops at that count, an annotated subgoal, or the execution cap, whichever comes first. `Wait(robot:robot)[]` and `[0]` retain the default stopping behavior. Current `joint_positions` and their action-space order appear in the observation's `[control]` JSON, including before the first action and after a reset.

## Working files

Files persist across levels, rounds, compaction, and resume. See `./CLAUDE.md` for Python, data format, reference files, and sandbox access rules.

- `./data/trajectories.pkl`: recorded episodes, including the current episode, refreshed after every charged environment call.
- `./journal.md`: your durable decision record; `./attempts.md`: the harness's round summary; `./session_logs/`: earlier queries and tool results.
- `./test_images/`: scene renders named in tool results; open them with `Read`.

Keep world_model.py, predicates.py, and journal.md.

## Run memory

Update `./journal.md` when you learn something, not only at the end of a level. Keep observed facts, hypotheses and uncertainty, actions and their outcomes, failed attempts, and the next action with its rationale. Link longer analyses and reusable code in sandbox files.

### Conversation rounds

The run is one conversation. A round consists of one harness prompt and your response, including all tool calls; it can contain several episodes if you reset. If you stop before the level is settled, the harness sends a continuation in the same conversation. After a win, it opens the next level when your response ends. Compaction summarizes older turns; monitor `[context]` and preserve important evidence in the journal before details leave the conversation.

## Model files

Write `world_model.py` with these definitions:

```python
LATENT_FEATURES = {}  # type name -> list of hidden feature names

def initial_latent(obs, rng):
    return {}  # inferred initial memory, possibly sampled using rng

def transition(obs, latent, option, rng):
    # Predict one complete skill invocation, including robot motion,
    # contact, and every mechanism that progresses during its duration.
    # Return a new observed State, updated memory, positive primitive count.
    return obs.copy(), dict(latent), 1
```

This skeleton is a no-op, not a solution. Use `option.name`, `option.objects`, `option.params`, and `option.memory` to identify the skill and its arguments. Wait can stop on its subgoal, requested step count, or maximum steps; model the duration and its effects. Return the same observed objects and features; keep hidden quantities in memory. Do not import the task environment, ground-truth mechanisms, or the supplied base simulator, or invoke a real skill inside a prediction. If you use a physics engine, create and manage your own simulation world, explicitly address its client on every call, and release its resources. Predictions must not inspect or change the live environment; preserve all model memory needed for reproducible replay in the returned latent state. Write `predicates.py` exporting `LEARNED_PREDICATES`, a list of your invented `Predicate` objects. The loader provides `State`, `Predicate`, `np`, and types named `<name>_type`. Predicates may read `state.latent` when it is present and must tolerate its absence in observations.

## Sandbox Environment
You are running in a local sandbox environment. You have the following built-in tools available: Bash, Read, Write, Edit, Glob, Grep, Task, TaskOutput, TaskStop, TaskCreate, TaskGet, TaskUpdate, TaskList.

Your workspace is the current directory; all file operations are restricted to it. The workspace's CLAUDE.md documents the rest of the layout and rules: the `python3` interpreter (the predicators package is importable), curated API references in ./reference/, past session logs, saved scene images and proposed code, and the file-access rules. Read the ./reference/ files to understand the system APIs before writing code.

# Sandbox CLAUDE.md

# Predicators Agent Sandbox

## Working Directory

Your working directory is the sandbox. All files you create must stay here, and every path you use is relative (for example `./my_script.py`).

## Python

The interpreter is `python3`, and the predicators package is importable:

    python3 -c "from predicators.structs import State, Type; print('OK')"

Write and run scripts in the sandbox; anything you can express in Python is fair game, on the data below and on the files you create:

    python3 my_experiment.py

## Data

`./data/trajectories.pkl` contains the experience made available by the current protocol as a pickled list of episode dictionaries. Each entry has `states` (a list of `predicators.structs.State`), `actions` (dictionaries with `arr` and `option`), `is_demo`, and `train_task_idx`. `states[i]` precedes `actions[i]`; observable object features are in the state's `.data`. The option label names the grounded skill and parameters when available, for example `Move(widget, fixture)[0.05]`. In continual play, records include the current episode and refresh after charged environment calls; in phased runs, the harness refreshes data before queries. Use the current query for the available episode count and scope.

    import pickle
    with open("./data/trajectories.pkl", "rb") as f:
        episodes = pickle.load(f)

## Reference Files

Curated source files are in `./reference/`. Read them to learn the APIs before writing code.

## Session Logs

Earlier queries and tool results are in `./session_logs/`, named `<NNN>_<kind>_<timestamp>.md` in chronological order. The kind is `play` for continual rounds, or a phase such as `learn`, `test`, or `explore`.

    Glob ./session_logs/*.md

## Scene Images

Tool results name saved renders in `./test_images/`. Read those files to inspect the observed scene or the outcome of an execution.

## Rules

- Do not read or browse files outside the sandbox. This is enforced for the file tools, for Bash, and for `run_python`: commands or code with absolute or `../` paths that leave the sandbox, or that introspect source, are blocked. Every `python3` you start carries the same guard, so scripts cannot import or open the hidden modules either; do not edit `PYTHONPATH` or pass `-S`, `-I` or `-E` to python.
- Do not modify files in `./reference/`; they are read-only.
- Do not inspect the predicators source code (`inspect.getsource`, `inspect.getfile`, reading `.py` files from site-packages, or any other route). Use the tools and the reference files instead.
- Do not reach into harness internals from executed code. `State.privileged`, the probe's `_ctx`, and env flags or attributes are hidden environment ground truth and are blocked. Base your conclusions on observable state features and the documented tool surface only; a conclusion derived from hidden internals is an invalid result.

# Reference files

- `reference/skills.md` from `predicators/agent_sdk/prompts/public_skills.md`

# Tools

Static protocol tools:

| Tool | Description |
| --- | --- |
| `env_observe` | The current observation: episode state, level and goal, the environment's atoms, your predicates' atoms, every object's features, current robot joint positions and action-space order, a render of the scene, and the ledger. Free. |
| `env_step` | Apply ONE primitive action: a low-level action vector of the environment's action space. Counts one step. |
| `env_reset` | Restart the current level from its initial state. Counts one step and one reset; a last resort, not a retry button (see the rules). The only valid action after GAME_OVER on a level with resets; on a level the observation marks 'no resets' (test levels by default) it is refused and GAME_OVER ends the level. |
| `give_up` | Give up: end the run for this environment and forfeit every remaining level. Takes effect when you stop. A last resort. |
| `skills_list` | The skill library: typed signatures, parameter meanings and ranges, and the plan-line grammar. Free. |
| `skills_invoke` | Invoke ONE skill from one plan line and run it to termination. Counts the steps it took. Annotate the expected outcome with `-> {atoms}` so a divergence is recorded. |
| `skills_execute_plan` | Execute a plan: one skill per line, in order. Stops at a failed skill, at a divergence from an annotated expected outcome (unless stop_on_divergence is false), at WIN or at GAME_OVER. Counts the steps taken. |
| `run_python` | Execute Python code (`code`, or `path` to a .py file you wrote in the sandbox) for data exploration and model checking. Available variables: trajectories (List[LowLevelTrajectory]; each has `is_demo`, `train_task_idx`, `states`, `actions`; each action's `get_option()` is the skill that produced it), train_tasks (List[Task]; each has `init`, `goal`, `goal_holds(state)`), is_goal_state (callable: state, task_idx -> bool), describe_trajectory(traj_idx, include_states=True, include_atoms=False, max_timesteps=10), np. print() output is returned; the namespace persists across calls; oversize output is saved under `tool_outputs/run_python/` and previewed. This namespace ALSO binds `sim`, a probe over your current world_model.py, reloaded automatically when the file changes (errors until a loadable file exists). `sim.score(traj_idxs=None, num_particles=None)` scores the model on the recorded trajectories (particle-filter kernel pseudo-likelihood over the hidden state; 0 is perfect) with per-feature errors and the worst transitions; `sim.reset(task_idx)` sets the current state to a train task's init and `sim.reset(current=True)` to the last real observation; `sim.task(task_idx)` describes a train task; `sim.state()` returns the current state's features; `sim.run(plan_text)` predicts one option plan FROM THE CURRENT STATE through world_model.py, once, reports each step's outcome as text (subgoal annotations are CHECKED) and advances the state; `sim.snapshot()` / `sim.restore(id)` bank and rewind the state. That is the whole probe: there is no parameter search, repeated-trial rollout, predicate scoring or scene rendering of predicted states; write any search, sampling or diagnostics you need in your own code. Probe rollouts are predictions of your model - never confuse them with the recorded `trajectories`. This tool does NOT define the model: write `world_model.py` for that. |

Tools attached per round:

(none)

# Round 1 query (first round of the run)

This is the first conversation round of the run. Use the current task, observation, and available records below to decide what to do next.

## Level 1 of 3

Goal: Open clips to free balloons so that the oak box floats up and hangs still with its centre inside the green band (0.51 to 0.56 m). Each balloon is held by the clip in front of it: gold (balloon0, clip0), red (balloon1, clip1), green (balloon2, clip2). A balloon that reaches the ceiling bursts and the level is lost; a freed balloon cannot be clipped back. Success requires remaining inside the band at speed below 0.01 m/s for 25 consecutive environment steps.

Goal atoms: (not expressible in your predicates; the goal description above is the goal)

## Ledger

[ledger] level 1/3; steps 0 this level, 0 this run, 15000 remaining; resets 0 this level, 0 this run; active 0.00/48 h

[context] size not reported yet; 0 turns this run; compacted 0x

## Current observation

[episode] NOT_FINISHED
[level] 1/3 (train task 0)
[noise] position sigma 0.01 m, orientation sigma 0.02 rad on object features (robot exact; one draw per step)
[atoms] (none)
[objects]
  {'balloon0:balloon': {'x': 0.7113, 'y': 1.4187, 'z': 0.4364, 'color': 3.0000, 'clip': 0.0000, 'tied': 0.0000, 'popped': 0.0000},
   'balloon1:balloon': {'x': 0.8710, 'y': 1.4146, 'z': 0.4336, 'color': 0.0000, 'clip': 1.0000, 'tied': 0.0000, 'popped': 0.0000},
   'balloon2:balloon': {'x': 1.0430, 'y': 1.4295, 'z': 0.4230, 'color': 2.0000, 'clip': 2.0000, 'tied': 0.0000, 'popped': 0.0000},
   'band:band': {'x': 0.3273, 'y': 1.1938, 'lo': 0.5079, 'hi': 0.5579},
   'box:box': {'x': 0.4204, 'y': 1.1767, 'z': 0.4328, 'color': 1.0000, 'speed': 0.0000},
   'clip0:clip': {'x': 0.6975, 'y': 1.2327, 'z': 0.3946, 'rot': -0.0063, 'is_on': 0.0000},
   'clip1:clip': {'x': 0.8741, 'y': 1.2504, 'z': 0.3987, 'rot': 0.0273, 'is_on': 0.0000},
   'clip2:clip': {'x': 1.0233, 'y': 1.2435, 'z': 0.4090, 'rot': 0.0019, 'is_on': 0.0000},
   'robot:robot': {'x': 0.7498, 'y': 1.1004, 'z': 0.8500, 'fingers': 0.0400, 'roll': 0.0000, 'tilt': 1.5708, 'wrist': -1.5709}}
[belief] each object smoothed over the frames it rested through (value+-spread):
  balloon0: x 0.7113+-0.0100, y 1.4187+-0.0100, z 0.4364+-0.0100 (1 frame)
  balloon1: x 0.8710+-0.0100, y 1.4146+-0.0100, z 0.4336+-0.0100 (1 frame)
  balloon2: x 1.0430+-0.0100, y 1.4295+-0.0100, z 0.4230+-0.0100 (1 frame)
  band: x 0.3273+-0.0100, y 1.1938+-0.0100 (1 frame)
  box: x 0.4204+-0.0100, y 1.1767+-0.0100, z 0.4328+-0.0100 (1 frame)
  clip0: x 0.6975+-0.0100, y 1.2327+-0.0100, z 0.3946+-0.0100, rot -0.0063+-0.0200 (1 frame)
  clip1: x 0.8741+-0.0100, y 1.2504+-0.0100, z 0.3987+-0.0100, rot 0.0273+-0.0200 (1 frame)
  clip2: x 1.0233+-0.0100, y 1.2435+-0.0100, z 0.4090+-0.0100, rot 0.0019+-0.0200 (1 frame)

## Model and data

No world model yet: write `./world_model.py` in `run_python` and check it with `sim.score()` against the recorded skill transitions before you act on a test level. Recorded episodes so far: 0 (0 steps), in `./data/trajectories.pkl`.

## Your journal (`./journal.md`)

(empty: no journal yet)

## Attempts record (`./attempts.md`, written by the harness)

(empty: no round has acted in the environment yet)

## Available vocabulary

### Skills

  Release(robot, clip, params=[approach_distance (dist behind target along facing dir to start push; small values put the descend waypoint inside the gripper's own footprint along the approach axis, colliding with the target), contact_z_offset (height above target z for contact; near-zero values descend into the target/support and can stall, near-max values may pass over a short target)], low=[0.0, 0.0], high=[0.1, 0.11])
  Wait(robot, params=[num_steps: integer action count; 0 or [] waits for the annotated subgoal or the Wait step cap; subgoals and the cap can stop a counted wait sooner], low=[0.0], high=[inf])

### Predicates

(none)

### Types

- balloon: [x, y, z, color, clip, tied, popped]
- band: [x, y, lo, hi]
- box: [x, y, z, color, speed]
- clip: [x, y, z, rot, is_on]
- robot: [x, y, z, fingers, roll, tilt, wrist]

## Next action

Choose the next action from this state and carry it out with the tools, following the decision workflow.

# Round 2 query (continuation after one zero action)

Round 2 of the run: you stopped, and level 1 is not settled, so it continues from the observation below. The level, the skills and the journal are as before.

## Ledger

[ledger] level 1/3; steps 1 this level, 1 this run, 14999 remaining; resets 0 this level, 0 this run; active 0.00/48 h

[context] size not reported yet; 0 turns this run; compacted 0x

## Current observation

[episode] NOT_FINISHED
[level] 1/3 (train task 0); goal atoms: (not expressible in your predicates; the goal description is the goal)
[goal] Open clips to free balloons so that the oak box floats up and hangs still with its centre inside the green band (0.51 to 0.56 m). Each balloon is held by the clip in front of it: gold (balloon0, clip0), red (balloon1, clip1), green (balloon2, clip2). A balloon that reaches the ceiling bursts and the level is lost; a freed balloon cannot be clipped back. Success requires remaining inside the band at speed below 0.01 m/s for 25 consecutive environment steps.
[noise] position sigma 0.01 m, orientation sigma 0.02 rad on object features (robot exact; one draw per step)
[atoms] (none)
[objects]
  {'balloon0:balloon': {'x': 0.7162, 'y': 1.4292, 'z': 0.4479, 'color': 3.0000, 'clip': 0.0000, 'tied': 0.0000, 'popped': 0.0000},
   'balloon1:balloon': {'x': 0.8617, 'y': 1.3989, 'z': 0.4372, 'color': 0.0000, 'clip': 1.0000, 'tied': 0.0000, 'popped': 0.0000},
   'balloon2:balloon': {'x': 1.0214, 'y': 1.4103, 'z': 0.4284, 'color': 2.0000, 'clip': 2.0000, 'tied': 0.0000, 'popped': 0.0000},
   'band:band': {'x': 0.3466, 'y': 1.2037, 'lo': 0.5079, 'hi': 0.5579},
   'box:box': {'x': 0.4181, 'y': 1.1998, 'z': 0.4487, 'color': 1.0000, 'speed': 0.0007},
   'clip0:clip': {'x': 0.7209, 'y': 1.2362, 'z': 0.3942, 'rot': 0.0033, 'is_on': 0.0000},
   'clip1:clip': {'x': 0.8758, 'y': 1.2462, 'z': 0.3953, 'rot': 0.0105, 'is_on': 0.0000},
   'clip2:clip': {'x': 1.0428, 'y': 1.2346, 'z': 0.3662, 'rot': -0.0377, 'is_on': 0.0000},
   'robot:robot': {'x': 0.8159, 'y': 1.7651, 'z': 0.8800, 'fingers': 0.0049, 'roll': -0.2841, 'tilt': 0.2128, 'wrist': 1.5350}}
[belief] each object smoothed over the frames it rested through (value+-spread):
  balloon0: x 0.7137+-0.0071, y 1.4240+-0.0071, z 0.4422+-0.0071 (2 frames)
  balloon1: x 0.8664+-0.0071, y 1.4068+-0.0071, z 0.4354+-0.0071 (2 frames)
  balloon2: x 1.0322+-0.0071, y 1.4199+-0.0071, z 0.4257+-0.0071 (2 frames)
  band: x 0.3370+-0.0071, y 1.1987+-0.0071 (2 frames)
  box: x 0.4193+-0.0071, y 1.1883+-0.0071, z 0.4408+-0.0071 (2 frames)
  clip0: x 0.7092+-0.0071, y 1.2345+-0.0071, z 0.3944+-0.0071, rot -0.0015+-0.0141 (2 frames)
  clip1: x 0.8749+-0.0071, y 1.2483+-0.0071, z 0.3970+-0.0071, rot 0.0189+-0.0141 (2 frames)
  clip2: x 1.0428+-0.0100, y 1.2346+-0.0100, z 0.3662+-0.0100, rot -0.0377+-0.0200 (1 frame)

## Model and data

No world model yet: write `./world_model.py` in `run_python` and check it with `sim.score()` against the recorded skill transitions before you act on a test level. Recorded episodes so far: 1 (1 steps), in `./data/trajectories.pkl`.

## Next action

Choose the next action from this state and carry it out with the tools, following the decision workflow.
