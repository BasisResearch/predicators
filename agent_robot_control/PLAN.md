# Agent-Harness Robot Control: Implementation Plan

Branch: `top/dynamic-env`. New code lives in `agent_robot_control/` (this
directory). Env changes live in `predicators/envs/pybullet_airport.py`,
`predicators/envs/pybullet_donut.py`, and the new
`predicators/envs/pybullet_plug_outlet.py`, which this branch owns.

## 1. Goal of the experiment

An off-the-shelf coding-agent harness (Claude Code now, OpenCode later)
controls a simulated robot arm through a small set of MCP tools and tries to
solve manipulation tasks in three PyBullet domains: Airport and Donut (coarse
manipulation, existing) and Plug-Outlet (precise insertion, new; Section 4.3).

Hypothesis: the harness is good at coarse manipulation (move the gripper near
things, push, press) but bad at fine-grained motor control. Giving it a tool
that runs model-free RL (PPO) on a reward it writes itself lets it delegate
the precise part to trial-and-error. A later model-based tool (particle world
model + MPC) should generalize better than PPO.

Conditions (each run: one env, one task, one seed, one harness):

| Condition | Tools exposed |
|---|---|
| `move_to` | `move_to`, `pixels_to_particles` |
| `model_free` | + `run_rl_on_particles` |
| `model_based` | + `run_model_based_rl_on_particles` (stub in phase 1) |

Metric: success rate as a function of total env interactions, where one
interaction is one `env.step` call (one joint-target command, 20 sim
substeps), regardless of who issued it (move_to interpolation, PPO training,
PPO policy execution, pseudo-resets). Hard cap: 100,000 interactions per run.
Success is recorded at the first interaction where the env's goal predicate
holds. The goal predicate is never revealed to the agent, only a natural-
language task description.

Scale for the first sweep: 3 envs x 3 conditions x 3 seeds = 27 runs with
Claude Code on `sonnet-5`. OpenCode doubles this later; the code supports it
from the start but it is not run until an API key is provided.

## 2. Design principles and constraints

- **No env resets.** One live simulation per run, as with a real robot.
  PPO runs sequentially in that sim, single environment, no parallel workers,
  no `saveState`/`restoreState`. The only allowed "pseudo-reset" is moving
  the arm back to an anchor pose with the normal controller. Objects stay
  wherever they end up. This means the Airport belt must loop and Donut must
  recycle donuts (Section 4).
- **The agent sees pixels, EE pose, and gripper state.** Every tool result
  includes the current camera image. The agent gets object-level geometry
  only by calling `pixels_to_particles`, which writes a file it can read.
- **Names are a known cheat.** The particle file labels each cloud with the
  env object name (e.g. `donut_0`, `target`) taken from PyBullet's
  segmentation mask. This leaks semantics a real perception stack would have
  to earn. Documented, accepted for now.
- **Reward = 1 means goal accomplished.** The agent's reward function
  returns a float; `>= 1.0` terminates the RL episode as a success and drives
  early stopping. Anything else is shaped reward.
- **Harness-agnostic.** All tools live in one standalone stdio MCP server
  process. Claude Code and OpenCode connect through their own config files
  and are launched headless by the same runner. We do not build on
  `predicators/agent_sdk/` (in-process, Claude-only, coupled to the bilevel
  planning stack); we borrow its patterns for image results, long-output
  spilling, and markdown transcripts.
- **Hydra config everywhere.** `hydra-core` (the framework now maintained at
  github.com/hydra-ecosystem/hydra) for env/harness/condition/RL/server
  settings; multirun for seeds.
- **Model-based RL is boilerplate now.** Shared backend interface, a stub
  tool that reports "not implemented", and transition logging from day one so
  the world model has data later.

## 3. Directory layout

```
agent_robot_control/
  PLAN.md                      this file
  README.md                    how to run a single tool test, a run, a sweep
  __init__.py
  conf/                        Hydra configs
    config.yaml                defaults: env=donut harness=claude_code condition=model_free seed=0
    env/airport.yaml           env name, task idx, task prompt file, camera, particle settings
    env/donut.yaml
    env/plug_outlet.yaml       clearance tier, outlet orientation, respawn policy
    harness/claude_code.yaml   binary, model=sonnet-5, max_turns, budget_usd, allowed tools
    harness/opencode.yaml      binary, model, config template
    condition/move_to.yaml     exposed tool list
    condition/model_free.yaml
    condition/model_based.yaml
    rl/sac.yaml rl/ppo.yaml    algo hparams, episode_length, action mode, workspace box, early stop
    server.yaml                interaction cap, image size, workspace dir names
  sim/
    session.py                 SimSession: owns the PyBulletEnv, interaction counter,
                               goal tracking, transition logger, run-state JSONL
    ee_control.py              EE-pose -> joint-target interpolation via IK; gripper control
    perception.py              render RGB/depth/seg, per-object particles with names,
                               particle file writer (npz + json summary); fixes Donut id mapping
    budget.py                  InteractionBudget (cap, remaining, exhaustion exception)
    anchor.py                  pseudo-reset: return arm to anchor pose
  mcp_server/
    server.py                  stdio MCP server (mcp 2.x `MCPServer`); registers tools
                               per condition; loads Hydra config from env var / argv
    tool_move_to.py
    tool_particles.py
    tool_rl.py                 run_rl_on_particles + run_model_based_rl_on_particles
    results.py                 text+image result builders, spill long text to files
  rl/
    backend.py                 RLBackend protocol, RLRequest, RLResult
    reward_loader.py           compile agent reward code in a restricted namespace, validate
    particle_env.py            gymnasium.Env over the live sim: reset-free episodes,
                               particle observations, EE-delta actions
    sb3_backend.py             SB3 SAC/PPO with early stopping, budget, policy execution
    model_based_backend.py     stub: raises NotImplemented with the documented plan
    transitions.py             (particles_t, ee_t, action_t, particles_t+1) logger (npz shards)
  harness/
    base.py                    HarnessRunner protocol: build config, launch, stream transcript
    claude_code.py             `claude -p ... --mcp-config ... --strict-mcp-config`
    opencode.py                `opencode run ...` with generated opencode.json
    prompts/
      system.md                tool semantics, budget, no-reset warning, reward contract
      task_airport.md
      task_donut.md
      task_plug_outlet.md
  experiments/
    run_experiment.py          @hydra.main: one run -> run_dir with results.json
    analyze.py                 success-vs-interactions curves per env/condition
  slurm/
    run_one.sub                sbatch template for one run on partition ellis
    submit_sweep.py            expands a sweep into a Slurm array job
  tests/
    test_ee_control.py test_perception.py test_budget.py test_particle_env.py
    test_reward_loader.py test_mcp_server.py test_env_loops.py
```

Run directory (`outputs/<env>/<condition>/<harness>/seed_<k>/`):

```
config.yaml           resolved Hydra config
workspace/            the harness's cwd: images, particle files, agent notes
  obs_0000.png ...
  particles_0003.npz / particles_0003.json
  rl_0001/            reward.py, ppo logs, policy.zip, training curve
events.jsonl          one line per tool call and per PPO window (see Section 8)
transitions/          npz shards for the world model
transcript.jsonl      raw harness stream
transcript.md         readable transcript
results.json          success, first_success_interaction, total_interactions, tool counts
```

## 4. Environment changes (predicators/envs)

### 4.1 Airport (`pybullet_airport.py`)

- **Looping belt.** In `_domain_specific_step`, an item whose x passes the
  belt end is teleported back to the belt start (same y, z, and lateral
  offset). Items recirculate forever, so the arm gets unlimited chances.
- **Goal.** Add predicate `OnTable(item)` (item's xy inside the table's
  footprint and z near table height; not on the belt). Tasks:
  `{OnTable(item_k)}` for a designated item, indexed by task idx. Both routes
  count: pushing the item off the belt with the gripper, or pressing the
  button so the pusher shoves it. The button/pusher mechanism stays.
- Register `robodisco/Airport-v0` already exists; add nothing else there.

### 4.2 Donut (`pybullet_donut.py`)

- **Cap live donuts at 8.** When a spawn would exceed the cap, remove the
  oldest donut that is not `donut_0` (or, if all are recent, the one farthest
  from the target). Removed donuts leave the sim; predicators State objects
  are reconciled accordingly.
- **Goal** stays `{InTarget(donut_0, target)}`.
- **Object ids.** Set `obj.id` for donuts, target, and robot the way Airport
  does, so segmentation ids map to names. `perception.py` also carries a
  generic fallback: build the id->name map from each env's known id
  attributes (`_donut_ids`, `_target_id`, robot id, table id).

### 4.3 Plug-Outlet insertion (new env `pybullet_plug_outlet.py`)

> Status note: implemented; see Section 12 and DEBUG_LOG entries 3, 8, 10, 11
> for what changed (3x torque caps, force limit, effective tolerances).

The task that actually tests the hypothesis: getting the plug near the outlet
is easy, seating it is not.

**Scene.** Table, Fetch robot, one **outlet** fixture and one **plug**.

- *Outlet*: a static block (8 x 8 x 3 cm) with a rectangular socket hole
  through its top face, built as a compound of five boxes (four walls plus a
  floor) with `createCollisionShapeArray`, so the hole is genuinely concave
  and a body can enter it. Default orientation `top`: the socket faces up and
  insertion is vertical, which suits the Fetch gripper's default downward
  orientation and its IK. A `wall` orientation (socket facing the robot,
  horizontal insertion) is a config switch for later.
- *Plug*: a dynamic compound body: a grip block (3 x 2 x 2 cm) on top of a
  rectangular prong (2 x 1 cm cross-section, 2 cm long). The robot grasps the
  block; the prong goes into the socket. Starts lying prong-down on the table
  at a random reachable pose about 15 cm from the outlet.
- *Clearance tiers* (`CFG.plug_outlet_clearance`, per-side gap between prong
  and socket): easy 4 mm, medium 2 mm, hard 1 mm. Default medium. Physics
  settings for tight fits: 240 Hz substeps, `numSolverIterations=150`,
  moderate `maxForce` on the grasp constraint so a misaligned plug stalls
  against the socket rim instead of exploding.

**Types and predicates.** `robot`, `plug(x, y, z, qx, qy, qz, qw, is_held)`,
`outlet(x, y, z, yaw)`. `PluggedIn(plug, outlet)`: prong tip inside the
socket footprint, insertion depth at least 1.5 cm, plug axis within 10
degrees of the socket axis. Held or released both count. `Holding(plug)` is
available for analysis only.

**Goal.** `{PluggedIn(plug, outlet)}`. Never shown to the agent; the prompt
says "plug the plug into the outlet".

**No-reset behaviour.** Nothing moves on its own. If the plug is dropped it
stays where it lands and can be re-grasped. If it leaves the table, it is
respawned on the table at a random pose and the event is logged as a human
intervention (`events.jsonl` kind `respawn`), reported alongside success.

**Why it is hard for `move_to` alone.** The socket interior is occluded, so
the agent must infer the hole centre from the outlet's top-face particles
(the ring around the hole). Alignment error must be under the clearance
(2 mm) in two axes plus yaw, from a 64-point-per-object cloud captured at
335 x 180 pixels. Position-controlled straight-line `move_to` into a tight
hole jams on the rim; the RL policy can learn small compliant wiggles.

**Reward the agent is likely to write.** Negative distance between the
lowest plug particles and the outlet hole centre, plus a bonus when the plug
top drops below a height threshold; `1.0` when the plug's lowest points are
1.5 cm below the outlet top face and inside the footprint.

**Tests.** Scripted oracle: grasp, lift, align to the true hole centre, lower;
must reach `PluggedIn` at every clearance tier (proves the physics allows
insertion). Offset oracle: the same with a 3 mm lateral error at medium
clearance must *not* succeed (proves the task demands precision). Plug
respawn triggers when pushed off the table.

### 4.4 Shared

- All envs constructed headless with `use_gui=False`, default Fetch robot,
  `CFG.pybullet_control_mode="position"`, 20 sim substeps per action.
- Tests: belt wrap-around preserves item count; donut cap holds over 2,000
  steps; plug-outlet oracles (above); goals evaluate correctly on hand-built
  states.

## 5. Simulation session (`sim/`)

`SimSession` is the single object the MCP server owns.

```python
class SimSession:
    def __init__(self, cfg): ...            # builds env, resets task, sets anchor pose
    def step(self, action: Action) -> None  # counts one interaction, logs transition,
                                            # checks goal, raises BudgetExhausted
    def ee_pose(self) -> Pose; def gripper(self) -> float
    def render_rgb(self) -> np.ndarray; def render_rgbd_seg(self) -> ...
    def particles(self, max_per_object) -> ParticleSnapshot   # names, points, colors, visible
    def goal_reached(self) -> bool
    def snapshot_image(self, tag) -> Path   # saves workspace/obs_NNNN.png
    def record_event(self, kind, **payload) # appends to events.jsonl
```

`ee_control.move_to(session, target_pose, gripper, max_steps, step_size)`:
straight-line interpolation of EE position (and slerp of orientation) from
the current pose to the target in increments of `step_size` (2 cm default);
each waypoint is converted to joint targets through `robot.inverse_kinematics`
(PyBullet IK for Fetch, IKFast for Panda) and executed with `session.step`.
Stops when within tolerance or when `max_steps` is spent. Gripper commands
set the finger joints over a few steps; grasp detection is the existing
`PyBulletEnv` machinery. IK failure returns a structured error, not an
exception, so the agent can adjust. The orientation input is roll/pitch/yaw
in degrees with the home orientation as default.

`anchor.return_to_anchor(session)` is `move_to` back to the pose the arm had
when the current RL call started, with the gripper state unchanged. It counts
interactions like everything else.

`perception.extract_named_particles(session, max_per_object)` wraps
`particle_world_model.particles.get_particles_from_rgbd_and_matrices`,
downsamples with the existing voxel routine (default raised from 20 to 64
points per object; Hydra-configurable), and attaches names. Output file:

```
particles_NNNN.npz   points_<name> (K,3) float32, colors_<name> (K,3), visible_<name> (K,) bool,
                     ee_pos (3,), ee_quat (4,), gripper (), interaction_count ()
particles_NNNN.json  {"objects": {name: {"num_points", "centroid", "bbox_min", "bbox_max"}},
                      "ee_pos", "ee_quat", "gripper", "interaction_count", "npz_path"}
```

Objects occluded or out of frame get zero points and appear in the JSON with
`num_points: 0`. The robot's own links are included under `robot`.

## 6. MCP server (`mcp_server/`)

Standalone stdio server built on the installed `mcp` 2.x Python SDK
(`from mcp.server.mcpserver import MCPServer`). Launched by the harness as
`uv run python -m agent_robot_control.mcp_server.server --run-dir <dir>`.
It reads the resolved Hydra config from `<run_dir>/config.yaml`, builds the
`SimSession`, and registers only the tools the condition allows. All state is
in-process, so it persists across tool calls for the life of the harness
session. Every tool result is text plus the current camera image (PNG, base64
image content) and ends with `interactions_used / interactions_cap`.

### Tools

`move_to(x, y, z, roll=None, pitch=None, yaw=None, gripper="keep", max_steps=200)`
Returns achieved EE pose, distance to target, IK status, steps used, and the
image. `gripper` is `"open" | "close" | "keep"`.

`pixels_to_particles(max_points_per_object=64)`
Renders, extracts, writes the npz+json pair into the workspace, returns the
JSON summary text plus both paths. Does not consume interactions.

`run_rl_on_particles(reward_code, budget_interactions, episode_length=50, action_mode="xyz", workspace_half_extent=0.15, control_gripper=false)`
Runs the model-free backend (Section 7; algorithm fixed per run by config, not chosen by the agent). Returns a training summary: windows
run, best/last mean reward, number of windows that reached reward 1, whether
early stopping fired, interactions spent, path to the RL sub-directory, and
the image after the final policy execution.

`run_model_based_rl_on_particles(...)` (same signature)
Phase 1: returns an `is_error` result saying the backend is not implemented.
Exposed only in the `model_based` condition so the agent's tool list matches
the condition even before the backend exists.

No `get_observation` tool: the initial image path, EE pose, and gripper state
are placed in the task prompt, and every tool call refreshes them.

### Reward contract (in the tool description and system prompt)

```python
import numpy as np
def reward(particles: dict[str, np.ndarray],   # name -> (K,3) world xyz, K may be 0
           visible: dict[str, np.ndarray],     # name -> (K,) bool
           ee_pos: np.ndarray, ee_quat: np.ndarray, gripper: float) -> float:
    """Return >= 1.0 when the goal is accomplished; smaller values shape progress."""
```

`reward_loader` compiles the code with a whitelist of builtins and `numpy`
only, runs it once on the current particles with a timeout, and rejects it
with the traceback if it fails or returns a non-finite number.

### Budget and termination

`InteractionBudget` raises `BudgetExhausted` inside `session.step`. Tools
catch it, return a final result marked `budget_exhausted: true`, and further
tool calls return the same error. The runner also kills the harness if the
budget is exhausted and the harness keeps talking.

Tool call timeouts: RL calls can run for a long time. Claude Code's MCP tool
timeout is raised through the `MCP_TOOL_TIMEOUT` environment variable in the
harness launcher; OpenCode's equivalent is set in its generated config. The
RL tool also writes progress to `events.jsonl` so a stalled run is visible.

## 7. Model-free RL backend (`rl/`)

### Algorithm choice

The setting is single-stream, reset-free, low-dimensional continuous control
with a hard interaction budget. That favours **off-policy** methods: every
transition stays in the replay buffer and is reused many times, and data from
earlier windows remains useful as the object configuration drifts. PPO
discards its data after each update and needs far more samples.

The algorithm is a Hydra option behind the same backend interface
(`rl.algo`): `sac` (default) or `ppo`. Both come from Stable-Baselines3, so
the implementations stay clean and interchangeable:

- **SAC** (SB3): the default. Well-understood, stable, and in the reset-free
  real-robot literature (SERL and follow-ups) it is the workhorse. Run with
  `train_freq=1` and `gradient_steps` in 1 to 4 (update-to-data ratio) to
  squeeze the budget.
- **PPO** (SB3): kept for comparison and because it is the least sensitive
  to reward scale; expected to be the least sample-efficient here.

Version pin: the repo pins `torch==2.7.1` for the PTv3 stack, and
`stable-baselines3>=2.9` requires torch 2.8, so pin
`stable-baselines3==2.7.0`.

Reward scaling: off-policy critics are sensitive to reward magnitude and the
agent's rewards vary. `ParticleEnv` clips the agent reward to [-10, 1] and
applies a fixed scale, logged in `events.jsonl`, before it reaches the
learner. The success threshold (`>= 1.0`) is checked on the raw value.

Observation encoder: the flat per-object point-cloud observation (up to
~1.5k floats) works with the default MLP; a small per-object PointNet
feature extractor (shared MLP + max-pool per object, SB3 custom
`BaseFeaturesExtractor`) is a config option if the MLP struggles.

Pilot: before the sweep, compare `sac` and `ppo` on the hand-written
Donut reward and the medium-clearance plug-outlet oracle reward, 3 seeds
each, on `ellis`. The winner becomes the default for the sweep.


### Reset-free episode structure

`ParticleEnv(gymnasium.Env)` wraps the live `SimSession`:

- `reset()` performs the pseudo-reset (arm to anchor pose via
  `return_to_anchor`), re-extracts particles, and returns the observation. It
  never touches objects.
- `step(a)` converts the action to an EE target (Section 7.2), executes one
  `session.step`, extracts particles, evaluates the agent reward, and returns
  `terminated = reward >= 1.0`, `truncated = t >= episode_length`.
- Object list and per-object point count `K` are fixed at the first
  extraction of the call; missing objects are zero-padded with
  `visible=False`.

SB3 sees an ordinary single env (`DummyVecEnv` of size 1). Off-policy
learners update every step after a short warm-up (`learning_starts=500`);
for PPO, `n_steps` is small (256) so updates happen often relative to the
budget.

### Observation and action

Observation: `concat(points of every object (K*3 each), visible masks, ee_pos,
ee_quat, gripper)`, normalized relative to the anchor EE position. Objects are
ordered by name for stability.

Action modes (Hydra `rl.action_mode`, agent-selectable per call):
`xyz` (3-D EE delta), `xyz_yaw` (4-D), each optionally plus a gripper scalar.
Each component is in [-1, 1] and scaled to at most 1 cm of translation or
5 degrees of yaw per step. Targets are clipped to a workspace box of half-
extent `workspace_half_extent` around the anchor position. The EE target goes
through the same IK path as `move_to`.

### Training loop and early stopping

`SB3Backend.run(request)` (one class, algorithm chosen by config):

1. Validate reward. Record the anchor pose. Open `rl_NNNN/`.
2. Train the SB3 learner with a callback that: logs every episode (window) to
   `events.jsonl`; stops when `budget_interactions` is spent; stops early
   when at least `early_stop.successes` of the last `early_stop.window`
   episodes reached reward 1 (defaults 5 of 8).
3. Execute the trained policy deterministically from a fresh pseudo-reset
   for up to `episode_length` steps or until reward reaches 1 (repeats up to
   `final_exec_attempts` times, default 3).
4. Save `policy.zip`, `reward.py`, `training_curve.png`, and return
   `RLResult`.

The sim ends wherever the last execution left it; that is the state the agent
continues from.

### Model-based stub and transition logging

`RLBackend` protocol: `run(request: RLRequest) -> RLResult`. `RLRequest`
carries the reward callable, budget, episode length, action mode, workspace
box, and paths. `ModelBasedBackend.run` raises `NotImplementedError` with a
message pointing here.

`transitions.py` logs every `session.step` as `(particles_t, visible_t, ee_t,
gripper_t, action_t as EE delta, particles_t+1, visible_t+1)` into npz shards
of 1,000 transitions, regardless of whether the step came from move_to or
PPO. Particle extraction is already needed for the PPO observation; for
move_to steps it adds a render per step, gated by `server.log_transitions`.

Planned model-based design (documented, not implemented): action-conditioned
PTv3 flow model (per-point features = xyz + broadcast EE delta + one-hot
object id), trained on the logged transitions plus the agent's task data;
MPC by sampling EE-delta sequences (CEM or MPPI) through autoregressive
rollouts and scoring with the same reward callable; execute first action,
replan. It reuses `particle_world_model.rollout_ptv3_flow.rollout_model`
once an action input is added there. Runs on Supercloud (CUDA required).

## 8. Harness launchers (`harness/`)

Common flow (`HarnessRunner.run(run_dir) -> HarnessOutcome`):

1. Write `<run_dir>/config.yaml` (resolved Hydra) and the MCP server config
   pointing at `python -m agent_robot_control.mcp_server.server --run-dir`.
2. Create `workspace/`, save the initial image, render the task prompt from
   `prompts/task_<env>.md` with the image path, EE pose, gripper, and budget.
3. Launch the harness headless with cwd = `workspace/`, stream stdout to
   `transcript.jsonl`, enforce a wall-clock limit and the interaction cap.
4. After exit, read `events.jsonl` and produce `results.json` and
   `transcript.md`.

Claude Code: `claude -p "<prompt>" --model sonnet-5 --mcp-config mcp.json
--strict-mcp-config --allowedTools "mcp__robot__*,Read,Write,Bash(python*)"
--append-system-prompt-file system.md --output-format stream-json
--max-turns N --max-budget-usd X --permission-mode bypassPermissions`.
Verified against Claude Code 2.1.263.

OpenCode: generate `opencode.json` with an `mcp.robot` local server entry and
the model, run `opencode run "<prompt>"` with output captured. Implemented
and unit-tested for config generation; not exercised end-to-end until an API
key is supplied.

Built-in tools policy: the agent may read and write files in `workspace/`
(particle files, notes, reward drafts) and run Python there to analyze
particles, but the MCP tools are the only way to act on the sim.

`events.jsonl` line schema: `{"t": iso, "kind": "tool_call"|"tool_result"|
"rl_window"|"goal_reached"|"budget_exhausted", "tool", "args" (reward code
hashed, stored separately), "interactions_before", "interactions_after",
"goal", "image"}`.

## 9. Experiment runner and analysis (`experiments/`)

`run_experiment.py` is a `@hydra.main` entry point. One invocation is one
run. Sweeps use multirun:

```
uv run python -m agent_robot_control.experiments.run_experiment -m \
  env=airport,donut condition=move_to,model_free,model_based seed=0,1,2 harness=claude_code
```

`analyze.py` reads every `results.json`, builds for each (env, condition,
harness) the curve "fraction of seeds with success by interaction t" over
[0, 100k], plus a table of first-success interaction, tool-call counts, RL
calls, and wall clock. Output: PNG plots and a markdown table.

## 9b. Compute: everything heavy runs through Slurm

This machine (`en-cc-unicorn-login-05`) is a **login node**. Nothing compute-
heavy runs here: no PPO, no sweeps, no harness runs, no world-model training.
Local use is limited to editing, unit tests that finish in seconds, and short
smoke checks (a few hundred sim steps). Everything else is an `sbatch` job.

Partition choice: **CPU-only jobs go to `default_partition`** (the large
general pool; `ellis`, the group's two nodes, is often fully booked and jobs
sit in `PENDING`). **GPU jobs go to the `gpu` partition.** Every script here
defaults to `default_partition`; `submit_sweep.py --partition ...` overrides
it.

Job scripts follow the user's existing convention (`~/scripts/jepa.sub`,
`~/scripts/ptv3_uv_test.sub`): `#SBATCH --get-user-env`, `-N 1`,
`--ntasks-per-node=1`, logs to `logs/%j.out` / `logs/%j.err` relative to the
submission directory, `cd /home/wp237/predicators`, `export PYTHONHASHSEED=0`,
commands through `uv run`, optional job rename from `$1` via `scontrol`.

- `agent_robot_control/slurm/run_one.sub`: one experiment run. CPU only
  (`--cpus-per-task=4`, `--mem=32000`, no `--gres`), wall clock from config
  (default `-t 24:00:00`). Sources the Anthropic API key from a file outside
  the repo, raises the MCP tool timeout, then calls `run_experiment.py` with
  Hydra overrides passed as script arguments.
- `agent_robot_control/slurm/submit_sweep.py`: expands env x condition x
  seed x harness into one Slurm **array** job, one task per run,
  each writing its own run directory. Preferred over Hydra's multirun
  launcher so `squeue` shows per-run state and single tasks can be
  resubmitted. Uses `--requeue` like `~/scripts/quick.sub`.
- World-model training and MPC (later phase) add `--gres=gpu:1`,
  `--partition=gpu` and `uv sync --extra ptv3`.
- Phase 0 includes a short job that confirms compute nodes can reach
  the Anthropic API and run Claude Code headless, and that headless PyBullet
  rendering works there.

## 9c. Debugging discipline

The system has many moving parts and the first sweeps are expected to fail
in places. Rules for every phase:

- Each component gets a standalone test or script that exercises it with no
  harness in the loop (env loops, move_to accuracy, particle files, reward
  loader, PPO with a hand-written reward, MCP tools through an in-process
  client, harness launch with a mock server).
- A failure is not "fixed" until the cause is known and written down in
  `agent_robot_control/DEBUG_LOG.md`: symptom, evidence (log lines, numbers),
  cause, fix, and how the test now guards against it.
- Every run directory is self-describing (`events.jsonl`, images per tool
  call, transcript, RL curves), so a failed cluster run can be diagnosed
  without rerunning it.
- Component-level expectations are stated up front and checked: move_to
  final EE error under 5 mm on reachable targets; interaction counts match
  `env.step` calls exactly; particle names present for every visible object
  in both envs; PPO reaches reward 1 on the hand-written Donut reward on at
  least one of three seeds within 30k interactions.

## 10. Phases and deliverables

Each phase ends with its tests passing under `uv run python -m pytest
agent_robot_control/tests`.

**Phase 0: dependencies, env changes, cluster check.** Add `hydra-core`,
`stable-baselines3==2.7.0` via `uv add`. New
`pybullet_plug_outlet` env with oracle tests. Slurm job template and sweep submitter;
`srun` smoke test of Claude Code headless from a compute node. Airport looping belt and `OnTable` goal;
Donut cap and object ids. Tests in `test_env_loops.py`.

**Phase 1: sim session, perception, move_to.** `SimSession`, budget,
`ee_control`, `perception` with named particles for both envs, transition
logger. Tests: move_to reaches targets within tolerance in both envs, IK
failure path, budget exhaustion, particle files well-formed, Donut names
present.

**Phase 2: MCP server.** Tools `move_to`, `pixels_to_particles`, the RL tool
shells wired to the backend protocol, image results, spill-to-file. Test with
an in-process MCP client (`mcp` client over stdio) in `test_mcp_server.py`.
Manual smoke run: Claude Code on Donut with `move_to` only.

**Phase 3: RL backend.** `reward_loader`, `ParticleEnv`, `SB3Backend`, early
stopping. Offline tests without a harness, on `ellis`: hand-written reward
"donut_0 centroid within 3 cm of target centroid" from an anchor pose next
to the donut, and the plug-outlet insertion reward from an anchor 3 cm above
the socket; assert reward 1 is reached within budget on at least one seed and
that interactions are counted exactly. SAC-vs-PPO pilot (Section 7).

**Phase 4: harness launchers and runner.** Claude Code launcher, OpenCode
config generation, `run_experiment.py`, `analyze.py`, prompts. Smoke run of
one seed of Donut `model_free` end to end.

**Phase 5: model-based boilerplate.** `ModelBasedBackend` stub, the fourth
tool in the `model_based` condition, transition shard validation, a note in
`particle_world_model/README.md` about the planned action-conditioned model.

**Phase 6: first sweep.** 27 runs with Claude Code on `sonnet-5`, analysis,
write-up of failure modes.

## 11. Risks and open items

- **Sample efficiency of reset-free RL.** Even off-policy learners may not
  solve a one-shot agent-written reward within the budget. Mitigations:
  short episodes, high update-to-data ratio, dense rewards encouraged in the
  prompt, small workspace boxes, the algorithm pilot before the sweep.
- **Tight-clearance physics.** PyBullet with 1 to 2 mm gaps can jitter or
  tunnel. The oracle tests in Section 4.3 gate the tiers we actually use.
- **Rendering cost.** PPO needs a render per step for particles. Headless
  TinyRenderer at 335x180 is milliseconds; if it dominates, render every
  other step or reduce K.
- **Occlusion and single camera.** Particles are partial and view-dependent.
  A second camera is a config switch in `perception.py` if needed.
- **Harness behavior.** The agent may burn budget on tiny `move_to` calls or
  write degenerate rewards. Prompts state the budget and the reward contract
  explicitly; the tool results always echo remaining budget.
- **Long tool calls.** MCP tool timeouts must be raised in both harnesses;
  verified in Phase 2's smoke run.
- **Airport `OnTable` definition** may need tuning once the pusher trajectory
  is observed; the predicate is a single function.
- **Cheats to keep in mind for the write-up:** object names from
  segmentation, perfect depth, goal predicate used only for scoring, plug
  respawn when it leaves the table.

## 12. Status (updated 2026-09-07)

Implemented and tested on the login node (`agent_robot_control/tests`, 32
tests, about 90 seconds): Phases 0 to 5. First cluster runs launched.

Deviations from the sections above, all recorded in `DEBUG_LOG.md`:

- **Torque limits are 3x URDF**, not 1x: raw limits make PyBullet's
  gravity-blind position controller sag by 20 cm in stretched poses; 3x
  still stops a misaligned plug at the rim (entries 3, 8).
- **IK is plain-first with a joint-limit check, null-space fallback** (entry
  9); `move_to` results list the bodies the robot touches.
- **Particles use farthest-point sampling** and `np.einsum` back-projection
  because of a NumPy/OpenBLAS bug on these CPUs (entries 1, 2);
  `OPENBLAS_NUM_THREADS=1` is set everywhere.
- **Camera stays at 335x180** for RL observations (55 ms per frame on CPU;
  640x360 costs 175 ms). Higher resolution is a config switch.
- **Goals require resting, unheld objects** (`InTarget`, `OnTable`): the first
  smoke run "succeeded" while carrying the donut above the target.
- **RL calls abort when a held object is dropped** (entry 10) or when the
  reward stagnates for 40 episodes, instead of burning the budget on an
  unreachable object.
- **Airport's button is a static contact switch** (entry 13); the pusher route
  has a timing window of about 0.40 to 0.45 m of item lead, verified by a
  gate test.
- **The controller is force-limited** (80 N, entry 11): `move_to` stops and
  reports the contact, the RL env retracts and penalises. Grasp compliance
  self-aligns lateral errors up to about 4 mm at 2 mm clearance, so the
  medium tier is solvable open-loop; `env=plug_outlet_hard` (1 mm) is
  exposed for the sweep but no oracle passes it.
- **Outputs live outside the repo** (`$HOME/arc_outputs`) and a Claude Code
  `PreToolUse` hook confines file access to the workspace (entry 7).
- **Model ID** is `claude-sonnet-5` (the alias `sonnet-5` is rejected).
- Claude Code spends 1 to 3 turns on `ToolSearch` to load the MCP tool
  schemas; harmless but counts toward `max_turns` (60).

Verified end to end: Claude Code headless, from a login-node smoke test and
from an `ellis` compute node, connected to the server, extracted particles,
grasped and moved objects, and left `events.jsonl` / `results.json` /
`transcript.md` per run. OpenCode remains untested pending an API key.

Sweep `arc_sweep1` (Slurm array 530099; arrays 529920 and 530037 were cancelled
after the finger-force and stagnation-rule fixes): 4 env configs (airport, donut,
plug_outlet medium, plug_outlet hard) x 2 conditions (move_to, model_free)
x 3 seeds = 24 runs, 6 concurrent, outputs in `$HOME/arc_outputs`. The
`model_based` condition is deferred until its backend exists (running it now
would only duplicate `model_free` plus a failing tool). First single runs:
donut and plug-outlet (medium) succeeded with `move_to` alone at 145 and 97
interactions; airport failed (button had no travel, fixed since).

Airport tasks of array 530099 ran without the `wait` tool (added later,
entry 15) and must be re-run: `submit_sweep.py --envs airport --conditions
move_to model_free`.

Sweep 1 finished (all 24 cells valid after re-running 8 usage-cap-truncated
runs): see `RESULTS_sweep1.md`. Headline: RL tool raised precision-tier
success from 4/9 to 7/9 at a 5x to 50x interaction cost; hurt on Airport. Hand-written-reward pilots (SAC
and PPO, plug insertion, 20k budget) both stagnated without inserting.

Open items: model-based backend, OpenCode end-to-end, a second camera or
occlusion-aware reward guidance for insertion, Airport prompt/tooling for the
timing task, `model_based` condition once the backend exists.
