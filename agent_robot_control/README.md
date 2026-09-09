# agent_robot_control

An agent harness (Claude Code now, OpenCode later) controls a simulated robot
arm through MCP tools. See `PLAN.md` for the design and `DEBUG_LOG.md` for
every failure we hit and why.

## Layout

| Path | What |
|---|---|
| `sim/` | `SimSession` (one live sim, interaction budget, event/transition logs), `EEController` (IK-based move_to), `perception` (named particles) |
| `mcp_server/` | stdio MCP server: `move_to`, `pixels_to_particles`, `run_rl_on_particles`, `run_model_based_rl_on_particles` (stub) |
| `rl/` | reward loader, reset-free `ParticleEnv`, SB3 SAC/PPO backend, model-based stub |
| `harness/` | Claude Code and OpenCode launchers, prompts, workspace sandbox hook |
| `experiments/` | `run_experiment` (Hydra), `analyze`, `rl_pilot` (hand-written rewards, no harness) |
| `conf/` | Hydra config groups: env, harness, condition, rl |
| `slurm/` | `run_one.sub`, `rl_pilot.sub`, `make_videos*.sub`, `submit_sweep.py` (array job). CPU jobs use `default_partition`; GPU work uses the `gpu` partition |
| `tests/` | pytest suite; ~2 min, but it drives PyBullet, so it goes through `sbatch` like everything else |

Envs live in `predicators/envs/`: `pybullet_airport.py` (looping belt, delayed
pusher, `OnTable` goal), `pybullet_donut.py` (long-table push with ungraspable
discs, `InTarget` goal), `pybullet_plug_outlet.py` (US three-leg insertion,
`PluggedIn` goal). Each has a ground-truth oracle under `experiments/` that
gates feasibility: `plug_oracle`, `donut_oracle`, `airport_oracle`.

## Running

Tests (cluster; the suite drives PyBullet, so it is not login-node work):

```bash
sbatch agent_robot_control/slurm/tests.sub
```

One run (cluster):

```bash
sbatch -t 24:00:00 agent_robot_control/slurm/run_one.sub env=plug_outlet condition=model_free seed=0
```

Full sweep (27 runs as one array job):

```bash
uv run python -m agent_robot_control.slurm.submit_sweep --harness claude_code
uv run python -m agent_robot_control.experiments.analyze $HOME/arc_outputs --out $HOME/arc_outputs/analysis
```

RL pilot without a harness (SAC vs PPO on hand-written rewards):

```bash
sbatch agent_robot_control/slurm/rl_pilot.sub --task plug_insert --algo sac --seed 0
```

Outputs go to `$ARC_OUTPUT_ROOT` (default `$HOME/arc_outputs`), which must be
outside this repo (see DEBUG_LOG entry 7). Each run directory holds
`config.yaml`, `prompt.md`, `workspace/` (images, particle files),
`events.jsonl`, `rl_NNNN/`, `transitions/`, `transcript.{jsonl,md}`,
`results.json`.

## Environment notes

* `OPENBLAS_NUM_THREADS=1` everywhere (numpy's bundled OpenBLAS miscomputes
  tall-skinny matmuls on the AVX-512 Xeons here; DEBUG_LOG entry 1).
* Rendering is CPU TinyRenderer: about 55 ms per frame at 335x180. RL
  observations use that resolution; `pixels_to_particles` can request more
  points but not more pixels.
* The Claude Code harness runs headless with `--model claude-sonnet-5`,
  `--strict-mcp-config`, `--allowedTools` for the MCP tools plus Read/Write/
  Bash(python*), and a `--settings` hook confining file access to the run's
  workspace.
