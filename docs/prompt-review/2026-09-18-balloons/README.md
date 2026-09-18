# Prompt review of the comparison arms, 2026-09-18

One review copy per arm of the eight-agent benchmark sweep (`scripts/configs/predicatorv3/continual_eight_agent_noisy_sweep.yaml`), plus the agentic real-to-sim baseline added later that day (`continual_real_to_sim_benchmark_r1.yaml`), rendered on the Balloons benchmark setting (composition test levels with the 25-step dwell, seed 0).
Each copy holds everything the agent is sent at the start of a run, in the order the harness assembles it:

1. The full system prompt, including the sandbox suffix appended by the local sandbox session.
2. The sandbox `CLAUDE.md` written into the agent's working directory.
3. The reference files copied under `./reference/`.
4. The MCP tools, with the descriptions the agent sees, split into the static protocol tools and the tools attached per round (`run_python` over the model workbench for the model arms).
5. The first-round query, with the real level, ledger, noisy observation, model status, journal and attempts placeholders, and the skill, predicate and type vocabulary.
6. The continuation query after one zero action.

The copies come from `scripts/dump_continual_arm_prompts.py`, which builds the real approach and environment, scripts two conversation rounds (one zero action, then give up) and records the text instead of querying a model.
Regenerate them with:

```bash
python -m scripts.dump_continual_arm_prompts --config predicatorv3/continual_eight_agent_noisy_sweep.yaml --domain balloons --out docs/prompt-review/2026-09-18-balloons
```

The same command with another `--domain` (`boil`, `bridge`, `fan`, `domino_high_friction_turn`) renders the other benchmark settings, and `--config predicatorv3/continual_real_to_sim_benchmark_r1.yaml` renders the real-to-sim arm.

## Arms

| Arm | File | What the prompt says the agent can do |
| --- | --- | --- |
| EMPIRIC (model-based) | [agent_continual.md](agent_continual.md) | Write and revise `simulator.py`, fit it with `sim.fit`, plan under parameter and pose uncertainty, repair the model when it disagrees with recordings. |
| Direct agent (model-free) | [agent_continual_model_free.md](agent_continual_model_free.md) | Act with the skill library and the sandbox; no simulator, no `run_python` workbench. |
| Standalone program world model | [agent_continual_program_world_model.md](agent_continual_program_world_model.md) | Write `world_model.py` over skill transitions without the base physics; `sim.score`, `sim.run`, `sim.refine` on that program; no engine diagnostics. |
| Oracle dynamics | [agent_continual_oracle_dynamics.md](agent_continual_oracle_dynamics.md) | The supplied simulator carries the true mechanisms and parameters, fixed for the run; predicates only; no fitting, no model edits. |
| Scene-only | [agent_continual_scene_only.md](agent_continual_scene_only.md) | The supplied simulator is the exact scene twin with corrected base calibration and no mechanism code, fixed for the run; predicates only; no fitting, no model edits. |
| Zero-shot model | [agent_continual_zero_shot.md](agent_continual_zero_shot.md) | Write `simulator.py` once before the first real action; the dynamics are sealed at that point; no fitting afterwards. |
| No harness fitting | [agent_continual_no_fitting.md](agent_continual_no_fitting.md) | Write and revise the model, but the harness fits nothing: declared values deploy as written, and the agent may estimate them in its own sandbox code. |
| No explicit uncertainty handling | [agent_continual_no_uncertainty.md](agent_continual_no_uncertainty.md) | Fit raw noisy transitions and plan from the latest raw observation: no observation smoothing or denoising, belief draws, parameter sweeps, `sim.belief` or `sim.suggest_probes`, including agent-written substitutes. |
| Agentic real-to-sim | [agent_continual_real_to_sim.md](agent_continual_real_to_sim.md) | No domain twin: build `simulator.py` on `SceneBase` from the generic engine wrapper, the scene manifest and the URDF and mesh files; declare and set parameters yourself; no harness fitting, no uncertainty machinery; `sim` has no world until the file loads. |

## What differs between the arms

- The identity line, the arm statement after the observation-noise rules (frozen arms), the decision workflow, the model workbench table, the model or predicate API reference, and the closing point-estimate section are all rendered per arm by `predicators/agent_sdk/play_prompts.py` from `play_system.md`, `play_frozen.md` and `play_model_contract.md`.
- The `run_python` tool description follows the arm's probe surface: fitting, model writing, alternative parameter values and uncertainty sweeps appear only where the probe accepts them.
- The supplied models of scene-only and oracle dynamics never enter the sandbox: they run inside `sim`, and no report or file names their constants.
- The query's model-status line is per arm: supplied models are reported as present and fixed from the first round, the zero-shot arm is told to write the model before its first action, the no-fitting arm is told to set declared values, and the standalone arm is told to write `world_model.py`.
- The real-to-sim arm's references are the engine sources (`base_sim/pybullet_env.py`, `base_env.py`, `scene_base.py`), `scene/scene_manifest.json` (bodies, shapes, mesh files, joints, colours, the observed object of each body; no masses, frictions or damping) and `assets/` (the URDF and mesh files those bodies and the robot were loaded from); the other model arms get the domain base-simulator sources instead.
- The sandbox `CLAUDE.md` and the static protocol tools are shared.

## Size

Word counts of the rendered text, Balloons seed 0.

| Arm | System prompt | Round 1 query | Round 2 query |
| --- | ---: | ---: | ---: |
| `agent_continual` | 3388 | 638 | 476 |
| `agent_continual_model_free` | 1072 | 515 | 353 |
| `agent_continual_no_fitting` | 3457 | 647 | 485 |
| `agent_continual_no_uncertainty` | 3432 | 542 | 380 |
| `agent_continual_oracle_dynamics` | 1997 | 618 | 456 |
| `agent_continual_program_world_model` | 1461 | 621 | 459 |
| `agent_continual_scene_only` | 2013 | 616 | 454 |
| `agent_continual_zero_shot` | 3005 | 647 | 485 |
| `agent_continual_real_to_sim` | 3655 | 556 | 394 |
