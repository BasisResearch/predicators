# Prompt review: direct agent with the scene files, 2026-09-19

One review copy of the arm launched by `scripts/configs/predicatorv3/continual_direct_scene_files_benchmark_r1.yaml` (menu entry `mf_scene_package_opus`), rendered on the Balloons benchmark setting, seed 0, at commit `f6f609636`.
It is the direct agent (`agent_continual_model_free`) with the engine wrapper, the scene manifest and the URDF and mesh files as read-only references.
The prompt only asks the agent to solve the levels: it has no simulator, no model files, no fitting and no model gate.
The one addition over the plain direct agent is the "Scene files" section of the system prompt.

The copy holds the full system prompt, the sandbox `CLAUDE.md`, the reference files, the tools, the first-round query and the continuation query, in the layout of [the Sept 18 review](../2026-09-18-balloons/README.md).
Regenerate it with:

```bash
python -m scripts.dump_continual_arm_prompts --config predicatorv3/continual_direct_scene_files_benchmark_r1.yaml --domain balloons --out docs/prompt-review/2026-09-19-direct-scene-files
```

The script names its output after the approach (`agent_continual_model_free.md`); this copy was renamed to [direct_agent_scene_files.md](direct_agent_scene_files.md) so it does not read as the plain direct agent.
