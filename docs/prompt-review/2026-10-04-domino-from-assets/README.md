# Prompt review: EMPIRIC from assets on Domino, 2026-10-04

This folder holds what the EMPIRIC-from-assets arm is told on the Domino benchmark setting (`domino_high_friction_turn`).
The arm is `agent_continual_from_assets`, menu entry `from_assets_opus` in `scripts/configs/empiric/approaches.yaml`.
Everything here comes from master `be822d71f`, the same tree as the round `fixes_r1` runs (`33a77cf2c`).
Seed 1 of that round won level 1 and lost the test level: its first bridge, struck 46 degrees off its topple axis, slid instead of toppling.

## Files

- [agent_continual_from_assets.md](agent_continual_from_assets.md): the review copy, in the layout of [the Sept 18 review](../2026-09-18-balloons/README.md), rendered on seed 1's tasks.
  It holds the system prompt with the sandbox suffix, the sandbox `CLAUDE.md`, the reference files, the tools, the first-round query and the continuation query.
- [seed1_as_delivered.md](seed1_as_delivered.md): the system prompt and the three queries seed 1 actually received, copied from its run folder.
- [scene_manifest.json](scene_manifest.json): the scene manifest seed 1 built its simulator from.

The review copy's system prompt matches the delivered one line for line except for the body and asset counts (defect 3 below).
Its sandbox `CLAUDE.md` is identical to the delivered one.
Source paths in its reference list are shortened to the repository root and `<run>`.

`scripts/dump_continual_arm_prompts.py --config` skips parked arms, so the copy was rendered by calling its `dump_arm` on the arm's run configuration:

```python
from scripts.cluster_utils import generate_run_configs
from scripts.dump_continual_arm_prompts import dump_arm

cfg = next(c for c in generate_run_configs(
    "empiric/benchmark.yaml", False, approaches=["from_assets_opus"])
           if c.env == "pybullet_domino")
dump_arm(cfg, 1, "<scratch dir>", "docs/prompt-review/2026-10-04-domino-from-assets")
```

## Defects in what the arm is told

1. Before a model exists, the model-status line is the agentic real-to-sim arm's: "set declared values from the recordings before you act on a test level; the harness fits nothing".
   This arm fits (`sim.fit()`), and its own system prompt asks for a fitted `simulator.py` before a test level.
   `AgentContinualFromAssetsApproach._no_model_section` returns the `no_model_scene` section of `play_query.md`.
2. After a restart between levels, the next query reports "No model yet" although `simulator.py` exists, and it shows an empty attempts record.
   It also says the level "resumes after compute preemption" although the level had not started.
   Seed 1 restarted at the level boundary when it moved to Claude account b.
   Seed 0 did not restart, and its level-2 query named its current files and fit status.
3. The reference list reads `scene_manifest.json (0 bodies)` and `assets/ (0 URDF and mesh files ...)` in the first round of every process.
   `AgentBaseApproach._ensure_agent_session` renders the system prompt (`system_prompt=`) before it builds the reference files (`extra_reference_files=`), and building them is what sets `_scene_package_summary`, which starts at `(0, 0)`.
   Later rounds of the same process show the previous round's counts, so a level whose scene changed would show the old level's counts.
   All three of seed 1's queries opened a new process and showed zeros.
   The review copy shows 8 bodies and 38 files because it renders the prompt after the run.

## What seed 1 was told about materials

The system prompt names `<type>_spinning_friction` and `<type>_rolling_friction` among the engine materials the base offers.
It says an undeclared material runs at PyBullet's default (spinning and rolling friction 0) and asks the agent to declare the ones its tasks' contacts depend on.
Seed 1 declared the dominoes' mass, lateral friction and restitution and the support's lateral friction and restitution.
It left spinning and rolling friction undeclared.
The world's dominoes have spinning friction 0.5 and rolling friction 0.006.
Replaying the level-2 episode in the real environment from its recorded actions reproduces the slide.
With either of the two set to 0, the same push cascades to the purple.
In the same replay, starting from the agent's values and setting only these two to the world's stops the first bridge from toppling.
The slide also needs the gripper body, which brushed the first bridge on the push's follow-through: with only the fingertips tangible, the same layout cascades at the world's materials.
Level 1's straight chain does not identify either friction: on its recording, the agent's own model scores the agent's values as well as the world's.
