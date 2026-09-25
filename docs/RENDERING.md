# Rendering EMPIRIC paper figures

This is the canonical rendering guide.
Run the commands below from the Predicators checkout.

## Locations

- Tools: `scripts/paper_figures/`.
- Archived image inputs: `scripts/paper_figures/figures/sources/`.
- Scene-to-image mapping and trajectory provenance: `scripts/paper_figures/data/`.
- Paper output: `figures/` in the sibling `sim-predicator-paper` checkout, or the checkout specified by `EMPIRIC_PAPER_ROOT`.
- The paper retains only the table inputs needed by LaTeX under `data/current-results/`.

## Build Figures 1 to 3

Install CairoSVG, Pillow, and PyMuPDF, plus the system Cairo library and DejaVu Sans fonts.
Fontconfig must see DejaVu Sans in its book, bold, and oblique styles; if only DejaVu Sans Mono is installed, every label silently falls back to the monospace face.
Matplotlib ships the four DejaVu Sans files, which can be copied to `~/.local/share/fonts` before running `fc-cache`.
The compositor uses archived images and recorded data and does not launch physics or agents.

```bash
export EMPIRIC_PAPER_ROOT=/path/to/sim-predicator-paper
python scripts/paper_figures/build_figures.py --overview
python scripts/paper_figures/verify_figures.py
```

Figure 1 uses `fig1_residual`, Figure 2 uses `fig2_method`, Figure 3 uses `fig3_trajectories`, and the appendix trajectory figure uses `figA_trajectories`.
PDF, SVG, and PNG outputs are generated.
The generated manifest records input and output hashes.
Do not edit generated images or manifests manually.
Compile the paper from its own checkout with `latexmk -pdf main.tex`.

## Blender Cycles renders

The committed Cycles images were generated with Blender 4.5.3, Cycles CPU, 48 samples, eight render threads, AgX Medium High Contrast, and deterministic seed zero.
The renderer environment needs `bpy==4.5.3` and `pycollada`.
The scene exporters restore recorded states and verify that rendering does not advance physics.
Run the expensive render step on a compute node.

The complete refresh sequence is:

```bash
# Run these importers and exporters in the Predicators environment.
python scripts/paper_figures/import_figure3_trajectories.py
python scripts/paper_figures/export_trajectory_scenes.py
python scripts/paper_figures/export_static_scenes.py

# Run this renderer in a Blender 4.5.3 Python environment on a compute node.
python scripts/paper_figures/render_cycles_scenes.py \
  --blender-python /path/to/blender-python \
  --samples 48 \
  --threads 8

# These steps are inexpensive and can run on a login node.
python scripts/paper_figures/prepare_figma_assets.py \
  --output-dir /tmp/empiric-figma-assets
python scripts/paper_figures/build_figures.py --overview --renderer cycles
python scripts/paper_figures/verify_figures.py
```

To render only selected Bridge scenes after changing their exported state or appearance, use:

```bash
python scripts/paper_figures/render_cycles_scenes.py \
  --blender-python /path/to/blender-python \
  --domains Bridge \
  --scenes bridge_start trajectory_bridge_train_1 bridge_pair_observed \
  --samples 48 \
  --threads 8
```

On the cluster, the equivalent self-contained invocation is:

```bash
uv run --no-project --python 3.11 \
  --with bpy==4.5.3 --with pycollada \
  python scripts/paper_figures/render_cycles_scenes.py \
  --samples 48 --threads 8
```

`--domains` now filters both static and trajectory scenes.
`--scenes` selects exact scene stems.
Add `--dry-run` to inspect the selected scene list without invoking Blender.
Renderer logs are temporary unless `--log-dir` is supplied.

The renderer preserves recorded transforms and authored mesh normals, uses flat normals on planar compound boxes, and records hashes and settings in `scripts/paper_figures/data/cycles-render-manifest.json` and beside each rendered PNG.
The saturated-blue Boil jug has a local ambient material response so its deep interior remains readable under the shared studio lighting.
Commit the scene JSON, rendered PNGs, PNG sidecars, and render manifest.
Do not commit transient renderer logs or the temporary Figma assets.

The Bridge selection behind Figure 2's act panel and Figure 1's lift illustration, with its steps, labels, source hashes, and display crops, is recorded in `scripts/paper_figures/data/trajectories/figure3.json`, which `import_figure3_trajectories.py` writes.
The selected Domino, Fan and Balloons runs and their environment settings are declared in `scripts/paper_figures/export_static_scenes.py`; `--domains` re-exports a subset.
Balloons draws its ceiling, the height at which balloons burst, as a red cap over the chute instead of the environment's translucent plate over the table, and draws the strings of tied balloons, which the environment otherwise draws only when it renders an image.
Update those declarations before re-exporting when new data should replace an existing figure panel.

The Bridge crop constants in `scripts/paper_figures/build_figures.py` reproduce the tighter framing used by the earlier figures.
`scripts/paper_figures/prepare_figma_assets.py` imports those same constants for the editable Figma teaser.

For reproducible appearance, use Blender 4.5.3, Cycles CPU, the same sample count, AgX settings, and the archived scene JSON.
CPU renders on Linux and macOS should look effectively identical, although denoising and floating-point implementation details may prevent byte-identical files.
Changing Blender versions or switching to Metal or CUDA rendering can produce small differences in noise, denoising, and color.

## Figure 2

The compositor draws Figure 2 as `fig2_method`, a six-step loop that follows the method section's notation.
Its code panel is simplified from the program written in the recorded Bridge run (`sandbox/simulator.py`), including the replaced bond rule; its plots are schematic.
Its only raster panel is the recorded state `trajectory_bridge_train_1`: the robot lowers span2, glued on the end that faces the row, toward span1's glued end.
The run's first bond rule tested face-centre proximity and welded blocks mid-descent, 28 mm above their seats; the recorded joint bonded only 25 steps after span2 was seated, which the code panel's revised line reflects.
The earlier illustrated pipeline remains in the [Figma file](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=133-2) but is no longer used by the paper.
Figure 4 quantitative plots are outside this compositor.
The same Figma file contains the editable [Figure 1 teaser](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=117-62).
Generate its eight raster fills with `prepare_figma_assets.py`, then upload every PNG to the node listed in the generated `manifest.json`.

## Re-render with a local GUI

The compositor arranges images; it does not perform GUI captures.
Use the original run's runtime, configuration, primitive actions, and recording/state files.
The replay helpers are `predicators/run/continual_video.py` and `scripts/continual_video.py`.
A local capture driver must select the archived events or restore complete saved states.

The earlier Figure 3 Bridge selection is recorded in `scripts/paper_figures/data/trajectories/figure3.json`; Figure 3 now uses the stripes below, but these frames still supply Figure 2's act panel and Figure 1's lift illustration:

| Row | Run under logs/agent_continual | Level | Within-level steps |
|---|---|---|---|
| Bridge training | bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710 | L01 | 0, 1189, 1290, 1362 |
| Bridge test | bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710 | L02 | 0, 563, 1652, 1702, 1940 |

Steps 1189 (mid-descent) and 1290 (mid-carry) fall inside a skill, so they have no GUI render and are archived by their recorded state index alone.
Step 1189 is drawn from its own camera at 900 by 540 (`ACT_CAMERA` in `export_trajectory_scenes.py`).

Do not reconstruct execution by resetting visible object poses alone.
Preserve attachments, velocities, glue/contact history, and other hidden state.
Verify replayed state, selected event, and terminal outcome against the recordings.
Rendering must not advance physics.
Camera, lighting, antialiasing, and resolution may change, but geometry, actions, and outcomes must not.
Keep the camera fixed within a trajectory.
If exact replay cannot be verified, retain the original or label the replacement as an illustration.

Save improved images under `scripts/paper_figures/figures/sources/local_gui/`.
Add overrides to `scripts/paper_figures/data/figure-render-overrides.json`.
Paths inside that file are relative to `scripts/paper_figures/`.
Example:

```json
{
  "281:107:18": {
    "path": "figures/sources/local_gui/domino_initial.png"
  },
  "trajectory_bridge_test_3": {
    "path": "figures/sources/local_gui/bridge_step1702.png",
    "crop": [250, 224, 820, 670]
  }
}
```

Crop coordinates refer to the replacement image; omit them for an already cropped panel.
Both Bridge rows use the crop `(250, 224, 820, 670)`, recorded in `figure3.json`, so the table sits in the same place in every frame.
Semantic frame names end in their index within the row.
Explicit gallery keys take precedence over semantic names.

| Domain | Initial key | Final key |
|---|---|---|
| Domino | 281:107:18 | 281:107:136 |
| Bridge | 281:321:18 | 281:321:136 |
| Balloons | 281:428:18 | 281:428:136 |
| Boil | 281:0:18 | 281:0:136 |
| Fan | 281:214:18 | 281:214:136 |

The Bridge solved panel of the teaser uses key `304:370:57`.
The teaser's two lift panels, `bridge_pair_predicted` and `bridge_pair_observed`, are Cycles renders of states built from the recorded training level: the robot and the grasped block take the mid-lift pose of step 1276, the glued partner either rises with it or keeps its pre-lift pose of step 1248, and every other object keeps its initial pose.
They are illustrations, not execution frames, and have no GUI counterpart.
Keep a provenance note with runtime commit, renderer, camera matrices, resolution, event, and replay checks.

To refresh the archived Figure 3 selection deliberately, run:
`python scripts/paper_figures/import_figure3_trajectories.py`.
This reads local logs and restores original images; it is not the command for installing GUI replacements.

## Trajectory stripes (Figure 3 and the appendix)

Figure 3 and the appendix figure show one recorded EMPIRIC run per domain as a stripe of five frames, from the run's experiments to the solved test task.
`scripts/paper_figures/data/trajectories/stripes.json` records each stripe's run, frames (level, episode and level step, or a saved model state), captions, and display settings; `MAIN_STRIPES` and `APPENDIX_STRIPES` in `build_figures.py` choose which stripes go where.
Each frame's caption names what the agent does, and the line below it what follows: the observed outcome, or in a dashed model frame the model's prediction.
Each row's `learn_after` and `learned` place a purple bar after that frame, labelled with what the agent learns there: its program $P$ and parameters $\theta$, or $\theta$ alone when the program adds no mechanism (Domino).
The bar takes a slot of its own, so every gap, between two frames or beside the bar, has the same width (`STRIPE_GAP` in `build_figures.py`) and holds an arrow.
The real-robot stripe learns after its two probes, as `_robot_stripe()` in `build_figures.py` states.
Level steps run on across a level's episodes, and a reset costs one step.

Dashed frames show the agent's own model.
Most runs checked their plans with `render=False`, so these frames are re-created with `scripts/paper_figures/render_model_rollout.py`: it loads the run's launch flags, executes the saved simulator version the agent had at the time, applies the parameter values the harness deployed then (the carry rule for unfitted versions included), rebuilds the noisy observations and belief the agent saw at the plan's start, and runs the same plan through the same probe.
Run it on the code the run used: the recorded commit is in the run's `info.log`, and the frozen worktrees or runtime folders at those commits are listed in the table below.
Check each replay against numbers the agent printed before using it.

| Stripe | Run | Model frames (replay code) |
|---|---|---|
| Domino | `domino_high_friction_turn-mb_opus_gate_r1/seed0` | two level-2 probes (`predicators-rebase`, 5d1c857c5) |
| Bridge | `bridge-mb_opus_benchmark_r2/seed3` | the six-dab check (`predicators-empiric-r2-frozen-20260919`, f2ed37aef) |
| Balloons | `balloons-mb_opus_benchmark_r2/seed3` | the gold-then-blue rehearsal (f2ed37aef) |
| Boil | `boil-mb_opus_gate_preflight_two_jug_tight_r1/seed1` | the two-jug rehearsal (5d1c857c5) |
| Fan | `fan_ramp-mb_opus_ramp_skill_repair_r1/seed2` | the level-1 burst replay and the level-2 plan (`logs/fan-ramp-skill-repair-runtime-20260921`, ff11bc76f) |

The replays' outputs, plans and parameters are under `logs/figure_model_rollouts/`.
Then export and render the stripe scenes and rebuild:

```bash
python scripts/paper_figures/export_stripe_scenes.py --preview /tmp/stripe-previews
uv run --no-project --python 3.11 --with bpy==4.5.3 --with pycollada \
  python scripts/paper_figures/render_cycles_scenes.py --samples 48 --threads 8 \
  --scenes trajectory_domino_stripe_0 ...  # every trajectory_*_stripe_* scene
python scripts/paper_figures/build_figures.py --overview --renderer cycles
python scripts/paper_figures/verify_figures.py
```

The exporter records each display adjustment in the scene metadata.
Fan and Balloons states move into the current scene layouts that Figure 1 uses, so positions relative to the platforms and the chute are unchanged.
Balloons shares Figure 1's red chute cap and balloon strings.
Boil liquid is drawn no higher than the jug rim, because the environment lets water rise above the rim before it overflows, which reads as an upturned jug.
The Bridge model frame draws the glue the model remembers as the environment's glue patches.

## Real-world trajectory

The last row of Figure 3 shows the real-robot Fan-Domino cascade run of 2026-09-22 (`exp_20260922_134142`).
Its logs, videos, posterior, and report are in the [shared run folder](https://drive.google.com/drive/folders/1bcFFkaMb1ZKa0p1sK5KuMdQQ92xojnBO).
Download it to `logs/real_robot/fan_domino_drive`, for example with `gdown` file by file, and then run:

```bash
uv run --no-project --with imageio-ffmpeg --with pillow \
  python scripts/paper_figures/import_real_robot_frames.py
```

The importer extracts five gust-camera frames from the left half of the side-by-side episode videos (`casc_explore.mp4` and `casc_test.mp4`), crops them identically, and writes their video hashes, frame indices, measured slides, the test plan's prediction, and the posterior summary to `scripts/paper_figures/data/trajectories/real_fan_domino.json`.
These videos carry no overlays; the `epNN_gust_tracked.mp4` clips show the same camera at twice the resolution but draw the tracker's fitted boxes and angles, so the paper does not use them.
Keep this applicability case study distinct from the simulated baseline comparison.

## Removed legacy tooling

Old preliminary-result builders, manuscript audits, renderer-comparison outputs, and static scene exports are not part of this workflow.
They have been removed from the paper checkout.
The old unfinished results-snapshot script has been replaced for appendix tables by `scripts/plotting/export_paper_results_tables.py`.

## Result tables

Generate the appendix tables from the same selected-run manifest as the paper result figure:

```bash
python scripts/plotting/export_paper_results_tables.py \
  docs/comparisons/figures/paper-results-opus-summary.json \
  /home/ycliang/sim-predicator-paper/data/current-results
```

Use the local paper checkout's path for the last argument when rendering on another machine.
This command preserves the figure's domain variants and selected seeds, and leaves missing results blank instead of substituting an older run.
Regenerate the tables whenever the figure's selected runs change.
