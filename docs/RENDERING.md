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

Figure 1 uses `fig1_residual`, Figure 2 uses `fig2_method`, and Figure 3 uses `fig3_trajectories`.
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

The Figure 3 Bridge selection, steps, labels, source hashes, and display crops are recorded in `scripts/paper_figures/data/trajectories/figure3.json`, which `import_figure3_trajectories.py` writes.
The selected Domino and Fan runs and their environment settings are declared in `scripts/paper_figures/export_static_scenes.py`.
Update those declarations before re-exporting when new data should replace an existing figure panel.

The Bridge crop constants in `scripts/paper_figures/build_figures.py` reproduce the tighter framing used by the earlier figures.
`scripts/paper_figures/prepare_figma_assets.py` imports those same constants for the editable Figma teaser.

For reproducible appearance, use Blender 4.5.3, Cycles CPU, the same sample count, AgX settings, and the archived scene JSON.
CPU renders on Linux and macOS should look effectively identical, although denoising and floating-point implementation details may prevent byte-identical files.
Changing Blender versions or switching to Metal or CUDA rendering can produce small differences in noise, denoising, and color.

## Figure 2

The compositor draws Figure 2 as `fig2_method`, a six-step loop that follows the method section's notation.
Its code panel is simplified from the program written in the recorded Bridge run (`sandbox/simulator.py`), including the replaced bond rule; its plots are schematic.
Its only raster panel is the recorded mid-dip state `trajectory_bridge_train_1`.
The earlier illustrated pipeline remains in the [Figma file](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=133-2) but is no longer used by the paper.
Figure 4 quantitative plots are outside this compositor.
The same Figma file contains the editable [Figure 1 teaser](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=117-62).
Generate its eight raster fills with `prepare_figma_assets.py`, then upload every PNG to the node listed in the generated `manifest.json`.

## Re-render with a local GUI

The compositor arranges images; it does not perform GUI captures.
Use the original run's runtime, configuration, primitive actions, and recording/state files.
The replay helpers are `predicators/run/continual_video.py` and `scripts/continual_video.py`.
A local capture driver must select the archived events or restore complete saved states.

The Figure 3 Bridge selection is recorded in `scripts/paper_figures/data/trajectories/figure3.json`:

| Row | Run under logs/agent_continual | Level | Within-level steps |
|---|---|---|---|
| Bridge training | bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710 | L01 | 0, 122, 1290, 1362 |
| Bridge test | bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710 | L02 | 0, 563, 1652, 1702, 1940 |

Steps 122 (mid-dip) and 1290 (mid-carry) fall inside a skill, so they have no GUI render and are archived by their recorded state index alone.
The between-levels panel summarizes the run's journal (`sandbox/journal.md`): the written glue program, the replay of the six recorded dips, the rejected first bond model, and the rehearsed test plan.

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
    "crop": [270, 280, 810, 860]
  }
}
```

Crop coordinates refer to the replacement image; omit them for an already cropped panel.
The Bridge test-level crop is `(270, 280, 810, 860)`; the training level uses its own crop in `figure3.json`.
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

## Real-world trajectory

The third row of Figure 3 shows the real-robot Fan-Domino cascade run of 2026-09-22 (`exp_20260922_134142`).
Its logs, videos, posterior, and report are in the [shared run folder](https://drive.google.com/drive/folders/1bcFFkaMb1ZKa0p1sK5KuMdQQ92xojnBO).
Download it to `logs/real_robot/fan_domino_drive`, for example with `gdown` file by file, and then run:

```bash
uv run --no-project --with imageio-ffmpeg --with pillow \
  python scripts/paper_figures/import_real_robot_frames.py
```

The importer extracts five frames from the gust camera's tracking videos, crops them identically, and writes their video hashes, frame indices, measured slides, the test plan's prediction, and the posterior summary to `scripts/paper_figures/data/trajectories/real_fan_domino.json`.
The tracking overlays are the tracker's fitted boxes, not predictions.
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
