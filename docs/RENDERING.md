# Rendering EMPIRIC paper figures

This is the canonical rendering guide.
Run the commands below from the Predicators checkout.

## Locations

- Tools: `scripts/paper_figures/`.
- Archived image inputs: `scripts/paper_figures/figures/sources/`.
- Scene-to-image mapping and trajectory provenance: `scripts/paper_figures/data/`.
- Paper output: `figures/` in the sibling `sim-predicator-paper` checkout, or the checkout specified by `EMPIRIC_PAPER_ROOT`.
- The paper retains only the table inputs needed by LaTeX under `data/current-results/`.

## Build Figures 1 and 3

Install CairoSVG, Pillow, and PyMuPDF, plus the system Cairo library and DejaVu Sans fonts.
The compositor uses archived images and does not launch physics or agents.

```bash
export EMPIRIC_PAPER_ROOT=/path/to/sim-predicator-paper
python scripts/paper_figures/build_figures.py --overview
python scripts/paper_figures/verify_figures.py
```

Figure 1 uses `fig1_residual`; Figure 3 uses `fig4_environments`.
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
# Run these two exporters in the Predicators environment.
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
  --scenes bridge_start trajectory_bridge_0 bridge_wet_lift \
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

The Figure 3 trajectory selection, steps, labels, source hashes, and display crops are recorded in `scripts/paper_figures/data/trajectories/figure3.json`.
The selected Domino and Fan runs and their environment settings are declared in `scripts/paper_figures/export_static_scenes.py`.
Update those declarations before re-exporting when new data should replace an existing figure panel.

The selected Balloons run predates the current chute placement, so the exporter translates the box and attached assembly to the current chute column and records that visualization-only migration in each scene file.
It also uses the current robot-facing camera and centres the target-height marker inside the chute with clearance from both walls.
The Bridge crop constants in `scripts/paper_figures/build_figures.py` reproduce the tighter framing used by the earlier figures.
`scripts/paper_figures/prepare_figma_assets.py` imports those same constants and adds phase-specific wide crops for Figure 2.

For reproducible appearance, use Blender 4.5.3, Cycles CPU, the same sample count, AgX settings, and the archived scene JSON.
CPU renders on Linux and macOS should look effectively identical, although denoising and floating-point implementation details may prevent byte-identical files.
Changing Blender versions or switching to Metal or CUDA rendering can produce small differences in noise, denoising, and color.

## Figure 2: Figma

The paper uses `fig2_illustrated_figma.png`, exported from the [editable illustrated pipeline](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=133-2).
Generate the image fills with `prepare_figma_assets.py`, then upload every PNG to the node listed in its generated `manifest.json`.
The current phase mapping is Bridge steps 0, 563, 1652, the illustrative wet-lift state, and step 1940 for Explore, Hypothesize, Design experiment, Execute and observe, and Rehearse and solve.
Export Figma node `133:2` at its native 924 by 512 pixels to `figures/fig2_illustrated_figma.png` in the paper checkout.
The compositor does not overwrite Figure 2.
Its thought annotations and code are illustrative, not verbatim agent traces.
Figure 4 quantitative plots are also outside this compositor.
The same Figma file contains the editable [Figure 1 teaser](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=117-62).
The generated Figma asset manifest also maps Figure 1's eight raster fills so its editable source remains synchronized with the paper compositor.

## Re-render with a local GUI

The compositor arranges images; it does not perform GUI captures.
Use the original run's runtime, configuration, primitive actions, and recording/state files.
The replay helpers are `predicators/run/continual_video.py` and `scripts/continual_video.py`.
A local capture driver must select the archived events or restore complete saved states.

The Figure 3 selection is recorded in `scripts/paper_figures/data/trajectories/figure3.json`:

| Domain | Run under logs/agent_continual | Level | Within-level steps |
|---|---|---|---|
| Bridge | bridge-mb_opus_span_transfer_r2/seed0/run_20260916_190710 | L02 | 0, 563, 1652, 1702, 1940 |
| Balloons | balloons-mb_opus_compose_r2/seed0/run_20260917_082044 | L03 | 0, 124, 241, 264, 348 |

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
  "trajectory_bridge_3": {
    "path": "figures/sources/local_gui/bridge_step1702.png",
    "crop": [270, 280, 810, 860]
  }
}
```

Crop coordinates refer to the replacement image; omit them for an already cropped panel.
Original Bridge and Balloons trajectory crops are respectively `(270, 280, 810, 860)` and `(410, 170, 880, 720)`.
The Balloons crop is shifted right relative to the earlier composition so that both chute walls remain visible throughout the trajectory.
Semantic frame names end in indices 0 through 4.
Explicit gallery keys take precedence over semantic names.

| Domain | Initial key | Final key |
|---|---|---|
| Domino | 281:107:18 | 281:107:136 |
| Bridge | 281:321:18 | 281:321:136 |
| Balloons | 281:428:18 | 281:428:136 |
| Boil | 281:0:18 | 281:0:136 |
| Fan | 281:214:18 | 281:214:136 |

Bridge teaser keys are `304:7:57`, `304:87:57`, and `304:370:57`.
These are schematic mechanism illustrations, not consecutive execution frames.
Keep a provenance note with runtime commit, renderer, camera matrices, resolution, event, and replay checks.

To refresh the original archived Figure 3 images deliberately, run:
`python scripts/paper_figures/import_figure3_trajectories.py`.
This reads local logs and restores original images; it is not the command for installing GUI replacements.

## Real-world trajectory

The third row of Figure 3 is currently a LaTeX placeholder in the paper.
Replace it with verified real-robot recordings when available.
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
