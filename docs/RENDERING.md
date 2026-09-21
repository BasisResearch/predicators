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

## Figure 2: Figma

The paper uses `fig2_illustrated_figma.png`, exported from the [editable illustrated pipeline](https://www.figma.com/design/PgS1btsW3SH28tvW52xjrk/EMPIRIC?node-id=133-2).
Re-export that frame after editing it.
The compositor does not overwrite Figure 2.
Its thought annotations and code are illustrative, not verbatim agent traces.
Figure 4 quantitative plots are also outside this compositor.

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
Original Bridge and Balloons trajectory crops are respectively `(270, 280, 810, 860)` and `(290, 170, 760, 720)`.
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
The unfinished results-snapshot script is not a supported figure-generation tool.
