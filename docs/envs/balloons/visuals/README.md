# Chute and hatch illustrations

The [overview slides](../overview_slides.html) embed the images and videos and can be shared as a single HTML file.
The Hatch navigation button opens the new visual section.
The PDF uses each video's final frame.
The [visual refinement notes](../visual-refinement.md) explain the ropes, clip colours, target outline and smaller ceiling canopy.

| Asset | Content |
|---|---|
| [Chute](chute-initial.png) | Current chute geometry, rendered for explanation rather than a scored run |
| [Hatch](hatch-initial.png) | Audited seed 5 test task before release |
| [Jam video](hatch-jam.mp4) | Red then gold: sustained support below the hatch, 65 primitive actions |
| [Passage video](hatch-pass.mp4) | Gold then red: evaluator win, 79 primitive actions |
| [Height plot](hatch-height.png) | Measured box-center height along both replays; also available as [SVG](hatch-height.svg) |
| [Manifest](manifest.json) | Source commit, initial-state hash, per-step measurements and mechanical outcomes |

These videos replay fixed Release sequences from the existing audited initial state in `logs/balloons_followup_20260909/hatch-audit-seed5/test0-initial.pkl`.
Their status, step counts, contact support and final heights were checked against that audit's `report.json`.
The refined videos play back the original primitive-action arrays from the two `*-actions.npz` files directly, avoiding controller-version differences when rebuilding the illustrations.
Per-run action hashes are included in the manifest.
The jam's winning counterfactual without panel collisions is recorded in that audit; the videos themselves retain the normal panel collisions.
The native replay outcome therefore leaves the counterfactual field unset (`wall_free_won=false`), rather than claiming it was rerun while rendering.
Neither video is an MB, MF or continual-oracle run, and their step counts are not agent sample-efficiency measurements.

The scene uses position observation noise of 0.01 m and orientation observation noise of 0.02 rad.
Rendering shows the underlying physical scene; noise affects observations and is not painted into the rendered frames.
The fixed mechanical sequence does not choose actions from noisy observations.
Playback samples every second primitive action at eight frames per second, with a one-second initial hold and a two-second final hold.
This is an explanatory playback speed, not a wall-clock recording.

The original chute walls and hatch panels collide with the box only.
The balloons and robot pass through them.
Balloon bursting uses a separate ceiling-height threshold.
The canopy depicts that threshold over the payload column, with the same height and thickness as before.
The rendered ropes illustrate the existing rigid attachments and appear after release; they do not add flexible-rope dynamics.
Colour-matched clip sliders identify which balloon each switch controls.
The green outline marks the goal's box-centre height range.

To reproduce the updated visuals, run [render_visuals.py](../render_visuals.py) on a compute node with the source commit in [manifest.json](manifest.json) on `PYTHONPATH` and the saved audit input available.
The refined renderer is in `/home/ycliang/predicators-balloons-visuals-r1` on branch `balloons-visual-refinement`.
Run the renderer from that checkout so the manifest records the imported environment's commit.
Run [the slide generator](../../../slides/make_balloons_domain_slides.py) with `--pdf` from the main checkout to rebuild both HTML copies and the PDF.
The original render job was `22393188`, using source `71f0dbe20` before the visual refinement.
The refined replay job was `22406226`, using source `a698a7775`; both action arrays matched exactly and both box-height/pitch traces matched the originals to absolute tolerance `1e-7`.
