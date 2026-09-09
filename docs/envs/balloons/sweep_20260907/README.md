# Balloons slides: uniform-sweep snapshot

The slides describe the contact-decoy chute environment used in the September 7, 2026 uniform sweep, with results completed through September 8.
The two standalone HTML copies contain the same 12 slides and embed their images, with no network dependencies.

- Main deck: `docs/slides/balloons_domain_slides.html`.
- Environment overview: `docs/envs/balloons/overview_slides.html`.
- PDF: `docs/slides/balloons_domain_slides.pdf`.

`manifest.json` pins all six balloons scorecard paths, source Git SHAs, checksums, per-level results and image sources.
The screenshots are copied without alteration from MB seed 0's test level in `run_20260907_102820`.
The original September 6 oracle video and JPEGs remain historical assets; they are not used by the updated slides.

The deck reports means over seeds 0-2, counting all two train levels and one test level per seed.
Real environment steps and agent resets include unsuccessful runs; sandbox simulations are excluded.
MB solved 9/9 levels with 481.7 mean steps and 0.67 mean resets; MF solved 8/9 with 495.7 steps and 1.00 reset.
The worked example's alternative hover height is calculated from the source law, not a fresh counterfactual replay.

Rebuild the HTML copies from the repository root:

```bash
python docs/slides/make_balloons_domain_slides.py
```

To regenerate and check the 12-page PDF, run on a compute node with WeasyPrint and PyMuPDF available:

```bash
LD_PRELOAD=/orcd/software/core/001/pkg/miniforge/25.11.0-0/lib/libstdc++.so.6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python docs/slides/make_balloons_domain_slides.py --pdf
```

The PDF check verifies page count, page size and text bounds, and exports page previews under `logs/analysis/balloons_slides_20260908/` for visual inspection.
