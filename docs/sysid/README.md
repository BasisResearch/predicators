# Physical system identification pipeline

`sysid_pipeline_slides.html` is a self-contained reveal.js deck explaining the sysID stack end to end: why it exists, every pipeline stage with the implementing module, the measured failure modes of run_20260724_232411, the stage-1 honesty fixes (commits `8d32ab853` + `6cc812a99`), honest limits, and the stage-2 roadmap.
Open it in a browser; press S for speaker notes.

`sysid_pipeline.png` is the one-page diagram of the same pipeline (data collection, fit orchestration in `predicators/code_sim_learning/`, uncertainty accounting, consumers), with red badges on the observed failure modes.

`sysid_landscapes.png` shows measured replay-SSE landscapes for four recordings at the true friction: which interaction data identifies the parameter (slide-rich pushes) and which cannot (pure topples with deterministic chaos spikes, carry motion).

Regenerate the figures with:

```bash
PYTHONPATH=. python docs/sysid/make_sysid_pipeline_fig.py
PYTHONPATH=. python docs/sysid/make_sysid_landscape_fig.py
```
