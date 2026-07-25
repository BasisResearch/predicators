# Physical system identification pipeline

`sysid_pipeline.png` diagrams the sysID stack end to end: data collection, the fit orchestration in `predicators/code_sim_learning/`, the uncertainty accounting, and the downstream consumers (belief env, capture gate, agent pre-check, exploration ensemble).
Red badges mark the failure modes observed in `run_20260724_232411` (domino_high_friction_turn, seeds 1-2), with the planned remedies in the legend.

Regenerate with:

```bash
PYTHONPATH=. python docs/sysid/make_sysid_pipeline_fig.py
```
