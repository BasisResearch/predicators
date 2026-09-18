# Model-file notes for continual play

The model reference follows the workflow and workbench API.
Examples are illustrative; use the current environment's types and features.

<!-- section: intro -->
## Model API reference

Write dynamics in `./simulator.py` and optional monitoring predicates in `./predicates.py`.
Use observed evidence to distinguish parameter errors from missing mechanisms; do not encode an unexplained task answer.
The example names below are placeholders for this environment's types and features.

<!-- section: paramspec -->
## Parameter declarations

```python
ParamSpec(name, init_value, lo=None, hi=None, scale="linear", discrete=False)
```

Declare learnable constants in `AGENT_PARAM_SPECS` with finite, plausible bounds.
Use `scale="log"` for positive multiplicative scales, with a strictly positive lower bound; use `discrete=True` for integer choices or counts.
A parameter needs an effect on scored recorded features to be identifiable.
Values used only by predicates stay at their initial values unless set explicitly.

<!-- section: observation_noise -->
## Observation noise and the fit

Do not smooth or filter the data before `sim.fit`; retain the raw recorded features.
The fit accounts for the declared observation channel and model noise floor.
Inspect residuals relative to that noise model and the report's units.
A rejected fit alone does not identify whether the cause is model structure, parameter values, starting-state uncertainty, or a fitting limitation.
A rollout starts from an uncertain observation or belief estimate; use recorded transitions to constrain effects too small to identify from one frame.

<!-- section: observation_noise_declared -->
## Observation noise and the declared values

Retain the raw recorded features; the declared observation channel and model noise floor set the scale on which residuals are judged.
Inspect residuals relative to that noise model and the report's units.
A mismatch alone does not identify whether the cause is model structure, declared values, starting-state uncertainty, or a limitation of an estimator you wrote.
A rollout starts from an uncertain observation or belief estimate; use recorded transitions to constrain effects too small to identify from one frame.

<!-- section: predicates -->
## `predicates.py`

Export `LEARNED_PREDICATES`, a list of `Predicate` objects.
The loader supplies `Predicate`, `np`, `<typename>_type` for each environment type, and `params`, a live view of model parameter values.
Classifiers receive a state and their bound objects and return a boolean.

```python
LEARNED_PREDICATES = [
    Predicate("Ready", [widget_type],
              lambda state, objs:
              state.get(objs[0], "glow") >= params["ready_glow"]),
]
```

Define predicates for outcomes you rely on: they support skill expectations, divergence checks, and `Wait` targets.
Share a physical threshold with its mechanism and keep completion thresholds reachable within the model's output range.
Call `sim.predicates()` after edits to load the definitions and inspect whether each grounding ever holds, changes, or latches in the recordings.
Supplied environment predicates and invented predicates remain distinct even if they have the same name.

<!-- section: predicates_latent -->
A classifier can accept a keyword argument named exactly `latent` to read inferred model memory, for example `lambda state, objs, latent=None: (latent or {}).get(objs[0].name, {}).get("charge", 0.0) >= params["done"]`.
`sim.predicates()` reconstructs that memory over recordings before scoring such classifiers.
Treat their output as model-dependent; prefer an observable classifier when its readings already carry the needed signal.

<!-- section: no_harness_fitting -->
### No harness parameter fitting

The harness estimates nothing in this run: `sim.fit`, fitted residuals, and automatic parameter sweeps are disabled, and the deployed model uses each declaration's `init_value` and `[lo, hi]` exactly as written.
Set and revise those declarations yourself from recorded experience.
You may estimate values in your own sandbox code by any method, from qualitative checks against recordings and model rollouts to fits you write against `trajectories`; the result only takes effect once you write it into the declaration.
Uncertainty-aware planning over your declared ranges remains available.

<!-- section: intro_supplied -->
## Predicate API reference

The dynamics model is supplied and fixed; it runs inside `sim` and is not exposed as source.
Write optional monitoring predicates in `./predicates.py`.
