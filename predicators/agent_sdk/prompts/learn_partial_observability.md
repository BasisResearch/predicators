# Hidden-state guidance for messages and predicates

<!-- section: message -->
## Partial observability

Some causally important quantities may be absent from the observation
entirely (under no name), possibly several, possibly none. Inspect the
trajectories first to judge whether any hidden process is at work and
which observable features are your window into it; then, if latents
are needed, declare subclass `MODEL_STATE_INIT` and implement `update_model_state`.

<!-- section: predicates -->
### Predicate signature

Classifiers may stay observation-only or take an optional `latent`
kwarg. The latent block is available at refinement time too: the
planner threads it through `state.latent` across search nodes, and
`Predicate.holds` routes it into classifiers that opted in. Be
defensive: at the very first step `state.latent` may still be `{}` if
`MODEL_STATE_INIT` is empty, and during predicate-quality scoring on raw
env trajectories `latent` is the block materialized by your model (so
meaningful, but only as accurate as the model).

```python
# Observation-only (robust to an inaccurate model; preferred when the
# observable carries enough signal):
Predicate("ProcessDone", [widget_type],
          lambda s, objs, latent=None:
              s.get(objs[0], "progress") > 0.5)

# Latent-aware (inherits simulator correctness; defend against
# missing keys at step 0):
Predicate("ProcessDone", [widget_type],
          lambda s, objs, latent=None:
              (latent or {}).get("level", 0.0) >= params["done_thresh"])
```

The kwarg must be named exactly `latent` for the routing to apply.
Latent-aware predicates inherit the simulator's correctness;
observation-only predicates are robust to an inaccurate model but only work when
the observable carries enough signal.

`sim.predicates()` rolls each trajectory through your simulator to
materialize the latent before scoring classifiers, so latent-aware
predicates get a real block there. Use its report to localize failures
(bad model versus bad threshold).
