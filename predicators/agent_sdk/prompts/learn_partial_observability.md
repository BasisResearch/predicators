# Partial observability (learn phase)

Appended under `CFG.partially_observable`: the simulator-side
recurrent-rules tutorial (all sim-learning arms), the short first-
message note, and the predicate-side latent guidance (predicate
invention arms only).

<!-- section: rules -->
## Recurrent rules (partial observability)

The observation may omit causally important quantities: several, one,
or none. Anything omitted is absent from the state entirely (under no
name, not even as a placeholder), so you cannot read it and must infer
its existence and dynamics from how the observable features evolve.
Inspect the trajectories first to judge how many latents you need: a
feature that drifts or ramps with no visible observed driver is likely
downstream of an accumulating latent; if every observable is explained
by other observed quantities, you need no latent at all, so keep the
5-argument signature and leave `latent` untouched. A common case is a
hidden continuous quantity surfaced only through a derived observable
that ramps once the latent crosses a threshold.

Model the hidden state explicitly: each state you predict is one
sample of an augmented state, the observable features plus the latent
dimensions you infer (a free-form dict such as `{"level": 0.73}` or
`{"count": 22}`). Rules read and advance that latent through the
`latent` argument of the 5-argument signature. Declare the initial
latent block:

```python
LATENT_INIT = {"level": 0.0, "count": 0}
# OR a zero-arg callable returning such a dict.
# Use ParamSpec("name", ...) values to make an init value learnable.
```

### Structure the latent like the state (per object)

A hidden quantity almost always belongs to an individual object: a
vessel's hidden `heat` is another feature of that vessel that happens
to be unobserved. Shape the latent like `data`, object first and then
feature, so that `latent[jug.name]["heat"]` reads in parallel with
`observation.get(jug, "water_volume")`. With several same-type objects
a flat `{"heat": 0.0}` collapses them into one shared accumulator,
which is wrong, exactly as rules must loop over every object rather
than indexing `[0]`.

```python
LATENT_INIT = {}          # {jug_name: {"heat": value}}, filled lazily

def heat_rule(observation, latent, history, updates, params):
    jugs = [o for o in observation.data if o.type.name == "jug"]
    for jug in jugs:
        jl = latent.setdefault(jug.name, {})    # this jug's hidden dims
        h = jl.get("heat", 0.0)
        if on_active_burner(observation, jug, params):
            h += 1.0
        jl["heat"] = h
        updates.setdefault(jug, {})["bubbling_level"] = readout(h, params)
    return updates
```

Two deliberate differences from `data`: (1) key by the stable string
`obj.name`, not the live `Object` (the latent is deep-copied and
reconstructed at every search node, so a live key risks identity
mismatch); (2) keep it a free-form JSON-like nest of dicts and numbers
with no registered schema. A genuinely global hidden quantity (a world
clock, an ambient temperature) stays a top-level scalar. Top-level
scalar entries may be `ParamSpec`s to make their initial value
learnable; seed each per-object slot lazily from such a shared init.

### Two synthesis patterns (pick per latent)

Pattern A, counter plus threshold: carry a step counter and flip the
observable when it crosses a learnable threshold. Same statistical
shape as a delayed discrete event.

```python
PARAM_SPECS = [ParamSpec("delay", init_value=33, lo=1, hi=200)]
LATENT_INIT = {"count": 0}

def count_rule(observation, latent, history, updates, params):
    active = is_widget_at_fixture(observation)  # observable check
    fixture_on = observation.get(fixture, "is_on") > 0.5
    if active and fixture_on:
        latent["count"] += 1
    else:
        latent["count"] = 0
    fired = latent["count"] >= params["delay"]
    updates[widget]["progress"] = 1.0 if fired else 0.0
    return updates
```

Pattern B, physical latent plus readout: carry an estimate of the
unobserved quantity and map it through a (typically monotone) function
to predict the observable. Higher resolution: the observable co-varies
smoothly with the latent before the symbolic "done" point.

```python
PARAM_SPECS = [ParamSpec("rate", init_value=0.03, lo=0.0, hi=0.1)]
LATENT_INIT = {"level": 0.0}

def level_rule(observation, latent, history, updates, params):
    active = is_widget_at_fixture(observation)
    fixture_on = observation.get(fixture, "is_on") > 0.5
    if active and fixture_on:
        latent["level"] += params["rate"]
    lvl = latent["level"]
    # monotone readout: ramps from 0 once `lvl` passes an onset (~0.85)
    updates[widget]["progress"] = max(0.0, min(1.0, (lvl - 0.85) / 0.15))
    return updates
```

How to choose: a smooth ramp across many steps in the derived
observable calls for Pattern B (a partial-progress signal whose rate is
identifiable from a single trajectory by slope fit); a clean discrete
flip at a variable tick may only need Pattern A (one learnable
threshold calibrated from the flip-time distribution across
trajectories). Different latents may use different patterns within one
simulator.

### Keep carried state in `latent`, not in emitted observables

Anything a rule must remember across steps (a counter, an accumulated
level, an irreversible "done" flag) belongs in `latent`. The
observables you write to `updates` are outputs only: recompute them
from `latent` and base-owned inputs each step, and never read one of
your own emitted features back in as state. The planner resets and
replays states during refinement, and only `latent` is guaranteed to
be threaded across those jumps, so a rule that latches on its own
output can pass a step-by-step rollout and break at refinement time.
Reading features the base sim owns (positions, `is_on`, `is_held`) is
fine; those are restored faithfully.

<!-- section: message -->
## Partial observability

Some causally important quantities may be absent from the observation
entirely (under no name), possibly several, possibly none. Inspect the
trajectories first to judge whether any hidden process is at work and
which observable features are your window into it; then, if latents
are needed, model them in `latent` with Pattern A or Pattern B (or a
mix).

<!-- section: predicates -->
### Predicate signature

Classifiers may stay observation-only or take an optional `latent`
kwarg. The latent block is available at refinement time too: the
planner threads it through `state.latent` across search nodes, and
`Predicate.holds` routes it into classifiers that opted in. Be
defensive: at the very first step `state.latent` may still be `{}` if
`LATENT_INIT` is empty, and during predicate-quality scoring on raw
env trajectories `latent` is the block materialized by your rules (so
meaningful, but only as accurate as the rules).

```python
# Observation-only (robust to bad rule chains; preferred when the
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
observation-only predicates are robust to bad rules but only work when
the observable carries enough signal.

`sim.predicates()` rolls each trajectory through your simulator to
materialize the latent before scoring classifiers, so latent-aware
predicates get a real block there. Use its report to localize failures
(bad rule chain versus bad threshold).
