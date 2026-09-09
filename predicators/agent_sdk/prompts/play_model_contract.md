# Model-file notes shared by continual play prompts

<!-- section: intro -->
## The model files

What `./simulator.py` and `./predicates.py` must contain and how the
`sim` probe and the harness read them. The names in the examples
(`widget`, `fixture`, `charge`, ...) are illustrative; use this
environment's types and features as the observation and `skills_list`
report them.

<!-- section: paramspec -->
## `ParamSpec`

```python
ParamSpec(name, init_value, lo=None, hi=None, scale="linear", discrete=False)
```

Declare a finite box for every parameter: it shapes the prior, the
sweep and the validation ensemble. `lo=0.0` for rates and distances;
`scale="log"` for a positive scale-like quantity whose effect is
multiplicative; `discrete=True` for an index or a count the rule
rounds (a wiring slot), which the fit then treats as a choice, not a
knob.

<!-- section: observation_noise -->
## Observation noise and the fit

The declared channel (__NOISE_LINE__) is part of the fit's likelihood:
every residual is scaled so that the observation noise on its feature
counts as noise, not as model error.

- Do not smooth or filter the data before `sim.fit`. The fit weights
  the noise itself, and smoothing removes real motion with it.
- The report's RMS thresholds are in units of the total noise (the
  model floor plus the observation sigma). A fit whose residuals sit
  inside that floor is at the floor, and a refusal means the model is
  wrong, not that the data is noisy.
- `sim.reset(current=True)` starts a rollout from the observed frame,
  which is itself one noisy draw. A parameter whose effect over the
  rollout is within a sigma of the start is not identifiable from one
  frame; look for it in the recorded data instead.

<!-- section: predicates -->
## `predicates.py`

```python
LEARNED_PREDICATES: List[Predicate]
```

The file runs with `Predicate`, `np`, a `<typename>_type` binding per
type of this environment, and `params`, a live view of the current
fitted value of every `ParamSpec` in `simulator.py`. A classifier takes
a state and the objects bound to the predicate's argument types and
returns a bool, reading observable features.

```python
def _at_fixture(state, objs):
    widget, fixture = objs
    ax, ay = anchor_point(state, fixture, params)   # origin + rotated offset, as in the mechanism
    dist = np.hypot(state.get(widget, "x") - ax, state.get(widget, "y") - ay)
    return dist < params["at_dist"]

LEARNED_PREDICATES = [
    Predicate("AtFixture", [widget_type, fixture_type], _at_fixture),
    Predicate("Ready", [widget_type],
              lambda state, objs: state.get(objs[0], "glow") >= params["ready_glow"]),
]
```

Under the protocol your predicates are the atoms of every observation
(the environment's own atoms are listed separately), the vocabulary of
the expectations you annotate on a skill invocation or a plan line (a
divergence is a missing or an unexpected atom of yours on the real
state after the skill), and the targets a `Wait` terminates on. A step
you cannot annotate is unmonitored: an outcome you rely on deserves a
predicate. Keep a completion predicate's threshold consistent with
where its model saturates, or the fit looks fine while a `Wait` on it
never ends. A parameter only predicates read has no fitting signal and
stays at its `init_value`.

`sim.predicates()` is the loader and the check: it reloads the file,
installs the set for the observation, the rollouts, the expectations
and the divergence checks, and reports per predicate and grounding,
over the recorded episodes, whether it ever held, how often it flipped
and whether it flipped once and stayed true. Call it after every edit.

<!-- section: predicates_latent -->
A classifier may take an optional kwarg named exactly `latent` to read
the model's hidden state, `lambda state, objs, latent=None: (latent or
{}).get(objs[0].name, {}).get("charge", 0.0) >= params["done"]`; it is
then only as accurate as the model, and `sim.predicates()` rolls each
recording through your simulator to fill the latent before scoring.
Prefer an observation-only classifier when the observable carries the
signal.
