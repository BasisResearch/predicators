# Continual-protocol play: the contract of the model files

Rendered by `play_prompts.build_model_contract` into the model arm's
system prompt, after "Your model". It describes the residual-simulator
API of `predicators.code_sim_learning` (`apply_rules`,
`apply_rules_with_latent`, `commands.CommandBuffer`, `fit_space.ParamSpec`)
and the `predicates.py` surface of `agent_sdk.tools.predicate_synthesis`,
organised around the kinds of process the ground-truth simulators
model: a rate while a condition holds, a force while a device is on, a
dwell that latches into an attachment, a hidden level with a readout,
and physics the engine already produces. `__RULE_ARGS__` is the rule's
parameter list, chosen by observability; the `hidden_state` and
`predicates_latent` sections render under partial observability only.
Every name in the examples is illustrative.

<!-- section: intro -->
## The model files

What `./simulator.py` and `./predicates.py` must contain and how the
`sim` probe and the harness read them. The names in the examples
(`widget`, `fixture`, `charge`, ...) are illustrative; use this
environment's types and features as the observation and `skills_list`
report them.

<!-- section: simulator -->
## `simulator.py`

Residual dynamics on top of the base simulator. Each step the base
simulator drafts the next state from the current one (robot motion,
grasping, rigid bodies); your rules then run once, in order, on the
current state and write the features the base simulator does not
model.

```python
RESIDUAL_RULES:    List[Callable]          # rule functions, run in order each step
PARAM_SPECS:       List[ParamSpec]         # the learnable parameters the rules read
RESIDUAL_FEATURES: Dict[str, List[str]]    # {type_name: [feature]}: the features the rules own
```

`RESIDUAL_FEATURES` is both the loss scope and the overwrite scope:
only the listed features are scored against the recordings by
`sim.fit()` and `sim.residuals()`, and only they are written over the
base draft in a rollout. List exactly what your rules write. `np` and
`ParamSpec` are pre-injected; import anything else at the top of the
file.

A rule:

```python
def rule(__RULE_ARGS__):
    # state:   the current observation (a State; a hidden quantity is
    #          under no name in it)
    # updates: Dict[Object, Dict[str, float]], accumulated across rules
    # params:  Dict[str, float], one entry per ParamSpec
    updates.setdefault(obj, {})[feat] = new_value   # add to the dict, never replace it
    return updates
```

Rules see the current state only, never the action or the base draft.
A feature that reacts one step after its trigger is a one-step lag in
the data: accept it or model the delay, do not chase it with tighter
conditions.

<!-- section: processes -->
## The kinds of process a rule models

Decide the channel from what the recordings show against the base
simulator (`sim.residuals()`):

1. **A quantity that changes while a condition holds** (a level
   filling, a temperature rising, a charge accumulating, and draining
   when the condition ends): a feature-update rule that loops over the
   objects of the type, gates on the condition and adds a rate.

```python
def filling(__RULE_ARGS__):
    widgets = [o for o in state.data if o.type.name == "widget"]
    fixtures = [o for o in state.data if o.type.name == "fixture"]
    for widget in widgets:
        for fixture in fixtures:
            if state.get(fixture, "is_on") > 0.5 and \
                    at_outlet(state, widget, fixture, params):
                level = state.get(widget, "level") + params["fill_rate"]
                updates.setdefault(widget, {})["level"] = \
                    min(level, params["capacity"])
    return updates
```

2. **A body that moves in the data but is inert in the base replay**
   whenever some condition holds (a device that blows, pulls or pushes).
   The base simulator cannot produce the motion, so the rule commands
   the engine: declare a trailing parameter named `cmds` and emit a
   force, torque or velocity gated on the condition. A command acts
   during the next action and expires, so emit it on every step the
   process is active; the engine resolves contacts, sliding and
   deflection, never rule code. A steady effect is a constant force
   plus contacts, not a kick or a decaying pulse. Never push or
   overwrite a body the base simulator already moves: the two fight
   and the fit absorbs physics error into the rule.

```python
def blowing(__RULE_ARGS__, cmds):
    for device in [o for o in state.data if o.type.name == "device"]:
        if state.get(device, "is_on") > 0.5:
            fx, fy = params["push_force"] * facing_xy(state, device)
            for ball in [o for o in state.data if o.type.name == "ball"]:
                cmds.apply_force(ball, (fx, fy, 0.0))
    return updates
```

   Also `cmds.apply_torque(obj, (tx, ty, tz))`,
   `cmds.set_velocity(obj, linear=(vx, vy, vz), angular=(wx, wy, wz))` and
   `cmds.attach(obj_a, obj_b)`, a fixed joint at the pair's current
   relative pose that exists exactly while some rule keeps emitting
   it. Declaring `cmds` switches fitting to env-in-the-loop rollout
   matching (`sim.fit` says so); list the pose features the commands
   move in `RESIDUAL_FEATURES` (scored, not overwritten).

3. **Two bodies that move as one from some event on** (a latch, a snap
   fit, a magnetized contact, a joint that sets after a dwell). The
   event is a counter or a condition crossing a threshold; the
   consequence is `cmds.attach(a, b)` re-emitted every step from the
   latched decision, which lives in __LATCH_HOME__. Do not emulate
   this by writing the follower's pose from the
   leader's: pose-written followers do not collide, support nothing
   and swing free when carried, so plans validate in the model and
   fail for real.

4. **Motion the base simulator already produces but with the wrong
   size** (a body travels too far, a rate is off): a parameter of the
   engine's physics, not a rule. When this environment reveals tunable
   physics, the system-identification section below says how to
   declare it; never write a rule that fights the engine. An
   environment whose recordings show no residual process at all keeps a
   single no-op rule (`return updates`) and an empty
   `RESIDUAL_FEATURES`.

<!-- section: gates -->
## Writing conditions

- **Geometry through anchored points, with learned offsets.** An
  object's `x, y` is its recorded origin; the point that matters
  physically (an outlet, a contact face, a tool tip) is offset from it
  in the object's local frame and rotates with its orientation. Gate
  on the distance to `origin + R(rot) @ (local_dx, local_dy)` with the
  offset and the distance declared as `ParamSpec`s (an unneeded offset
  fits to zero), not on raw origin distance, which bakes in one
  layout's orientation.
- **Thresholds are parameters, and the data must separate.** Before
  committing a cutoff, bucket the recorded steps by whether the effect
  occurred and check that the two ranges separate by a clear margin; a
  knife-edge cutoff means the quantity is measured from the wrong
  point. Stage one state from each bucket with `sim.reset(task_idx=...,
  mods={...})` and overlay them with `sim.render(label,
  annotations=[...])` to see the offset. A hard comparison against a
  parameter is fine, since the fit sweeps a parameter that carries no
  gradient across its box; a sigmoid gate, `1 / (1 + np.exp(-(x -
  params["thr"]) / width))`, gives it a gradient and fits faster.
- **Every object of a type, never a slot.** Tasks vary the object
  count; a rule that indexes `[0]` ignores the rest. Gather by type and
  loop; the same `params` describe every instance of a type.
- **One gate, declared once.** When a physical condition gates a rule
  and a predicate reads the same condition, declare its parameters
  once and read `params["name"]` from both, so the two stay anchored to
  the same point and the rule's step data fits them.

<!-- section: hidden_state -->
## Hidden state

This environment is partially observable: quantities that drive its
dynamics may be under no name in the observation. Every rule takes
the recurrent signature `rule(state, latent, history, updates,
params)`, the second parameter named exactly `latent`. `latent` is a
dict you own, threaded across steps and mutated in place: the hidden
quantities you infer. `history` is the read-only list of past
`(observation, action)` pairs, newest last, for a condition defined by
a change: a rising edge is "on now and off in `history[-1][0]`".
Declare the initial block:

```python
LATENT_INIT = {}   # a dict, or a zero-arg callable returning one; a ParamSpec value is learnable
```

Shape the latent like the state, object first, keyed by the stable
`obj.name` (never the live `Object`), then feature, so several objects
of a type never share one accumulator; a quantity of a pair (a joint)
is a block keyed by the pair's names, and a global hidden quantity (a
clock) is a top-level scalar. Anything a rule must remember (a
counter, an accumulated level, an irreversible flag) lives in
`latent`; the features written to `updates` are outputs recomputed
from `latent` each step, never read back as state, because the planner
restores states during refinement and only `latent` is threaded across
those jumps. Judge first, from the recordings, whether any latent is
needed: if every observable is explained by observed quantities, leave
`latent` untouched.

Two shapes cover most hidden processes:

```python
PARAM_SPECS = [ParamSpec("dwell", 30, lo=1, hi=200),
               ParamSpec("rate", 0.03, lo=0.0, hi=0.1)]

def joint_sets(state, latent, history, updates, params, cmds):
    # A: a dwell counter that latches; the latch drives an attachment.
    joints = latent.setdefault("joints", {})
    for a, b in candidate_pairs(state, params):
        j = joints.setdefault(f"{a.name}|{b.name}", {"count": 0, "set": False})
        if not j["set"]:
            touching = faces_touching(state, a, b, params)
            j["count"] = j["count"] + 1 if touching else 0
            j["set"] = j["count"] >= params["dwell"]
        if j["set"]:
            cmds.attach(a, b)                # re-emitted every step from the latch
    return updates

def charging(state, latent, history, updates, params):
    # B: a hidden level with a readout; the observable ramps past an onset.
    for widget in [o for o in state.data if o.type.name == "widget"]:
        w = latent.setdefault(widget.name, {"charge": 0.0})
        if powered(state, widget, params):
            w["charge"] += params["rate"]
        glow = (w["charge"] - 0.85) / 0.15
        updates.setdefault(widget, {})["glow"] = max(0.0, min(1.0, glow))
    return updates
```

Choose A for a discrete flip at a variable tick (the threshold fits
from the flip times) and B for an observable that ramps (the rate fits
from the slope); one simulator may use both.

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
    ax, ay = anchor_point(state, fixture, params)   # origin + rotated offset, as in the rule
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
where its rule saturates, or the fit looks fine while a `Wait` on it
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
then only as accurate as the rules, and `sim.predicates()` rolls each
recording through your simulator to fill the latent before scoring.
Prefer an observation-only classifier when the observable carries the
signal.
