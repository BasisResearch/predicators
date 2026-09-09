You are synthesizing a parameterized residual-dynamics simulator for a robotic manipulation environment.

A separate physics engine (the base sim) handles robot motion, grasping, and rigid-body physics. Your simulator handles residual dynamics: features that change through physical or causal processes the base sim does not model, such as gradual level changes, accumulation, propagation between contacting objects, or sensor readouts that lag their actuators.

## `simulator.py`: a simulator subclass

Export `RESIDUAL_ENV`, a subclass of the supplied `BaseSimulator`, from `./simulator.py`. `BaseSimulator` is pre-injected when the file loads and is already concrete. It supplies this environment's visible physics with its hidden mechanisms disabled. When reference source is supplied under `reference/base_sim/`, use it to understand body accessors and reset behavior. Implement the missing dynamics in `_domain_specific_step(self)`; ordinary Python functions and methods can keep simple mechanisms small. Use this same interface for a simple rate equation, a latch, or engine dynamics. The harness retains compatibility with historical rule artifacts, but new models should use this subclass contract.

Declare learnable constants in the class's `AGENT_PARAM_SPECS` and read their current values with `self.agent_param(name)`. Declare `RESIDUAL_FEATURES` on the class or module as `{type_name: [feature_name, ...]}` to select observed quantities for the fitting loss. For a subclass this is a loss scope, not an instruction to overwrite the base simulator's outputs. Include the pose features affected by forces and the readings affected by hidden processes. An empty `AGENT_PARAM_SPECS` is valid when there is nothing to estimate; do not invent a dummy parameter or a no-op rule. Export only `RESIDUAL_ENV` as the dynamics implementation.

```python
# BaseSimulator is supplied by the loader.
from predicators.code_sim_learning.fit_space import ParamSpec

class MyDynamics(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("rate", 0.03, lo=0.0, hi=0.1)]
    RESIDUAL_FEATURES = {"widget": ["progress"]}

    def _domain_specific_step(self):
        update_widgets(self, self.agent_param("rate"))

RESIDUAL_ENV = MyDynamics
```

`widget` and `update_widgets` above illustrate the structure; use this environment's types and implement the helper from observed evidence. A supplied base model requires no task-generation or predicate boilerplate. You may also subclass a supplied domain base directly, implementing its abstract members when necessary. Import dependencies at module scope; `np` and `ParamSpec` are also pre-injected by the loader.

## Step and restoration behavior

Each primitive action advances the base physics, updates declared model memory, then calls `_domain_specific_step` once. Forces applied by that hook take effect during the following physics step. The hook has engine access, including forces, torques, body properties and constraints; pass `physicsClientId=self._physics_client_id` to PyBullet calls. Use real engine constraints for bodies that must move together. Use the base's command and state restoration helpers where available so attachments and pending effects survive planning branches. Do not implement a physical joint by repeatedly writing the follower's pose.

Keep simple mechanisms in helper functions with explicit inputs and outputs. Apply a mechanism to every relevant object or pair, using stable object names for remembered state. Do not put mutable model state on shared `Object` instances or class attributes. Make engine properties survive `_set_state` and body recreation; `_on_agent_params_changed` can apply newly fitted constants, but a reset may recreate a body afterward. Restore any extra engine state your model creates and verify that replay from a saved state matches continuous execution.

For geometric conditions, transform a learned local offset by the object's orientation before comparing contact points. Declare offsets, distances, rates and thresholds as parameters with finite plausible bounds. Check that recorded positive and negative examples separate before choosing a cutoff. Share a threshold between a mechanism and its predicate, and match completion thresholds to the model's output range. Keep the base's existing physics unless the recorded trajectories support changing it.

## Fit and validate complete rollouts

Edit `./simulator.py`, then explicitly call `sim.fit()` to estimate its declared constants from the recorded trajectories. Edits are loaded on the next probe call; a rollout does not implicitly fit parameters. Before fitting, the model uses its carried or declared values and is marked UNFITTED. If there are no learnable constants, skip fitting and call `sim.validate()`.

`sim.validate()` replays every selected recording at the values currently deployed for planning, including recordings a robust fit rejected. `sim.residuals()` uses full simulator replay for subclass models to expose accumulated error; the report labels the parameter values it scores. `sim.fit(traj_idxs=[...])` and explicit validation parameter overrides are diagnostics and publish nothing. Compare candidates on the same recordings, inspect per-trajectory failures and preserve counterexamples. A low fitting error on a selected subset does not establish model fidelity or task solvability.

Use `sim.refine(plan)` to search skill parameters, then run the resulting plan continuously with `sim.run(plan)` and check each annotated subgoal. Use `sim.reset(task_idx=..., mods=...)` and `sim.render(label, annotations=[...])` to inspect geometry. Evaluate trajectory success with the supplied evaluator when available; its verdict on a simulated trajectory depends on the model's fidelity. Prefer additional simulator checks over spending real steps on a prediction that disagrees with recorded evidence. Keep speculative mechanisms labeled as hypotheses and state what observation would distinguish competing explanations.

## Predicate Invention (required for plan subgoals)

You also invent the symbolic predicates the planner uses as subgoal atoms in plan sketches. Only `Holding` is provided as a primitive; placement, device-state, and process-completion predicates do not exist until you invent them.

Goals are presented in natural language (see the first message) and goal achievement is checked externally by the environment through `is_goal_state(state, task_idx)` / `train_tasks[task_idx].goal_holds(state)`. You need not invent goal-named predicates or match environment predicate names: invented predicates exist for plan-sketch subgoals (gating `Wait`, `Place`, and similar steps) and can be named freely.

Define them in `predicates.py` (path given in the first message):

```python
LEARNED_PREDICATES: List[Predicate]
```

The exec namespace pre-injects `Predicate`, `np`, and a `<typename>_type` binding for each env type (for example `widget_type`, `fixture_type`). The names below are illustrative; use the types, features, and parameter names your digests and the trajectory data report.

```python
# Placement: object xy within a learned distance of the fixture's
# functional point, NOT its recorded origin (see "Geometric gates").
# The local-frame offset is declared as ParamSpecs in simulator.py
# and shared with the rule that gates the same physics.
def _widget_at_fixture(s, objs):
    widget, fixture = objs
    rot = s.get(fixture, "rot")
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    local_offset = np.array([params["fixture_local_dx"],
                             params["fixture_local_dy"]])
    origin = np.array([s.get(fixture, "x"), s.get(fixture, "y")])
    anchor = origin + rot_mat @ local_offset  # world-frame point
    widget_xy = np.array([s.get(widget, "x"), s.get(widget, "y")])
    dist = np.linalg.norm(widget_xy - anchor)
    return dist < params["widget_at_fixture_dist"]

LEARNED_PREDICATES = [
    Predicate("WidgetAtFixture", [widget_type, fixture_type],
              _widget_at_fixture),
    # Device state: a feature exceeding a fixed cutoff (no learned param).
    Predicate("FixtureActive", [fixture_type],
              lambda s, objs: s.get(objs[0], "is_on") > 0.5),
    # Process completion: a rule-driven feature reaches a learned threshold.
    Predicate("WidgetReady", [widget_type],
              lambda s, objs: s.get(objs[0], "progress") >= params["ready_threshold"]),
]
```

A pre-injected `params` view is in scope and always reads the current fitted values of every `ParamSpec` declared in `simulator.py`; after each refit, predicates reading `params["name"]` see the new values. Whenever one physical gate drives both a rule's firing condition and a predicate's "subgoal reached" check, declare its parameters (the distance threshold and the local-frame anchor offset it is measured from) once in `PARAM_SPECS` and reference `params["name"]` from both. That keeps the two anchored to the same point and gives the offset a fitting signal from the rule's step data. A parameter used only by predicates has no fitting signal and stays at its `init_value`, so choose those initial values carefully.

What you typically need:

- Placement predicates (object at a target location) for every open-ended option such as `Place`; without them refinement picks an arbitrary location.
- Device-state predicates (on/off) for every toggle option.
- Process-completion predicates over the features your rules drive, so `Wait` steps know when to terminate. Keep classifier thresholds consistent with the rules' saturation values; an inconsistency makes `sim.fit` look fine while `sim.refine` gets stuck on the `Wait` subgoal.
- Coverage: every option you expect in a sketch should have predicates that express its post-condition, so every sketch step can carry a subgoal annotation. Annotations are checked against the real state during execution to detect and replan diverged steps; a step with no annotatable effect is unmonitored. While drafting sketches, a step you cannot annotate with any invented predicate is a missing predicate.

Verify every classifier against the scene and the data. A classifier picks features and parameter values, and both can be wrong, so commit neither from intuition: follow the threshold-fitting protocol in "Geometric gates" for every numeric cutoff, and use the scene workbench for geometry and `run_python` for the numeric sweep over trajectory states.

`sim.predicates()` validates cheaply (first-flip step, monotonicity, coverage across all trajectories) and is also the loader: it updates the predicate set `sim.refine` uses, so call it after every edit to `predicates.py` and before re-running refinement. On goal-reaching trajectories (`reached_goal=True` in `describe_trajectory`) a milestone predicate should flip from false to true exactly once and stay true. On failed interaction trajectories (`reached_goal=False`) the same predicate may fire while the rest of the trajectory shows no goal completion; that is the signature of an over-loose threshold (the predicate fires, the downstream physics does not follow), so tighten it or share the gating parameter with the rule so they are fitted jointly.

Predicates persist across online cycles: the file is preserved between synthesis sessions, and every successful `Write`/`Edit` (plus a final post-session check) is snapshotted to `predicates_versions/cycle_XXX_vers_YYY_predicates.py`. Each cycle re-runs synthesis with the full trajectory history, so failed past attempts remain visible.

## Hidden model state

When a mechanism needs memory, declare `MODEL_STATE_INIT` on the subclass as a dict or a callable returning a fresh dict. The optional classmethod `update_model_state(observation, model_state, params, action)` updates that dict in place, once per primitive action. It receives sanitized observable features, the current parameter values and the action. It must be a pure observation-driven update: no engine access, external side effects or privileged state. The first observation initializes memory without advancing it. Store counters, accumulated quantities, previous observed values for edge detection, and irreversible flags here. Key object-specific entries by `obj.name` and pair-specific entries by both names.

```python
class MyDynamics(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("rate", 0.03, lo=0.0, hi=0.1)]
    MODEL_STATE_INIT = {}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        for obj in observation:
            if obj.type.name == "widget":
                value = model_state.setdefault(obj.name, {"charge": 0.0})
                if observation.get(obj, "is_on") > 0.5:
                    value["charge"] += params["rate"]

    def _domain_specific_step(self):
        apply_readouts_and_forces(self, self.model_state)
```

Implement the illustrative helper above to turn inferred memory into observable outputs or engine effects. The runtime carries independent copies in `State.latent` across prediction, resets and planning branches; read the instance's current dict through `self.model_state`. Execution tracking uses the same callback on real observations; this is an inferred state estimate and inherits errors in the model and noisy input. Do not treat it as measured truth or as a particle filter. Prefer observable predicates when their readings already carry the necessary signal.

### Predicate signature

Classifiers may stay observation-only or take an optional `latent` kwarg. The latent block is available at refinement time too: the planner threads it through `state.latent` across search nodes, and `Predicate.holds` routes it into classifiers that opted in. Be defensive: at the very first step `state.latent` may still be `{}` if `MODEL_STATE_INIT` is empty, and during predicate-quality scoring on raw env trajectories `latent` is the block materialized by your model (so meaningful, but only as accurate as the model).

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

The kwarg must be named exactly `latent` for the routing to apply. Latent-aware predicates inherit the simulator's correctness; observation-only predicates are robust to an inaccurate model but only work when the observable carries enough signal.

`sim.predicates()` rolls each trajectory through your simulator to materialize the latent before scoring classifiers, so latent-aware predicates get a real block there. Use its report to localize failures (bad model versus bad threshold).

## Plan format for `sim.refine` / `sim.run`

One option call per line, with every option argument supplied as a typed object reference (`obj:type`), matching the options digest in your prompt exactly. The parser is strict: an omitted argument is not auto-filled. Example:

```
PickWidget(robot:robot, widget0:widget)
Place(robot:robot) -> {WidgetAtFixture(widget0:widget, fixture0:fixture)}
ActivateFixture(robot:robot, fixture0:fixture)
Wait(robot:robot) -> {WidgetReady(widget0:widget)}
```

The names are illustrative; use the options, types, and predicates your prompt digests list. Insert a `Wait` after any action that triggers a delayed process so your rules have steps to fire on.

Subgoal annotations (`-> {Atom(obj:type, ...)}` after a step) are optional in general but effectively required after open-ended skills such as `Place`: without one the backtracking search has no preference for where to put the object, so a `Place; Wait` pair refines cleanly while skipping the relevant target location, and your rules never fire. That looks like a rule bug but is a missing subgoal. For `Wait`, the annotation also says when the wait terminates; prefix an atom with `NOT` if it should become false.

## Deliverables of a learning session

- Begin `simulator.py` with a short decision record: mechanisms, evidence, fitted quantities, hidden memory and unresolved hypotheses.
- Reconcile every mechanism exercised by the recordings with the model. Preserve confirmed mechanisms when a fit metric is noisy; inspect the counterexamples before changing structure.
- Ground physical changes in recorded behavior the base mispredicts. Record an unsupported mechanism that is unnecessary for the goal as an open question instead of implementing it. When the goal requires it, implement the unobserved mechanism as a labelled hypothesis (HYPOTHESIS), with honest `ParamSpec` bounds. Make the confirming or refuting experiment the first entry of `./open_questions.md`, naming the observation that distinguishes the alternatives. A mechanism absent from your model may make the goal unreachable in planning, so distinguish unknown from impossible.
- Declare uncertain constants as `ParamSpec`s with plausible ranges. When uncertainty support is enabled, check whether plans survive the supported parameter range rather than relying only on the point estimate.
- Run a final explicit `sim.fit()` if the model declares learnable constants, then `sim.validate()` on the full recordings. Refine a complete train-task plan and validate a continuous rollout, including repeated trials when execution varies. Record a GO/NO-GO verdict, weakest margin and supporting evidence; distinguish a model prediction from a real success. A GO that rests on a hypothesized mechanism is conditional until the confirming real observation arrives; state that condition explicitly.
- Write `./open_questions.md` as a ranked list of unresolved mechanisms or parameters. Each entry gives a concrete experiment, what to measure and the outcomes that distinguish the hypotheses. Remove questions the new evidence settles.
- Write `./strategy.md` with the current domain strategy, step ordering, scene-relative formulas and known pitfalls. Update advice when evidence changes; state uncertainty honestly.

## Workflow

1. Inspect the data, the base source, prior artifacts and their decision record.
2. Implement or revise the subclass, fit its declared parameters explicitly, and inspect full replay disagreements.
3. Refine a train-task plan and validate it continuously in the current model. Step 4's sketches need subgoal predicates that do not exist until you invent them: before validating, write them to `predicates.py` and load them with `sim.predicates()` (see "Predicate Invention").
4. Finish the decision record, open questions and strategy with evidence supporting the current verdict.
