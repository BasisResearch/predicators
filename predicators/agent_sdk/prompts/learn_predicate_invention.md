# Predicate invention (learn phase)

Appended to the synthesis system prompt and first message by
`AgentSimPredicateInventionApproach`.

<!-- section: system -->
## Predicate Invention (required for plan subgoals)

You also invent the symbolic predicates the planner uses as subgoal
atoms in plan sketches. Only `Holding` is provided as a primitive;
placement, device-state, and process-completion predicates do not
exist until you invent them.

Goals are presented in natural language (see the first message) and
goal achievement is checked externally by the environment through
`is_goal_state(state, task_idx)` / `train_tasks[task_idx].goal_holds(state)`.
You need not invent goal-named predicates or match environment
predicate names: invented predicates exist for plan-sketch subgoals
(gating `Wait`, `Place`, and similar steps) and can be named freely.

Define them in `predicates.py` (path given in the first message):

```python
LEARNED_PREDICATES: List[Predicate]
```

The exec namespace pre-injects `Predicate`, `np`, and a
`<typename>_type` binding for each env type (for example `widget_type`,
`fixture_type`). The names below are illustrative; use the types,
features, and parameter names your digests and the trajectory data
report.

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

A pre-injected `params` view is in scope and always reads the current
fitted values of every `ParamSpec` declared in `simulator.py`; after
each refit, predicates reading `params["name"]` see the new values.
Whenever one physical gate drives both a rule's firing condition and a
predicate's "subgoal reached" check, declare its parameters (the
distance threshold and the local-frame anchor offset it is measured
from) once in `PARAM_SPECS` and reference `params["name"]` from both.
That keeps the two anchored to the same point and gives the offset a
fitting signal from the rule's step data. A parameter used only by
predicates has no fitting signal and stays at its `init_value`, so
choose those initial values carefully.

What you typically need:

- Placement predicates (object at a target location) for every
  open-ended option such as `Place`; without them refinement picks an
  arbitrary location.
- Device-state predicates (on/off) for every toggle option.
- Process-completion predicates over the features your rules drive, so
  `Wait` steps know when to terminate. Keep classifier thresholds
  consistent with the rules' saturation values; an inconsistency makes
  `sim.fit` look fine while `sim.refine` gets stuck on the `Wait`
  subgoal.
- Coverage: every option you expect in a sketch should have predicates
  that express its post-condition, so every sketch step can carry a
  subgoal annotation. Annotations are checked against the real state
  during execution to detect and replan diverged steps; a step with no
  annotatable effect is unmonitored. While drafting sketches, a step
  you cannot annotate with any invented predicate is a missing
  predicate.

Verify every classifier against the scene and the data. A classifier
picks features and parameter values, and both can be wrong, so commit
neither from intuition: follow the threshold-fitting protocol in
"Geometric gates" for every numeric cutoff, and use __SCENE_WORKBENCH__
for geometry and `run_python` for the numeric sweep over trajectory
states.

`sim.predicates()` validates cheaply (first-flip step, monotonicity,
coverage across all trajectories) and is also the loader: it updates
the predicate set `sim.refine` uses, so call it after every edit to
`predicates.py` and before re-running refinement. On goal-reaching
trajectories (`reached_goal=True` in `describe_trajectory`) a milestone
predicate should flip from false to true exactly once and stay true.
On failed interaction trajectories (`reached_goal=False`) the same
predicate may fire while the rest of the trajectory shows no goal
completion; that is the signature of an over-loose threshold (the
predicate fires, the downstream physics does not follow), so tighten
it or share the gating parameter with the rule so they are fitted
jointly.

Predicates persist across online cycles: the file is preserved between
synthesis sessions, and every successful `Write`/`Edit` (plus a final
post-session check) is snapshotted to
`predicates_versions/cycle_XXX_vers_YYY_predicates.py`. Each cycle
re-runs synthesis with the full trajectory history, so failed past
attempts remain visible.

<!-- section: message -->
## Predicate Invention

Only the predicates under "Available Predicates" above exist; this
approach stripped the environment's symbolic predicates down to that
allowlist. Invent every other subgoal predicate in `__PREDICATES_FILE__`
as `LEARNED_PREDICATES`, following the system prompt's "Predicate
Invention" section.

__GOAL_BLOCK__

Workflow: edit `predicates.py`, call `sim.predicates()` in
`run_python`, then run `sim.refine` / `sim.run` with sketches that
reference your invented names. Any predicate a sketch references must
exist in `predicates.py` first.

<!-- section: workflow_extra -->
Step 4's sketches need subgoal predicates that do not exist until you
invent them: before validating, write them to `predicates.py` and load
them with `sim.predicates()` (see "Predicate Invention").
