# Simulator learning session instructions

<!-- section: intro -->
You are synthesizing a parameterized residual-dynamics simulator for a
robotic manipulation environment.

A separate physics engine (the base sim) handles robot motion,
grasping, and rigid-body physics. Your simulator handles residual
dynamics: features that change through physical or causal processes
the base sim does not model, such as gradual level changes,
accumulation, propagation between contacting objects, or sensor
readouts that lag their actuators.

<!-- section: declared_params -->
### Parameter estimation is DISABLED in this run

No parameter is fitted from data, by you or by the harness: `sim.fit`
refuses, `sim.residuals(fit_params=True)` and `sweep_params=` are
unavailable, and the deployed model uses every `ParamSpec` and
`AGENT_PARAM_SPECS` entry exactly as you declared it. That makes the
declaration itself the estimate:

- `init_value` is the point estimate the planner uses. Choose it from
  your knowledge of the mechanism and from what the recorded data
  shows qualitatively (`sim.residuals()` at the declared values,
  `sim.run` / `sim.refine` rollouts, `describe_trajectory`); do not
  leave a placeholder.
- `lo` / `hi` is the plausible interval. It is used as such: the
  validation gate re-rolls plans at values across this interval and the
  exploration ensemble is drawn uniformly from it, so a box that is too
  wide rejects every plan and one that is too narrow hides your own
  uncertainty. Declare a finite box for every parameter.

Everywhere the rest of this prompt says to fit, score, or refit a
parameter, read "declare it and check the rollouts at the declared
values" instead.

<!-- section: plan_format -->
## Plan format for `sim.refine` / `sim.run`

One option call per line, with every option argument supplied as a
typed object reference (`obj:type`), matching the options digest in
your prompt exactly. The parser is strict: an omitted argument is not
auto-filled. Example:

```
PickWidget(robot:robot, widget0:widget)
Place(robot:robot) -> {WidgetAtFixture(widget0:widget, fixture0:fixture)}
ActivateFixture(robot:robot, fixture0:fixture)
Wait(robot:robot) -> {WidgetReady(widget0:widget)}
```

The names are illustrative; use the options, types, and predicates your
prompt digests list. Insert a `Wait` after any action that triggers a
delayed process so your rules have steps to fire on.

Subgoal annotations (`-> {Atom(obj:type, ...)}` after a step) are
optional in general but effectively required after open-ended skills
such as `Place`: without one the backtracking search has no preference
for where to put the object, so a `Place; Wait` pair refines cleanly
while skipping the relevant target location, and your rules never
fire. That looks like a rule bug but is a missing subgoal. For `Wait`,
the annotation also says when the wait terminates; prefix an atom with
`NOT` if it should become false.

<!-- section: deliverables -->
## Deliverables of a learning session

- Begin `simulator.py` with a short decision record: mechanisms, evidence, fitted quantities, hidden memory and unresolved hypotheses.
- Reconcile every mechanism exercised by the recordings with the model.
  Preserve confirmed mechanisms when a fit metric is noisy; inspect the counterexamples before changing structure.
- Ground physical changes in recorded behavior the base mispredicts.
  Record an unsupported mechanism that is unnecessary for the goal as an open question instead of implementing it.
  When the goal requires it, implement the unobserved mechanism as a labelled hypothesis (HYPOTHESIS), with honest `ParamSpec` bounds.
  Make the confirming or refuting experiment the first entry of `./open_questions.md`, naming the observation that distinguishes the alternatives.
  A mechanism absent from your model may make the goal unreachable in planning, so distinguish unknown from impossible.
- Declare uncertain constants as `ParamSpec`s with plausible ranges.
  When uncertainty support is enabled, check whether plans survive the supported parameter range rather than relying only on the point estimate.
- Run a final explicit `sim.fit()` if the model declares learnable constants, then `sim.validate()` on the full recordings.
  Refine a complete train-task plan and validate a continuous rollout, including repeated trials when execution varies.
  Record a GO/NO-GO verdict, weakest margin and supporting evidence; distinguish a model prediction from a real success.
  A GO that rests on a hypothesized mechanism is conditional until the confirming real observation arrives; state that condition explicitly.
- Write `./open_questions.md` as a ranked list of unresolved mechanisms or parameters.
  Each entry gives a concrete experiment, what to measure and the outcomes that distinguish the hypotheses.
  Remove questions the new evidence settles.
- Write `./strategy.md` with the current domain strategy, step ordering, scene-relative formulas and known pitfalls.
  Update advice when evidence changes; state uncertainty honestly.

<!-- section: workflow -->
## Workflow

1. Inspect the data, the base source, prior artifacts and their decision record.
2. Implement or revise the subclass, fit its declared parameters explicitly, and inspect full replay disagreements.
3. Refine a train-task plan and validate it continuously in the current model.__WORKFLOW_EXTRA__
4. Finish the decision record, open questions and strategy with evidence supporting the current verdict.
