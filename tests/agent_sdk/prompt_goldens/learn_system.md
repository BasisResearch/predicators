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

## Supplied physical parameter menu

The base simulator exposes these tunable quantities:

- `lateral_friction` (built-in 0.5, fit box [0.05, 2], fitted in log-space): sliding friction of every body

Declare parameters you want to estimate in `AGENT_PARAM_SPECS`. Names from this menu are connected to the supplied base's physical setters automatically. For additional constants, implement their effect in your own hook, read them with `self.agent_param(name)`, and ensure changes survive state restoration. Use full rollout validation to check that each fitted parameter affects the intended mechanism.

## Fit and validate complete rollouts

Edit `./simulator.py`, then explicitly call `sim.fit()` to estimate its declared constants from the recorded trajectories. Edits are loaded on the next probe call; a rollout does not implicitly fit parameters. Before fitting, the model uses its carried or declared values and is marked UNFITTED. If there are no learnable constants, skip fitting and call `sim.validate()`.

`sim.validate()` replays every selected recording at the values currently deployed for planning, including recordings a robust fit rejected. `sim.residuals()` uses full simulator replay for subclass models to expose accumulated error; the report labels the parameter values it scores. `sim.fit(traj_idxs=[...])` and explicit validation parameter overrides are diagnostics and publish nothing. Compare candidates on the same recordings, inspect per-trajectory failures and preserve counterexamples. A low fitting error on a selected subset does not establish model fidelity or task solvability.

Use `sim.refine(plan)` to search skill parameters, then run the resulting plan continuously with `sim.run(plan)` and check each annotated subgoal. Use `sim.reset(task_idx=..., mods=...)` and `sim.render(label, annotations=[...])` to inspect geometry. Evaluate trajectory success with the supplied evaluator when available; its verdict on a simulated trajectory depends on the model's fidelity. Prefer additional simulator checks over spending real steps on a prediction that disagrees with recorded evidence. Keep speculative mechanisms labeled as hypotheses and state what observation would distinguish competing explanations.

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
3. Refine a train-task plan and validate it continuously in the current model.
4. Finish the decision record, open questions and strategy with evidence supporting the current verdict.
