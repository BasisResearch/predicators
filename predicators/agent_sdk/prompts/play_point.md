<!-- Wording of EMPIRIC from assets at one value per parameter
     (continual_point_values_only). A section named `<template>__<section>`
     stands in for that section of that template when play_prompts.py
     renders this arm's prompt. Each one says what the agent has. None
     names the noise, a belief, a draw or a stress test, and none says that
     something is left out. -->

<!-- section: play_system__observations -->
## Reading observations

Object features and renders describe the observed scene.
`[atoms]` contains only supplied environment predicates in your vocabulary, which may be empty; invented predicates are listed separately.
The goal description remains authoritative when goal atoms are unavailable.
A predicate inferred from model memory is an inference, not a measured fact.

<!-- section: play_system__model_gate_fit -->
### Test levels require a fitted model

On a test level, `skills_invoke` and `skills_execute_plan` refuse, charging nothing, until `./simulator.py` loads, declares `RESIDUAL_FEATURES` and, when it declares parameters, has been fitted with `sim.fit()` in its current form.
Every edit of the file needs a new fit before the next skill on a test level, so finish editing first.
The refusal says which condition is unmet.
Train levels are not gated: collect evidence there first.
Use `sim` to check plans before execution.

<!-- section: play_system__model_repair_point_estimate -->
### When the model disagrees with evidence

Treat a rejected fit as evidence to investigate, not a hard action gate or a reason to give up.

1. Replay the recordings with `sim.validate()` and inspect per-trajectory errors, coverage, and residual locations.
   `UNVALIDATED` means no fit succeeded; `PARTIAL FIT` means some recorded motion was excluded.
   A low error on accepted segments can hide important counterexamples.
2. Compare alternative dynamics structures as well as parameter values.
   Check units, timestep, coordinates, forces, object-specific behavior, and missing interactions against observations and __REPAIR_REFERENCE__.
   Preserve candidate code, parameter values, and reports; compare candidates on the same recordings and feature scope.
   Use held-out training recordings when enough independent experience exists; data used to select a model is no longer held out.
   Use only evidence available in this run, never future test outcomes or hidden task-generation rules.
3. Keep one candidate deployed at its __REPAIR_VALUES__ values and rehearse plans under it.
   Choose the candidate the recordings support best, then act.
   Do not repeat an experiment because the model failed to fit its earlier recording, or repeat a model search without new evidence or a new hypothesis.

Record how the candidates compare, and the hypotheses you rejected, in the journal.
Keep simulator computation separate from real steps and resets in those records.

<!-- section: play_system__robustness_point_estimate -->
| Check reliability | Repeated rehearsals at the same state and dynamics (`trials>=2`) check controller reliability. |

<!-- section: play_system__journal -->
## Run memory

Update `./journal.md` when you learn something, not only at the end of a level.
Keep observed facts, hypotheses and open questions, candidate models and validation results, failed attempts, and the next action with its rationale.
Link longer analyses and reusable code in sandbox files.

<!-- section: play_system__point_estimate_decisions_raw -->
## State and parameter values

Use the latest observation as the current state and the fitted value of each parameter for every decision.
`sim.reset(current=True)` stages that observation, with the model memory that `update_model_state` infers from the observations so far.
Model memory tracks action history and hidden processes; the observed features come from the observation.
Repeated rehearsals at the same state and dynamics (`trials>=2`) check controller reliability.

<!-- section: subclass_model__memory -->
## Hidden model state

When a mechanism needs memory, declare `MODEL_STATE_INIT` on the subclass as a dict or a callable returning a fresh dict.
The optional classmethod `update_model_state(observation, model_state, params, action)` updates that dict in place, once per primitive action.
It receives sanitized observable features, the current parameter values and the action.
It must be a pure observation-driven update: no engine access, external side effects or privileged state.
The first observation initializes memory without advancing it.
Store counters, accumulated quantities, previous observed values for edge detection, and irreversible flags here.
Key object-specific entries by `obj.name` and pair-specific entries by both names.

```python
class MyDynamics(__BASE_CLASS__):
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

Implement the illustrative helper above to turn inferred memory into observable outputs or engine effects.
The runtime carries independent copies in `State.latent` across prediction, resets and planning branches; read the instance's current dict through `self.model_state`.
Execution tracking uses the same callback on real observations; this is an inferred state estimate and inherits __ESTIMATE_ERRORS__.
Do not treat it as measured truth.
Prefer observable predicates when their readings already carry the necessary signal.
