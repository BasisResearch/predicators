# Shared simulator contract for model-based agents

<!-- section: simulator -->
## `simulator.py`: a simulator subclass

Export `RESIDUAL_ENV`, a subclass of the supplied `BaseSimulator`, from `./simulator.py`.
`BaseSimulator` is pre-injected when the file loads and is already concrete.
It supplies this environment's visible physics with its hidden mechanisms disabled.
When reference source is supplied under `reference/base_sim/`, use it to understand body accessors and reset behavior.
Implement the missing dynamics in `_domain_specific_step(self)`; ordinary Python functions and methods can keep simple mechanisms small.
Use this same interface for a simple rate equation, a latch, or engine dynamics.
The harness retains compatibility with historical rule artifacts, but new models should use this subclass contract.

Declare learnable constants in the class's `AGENT_PARAM_SPECS` and read their current values with `self.agent_param(name)`.
Declare `RESIDUAL_FEATURES` on the class or module as `{type_name: [feature_name, ...]}` to select observed quantities for the fitting loss.
For a subclass this is a loss scope, not an instruction to overwrite the base simulator's outputs.
Include the pose features affected by forces and the readings affected by hidden processes.
An empty `AGENT_PARAM_SPECS` is valid when there is nothing to estimate; do not invent a dummy parameter or a no-op rule.
Export only `RESIDUAL_ENV` as the dynamics implementation.

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

`widget` and `update_widgets` above illustrate the structure; use this environment's types and implement the helper from observed evidence.
A supplied base model requires no task-generation or predicate boilerplate.
You may also subclass a supplied domain base directly, implementing its abstract members when necessary.
Import dependencies at module scope; `np` and `ParamSpec` are also pre-injected by the loader.

<!-- section: simulator_scene -->
## `simulator.py`: a scene simulator subclass

Export `RESIDUAL_ENV`, a subclass of the supplied `SceneBase`, from `./simulator.py`.
`SceneBase` is pre-injected when the file loads; its source is at `reference/base_sim/scene_base.py`, read-only.
It supplies the robot, its home pose and gripper conventions, the observation types, the render camera, and the binding of observed object names to the bodies you load; it holds no scene, no mechanism and no calibration.
Override the classmethod `initialize_pybullet(cls, using_gui)`: call `super()` (it connects the engine and loads the ground plane and the robot), load every body the manifest lists with `p.loadURDF(cls.asset(path), globalScaling=scale, useFixedBase=..., physicsClientId=client)` or `p.createMultiBody` for primitive shapes, and return the bodies dict with each observed object's body id under its observed name (`None` under the name of an observed object that has no body).
Observed poses set body poses through the base; features no pose carries (a switch reading, a level, a flag) round-trip through the base's feature store unless you override `_set_domain_specific_state` and `_get_domain_specific_feature` to back them with joints or your own state, calling `super()` for the rest.
Implement the mechanisms in `_domain_specific_step(self)`; ordinary Python functions and methods can keep simple mechanisms small.

Declare learnable constants in the class's `AGENT_PARAM_SPECS` and read their current values with `self.agent_param(name)`; masses, frictions and other engine properties you set are constants of your scene until you declare them.
Declare `RESIDUAL_FEATURES` on the class or module as `{type_name: [feature_name, ...]}` to name the observed quantities your model owns; the deployment gate requires it (`{}` if none).
Export only `RESIDUAL_ENV` as the dynamics implementation.

```python
# SceneBase, np and ParamSpec are supplied by the loader.
import pybullet as p

class MyScene(SceneBase):
    AGENT_PARAM_SPECS = [ParamSpec("rate", 0.03, lo=0.0, hi=0.1)]
    RESIDUAL_FEATURES = {"widget": ["progress"]}

    @classmethod
    def initialize_pybullet(cls, using_gui):
        client, robot, bodies = super().initialize_pybullet(using_gui)
        bodies["widget0"] = p.loadURDF(cls.asset("urdf/widget.urdf"),
                                       globalScaling=0.2,
                                       physicsClientId=client)
        return client, robot, bodies

    def _domain_specific_step(self):
        update_widgets(self, self.agent_param("rate"))

RESIDUAL_ENV = MyScene
```

`widget` and `update_widgets` above illustrate the structure; use this environment's manifest, types and features.
A URDF may place a body's origin away from the observed pose reference; check that `sim.reset(current=True)` reconstructs the initial observation before relying on the scene.
When a level's scene changes, edit `initialize_pybullet` to load the new bodies; the harness rebuilds your world when the file changes.

<!-- section: dynamics -->
## Step and restoration behavior

Each primitive action advances the base physics, updates declared model memory, then calls `_domain_specific_step` once.
Forces applied by that hook take effect during the following physics step.
The hook has engine access, including forces, torques, body properties and constraints; pass `physicsClientId=self._physics_client_id` to PyBullet calls.
Use real engine constraints for bodies that must move together.
Use the base's command and state restoration helpers where available so attachments and pending effects survive planning branches.
Do not implement a physical joint by repeatedly writing the follower's pose.

Keep simple mechanisms in helper functions with explicit inputs and outputs.
Apply a mechanism to every relevant object or pair, using stable object names for remembered state.
Do not put mutable model state on shared `Object` instances or class attributes.
Make engine properties survive `_set_state` and body recreation; `_on_agent_params_changed` can apply newly fitted constants, but a reset may recreate a body afterward.
Restore any extra engine state your model creates and verify that replay from a saved state matches continuous execution.
If inferred memory creates attachments or other persistent engine effects, implement `restore_model_state(self)` to realize them immediately after reset, before controller initiation and motion planning.
The hook must be idempotent: do not step physics, advance counters, snap poses, or infer new joints there.
For rigid links inferred by your own observation-driven model, call `self.restore_model_attachments([(name_a, name_b), ...])` from this hook and when the inferred links change during dynamics.
This registers links for held-assembly collision checking and snapshot restoration; creating an unregistered engine constraint is insufficient.
The helper does not supply attachment rules or infer links from the real environment.
Run `sim.reset(current=True).check_restore()` after model edits and before trusting a held-assembly rehearsal.
It checks pose and inferred-memory round trips in fresh worlds without physical steps; a pass does not establish that your inferred memory is correct.

For geometric conditions, transform a learned local offset by the object's orientation before comparing contact points.
Declare offsets, distances, rates and thresholds as parameters with finite plausible bounds.
Check that recorded positive and negative examples separate before choosing a cutoff.
Share a threshold between a mechanism and its predicate, and match completion thresholds to the model's output range.
Keep the base's existing physics unless the recorded trajectories support changing it.

<!-- section: memory -->
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
Do not treat it as measured truth or as a particle filter.
Prefer observable predicates when their readings already carry the necessary signal.

<!-- section: tools -->
## Fit and validate complete rollouts

Edit `./simulator.py`, then explicitly call `sim.fit()` to estimate its declared constants from the recorded trajectories.
Edits are loaded on the next probe call; a rollout does not implicitly fit parameters.
Before fitting, the model uses its carried or declared values and is marked UNFITTED.
If there are no learnable constants, skip fitting and call `sim.validate()`.

`sim.validate()` replays every selected recording at the values currently deployed for planning, including recordings a robust fit rejected.
`sim.residuals()` uses full simulator replay for subclass models to expose accumulated error; the report labels the parameter values it scores.
`sim.fit(traj_idxs=[...])` and explicit validation parameter overrides are diagnostics and publish nothing.
Compare candidates on the same recordings, inspect per-trajectory failures and preserve counterexamples.
A low fitting error on a selected subset does not establish model fidelity or task solvability.

Use `sim.refine(plan)` to search skill parameters, then run the resulting plan continuously with `sim.run(plan)` and check each annotated subgoal.
Use `sim.reset(task_idx=..., mods=...)` and `sim.render(label, annotations=[...])` to inspect geometry.
Evaluate trajectory success with the supplied evaluator when available; its verdict on a simulated trajectory depends on the model's fidelity.
Prefer additional simulator checks over spending real steps on a prediction that disagrees with recorded evidence.
Keep speculative mechanisms labeled as hypotheses and state what observation would distinguish competing explanations.

<!-- section: physical_params -->
## Supplied physical parameter menu

The base simulator exposes these tunable quantities:

__PARAM_LIST__

Declare parameters you want to estimate in `AGENT_PARAM_SPECS`.
Names from this menu are connected to the supplied base's physical setters automatically.
For additional constants, implement their effect in your own hook, read them with `self.agent_param(name)`, and ensure changes survive state restoration.
Use full rollout validation to check that each fitted parameter affects the intended mechanism.
