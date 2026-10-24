# Continual standalone program model

<!-- section: workflow -->
## Modeling and acting

Learn a standalone executable world model from your recorded interaction and use it to rehearse candidate skill sequences.
No prepared physical scene or base simulator is supplied underneath this model; there is no phased learning or execution gate.
You may use PyBullet or other available simulation libraries to build your own predictive model from observations and recorded interactions.
Within this conversation you can collect evidence, edit the model, score it on recordings, rehearse plans, and act.
Use `run_python` to access `trajectories`, `describe_trajectory`, and the `sim` probe.
`sim.score()` scores world_model.py against recorded skill transitions, and `sim.run` predicts a plan once through that program.
The harness offers no plan search, repeated-trial rollouts, predicate scoring or renders of predicted states; write any search, sampling or diagnostics you need in your own code.
The supplied residual-fitting API, physical contact diagnostics, and engine-based evaluator replays are unavailable.
You may implement fitting and diagnostics for your own model.
An environment win is authoritative; a predicted goal alone cannot certify it.
Retain programs and your journal across levels and revise them as new evidence arrives.
`sim.reset(current=True)` reconstructs memory by replaying this episode's recorded skill invocations under your latest program, starting each prediction from its observed pre-state.
Model edits and resets therefore discard stale inferred memory.
The skill-level model cannot replay raw `env_step` actions; after such actions, current-state memory reconstruction reports this limitation instead of assuming fresh memory.

<!-- section: contract -->
## Model files

Write `world_model.py` with these definitions:

```python
LATENT_FEATURES = {}  # type name -> list of hidden feature names

def initial_latent(obs, rng):
    return {}  # inferred initial memory, possibly sampled using rng

def transition(obs, latent, option, rng):
    # Predict one complete skill invocation, including robot motion,
    # contact, and every mechanism that progresses during its duration.
    # Return a new observed State, updated memory, positive primitive count.
    return obs.copy(), dict(latent), 1
```

This skeleton is a no-op, not a solution.
Use `option.name`, `option.objects`, `option.params`, and `option.memory` to identify the skill and its arguments.
Wait can stop on its subgoal, requested step count, or maximum steps; model the duration and its effects.
Return the same observed objects and features; keep hidden quantities in memory.
Do not import the task environment, ground-truth mechanisms, or the supplied base simulator, or invoke a real skill inside a prediction.
If you use a physics engine, create and manage your own simulation world, explicitly address its client on every call, and release its resources.
Predictions must not inspect or change the live environment; preserve all model memory needed for reproducible replay in the returned latent state.
Write `predicates.py` exporting `LEARNED_PREDICATES`, a list of your invented `Predicate` objects.
The loader provides `State`, `Predicate`, `np`, and types named `<name>_type`.
Predicates may read `state.latent` when it is present and must tolerate its absence in observations.
