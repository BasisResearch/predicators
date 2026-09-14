# domino-fan

The robot bridges a start block to a target with blue dominoes, presses
a switch, and the **wind** topples the chain. It never pushes a domino
itself — `Push` is withheld in this env, option and process both, so the
only way to start a cascade is the fan.

Run it:

```bash
scripts/domino_fan/run_rung.sh 1        # or 2, 4
scripts/domino_fan/run_rung.sh --declare 1   # the button-free variant
```

## Results

| rung | what it is handed | result |
|-----:|-------------------|--------|
| 1 | ground-truth simulator and predicates; the process planner plans | 1/1 &middot; **0.900** |
| 2 | ground-truth simulator and predicates; the AGENT plans | 1/1 &middot; **0.950** |
| 3 | structure only, parameters fitted from data | *skipped, see below* |
| 4 | **the base simulator alone** | 1/1 &middot; **0.950** |

Reward is `1 - 0.05 x blues consumed`, so 0.950 is a one-block solve
and 0.900 a two-block one. Rung 4 matches rung 2 and beats the oracle
while being handed none of the model.

**Rung 3 is skipped on purpose.** Fitting the wind's magnitude is not
a learnable problem here: the wind acts for about two steps before the
start block tips, so the whole observation is "tipped or did not", and
1.5 N and 2.0 N produce identical trajectories. Measure it yourself
with `scripts/domino_debug/probe_wind_identifiability.py`. Contrast
`pybullet_fan`, which fits the same parameter happily because a ball's
entire trajectory is wind. What a wind parameter is pushing decides
whether it can be fitted.

## What rung 4 discovered

`discovered_simulator.py` and `discovered_predicates.py` in this
directory are the agent's own, copied from run_20260831_202652. Its
goal text never contained the words "wind" or "fan", and its predicate
vocabulary was stripped to `Holding` alone.

It found the mechanism by comparing its model against reality: "exactly
one step after `fan_0.is_on` flips to 1 ... the GREEN domino - the one
nearest the fan along the fan's facing axis - begins to translate in +x
and to roll ... Nothing else moves." It modelled it as a force through
the engine rather than a feature overwrite, because "the engine must
resolve the resulting contacts (that is what produces the cascade)".

It also found something the hand-written ground-truth simulator does
not model. The wind is **occluded**: a staged blue on the fan's axis
shows roll exactly 0.000 until the toppling green reaches it, and "a
sub-threshold force would have produced a visible lean", so an upright
domino blocks the beam. `domino_fan/gt_simulator.py` sidesteps this by
applying wind only to whichever block is painted green - a role lookup,
not physics.

And it rebuilt the vocabulary it had been denied: `_toppled`,
`_upright`, `_fan_on`, `_fan_off`, `_switch_on`, and `_bridges_gap` -
its own `InFront`, named for what it does.

The test layout has a 0.294 m gap against the 0.196 m it practised on,
so the model generalized rather than memorizing one scene.

## oracle_solve.gif

![oracle](oracle_solve.gif)

Rung 1 — `oracle_process_planning`, ground-truth simulator and
predicates. **1/1 at reward 0.900**, ~2 minutes, no LLM. Plan:

```
PickDomino  → PlaceDomino  → PickDomino  → PlaceDomino
→ TurnFanOn → Wait
```

This is the ceiling the learning rungs are measured against, not a
result in itself.

## rung2_agent_planner.gif

![rung2](rung2_agent_planner.gif)

Rung 2 — `agent_model_based_planning`, same ground-truth simulator and
predicates, but the **agent** writes the plan instead of the process
planner. **1/1 at reward 0.950**, early-stopped at cycle 1.

Higher than the oracle, and not by luck: the oracle's grid-based
planner bridges with TWO blues, while the agent found that ONE
suffices, since a domino topples further than one `pos_gap`. In the
video the second blue is still sitting untouched at its staging spot.

The agent also probed the limits and diagnosed them correctly - x=0.700
and x=0.570 both fail on **gripper clearance**, not physics ("the
opening fingers need ~0.044 m of side clearance"), symmetric about both
neighbours. Its notes are in the run's sandbox as `notes_domino_fan.md`.

## wind_cascade.gif

![wind](wind_cascade.gif)

The mechanism in isolation: a chain laid by hand at `pos_gap`, fan
switched on, **no robot**. 4/4 dominoes topple (final rolls
`[82, 81, 83, 90]` degrees). Useful for showing that the wind physics
work independently of whether the manipulation does.

## Reading the reward

`reward = 1[goal reached via a certified wind cascade] − 0.05 × blues used`

The certificate ([`cascade_certificate.py`](../../../predicators/envs/pybullet_domino/cascade_certificate.py))
rejects an episode where the arm knocks the target over rather than the
wind — `TurnFanOn` is the sanctioned trigger here, in place of `Push`.

Ceilings differ between task sets, so do not read a drop as a
regression:

| task set | bridge  | blues (grid plan) | reward |
|----------|---------|-------------------|--------|
| train    | 0.196 m | 1                 | 0.95   |
| test     | 0.294 m | 2                 | 0.900  |

"Blues needed" is what the ORACLE's grid planner uses, not a floor: rung
2 solved the test task with one blue for 0.950. Treat 0.900 as the
oracle's score, not the task's ceiling.

## Results so far

| rung | arm | what it must supply | result |
|------|-----|---------------------|--------|
| 1 | `oracle` | nothing (GT everything) | **1/1, 0.900** |
| 2 | `agent_model_based_planning` | the plan + its continuous params | **1/1, 0.950** |
| 3 | `agent_param_learning` | the wind's parameters | not run |
| 4 | `agent_po_predicate_invention_al` | the wind's code, its params, and predicates | partial |
