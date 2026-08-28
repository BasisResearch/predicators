# domino-fan

The robot bridges a start block to a target with blue dominoes, presses
a switch, and the **wind** topples the chain. It never pushes a domino
itself — `Push` is withheld in this env, option and process both, so the
only way to start a cascade is the fan.

Run it:

```bash
PYTHONHASHSEED=0 .venv/bin/python scripts/local/launch_simp.py \
    -c predicatorv3/exp_domino_fan.yaml
```

`exp_domino_fan.yaml` carries the whole ladder; un-skip one arm at a time.

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

| task set | bridge  | blues needed | best reward |
|----------|---------|--------------|-------------|
| train    | 0.196 m | 1            | **0.95**    |
| test     | 0.294 m | 2            | **0.900**   |
