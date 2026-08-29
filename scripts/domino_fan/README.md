# domino-fan: running the ladder

Three scripts. No YAML editing.

```bash
scripts/domino_fan/run_rung.sh 1        # or 2, 3, 4, or "all"
scripts/domino_fan/dashboard.sh         # watch results at :8765
scripts/domino_fan/reset_runs.sh --yes  # clear old runs
```

## The ladder

Each rung hands the agent less and asks it to recover more.

| rung | arm | given | must supply |
|------|-----|-------|-------------|
| 1 | `oracle` | simulator, predicates | nothing — the process planner plans |
| 2 | `agent_model_based_planning` | simulator, predicates | the plan and its continuous parameters |
| 3 | `agent_param_learning` | simulator *structure* | the wind's parameters, fitted from data |
| 4 | `agent_po_predicate_invention_al` | base simulator only | that a wind exists, code modelling it, and predicates |

Rung 1 takes ~2 minutes and calls no LLM. Rungs 2–4 drive Claude and
run from minutes to hours; `run_rung.sh` refuses to start a second run
while one is going, and so does the dashboard button.

## The task

The robot bridges a start block to a target with blue dominoes, presses
a switch, and the **wind** topples the chain. It never pushes a domino:
`Push` is withheld in this env, option and process both, so the fan is
the only way to start a cascade.

## Reading the score

```
reward = 1[goal reached via a certified wind cascade] − 0.05 × blues used
```

The certificate rejects an episode where the arm knocks the target over
instead of the wind — `TurnFanOn` is the sanctioned trigger here, in
place of `Push`. So **fewer blocks scores higher**, and the rungs are
not all chasing the same number:

| | blocks used | reward |
|---|---|---|
| rung 1 (oracle, grid planner) | 2 | 0.900 |
| rung 2 (agent plans) | 1 | **0.950** |

Rung 2 beats the oracle because the oracle's grid planner bridges with
two blues while a domino actually topples further than one `pos_gap`.
Treat 0.900 as the oracle's score, not the task's ceiling.

Train tasks are shorter than test tasks (a 1-block bridge against a
2-block one), so a run at 0.95 on train and 0.900 on test is at ceiling
on both — not declining.

## Where things land

```
logs/<approach>/domino_fan-<arm>/seed0/run_<ts>/    transcripts, sandbox, fits
videos/<same path>/                                 .mp4 per episode
results/                                            .pkl metrics per cycle
docs/envs/domino_fan/                               curated gifs, committed
```

Run artifacts are working data and not committed. Anything worth
keeping gets converted and put in `docs/envs/domino_fan/` deliberately.
