# Real-Robot Bring-Up: from a captured scene to closed-loop active learning

A staged procedure for bringing the domino scene up on the real Franka. The
organizing principle is that **each stage adds exactly one new source of
failure**, so when something breaks you know what caused it. Run them in order;
do not skip ahead to save time, because a failure three stages later is much
harder to attribute.

Nothing below has been validated on hardware. The pass criteria are what to
check, not results anyone has seen. Two of the stages exist specifically to
replace guessed defaults with measurements — see
[Numbers to calibrate](#numbers-to-calibrate).

## The pieces

| Piece | Where | Role |
| --- | --- | --- |
| `PyBulletDominoRealEnv` | `predicators/envs/pybullet_domino_real.py` | Pure simulation. Sizes itself from a captured scene; converts perception into `State` / `EnvironmentTask`. Holds no robot. |
| `RealRobotExecutor` | `predicators/pybullet_helpers/real_robot_executor.py` | Attached to the env. Ships each option's trajectory to the arm, looks at the scene, writes what it saw into the simulated **twin**. |
| `RealRobot` | babyrobot submodule | The arm and the cameras, in-process. |
| `replay_plan.py` | `scripts/domino_debug/replay_plan.py` | Debug tool: replays one exact recorded plan. No planning, no LLM. |

The env stays pure sim throughout; "real mode" is a property of an executor
being attached, which happens only at `main.setup_environment`. Bilevel search
cannot reach the executor (`PyBulletEnv.simulate` bypasses it), so planning
never moves the arm.

## The flags that matter

| Flag | Default | Meaning |
| --- | --- | --- |
| `real_robot_execute` | `False` | Attach an executor at all. Everything else is ignored when this is off. |
| `real_robot_dry` | `False` | Build the `RealRobot` with no arm: every call runs, nothing moves. |
| `real_robot_perception` | `"zed"` | `"zed"` live cameras, `"scene_file"` replays `domino_real_scene`, `"none"` blind. |
| `real_robot_observe_at_option_boundary` | `True` | Look at the scene between options and correct the twin. |
| `real_robot_human_reset` | `True` | Between episodes, home the arm, wait for a human, then rebuild the task from what the cameras see. |
| `real_robot_settle_s` | `0.5` | Dwell before a capture, so dominoes come to rest. |
| `real_robot_divergence_atol` | `0.02` | Metres of twin-vs-scene disagreement worth logging. |

Canonical launcher:

```bash
python scripts/local/launch_simp.py -c predicatorv3/<config>.yaml
```

For one-off overrides during bring-up it is usually easier to drive `main.py`
directly and pass flags, rather than editing YAML for each stage:

```bash
PYTHONHASHSEED=0 python predicators/main.py --env pybullet_domino_real \
    --approach oracle --seed 0 --real_robot_execute True --real_robot_dry True
```

---

## Stage overview

| # | Arm | Cameras | Human | What it proves |
| --- | --- | --- | --- | --- |
| 0 | — | — | — | A scene exists and its roles resolve |
| 1 | — | — | — | The scene is solvable at all, in sim |
| 2 | dry | replay | — | Executor plumbing survives a full episode |
| 3 | dry | **live** | — | Perception maps to the twin correctly |
| 4 | dry | live | yes | Reset cadence and task rebuild |
| 5 | **moves** | — | — | Per-option shipping at the hand |
| 6 | moves | — | — | A whole recorded plan |
| 7 | moves | live | — | The closed loop |
| 8 | moves | live | yes | Real active learning |

---

## Stage 0 — a scene to work from

```bash
babyrobot scene capture --out scenes/domino_real_00NN.json
```

Point `domino_real_scene` at it and set `domino_real_start_id` /
`domino_real_target_id` to the green (start) and purple (target) dominoes. The
raw capture carries no roles; they are assigned by id.

**Pass:** the env builds and allocates exactly one target. You can skip this
stage and reuse `domino_real_0000.json`.

## Stage 1 — sim only, oracle approach

Nothing real. `real_robot_execute: False`, approach `oracle`. Un-skip
`domino_real` in `predicatorv3/oracle.yaml`.

**Proves:** the captured scene is *solvable at all* — reach, geometry, the
base→world transplant, and the `Toppled(target)` goal.

**Pass:** a plan is found and the rollout topples the target in sim.

**This is the gate for everything after it.** If oracle cannot solve the
captured scene, stop: no amount of hardware work fixes an unsolvable scene.
Re-capture with the dominoes closer together, or reconsider the layout.

Two things to try before concluding it is unsolvable:

- `domino_restricted_push: True`, which is what the other domino envs use for
  oracle runs (the target is then inferred from state).
- Check `envs/all.yaml`, which notes `max_initial_demos: 0` because the grid
  oracle "can't solve the real scene". That is about the *demo generator*, not
  the oracle approach — but it is a warning that this scene is hard.

## Stage 2 — dry executor, replayed perception

First time the executor runs inside the real pipeline. No arm, no cameras.

```
real_robot_execute: True
real_robot_dry: True
real_robot_perception: "scene_file"
real_robot_human_reset: False
```

**Proves:** attachment, per-option chunking, the twin sync and the gripper
split all survive a full episode in the stock loop.

**Pass:** the episode completes, and the log shows one look per option
boundary.

**Expected failure:** shape mismatches between predicators and babyrobot
message types.

File perception always replays the capture, so divergence reads ≈0 and no
topple is ever reported. That is expected here — this stage is plumbing, not
perception.

## Stage 3 — dry executor, live cameras, no motion

Flip `real_robot_perception: "zed"`. Still `real_robot_dry: True`.

**Proves:** the ZED path works end-to-end through predicators' conversion —
capture ids map to component slots, and poses land where they should.

**This is the highest-value stage before any motion.** Do all three checks:

1. **Static.** Run against an untouched scene. Divergence should sit at the
   perception noise floor. Write the numbers down; they set
   `real_robot_divergence_atol`.
2. **Moved.** Physically move one domino ≈5 cm and re-run. Divergence should
   report ≈0.05 m, and the twin should move *that* domino, not another one. A
   wrong domino moving means the id→slot mapping is wrong.
3. **Toppled.** Lay one domino on its face. `Toppled` should become true for
   it. The toppled-pose reading is verified offline against captures `0000`
   and `0001` but has never seen live data.

**Watch for a constant z offset**, which means a table-height disagreement.
predicators passes `domino_real_table_z` (−0.041) into `DominoPerception`,
deliberately overriding babyrobot's own default (−0.045). Confirm the value in
the config is the calibrated one for your table.

## Stage 4 — dry executor, human resets

Add `real_robot_human_reset: True` and run two episodes.

**Proves:** the prompt cadence and the task rebuild.

**Pass:** exactly one prompt per episode, and each episode's initial state
reflects how you just arranged the dominoes — not the capture. `executor.resets_done`
should equal the episode count.

**Also verify the safety property here:** the arm homes *before* the prompt
appears, i.e. before a human reaches into the workspace. In dry mode nothing
moves, so confirm the ordering in the log; you are checking it now so that it
is already trusted at Stage 8.

## Stage 5 — first motion: one option, open loop

```
real_robot_dry: False
real_robot_observe_at_option_boundary: False
real_robot_human_reset: False
```

Use `replay_plan` with a short known-good plan — ideally a single `Push`.

```bash
PYTHONPATH=. python scripts/domino_debug/replay_plan.py --plan plan.txt --execute
```

**This is the riskiest step in the plan, and the one thing that has never
run.** Every prior hardware session shipped the whole episode in one call;
per-option shipping is new. What makes it safe is `RealRobot`'s session-wide
gripper dedup: a `Place` chunk re-emits a leading `close`, and the dedup is
what stops it force-grasping a domino it is already holding.

**Verify at the hand, not in the log.** Hand on the e-stop. Clear the table of
anything you are not willing to have knocked over.

Then extend to a `Pick` → `Place` pair, which is the exact case the dedup
exists for: watch that the gripper closes once, carries, and releases.

## Stage 6 — arm, full recorded plan

Same settings, the whole plan.

**Proves:** multi-option shipping and the drift guard across a full episode.

**Pass:** the arm completes without the drift guard tripping between options.
A trip here means the arm did not end an option where the twin thought it did.

## Stage 7 — close the loop

`real_robot_observe_at_option_boundary: True`. Still `replay_plan`, still no
human reset.

**Proves:** looking mid-episode does not destabilize execution — the twin is
corrected between options and the next option still starts from a sane state.

**Pass:** the episode completes and per-boundary divergence stays near the
noise floor from Stage 3.

If divergence is routinely above tolerance on a scene nothing touched, either
`real_robot_settle_s` (0.5 s) is too short for the dominoes to come to rest, or
perception is noisier than Stage 3 suggested. Both are measurable — do not
just raise the tolerance until the warning stops.

## Stage 8 — real active learning

`predicatorv3/exp_domino_real.yaml` as shipped: 2 online-learning cycles × 1
seed, human resets on, closed loop on.

**Pass:** the run completes, one human reset per episode, and the learner
consumes *perceived* transitions rather than the simulator's guesses.

**Budget:** on the order of a couple of hours. Every episode costs a physical
reset by a human, which is why the cycle and seed counts are cut from
`common.yaml`'s 10 × 3.

---

## Numbers to calibrate

Every other stage is pass/fail. These two produce numbers, and both currently
ship as guesses:

| Setting | Ships as | Set it from |
| --- | --- | --- |
| `real_robot_divergence_atol` | 0.02 m | Stage 3's static noise floor, with headroom |
| `real_robot_settle_s` | 0.5 s | Stage 7: the shortest dwell at which post-motion divergence returns to the noise floor |

## If something goes wrong

- **Wrong domino moves in the twin** — id→slot mapping. The env maps capture
  ids to slots through `self._scene_ids`, in scene order. A re-capture that
  reorders or renumbers dominoes invalidates the configured start/target ids.
- **Constant z offset** — table height. See Stage 3.
- **Gripper stuck closed after a `Place`** — the gripper dedup. This is the
  failure per-option shipping was designed around; capture the segment log and
  check whether a second `close` reached the hand.
- **Drift guard trips between options** — the arm did not arrive where the twin
  predicted. Look at the last shipped chunk's final waypoint against the arm's
  reported joints.
- **A human reset prompts twice in one episode** — `_reset_pending` bookkeeping.
  It is set on `after_reset`, so exactly one prompt should be owed per episode.
