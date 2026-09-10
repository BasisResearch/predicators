# The domino pilot: what we did after merging master, and why

Written 2026-09-09. Covers everything between `git merge origin/master` and the
six-run domino pilot: what your friend's launch command would actually do, how
his experiment differs from ours, and what each domino test bought us.

---

## 1. What happened, in order

| # | Step | Outcome |
|---|---|---|
| 1 | Merged `origin/master` into `top/dynamic-env` (commit `2f309ee`) | Two conflicts: `scripts/run_checks.sh` (kept our `uv run` prefixes plus master's `set -euo pipefail`), `setup.py` (stays deleted; dropped master's `emcee` and regenerated `uv.lock`) |
| 2 | Read his framework before touching it | Findings in sections 2 and 3 |
| 3 | Wired `pybullet_domino` into our stack (commit `d7a4bad`) | Env registry entry, `cfg_overrides` plumbing, `conf/env/domino.yaml`, optional full state recording, his certificate recorded in our `results.json` |
| 4 | Wrote a ground-truth oracle + negative controls | Solves 5/5 seeds, certifies 5/5; controls behave |
| 5 | Ran the pilot: `move_to` and `no_particles`, seeds 0-2 | 6 runs, ~$22, all finished |
| 6 | Found and fixed a defect in **our** controller that the pilot exposed | Reachability pre-flight in `move_to` + 2 regression tests |
| 7 | Rendered replay videos of all six trajectories | `~/arc_outputs/videos/domino/*.mp4` |

---

## 2. What his command would do — and why we must not run it here

```
python scripts/local/launch.py -c predicatorv3/protocol_continual_minimal_knowledge_sweep.yaml --parallel
```

`launch.py` calls `get_cmds_to_prep_repo()` (`scripts/cluster_utils.py:211`)
**before** launching anything, and runs it with `shell=True` in this repo:

```bash
git stash                 # our uncommitted work disappears into a stash
git fetch --all
git checkout master       # DEFAULT_BRANCH — leaves top/dynamic-env
git pull
rm -rf results/ logs/ saved_datasets/ saved_approaches/ eval_trajectories/
mkdir -p logs
```

Three separate problems for us:

1. **It moves the checkout to `master`.** Everything in `agent_robot_control/`
   lives on `top/dynamic-env`; after the checkout our experiment code is not on
   disk, and any in-flight Slurm job reading the working tree changes underfoot.
2. **`rm -rf logs/`** is where every one of our Slurm `.out`/`.err` files lives.
3. **`--parallel` is macOS-only** — it opens one `Terminal.app` window per
   experiment (`subprocess.Popen(["open", ...])`). On a Linux login node it
   does nothing useful, and running sequentially instead would put a 48-hour,
   five-domain LLM sweep on the login node.

Also, his configs hardcode his own paths (`continual_runs_dir:
/home/ycliang/predicators/logs`), so his runs would not write anywhere we can
read. **Conclusion: to reproduce his numbers, do it in a separate clone on
`master`, never in this working tree.**

---

## 3. What his experiment actually is

From `docs/continual-protocol-overview.md`, `docs/continual-protocol.md` and
`docs/continual-minimal-knowledge.md`.

**The protocol.** An agent is dropped into one environment and plays its tasks
in order as *levels*, in one continuous run. Train levels come first, then test
levels; a test level is one-shot (no resets). The protocol charges for exactly
one thing: a low-level environment step. The budget is pooled across levels
(5000 steps per level for domino, plus a 48-hour wall clock), so early learning
is amortised. Each run writes a `scorecard.json`: levels won, steps and resets
per level, steps before first win, LLM cost, plus a `run.mp4`.

**`WIN` is the evaluator's certified success, not the goal atoms.** His overview
says this in as many words, and names domino as the example: the goal atoms can
hold after an illegitimate topple and the evaluator rejects the episode.

**The four arms.** His "model-based" and "model-free" are *not* our RL
conditions — no RL is involved anywhere:

| Arm | Gets a simulator | Gets primitive skills |
|---|---|---|
| `agent_continual` | domain-specific base simulator | yes |
| `agent_continual_model_free` | none | yes |
| `agent_continual_model_based_minimal` | the generic `PyBulletEnv` **class source** only | no |
| `agent_continual_model_free_minimal` | none | no |

"Model-based" means the agent **writes its own `simulator.py`** in a sandbox,
fits its parameters against recorded experience, rolls hypotheses out in its own
code, and then acts. "Model-free" means it is denied a simulator and must probe
the real environment. The YAML he sent (`protocol_continual_minimal_knowledge_sweep.yaml`)
runs the two *minimal* arms — no skills, raw actions — over balloons, boil,
bridge, `domino_high_friction_turn` and fan, seed 0.

**Their interface.** `env_observe`, `env_step`, `env_reset`, `give_up`,
`env_run_policy`. Actions are **absolute joint position targets**. An
observation is both numbers and pixels: an object feature table plus observable
atoms plus a render path. `env_run_policy` lets the agent write
`get_action(observation, memory)` to a file and run it for up to N steps — its
closest analogue to our RL tool, except the policy is agent-written code, not
learned.

---

## 4. How our approach differs

| | His minimal arms | Ours (`move_to` condition) |
|---|---|---|
| Action space | absolute joint position targets | Cartesian EE pose via IK (`move_to`), plus `wait` |
| Perception | object feature table + atoms + render | **pixels only**, plus per-object point clouds when the agent calls `pixels_to_particles` |
| Object poses | given as numbers | never given; must be estimated from pixels/particles |
| Resets | `env_reset` available; test levels one-shot | **never** — one continuous session, real-robot style |
| Structure | levels, train then test | one task, one session |
| Budget | 5000 steps/level pooled + 48 h | 100k interactions, 300 turns, $20 |
| Learning | agent writes `simulator.py` (model-based) or probes (model-free) | agent may call SB3 SAC/PPO on particles with its own reward (`model_free`/`model_based` conditions; not used in this pilot) |
| Success | evaluator-certified `WIN` | `goal_reached()` (goal atoms), **plus** his certificate now recorded alongside |
| Domino variant | `domino_high_friction_turn` | plain min-block, seeds 0-2 |

Same env class, different everything else. The comparison to make is not
"our number vs his number" but "what does each interface make hard".

---

## 5. Why the oracle and all the domino tests

Not because his environment is suspect. Because **our interface has never
touched it**, and every failure in `DEBUG_LOG.md` came from exactly that seam:
our IK, our torque caps, our contact-force reflex, our gripper-hold logic, our
camera. Concretely, the tests earned their cost five times over:

1. **They caught a bug in our code.** The first oracle solved 1 of 3 seeds. The
   blue spare is staged rotated 90° from the row, so it has to be grasped across
   its 15 mm face and turned on the way over; my first version carried it at the
   wrong wrist angle and stood it broadside, which sometimes fails to transmit.
   Fixed: 5/5. Had we launched the sweep first, that would have read as "the
   domain is hard for our stack".
2. **They exposed a scoring gap.** The `cheat` control shoves the target over
   with the arm. Our `goal_reached()` calls that a success; his certificate
   rejects it ("the goal topples are owed to the robot's body, not the built
   layout"). That is why `results.json` now carries `certified`,
   `evaluator_reward` and `evaluator_solved` — and it mattered: 2 of 3
   `no_particles` runs are exactly this case.
3. **They proved a solve is reachable at all** before spending Opus budget:
   pick, turn, place in the single 98 mm gap, push the green block — 5/5
   certified in under 400 interactions.
4. **They calibrated the tolerance.** Placement error up to ±35 mm still
   cascades; at 49 mm the target still topples but the clean-push replay fails,
   i.e. the run reaches our goal and fails his certificate. Useful to know
   before reading agent numbers.
5. **They gave the negative controls** that make the positive result mean
   something: pushing with no bridge leaves the target standing, placing without
   pushing scores 0.

Then the pilot itself exposed a **defect in our controller** (section 7), which
the oracle could not have found because the oracle never asks for an impossible
pose.

---

## 6. Pilot results (9 runs, 3 seeds per cell)

| model | condition | goal atom | **certified** | median first success | mean interactions | mean turns | mean $ | IK failures |
|---|---|---|---|---|---|---|---|---|
| Opus 5 | `move_to` (particles) | 3/3 | **3/3** | 154 | 250 | 65 | $2.05 | 0, 0, 0 |
| Opus 5 | `no_particles` | 3/3 | **1/3** | 314 | 719 | 192 | $5.53 | 18, 21, 6 |
| Fable 5.1 | `move_to` (particles) | 3/3 | **3/3** | 209 | 222 | 47 | $1.67 | 0, 0, 0 |

Every certified run scores the same 0.95 (the success bonus minus one consumed
blue), so the models are separated only by cost, not by quality of solve. Fable
used fewer interactions (222 vs 250) and noticeably fewer turns (47 vs 65) for
$5.00 against Opus's $6.14 over the cell, but took longer to its first success
(median 209 vs 154) - it spends more of its budget probing before committing.
Neither model triggered a single reachability refusal, so the controller fix
(section 7) is not a confound for this comparison; only the `no_particles` cell
is affected. Controller provenance: all three Fable processes started at
16:47:39, before the other session's 16:51:18 edit adding a configuration-jump
guard, and Python imports the module at process start - so these runs used the
reachability pre-flight *without* that guard. A later re-run would include it.

Fable's slowest seed is a good illustration of probing without ground truth: it
lowered the open gripper onto the blue block at two different wrist angles,
noticed both stopped at the same height, inferred "the domino top is hitting the
gripper palm with the fingers already straddling it", closed there, lifted, and
measured from the particles that the block hung 11.6 cm below the EE frame -
then rotated it 90 degrees in the air and set it in the gap.

The headline is the gap between the two success columns. **On the goal atom
alone the two conditions look identical (3/3 vs 3/3), and the conclusion would
have been "perception tools don't matter here".** Under the domain's own
certificate, they decide the outcome:

- All three `move_to` agents deliberately built a bridge, pushed the green
  block, said they had succeeded, and were right. Two of them explicitly
  reasoned about the certificate ("only the fingertips touched green"; "the arm
  was 15 cm away and 25 cm up while blue struck purple").
- All three `no_particles` agents reported **failure**. Two were right: they
  knocked the staged blocks over while probing, and the target's topple was
  owed to the arm. The third had, in fact, built a legitimate cascade and
  scored 0.95 without knowing it — the same invisible-success pattern we saw
  with RL in sweep 1.

Videos:
- Opus, both conditions: `~/arc_outputs/videos/domino/domino_{move_to,no_particles}_seed{0,1,2}.mp4`
- Fable, `move_to`: `~/arc_outputs3/videos/domino/domino_move_to_seed{0,1,2}.mp4`

Runs live under `~/arc_outputs/pybullet_domino/` (Opus) and
`~/arc_outputs3/pybullet_domino/` (Fable). Keep them in separate roots:
`run_dir` is `${output_root}/${env.name}/${condition}/${harness}/seed_${seed}`
with `exist_ok=True`, so re-running a cell at the same root overwrites it.
Launch with a Hydra override (`output_root=$HOME/arc_outputs3`) rather than the
`ARC_OUTPUT_ROOT` env var, which `--get-user-env` can drop.

---

## 7. The controller defect the pilot found (and the confound it creates)

`move_to` walks a straight Cartesian line to its target, waypoint by waypoint.
If the target had no joint solution, the old code discovered that only when some
*waypoint* failed — by which point the arm had already traversed most of the
line, sweeping whatever lay on it. Measured directly: an out-of-reach target
dragged a staged domino **32 cm** across the table and permanently failed the
task (the staging rule forbids disturbing the green and purple blocks).

Fixed: `move_to` now checks reachability of the final pose *before* moving and
refuses outright — zero steps, zero interactions, nothing touched, and a message
saying so. The refusal is kinematic only, so press-into-contact moves (plug
insertion, disc pushes) still work and still report a stall. Two regression
tests in `tests/test_ee_control.py` pin both halves.

**The confound:** the pilot's `no_particles` cell ran *before* the fix, and hit
18/21/6 IK failures, versus 0 in every `move_to` run. Part of that gap is a real
consequence of having no perception (you guess targets, you leave the envelope),
but part is our controller punishing a bad guess by ploughing through the scene.
Those two causes are not separated in the numbers above. To separate them the
`no_particles` cell should be re-run on the fixed controller (3 runs, ~$17).
That is the one open decision; I have not re-run it, because it also means the
domino numbers no longer sit on the same controller as sweeps 1 and 2.

---

## 8. The hard variant (his `domino_high_friction_turn`)

Same env class, three things harder at once: every task is an L (90 degree
corner), four blues are staged against a searched minimum of **K\* = 2** with
each one spent costing 0.1 reward, and the true friction is 0.5 while the
planner that generated the task believes 0.1 - with span/leg bands chosen so
the believed-cheapest layout is not the one that works. His own anatomy figure
(`docs/envs/domino_min_block/task_anatomy.png`) states the trap: believing the
blocks reach further than they do, you plan one blue and the chain dies.

Task generation runs simulated minimum-block searches (217-345 s per seed,
~1 turn candidate in 30 survives) and caches under
`saved_datasets/domino_min_block_tasks`.

Opus 5, `move_to`, three seeds:

| seed | blues used | certified | reward | first success | interactions | cost |
|---|---|---|---|---|---|---|
| 0 | 3 | yes | +0.7 | 358 | 425 | $7.06 |
| 1 | 2 (= K\*) | yes | +0.8 | 310 | 374 | $3.29 |
| 2 | 2 (= K\*) | yes | +0.8 | 317 | 343 | $4.08 |

3/3 certified, 2/3 at the oracle minimum. The corner makes it a real placement
problem - a block square to either leg does not transmit round the bend, so the
solves place blues at oblique yaws (seed 1 at -60 and -30 degrees). Median first
success moves from 154 interactions on the straight task to 317 here.

Four runs were discarded and re-run before these three: three whose robot tool
server never started (section 9) and one cut off by the account usage limit.

## 9. Two bugs the hard variant exposed in our stack

1. **The MCP server inherited the harness's working directory** - the agent's
   workspace sandbox - so any setting naming a *relative* path resolved inside
   that sandbox. The domino task cache is one, so every server missed the warm
   cache and regenerated its task (217-345 s) against Claude Code's 300 s
   tool-server connect timeout. Two of the first three runs began with no robot
   tools at all; the third survived only because its task takes 217 s. The
   server now runs from the repo, is launched by naming the venv interpreter
   instead of `uv run` (which locks the shared venv), and keeps its stderr in
   `<run_dir>/mcp_server.log` - previously piped somewhere unreadable, which is
   why a server that never started left no trace.
2. **Validity keyed on the goal atom.** A run truncated by the usage limit
   counted as a result whenever the atom held - but on a domain with a
   certificate the atom is not success, and an agent cut off having toppled the
   target illegitimately might well have gone on to fix its layout. Validity now
   takes the evaluator's verdict where there is one, and a run whose server never
   connected (zero interactions, no robot tool call) is invalid rather than a
   failure.

## 10. State of the tree

- Committed: the merge (`2f309ee`) and the domino wiring (`d7a4bad`).
- Uncommitted: the `move_to` reachability fix, its tests, the analyzer's new
  certificate table, `slurm/make_videos_domino.sub`.
- The full suite passes: **54 passed** (job 654468). Earlier: **50 passed** (Slurm job
  636001, 1m54s), including the 4 new domino gate tests and the 2 new
  controller regression tests. One failure on the first run was unrelated —
  `test_harness.py` pinned `--model claude-opus-5` while the sweep-3 commit
  switched the harness to `claude-fable-5-1`; the assertion now checks the
  configured model is passed through rather than a fixed name.
- Stashed, not lost: `stash@{0}` holds formatting churn `run_autoformat.sh`
  produced across 53 files of freshly merged master code, which does not belong
  in our commits.
- **Another session is working in this same tree.** It committed `fdfbaefa4`
  ("Sweep 3: slant the outlet, unblock the push target, shrink the pusher") on
  top of the domino commit, which switched the harness to `claude-fable-5-1`
  with a $40 cap, and it launched the `pusher_len` (632420) and `arc_sweep3`
  (634407, nine `move_to` runs on airport/donut/plug_outlet) jobs. Because
  `arc_sweep3` started at 14:34, after the controller fix landed in the working
  tree, sweep 3 is running with the new refusal behaviour — on Fable 5.1,
  whereas the domino pilot above ran on Opus 5.
