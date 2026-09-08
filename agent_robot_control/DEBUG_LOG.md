# Debug log

Every failure that cost more than a few minutes, with its root cause. A
failure is not "fixed" until it is understood and written down here.

## 1. Back-projected particles contained garbage points (2026-09-07)

**Symptom.** With farthest-point downsampling, `donut_0`'s centroid came out
at (151, 47, -26) m. About a quarter of all back-projected points had
coordinates in the hundreds, with normal depth values.

**Evidence.** Per-pixel `inv @ [x, y, z, 1]` gave correct world points for the
same pixels; the vectorised `pix @ inv.T` gave wrong 3rd and 4th columns for
59,876 of 60,300 rows. A standalone repro (`(100000, 4) @ (4, 4)` random
matrices) reproduced 99,840 wrong rows against einsum. Zero wrong rows with
`OPENBLAS_NUM_THREADS=1`, `OPENBLAS_CORETYPE=Haswell`, or `SkylakeX`.

**Cause.** The OpenBLAS 0.3.20 bundled with numpy 1.23.5
(`numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so`) has a multithreaded dgemm
bug on this CPU (Intel Xeon Gold 6448Y, Sapphire Rapids, AVX-512) for tall
skinny products. It went unnoticed before because the old voxel downsampler
rarely picked the outliers.

**Fix.** `particle_world_model/particles.py` and `geometry.py` use
`np.einsum` for the back-projection (no BLAS). `agent_robot_control/__init__`
and all entry points set `OPENBLAS_NUM_THREADS=1`; Slurm job scripts export
it too. Guard: `tests/test_perception.py` checks every particle lies within
2 m of the camera target.

## 2. Particle counts far below the requested number

**Symptom.** Requesting 64 points per object returned 3 to 8, even at
960x540.

**Cause.** `downsample_particles` sizes voxels as
`(prod(bbox ranges) / K)^(1/3)`, which assumes points fill a 3-D volume.
Camera points lie on 2-D surfaces, so only a small fraction of voxels are
occupied.

**Fix.** `agent_robot_control/sim/perception.py` uses farthest-point sampling
on a random subset of at most 4,096 pixels; it returns exactly K points when
at least K pixels are visible.

## 3. Misaligned plug tunnelled through the socket wall

**Symptom.** With a 4 mm lateral error and 2 mm clearance the prong still
went 1.8 cm into the socket; contact penetration was 2.9 mm at only 6 N.

**Cause.** PyBullet's position controller applies effectively unbounded
torque by default, and the arm is laterally stiff, so the prong is pressed
into the wall until the soft-contact solver yields. Stiffer contact
parameters did not help (1.9 mm penetration); physics substeps broke the
aligned insertion (14 degree tilt, kN forces).

**Fix.** `apply_urdf_torque_limits` caps the arm motors at the URDF effort
limits (34/132/77/66/29/26/7 N m). A 4 mm offset now jams at 0.6 mm depth
with 219 N on the rim; aligned insertion still succeeds. Guard:
`tests/test_env_plug_outlet.py::test_offset_oracle_jams_on_rim`.

## 4. Airport env could not be reset through the standard path

**Symptom.** `PyBulletAirportEnv.reset` raised `NotImplementedError` after
the custom override was removed.

**Cause.** The abstract `AirportEnv` had `reset`/`step` stubs that raise, and
in the MRO `PyBulletEnv.reset -> super().reset` lands on them. The previous
override had bypassed `BaseEnv.reset`, so `_current_task` was never set and
`goal_reached()` evaluated a default task (vacuously true).

**Fix.** Removed the stubs; the standard `PyBulletEnv.reset` runs and the
goal is tracked correctly.

## 5. Donut particles had no object names

**Symptom.** `extract_particles` returned nothing for the Donut env.

**Cause.** The env never set `Object.id`, so body ids could not be mapped.

**Fix.** `_store_pybullet_bodies` now assigns ids for donuts and the target.

## 6. Hard clearance tier (1 mm) is beyond the oracle

**Observation.** Aligned oracle insertion at 1 mm clearance lands 2 to 3 mm
off (grasp offset plus position tracking) and is not reliably reported as
plugged in. The tier stays available but medium (2 mm) is the default and
the only one gated by tests.

## 7. The agent could read the repo and this project's notes (2026-09-07)

**Symptom.** In the first end-to-end smoke run the agent's second action was
to `Read` the Claude Code memory file describing this experiment.

**Cause.** The run directory lived under the repo (`outputs/...`), so Claude
Code treated `/home/wp237/predicators` as its project: it loaded the repo's
`CLAUDE.md` and the per-project memory index, and its file tools could reach
the env source (goal predicates included).

**Fix.** Default `output_root` is `$HOME/arc_outputs` (outside any repo) and
the launcher refuses run dirs inside the repo. A per-run `--settings` file
installs a `PreToolUse` hook (`harness/sandbox_hook.py`) that denies
Read/Write/Edit/Glob/Grep outside the workspace and Bash commands that
reference the repo, `~/.claude`, `..`, or other absolute paths. This is a
guard against casual leakage, not a security sandbox.

## 8. Arm sagged 22 cm under raw URDF torque limits (2026-09-07)

**Symptom.** Replaying the smoke run's moves with torque limits on, a
free-space `move_to` to (1.2, 0.9, 0.45) stopped 22 cm short; the first
smoke run (no limits in the Donut env then) had shown a 3 cm miss instead.

**Evidence.** With 1x URDF limits, `|torque|/cap` was 1.0 on the shoulder
pan, elbow flex, forearm roll and wrist roll joints and the joint error vs the
IK target reached 0.37 rad. Without limits every joint tracked within 0.01 rad
except one (see entry 9). Multiplier sweep on the same moves plus the
insertion oracles: 1x sags 21.8 cm; 2x sags 2.7 cm and only inserts 1.45 cm;
3x and 5x track within 0.4 cm, insert fully when aligned, and still jam a 4 mm
misaligned prong at < 0.5 mm depth.

**Cause.** PyBullet's position controller is stiff and gravity-blind; Fetch's
URDF efforts are sized for a compliant real controller, so in stretched poses
the PyBullet motors saturate and the arm droops.

**Fix.** `apply_urdf_torque_limits(robot, scale=3.0)` everywhere
(`torque_limit_scale` in the env class and `SessionConfig`). Guard: the
plug-outlet offset test still passes; the donut replay reaches its targets.

## 9. `move_to` "stalled" 3 cm short in free space

**Symptom.** After the sag was ruled out, (1.2, 0.9, 0.45) still ended 3.4 cm
off with no contacts; only the wrist-flex joint was 0.11 rad from its target
at 2% of its torque cap.

**Cause.** PyBullet's plain IK ignores joint limits. Its solution put wrist
flex past its 2.16 rad limit; `_action_from_joints` clips actions to the
joint range, so the arm converged to a clamped, wrong configuration and the
stall detector fired.

**Fix.** `EEController._ik` first runs plain IK and accepts it only if the
limit-clipped solution still reaches the target (FK check, joints restored);
otherwise it runs null-space IK with joint limits and rest poses. Null-space
IK alone was tried first and rejected: it produced different descent paths in
the insertion oracle, letting the plug catch the rim and rotate 34 degrees in
the grasp. Plain-first keeps the previously verified paths.

**Also learned.** "IK failed" often means the fingertip pads (3.2 cm below the
EE frame) hit the table; `move_to` results now list bodies in contact and
the prompt states the pad offset and the true gripper opening (about 8 cm,
not the 0.04 joint value the agent had read as 4 cm).

## 10. RL pilot on plug insertion flat-lined at reward -4 after 20 episodes

**Symptom.** Both SAC and PPO pilots (hand-written insertion reward) had
max episode reward around -0.3 for 20 episodes, then exactly -4.0 for every
later step.

**Cause.** SAC's exploration pressed the held plug into the outlet; the grasp
constraint (PyBullet default `maxForce` 500 N) gave way, the plug fell and
toppled, and after 50 steps the env's intervention rule respawned it upright
in the holder 40 cm away, outside the RL workspace box. -4.0 is exactly the
reward for a plug sitting in the holder. Without resets, every remaining
episode was wasted (18k interactions).

**Fix.** `ParticleEnv` records whether an object was held when the RL call
started; losing it terminates the episode with a -1 penalty and
`SB3Backend` aborts the call with an explicit message telling the agent to
re-grasp and retry (smaller workspace, gentler reward). The intervention
count is reported in `results.json`. Guard:
`tests/test_particle_env.py::test_drop_aborts_rl_call`.

## 11. Emulated force limit; effective insertion tolerance (2026-09-07)

**Symptom.** With 3x torque caps, the resubmitted plug pilots aborted within
12 s: SAC's random warm-up pushed the held plug into the outlet; measured
contact forces reached 600 to 1300 N within two 1 cm steps while an aligned
insertion produces about 0 to 30 N.

**Cause.** PyBullet's position servo has no force limit; a real arm runs a
compliant, force-limited controller, which is what makes exploration safe in
real-robot insertion RL.

**Fix.** `EEController.contact_force()` measures the peak normal force the
robot or its held object exerts on other bodies (finger-on-held-object
contacts excluded). `move_to` stops and reports when it exceeds
`max_contact_force` (80 N, `server.max_contact_force`); `ParticleEnv`
penalises the violating step (-0.1) and spends the next step retracting along
the last delta. The system prompt tells the agent the controller is
force-limited.

**Measured tolerance (ground-truth-aligned oracle, 1 cm descent steps).**
At 2 mm clearance: lateral errors up to 4 mm self-align through grasp
compliance and insert fully; 6 mm jams at 6.7 mm depth; 8 mm and 10 mm jam at
about 2 mm depth. At 1 mm clearance the oracle fails even with 0 mm
commanded error (the plug shifts 2.8 mm in the grasp on contact). The jam
test therefore uses 8 mm; the "hard" tier is exposed as
`env=plug_outlet_hard` but is not gated by a passing oracle.

**Harness observation.** The first Claude Code run on the medium tier
(`model_free` condition) succeeded at interaction 97 using `move_to` only
(300-point particles were precise enough); its single RL call aborted at
once because the plug had been released. Medium clearance does not separate
the conditions by itself.

## 12. Held objects released by a spurious "open" (2026-09-07)

**Symptom.** Force-limited plug pilots still aborted with `dropped=True`
after one or zero force violations; separately, the scripted grasp succeeded
for seeds 0 and 4 but not 1 to 3.

**Cause.** `PyBulletEnv` removes the grasp constraint whenever the commanded
finger target exceeds the current finger position by 1e-4. Our "keep closed"
command was the nominal closed value (0.01). When the block shifted slightly
in the grasp and the fingers closed to 0.0099, the same command read as an
open and the object was released. Commanding fully closed (0.0) fixed the
release but squeezed so hard (finger position error x gain) that the block
pivoted in the grasp: the medium oracle then failed with a 7 degree tilt.

**Fix.** A closed command tracks `min(closed, current - 2 mm)`, so it never
reads as opening and never over-squeezes. Grasps now succeed for seeds 0 to
4 (plug lifted, tilt about 2.5 degrees) and the medium/easy oracles insert
(0 N contact) while 8 mm offsets jam (939 N, stopped by the force limit).

## 13. Airport button could not be pressed under the force limit (2026-09-07)

**Symptom.** The first Airport harness run pressed the button 20+ times and
never got item_2 onto the table; after the force limit was added, a scripted
press stopped at 80 N with `is_pressed = 0`.

**Cause.** The button (5 g, prismatic constraint) rested directly on its
stand with zero travel. "Pressed" was defined as the button being 1 mm lower
than its rest height, which only happened when the unbounded-force controller
drove it into the stand. Separately, the pusher only shoves items during its
60-step sweep across the belt; items arriving after it is fully extended pass
behind it, so pressing has a timing window.

**Fix.** The button is a static contact switch: pressed while the robot
touches it with at least 0.5 N. Timing sweep with the oracle: pressing when
item_2 is 0.40 to 0.45 m upstream of the pusher succeeds (about 300
interactions total); 0.20 to 0.35 m nudges the item sideways but not onto the
table; 0.50 m or more is too early. Guard:
`tests/test_env_airport.py::test_timed_button_press_puts_goal_item_on_table`.

**Addendum (same day).** The tracking target alone was not enough: the finger
joint reads about -1e-4 when fully closed, and a target clipped at the action
space minimum (0.0) still exceeded it by the release tolerance. Two more
changes: (a) closed commands may go slightly negative (down to -0.01), so a
closed gripper can never read as opening; (b) finger motors are capped at
their URDF effort (60 N) in `apply_urdf_torque_limits`, because with
unbounded finger force the pads squeezed through the constrained block until
it left the grasp (roll about 10 degrees, fingers fully closed). Stack trace
confirmed the release came from `_step_base`'s opening branch. After the fix,
600 random RL steps on the held plug for two seeds produced 13 to 14
force-limit reflexes and no drop, with the fingers steady at 0.015.

## 14. Stagnation abort fired during genuine learning (2026-09-07)

**Symptom.** Force-limited plug pilots (v5) ran without drops but SAC and
PPO both aborted as "stagnated" after 80 to 99 episodes (about 5k of 20k
interactions).

**Cause.** The rule compared only the best per-episode max reward over 40
episodes, which sat at about -0.15 while the mean return was still improving
(SAC: -33 to -22 per episode). Early RL often raises the average long before
the best single step moves.

**Fix.** Stagnation now requires 40% of the call's budget spent, at least
three windows of episodes, and no improvement in BOTH the best reward and the
mean return between the last two windows. Pilots resubmitted (v6).

## 15. Airport agents burn turns to make time pass (2026-09-07)

**Symptom.** Airport `move_to` runs hit the 120-turn cap with only 500 to 600
interactions used (seed 1: 82 `move_to` calls, 35 particle calls, $4). The
agent's last message: "continue advancing time toward the pusher zone".

**Cause.** Simulation time only advances through actions, so waiting for
item_2 to reach the pusher required a tool call per few centimetres of belt
travel. Turns, not interactions, became the binding constraint.

**Fix.** New `wait(steps)` tool (holds the arm, costs `steps` interactions),
in every condition, with a prompt note. Added after sweep array 530099 had
started its six Airport tasks, so those tasks ran without it; the Airport
runs are to be re-run as a separate array once the sweep finishes, and the
later Donut/Plug tasks in the sweep see the tool (unused by them).

**Pilot note.** Plug-insertion pilots v6 (hand-written reward, force
limited, no drops): PPO stagnated at 8k interactions with best reward flat at
about -0.13 (1.3 cm lateral); SAC's mean return improved from -31 to -17 per
episode but its best reward stayed flat too. Neither inserted within the
budget. The agent-written rewards in the sweep may differ; this is the
reset-free single-stream regime the plan warned about.

## 16. Observation: RL inserted the plug but the agent's reward could not see it (2026-09-07)

Sweep run `pybullet_plug_outletmedium/model_free/seed_1`: the agent grasped
the plug, then called `run_rl_on_particles` (budget 4000, 30-step episodes,
xyz) with a reward based on the lowest visible plug points versus the outlet
centroid. The env's `PluggedIn` predicate fired at interaction 1651 during RL
exploration, yet `successes_during_training` stayed 0 and RL ran to its
budget: once the prong is inside the socket its points are occluded, so the
particle-based tip estimate never reaches the reward's success region. The
run counts as a success (first success 1651, 4233 total) because success is
scored by the env, but the agent itself never learned it had succeeded. This
is the single-camera occlusion limitation anticipated in PLAN.md Section 11;
a second camera or a reward built on the visible plug top (block height) would
avoid it.

## 17. Observation: RL tool used as a clock (2026-09-07)

Airport `model_free` seed 2 (array 530099, before the `wait` tool existed)
called `run_rl_on_particles` three times with the reward
`(ee_z - 1.0) / 0.5`, i.e. no task content at all, spending about 12,000
interactions. The transcript shows the intent: let the belt run while
spending few turns. Two of the calls ended by the stagnation rule. This is
the clearest evidence that the missing `wait` primitive distorted the Airport
condition; the six Airport runs were re-submitted with `wait` (array 530684)
and the original outputs are kept under `first_runs/airport_no_wait/`.

## 18. Airport agent declared success; replay shows the item on the belt edge (2026-09-07)

**Symptom.** Airport `model_free` seed 0 (with `wait`) ended after 408
interactions with the agent stating "item_2 was pushed onto the table" and
"vanished from the belt cycle"; the goal never fired.

**Method.** `experiments/replay.py` re-executes a run's move_to/wait calls
from `events.jsonl` in a fresh sim; the replayed interaction counts matched
the log at every call (PyBullet is deterministic).

**Cause.** The button press at interaction 299 came slightly late: the
pusher's sweep caught item_2 only partially and left it at y = 0.689, i.e.
still on the belt (edge at 0.70), cycling as before. The agent's later
particle snapshots showed item_2 as not visible (it was at the far end of
the belt, out of the camera frame) and it concluded the item had left the
belt. All six Airport reruns failed; the button route needs the press to
start when the item is 0.40 to 0.45 m upstream (entry 13), and the agent's
timing was off by a few centimetres in every attempt it made.

## 19. Account usage cap truncated 8 sweep runs (2026-09-07)

**Symptom.** Late sweep tasks ended with the assistant message "You've hit
your session limit - resets 11:50pm" and exit code 1; all three hard-tier
`model_free` runs and five of six Airport reruns were affected.

**Cause.** The Claude account's rolling usage limit, reached while six runs
ran concurrently for hours. The harness reported it as an ordinary turn.

**Fix.** `parse_stream_json` detects the message; `results.json` carries
`account_limit_hit` and `invalid` (unless the goal was already reached), and
`analyze.py` skips invalid runs. Truncated runs archived and resubmitted with
4 concurrent tasks (`submit_sweep.py --only ...`). For future sweeps, keep
concurrency low enough for the account's limit, or run OpenCode/API-key
harnesses for the bulk of the runs.

## 20. Slurm jobs sat in PENDING for hours (2026-09-08)

**Symptom.** Video-render jobs (1 CPU, minutes of work) stayed
`PENDING (Priority)` on `ellis` and then on `default_partition`, with an
estimated start ten hours out, while `sinfo` showed thousands of idle CPUs.

**Cause, two independent ones.**
1. Partition priority tiers: `ellis` 20, `gpu` 15, `default_partition` 10.
   The shared pool's idle nodes belong to other groups' tier-20 partitions
   and are held for them, so a tier-10 job waits. A 1-CPU/2-minute probe job
   reproduced this. `squeue` only shows this user's own jobs
   (`PrivateData`), which made the pool look empty.
2. Memory, not CPUs, on `ellis`: `ellis-compute-01` had 4 idle CPUs but
   `AllocMem=253952` of `RealMemory=257500`, i.e. about 3.5 GB unallocated.
   Tasks requesting 4 GB could not fit; at `--mem=1500` two started at once
   immediately.

**Fix.** Job scripts request `--mem=1500` per render task and list
`--partition=ellis,gpu,default_partition` so a task starts wherever capacity
frees first. CPU-only experiment scripts default to `default_partition`,
GPU work to `gpu`. Diagnostic order for a stuck job: `scontrol show job` for
`Reason`, then `scontrol show node` for `AllocMem` versus `RealMemory`, then
`scontrol show partition` for `PriorityTier`.

## 21. Replay fidelity confirmed (2026-09-08)

All twelve deterministically replayable successes were re-executed from their
event logs and rendered to video. Every one reproduced its logged
first-success interaction exactly, with zero per-call step-count mismatches,
which also validates that the interaction accounting in `SimSession` is
faithful and that the runs contain no hidden nondeterminism outside RL.

## 22. Sweep 2 feasibility gates (2026-09-08)

Each domain change was gated with a ground-truth oracle before any Opus budget
was spent. Three of the four gates failed first time, for reasons worth
keeping.

**Three-leg plug, first attempt: 0/5 at every clearance from 5 mm down.** The
oracle jammed at about 9 mm of depth with 900 N of contact. Cause: the finger
pads hang about 32 mm below the end-effector frame, so with 20 mm legs the
gripper reached the outlet plate before the legs were deep enough. Fix: 30 mm
blades and a 32 mm ground pin, which leaves the pads 4.5 mm clear of the plate
at full depth. Second attempt: 4.0 mm clearance inserts 5/5 at zero contact
force, 3.0 mm only 3/5, and 2.5 mm and below jam at about 900 N with the plug
8 degrees off square. The floor is the plug tilting in the jaws, not the hole
size, so the sweep runs at 4.0 mm. Misalignment probe there: 3 mm lateral
error still inserts 3/3, 5 mm inserts 1/3, and a 15 degree yaw error never
inserts. The yaw constraint is what a single prong never demanded.

**Donut push, first attempt: 0/3, though the discs travelled most of the way.**
Two separate faults. The oracle pushed with the end effector at table + 20 mm,
which puts the finger pads 12 mm *below* the table surface, so the arm was
skidding its pads on the table rather than pushing the disc; pushing height is
now table + 45 mm. And the goal disc started at y = 0.28, which puts the
pose behind it (about 12 cm further back) outside the arm's reachable band, so
one seed never made contact at all; the disc now starts at y = 0.35.

**Airport press window with the 20-step lag.** Leads of 0.60 to 0.70 m succeed
where 0.30 to 0.50 m all fail, so the delay moved the working press about
20 cm upstream, exactly the belt travel during the lag. The window itself
stayed roughly as wide as before, which is the point: the task is harder
because the agent must anticipate, not because the target got smaller.
