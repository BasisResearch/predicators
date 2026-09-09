# Agent harnesses as robot controllers, with RL as a motor-skill subroutine

Report on the experiments. Sweep 1 is below in full; sweep 2's numbers and
reading are in `RESULTS_sweep2.md`, and the published page at the link in the
session notes carries both.

**Sweep 2 headline (2026-09-09):** removing the particle tool drops success
from 7/9 to 3/8, which is a larger effect than the RL tool produced in either
sweep; the RL tool was called once in 26 runs and cost 75% of that run's
interactions for nothing; the pusher-delay change made Airport properly hard
while the three-leg plug came out easier than the single prong it replaced.

Original sweep-1 report follows. Code in `agent_robot_control/`,
design in `PLAN.md`, every root-caused failure in `DEBUG_LOG.md`, raw sweep
table in `RESULTS_sweep1.md`. Written 2026-09-08.

## 1. Question

A coding-agent harness can plan and call tools, and it is good at coarse
manipulation: look at a scene, decide where the gripper should go, send it
there. It is bad at the last millimetres. The premise of this experiment is
that the harness should delegate that part to trial-and-error learning: it
writes a reward over what it can perceive, an RL algorithm optimises it in
the live environment, and control returns to the agent.

We test three conditions on the same tasks, measuring success as a function
of **environment interactions** (one interaction = one low-level control step
of the simulator, whoever issued it):

| condition | tools the agent gets |
|---|---|
| `move_to` | `move_to`, `wait`, `pixels_to_particles` |
| `model_free` | the above plus `run_rl_on_particles` (SAC or PPO) |
| `model_based` | the above plus `run_model_based_rl_on_particles` (not yet implemented) |

## 2. System

A standalone stdio MCP server owns one live PyBullet simulation for the
lifetime of an agent session. The harness (Claude Code headless,
`claude-sonnet-5`) connects to it through its own config, so the same server
serves any harness that speaks MCP.

- **`move_to(x, y, z, roll, pitch, yaw, gripper, max_steps)`** interpolates
  the end effector along a straight line, solving IK per waypoint and
  stepping the simulator once per waypoint. The controller is force-limited
  (80 N): a move that presses harder stops and says what it hit.
- **`wait(steps)`** holds the arm while time passes, for tasks where
  something must come to the robot.
- **`pixels_to_particles(max_points_per_object)`** renders RGB-D, splits it
  by PyBullet's segmentation mask, back-projects into per-object point clouds
  labelled with object names, and writes an `.npz` plus a `.json` summary
  into the agent's working directory. Free of interaction cost.
- **`run_rl_on_particles(reward_code, budget_interactions, ...)`** compiles
  the agent's reward, validates it on the current scene, and runs SAC (or
  PPO) in the live simulation: small end-effector deltas inside a box around
  the pose where the tool was called, short episodes, **no environment
  resets** — between episodes only the arm returns to that pose, objects stay
  where they are. It aborts early on success, on dropping a held object, or
  on reward stagnation, then executes the learned policy and leaves the world
  in the resulting state.
- **`run_model_based_rl_on_particles`** is a stub with the same signature;
  transitions are already logged so its world model has training data.

The agent never sees ground-truth state, only the camera image, its own
end-effector pose and gripper, and whatever it extracts through particles. A
`PreToolUse` hook confines its file access to the run's workspace so it
cannot read the environment source. Success is scored by the environment's
goal predicate, never revealed to the agent.

## 3. Tasks

- **Donut** (existing): put `donut_0` inside a target square. Donuts keep
  appearing, capped at 8 live.
- **Airport** (existing, reworked): get `item_2` from a looping conveyor onto
  a table, either by pushing it with the gripper or by pressing a button that
  extends a pusher across the belt. The pusher only shoves during its sweep,
  so the press has a timing window of about 5 cm of belt travel.
- **Plug-Outlet** (new): grasp a plug standing in a holder and seat its prong
  in a socket. Clearance per side is the difficulty knob: 4 mm easy, 2 mm
  medium, 1 mm hard.

## 4. Results

Three seeds per cell, 100k-interaction cap, 120-turn cap. Eight runs that the
Claude account's usage limit cut short were discarded and re-run.

| task | `move_to` | `model_free` | median first success (`model_free`) |
|---|---|---|---|
| Donut | 3/3 | 3/3 (RL never called) | 103 |
| Plug-Outlet medium (2 mm) | 1/3 | 3/3 | 1,651 |
| Plug-Outlet hard (1 mm) | 2/3 | 3/3 | 2,660 |
| Airport | 2/3 | 1/3 | 1,129 |

Mean interactions per run: Donut about 135 in both conditions; plug medium
2,645 with `move_to` versus 11,047 with RL; plug hard 146 versus 13,555;
Airport 2,120 versus 4,337.

**RL helps where precision binds, and costs one to two orders of magnitude
more interactions.** Across the two precision tiers the RL condition went
from 4/9 to 7/9 successes. The success-versus-interaction curves cross only
after roughly a thousand interactions: any agent that can solve the task
open-loop wins on the interaction axis by a wide margin.

**Coarse control is better than expected at 1 mm clearance.** With `move_to`
alone the agent solved the hard tier twice, at 87 and 162 interactions, by
grasping across the plug's narrow side so the jaws centred it. My scripted
oracle, which aligns using ground-truth poses, fails at that clearance. The
agent's strategy is better than the one I wrote.

**The agent cannot see its own success.** In five of six plug RL successes
the goal predicate fired *during* RL exploration while the agent's own
particle reward never registered it: once the prong enters the socket its
points are occluded from the single camera, so the reward's success region is
unreachable in the observation. RL therefore ran to its full budget and the
agent never learned it had won. This is the largest single defect in the
setup and the first thing to fix.

**Airport is a timing task and RL hurt there.** With `move_to` plus `wait`
the agent solved it twice; with the RL tool available it solved it once, and
the failures spent thousands of interactions on RL calls that stagnated.
Before `wait` existed, one agent called the RL tool with the reward
`(ee_z - 1.0)/0.5`, i.e. no task content at all, purely to make the belt
advance while spending few conversational turns. Giving the agent an explicit
way to wait removed that pathology.

**Hand-written rewards did worse than the agents'.** Harness-free pilots on
plug insertion (SAC and PPO, 20k interactions, reward over the same
particles) both stagnated without inserting. The agents' rewards succeeded
more often, mostly because they chose tighter starting poses and smaller
workspace boxes than I did.

## 4b. What the RL tool actually contributes

The agent does call the tool: one to four accepted calls per insertion run, and RL
consumes almost all interactions in those runs (for example medium seed 0: 17,350 RL
steps against 84 of the agent's own). What it contributes, though, is not a learned
skill but a randomised fine-alignment search under contact.

| run | goal at | RL calls | RL steps | own steps | reward successes per call | policy replay | goal inside an RL call |
|---|---|---|---|---|---|---|---|
| hard, seed 0 | 2,660 | 3 | 2,479 | 205 | 16, 6, 7 | none | no, 43 steps after |
| hard, seed 1 | 11,422 | 3 | 13,167 | 239 | 0, 0, 11 | 1 of 3 | yes |
| hard, seed 2 | 1,574 | 4 | 19,298 | 5,276 | 0, 2, 0 | none | yes |
| medium, seed 0 | 1,265 | 3 | 17,350 | 84 | 0, 0, 0 | none | yes |
| medium, seed 1 | 1,651 | 1 | 4,096 | 137 | 0 | none | yes |
| medium, seed 2 | 2,612 | 2 | 11,374 | 101 | 0, 0 | none | yes |

Of fifteen accepted RL calls the trained policy reproduced success on replay exactly
once. The insertions happen while the learner is still exploring: 1 cm end-effector
deltas under the 80 N force limit, jiggling the prong until geometry and friction accept
it. That is a contact-rich search the straight-line controller cannot express.

Most agent-written rewards are position trackers, not task rewards. The first reward from
hard seed 0, verbatim, rewards holding the end effector at a fixed offset above the
outlet centroid and never mentions the plug, so nothing in it distinguishes a seated plug
from one held above the socket:

```python
def reward(particles, visible, ee_pos, ee_quat, gripper):
    outlet = np.array(particles['outlet'])
    ox, oy = float(outlet[:, 0].mean()), float(outlet[:, 1].mean())
    outlet_top = float(outlet[:, 2].max())
    ex, ey, ez = ee_pos
    xy_err = ((ex - ox) ** 2 + (ey - oy) ** 2) ** 0.5
    z_err = abs(ez - (outlet_top + 0.0388))
    return float(1.3 - 10.0 * xy_err - 15.0 * z_err)
```

Twice an agent reached for RL not for precision but because straight-line moves were
failing outright, reasoning that the tool "acts via small incremental deltas rather than
straight-line IK". On Airport that produced two stagnated calls and no progress.

### Every reward the agents wrote

Twelve rewards across the sweep, written from the tool description and the particle
files with no examples to copy. They fall into five kinds, and the kind an agent chose
predicted whether its reward could see success at all. Donut is absent because no donut
run ever called the tool.

| kind | what it measures | used in | reward successes | policy replay |
|---|---|---|---|---|
| pose tracker | end effector held at a fixed offset above the outlet centroid; never mentions the plug | plug hard, seed 0 (3 calls) | 16, 6, 7 | none |
| dead reckoning | prong tip inferred from the EE pose plus measured grasp and tip offsets, socket centre hardcoded | plug hard, seed 1 (last call) | 11 | yes, the only one |
| visible-geometry depth | lowest visible plug points against the outlet top face | plug medium, all 3 seeds; plug hard, seed 1 (2 calls) | 0 | none |
| goal detector | any item centroid inside the table footprint, distance-shaped while outside | airport, seed 1 | 0 | none |
| mechanism proxy | the pusher becoming visible, as a stand-in for a successful press | airport, seed 1 | 0 | none |

Only the dead-reckoning reward produced a policy that reproduced success on replay, and
it is the only reward that does not depend on seeing the prong. The agent measured the
grasp offset and tip height once from particles, hardcoded the socket centre, then
computed the tip from the end-effector pose, which stays observable after the prong
disappears into the socket. Working around the occlusion was something one agent
discovered on its own:

```python
def reward(particles, visible, ee_pos, ee_quat, gripper):
    ee = np.asarray(ee_pos)
    hole_xy = np.array([1.3477, 0.9495])       # measured earlier from particles
    outlet_top_z = 0.2277
    grasp_offset = np.array([-0.0017, -0.0055])
    tip_z_offset = 0.0544                      # prong tip below the EE frame
    tip_xy = ee[:2] + grasp_offset
    tip_z = ee[2] - tip_z_offset
    xy_dist = float(np.linalg.norm(tip_xy - hole_xy))
    depth = max(0.0, outlet_top_z - tip_z)
    depth_error = max(0.0, 0.0096 - depth)
    r = -100.0 * xy_dist - 100.0 * depth_error
    if xy_dist < 0.003 and depth >= 0.008:
        r = 1.0 + (0.008 - xy_dist) * 10.0
    return float(np.clip(r, -10.0, 2.0))
```

Every reward built on the plug's visible lowest points reported zero successes on both
tiers. They are well written and would be correct with a second camera; they fail
because the quantity they measure stops existing at the moment of success. Plug medium,
seed 2, is representative:

```python
def reward(particles, visible, ee_pos, ee_quat, gripper):
    plug, outlet = np.asarray(particles['plug']), np.asarray(particles['outlet'])
    order = np.argsort(plug[:, 2])
    tip = plug[order[:max(1, len(plug) // 8)]].mean(axis=0)   # lowest points = prong tip
    outlet_top = outlet[:, 2].max()
    horiz_err = float(np.linalg.norm(tip[:2] - outlet[:, :2].mean(axis=0)))
    depth = float(outlet_top - tip[2])
    r = -horiz_err * 20.0
    r += depth * 60.0 if depth > 0.0 else -2.0 * min(-depth, 0.02)
    return float(r)          # success = depth > 0.015 and small horiz_err, never observed
```

On Airport the two rewards are of different kinds and both stagnated. The first is a
correct goal detector; it fails because its shaping term rewards the arm being near an
item and the item being near the table, neither of which a policy can change from a pose
above the button, and the mechanism that actually solves the task, a timed press, is not
reachable by centimetre end-effector deltas:

```python
def reward(particles, visible, ee_pos, ee_quat, gripper):
    table_x_min, table_x_max, target_y, best = 1.65, 2.35, 0.9, -2.0
    for i in range(5):
        pts = np.array(particles.get(f"item_{i}", []))
        if pts.shape[0] == 0:
            continue
        x, y, z = pts.mean(axis=0)
        if table_x_min <= x <= table_x_max and y > 0.72:
            return 1.0                                  # item is on the table
        if x < 1.5:
            continue
        best = max(best, -abs(target_y - y) - 0.5 * np.linalg.norm(pts.mean(axis=0) - ee_pos))
    return float(best)
```

The second replaces the goal with a proxy for the mechanism, returning 1.0 as soon as
the pusher is visible at all, on the reasoning that a visible pusher means the press
worked. And in the discarded pre-`wait` Airport runs one agent submitted
`return (ee_pos[2] - 1.0) / 0.5`, a reward for lifting the gripper, purely to make the
conveyor advance while spending few turns.

## 4c. When runs stop, and why

Four limits are configured; only two ever ended a run, and the interaction budget was
never one of them (largest consumption 24,574 of 100,000).

| limit | value | ended a coarse run | ended an RL run | largest observed |
|---|---|---|---|---|
| conversational turns | 120 | 3 | 1 | 121 (the cap) |
| agent stops on its own | none | 9 | 11 | -- |
| environment interactions | 100,000 | 0 | 0 | 24,574 (25%) |
| API spend | $8.00 | 0 | 0 | $4.75 |
| wall clock | 10 h | 0 | 0 | 102 min |

Twenty of twenty-four runs ended because the agent decided it was finished; sixteen were
right. In the other four the agent asserted success while the goal predicate did not
hold, confidently and specifically: "the block now sits flush on the outlet's top surface
and stays upright with no support" (hard, coarse, seed 1, stopped at 158 interactions,
prong never entered), and "item_2 vanished from the belt's item cycle ... consistent with
landing on the adjacent table" (airport, RL, seed 0, stopped at 408; replay puts the item
at y = 0.689 with the belt edge at 0.707). Both are inferences from absence: a plausible
geometry read as a seat, an object leaving the camera frame read as an object leaving the
belt. Self-stopping is the largest source of unforced failure in the sweep.

Three of four coarse-control failures ended at the turn cap, and coarse runs make more
tool calls per run than RL runs (31 move_to calls on average against 22). A coarse failure
therefore means "ran out of conversation or quit", never "exhausted the interaction
budget", while RL runs spend interactions freely. Rerunning the coarse cells with a
400-turn cap would settle the comparison.

## 4d. How the agent uses perception and time

Calls per run, six runs per domain, both conditions pooled:

| domain | pixels_to_particles | wait | what for |
|---|---|---|---|
| Donut | 2, 2, 2, 2, 2, 3 | none | locate donut, locate target, confirm placement |
| Plug, medium | 8, 10, 11, 12, 15, 23 | 0 x5, 31 | socket centre, prong depth, re-check per descent |
| Plug, hard | 7, 7, 8, 10, 10, 27 | 0 x5, 16 | as above, plus verifying the plug stays seated |
| Airport | 5, 12, 23, 26, 32, 44 | 3, 8, 18, 20, 23, 30 | track the item, estimate belt speed, time the press |

Particle use scales with the precision demanded: two or three calls for a centroid, 7 to
27 interleaved with descent steps for insertion. `wait` has two distinct uses. On Airport
all six runs use it (3 to 30 calls, 10 to 500 steps each) because simulation time only
advances when the agent acts, so waiting is the only way to let the conveyor bring the
item round, and the agent samples particles during the interval to estimate belt speed.
In the two insertion runs that use it, the agent is recovering from a dropped or wedged
plug and waits in blocks of 100 to 400 steps to confirm the object has stopped moving
before trying again -- a stability test the tool surface does not otherwise provide.

## 4e. One trajectory in full

Hard tier, coarse control, seed 0, the fastest insertion in the sweep. One particle call
to locate the plug, a grasp, one particle call to locate the socket, then a descent in
shrinking increments of 100, 50, 12 and 8 mm with a particle check between each. The goal
holds at interaction 87 on the 8 mm step; the agent spends 19 more interactions verifying
and releasing.

| call | tool | arguments | steps | cumulative | goal |
|---|---|---|---|---|---|
| 1 | pixels_to_particles | max_points 64 | 0 | 0 | -- |
| 2 | move_to | (1.3695, 0.5475, 0.450) open | 19 | 19 | -- |
| 3 | move_to | (1.3695, 0.5475, 0.270) open | 18 | 37 | -- |
| 4 | move_to | (1.3696, 0.5475, 0.2725) close | 9 | 46 | -- |
| 5 | move_to | (1.3696, 0.5475, 0.450) keep | 9 | 55 | -- |
| 6 | pixels_to_particles | max_points 64 | 0 | 55 | -- |
| 7 | move_to | (1.3477, 0.9495, 0.450) keep | 21 | 76 | -- |
| 8 | pixels_to_particles | max_points 64 | 0 | 76 | -- |
| 9 | move_to | (1.3492, 0.9502, 0.350) keep | 6 | 82 | -- |
| 10 | pixels_to_particles | max_points 64 | 0 | 82 | -- |
| 11 | move_to | (1.3492, 0.9501, 0.300) keep | 3 | 85 | -- |
| 12 | move_to | (1.3493, 0.9501, 0.288) keep | 1 | 86 | -- |
| 13 | move_to | (1.3493, 0.9501, 0.280) keep | 1 | 87 | **holds** |
| 14-19 | particles, move_to x3, particles | verify, release, retract | 19 | 106 | holds |

The same domain in the RL condition, seed 0: coarse moves to a pose above the socket by
interaction 138, then three RL calls with budgets of 4,000, 6,000 and 15,000 and episode
lengths of 40, 30 and 50. They consumed 1,250, 229 and 1,000 interactions and reported
16, 6 and 7 reward successes; none of the three policies reproduced a success on replay.
The goal held at 2,660, forty-three interactions after the last call returned.

A per-run ledger of all 24 runs (termination, cost, wall clock and per-tool call counts)
is in `RESULTS_sweep1.md`. Videos of all twelve replayable successes are in
`~/arc_outputs/videos`; five are embedded in the published web version of this report.

## 5. Reproducibility

PyBullet is deterministic for a fixed action sequence, so any run whose
success precedes its first RL call can be replayed exactly from its event
log. All twelve such successes were replayed and rendered to video: every one
reproduced its logged first-success interaction with zero per-call
step-count mismatches, which also validates the interaction accounting.
Videos are in `~/arc_outputs/videos/`, one per run, with a caption strip
showing the current tool, the interaction counter and the goal state.

## 6. What went wrong along the way

Twenty-one issues are root-caused in `DEBUG_LOG.md`. The ones that changed
conclusions rather than just costing time:

- **A numerical bug in the stack, not our code.** The OpenBLAS bundled with
  numpy 1.23.5 returns wrong rows for tall-skinny matrix products when
  multithreaded on these AVX-512 Xeons, corrupting a quarter of all
  back-projected particles. Everything now back-projects with `einsum` and
  pins `OPENBLAS_NUM_THREADS=1`.
- **Physics that let the robot cheat.** PyBullet's position controller
  applies unbounded torque, so a misaligned plug was simply forced through a
  socket wall. Capping arm torques at three times the URDF limits and adding
  the 80 N contact-force limit made insertion an actual alignment problem.
  Raw URDF limits were too low: the gravity-blind controller then sagged the
  arm by 20 cm.
- **Grasps that silently released.** The environment releases a grasp when
  the commanded finger target exceeds the current position, so "hold closed"
  at the nominal closed value read as an open as soon as a held object
  shifted, and RL calls aborted with the object on the floor.
- **A task that was impossible as written.** Airport's button had zero travel
  and counted as pressed only when driven into its stand by unbounded force.
  It is now a contact switch.
- **Goals that could be satisfied while cheating.** `InTarget` and `OnTable`
  now require the object to be at rest and not held; the first smoke run
  "succeeded" while carrying the donut above the target.
- **An experiment confound from the harness side.** Runs truncated by the
  account usage limit are now detected and marked invalid rather than
  counted as failures.

## 7. Limitations

Single fixed camera, so occlusion is severe exactly where precision matters.
Object names come from the segmentation mask and depth is perfect, both
cheats a real perception stack would have to earn. Three seeds, one harness,
one model. The `model_based` condition has not run. Success is scored by a
predicate the agent cannot see, which is right for measurement but means the
agent's own stopping decision is unmeasured.

## 8. Next steps, in order

1. **Fix the occlusion.** Add a second camera, or teach the prompt to build
   rewards from visible geometry only, for example the height of the plug
   block rather than the position of the hidden prong. Without this the RL
   tool cannot report its own success and burns its whole budget.
2. **Implement the model-based backend** so the third condition is real: an
   action-conditioned PTv3 particle model on the already-logged transitions,
   with sampling-based MPC scored by the same agent reward.
3. **Harder insertion, and more seeds.** The medium tier is solvable
   open-loop; sub-millimetre clearance or a wall-mounted socket would
   separate the conditions more sharply, and five to ten seeds would make the
   differences statistically meaningful.
4. **Run OpenCode** for the harness comparison, on an API key, which also
   removes the account usage limit from the critical path.
