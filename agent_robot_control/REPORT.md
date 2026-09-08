# Agent harnesses as robot controllers, with RL as a motor-skill subroutine

Report on the first round of experiments. Code in `agent_robot_control/`,
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
