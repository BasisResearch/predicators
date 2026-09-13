# Four-block Bridge placement audit

This audit concerns MB seeds 1 and 2 from `bridge-agent_continual_span_transfer_r1`, frozen at `3e95b1798c6fc7b6e24dccf0fd0b6702c8801d94`.
It does not change their outcomes or authorize resuming held Bridge comparisons.

## Seed 2: candidate geometry and the terminal certificate

The last L02 invocation ends at step 1267 with both `Bridged` directions, both supporting `SeatedOn` relations, and all three span attachment pairs present in its recorded environment atoms.
Thus the recorded candidate satisfies the geometric goal and has registered attachments.
The generic rejection suffix, "an unwelded row cannot stand", is not a diagnosis of this construction.

`PyBulletBridgeEnv.check_episode_trajectory` advances the live physics client for 60 raw substeps and then rechecks the geometric goal.
The normal episode step appends its observation before calling that certificate and returns that same observation afterward.
The post-certificate state is not appended to the episode trajectory.
Consequently, the saved candidate geometry does not show which relation failed during certification.
The certificate also bypasses ordinary domain updates, including resting-weld relaxation.
The gripper has not necessarily completed the Place retreat when the geometric goal triggers termination.
These are concrete evaluation and observability concerns, but they do not establish that the rejection itself was incorrect.

Earlier full recorded-action and public-controller replays diverged before the final placement, so they cannot determine the original failure's physical cause.
See [the seed-2 investigation](bridge-span-mb-seed2-analysis.md) for the replay limitations and the successful matched-layout oracle-scene run.

## Seed 1: rehearsal existed, but its assumptions did not cover execution

The play transcript explicitly rehearses PickBlock plus Place, including 27 nearby placement targets, before executing `PickBlock(span1)` followed by `Place[0.7206, 1.3004, 0.5320, 0.0093]`.
The real pick consumes 46 steps and Place consumes 128 steps, ending at L02 step 1344.
The following observed state and saved render show leg0 lying flat and the span sloping down from the other leg.
The transcript's inspection of recorded states places the toppling within the first 20 steps of that Place invocation, during the carry.
Its causal account of the exact contacting body remains an interpretation, because contact manifolds were not recorded.

The public Place controller does perform motion planning.
Its collision context includes transitively welded followers, using ideal weld transforms rather than their instantaneous moving poses.
MoveAbove targets the destination XY at transport height; it does not first require a vertical lift at the starting XY.
The planner is kinematic and does not certify dynamic tracking error, grasp compliance, or transient contact during the physics substeps.
Bridge's configured carried-object bystander clearance is 1 cm, and the general planner permits contact penetration down to its configured -5 mm hard margin in the applicable contact cases.
Neither is a guarantee that a movable leg cannot be disturbed.

Both run configurations enable `pybullet_pin_held_weld_assemblies`, motion planning, and a 3 N placement preload.
The pinning occurs once after an action's 20 physics substeps and poses followers relative to the actual held root, not an exactly tracked gripper pose.
Therefore the transcript's description of a "floppy" span should not be accepted as proof of flexible welds: whole-assembly tilt at the grasp and within-action constraint error must be distinguished from deformation between blocks.
The 3 N preload is also not established as the cause of the initial carry collision.

The saved learned simulator is a no-op residual, and the agent's rehearsal consequently does not establish a prediction of the real bonded assembly dynamics.
Successful geometric/controller rehearsal and a physically safe welded carry are different claims.

## Recommended repair order

1. Record pre- and post-certificate poses, velocities, contact information, attachment state, and the specific failed geometric relations, with a render of the rejected state.
2. Reproduce the carry through the public controller in a diagnostic run that captures native physics snapshots and contact traces before the collision and release.
3. Test lift-before-translate transport and assembly-clearance tracking against that reproduction, separating grasp motion from weld error.
4. Validate the released rigid assembly under the same intended physics semantics used during ordinary execution before changing the terminal certificate.

Do not count a transient geometric match as a win or loosen the four-block task based solely on the current video.
Apply any shared controller repair to both MB and MF, and keep resulting experiments separate from this frozen cohort.

## Follow-up measurements, 2026-09-13

The direct [recorded-pose measurement](/home/ycliang/predicators/logs/bridge_placement_diagnostic_20260913/recorded-geometry.json) materially narrows seed 1's diagnosis.
While span1 remains held during the final Place, the other span blocks' positions relative to span1 change by at most 0.00039 mm across recorded action boundaries.
Thus the saved trajectory shows a rigid assembly moving as a whole, not a beam bending at its welds.
Before leg0 is disturbed at step 1236, the grasped block's pitch reaches approximately 15.9 degrees and span3 is as much as 55.5 mm below span1.
The exact within-action contact impulse is not recorded, so these measurements do not identify the contacting robot link or span block.
They do establish that the earlier description of a persistently floppy welded span was inaccurate.
The agent-facing transcript contains noisy observations; the figures above use the environment recording directly.

Compute job 22680231 ran three reconstructed public-controller carry variants and three initial settling diagnostics.
The public PickBlock consumes the same 46 steps as the original in the baseline reconstruction, but the reconstructed Place rejects its descent instead of reproducing the original leg toppling.
Native substep traces nevertheless show up to 25.1 mm of transient weld position error during this reconstructed carry before the ordinary action-boundary rigidity correction.
A vertical-lift-first diagnostic reduces the measured carry error to 8.2 mm; correcting rigidity every substep reduces it to 2.8 mm.
All three variants reject the descent, so none demonstrates a repaired successful placement.
These variants have different generated trajectories and should not be interpreted as a controlled stiffness-only effect on task success.

The first seed-2 reconstruction had zero body velocities and retained initialization motor targets, which made its raw certificate fail while a new hold command survived.
That is a reconstruction artifact and is excluded from conclusions about the original failure.
The corrected reconstruction restores recorded body velocities and the final action's motor command.
All five variants in job 22680275 survive: the raw certificate, raw physics with periodic weld relaxation, ordinary hold steps, a new hold motor target without domain updates, and disabled robot-to-block collisions.
Job 22680301 repeats the raw certificate with robot contact traces and tests full finger opening and ordinary steps retaining the final command; all three survive and record no positive-force robot-to-span contact.
Job 22680312 additionally reconstructs weld frames from step 1266, the last held observation; all four tested certificate/control variants also survive.
Consequently, these tests do not establish that robot pressure, incomplete release, or skipped weld relaxation caused the original seed-2 rejection.

Job 22680283 applies the original seed-1 action suffixes to reconstructed late scenes.
Those trajectories also diverge, including from the first post-restoration action in the held-state cases, and their contacts do not reproduce the original leg0 toppling.
They are diagnostic reconstructions, not new agent results or faithful counterfactuals for the original run.

The original seed-2 rejection remains mechanically unresolved because the recorded terminal trajectory omits the post-certificate state and does not preserve all native constraint, joint-velocity, and contact-solver state.
The next reliable reproduction should capture that state in-process before certification and before the critical carry, preserving native snapshots for branching comparisons.
The concrete repair priorities are terminal-state diagnostic recording and transport that verifies actual whole-assembly clearance before crossing a support.
There is no evidence here for relaxing the task's success criterion, changing historical scores, or treating either failed MB seed as solved.
All unfinished Bridge baseline jobs remain held.
