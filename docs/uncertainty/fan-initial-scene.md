# Fan initial scene and complete-recording preflight

September 12, 2026.
This development experiment composes the Fan components into a declared root-state law for the [uncertainty simplification plan](simplification-proposal.md).
It is an offline model and support investigation, not a fitted posterior or a production replacement.

## Inputs and original law

The experiment uses the previously frozen first Fan training level: 132 actions, 133 public noisy observations and the latest saved cycle-000 simulator program.
The program and recording hashes remain in the [experiment manifest](../../logs/uncertainty_fan_full_scene_v2_20260912/plan.json).
No task generator, privileged recording velocities or evaluator reset state initializes a candidate.
The unchanged dynamics program has one parameter, `fan_speed`, with the explicitly declared uniform original prior on [0, 1] described in the [provenance audit](offline-fitter-comparison.md#fan-prior-provenance).
The scene law below is independent of that airflow parameter.

| Quantity | Declared original distribution or input | Continuous dimensions after initial conditioning |
| --- | --- | ---: |
| Robot positions | Independent zero-mean Gaussian reset coordinates, standard deviation pi for revolute joints and 0.1 m for prismatic joints; condition nine controlled positions on their exact readings and retain their density | 4 |
| Robot motion | Probability 0.8 on all joints resting; otherwise independent uniform velocities on [-0.1, 0.1] in each joint's units per second | 0 or 13 |
| Ball supported component | Probability 0.8; uniform workspace xy eroded by radius, uniform yaw, height at table plus radius, zero twist | 3 |
| Ball free component | Probability 0.2; uniform eroded workspace xyz, Haar orientation, independent uniform linear velocities on [-0.1, 0.1] m/s and angular velocities on [-0.2, 0.2] rad/s | 12 |
| Ten fixed fixture positions | Independent uniform xyz, with x/y bounds equal to the public workspace expanded by 0.3 m and z in [0, z_ub]; zero base velocity under the fixed mounting law | 30 |
| Four switch bases and target yaw | Independent uniform angles on [-pi, pi] | 5 |
| Four boundaries and one obstacle yaw | Independent uniform mass on four quarter turns; boundary body dimensions are derived from their exact world extents and the selected rotation | 0, with 4^5 discrete cases |
| Fan-bank bases | Derived from the visible placement routine, with noisy fan-pose observations retained in the likelihood | 0 |
| Twenty rotor joints | Independent uniform finite-winding positions on [-pi, pi] and velocities on [-2pi, 2pi] rad/s | 40 |
| Four switch sliders | The [conditioned rest/motion law](fan-articulated-prior.md), with rest mass 0.8 and moving velocity half-width 0.1 m/s; retain the probability of each exact initial switch flag | 0 to 8 |
| Attachments and model memory | No initial attachment or queued command under this declared root protocol; the frozen program has no declared memory; later commands evolve in replay | 0 |

This defines 82 to 112 continuous initial-state coordinates, or 83 to 113 jointly with `fan_speed`.
The two robot-motion cases, two ball cases, sixteen conditional switch-motion cases and 1,024 fixture-orientation cases give 65,536 combinations before geometric support and other observation constraints.
These are representation counts, not a claim that every case retains positive posterior mass or that the sampler has explored them.
The ball's unobserved orientation is deliberately retained here; no unverified spherical-symmetry reduction is used.
Hyperparameters and the independent fixture law are engineering assumptions, not estimates of the task generator's calibrated distribution.

## Geometry and conditioning

Quarter turns preserve the visible box reconstruction's relation between body-frame dimensions and exact world-axis-aligned extents.
The obstacle wall has its source-defined physical dimensions, while boundary dimensions come from the supplied exact descriptors.
The original noisy positions and rotations are likelihood evidence, not fixed geometry.

The mounting model permits overlapping mass-zero fixtures to form a fixed compound geometry.
It does not treat them as free rigid bodies that should separate under contact forces.
The feasibility gate rejects a whole candidate if the robot or ball penetrates any other native body beyond the declared 1e-7 m geometry roundoff.
The supplied Fetch wheel/floor fixture intersections are the sole pair-specific exception and must equal the previously audited -0.011075 m distance under this initialization protocol.
This is the stated simulator support policy, not certification of arbitrary physical mounting arrangements or every possible robot self-contact model.

Airflow speed does not affect these initial collision shapes or the placement protocol.
Consequently the whole-scene support normalizer is independent of that parameter and cancels within this fixed posterior target.
The gate rejects the entire draw, including its motion and orientation cases; it does not normalize each case separately.
Acceptance rates from observation-informed proposals are not estimates of the original prior's support normalizer.

The proposal draws continuous positions and non-quarter fixture yaw from truncated Gaussians around the initial noisy readings, restricted to the declared original support.
Every such coordinate retains `log p0 - log q`.
Quarter-turn proposals mix 98% of the angle-likelihood-weighted categorical distribution with 2% of the original uniform distribution.
This keeps all four cases reachable even when exponentiating a very small likelihood would otherwise underflow to zero.
Their original masses and proposal probabilities receive the same correction.
The first scene prototype lacked that categorical mixture and remains a geometric diagnostic, not a validated posterior sampler.
No initial measurement is removed from the complete likelihood to compensate for proposal use.

The candidate is initialized from sampled values, source-derived fan placements and exact descriptors before action replay.
All native robot, slider and rotor positions and velocities are explicitly installed, and full ball orientation and motion are preserved.
Recorded actions then run uninterrupted in the candidate simulator.
The likelihood uses the previously declared [complete output-discrepancy composition](orientation-discrepancy.md), including its retained exact event constraints; it does not add Boolean sensor noise to force acceptance.

## Native results and limits

The corrected scene preflight `22640582_0` accepts eight scenes from 23 proposals, rejecting fifteen for the declared geometry policy.
Every accepted scene exactly repeats its 32-action continuation in a fresh second world, including robot joints, articulated joints and projected observations at all 33 boundaries.
Seven candidates have finite complete-output likelihood; one violates a later exact event.
These eight cases do not establish prior calibration or usable posterior uncertainty.

The earlier eight saved physical candidates were also replayed for all 132 training actions in job `22640308_0`.
All initial predictions and first-32-action likelihoods match their saved counterparts, and both full continuations repeat exactly.
However, none has finite full-recording likelihood: seven first disagree on switch timing and one first disagrees on target-hit timing.
This is why a positive short-prefix result cannot close the recorded-prediction gate.

| Saved draw | First exact-event disagreement | Step |
| --- | --- | ---: |
| 1 | Fan 2 stays on after the observed off transition | 70 |
| 4 | Fan 2 is off when the observation is on | 17 |
| 6 | Fan 2 stays on after the observed off transition | 70 |
| 8 | Fan 2 stays on after the observed off transition | 70 |
| 10 | Fan 0 is off when the observation is on | 98 |
| 13 | Target-hit prediction becomes true too early | 108 |
| 19 | Fan 0 is off when the observation is on | 98 |
| 22 | Fan 2 stays on after the observed off transition | 70 |

The follow-up [parameter support scan](../../logs/uncertainty_fan_event_support_20260912/plan.json) holds saved scene 13 fixed and probes airflow speed over the original [0, 1] support, including historical fitted values as numerical probes.
It uses the full training recording to search for event-compatible parameters; it is neither a parameter estimate nor a held-out prediction comparison.
A failed finite scan would not prove that the joint initial-state/parameter target has empty support.

An interim scan at speed 0.09 moves the first disagreement to the final switch-off event at step 132, with two mismatched readings from the switch and its linked fan.
Thus the candidate's earlier target-first failure did not establish that every switch transition matched.
The [fixture-placement support probe](../../logs/uncertainty_fan_switch_support_20260912/plan.json) fixes speed at 0.09 and varies switch 0's initial xy by up to 2.5 mm within the unchanged uniform placement prior.
Its [scope addendum](../../logs/uncertainty_fan_switch_support_20260912/scope-audit.json) distinguishes the actual placement scan from an inherited description of the preceding speed scan.
It checks the complete 132-action history for each point and does not reuse the source proposal weight after modifying the scene.
These directed probes use the full training data and remain separate from held-out predictive evaluation and posterior sampling.

## Runtime provenance

The first scene preflight ran on node3504, not node1412 as incorrectly stated by inherited launcher metadata.
Its [runtime addendum](../../logs/uncertainty_fan_full_scene_20260912/runtime-audit.json) records the authoritative allocation without changing the raw worker or result artifacts.
Related earlier articulated diagnostics have corresponding addenda, linked from their own experiment directories.
The full-history audit records its actual node1390 Intel Xeon Gold 6230 runtime, and the corrected scene preflight records node1622 AMD EPYC 9654.
Both record Python, NumPy, SciPy, PyBullet API and hash-seed information at worker startup.
Within-job exact repeats remain measured evidence; identical random seeds across different CPUs are not claimed to generate identical physical candidates.

Artifacts: [corrected scene report](../../logs/uncertainty_fan_full_scene_v2_20260912/pilot-22640582_0.json), [saved-scene full histories](../../logs/uncertainty_fan_full_history_20260912/pilot-22640308_0.json), and [initial scene prototype](../../logs/uncertainty_fan_full_scene_20260912/pilot-22640065_1.json).
