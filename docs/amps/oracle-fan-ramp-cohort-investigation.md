# Oracle Fan + Ramp: nine-seed failure audit

Audited September 22, 2026, using the frozen `ramp_skill_repair_r1` runtime.
Oracle solved both levels in seeds 0, 2, 3, and 5: 4/9 full successes.
All nine seeds solved training; seeds 1, 4, 6, 7, and 8 failed test.
None of those five failures was a usage-limit termination or a step-budget exhaustion.
Seeds 1 and 6 explicitly gave up with the ball wedged; seeds 4, 7, and 8 received a fall rejection.

## What failed

| Seed | Failure supported by the execution record | Interpretation and limits |
| --- | --- | --- |
| 1 | A braking switch took 45 actions instead of the reconstructed rehearsal's 20; a following 150-action wait carried the ball past the landing. | Previous controlled replays isolate scene-dependent switch timing; exact-state Oracle predicts the danger. |
| 4 | The ramp crossing succeeded, the subsequent turn stopped at a platform junction, and repeated recovery pushes eventually carried the ball off the edge. | The trace reports a shortened braking sequence and ineffective cross-junction pushes. The precise contact mechanism is not established by the journal's gap calculation. |
| 6 | A hand-integrated traverse predicted a safe coast, but the ball accelerated down the ramp and overshot into a wedge. | The hand model used the wrong action-time scale. Its claim that native ramp physics was four times too strong is misleading. |
| 7 | The final positioning request predicted about 0.117 m of travel and instead lost the ball over the side. All three preceding skills succeeded. | The decision used a hand-estimated pulse response. The later claim that platform 2 has half the friction is contradicted by the native material assignments. |
| 8 | A braking switch took 46 actions instead of the reported prediction of 29; the ball crossed the open landing edge. | The actual trace confirms 46 actions and the fall. Scene/controller timing is the leading explanation, but unlike seed 1 it has not been isolated with a switch-pose intervention. |

The scorecard's seed-4 test rollout count is zero, but its session contains simulation work.
Do not interpret that counter as evidence that it never simulated, especially across resumed sessions.

## Two misleading postmortems

Seed 6 constructed its own model with `g=0.01663` in metres per action squared, then computed the ramp term as `(5/7)*g*0.0075`.
The frozen native engine uses gravity 10 m/s², a 1/240-second physics step, and 20 physics steps per action.
An action therefore spans 1/12 second, giving gravity 0.06944 m/action² and the same approximate rolling-sphere ramp term 0.000372 m/action².
The agent's term was 0.0000891, smaller by a factor of 4.18.
The corrected approximation is close to its retrospective estimate of 0.00036; this is not evidence that the provided simulator secretly applies four times the intended ramp gravity.
The approximation does not account for every contact transient or drag term.
Native replay of the recorded unpowered ramp segment gives net acceleration about 0.000243 m/action².

Seed 7 inferred a friction change from different pulse displacements.
Native inspection gives platform 1 and platform 2 identical lateral friction 0.5 and rolling friction 0.0001, with both top surfaces at z=0.4.
Its claimed 2.8 mm height difference is also absent from the exact generated scene.
Different displacements therefore cannot establish the asserted difference in material coefficients.
Contacts, initial motion, and the validity of the pulse approximation require separate attribution.
The established decision error is using a prediction that underestimated the dangerous excursion, despite already observing a larger response during the preceding maneuver.
In an additional compute-node check (`23463512`), a fresh supplied Oracle was initialized from the native replay at test action 3387 and given the final SwitchOn, Wait[5], SwitchOff, Wait[200] sequence.
It predicts the ball leaving the platform and falling to the floor.
This demonstrates that the supplied dynamics can flag this action from a faithfully reconstructed decision state; it does not establish that the agent's historical noisy reconstruction would have done so.
The replay prefix retains the approximately 0.00195 maximum joint discrepancy noted below.

## What the successful runs tell us

The supplied Oracle supplies the domain dynamics, but still reconstructs execution state from noisy observations and leaves planning to the agent.
It is not an oracle scene estimate or an oracle controller.
Several successful and failed runs encountered the same noisy junctions and variable actuation durations.
The successful seed-5 journal describes averaging static geometry, using fresh simulation worlds, checking several planner seeds and geometry perturbations, braking during descent, and choosing recoverable undershoots.
Seed 0 adjusted braking using the ball position observed after the switch completed.
These support a practical explanation for the variability: success depends on how the agent corrects its reconstructed scene and uses the model, not merely whether correct dynamics were supplied.

These results do not isolate an EMPIRIC learning advantage.
EMPIRIC and both ablations currently solve 5/5 on this layout, and the Oracle traces themselves include manual calibration and model approximation.
Comparisons would need matched decision-state tests to distinguish better inference, better planning, and variation in agent choices.

## Recommended next work

1. Make simulation metadata explicit: action duration, physics substeps, deployed model identity, and the scene estimate used for the rollout.
   Seed 6 is a concrete regression case for interpreting time units.
2. Verify scene edits against the actual collision bodies in both fresh and reused simulators.
   Agents repeatedly report that geometry edits do not have the expected effect; those reports warrant end-to-end tests, not acceptance as API facts.
   Use accumulated observations and coherent geometry estimates instead of treating independently noisy platform heights as exact.
3. Evaluate risky plans over plausible switch poses and actuation latencies, and report the maximum excursion and remaining recovery margin.
   Repeating one nominal scene is insufficient when that scene predicts the wrong switch timing.
4. Reassess after a timing or motion discrepancy, before executing a long wait or a positioning pulse near an open edge.
   Stopping the robot alone does not stop the ball; the continuation must account for active fans and momentum.
5. Preserve saved decision points from these failures for matched exact-state, reconstructed-state, and alternative-action checks before changing shared skills or enabling mandatory preflight.

A preflight that only checks whether a switch skill succeeds would miss the immediate problem in four of the five failed test runs, which recorded zero failed skill invocations.
The useful check is whether the full maneuver remains safe under state and timing error.
No agent, controller, or environment change was made for the two additional seeds.

## Evidence and reproduction

- [Seed 1 controlled attribution](oracle-fan-ramp-seed1-investigation.md).
- [Seed 4 execution and recovery](/home/ycliang/predicators/logs/agent_continual_oracle_dynamics/fan_ramp-oracle_dynamics_opus_ramp_skill_repair_r1/seed4/run_20260921_121409/agent/004_play_20260921_132651.md:2397).
- [Seed 6 hand model](/home/ycliang/predicators/logs/agent_continual_oracle_dynamics/fan_ramp-oracle_dynamics_opus_ramp_skill_repair_r1/seed6/run_20260921_155606/agent/003_play_20260921_173628.md:507).
- [Seed 7 final action](/home/ycliang/predicators/logs/agent_continual_oracle_dynamics/fan_ramp-oracle_dynamics_opus_ramp_skill_repair_r1/seed7/run_20260921_164759/agent/002_play_20260921_190143.md:2878).
- [Seed 8 delayed braking](/home/ycliang/predicators/logs/agent_continual_oracle_dynamics/fan_ramp-oracle_dynamics_opus_ramp_skill_repair_r1/seed8/run_20260921_164759/agent/002_play_20260921_184746.md:1420).
- [Successful seed 5 strategy](/home/ycliang/predicators/logs/agent_continual_oracle_dynamics/fan_ramp-oracle_dynamics_opus_ramp_skill_repair_r1/seed5/run_20260921_155606/agent/sandbox/journal.md).

Compute array `23463338` replayed saved low-level test actions for seeds 4, 6, 7, and 8 in the original frozen runtime.
All reproduce the qualitative terminal failure: seed 6 wedges beyond the landing and the other three fall.
This is not a claim of bitwise trajectory equivalence: maximum robot-joint differences from the saved states were approximately 0.29, 4.5e-7, 0.00195, and 1.1e-8 respectively.
In particular, seed 4's divergence limits precise timing attribution from this replay.
Native material and engine metadata were read directly from each recreated environment.
The driver is [audit_oracle_ramp_20260922.py](/home/ycliang/predicators/logs/audit_oracle_ramp_20260922.py), with outputs `logs/oracle-ramp-audit-23463338-{4,6,7,8}.log`.

## Additional seeds

Seeds 9 and 10 were submitted as array `23463117` and started on compute nodes using accounts b and d.
The launch flags and arguments were checked against the previous extra-seed config; only the seed range changes.
The [launch configuration](/home/ycliang/predicators/scripts/configs/predicatorv3/continual_fan_ramp_oracle_extra_r3.yaml) retains the frozen runtime and original cohort identifier.
The Markdown benchmark tracks all eleven seeds, retaining all failures.
Replacement plot monitor `23463368` refreshes the report and figures every 60 seconds when results change.
The expanded-cohort report tests passed: 8 tests.

## Prompt alignment and current comparison

The working-tree prompts now share state-estimation, action-time, and discrepancy-response guidance between EMPIRIC and Oracle.
Oracle retains fixed supplied dynamics and is explicitly instructed not to replace them with hand-built approximations.
The model-loading gate no longer claims that loading a model enables automatic skill rehearsal.
Preflight remains controlled by its existing configuration flag; these changes do not enable it.
Prompt regression tests cover both flag values and require the same shared guidance in EMPIRIC and Oracle.
Frozen running experiments retain their original prompts, so their outcomes cannot measure the effect of this alignment.

The successful no-explicit-uncertainty ablation makes missing parameter uncertainty an inadequate explanation for Oracle's lower solve rate.
Oracle already receives the correct dynamics parameters; reconstructed geometry, state estimates, controller timing, and agent choices remain uncertain.
Prompt differences and learned compensation for reconstruction errors are plausible explanations, not isolated causal findings.
Matched prompts and saved-decision-state comparisons are needed to distinguish them.

Both current figures display the repaired-skill ramp cohort as Fan, with maze and inertial results retained only as historical records.
New benchmark menu selections of `fan` use the reviewed 3 mm ramp and 10 cm landing extension.
Historical pilot configurations explicitly inherit `fan_maze` so this default change does not silently change their layouts.
