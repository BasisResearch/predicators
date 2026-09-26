# Lower-drop Fan seed 4 switch investigation

The completed lower-drop EMPIRIC cohort solved seeds 0 through 3 and lost seed 4.
The initial investigation changed no agent, shared controller, benchmark configuration, or paper result.
The subsequent shared-controller repair is described below; active frozen experiments and paper results remain unchanged.

## Reproduction

Diagnostics ran on compute nodes with the original frozen runtime `ed7fb86a3ad0`.
The diagnostic replays recorded low-level actions from the test reset up to the brake invocation, then executes a newly grounded SwitchOn skill.
Seed 4's decision point is step 336; seed 2's corresponding successful invocation starts at step 177.
Replayed joint positions match the saved decision states within 7.1e-9 radians for seed 4 and 2.4e-9 radians for seed 2.

| Decision state | Approach / contact height | Result |
| --- | --- | --- |
| Seed 4 | 0.05 / 0.10 | Same approach-descent stall, 58 steps, fan remains off |
| Seed 4 | 0.05 / 0.105 | Skill completes in 27 steps, fan on |
| Seed 4 | 0.05 / 0.11 | Skill completes in 27 steps, fan on |
| Seed 2 | 0.05 / 0.10 | Skill completes in 27 steps, fan on |
| Seed 2 | 0.05 / 0.105 | Skill completes in 27 steps, fan on |
| Seed 2 | 0.05 / 0.11 | Skill completes in 27 steps, fan on |

Replaying seed 4's original 58 action commands also leaves the fan off, reproducing the recorded failure's physical outcome.
The newly grounded skill reproduces the exact diagnostic: Waypoint_1 incremental IK stalls after 25 steps without progress, 0.015 m from its target, with robot-switch contact reported at -0.0155 m.
Explicitly restoring the exact replayed seed-4 state through `_set_state` before the invocation preserves both the failure and the successful higher-height counterfactuals.
Thus exact state restoration alone does not explain away this failure.

## Interpretation and limits

The immediate cause is a reproducible state-dependent failure of the switch approach, before the intended push phase.
A 5 mm increase in the requested contact height avoids it at the same decision point.
The same original parameters succeed in seed 2, so they are not universally invalid.
Follow-up reciprocal arm-state swaps isolate starting arm state as a causal factor: seed 4 with seed 2's robot features and joints completes the original request in 27 steps, while seed 2 with seed 4's arm state stalls after 58 steps.
The latter stall is not identical: it reports a 30 mm target shortfall and the fan has switched on, whereas the original seed-4 failure leaves the fan off.
Changing the contact height to 0.105 or 0.11 succeeds in both swapped cases.
The original failing descent first contacts the switch with the left gripper finger at step 23, followed by both fingers during the stall.
Actual-world contact depth reaches approximately 1.5 mm, not the 15.5 mm reported by the controller's separate contact diagnostic; these measurements must not be conflated.
This localizes the fragility to the arm-state-dependent approach rather than proving that the requested Cartesian endpoint is universally infeasible.
These experiments do not establish parity with the agent's noisy reconstructed simulator, which this exact-state diagnostic does not load.
The agent's partial rehearsal and reliance on timely braking made the failed skill catastrophic, but it is not accurate to attribute the physical failure solely to a lack of planning.
The higher-height counterfactuals establish switch success, not completion of the full task.

The next repair investigation should target collision-safe approach descent and planning/execution consistency in the shared skill, tested across starting arm states, before further domain tuning.
Do not globally hard-code the higher height from these two examples without broader checks.

## Controller trace follow-up

An instrumented replay confirms that motion planning is enabled for the failing approach descent and that the skill's own IK-validation setting is already true.
The command-line `pybullet_ik_validate=False` does not disable that setting in this Fan skill factory.
Consequently, toggling that global flag alone is not a test of enabling validation in this controller and is not a demonstrated repair.

The planner returns an eight-configuration path for the failing descent, including its starting configuration.
The controller consumes the seven subsequent commands, then remains in the same descent phase through the failure.
It attempts to settle on the final planned configuration and subsequently invokes incremental IK repeatedly to converge to the Cartesian target.
The first physical switch contact occurs before the later incremental-IK convergence loop, so it would be incorrect to blame that loop alone for initiating the collision.
The higher-height counterfactuals use six-configuration descent paths and progress to the intended push without this prolonged convergence.

The shared controller permits a non-direct planned motion to advance after its tracking-hold limit, and its final convergence path has a stall guard and joint-jump guard rather than another collision-planned trajectory.
This identifies a concrete execution robustness gap: acceptance of a collision-checked joint path does not guarantee that the physical arm tracks it safely or reaches its endpoint.
It does not yet distinguish all possible contributors to the initial contact, such as tracking error versus planning-world geometry or collision clearance.

Recommended repair scope is the shared switch approach, not an EMPIRIC-only policy change:

1. Add this saved-state failure and the successful seed-2 invocation as end-to-end regression fixtures.
2. Check physical tracking and unexpected contact during the non-contact descent, with prompt abort or a validated retreat/replan instead of prolonged pressing.
3. Validate the final convergence motion as well as the original planned path.
4. Compare planning-world and execution-world switch/finger geometry before choosing a clearance margin or a different approach trajectory.
5. Verify the repair across all five saved starting arm states and both successful and failed parameter choices before any benchmark rerun.

An early abort alone saves time but does not establish task success; recovery and complete-task outcomes require separate tests.
No production skill, agent, environment, or existing experiment was changed during that diagnostic follow-up.

## Domain-general repair, September 21

The accepted descent endpoint has approximately 5 mm finger-to-switch clearance in both the planning and execution worlds.
The switch base pose, lever configuration, and endpoint closest-point distances agree in the geometry probe.
This rules out an endpoint-geometry mismatch in this particular failure; it does not prove exact parity throughout every rollout.
The geometry probe uses save/restore for inspection and slightly perturbs subsequent contact transients, so the uninstrumented replay remains the authoritative failure reproduction.

The candidate repair changes the shared push factory, not a Fan-specific skill or agent prompt.
Its non-contact approach prefers the existing collision-checked Cartesian descent used by grasp/place controllers.
When joint limits prevent that descent, it can use a collision-checked free-space detour to the same requested pose.
The executor distinguishes the actual path type: a detour must not inherit the shorter straight-descent tracking budget.
Final approach corrections replan from the measured arm pose at most once, instead of falling back to unchecked incremental IK.
The intentional push direction, requested contact height, unplanned execution mode, and grasp/place defaults are unchanged.
No special fan name, switch index, or 5 mm height adjustment appears in the repair.

This is narrower than the initially proposed contact-triggered retreat system: the reproduced failure is avoided through path construction, and recovery uses a bounded checked replan rather than a new general-purpose retreat policy.

### Regression evidence and cost

- All 62 recorded test-level switch invocations across the five lower-drop Fan seeds complete with the final candidate.
  The legacy controller completes 61/62 in the paired replays.
  Seed 4's original failed invocation now completes in 37 steps with its original parameters, versus the legacy 58-step failure.
- The 61 previously successful Fan invocations take 0-10 additional steps, with a median increase of 10.
- All eight sampled Boil switch invocations from benchmark seed 3 complete in both versions.
  The candidate adds 10 steps per invocation.
- Both sampled Balloons releases from benchmark seed 3 complete in both versions.
  The candidate adds 6 and 7 steps, respectively.
- The initial Domino comparison completes its sampled push in both versions, taking 46 candidate steps versus 47 legacy steps.
  The final Domino replay has zero decision-state joint error against the historical recording; the Boil replays differ by at most 0.00435 radians.
  Joint agreement is a useful reproduction check, not a claim that all hidden physical state was independently compared.
- The shared-skill, motion-planning, and compact saved-arm tests pass: 126 tests, with four existing expected failures.
  The seven plotting/report tests also pass after registering the new inertial cohorts.
  Targeted mypy passes on the six changed production/report/replay modules.
  Targeted pylint passes with only obsolete configuration-option diagnostics suppressed; the repository lint configuration contains options rejected by its pinned pylint.

The first straight-descent-only candidate broke previously successful joint-limit cases.
Adding a planned detour fixed those, but an initial implementation wrongly applied the short descent tracking budget to the detour, aborting a seed-1 invocation that the legacy controller completed in 184 steps.
The final candidate preserves that 184-step success; a dedicated unit test guards the distinction.

These are local skill regressions, not new whole-task solve rates.
Extra manipulation time can change heating or ball motion during execution, so the evidence does not establish unchanged end-to-end performance or a 5/5 Fan ramp solve rate.
The repair is not included in the active Fan inertial jobs and has not been substituted into any paper result.
Matched full-agent evaluation is still necessary before adopting it as a new benchmark runtime.

Final replay artifacts are `logs/push-fan-five-seed-final.log`, `logs/push-boil-current-final.log`, `logs/push-balloons-final.log`, `logs/push-domino-current-final.log`, and `logs/push-domino-legacy-final.log`.
The paired legacy Boil records are in `logs/push-cross-domain-replays-v2.log`.
The compact physical regression is `tests/test_push_approach_regression.py`; the reusable comparison driver is `scripts/replay_push_approaches.py`.

## Artifacts

- Diagnostic driver: `scripts/diagnose_fan_switch.py`.
- Seed 4 replay: `logs/fan-switch-seed4-diagnostic.log`.
- Seed 2 comparison: `logs/fan-switch-seed2-diagnostic.log`.
- Seed 4 exact-restoration comparison: `logs/fan-switch-seed4-restored-diagnostic.log`.
- Contact-link trace: `logs/fan-switch-seed4-contacts.log`.
- Reciprocal arm swaps: `logs/fan-switch-seed4-arm2.log` and `logs/fan-switch-seed2-arm4.log`.
- Controller dispatch and planning trace: `logs/fan-switch-seed4-controller.log`.
- Planned-path progress trace: `logs/fan-switch-seed4-tracking.log`.
