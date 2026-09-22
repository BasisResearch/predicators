# Oracle Fan + Ramp seed 1 investigation

## Conclusion

The failure combines reconstruction-sensitive controller timing with an open-loop action choice that assumes the optimistic timing is safe.
It is not the previously repaired switch-descent failure: all 13 test-level skill invocations completed successfully.
The supplied Oracle predicts the dangerous outcome when given the actual scene and arm state.
Reconstructing the agent's noisy scene and its recorded geometry modifications reproduces its optimistic 20-step prediction instead.
No production code, agent prompt, domain setting, active experiment, or benchmark result was changed in this investigation.

## Historical evidence

The run is `logs/agent_continual_oracle_dynamics/fan_ramp-oracle_dynamics_opus_ramp_skill_repair_r1/seed1/run_20260921_121401`, frozen runtime `ff11bc76f465`.
Training succeeded in 550 real steps.
Test ended by the agent's explicit give-up after 938 steps, not a usage limit, step cap, controller exception, or terminal evaluator rejection.
The scorecard records `end_reason=agent_ended`, zero failed test skills, and no game-over event.
The ball remained wedged above the evaluator's fall threshold, which explains why the episode was still technically in progress when the agent gave up.

The decisive sequence appears in `agent/002_play_20260921_124512.md`, turns 34 through 37.
The agent simulated a calibration from the test initial observation, with hand-adjusted platform geometry, using `sim.run(CAL, render=False, seed=0, fresh=True)`.
It predicted that enabling the opposing fan would take 20 steps, and that the ball would stop near x=0.474 m on the starting platform.
In execution it first pinned the ball against the left wall and switched that fan off.
It then enabled the forward fan, enabled the opposing fan, and waited 150 steps in one execution request.
The second switch took 45 steps, not 20.
The opposing fan cancels the forward wind but does not remove existing momentum; gravity then accelerates the ball down the ramp.
The ball ended at x=1.52625 m, z=0.41002 m, beyond the landing edge and wedged next to the fan bank.
Subsequent attempts did not recover it.

The agent already recorded timing discrepancies in its training journal, but still described this test calibration as zero-risk.
Its retrospective claim that the model does not represent controller timing is too strong: the controlled replays below demonstrate that it can predict the 45-step timing.
Its claim that switches flip only at the very end is also imprecise: the decisive real skill flips at action 40 of 45, versus action 15 of 20 in the reconstructed rehearsal.

## Controlled compute-node replays

All diagnostics used the original frozen runtime, supplied Oracle source, recorded launch configuration, and saved low-level actions.
No LLM benchmark runs were launched.
The prefix to test step 284 reproduces recorded robot joints exactly in these diagnostics.
The complete 938-action replay reproduces the overshoot and subsequent stuck position.

| Rehearsal substrate | Planner seed | Second SwitchOn duration | Outcome after 150 hold steps |
| --- | --- | --- | --- |
| Native world after recorded prefix | 1 | 45, flips at 40 | Wedged, x=1.52625 |
| Native world after explicit state restoration | 1 | 45, flips at 40 | Wedged, x=1.52625 |
| Fresh Oracle from the exact decision state | 1 | 45, flips at 40 | Wedged, x=1.52625 |
| Full calibration, exact initial geometry | 1 | 45, flips at 40 | Wedged, x=1.52625 |
| Full calibration, historical noisy observation plus recorded FL3 modifications | 0 | 20, flips at 15 | Safe starting-platform stop, x=0.47351 |
| Same reconstructed calibration | 1 | 20, flips at 15 | Same safe stop |
| Same reconstructed calibration, only switch poses replaced with true poses | 1 | 45, flips at 40 | Wedged, x=1.52624 |
| Exact initial geometry, only switch poses made noisy | 1 | 36, flips at 31 | Starting-platform stop, x=0.61349 |

The noisy observation is regenerated with the recorded noise configuration and `step_rng(1, 1, 0, 0)`; its full-precision initial features match the logged observation.
The FL3 modifications are copied from turn 25 of the historical trace.
The full-calibration diagnostic executes the same grounded skills and hold actions in the supplied Oracle, rather than recreating the complete historical sandbox session.
It reproduces all four relevant historical rehearsal skill durations, 23/39/39/20, and the reported x=0.4735 result.

Replacing only the switch geometry is sufficient to restore the dangerous timing and outcome in that reconstructed scene.
The reciprocal intervention gives an intermediate 36-step timing, so switch geometry is a causal contributor, not an exclusive explanation independent of other scene geometry and preceding arm motions.
These experiments do not isolate a single coordinate or a particular internal motion-planner waypoint as the entire cause.
They show that millimetre-scale pose errors can change the shared controller's path and actuation latency substantially even when the skill ultimately succeeds.

Planner randomness is a second sensitivity, not a cure: from the exact decision state, planner seed 0 fails the approach after 21 actions, while seed 1 succeeds in 45.
The subsequent holds in that diagnostic are failure-consequence probes, not an assertion that a production executor would continue the plan after a skill exception.
Simply repeating the optimistic reconstructed rollout under seeds 0 and 1 leaves its 20-step prediction unchanged.

Exact-state restoration changes the ball's transient position by about 8 mm at the end of the skill, although the timing and eventual wedge agree.
Thus these results establish the relevant failure reproduction, not bitwise parity of every physical transient or universal completeness of hidden-state restoration.

## Why successful runs differ

Oracle seed 0 also encountered timing and reconstruction discrepancies, but used a forward pulse with the forward fan switched off before engaging the opposing fan, followed by feedback-based braking.
Its journal records adjustment of the brake hold from the observed ball position after the braking skill.
EMPIRIC seed 1 explicitly recorded broad actuation-latency variation and used observations between braking decisions rather than assuming a fixed safe nudge.
These are supported strategies in the successful runs' journals, not independently replayed guarantees of success on every seed.
The failed run's 225 test simulation rollouts show that the issue was not simply too little simulation; many rollouts shared an incorrect scene and a fragile timing assumption.

## Missed recovery opportunity

At test step 329, after the unexpectedly slow switch but before the 150-step wait, the ball is still on the starting platform at x=0.55107 m.
A counterfactual replay replaces that wait with `SwitchOff(fan_0)[0.08,0.10]`, leaving the opposing fan on for active braking.
That skill succeeds in 40 steps and flips the forward fan off at its action 33.
Across the following 750 hold steps the ball stays on the deck, reaches at most x=1.08963 m, and eventually rests near x=1.07055 m.
This prevents the demonstrated east-edge overshoot.
It does not demonstrate a complete level win or establish that resting near the ramp seam is a robust long-term recovery state.
The diagnostic output label `SETTLED` denotes only the first 150-step hold endpoint; the later `EXTENDED_BRAKE` records are needed to assess subsequent motion.

## Recommended repair

1. Preserve this exact/reconstructed pair as an end-to-end regression fixture for simulation and skill timing.
   Record time to the actual switch transition, total skill duration, and the ball trajectory, not only whether the skill completed.
2. Include contact-object pose uncertainty in selective rehearsal, including switches, and refresh exact robot proprioception from the current observation before committing a maneuver.
   Use coherent scene estimates and varied planner seeds; do not supply privileged true object poses to the agent.
3. Turn observed rehearsal-versus-execution timing errors into a conservative actuation-latency range.
   Evaluate whether the plan remains recoverable under that range before an irreversible maneuver.
   Repeating a single reconstructed scene or checking only a terminal goal cannot provide that assurance.
4. End an execution batch and reassess when an observed skill duration or outcome exceeds its predicted range, especially before a long wait.
   A generic deviation response must invoke a validated continuation, not assume that stopping the robot stops the moving object.
5. Investigate controller path sensitivity across nearby contact poses and starting arm states before another shared-skill change.
   Keep the prior repair and avoid hard-coded Fan-specific heights or an assumed fixed 20-step duration.
   Any shared controller change needs cross-domain replay checks and consistent application to comparison arms.

A blanket requirement that each skill merely succeed in one preflight rollout would not address this failure: the historical rehearsal already passed every skill.
No environment simplification or automatic replacement of the failed seed is recommended from this evidence.

## Diagnostic provenance

- Driver: `logs/diagnose_oracle_fan_ramp_seed1.py`.
- Authoritative recorded/exact/restored replays: job array `23408841`, logs `logs/oracle-fan-replay-v2-23408841-*.log`.
- Historical reconstructed scene and planner-seed controls: array `23408983`, logs `logs/oracle-fan-attribution-23408983-*.log`.
- Switch-geometry interventions and immediate-braking counterfactual: array `23409045`, logs `logs/oracle-fan-controls-23409045-*.log`.
- Extended braking counterfactual: job `23409067`, log `logs/oracle-fan-brake-23409067.log`.
- Initial diagnostic array `23408723` had a diagnostic argument-handling error and ran the native decision-point case four times; it is excluded from the comparative evidence above.

## Additional seeds requested September 21

At the user's request, Oracle seeds 5 and 6 were submitted as array `23412153` on compute nodes, using account d.
They retain the existing `ramp_skill_repair_r1` cohort name, frozen runtime, agent settings, geometry, budgets, and preflight settings.
The proposed reliability changes above are not enabled.
The resolved launch flags and arguments match those of the existing Oracle cohort; only the seed range changes.
The launch configuration is `scripts/configs/predicatorv3/continual_fan_ramp_oracle_extra_r1.yaml`.
All seven Oracle seeds are retained in the development report, including the failed seed 1; no paper figure is changed.
