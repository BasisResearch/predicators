# Boil transition discrepancy and initial fixture state

This extends the [historical-model control](historical-model-controls.md) under Stage A/B of the [simplification proposal](simplification-proposal.md).
The learned filling/heating program, parameter defaults and September 10 training recording stay fixed.
The acting agent and original parameter-free controls remain unchanged.
All corrected trajectories below condition on recorded joints throughout the episode; they are likelihood diagnostics, not unconditional forecasts.

## Joint correction alone does not resolve the failure

The model applies the same explicit joint-transition construction tested in [Domino](domino-joint-transition.md).
After each native action, each controlled physical joint has a Gaussian transition centered at its native prediction with standard deviation 0.001 in that joint's coordinate units.
Exact observed joints determine the corrections, while their normalized densities remain in the likelihood once.
Native joint velocities and the pre-correction cached Cartesian robot fields remain unchanged.
Joint AR output-error factors are absent from corrected trajectories; the other declared position/orientation discrepancies, sensor factors and checked finger readout remain.
Switch observations retain their exact semantics.

Native job `22685601_0` completes six 264-action replays in 75 allocation seconds on one CPU, using 1,584 native actions.
The uncorrected no-op and historical-default histories reproduce their archived controls exactly.
Two corrected default histories repeat exactly, including their model memory and all density factors.
Varying the fill rate changes the learned scalar dynamics but leaves the physical switch mismatch unresolved.

| Source or replay | Faucet turns on at action | Faucet turns off at action | Mismatched faucet readings |
| --- | ---: | ---: | ---: |
| Recording | 56 | 121 | 0 |
| Uncorrected historical model | 55 | 122 | 2 |
| Joint-corrected historical model at original point state | 55 | No switch-off through 264 | 145 |

The faucet object and its switch expose the same event, so the two 145-count witnesses are duplicate readouts of one physical mismatch, not 290 independent events.
The complete likelihood is zero in all six cases.
The output likelihood is finite through action 54 and first becomes zero at action 55.
Independent verification `22685666` checks 9,504 Gaussian joint factors to maximum absolute log-factor discrepancy 8.882e-16, recomputes the complete output likelihood, and rejects four deliberately corrupted inputs.
It takes 79 allocation seconds on one CPU.

The largest correction is 0.016831 radians in an angular joint and 0.008100 meters in a finger joint, both at action 56.
These are physical interventions, not harmless numerical adjustments.
Changing the Gaussian scale alone cannot alter this conditioned path: the exact observed positions determine every correction at any positive scale.
The scale changes density weights, not the missed switch-off for the fixed starting state.

## A control for cache and native-reset effects

Job `22685736_0` separates three operations while retaining the same starting state and historical program.
It completes six full histories in 34 allocation seconds on one CPU, using 1,584 native actions.

| Intervention | Native joint-reset calls | Predicted frames different from unmodified historical replay |
| --- | ---: | ---: |
| Refresh the observation cache only | 0 | 0 |
| Reset every joint to its identical native position and velocity | 2,376 | 0 |
| Repeat those identical resets in a fresh world | 2,376 | 0 |
| Correct joints to recorded positions | 936 | 256 |

Complete predictions, commands and model memory match for both identity-reset runs and the cache-only run.
The recorded-joint run reproduces the earlier failed corrected history exactly.
An independent JSON reader checks all six histories and every intervention, and rejects altered identity positions, reset counts and cache predictions.
This isolates the failure to changed joint positions among these interventions; it does not establish general engine-checkpoint portability or identify a particular contact-force error.

## Initial slider state versus initial fixture position

A bounded search varies the unobserved initial faucet-slider position and velocity while preserving the exact displayed off state.
The three positions cover fractions 0, 0.25 and 0.75 of the off interval, and velocities are -0.02, 0 and 0.02 meters per second.
An additional original-state repetition makes ten histories.
The native slider range is 0 to 0.0296 meters, with the off/on boundary at 0.0148 meters.
These grid points are candidate witnesses, not samples from a declared prior.

Job `22685891_0` completes all ten histories in 117 allocation seconds on one CPU, using 2,640 native actions.
All public initial observations remain identical, the original-state histories repeat exactly, and every candidate retains the same 145-reading faucet mismatch.
Independent reader `22686067` checks all ten histories, invariant initial readings and 23,760 Gaussian joint factors in 117 allocation seconds.
This bounded search does not prove the full initial-state prior infeasible.

A separate point-state test estimates fixed fixture x/y coordinates using only the first 65 public observations.
It changes neither height nor orientation, verifies that each adjusted body has a fixed base, and leaves movable bodies and initial joint settings unchanged.
The two alternatives average only the faucet switch, or all four fixed faucet/burner fixtures.
The resulting switch x position is 0.027114 meters from the noisy first-frame point estimate; its y shift is 0.001085 meters.
That displacement is substantial relative to switch contact geometry.

Job `22686011_0` completes the original point twice and both alternatives in 54 allocation seconds on one CPU, using 1,056 native actions.
Both prefix-mean alternatives eliminate the remaining exact switch witnesses and have finite complete likelihoods under the declared transition/output model.
The original point still has zero likelihood and repeats the earlier failed history exactly.
Independent reader `22686210` verifies the four histories, the first-65-frame mean calculations, unchanged non-target initial features, full observation likelihoods and 9,504 Gaussian joint factors in 62 allocation seconds.
It rejects the same four corrupted-input classes as the transition verifier.
Both supported histories reproduce every exact switch observation; Gaussian-factor roundoff is at most 3.553e-15 in this check.
The arithmetic mean is a diagnostic point estimate, not a posterior mean under the temporally correlated output-error model.
Its likelihood values are conditional on the chosen fixture state, not model evidence integrated over initial-state uncertainty.

## Implication for the implementation

The joint-transition model alone did not repair the point-state Boil replay.
A better fixture-state estimate supplies supported histories without changing the learned dynamics or relaxing exact switch observations.
This provides a useful starting point for explicit joint parameter/state inference and supports the plan's requirement to account for uncertain initial states.
It does not justify treating fixture positions as known, accepting a posterior, or enabling the replacement in planning.

Next define a normalized fixture-state prior and a density-corrected proposal using the available observation prefix, retain the initial-state likelihood and original parameter prior, and test independent inference replicas and causal future predictions.
The full source and artifacts are in `logs/uncertainty_boil_joint_transition_20260913/`, `logs/uncertainty_boil_joint_reset_control_20260913/`, `logs/uncertainty_boil_slider_initial_control_20260913/`, and `logs/uncertainty_boil_fixture_pose_control_20260913/`.

Each bundle's `verified-inputs.json` records the tested source, plans, reports and verification artifacts.
