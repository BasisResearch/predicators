# Checked observation reductions

September 12, 2026.
This records reductions justified by the observation interface for the [uncertainty simplification proposal](simplification-proposal.md).
The original observation ledger remains intact, and the acting agent's observations remain unchanged.

## Exact deterministic readouts

Suppose an exactly observed source `q` has an additional deterministic readout `r=g(q)`.
Its likelihood factors as the density or mass of `q`, followed by a conditional point mass at `g(q)`.
If the observed pair is compatible, the readout adds conditional mass one; it is not another independent measurement of `q`.
If the pair is incompatible, its likelihood is zero.
Changing the display scale does not create additional evidence or an extra Jacobian, because the observation measure uses `q` as the source coordinate.

`ExactReadout` identifies the source field, output field, sensor schema and reviewed map.
Its mapping identity must include the implementation, constants, precision and observation timing, and the map must not depend on parameters being inferred.
The generic interface does not discover or prove those dependencies.

`reduce_exact_readout` checks the observed pair before producing a reduced observation view.
The source remains in that view and must still contribute its likelihood under the candidate model, including any declared latent output error.
An inconsistent readout returns no reduced view and a negative-infinite factor, so it cannot silently disappear from the calculation.
A missing or noisy source requires a different conditional construction; the function refuses to discard its informative readout.
The original data identity and the readout-map identity both belong in the complete inference identity.

## Robot finger reading

The five frozen development programs inherit the standard robot finger readout.
`SingleArmPyBulletRobot.get_state()` first casts the left finger joint into its float32 state vector.
`PyBulletEnv._get_robot_state_dict()` then applies `_fingers_joint_to_state` using the fixed robot and feature endpoints.
The correct map is therefore the runtime's endpoint interpolation applied to `np.float32(observed_left_finger_joint)`.
Skipping that cast changes the exact readout.
Clipping or introducing an observation tolerance is not part of this map.

The public joint source is index 7 of the observed arm-joint vector in these five Fetch configurations.
Its index and endpoint constants come from each instantiated visible model, not a guessed global convention or evaluator-private state.
The frozen programs do not override the readout or infer its constants.
Other robot configurations or future program changes require a new declaration and audit.

| Domain | Recorded frames | Exact matches with the runtime map | Mismatches if float32 conversion is omitted |
| --- | ---: | ---: | ---: |
| Bridge | 1,187 | 1,187 | 1,186 |
| Fan | 133 | 133 | 123 |
| Domino | 162 | 162 | 161 |
| Boil | 265 | 265 | 262 |
| Balloons | 236 | 236 | 235 |

Audit `22634263` verifies all 1,983 frames using the source-derived map and retains the 1,967 failed direct-double readouts as a negative control.
Integration audit `22634439` then applies the new reduction to the same complete public recordings.
All 1,983 reductions preserve their source joint and all other measurements.
All 1,983 deliberately perturbed finger readings are rejected, with no reduced observation returned.
These are observation checks, not physical rollouts or agent seeds.
Artifacts are in `logs/uncertainty_exact_readout_audit_20260912` and `logs/uncertainty_exact_readout_audit_v2_20260912`.
The [integrated assessment](../../logs/uncertainty_exact_readout_audit_v2_20260912/assessment.json) checks that the data and sensor identities match the preceding five-domain output-error diagnostic and lists its remaining exact constraints after this reduction.

Final compute job `22634508` passed fifteen functional tests, two-file mypy and lint, and pinned formatting checks.
The tests verify source-likelihood preservation, display-scale invariance, precision-sensitive contradictions, missing/noisy-source handling and invalid mapping callbacks.
The first check attempt found a test-lambda type annotation issue; a subsequent attempt was canceled after an overlong assertion was found, and both were corrected before the final checks.
The library implementation used by the recorded integration audit matches the final checked implementation.

This reduction can remove the separate finger constraint from an inference likelihood after verifying it against the exact joint source.
It does not excuse a discrepancy in the modeled joint reading, supply its posterior density, or make the complete recording compatible.
In particular, the earlier recorded output-error comparison must still model the joint source and retain its remaining orientation, event and speed constraints.

## Cartesian robot pose is different

The [cached-link audit](transition-discrepancy.md#preserve-the-native-robot-observation-phase) showed that current joint positions do not directly reconstruct the historical Cartesian robot observation phase.
Calling fresh forward kinematics can change the reported pose without changing the joint readings.
That pose therefore cannot be removed using the finger-readout argument.
The exact robot orientation and Cartesian outputs need a model that respects their actual timing and dependencies.

## Native Euler readout support

Compute audit `22634634` completed successfully on `mit_preemptable` and examined the same 1,983 public frames.
It found that the proposed canonical Euler support was too restrictive; its `invalid_angles` field records violations of that proposed support, not invalid environment observations.
All 31 witnesses have zero roll, pitch exactly positive pi/2, and yaw outside [-pi, pi].
They must remain in the observation ledger.

| Domain | Ordinary pitch | Pitch exactly positive pi/2 | Pole readings with yaw outside [-pi, pi] |
| --- | ---: | ---: | ---: |
| Bridge | 401 | 786 | 2 |
| Fan | 60 | 73 | 13 |
| Domino | 66 | 96 | 3 |
| Boil | 118 | 147 | 13 |
| Balloons | 28 | 208 | 0 |

The audit also exercised the installed quaternion-to-Euler conversion directly, using roll 0.4 and yaw 0.7 at both pitch poles.
For both float64 and float32 quaternion inputs, tested pitch offsets through 0.004 radians returned pitch exactly at the pole and roll zero; offsets 0.005, 0.01 and 0.05 did not.
These probes bracket a change in this runtime's readout behavior; they do not identify an exact threshold or establish a probability law for orientations.
The complete probes and recorded witnesses are preserved in [the audit artifact](../../logs/uncertainty_robot_angle_audit_20260912/reference-22634634.json).

An orientation likelihood must account for this coupled readout, including its collapsed pole regime and native yaw branch.
Three independent continuous angle densities would assign the wrong measure to the pole readings.
Wrapping yaw and discarding its original branch would also change the exact observation target unless that reduction is separately justified.
No such likelihood or reduction is deployed here.
The next construction must retain these observations and validate its induced mass and density factors before using orientation evidence in a full-recording posterior.
