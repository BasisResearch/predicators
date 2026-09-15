# Bridge glue support after joint conditioning

September 13, 2026.
These are offline development diagnostics for the [uncertainty simplification plan](simplification-proposal.md).
They retain the revised subclass, original scene law and 600-action fitting prefix.
They do not change the production agent or establish posterior adequacy.

## Radius and height alone do not fix the saved path

The [joint-conditioning audit](bridge-initial-scene.md#conditional-robot-joint-trajectory-audit) removed joint and finger-readout contradictions but retained sixteen glue-reading mismatches on its all-rest root.
The next diagnostic partitions the original `drip_r` and `drip_h` boxes at every eligibility change point on the saved pre-rule geometry.
Within each resulting cell, the nearest eligible face is constant at every step.
It tests an interior representative of every positive-area cell and checks the selected/default choices through the literal deposition function.
The screen compares deposition labels where the observed previous glue level is not already latched; full causal updates and bond consumption require separate native validation.

Compute job `22701184` completed the five-root screen, and independent reader `22701286` verified 2,419 threshold cells, the saved geometry arrays and three corruption controls.
Independent rotation calculations differ from the saved geometry by at most 3.3307e-16 m.
The all-rest root improves from two deposition-label errors to one, but no tested cell has zero errors.
The four moving-root cases retain thirteen errors because their paths do not supply the required held-bottle deposition geometry.
The first screen attempt, `22701022`, completed its calculations but failed to serialize integer array indices; the corrected attempt preserves its computation and selection rule.
That failure is a diagnostic reporting error, not an agent or model failure.

The all-rest path supplies a concrete incompatibility:

| Recorded event | Necessary condition on the saved geometry |
| --- | --- |
| Deposition at action 87 on `span0.glue_end_a` | `drip_h >= 0.02439433018` m. |
| Deposition at action 200 on `span2.glue_end_a` | `drip_r >= 0.01606379755` m. |
| No deposition at action 57 on `span0.glue_end_a` | The candidate instead has horizontal distance 0.00483451437 m and height 0.02003017178 m, so the two necessary bounds force deposition. |

No competing face can become eligible at action 57 anywhere in the declared radius/height boxes.
This is a contradiction for the frozen geometry and deposition-label screen, not proof that every initial scene or every causal parameter trajectory is unsupported.

## Nearby initial scenes

The next native audit perturbs eight XY proposal coordinates for the three spans and the bottle around the all-rest root.
It uses predetermined unit-coordinate changes of +/-0.10 and +/-0.25 where those remain inside the original coordinate domain.
The root map, original prior, proposal-density corrections, default model parameters and conditional joint law remain unchanged.
These thirty candidates are deterministic support probes, not posterior samples or estimates of prior mass.

Job `22701392` completed in 3:13 with 29 feasible scenes, one retained geometry rejection and 34,800 native actions.
Every feasible 600-action path repeated exactly, including the original baseline failure.
Four candidates reduced the complete glue-reading mismatches from sixteen to thirteen without holding errors; other changes produced much larger errors.
Independent reader `22701745` checked all 156,600 conditional joint factors and 420 original-prior/proposal density terms.
Maximum discrepancies were 6.8479e-13 for joint factors and 8.8818e-16 for density ratios.
It also rejected a deliberately altered original-prior/proposal factor.

The radius/height screen was then applied to all 29 verified feasible paths.
The one rejected geometry remains identified in the upstream report and is not silently promoted to a successful path.
Job `22701914` completed in 44 seconds, and independent reader `22701927` checked all 73,082 threshold cells in 26 seconds.
Every path still has at least one deposition-label error at its best threshold cell.
Thus these local scene changes and threshold adjustments do not supply exact glue support.
They do not justify claiming that the full original scene law has empty support.

## Explicit stochastic reach trial

A separate discrepancy model adds an unobserved independent reach perturbation before each learned glue update:

```
eta_t ~ Normal(0, 0.005**2)       # metres
effective_drip_h_t = drip_h + eta_t
```

The other eligibility conditions, face ranking, glue accumulation, drainage and bond rules remain the same.
The recorded glue readings remain exact; they are not assigned sensor noise or a likelihood floor.
This is a changed stochastic simulator model, with its own identified assumptions, rather than a claim that the previous deterministic model already explained the recording.
The reach scale is a declared development assumption and still requires predictive assessment.

For a fixed pre-rule scene, the eligible face changes only at finitely many reach thresholds.
The conditional driver enumerates all intervals between those thresholds and evaluates the unchanged model update at one interior point per interval.
It sums the normalized Gaussian masses of intervals that produce exactly the observed glue outputs.
It eliminates the unobserved perturbation only after checking that every compatible interval produces identical complete next model memory and commands.
The retained interval mass is part of the conditional likelihood; choosing a compatible representative does not make that mass one.
No compatible interval gives zero likelihood and terminates that candidate as unsupported.
A future generator for this model must draw fresh reach perturbations without consulting future glue readings.

The initial native trial uses the same all-rest root, the screened radius/height pair and bond dwell 25, derived from the recorded first-bond step and the candidate dwell counter.
The original dwell 30 is retained as a negative control.
Job `22702160` completed both fresh repetitions of each case in 1:01, totaling 2,366 native actions.
The dwell-25 case matches all glue readings over the complete 600-action prefix and retains an event log-probability of approximately -6.38423.
The dwell-30 case matches the first 582 actions and correctly has no compatible transition at the recorded bond event, action 583.
The positive case still has six exact robot pose/orientation channels to represent in the complete output model.
Independent verifier `22702255` completed in ten seconds.
It checked all 1,471 reach intervals using separate Gaussian integration, reproduced every literal rule and memory update, and verified the retained joint factors.
The maximum interval log-mass discrepancy was 3.4107e-13 and the maximum joint-factor discrepancy was 6.8301e-13.
Compatible intervals also agree on their command sequences, so marginalizing the perturbation does not discard a hidden branch of future model behavior in these cases.
Changed probabilities, interval boundaries, compatibility labels and model memory are rejected.
The two earlier setup attempts failed at the public-feature projection boundary; the corrected driver checks the callback's public features against the candidate world and records public joint metadata separately.
Neither setup failure is an agent result.

## Complete output composition at the supported point

Job `22702471` composes the supported path with the existing continuous-output components: correlated scalar position errors, a coupled robot-orientation error and the checked exact finger readout.
The other recorded noisy fields retain their original sensor likelihood, and exact fields still reject contradictions.
The composition includes the original root-density correction, conditional joint factors, normalized reach-event masses and the exact-rate conditioning density `1 / 0.95`, retained once at the first partial glue increment.
Initial and remaining output factors sum to the complete output likelihood, and repeated scoring is exact.
Independent reader `22702562` completed in 34 seconds with zero output-accounting discrepancy.
It checks scalar innovations and sensor factors separately, verifies all 1,202 observed/predicted finger readouts and uses an independent quaternion conversion with the previously checked orientation-density kernel.

The resulting finite composed log weight is approximately -11,013,135, so support alone is not a useful inference starting point.
Three noisy orientation channels dominate the poor fit: `leg1.pitch`, `span0.roll` and `span1.roll` contribute approximately -7.414 million, -1.883 million and -1.826 million respectively.
Their initial-observation penalties are already large; the sampled resting-face choices must be investigated before launching a posterior from this point.
The original six-face prior remains unchanged, and a diagnostic face selection must not be misreported as a posterior sample or a normalized importance proposal.
The dwell-30 control retains zero complete likelihood at its exact glue contradiction.

These results do not establish a complete normalized Bridge target, a reliable posterior, better predictions or unchanged agent performance.
The remaining requirements include a complete parameter/scene inference map with full prior and data identities, adequate exploration, and causal future prediction tests.

## Initial orientation guidance with the original prior retained

Native scan `22702774` evaluates all six resting faces of each body using only its initial public noisy orientation and height readings.
The preferred choices correct `leg1` from face 3 to 2, `span0` from face 4 to 0, `span1` from face 5 to 0 and the bottle from face 1 to 0.
The other two blocks retain their original faces.
Independent checks reproduce the Gaussian factors, represented block rotations and exact native height readout for all 36 cases, and reject four corrupted reports.
These maximum-score selections are diagnostic points, not posterior draws.

The shared `CategoricalProposal` component converts that guidance into a normalized sampling proposal while retaining all six original prior cases.
For original masses `p_i` and guide likelihoods `L_i`, it samples from `q_i = 0.25 p_i + 0.75 p_i L_i / sum_j(p_j L_j)` and returns `log(p_i / q_i)`.
The caller must retain the likelihood used to construct the guide; it does not become the physical prior.
An impossible guide case still has positive defensive prior mass, and a numerically unrepresentable sampling interval is rejected explicitly.
Sixteen functional tests, focused type/lint checks and the pinned formatters pass in compute job `22702995`.
The tests recover prior moments and an independently specified posterior by exact finite enumeration, including a strongly concentrated guide and impossible guide cases.

Bridge adapter check `22703098` recovers all seven diagnostic roots exactly through the unchanged native initializer.
It checks all 36 per-body density corrections and sums all 46,656 joint face combinations back to unit original-prior mass.
Moving-body coordinates remain unchanged and acquire no spurious resting-face factor.
This validates a proposal boundary for the planned inference map; it does not supply a posterior population or satisfy the prediction gate.

## Complete conditional trajectories with corrected faces

Pipeline `22703020` completed in 7:49, including 7,306 native actions, independent reach/joint verification and four complete output-accounting checks.
Every feasible diagnostic is run twice from a fresh native world, and the baseline reproduces the earlier supported artifact exactly.

| Face intervention | Exact glue prefix supported | First unsupported action | Complete output log density |
| --- | ---: | ---: | ---: |
| Original sampled faces | 600 | None | -11,048,561.45 |
| Correct `leg1` only | 600 | None | -3,619,008.88 |
| Correct `span0` only | 86 | 87 | Zero complete likelihood |
| Correct `span1` only | 600 | None | -9,218,435.81 |
| Correct bottle only | 582 | 583 | Zero complete likelihood |
| All initial-observation-selected faces | 600 | None | 74,837.37 |
| All selected faces, dwell 30 | 582 | 583 | Zero complete likelihood |

The combined face correction removes the dominant orientation discrepancy while preserving all 600 exact glue updates, with retained reach-event log probability -5.32478.
The complete composition, before the additional face-proposal correction, has log weight 109,642.31.
Its independent output-accounting discrepancy is at most 2.9104e-11.
The largest remaining negative output terms are the two span roll channels, so the improved score does not establish good prediction or calibrated uncertainty.
These are log densities at deterministic development points, not marginal evidence, posterior comparisons, solve rates or successful agent seeds.

The face interventions cannot be assessed independently: changing the bottle or `span0` alone loses an exact event, whereas their combined correction with the other faces preserves the full prefix.
At the fixed radius, intersecting every compatible height interval on the corrected conditional path remains empty at action 87.
Thus this corrected scene does not yet justify removing the declared reach discrepancy and using a single deterministic height.

The first complete reader attempt exposed cancellation for a roughly 5.55e-17 m interval between nearly equal reach thresholds.
Subtracting rounded Gaussian log-CDF values produced a log-mass error of about 0.094 for an incompatible interval.
The corrected evaluator subtracts the original boundaries before scaling and uses the narrow-interval density expansion when the standardized width times the midpoint magnitude is below 1e-5.
The independent quadrature reader likewise preserves the original interval width.
All seven native cases were rerun after that correction, with maximum independent interval log-mass discrepancy 5.3718e-11.
The earlier path/probability artifacts remain retained separately under the failed attempt.
The earlier scan path-resolution failure and an incorrect float32 assumption in its height-readout checker also remain recorded as setup/verification failures, not model or agent outcomes.

Next, integrate the corrected face proposal into the complete parameter/scene target and validate an unconditional future generator for the explicit joint/reach laws.
Future reach perturbations must be sampled independently of future glue observations, with one posterior variance draw retained per joint throughout each future.
Only then should full posterior populations and their reserved-action forecasts be assessed against the incumbent.

Artifacts are in `logs/uncertainty_bridge_joint_glue_screen_20260913/`, `logs/uncertainty_bridge_local_scene_20260913/`, `logs/uncertainty_bridge_local_glue_screen_20260913/` and `logs/uncertainty_bridge_reach_discrepancy_20260913/`.
Orientation diagnostics and the face proposal are in `logs/uncertainty_bridge_orientation_support_20260913/` and `logs/uncertainty_bridge_face_proposal_20260913/`.
