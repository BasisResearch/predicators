# Balloons initial-scene conditioning reference

September 12, 2026.
This is a declared development prior for the first original non-hatch Balloons training recording in the [initial-state inventory](initial-state-inventory.md).
It connects public observations, a complete candidate scene, collision checks, and fresh-world replay of the frozen learned simulator.
It is not a dynamics-parameter fit, a full-trajectory posterior, or a deployed agent change.

## Information and probability model

The evidence is only the first observation returned by the corrected recording projection, including its exact public robot positions and exact box speed zero.
The model uses the historical `b09217bb3` runtime and frozen `cycle_000_vers_005_simulator.py`, with optional parameter sidecars absent in an isolated working directory.
Object handles and geometry come from fresh visible model worlds.
No evaluator poses, velocities, attachment frames, or hidden memory initialize candidates.
Later recorded actions and observations are used only for replay diagnostics after sampling.

The prior is an engineering assumption, not a recovered task-generation law or a claim that the interface guarantees each reset value.
Its original distribution is identified separately from the first-observation conditioning target.

| Quantity | Declared prior or input | Initial conditioning |
| --- | --- | --- |
| Box and balloon translations; clip translations; band xy | Independent Gaussians with standard deviation 0.15 m, centered on public base layout constants | Gaussian sensor conditioning for each noisy coordinate. |
| Box and balloon orientations | Independent uniform rotations on SO(3) | Retain all three coordinates per body; poses have no rotation sensor in this recording. |
| Clip orientation | Zero roll and pitch; uniform yaw on `[-pi, pi]` | Truncated Gaussian yaw distribution under the actual unwrapped observation channel. |
| Band z and orientation | Height is the midpoint of the supplied lo/hi descriptors; identity orientation | Derive the height; treat the descriptors as fixed context. |
| Robot positions | The previously declared Gaussian reset law, including unobserved movable joints | Condition on exact public arm/gripper positions and retain their original density. |
| Robot motion | Half mass at complete rest; half in independent uniform movable-joint velocities on `[-0.25, 0.25]` | No velocity reading; retain both cases. |
| Each dynamic body's motion | Half mass at zero linear/angular velocity; half in independent Gaussian velocity coordinates with standard deviation 0.25 | Exact box speed zero selects its rest component and retains mass 0.5; balloon motion remains unknown. |
| Static clip and band motion | Fixed zero velocity | Part of the anchored-fixture model. |
| Initial mechanisms | Intact untied balloons, closed clips at their canonical stop, no grasp, command weld, or pending command | This reset component rejects other root mechanisms as unsupported. |
| Labels and band limits | Supplied exact scene descriptors | Fixed inputs, not extra noisy likelihood terms. |

After the initial conditioning, the representation has `42 + 13*m + 6*k` continuous dimensions, where `m` is the robot moving-case indicator and `k` is the number of moving balloons.
The range is therefore 42 through 73 dimensions across 16 motion cases.
The 42 common dimensions comprise 23 translations, three clip yaws, twelve omitted body-orientation coordinates, and four unobserved robot positions.
This count applies only to the declared free-pose reference; adding supported-contact cases changes the representation and its original-prior identity.

The new `condition_gaussian_coordinate` function derives each scalar Gaussian proposal from its fixed original prior and one observation.
It returns the marginal observation density as well as the conditional mean and standard deviation.
For an exact coordinate, it returns zero remaining variance and retains the original prior density at the measured value.
Tests verify `p0(x) * p(y given x) / q(x given y)` against that returned factor, including repeated fitting and genuinely new independent observations.
The same reading must not be scored again after this analytic conditioning.

The geometry policy rejects a whole candidate on dynamic-body collisions or robot collisions, respecting the visible chute's wall-box-only collision rule.
It permits static-static overlap as part of the declared anchored-fixture support and the source-established wheel/plane fixture contact.
It does not certify every sampled static-fixture arrangement as physically mountable.
This is an explicit limitation of this prior, not an unrecorded repair of rejected bodies.
Rejection from the analytically conditioned distribution samples its globally geometry-conditioned initial-state law.
The acceptance fraction is not reported as the original scene normalizer or model evidence.

## Why the reset protocol matters

The earlier generated-component audit checked geometry directly after constructing the robot and resetting its joints.
The historical `_set_state` path additionally calls `robot.reset_state`, which places the base center of mass at its configured pose.
The URDF base inertial origin is `(-0.0036, 0, 0.0014)` relative to the base link.
Consequently, the model constructor has base COM `(0.75, 0.6464, 0.0014)`, while the complete reset has COM `(0.75, 0.65, 0)`.
The expected wheel/plane signed distance changes from -9.675 mm to -11.075 mm.
The source-derived formula after reset is `0.055325 - 0.0014 - 0.065` meters.
The new check uses that declared reset geometry and retains the earlier constructor-only result separately.
It does not change the robot reset implementation, geometry, contact tolerance, or historical agent behavior.

## Current evidence and remaining constraint

Final root reference `22631706` sampled eight accepted scenes for each of two seeds, requiring 1,242 and 1,715 complete draws respectively.
All sixteen samples reproduce every exact initial observation.
Each sample's 16-action replay is identical across two fresh model worlds for all projected public outputs.
These are conditional root samples and replay checks, not agent solve-rate seeds.

Every sample nevertheless disagrees with the first later exact observation.
Moving robot roots change the exact post-action joints; roots with resting robots first disagree on box speed.
Initial conditioning and reproducible replay therefore do not complete the trajectory-constraint problem.

A separate controlled support diagnostic fixes the box upright on the known table face instead of using the free-pose sample.
At the public nominal xy and support height, its first speed is within `2.63e-11 m/s` of the recorded value for one tested native-parameter setting, and its first public joints match exactly.
This is a substantial reduction from the free-pose discrepancies, but it is not exact equality, a fitted posterior, or permission to add a tolerance-band likelihood.
The native damping scan motivates an explicit parameter-elimination diagnostic, while the mixture of supported and free initial states still needs a complete conditional construction.
Negative-height controls that intersect the table remain inadmissible under the tested support policy.

Constraint diagnostic `22631837` then scans the visible native `air_drag` range `[0.01, 40]` at three fixed box masses.
Near damping 2.2, the first-speed residual falls to approximately `3.5e-16` through `4.6e-16 m/s` in this restricted supported component.
This remains a numerical candidate, not an exact conditional representation for all scene states or a whole-recording fit.
The local derivative estimates vary by approximately 2.5% through 5.9% over the tested finite-difference step sizes.
Two further sign-changing brackets near damping 12.82 and 14.19 do not converge to equality: they retain residuals around `3.46e-9` and `-8.89e-9 m/s`, while their finite-difference slopes grow by roughly 100 times when the step shrinks by 100 times.
They are discontinuity candidates, not valid smooth roots, and their illustrative inverse-derivative factors must not become posterior weights.
The three masses also retain later speed discrepancies of approximately `1e-4 m/s` within the sixteen-action prefix, although public joints match throughout that restricted replay.
No damping value was published or adopted by an agent.
This experiment establishes why a root solver's convergence flag alone is insufficient for the remaining nonlinear contact-conditioning work.

Artifacts: [declared root law and source snapshot](../../logs/uncertainty_balloons_root_v4_20260912/plan.json), [all root samples and replay diagnostics](../../logs/uncertainty_balloons_root_v4_20260912/reference-22631706.json), [controlled support diagnostic](../../logs/uncertainty_balloons_support_20260912/reference-22631750.json).
The [parameter-constraint diagnostic](../../logs/uncertainty_balloons_constraint_20260912/reference-22631837.json) preserves every scan point, root candidate, residual, derivative estimate, and later prediction error.

The follow-up `22632107` varies supported-box xy and yaw instead of fixing the canonical pose.
It obtains four admissible supported configurations per seed in 90 and 57 draws, then evaluates 21 damping values for each configuration.
None of those eight scans finds a sign-changing bracket for the first-speed constraint.
Their same-sign speed errors range in magnitude from approximately `3.57e-8` to `2.64e-6 m/s`.
This finite scan is not proof that no solution exists, but it does not support extending the canonical damping-elimination chart across uncertain initial geometry.
The [assessment](../../logs/uncertainty_balloons_supported_family_20260912/assessment.json) records the supported-component identity and corrects two inherited free-case metadata fields that were not used as weights in these diagnostics.

The next diagnostic, `22632307`, varies box yaw as the eliminated coordinate while retaining uncertain xy and testing three native damping settings.
It is a support investigation, not a posterior fit: root residuals, derivative variation, and unsearched brackets remain explicit.
