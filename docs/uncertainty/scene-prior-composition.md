# Reset-compatible priors and whole-candidate feasibility

September 12, 2026.
This extends the offline [joint-state](robot-state-prior.md) and [assembly](assembly-prior.md) components.
It resolves two incorrect hard-support assumptions in the generated physical reference, while keeping their failed versions as controls.
It does not change the acting agent or establish a fitted historical-domain posterior.

## A declared reset law instead of an ideal joint-limit bound

The robot wrapper writes joint positions through `resetJointState`.
The vanilla IK routine repeatedly resets its candidate and checks end-effector position for convergence; that validation is not a certificate that every returned joint respects the URDF interval.
A mechanical reproduction passes the recorded initial joint vector through the actual wrapper and reads it back exactly, including the balloons shoulder angle below the URDF lower limit.
Changing that archived reading or the agent's initialization is unnecessary for this inference experiment.

`JointStatePrior` now accepts either finite uniform position bounds or `GaussianJointPosition` distributions through its `position_priors` field.
Gaussian positions describe a simulator initialization law over the real-valued joint coordinate, including winding.
They do not wrap angles or assert an ideal hardware joint limit.
Exact measured positions contribute their original Gaussian log density when conditioned out, and the other coordinates keep their original distributions.
Unobserved Gaussian positions use uniform quantile coordinates; zero-measure endpoints have no finite lift and are rejected explicitly, while unrepresentable numerical results raise arithmetic errors.
The implementation uses the standard-library normal inverse CDF.

The new reference declares zero-mean Gaussian position priors with standard deviation pi radians for revolute coordinates and 0.1 m for prismatic coordinates.
These are engineering assumptions, fixed in the experiment plan rather than centered or fitted on each observed joint value.
The Gaussian family was introduced after identifying the bounded model's failure on development data, so positive support on these same recordings is not independent validation of its calibration.
The original bounded component still rejects the balloons start and remains a distinct negative control.
The new joint-prior schema identifies the changed family and representation; it does not reuse the old prior's identity.

## Collision conditioning must apply to the whole draw

Suppose a normalized base prior draws a complete candidate `x`, including any component case, robot state, and assembly state.
For a declared feasibility predicate `C`, rejecting the entire draw unless `C(x)` holds produces:

```
p(x | C) = p0(x) * I[C(x)] / Z
Z = Pr_p0(C)
```

[draw_feasible](../../predicators/code_sim_learning/inference_feasibility.py) implements this procedure with an explicit draw budget and separate identities for the base prior and support policy.
A failed search returns `budget_exhausted`, with no completed sample batch; it is not proof of empty support.
Predicate and setup errors propagate instead of becoming collision rejections.
These outputs are prior draws, not posterior samples.

Rejecting only an offending body while keeping the rest of the sample generally changes the distribution.
Likewise, normalizing feasibility separately within each mixture case preserves different case probabilities from global rejection.
For example, equally likely cases with feasibility probabilities 0.2 and 0.8 have accepted probabilities 0.2 and 0.8 under global rejection.
The numerical tests check this result and the dependence induced by a triangular joint constraint.

The acceptance fraction is not an exact normalizing constant or a model-evidence estimate.
A common `Z` cancels in ratios for one fixed posterior target, but a parameter-dependent or case-dependent normalizer cannot silently be discarded.
This sampler neither asserts that `Z` is parameter-independent nor implements evidence comparison across differently normalized models.
A full generative scene model still has to specify how observation conditioning, component probabilities, geometry, and dynamics parameters interact.

## Connecting feasible scenes to parameter inference

`FeasibleConditioning` now carries that distinction into the existing offline batch sampler.
It wraps an exact-conditioning map, evaluates feasibility on the complete lifted candidate, and preserves the map's observation-density and proposal corrections.
It adds no resampling or alternative inference algorithm.
The caller must explicitly choose which original distribution is intended:

| Declared law | Original distribution | Additional conditional-base factor |
| --- | --- | --- |
| `global_joint` | `p0(theta, s) I[C(theta, s)] / Z` | The feasibility indicator; one global constant cancels within this posterior. |
| `conditional_state` | `p0(theta) p0(s given theta) I[C(theta, s)] / Z(theta)` | The indicator and `1 / Z(theta)`, retaining the declared parameter marginal before observations. |

For `conditional_state`, `Z(theta)` is the support probability under the original state law before observing the data.
It is not the acceptance rate after conditioning on the current recording.
The adapter requires a separately identified deterministic log-normalizer callback; it cannot derive that normalizer from finite rejection samples.
Its implementation and dependence on the retained variables remain part of the caller's reviewed probability model.
Missing normalization, invalid probability values, and scene-construction exceptions remain explicit errors rather than zero likelihoods.
Zero support found in a finite candidate batch remains a search outcome, not a proof of inconsistency.

The supported original-prior identity includes the feasibility policy and the normalization choice, separately from the exact observations.
Changing observations therefore changes the inference target without redefining the original prior.
Weights flow through the same `PriorPoint` and `ConditionedPrior` interfaces, so the existing sampler retains them during initialization and every Metropolis move.
Noisy observations are applied afterward, once, through the remaining likelihood.

An analytic reference makes the distinction measurable.
Let `theta` be uniform on `[1, 4]`, let `x` be independently uniform on `[0, 4]`, require `x <= theta`, and observe `theta*x = 0.5` exactly.
The coordinate map eliminates `x`, retaining the factor `1/theta`.
Global joint conditioning then gives mean `theta = 3/log(4)`, approximately 2.164.
Normalizing the state prior separately uses `Z(theta) = theta/4`, giving density proportional to `1/theta^2` and mean `theta = log(4)/0.75`, approximately 1.848.
Omitting that factor changes the statistical question despite using the same feasible candidates.
This is a numerical reference, not a historical-domain result or evidence that either law is the right task prior.

For the proposal's fixed parameter-prior endpoint, use `conditional_state` when defining feasibility inside `p0(s given theta)`.
A support probability independent of all sampled parameters can cancel, but that independence needs justification.
If geometry or attachment parameters affect the normalizer and it is unavailable, retain an explicit unsupported construction until a normalized generative representation or validated normalization method is supplied.
Selecting `global_joint` just to avoid that calculation would generally change the declared parameter prior.
Full historical scene laws, attachment cases, and exact trajectory conditioning remain outstanding.

## Fixed fixture contacts are part of the geometry contract

Rejecting every robot/background intersection gives no accepted candidates in the strict reference.
The contact diagnostic identifies the same two pairs in every visible domain: each Fetch wheel against the floor plane.
The checked URDF places the spherical wheel collision centers at height 0.055325 m and gives them radius 0.065 m.
The fixed-base placement therefore produces signed distance -0.009675 m, independent of wheel rotation.
This is a property of the supplied fixture geometry; sampling different joint angles cannot remove it.
This particular placement is the constructor-only geometry used by that generated-component reference.
The later [historical-root reference](balloons-initial-scene.md#why-the-reset-protocol-matters) calls the full robot reset, which places the base COM at its configured pose and produces signed distance -0.011075 m instead.
The two fixture policies are tied to their explicit initialization protocols; the earlier result does not certify the later placement.

The revised reference permits only those two named wheel/plane fixture contacts at that expected signed distance.
The support identity records the exception, expected distance, source hashes, and geometric roundoff policy.
A different distance raises an error rather than expanding the allowance.
Every other tested robot/background or sampled-assembly intersection is still rejected.
The strict no-overlap result remains recorded separately.
No robot geometry, collision mask, observation, sensor noise, or production behavior changes to make the revised predicate pass.

## What the composed reference establishes

The generated reference combines exact recorded initial robot positions with uncertain unobserved joints and a synthetic rigid box/sphere assembly.
Its base distribution chooses equally between a joint/assembly rest case and a joint/assembly moving case, then rejects the whole candidate under the declared geometry policy.
That shared case is a stated correlation in this generated reference, not an inferred rest guarantee or a proposed final task-wide motion model.
The policy checks the new bodies against existing collision geometry and the robot against existing nonrobot geometry, with only the named fixed fixture contacts permitted.
It conservatively includes queried geometry even where a visible environment may disable a collision pair.
Robot self-collision is not added to the existing fixed-base model by this audit.

This tests a composition procedure and support under a particular declared model.
It does not reconstruct the historical object layouts, condition on their other exact readings, solve later exact robot/contact trajectories, or fit simulator parameters.
The finite accepted batch does not certify broad exploration or posterior quality.
Those checks remain necessary before replacing legacy inference.
See [the experiment record](experiments-20260912.md#reset-law-and-composed-scene-support).
