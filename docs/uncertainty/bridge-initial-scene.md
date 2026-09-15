# Revised Bridge initial-state model

September 13, 2026.
This extends the [revised subclass comparison](bridge-subclass-inference.md) with an explicit physical root law.
It is an offline development model; the incumbent remains the production estimator.
The 600-action fitting split in [the incumbent control](bridge-incumbent-control.md) remains the intended comparison boundary.
Short root audits do not replace that comparison.

## Geometry established by the native model

The visible scene has five movable blocks, a movable bottle and two static site markers.
All eight bodies have box collision shapes and no articulated joints.
The bottle's collision shape is a box, so its omitted roll and pitch cannot be discarded using rotational symmetry.
The sites also have collision shapes and remain in whole-scene collision checks.

| Body | Full collision extents, metres | Mass, kilograms |
| --- | --- | --- |
| Each block | 0.10, 0.05, 0.05 | 0.10 |
| Bottle | 0.024, 0.024, 0.060 | 0.05 |
| Each site | 0.090, 0.090, 0.0002 | 0 |

Native inventory job `22697529` completed 96 support probes in two discarded worlds with exact fresh-instance agreement.
The tilted block's native support depth differs from the ideal sharp-box depth by approximately 0.482 mm in the tested orientation.
Follow-up `22697773` checked 240 probes, adding twelve deterministic random orientations to the eight original cases for each movable body.
For these bodies, the reported collision margin is smaller than every half-extent, and the native support depth agrees within 1e-9 m with `sum(abs(R[2]) * (half_extents - margin)) + margin`.
Each inferred contact height is checked through a second native signed-distance query.
The margin audit repeats exactly across two fresh worlds.
These probes inspect candidate geometry only and execute no recorded actions.

The thin sites do not meet the formula's half-extent condition: their half-thickness is 0.0001 m while their reported margin is 0.001 m.
The first root audit, `22698166`, placed them at the ideal sharp-box contact height and found native table penetration in the tested upright cases.
All sixteen complete candidates failed geometry support, with zero recorded native actions executed.
The failed driver and probability-model assumptions are retained under `ideal-site-support-22698166/`.
The corrected law gives fixed sites uncertain mounting xyz positions; upright versus arbitrary orientation remains a separate mixture.
It does not change the native shapes or suppress site/table collision witnesses.

## Declared reset law and information boundary

This construction covers the recorded unheld, initially dry reset case of the revised program.
Every initial `is_held` and glue reading is checked against that scope.
The subclass declares empty glue/bond/dwell memory and an empty command queue at reset; the initializer verifies that declaration and the absence of native constraints in its new candidate world.
These are assumptions of the candidate program's reset model, not facts inferred from omitted evaluator metadata.
Continued episodes or an initially bonded model require a different initial-memory and assembly construction.
Later bonds, dwell counts, command queues and constraint frames must arise from the candidate's uninterrupted action history.

The initializer first performs a fixed numeric template reset.
It preserves only immutable public sizes/colors and the conditioned fixed-base pose in that template.
Recorded noisy object coordinates and measured joint values do not determine intermediate template geometry.
It then applies the entire sampled root and exact initial controlled-joint conditioning before stepping.
No private velocities, curing counters or recorded weld metadata enter the root law.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| --- | --- | --- | --- | --- | --- |
| Each fixed site, native shape and public xyz | Fixed mass-zero collision box, noisy xyz | Mounting position and omitted orientation | Uniform workspace xyz; probability 0.8 for upright orientation with uniform yaw, otherwise Haar-uniform rotation; whole-scene collision conditioning | Static base motion is absent under this fixed-fixture model; xyz readings remain likelihood factors | 4 or 6 per site; upright/free-rotation cases |
| Each block and bottle, native shape and public features | Fixed box dimensions; noisy xyz and available angles | True pose, omitted bottle tilt, motion and table support | Probability 0.8 for table-supported rest, divided equally among six local faces with uniform xy/yaw; otherwise uniform xyz, Haar rotation and uniform linear/angular velocity | Supported z follows the verified shape map; all noisy pose readings remain in the likelihood | 3 or 12 per body; six rest-face cases plus free moving |
| Robot, native joint schema and public joint positions | Nine exact controlled positions, four other movable joints and eleven fixed joints | Unobserved positions and all movable-joint velocities | Gaussian movable-joint positions; probability 0.8 for zero velocities, otherwise independent bounded velocities | Exact controlled positions conditioned with their original density retained; fixed joints and fixed base are declared inputs | 4 or 17; rest/moving cases |
| Block sizes/colors, public descriptors | Exact immutable inputs to this program | No sampled descriptor uncertainty under this comparison | Fixed known geometry and labels | Condition on descriptors and retain native shape checks | 0 |
| Holding, public flags | Every initial body is observed unheld | Other reset cases are outside this construction | Explicit unheld reset scope | Check initial flags, with no initial grasp constraint | 0 in this scoped case |
| Glue, dwell, bonds and pending commands, subclass reset declaration | Empty declared model memory; observed initial glue values are all zero | Continued-episode memory is unsupported by this root law | Revised program's declared empty memory and constraint-free reset case; future memory from uninterrupted history | Check the dry initial scope and the reset declaration; do not import private curing or weld metadata | 0 in this scoped model |

The workspace bounds are x in [0.2, 1.3], y in [0.8, 1.9] and z in [0.4, 0.95] metres.
Moving bodies have componentwise linear-velocity bounds of +/-0.1 m/s and angular-velocity bounds of +/-0.2 rad/s.
Robot velocity bounds are +/-0.1 in the native joint units.
Gaussian robot position priors use standard deviation pi for revolute joints and 0.1 m for prismatic joints, as in the earlier reset-compatible reference.
Resting bodies derive their z position from the verified native box support formula; a sampled yaw does not imply a fixed world orientation for standing blocks.
The physical root has 30 through 101 continuous coordinates across its declared cases, before adding model parameters or trajectory-discrepancy variables.
The implementation uses 140 augmented unit coordinates to include mixture selectors, proposal selectors and inactive-case coordinates; 140 is not the physical dimension.

## Proposals and normalization

Noisy initial xyz and available yaw readings guide defensive Gaussian/uniform proposals inside the fixed original coordinate bounds.
Every guided point retains its complete original-prior/proposal density ratio.
The likelihood must still score all those observations once; using them for proposal guidance does not condition them away.
Orientation cases are not selected by plugging noisy Euler angles into an exact physical state.
Unguided continuous rotations use a Haar-uniform quaternion map.

The geometry predicate checks the robot, every movable body and both sites against candidate collision geometry.
It retains the fixed wheel/ground contacts only at the previously audited -0.011075 m distance, within a 1e-8 m numerical geometry allowance.
Other penetrating pairs reject the entire candidate.
This allowance is a geometric roundoff policy, not an observation-likelihood tolerance.

For this constraint-free reset law, initial geometry and its feasibility probability are independent of all six fitted glue parameters.
Those parameters enter later model updates rather than initial collision dimensions, poses or attachments.
Thus the original parameter marginal is preserved when normalizing scene feasibility conditionally on parameters, and the same unknown scene-normalization constant cancels within this fixed target.
The audit also compares complete initialized roots under default and lower-bound parameter vectors.
That finite check supplements the construction's independence argument; it does not justify omitting normalization for a future parameter-dependent attachment prior or for cross-model evidence.

## Validation and remaining work

Corrected compute job `22698304` tests sixteen stratified roots covering the six rest faces, mixed/free body motion, site orientation cases and both robot motion cases.
These deliberately stratified probes are not claimed as independent prior samples or as a Monte Carlo estimate of feasible prior mass.
Every feasible root receives two independent fresh-world 64-action continuations, comparing all public predictions and learned memory exactly.
Every root also receives a separate initialization under the alternate parameter vector.
The job completed on `mit_preemptable` with one CPU and 12 GB in 44 allocation seconds.
One root was feasible and repeated all 64 actions exactly, for 128 native actions total; the other fifteen retained their geometry rejections.
Every root initialized identically under the two parameter vectors.
Independent reader `22698424` completed 182 density-ratio checks over 384 guided coordinates, 54 supported-body charts, 42 free-body cases, 384 stored joint states and 47 negative contact witnesses.
The maximum independently recomputed density-ratio difference is 1.776e-15.
The earlier corrected-mounting attempt `22698233` reached its first feasible 64-action continuation but failed report serialization because a shallow copy of initial model memory shared later mutations.
Its replacement deep-copies that diagnostic memory; it changes neither dynamics nor the probability law, and the failed attempt is not a model or agent failure.

Follow-up `22698483` changes only moving-body proposal guidance: it centers the proposed z coordinate at least 0.02 m above the orientation-dependent native support height.
This is a proposal center, not a new prior bound or collision allowance; the prior component retains full original support and every candidate retains its full conditional proposal-density correction.
The motivation is the retained moving-root table penetrations, rather than an assumption that the physical initial state is exactly at its noisy height.
The original root law, initial-observation likelihood requirement and whole-scene feasibility predicate remain unchanged.
The follow-up native audit completed in 53 allocation seconds with five feasible roots and 640 native actions.
Four supported roots include three moving bodies each, and two also have moving robot joints; the original all-rest positive root is retained.
Every complete predicted frame and learned-memory state repeats exactly, and all sixteen roots retain exact parameter-independent initialization.
The remaining eleven complete-scene rejections remain in the report.
Dependent density reader `22698502` passed, preserving the same independent density and support-chart checks.
This improves proposal support on the stratified audit; it is not a feasible-prior-mass estimate or a posterior-efficiency result.

The same frozen law and proposal now feed long-prefix job `22698641`, followed by a dependent independent density reader.
It extends the five supported roots to two complete 600-action continuations each and records exact-output mismatch attribution against the fitting prefix.
Future observations after action 600 are excluded from that attribution.
The extension completed in 2:32, with all five feasible roots repeating all 600 actions exactly, for 6,000 native actions.
Independent reader `22698656` completed in six seconds and verified the saved densities, support charts and original weights.
All five roots nevertheless contradict exact recorded outputs, with their first discrepancies at actions 34, 32, 1, 34 and 1 respectively.
The all-rest root has sixteen glue mismatches across two channels; the moving-root cases have much larger glue mismatches.
These are selected candidate failures, not proof that the full conditional support is empty.
The next target must represent the remaining continuous exact trajectory constraints and retain the geometry-dependent discrete glue likelihood.
Neither a completed replay nor a mismatch count is a continuous exact-conditioning method.

## Conditional robot-joint trajectory audit

The next isolated audit preserves all five feasible initial roots and compares their native continuations with a declared conditional joint-transition model on the same 600-action prefix.
After each native base step and before the learned model-memory update, it conditions the nine controlled joint positions on their exact observations, retaining native joint velocities and existing physical constraints.
It never restores a later recorded scene or imports private model memory.
One independent inverse-gamma variance per controlled joint is shared across the entire prefix, using the [tested variance component](shared-variance-discrepancy.md) with `alpha=2` and `beta=1e-6` in native squared units.
Every conditional correction retains its normalized Student-t factor; these factors alone are not the complete likelihood of the remaining outputs.

Compute job `22700282` completed ten trajectories and their fresh-world repetitions in 3:48, totaling 12,000 native actions.
All five native controls reproduce the previously saved failures exactly, including every prediction and learned-memory value.
Both modes repeat exactly for every root.
The conditional paths remove all controlled-joint mismatches and the exact finger-readout mismatches, while retaining other contradictions.
The all-rest root drops from eighteen mismatching channels to eight, with the same sixteen glue-reading mismatches.
Its remaining channels are the six robot pose/orientation fields and two glue fields.
The four moving roots each retain twelve mismatching channels, including holding and glue discrepancies.
Thus exact joint conditioning supplies one valid trajectory component but does not resolve the Bridge support problem.
Independent reader `22700337` completed in fifteen allocation seconds, checking all 27,000 causal Student-t factors against separate gamma-function calculations, the full shared-variance integral, posterior sufficient statistics and alignment of exact joints with the observed and predicted histories.
The maximum per-factor discrepancy is 6.8301e-13.
It also reconstructed every remaining mismatch count and rejected corrupted joint values, factors, variance statistics and missing transitions.
The remaining work is an explicit representation for the other continuous exact robot outputs and supported geometry-dependent glue/holding histories, while preserving this same fitting split and root prior.
Artifacts are in `logs/uncertainty_bridge_joint_conditioning_20260913/`.

Full 600-action exact-output support, independent density verification, the complete conditional inference target and posterior adequacy remain required.
In particular, the exact-rate component assumes an eligible deposition schedule and cannot simply replace the full geometry-dependent glue likelihood.
This root construction alone supplies neither a usable posterior nor evidence of improved agent performance.
Frozen inputs, scripts and reports are in `logs/uncertainty_bridge_initial_scene_20260913/` and `logs/uncertainty_bridge_scene_inventory_20260913/`.
