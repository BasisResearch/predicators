# Fan joint-inference integration

September 12, 2026.
This extends the [full Fan scene law](fan-initial-scene.md) into the offline batch sampler required by the [simplification proposal](simplification-proposal.md).
The production agent remains on the incumbent fitter.
These are fixed-program development experiments on the original 132-action training recording, not agent runs or held-out predictions.

## A fixed coordinate map for every initial case

The initializer now accepts a 120-dimensional unit vector instead of consuming a variable-length random stream.
Its physical prior, observation-informed proposal, initial observation factors and whole-scene geometric support policy remain the declared Fan model.
Each coordinate has a fixed role across rest/moving cases:

| Slots, inclusive | Role |
| --- | --- |
| 0 | Airflow speed under the original Uniform[0, 1] prior |
| 1 | Robot rest/motion case |
| 2-18 | Free robot positions and potentially moving velocities |
| 19-58 | Ten fixtures, four coordinates each: xyz and yaw/categorical orientation |
| 59 | Ball supported-rest/free-motion case |
| 60-71 | Ball pose and potentially moving twist |
| 72-79 | Four switch conditional position-mixture and velocity pairs |
| 80-119 | Twenty rotor positions and velocities |

Coordinates unused in a particular case remain normalized auxiliary uniforms.
The box dimension is therefore not the number of physical degrees of freedom in each scene.
The existing 83-113 active continuous dimensions, including airflow speed, and 65,536 discrete cases remain represented.
Exact controlled initial positions are still conditioned with their density retained.
No later observation is assigned directly into a simulator state.

The inverse helper only constructs a proposal center from the previously found supported-rest-ball witness.
It does not restrict the forward map to that case or claim an inverse for every free-ball quaternion chart.

Compute job `22643080_0` completed the native map audit.
The saved physical witness and its mapped reconstruction each have full-recording log likelihood 34241.55615373634.
Both repeat exactly in fresh worlds, including public observations, robot joints and nonrobot articulated state.
Four fresh unit draws also repeat exactly; three fail geometry and the fourth fails the full observation target.
Those outcomes are retained as failures to reach support, not evidence that the full target is impossible.
The [report](../../logs/uncertainty_fan_coordinate_map_20260912/pilot-22643080_0.json) records the complete coordinates, physical candidates and actual node1377 runtime.

## Informed proposals without changing the prior

Let `u` denote the original unit chart and `r0(u)` its existing original-prior/proposal correction, including exact initial joint and switch evidence.
The new proposal draws `u` from a mixture with weight 0.1 on the entire original unit cube and weights 0.45 each on nearby and wider components around the support witness.
The wider component has ten times the nearby standard deviations.
These proposal choices use the complete training recording; they are not new prior information or held-out evidence.

Each local component uses normalized truncated Gaussian laws on 57 active chart coordinates.
Matching discrete-case selectors use uniform laws over their corresponding chart intervals.
Unused auxiliaries, supported-ball yaw and all rotor states retain uniform proposal draws.
The broad component retains support for every original continuous region and discrete case, including other motion cases and fixture orientations.
It does not guarantee that a finite run actually visits those alternatives.

The mixture density `q(u)` is evaluated with log-sum-exp across all three components, including components other than the one that generated the draw.
The corrected factor is `log r0(u) - log q(u)`.
Using only the selected component's density would define different weights and is not done here.
The original parameter and scene prior remains fixed.
The whole-scene support normalizer is still common and independent of airflow speed; no model-evidence estimate is claimed.

Compute job `22643174_0` completed a stratified proposal-overlap audit with eight draws per component.
The component sample counts are diagnostic allocations, not mixture samples to pool as an unweighted posterior.

| Component | Geometry-feasible draws | Finite complete targets |
| --- | --- | --- |
| Broad unit cube | 2/8 | 0/8 |
| Nearby | 8/8 | 5/8 |
| Wider | 6/8 | 3/8 |

All three repeated first-draw trajectories match exactly, including the failing broad case.
Checks cover truncated-law CDF/quantile round trips, positive interval lengths and the broad component's density lower bound.
The largest CDF round-trip error is below 2.2e-13.
The [report](../../logs/uncertainty_fan_joint_proposal_20260912/pilot-22643174_0.json) retains every candidate and its full mixture-density correction.
This establishes usable proposal overlap for an integration experiment; it does not establish posterior adequacy.

## Submitted full-recording inference

Array `22643258` runs independent seeds 200 and 201 on `mit_preemptable`, pinned to node1412 with the same CPU and Python hash seed.
Each uses 64 particles, 32 cubic temperatures, eight Metropolis moves per temperature and a maximum of 16,448 target evaluations.
The eight-hour wall limit is an external compute cap, not a declaration of convergence.
Fresh-world 132-action replay occurs for each candidate, including candidates that change initial geometry or motion.

The sampler's proposal space has one additional mixture-selector coordinate.
Its returned joint rows contain airflow speed and the original 119 scene-chart coordinates, so downstream parameter projection uses `theta.fan_speed` explicitly.
Symmetric scalar-coordinate moves act in the 121-dimensional proposal space and include the complete density correction in their acceptance ratio.

At temperature zero the target includes the original-prior/proposal correction, initial-output likelihood, geometric support and all complete-recording exact constraints.
The tempered term is the complete-output log likelihood minus the initial-output log likelihood.
At temperature one their sum is the declared complete target, with every observation entering once.
No exact event is softened by tempering or by a tolerance window.

The [frozen manifest](../../logs/uncertainty_fan_joint_pilot_20260912/plan.json) identifies the historical runtime, offline overlays, program, source scripts, proposal, full training data and hashed prerequisite report.
Each worker first verifies a finite, exactly repeated mapped witness on its own runtime.
Sampler completion leaves numerical availability unevaluated.
Independent agreement, weight and ancestor concentration, parameter movement, budget/proposal sensitivity and predictive investigations remain necessary before returning a usable physical-domain posterior to an acting agent.
