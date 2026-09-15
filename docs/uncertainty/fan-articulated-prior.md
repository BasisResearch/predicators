# Fan articulated initial-state prior

September 12, 2026.
This offline component advances the Fan row of the [initial-state inventory](initial-state-inventory.md).
It does not establish a complete scene prior or replace production inference.

## Switch state from a Boolean reading

The visible Fan simulator has four prismatic switches.
Their positions are distances, not angles.
An exact `is_on` reading constrains which side of a threshold the slider occupies; it does not reveal its position or velocity.

The audited runtime reports nominal URDF travel from 0 to 296 mm through `getJointInfo`.
The [travel-cap helper](../../predicators/pybullet_helpers/objects.py) enforces an upper stop at 29.6 mm without changing those reported limits.
The public controller's off/on poses are 0 and 29.6 mm, and its on threshold is 14.8 mm.
The proposed component uses the enforced 0 to 29.6 mm interval as declared initial support.
This is an initialization assumption; arbitrary `resetJointState` calls can bypass mechanical limits, so the nominal metadata or cap alone is not proof that every archived initialization lies in this interval.

[RestingJointPrior](../../predicators/code_sim_learning/inference_joints.py) specifies a normalized distribution with total mass rho divided equally between two declared controller rest poses, each with zero velocity.
With probability 1-rho, position is uniform on the declared travel interval and velocity is independently uniform on a declared symmetric interval.
The generic component also permits controller rest poses inside the travel interval.
Neither a Boolean flag nor missing velocity metadata determines rho or the velocity width.

For a reading y indicating q > c, conditioning retains the compatible rest atoms and restricts the continuous interval to the compatible side of c.
The observation probability is the sum of compatible atom mass and the moving mass multiplied by the compatible fraction of travel.
`log_observation_factor` retains this probability in the joint inference target, including when c depends on an unknown parameter.
The conditional mixture weights are divided by that same observation probability.
No observed position is fabricated, and no later trajectory output is overwritten.

Each resting case has zero physical continuous dimensions; each moving case has two, position and velocity.
Two unit sampling coordinates represent the mixture, with unused auxiliary coordinates in resting cases.
For the four switches with endpoint rest poses and interior cuts, this gives 16 rest/moving combinations and 0 to 8 physical continuous coordinates.
These counts exclude scene placement, robot state, ball motion and rotor state.

## Native validation

The [corrected audit report](../../logs/uncertainty_fan_joint_prior_v2_20260912/pilot-22639208_1.json) records 2,048 sampled switch states: four switches, both flags and 256 draws per case.
All native Boolean readings and restored position/velocity pairs match the conditional samples exactly.
This development audit declares rho = 0.8 and moving velocities uniform on [-0.1, 0.1] m/s; those values are engineering choices, not fitted or calibrated conclusions.

An isolated control drives the same switch asset and scale toward 236.8 mm for 1,200 native simulation steps.
Without the cap, the slider reaches 236.8 mm; with the cap, it stops at 29.6 mm.
Both worlds continue to report the same nominal URDF range.
The 0.01 mm settling tolerance tests the mechanical control; it is not an observation-noise or likelihood tolerance.

The earlier audit `22639121_1` completed its 2,048 readback checks but failed its isolated control assertion because that control used the object helper's default scale of 0.2 instead of Fan's scale of 1.0.
Its report is retained as a diagnostic setup failure.
The corrected audit uses the environment scale explicitly and checks the uncapped target without a scale-dependent offset.
Neither audit is an agent performance result.

Compute job `22638962` passed 15 functional tests, focused mypy and pylint, and pinned formatting checks for the new component and its integration references.
The tests compare conditional moments and a parameter-dependent threshold posterior against analytic answers, including pure-rest, pure-motion, incompatible and interior-rest cases.
The checked source hashes match the native audit overlays.

## Rotor component and remaining scene state

The [native inventory](../../logs/uncertainty_fan_articulation_20260912/pilot-22638602_1.json) distinguishes four public fan-bank objects from twenty physical fan bodies, five per bank.
Each physical fan has a continuous rotor joint.
The visible placement routine resets fan bases but does not reset those rotor joints.
The rotor link in the supplied URDF has ten collision elements, so it cannot be eliminated merely by calling it decorative.

The [replay follow-up](articulated-replay.md) found that the offline snapshot omitted all nonrobot joints, losing the supplied states of all twenty rotors and four switches even in repeatable fresh-world replays.
The corrected snapshot includes these articulated states and verifies their native layout before restoring them.

The [rotor component audit](../../logs/uncertainty_fan_rotor_prior_20260912/pilot-22639509_1.json) declares independent uniform initial rotor positions on [-pi, pi] radians and velocities on [-2pi, 2pi] radians per second using the existing `JointStatePrior` implementation.
This is a finite-winding engineering initialization law, not a mechanical limit, an angle-wrapping equivalence or a calibrated distribution.
The twenty movable joints contribute forty continuous coordinates; fixed joints contribute none.
Together with the four conditional switches, the articulated portion has 40 to 48 continuous coordinates across sixteen rest/moving switch combinations.

Eight sampled articulated candidates preserve every supplied joint state and the original public initial switch flags after restoration.
Each repeats sixteen recorded actions in two fresh worlds with exact equality of the native joint states and projected public observations at all boundaries.
The sampled cases have 40, 42 or 44 continuous articulated coordinates; the experiment does not claim to enumerate all sixteen combinations.
The four symmetric switch readings contribute a retained log probability of -2.7725887222397807 in each trial, rather than being silently treated as exogenous inputs.

These tests hold the remaining scene and robot root at controlled public-frame inputs; they do not turn those noisy inputs into a valid full-scene prior.
The [whole-scene follow-up](fan-initial-scene.md) now declares fixture placement, box orientation cases, robot/ball components and an explicit geometric support policy, with 82 to 112 continuous initial-state coordinates.
It retains these articulated components and validates repeated whole-scene candidate trajectories.
Sensitivity assessment, calibrated scene assumptions and a usable complete-recording posterior remain unresolved.
