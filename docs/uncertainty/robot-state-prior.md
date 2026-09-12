# Robot initial-state prior and conditioning

September 12, 2026.
This offline component implements the robot-motion portion of the [initial-state inventory](initial-state-inventory.md).
It does not replace the acting agent's state estimator, establish collision-free full scenes, or solve exact trajectory constraints.

## Explicit initial-state assumptions

[JointStatePrior](../../predicators/code_sim_learning/inference_joints.py) requires an entry for every joint in URDF order.
Mechanically fixed joints have position and velocity zero and no free coordinates.
Every movable joint has an explicit position prior: finite uniform bounds or the later Gaussian reset-law extension.
A positive velocity half-width specifies a normalized uniform initial velocity distribution; zero specifies a prior atom at rest.
The component declares independence among these coordinates.
A full scene prior must justify dependencies, contact compatibility, and probabilities of its motion cases separately.

The interface deliberately does not infer position bounds or motion from recording omissions.
A URDF continuous joint needs a declared prior over its winding as well as its physical orientation; the API never wraps an exact observed angle into a preferred interval.
URDF limited-joint intervals are possible modeling assumptions, not automatically valid hard support for the simulator's recorded initialization.
The balloons counterexample below demonstrates why that distinction matters.

`condition_positions` takes exact initial joint measurements and eliminates those coordinates.
The remaining independent coordinates keep their normalized original distributions.
Each conditioned movable position contributes its original uniform or Gaussian density; a fixed joint's exact zero contributes unit mass.
`log_observation_factor` retains these factors, including when every coordinate is determined and no sampler is needed.
The factor can matter when comparing components with different position priors and must not be silently dropped.
No later observation is substituted into a rollout by this operation.

An out-of-support initial measurement raises `IncompatibleJointObservation`.
This means that the declared component assigns zero support to that measurement.
It does not mean a finite sampler missed feasible particles, that every possible prior is inconsistent, or that the agent failed its task.
Other malformed inputs raise ordinary validation errors.

## What the Fetch inventory permits

The five audited visible simulators share the same Fetch URDF: nine observed movable joints, four unobserved movable joints, and eleven fixed joints.
The unobserved joints are the two wheels and head pan/tilt.
After conditioning on the nine measured positions, the explicit components have:

| Component | Unknown positions | Unknown velocities | Total free coordinates |
| --- | ---: | ---: | ---: |
| Declared instantaneous rest | 4 | 0 | 4 |
| Declared moving robot | 4 | 13 | 17 |

These are dimensions of the component conditional when the readings lie in its support.
They are not a completed full-scene posterior dimension, a claim that the robot really starts at rest, or a reason to keep coordinates that a stronger program-and-scene invariant could legitimately eliminate.

The geometry audit changes the head configuration while keeping the observed arm/gripper positions fixed.
Public robot features remain identical, but a 1 cm-radius probe intersects the head in one configuration and is separated from it in the other.
The first witness is about 5.62 mm inside one collision envelope and 210.05 mm outside the other.
This demonstrates that identical forward kinematics does not imply identical collision behavior.
The intersecting probe is a geometric witness, not a proposed valid penetrating initial scene or evidence that a historical task hit the head.
Eliminating these joints would require proof that their geometry and all program reads are irrelevant throughout the declared scene support and action history.

## A real initial state outside an assumed hard bound

The frozen original balloons training recording starts its shoulder-lift joint at `-1.5119263197144368` rad.
The matching URDF limit interval is `[-1.221, 1.518]` rad.
Both rest and moving components using that interval therefore reject the initial observation, before candidate sampling.
The other four audited recordings admit their exact initial positions under the same declared rules.

The robot wrapper restores positions through `resetJointState`; the prior cannot treat an idealized joint-limit interval as a proven invariant of all simulator reset states.
This result does not justify clipping the observation, changing the archived task, silently widening the prior, or relabeling initial joint positions as external inputs to make this target pass.
A broader initialization law or an explicitly justified conditional-input formulation would be a different declared model and needs its own validation.
The rejected component remains a negative control.
The later [Gaussian reset-law and scene-composition reference](scene-prior-composition.md) covers this initial position under a separately declared prior, while retaining its original reading density.
That support result does not validate the Gaussian law's calibration or replace the full-scene and trajectory checks.

## Validation boundary

The numerical tests check exact conditioning, retained observation density, unobserved position/motion marginals, deterministic conditionals, and rejection without clipping or wrapping.
The visible-model audit uses the previously verified public initial joint readings and checks the URDF byte identity.
It restores generated joint states through the engine and checks the actual joint values and public robot features.
No hidden body motion, task generator, or live evaluator state supplies prior values.
The geometry witness uses synthetic probe positions solely to test whether hidden head configuration can matter for collision geometry.

These checks validate the component's declared semantics.
They do not certify that independent joint and assembly priors form a collision-free scene or that a learned simulator can satisfy all later exact proprioception.
See [the experiment record](experiments-20260912.md#robot-prior-conditioning-and-hidden-joint-geometry).
