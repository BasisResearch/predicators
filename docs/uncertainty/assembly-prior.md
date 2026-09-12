# Rigid-assembly initial-state component

September 12, 2026.
This is an offline component of Stage A in the [uncertainty simplification plan](simplification-proposal.md), implemented in [inference_assembly.py](../../predicators/code_sim_learning/inference_assembly.py).
It provides normalized candidate distributions for a fixed rigid geometry, including a planar-contact case.
It is not the historical balloons task prior, a full scene prior, or a deployed agent estimator.

## Why bodies cannot be initialized independently

A weld constrains relative pose and motion.
Giving attached bodies independent positions can create an inconsistent constraint immediately.
Copying the parent's linear velocity to an offset child is also incorrect when the assembly rotates.
For root position `x`, orientation `R`, linear velocity `v`, angular velocity `omega`, and child offset `r` in the root frame, initialize:

```
child_position = x + R r
child_linear_velocity = v + omega cross (R r)
child_angular_velocity = omega
```

Child orientation composes the root rotation with the declared local rotation.
The original parent weld frame is that same local pose, and the child weld frame is identity.
These quantities describe one correlated candidate; they are not independent noisy readings or separately fitted body states.
During simulation, finite-force engine constraints can deflect, so the exact initial construction does not assert exact rigidity at every later step.

## Explicit geometry and support assumptions

Each `AssemblyBody` declares a name, a fixed pose relative to the root, and a radius enclosing its collision geometry.
The first body is the root and has identity local pose.
The caller must establish these geometric facts from its declared model and permitted inputs.
The API cannot verify arbitrary mesh geometry or infer a free region from noisy observations.
Privileged weld metadata and evaluator reset states do not supply these declarations.

The caller also declares an axis-aligned obstacle-free cell.
Pairwise enclosing spheres must not overlap, which conservatively ensures that distinct bodies do not interpenetrate.
This rejects some physically valid tightly packed assemblies; rejection is an explicit limitation of this component's support.
The assembly radius about the root is the largest local-center distance plus that body's radius.
Eroding the cell by that radius supplies normalized uniform bounds for the root center, without candidate-dependent clipping or rejection normalization.
A cell too small to contain the declared assembly is rejected before sampling.
This guarantee covers the declared cell and collision envelopes, not undeclared obstacles, robot links, or articulated bodies.

## Three distinct component priors

| Component | Free coordinates | Normalized distribution | Derived quantities |
| --- | ---: | --- | --- |
| Free assembly at instantaneous rest | 6 | Uniform root xyz in the eroded cell; uniform orientation on SO(3) | All body poses; zero linear/angular motion; original weld frames |
| Free moving assembly | 12 | The same pose distribution plus independent uniform root linear and angular velocity components with declared positive half-widths | Correlated child velocities through the shared rigid twist |
| Assembly on a horizontal support face | 3 | Uniform root xy in the eroded footprint; uniform yaw | Root height, zero roll/pitch, zero twist, body poses and weld frames |

Uniform SO(3) orientation uses a uniform-quaternion construction from three unit-interval coordinates.
It does not use independent uniform Euler angles.
`coordinates` returns a normalized `BoxPrior`, and `lift` maps those coordinates into physical body states.
The generative probability measure is defined on these coordinates and pushed forward through the map.
There is no extra product of densities over derived child poses or velocities.
Geometry and prior support are part of the component digest.

For the supported component, `support_depth` declares an actual lowest horizontal root face at local `z = -depth`.
The root height is exactly the cell floor plus this depth, and only yaw is sampled.
The root envelope must fit below the cell ceiling, and attached-body enclosing spheres must remain above the support plane.
The caller must verify the root face against its collision geometry; a bounding radius alone does not establish that face.
This is geometric contact at instantaneous rest, not a proof of static balance or a guarantee that the assembly will remain upright.

These components do not assign probabilities to one another.
A complete scene prior must declare their case masses and any dependence on parameters, layout, attachments, and robot state.
In particular, the supported component is not obtained by projecting a sample from the free component onto the table.
It is a separately declared distribution on a lower-dimensional contact case.
The rest components likewise declare atoms in velocity; they do not infer zero motion from missing observations.

## Validation and scope

The tests compare generated states against actual PyBullet collision geometry, compose weld frames independently through Bullet, and compare the velocity map with a finite difference of rigid motion.
They also check rotational isotropy, explicit dimensions, identity changes, rejected geometry/support, and exact face contact up to engine distance roundoff.
That geometric comparison tolerance is not a likelihood or added sensor variance.

The separate physical reference generates a box and an attached sphere entirely from this declared model, then simulates gravity and plane contact in fresh worlds.
Each trial compares an uninterrupted 240-step trajectory with a fresh repeat and a full-prefix reconstruction returning steps 120 through 240.
It records initial penetrations, contacts, weld deflection, and every nonzero replay difference.
The references are generated mechanical cases, not historical-domain fits or agent solve-rate seeds.
See [the experiment record](experiments-20260912.md#rigid-assembly-prior-and-planar-contact-reference).

Before applying this to recorded tasks, define and justify root attachment cases, uncertain relative geometry, valid scene regions, and robot-state priors.
An exact tied flag does not identify a weld frame, and a noisy pose cannot silently become a geometric prior bound.
Subsequent exact robot/contact observations still require a valid conditional representation.
This component supplies one physical building block without bypassing those requirements.
