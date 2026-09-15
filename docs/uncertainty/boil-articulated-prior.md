# Boil articulated initial-state component

September 13, 2026.
This extends the [fixture-position work](boil-fixture-proposal.md) toward a complete episode initialization model.
The fixed historical Boil program and production agent remain unchanged.

## Three nonrobot movable joints

The public asset definitions contain one slider in each of the two switches and one revolute joint in the faucet itself.
The faucet joint is separate from the switch that controls the recorded faucet `is_on` value.
Its position and velocity are not observed by that Boolean reading.
The remaining joints in those assets are fixed links.
The native audit verifies this inventory against the instantiated runtime instead of relying only on the asset XML.

Each joint has an explicitly declared mixture: probability 0.8 split equally between rest at its two interval endpoints, and probability 0.2 on independent uniform position and velocity.
These engineering choices do not follow from absent velocity observations.
Switch position support uses the native travel limit multiplied by the visible `switch_joint_scale`; velocity support is `[-0.02, 0.02]` meters per second.
Faucet position support uses the native revolute interval, nominally `[0, pi/2]`; velocity support is `[-0.2, 0.2]` radians per second.
The proposal reads and records the actual instantiated bounds.

Each switch law is conditioned on its actual initial exact flag.
The probability of that event is retained once per underlying slider, even though the corresponding faucet or burner repeats the same switch state in its public features.
For the midpoint threshold and symmetric endpoint masses, the event probability is one half.
The unobserved faucet joint uses its original mixture directly and contributes no invented observation factor.

`RestingJointPrior` now exposes an unconditional coordinate map alongside its existing Boolean-conditioning interface.
The new map preserves both resting atoms and continuous motion; unused unit coordinates remain harmless auxiliaries in a rest case.
It does not modify the old conditional map or its prior identity.
Zero-measure coordinate seams and invalid numerical inputs remain explicit errors.

## Native validation

Bundle `logs/uncertainty_boil_articulated_prior_20260913` freezes the component, source program, recording inputs, and job scripts.
The experiment keeps the verified first-65-frame mean fixture point and the same historical parameter values.
It draws 16 joint-state triples with seed 401 and includes the original supported point twice as a repeat control.
Each of the 18 cases replays all 264 actions, for 4,752 native actions.
These are conditional histories with the recorded robot joints used throughout, not unconditional forecasts or agent seeds.

Before stepping, every requested joint position and velocity must match its native readback exactly, and every conditioned switch flag must match its actual observation.
The report records the original joint inventory, sampled units, original prior identities, event factors, complete trajectory likelihoods, and exact-output contradictions.
Initial joint-event factors are stored separately from the conditional trajectory likelihood and explicitly combined by the independent reader.
The reader independently reconstructs the mixture maps and rejects altered readbacks, missing event factors, and invented faucet-joint observations.
It also verifies all existing output and transition factors and both reference histories.

The driver saves each native observation before the historical rules execute.
A separate reader reconstructs those states and runs the literal historical rules, requiring exact agreement with all saved output predictions and model memory and rejecting emitted physical commands.
This checks a possible reuse boundary for cheaper later parameter evaluations.
It does not authorize reuse for other programs or establish that the entire joint-state inference problem is solved.

## Status and remaining composition

Compute checks `22687962` passed 16 functional tests, focused type checking, lint, and pinned formatting.
These include independent piecewise quadrature for the original mixed prior, both pure-component limits, unchanged conditional behavior, and invalid-coordinate cases.
The first check's untyped quadrature-library call was replaced by the analytic two-node Gaussian quadrature rule; the statistical reference itself had already passed.

Native job `22688041_0`, independent reader `22688102`, and literal-rule reader `22688171` all completed on `mit_preemptable`.
Their one-CPU allocation times are respectively 232, 208, and 22 seconds.
The instantiated inventory confirms two sliders with URDF interval `[0, 0.296]` and the separate faucet hinge with interval `[0, pi/2]`; the effective slider interval is `[0, 0.0296]` meters.
All 16 sampled joint triples have finite full conditional likelihoods, and both reference histories repeat exactly.
The samples include nonzero initial velocity in four faucet-switch draws, one burner-switch draw, and five faucet-hinge draws.
The independent reader verifies all 48 initial joint draws and 42,768 robot transition factors, with maximum arithmetic difference 3.553e-15, and rejects seven corrupted-input classes.
The literal-rule reader reproduces all 4,752 post-rule output frames and every recorded model-memory state exactly, with no new native actions.
This reuse check covers the tested parameter values; validating reuse across parameter changes remains a prerequisite for an optimized fitter.

The full scene model must still combine these joints with uncertain fixture height/orientation, jug pose and motion, robot nuisance state, physical support, and the fixed program's liquid/memory initialization contract.
Any feasibility conditioning must preserve its normalization and the original parameter prior.
The current fixture and articulated component studies do not replace that full composition, independent-fit agreement, held-out prediction checks, or live non-regression experiments.
