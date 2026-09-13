# Fan prefix comparison

This is a Stage B development comparison under the [simplification proposal](simplification-proposal.md).
The existing full-recording Fan inference pilots use all 132 actions, so they cannot provide an unused suffix from that recording.
The separate legacy fit `22638308_1` uses 64 actions and already supplies predictions for the remaining 68 actions.

## Prefix-only support check

Job `22674026` completed successfully on `mit_preemptable` on September 13, 2026, with one CPU on node1412.
It used 106 allocation seconds, 90.47 worker seconds and 4,160 native actions under its 20-minute allocation limit.
Its frozen bundle is `logs/uncertainty_fan_prefix_support_20260913`.
The submission manifest records worker and configuration hashes.

The worker retains only the initial observation and first 64 action/observation transitions before constructing any candidate scene or scoring predictions.
It checks that the sensor schema and every retained observation use exactly the first observation's feature keys.
The full recording hash remains provenance, while the inference data identity describes only the retained prefix.
It uses the same historical program, physical runtime, original scene prior and output discrepancy model as the full-recording Fan pilots.
The historical program may have seen later training experience during synthesis, so this is an estimator comparison on a reserved suffix, not an unseen-program evaluation.

The search does not import the full-recording supported candidate or its guided proposal.
It checks 16 stratified original-prior fan speeds at the first-observation proposal median scene, followed by 48 independently sampled scenes from the original first-observation proposal restricted to its robot/ball rest components.
Rest-component restriction is a declared support-search choice; these candidates are neither joint-prior samples nor posterior samples.
Unknown rotor state and fixture locations retain the declared chart semantics.
No omitted velocity is claimed to be an exact observed zero.

Every candidate is simulated for 64 native actions in a fresh world, with geometry witnesses, full-prefix likelihood, initial likelihood and complete prediction digest retained.
The first feasible finite candidate is repeated exactly; if none is found, the first median candidate is repeated instead.
A finite search with no supported candidate cannot prove that the target has no support.

## Acceptance and follow-up

The completed native audit found the following support:

| Candidate construction | Probes | Valid initial geometry | Finite likelihood and valid geometry |
| --- | ---: | ---: | ---: |
| First-observation proposal median, stratified speed | 16 | 0 | 0 |
| Sampled first-observation proposal, robot/ball rest cases | 48 | 12 | 8 |

The first supported candidate, index 26, repeated with exactly matching likelihoods, geometry witnesses and complete predicted-observation digest.
The median scene has a 5.11 mm initial penetration and is unsuitable as a fixed-state baseline without a separately declared feasible selection policy.
The independent verification checks all 64 result records, native action accounting and all submitted script hashes.
Reports and verification are retained in the frozen bundle.

This establishes prefix-only support for constructing a prefix-specific inference proposal; it does not establish numerical adequacy.
Any selected guide must depend only on the prefix, and its proposal density must be included in the inference correction.
Keep a component with full original support if adding local guidance.
Independent numerical replicas and budget sensitivity remain required before treating fitted samples as usable posterior estimates.
Only then compare the frozen-weight predictions on the remaining 68 actions against the existing legacy predictions, including ball trajectory, event timing, goal outcomes and computation cost.
This support audit produces no agent solve-rate result and does not change the production fitter.
