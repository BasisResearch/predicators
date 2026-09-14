# Domino transition model with uncertain starting states

September 14, 2026.
This is an offline composition check under the [simplification proposal](simplification-proposal.md).
It addresses a remaining limitation of the [joint-transition comparison](domino-joint-transition.md): that comparison fixes the initial physical state.
Its completed 128-particle results still disagree with one smaller-budget result on final toppling probability, so neither numerical adequacy nor migration acceptance is established.

## Probability model and scope

Retain the existing 95-coordinate proposal for five physical parameters and the uncertain initial scene.
The original state prior, geometry support, parameter prior, exact initial conditioning and prior-to-proposal correction remain unchanged.
Combine that initializer with the already declared joint-transition discrepancy: after each native action, joint positions have independent Gaussian transition error of scale 0.001 in their coordinate units.
The exact joint readings determine the innovations during fitting, and their transition densities enter once.
The old joint output-error factors are absent; other output factors and the native Cartesian observation phase remain unchanged.

The initial observation contributes through the original conditioned base weight.
The remaining 64-action likelihood excludes that initial factor, so their sum retains it exactly once.
Unlike the fixed-state approximation, the base weight varies across initial scenes and must not be discarded as a constant.
Each candidate is replayed from its own initialization.
There is no mid-trajectory state restoration.

This study checks composition and support at declared candidates.
It does not fit a posterior, estimate parameter coverage or evaluate an agent.
It also does not establish that the alternative transition model predicts better than the original output-error model.

## Cases and controls

The fourteen cases are selected before this native check:

- Eight retained joint proposals: four distinct positive-weight proposals from each completed 128-particle joint fit, selected by descending weight with lowest-index ties.
- Two archived fixed-state controls, expanded to the same 95-coordinate representation.
- Four unselected deterministic hash-uniform proposal vectors, retaining any failed scene support explicitly.

Retained proposals are diagnostic cases, not fresh samples from the prior or an adequate posterior.
The unselected vectors are in proposal coordinates; they are not automatically draws from the conditioned scene prior.
Their original density corrections and support checks remain necessary.

For every case, repeat the original joint target and verify retained proposals against their archived physical coordinates, base weights and likelihoods.
Evaluate the alternative transition target and repeat two complete uncertain-state histories.
The fixed-state controls must reproduce every archived numeric field and observation exactly; only their case identifiers are remapped.
Unsupported initial scenes remain explicit and consume no claimed native action steps.

For predeclared uncertain cases 0 and 4, generate two sixteen-action futures if the fitting prefix is supported.
Unsupported cases are reported as unavailable and are not replaced after inspecting future outcomes.
Generated future observations must reproduce the complete histories and densities under the separate conditional evaluator.
No future observation enters an unconditional prediction or the fitted prefix.
This recording is development data and cannot substitute for untouched final evaluation.

## Verification and status

The reader independently reconstructs every Gaussian joint factor, checks the initial-prior and remaining-likelihood accounting, verifies original proposal identities and recomputes native action counts.
It also repeats four complete histories in fresh worlds and rejects deliberately altered base weights and transition sums.
Other output-law components retain their existing validation; this reader is not a new independent implementation of the entire output model.

Preparation checks pass for Python and shell syntax, fourteen coordinate mappings, frozen input hashes and four archived-history scalar references, including rejection of corrupted base weights.
Those preparation checks initialize no simulator and do not establish native composition parity.
Native job `22766058` and dependent reader `22766059` are submitted on `node1412` in `mit_preemptable`, each with four CPUs and 20 GiB.
Their initial twenty-minute requests were subsequently reduced to five minutes using the verified workload evidence below.
Both were pending at the latest scheduler check.
Frozen inputs, selected cases, source hashes and outputs are in `logs/uncertainty_domino_uncertain_transition_20260914`.

Inspect native support, archived-control parity and the reader before deciding whether a full uncertain-state comparison is justified.
Even a successful preflight leaves the original numerical-budget disagreement, predictive assessment and later planning gates unresolved.
The acting agent remains unchanged.

## Allocation correction

Both jobs were still pending with zero elapsed time and restarts when their allocation time limits changed from twenty to five minutes.
The requested node was already idle; the scheduler reported a priority wait, so adding another node was not justified by insufficient capacity on this node.
Historical native preflight `22683003` completed 9,152 actions in a 91-second allocation on the same node and four CPUs.
The new preflight has an upper bound of 3,456 native actions, including both old-target repeats, all new cases, repeated histories and four generated/density round trips.
Earlier independent reader `22684436` completed its larger saved-history check in 31 seconds; the new reader adds four fresh 64-action replays.
These timing references support the five-minute requests with margin; they do not guarantee a start time.

Only pending allocation limits changed.
Job IDs, dependencies, CPUs, memory, node selection, frozen sources and every scientific evaluation count were verified unchanged.
The before/after scheduler records and input-seal verification are in the bundle's `allocation.json`.
Both jobs remained pending after the update.
