# Revised Bridge subclass and exact-rate conditioning

This follows the [Bridge program-revision diagnostics](bridge-glue-attribution.md).
It preserves the useful revised candidate through the current simulator interface and verifies one required probability-model component.
Neither result establishes a complete Bridge posterior or changes the acting agent.

## Complete subclass parity

The revised candidate now uses `AGENT_PARAM_SPECS`, observation-driven `MODEL_STATE_INIT`/`update_model_state`, and the native `_domain_specific_step` hook.
The observation callback carries glue levels, bond dwell counters and bonds, computes feature outputs, and records the attachment commands to apply.
The native hook queues those commands for the next action; the state getter exposes the predicted glue features.
The converted functions do not read trajectory history, which is checked against their source before conversion.
The historical rules remain a frozen reference for this conversion, rather than a second new production interface.

Job `22696083` completed four full 1,186-action native trajectories in 1:53 of allocation time on `node1412` under `mit_preemptable`.
The four cases are the revised default, its repetition, the clean-prefix-derived geometry parameter pair, and the historical geometry parameter pair.
Every observed frame, encoded learned-memory value and attachment-command count/type matches the independently verified literal reference exactly in all four cases.
The report contains all 4,744 native steps, and the completed artifacts were also compared directly against the frozen reference after the job finished.
This establishes conversion parity for these cases, not arbitrary simulator restoration or unchanged stochastic agent conversations.

The frozen bundle is `logs/uncertainty_bridge_revised_subclass_20260913/`.
It contains the subclass factory, complete native parity driver, source/input identities and verified artifacts.
The candidate remains an offline development model; its remaining exact-output disagreements are preserved.

## Exact glue readings constrain the rate directly

The revised program adds a rate `r` to its partial glue level and publishes `1` once accumulated progress reaches a latch threshold `L`.
Starting from an exactly observed zero level, a positive partial reading `y < 1` after one deposition implies `r = y`.
Drainage and bond consumption cannot create a positive partial reading.
The geometric event that permits deposition is still a separate constraint; solving for the rate does not guarantee it occurs.

The new component reference uses the already implemented affine-conditioning chart for `y = r`, retaining the original rate-prior density and unit Jacobian.
It does not draw continuous rates and reject unequal observations, introduce sensor noise on the exact reading, or count later deterministic repetitions as independent rate measurements.

Under the explicitly declared component priors `r ~ Uniform(0.05, 1)` and `L ~ Uniform(0.4, 1)`, the progression `0, 0.2, 0.4, 1` gives:

- `r = 0.2` from the partial reading, with density factor `1 / 0.95`.
- `L > 0.4` from the second partial reading.
- `L <= 0.6000000000000001` from the third addition in the actual floating-point program.

Thus the remaining latch distribution is uniform on that interval for this fixed deposition schedule.
Its mean is approximately `0.5`, and integrating the retained rate density and latch constraint gives approximately `0.3508771929824563`.
This is a density/mass factor for the stated component representation, not a probability of the full recording or a Bayes factor between arbitrary programs.
The uniform boxes are declared development assumptions inherited from the candidate's bounds, not a claim that those bounds were chosen before all historical learning data.

Job `22696234` completed the component reference in 15 allocation seconds on the compute node.
It evaluates 600 latch candidates through the actual revised deposition function and compares their exact supported/unsupported traces with an independent interval calculation.
The retained weights reproduce the analytic integral and conditional mean.
Additional checks preserve an uninformed height coordinate under the fixed eligible schedule, reproduce the same result when reusing the original prior, and verify that changing the original rate-prior width changes the evidence factor.
The check performs 1,812 rule steps and no recorded native actions.
Artifacts are frozen in `logs/uncertainty_bridge_rate_conditioning_20260913/`.

## What remains before a Bridge inference comparison

The component check supplies a valid treatment of the rate equality under its stated schedule; it does not condition the complete native trajectory.
A complete target must still retain geometric eligibility, partial-level drainage, latch and bond events, and every remaining exact and noisy observation.
It also needs the physical initial-state prior and a supported representation for exact joint/readout constraints, with their probability factors retained.
The discrete bond dwell and the other continuous parameters remain free unless the chosen fitting observations constrain them.

The current 124-action diagnostic prefix contains no bond event, so its later bond predictions should retain uncertainty about dwell rather than treating the default dwell as learned.
Choose and freeze the fitting split and complete prior before launching the estimator comparison, and distinguish any changed split from the existing geometry diagnostics.
Use the same revised subclass for both estimator arms and retain the old inconsistent programs as negative controls.
Stage A/B acceptance remains open; production continues with the incumbent uncertainty handling.
