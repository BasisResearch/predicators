# Boil fixture-state proposal

September 13, 2026.
This extends the [supported point-state audit](boil-transition-and-initial-state.md) toward uncertain initial-state inference.
It is a restricted component experiment with the historical filling/heating parameters and all other initial-state coordinates held fixed.
It does not complete the full scene prior or establish a replacement posterior.

## Prior and proposal are separate

The eight coordinates are x/y for `faucet_switch`, `faucet`, `burner_switch0`, and `burner0`.
Their declared development prior is a product of uniforms with x in `[0.35, 1.15]` and y in `[1.05, 1.65]` meters.
These bounds use the public workspace with five centimeters of padding; recorded fixture centers and exact task-generator placements do not define them.
The component is conditional on the diagnostic's fixed heights, orientations, and remaining point state.
It must not be presented as the original prior over the whole physical scene.

For each coordinate, the guide uses only the first 65 public observations and the existing constant-output likelihood helper.
It retains the declared sensor noise and AR output discrepancy with persistence 0.9, innovation standard deviation 0.005 meters, and zero initial error.
Its center is generally different from the arithmetic mean used in the earlier point-state probe.
The guide is a Gaussian truncated to the declared prior bounds.

The proposal is a mixture of the original uniform vector with probability 0.25 and the guided vector with probability 0.75.
One mixture indicator selects the whole vector; the density is a mixture of products, not a product of coordinate-wise mixtures.
The new offline `GaussianBoxProposal` component returns both the physical candidate and `log(prior/proposal)`.
The uniform branch preserves the complete box support and bounds this importance ratio above by four.

Every original output observation and joint-transition factor remains in the likelihood once.
Using the prefix to construct a proposal does not remove those observations or multiply their likelihood a second time.
Proposal identities change when the guide changes; the original prior identity remains fixed.

## Native audit

The frozen bundle is `logs/uncertainty_boil_fixture_proposal_20260913`.
The audit retains the original point, the two supported arithmetic-mean controls, and an exact repeat of the original point.
It adds the correlated-likelihood guide center and 24 stratified mixture draws: six from the uniform component and eighteen from the guided component.
All 29 cases replay the same 264 recorded actions, totaling 7,656 native actions.
These are conditional histories using the recorded exact joints throughout, not unconditional future predictions or new agent seeds.

Each history records the requested and actual fixture positions, complete output likelihood, nine joint-transition factors per action, exact-output contradictions, and initial fixture-pair intersections.
Intersection checks describe sampled geometry without silently rejecting candidates and changing the normalized box prior.
Passing fixture-pair checks alone does not establish whole-scene feasibility: moving bodies, table contacts, attachments, orientations, and articulated state still need their own supported composition.
If later inference conditions on a feasible scene, its normalization and any parameter dependence must be handled explicitly.

The independent reader reconstructs the 65-frame Gaussian covariance to verify the guide centers, scales, and likelihood constants.
It checks mixture densities and inverse quantiles without calling the proposal mapper, then verifies every output score and joint factor from saved histories.
Corrupted importance weights, proposal densities, joint readings, likelihood factors, exact-switch witnesses, and model identities must be rejected.
The point and repeat controls must retain the previously verified histories exactly.

## Validation and remaining gate

Independent quadrature tests check proposal normalization, the joint mixture law, recovery of original-prior moments, and fixed-prior posterior evidence and moments under three different guide centers.
Additional tests cover finite support endpoints, deterministic mapping, the defensive weight bound, identities, and invalid inputs.
The first compute check passed all 27 proposal/output-error tests and focused type checking; its two test-file lint findings were corrected before resubmission.
Final compute checks `22687335` completed successfully: 27 functional tests, two-file type checking and lint, and pinned formatting.
Native audit `22687337_0` and independent reader `22687339` completed on `mit_preemptable`, using respectively 379 and 327 allocation seconds on one CPU.
All 18 guided random draws have finite complete conditional likelihoods and no detected initial fixture-pair intersections.
All six uniform-component draws have zero complete likelihood; two also have fixture-pair intersections.
The guide center and both previous mean controls remain supported, and the original failed point repeats exactly.
The reader verifies all 68,904 joint-transition factors, with maximum independent calculation difference 3.638e-12, and rejects all six corrupted-input classes.
Its dense Gaussian reference validates all eight guide summaries and all 24 proposal densities and inverse quantiles.
The coordinate guide standard deviation is 4.74818 millimeters under the retained temporally correlated output model.
The cancelled dependent jobs from the first lint failure performed no native actions and are setup outcomes, not failed inference trials.

These support results justify incorporating this component into the remaining initial-state composition with the fixed parameter prior and complete likelihood accounting.
The 18 supported conditional draws establish local support under this proposal, not an 18-seed agent solve rate, a posterior estimate, or independent calibration.
Neither a supported draw nor a successful numerical component test establishes posterior adequacy, held-out prediction quality, or unchanged agent performance.
Production fitting, planning, and execution estimation remain unchanged.
