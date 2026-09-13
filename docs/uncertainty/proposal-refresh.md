# Exploring the fixed posterior with global block proposals

September 13, 2026.
This is an offline numerical change within Stage B of the [simplification proposal](simplification-proposal.md).
It leaves the original prior, conditional coordinate map, observation model and agent unchanged.

## Evidence motivating the change

The completed Domino runs select uniformly among 25 proposal blocks and make eight moves at each of 32 temperatures.
A given physical parameter therefore receives only 10.24 proposed updates per particle in expectation before accounting for acceptance or resampling ancestry.
The aggregate accepted-move count includes state and auxiliary coordinates, so it does not establish parameter exploration.

Compute job `22672492` evaluates conditional target slices at the two archived best scenes.
For each scene it checks seven prior quantiles of each physical parameter, holding all other coordinates fixed.
It also repeats both original reference points and reproduces their archived likelihoods exactly.
The worker performs 4,736 native actions and finishes in 40.19 seconds on four compute CPUs.

| Parameter | Finite log-target range at scene 100 | Finite log-target range at scene 101 | Finite grid points per scene |
| --- | ---: | ---: | --- |
| Lateral friction | 1334.23 | 402.91 | 7/7, 7/7 |
| Restitution | 0 | 0 | 7/7, 7/7 |
| Rolling friction | 727.03 | 108.52 | 7/7, 7/7 |
| Spinning friction | 135.64 | 6.98 | 7/7, 7/7 |
| Mass | 86.07 | 11.73 | 6/7, 6/7 |

Changing restitution leaves the complete predicted 64-action history byte-for-byte identical at both checked scenes.
Nevertheless, the completed populations place restitution in different narrow ranges: approximately [0.259, 0.635] and [0.689, 0.900].
The slices are conditional checks at two scenes, not a proof that restitution is globally unidentifiable or that its full posterior equals its prior.
They do show that narrow, different finite-population summaries cannot be taken as evidence of identification without an exploration check.
The other parameters have substantial conditional likelihood variation, so replacing all local moves with global proposals could sharply reduce their acceptance.
Inputs, physical predictions' hashes and complete slice values are retained in `logs/uncertainty_domino_parameter_slices_20260913`.

## Mixed proposal kernel

`SamplerConfig.refresh_probability` mixes local Gaussian random walks with independent uniform proposals within the selected block of the declared proposal box.
The block-selection distribution and mixture probability are fixed independently of the current state.
Both proposal components are symmetric on that box, so the existing Metropolis ratio remains valid, including the conditional-base density correction and tempered remaining likelihood.
A refresh proposes a new point; it does not automatically accept it, replace the fixed generative prior or discard accumulated evidence.
The mixture is a numerical exploration strategy, not another statistical uncertainty estimate.

Probability zero preserves the existing scalar and batch random streams exactly.
The checkpoint signature omits the newly added zero-valued field for compatibility, while a nonzero value participates in the signature and rejects mismatched continuation.
Serialized configuration reports include the explicit new field.
There is no agent-facing flag or deployment change.

## Numerical validation

A public-sampler reference reproduces concentration of an unobserved independent coordinate when an informative nuisance coordinate causes resampling to one ancestor and local moves are too small to explore its prior range.
Full-range block proposals recover the unobserved coordinate's uniform mean and variance in both scalar and batch execution across two seeds.
That recovery does not certify the separate approximation of the sharply observed coordinate.
A truncated Beta reference checks the nonlinear coordinate factor, mixed and fully global proposals, blocked and full-vector modes, and checkpoint replay.
Budget exhaustion still returns no usable posterior samples.

Compute job `22672869` passes all 41 functional tests and 32 comparisons against the previous sampler, including exact old-checkpoint continuation with refresh disabled.
Its static check exposed an inferred test-variable type of `object`; the fixture now explicitly annotates the callable-or-batched-target union.
Final check job `22673128` reran all nine new reference tests and 32 compatibility comparisons and passed two-file mypy, pylint and pinned formatting checks.
The follow-up checks and native process-equivalence validation retain frozen sources in `logs/uncertainty_refresh_checks_v2_20260913` and `logs/uncertainty_domino_refresh_validation_20260913`.

## Matched Domino experiment

The prepared comparison uses two numerical seeds, 100 and 101, for each of a local-only arm and a 50/50 local/full-range arm.
Both use 64 particles, 32 cubic-spaced temperatures, eight moves, the same 25 blocks, local scale 0.05 and at most 16,448 target evaluations.
Both evaluate complete targets through four isolated processes with the same frozen program, training prefix, conditional map, prior and output model.
The refresh probability is the manipulated setting; the sampled random trajectories can differ between arms.
Each run verifies an archived target point before fitting and records weighted parameter summaries and ancestry at every complete stage.
Checkpoints preserve cumulative numerical budgets across interruption.

The frozen plan also declares exploratory comparisons of their subsequent 97-action forecasts: at most 0.0025 m RMS difference between conditional position means, 0.20 maximum toppling-curve probability difference and 0.15 final toppling-probability difference between numerical replicas.
These are triage thresholds, not a posterior-adequacy certificate or confidence interval.
Parameter exploration, model mismatch, predictive errors and cost remain separate diagnostics.
Passing this pilot still requires budget sensitivity and broader comparisons before posterior use in planning.
The old production estimator remains the default.

Array `22673150` is submitted on `mit_preemptable`, dependent on successful final static and native validation (`22673128` and `22672945`).
Tasks 0 and 1 are the refresh arm; tasks 2 and 3 are local-only controls, with numerical seeds 100 and 101 in each arm.
Each task requests four CPUs on the same declared AMD worker node, with a four-hour limit and a maximum of two simultaneous array tasks.
The complete source, submission manifest, prospective prediction thresholds and checkpoints are in `logs/uncertainty_domino_refresh_comparison_20260913`.
Queued, interrupted and incomplete fits are not completed posterior results.

Native validation `22672945` completed successfully with a 50/50 proposal mixture.
All 32 fixed target evaluations and the subsequent 95-evaluation sampler result match exactly between synchronous and four-process execution.
The sampler portion took 96.91 seconds synchronously and 25.40 seconds in parallel.
This short run retains one ancestor and remains a kernel-equivalence check, not a numerically adequate physical posterior.
The validation dependency is satisfied, and comparison tasks 0 and 1 have started; the local-only controls await the two-task concurrency limit.
