# Ordered batch evaluation for offline inference

September 13, 2026.
This implements an evaluation-throughput component of the [simplification proposal](simplification-proposal.md), without changing the production estimator.
The motivation is the multi-hour physical posterior pilots and their unresolved prediction disagreement, documented in [Domino joint inference](domino-joint-inference.md).
More cores can reduce evaluation latency; they do not resolve missing posterior modes or certify numerical adequacy.

## Contract

`BatchedTarget` accepts an immutable ordered tuple of proposal coordinates and returns one `TargetEvaluation` per proposal.
Each result includes the original proposal identity, the complete joint coordinates, the conditional-base/proposal log factor and the remaining log likelihood.
The sampler verifies cardinality, proposal order, output dimensions and box-prior invariants before consuming any batch.
Coordinates must be finite, and log factors may be finite or negative infinity; NaN and positive infinity are errors.
A zero conditional-base factor requires zero target support.

Each worker must own the entire conditional map and likelihood evaluation.
The physical conditional map creates a candidate-specific native world that the likelihood consumes, so concurrently calling the existing two callbacks against shared state is unsafe.
The caller supplies a synchronous ordered map or an ordered process-executor map, owns the executor lifetime and identifies all dependencies in the immutable runtime identity.
Setup and worker failures propagate rather than becoming zero-likelihood candidates.
No worker may drop, replace or silently retry an evaluation.

## Sampler integration

`sample_batch` uses the same tempering, conditional density corrections, weights, resampling and symmetric Metropolis acceptance calculation in both modes.
The new mode evaluates all valid proposals in a mutation sweep together, then applies accepted updates in particle order.
Each particle's proposal is independent of the other particles' acceptance decisions within that sweep.
A proposal outside the declared support is rejected without clipping or target evaluation.

The new mode draws an acceptance uniform for every proposal, including proposals outside support and proposals whose likelihood is zero.
This makes its random stream independent of worker completion order and evaluation outcomes.
It is a separately identified random schedule, `tempered_smc_ordered_batch_v1`, and is not bitwise equivalent to the original scalar schedule.
The existing scalar path retains its original schedule and checkpoint signature.
Checkpoints reject cross-mode continuation, while synchronous and parallel maps within batch mode must reproduce the same population, weights and diagnostics exactly.
Callers record the selected schedule with their frozen runtime and experiment provenance.

The sampler reserves the remaining evaluation allowance before dispatching workers.
If a full batch would exceed it, only the permitted prefix is evaluated, and the run returns `budget_exhausted` with no posterior samples.
Only complete initialization and temperature boundaries produce checkpoints.
A failed or interrupted batch therefore cannot publish a partial stage as a completed posterior.
Resuming preserves the cumulative numerical budget; repeated work after interruption remains a separately reported compute cost.

## Validation

The numerical references include an analytically solvable truncated Beta distribution after a nonlinear coordinate transformation, an uninformed uniform parameter, exact process-map agreement, zero-support points, budget exhaustion and interrupted-stage recovery.
A separate comparison uses the pre-change scalar implementation to check complete result and checkpoint equality across conditional/unconditional targets, blocked/full-vector proposals and multiple seeds.
The checks run on compute nodes, with frozen sources retained in `logs/uncertainty_batch_evaluation_v2_20260913`.

A native Domino validation uses the existing conditional initialization and full 64-action likelihood as one indivisible worker operation.
It compares fixed candidate evaluations and a short complete sampler run between synchronous evaluation and four isolated processes.
It measures initialization and worker startup overhead separately from the steady-state sampler time.
Its small population and short schedule test execution equivalence and cost only; they do not provide a usable posterior or agent performance result.
Frozen inputs and worker source are retained in `logs/uncertainty_domino_batch_evaluation_20260913`.

## Completed native and reference checks

Compute job `22672354` completed on node1412 with four allocated CPUs.
All 32 fixed native target records match exactly across execution modes, including proposal/joint coordinates, conditional-base factors and remaining likelihoods.
Twenty-two of those records have finite remaining likelihood.
The subsequent 32-particle, two-temperature, one-move sampler completes all 96 evaluations in both modes with exactly identical samples, weights and diagnostics.
This deliberately short run retains one original ancestor and does not establish numerical adequacy.

| Measurement | Synchronous evaluation | Four isolated processes |
| --- | ---: | ---: |
| Initialization plus 32 fixed candidates | 52.08 s | 23.38 s |
| Subsequent 96-evaluation sampler | 98.50 s | 26.19 s |
| Combined measured work | 150.58 s | 49.57 s |

The measured sampler portion is 3.76 times faster; the combined work including initialization and fixed-candidate validation is 3.04 times faster.
These are single matched latency measurements on the declared hardware, not a reduction in logical evaluations or a guarantee for larger fits and other domains.

The corrected reference run `22672392` passes 32 functional tests and 16 exact comparisons against the original scalar implementation, including every emitted checkpoint.
The first run, `22672316`, exposed a test error: a support assertion inspected zero-weight rows retained when resampling is skipped.
The corrected assertion checks positive posterior mass, and the original failed report remains archived.

Final check job `22672431` completed successfully after replacing a dynamically constructed invalid-field test with explicit typed calls.
It reran all nine batch-target tests and the sixteen scalar parity comparisons, and passed three-file mypy, pylint and pinned formatting checks.
The preceding 32-test suite remains valid for the unchanged implementation; the final fixture and static-check artifacts are in `logs/uncertainty_batch_evaluation_v3_20260913`.
The implementation files match the frozen native-validation sources byte for byte.
