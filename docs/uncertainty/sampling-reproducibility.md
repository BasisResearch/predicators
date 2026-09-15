# Joint-sampling reproducibility and conditioning

September 12, 2026.

The first [Domino pilots](domino-joint-inference.md) completed but lost parameter diversity.
Follow-up diagnostics also found that their uncontrolled process initialization prevented a clean numerical repeat-run comparison.
The results below separate those issues before another inference experiment.

## Reproducing a physical candidate

The first weight audit, jobs `22636950_100` and `_101`, regenerated the same unit-coordinate candidate sequences but obtained different finite-likelihood counts from the original pilots.
That audit omitted the original preflight lifecycle, so it was not sufficient to attribute the difference.
Jobs `22637147_200`, `_201` and `_202` restored that lifecycle and used the same candidate RNG seed 100 while controlling the Python hash seed.

The initializer constructed its object dictionary from a set.
The environment copied that insertion order into its object list during `_set_state`, so hash order could affect the ensuing physical replay.
With different hash seeds, 28 of 32 prediction digests differed.
Even with hash seed 0 in both processes, four candidates differed slightly in their generated physical poses, and four prediction digests differed.
The pose differences were in the last floating-point bits, including an approximately 1e-16 m position difference.
The sampled unit coordinates and fitting-data identity were identical.
This observation does not identify a specific math-library routine as the cause, but it rules out assuming bitwise identical coordinate transforms on all compute nodes.

The corrected control sorts objects by name and type before semantic initialization and supplies identical saved physical candidates rather than recomputing their quantiles.
Jobs `22637238_200`, `_201` and `_202` replayed 31 such candidates on AMD EPYC 7542, AMD EPYC 9474F and Intel Xeon Platinum 8462Y+ CPUs, using hash seeds 0 and 1.
Every physical-candidate record, full 65-frame prediction digest and complete log likelihood matched exactly across all three processes.
Twelve of those candidates had finite complete likelihood in each process.
The remaining candidates contradicted exact outputs consistently; they were not setup failures.

This closes the tested cross-process replay discrepancy when physical candidate values and initialization order are fixed.
It does not establish complete runtime closure, hardware-independent candidate quantile generation or portable mid-trajectory engine restoration.
New sampling runs therefore use canonical object order, explicit hash seed 0 and the same declared compute node/CPU for independent chains.
Those controls are included in the experimental runtime identity, and the worker checks the hash seed and CPU model before fitting.
The incumbent environment and its action behavior have not been changed.

## Separating initial conditioning from future evidence

The regenerated weight diagnostics show concentration before a substantial future-likelihood temperature is applied.
For the two same-hash seed-100 processes, the effective sample size of prior/proposal weights restricted to finite-likelihood candidates was approximately 1.71 before tempering.
An informative proposal around the first observation can have poor importance weights against the unconditioned physical prior even though it is useful for the final posterior.
Merely choosing a smaller first temperature does not remove that initial weighting problem.

The next experiment changes the intermediate sampling distributions while preserving the complete target.
Writing the full likelihood as `L(o0, o1:T)` and the initial-frame likelihood as `L0(o0)`, it uses:

- Initial base weight: `(original prior / proposal) * L0`.
- Remaining likelihood: `L(o0, o1:T) / L0`.

The product is exactly the original full-data importance weight.
The fixed original parameter prior and the physical initial-state prior are unchanged.
The initial observation enters once, and the remaining factor is conditional on that observation, including dependencies represented by the output-discrepancy model.
An initial observation with zero likelihood rejects the candidate before this division; undefined conditional likelihoods are not assigned an invented value.
Geometric and exact-output support constraints remain in force.
This construction is an initial-observation-conditioned sampling path, not a new prior centered on a previous fit.

## Declared temperature schedules

`SamplerConfig.temperature_schedule` optionally supplies a fixed, strictly increasing sequence ending at one, with the configured stage count.
An empty sequence retains equally spaced temperatures and the original random stream.
Reweighting uses the actual temperature increment, and Metropolis moves target the corresponding distribution with the same conditional-base factors.
The schedule changes exploration, not the final posterior or the evaluation budget.
It is not an adaptive-temperature implementation or an adequacy certificate.

Job `22637164` passed 19 functional tests, two-file type checking and lint, pinned formatting, and eight exact default-result comparisons against commit `4455bdb1a`.
The new nonlinear-schedule reference checks a narrow conditional Gaussian against its analytic mean and variance while verifying an uninformed parameter's prior marginal.
Explicit linear schedules also reproduce the implicit default exactly, apart from the configuration field itself.

## Next physical experiment

Jobs `22637359_100` and `_101` use the initial-conditioned path, 64 particles, 32 cubic-spaced temperatures, eight moves per stage and at most 16,448 target evaluations per run.
Physical parameters move individually; body horizontal position, supported yaw/moving height, and other body coordinates use separate declared groups.
The rest/moving prior, complete observation model and 64-action fitting window are retained.
The runs are pinned to `node1412` on `mit_preemptable` to keep candidate transforms on the same CPU model.

Independent-run agreement, parameter movement, weight concentration, budget sensitivity and held-out prediction still need assessment.
Completion or a higher acceptance count cannot establish numerical adequacy, especially when some moves change only auxiliary coordinates.
The older pilots lack these runtime controls and cannot serve as a controlled sampler baseline.
These are offline inference experiments, not new agent seeds, and they do not authorize deploying or retiring either fitter.

Frozen scripts, configurations and reports are in `logs/uncertainty_domino_weight_audit_20260912`, `logs/uncertainty_domino_process_audit_20260912`, `logs/uncertainty_domino_fixed_candidates_20260912`, `logs/uncertainty_schedule_20260912` and `logs/uncertainty_domino_initial_conditioned_20260912`.
