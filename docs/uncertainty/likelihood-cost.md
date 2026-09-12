# Quaternion likelihood cost reduction

September 12, 2026.
The [Fan cost audit](fan-joint-inference.md#rollout-cost-audit) found that likelihood evaluation cost more than candidate initialization or native rollout.
The optimization changes scalar arithmetic inside the existing quadrature integrands, leaving the probability model and quadrature acceptance criteria intact.
It does not change an acting agent or the source snapshots of running fits.

## Profile and implementation

Profiling 64 archived Fan orientation readings under the original implementation found 42,189 calls to SciPy's general array `logsumexp` routine.
Those calls consumed 2.23 of the profiled 3.28 seconds.
Every hot integrand call sums exactly two nonnegative density contributions in log space, so general array construction and reduction are unnecessary there.

The implementation uses the stable scalar identity `max(a,b) + log1p(exp(min(a,b)-max(a,b)))`, with explicit handling when a contribution has log density minus infinity.
Both antipodal contributions remain in the density.
Fixed roll/yaw trigonometric terms and the fixed radial lower bound are also computed once per observation instead of once per quadrature callback.
The tail calculation, adaptive integration, numerical tolerances, ordinary/pole branches and statistical model identity are unchanged.
Runtime source identity still records the implementation change.

## Validation

Compute job `22645511` passed 23 existing functional tests, focused mypy and pylint, and the pinned formatting checks.
The functional suite includes independent analytic densities, native Gaussian sampling references, branch probabilities, angular support, the recorded near-pole regression and complete observation composition.
These probability checks complement comparison with the old implementation.

The same job evaluated every archived orientation case across five domains, four discrepancy scales and two quadrature tolerances.
All 2,560 old/new log densities matched exactly on the measured runtime.
Total unprofiled density time fell from 79.24 seconds to 19.34 seconds, a 4.10-fold speedup.
Method order alternated across paired cases.
The [comparison report](../../logs/uncertainty_orientation_scalar_checks_20260912/parity-22645511.json) retains every value and timing, along with the actual node1387 Intel Xeon Gold 6230 runtime.
Exact equality on these cases is measured evidence, not a universal claim about all floating-point inputs.

An independent complete-history check, `22645714_0`, reran five saved physical Fan candidates with both implementations on the same worker.
Every public prediction and complete likelihood matched exactly, including the two event-incompatible candidates.
Cold full-likelihood times fell from 4.88-5.01 seconds to 1.35-1.39 seconds, approximately 3.6-fold faster.
Initialization and native rollout still contribute to total inference cost, and active samplers also benefit from cross-candidate caching.
Consequently, the likelihood timing ratio is not a claimed end-to-end sampler speedup.
The [full Fan report](../../logs/uncertainty_orientation_e2e_20260912/pilot-22645714_0.json) records the paired histories, likelihood differences and timings.

The earlier standalone profile job `22645118` was cancelled while pending after these stronger paired checks and the original-code profile completed.
No running inference experiment was cancelled, restarted or modified.
Subsequent source snapshots can use the checked implementation while retaining all inference-adequacy and prediction gates.
