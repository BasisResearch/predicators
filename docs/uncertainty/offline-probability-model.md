# Offline probability model prototype

Updated September 12, 2026.
This implements part of Stage A in the [simplification proposal](simplification-proposal.md).
It is an offline reference implementation; the acting MB agent still uses the incumbent estimator.
The numerical reference suite and five-domain recording-integrity audit passed.
No five-domain posterior-inference or agent-performance result is claimed.

## Raw evidence and identity

[The data module](../../predicators/code_sim_learning/inference_data.py) snapshots scalar measurements, primitive actions, reset episode identities, and observation step indices into immutable tuples.
Repeated identical reads at one step are deduplicated; conflicting reads at that step are rejected.
Equal values at different steps remain separate measurements.
Missing measurements are explicitly absent entries rather than fabricated values or NaNs.
The likelihood requires a prediction for every primitive state index, including the initial state, and checks that episode identities and trajectory lengths match the ledger.

A level boundary does not by itself define a reset episode.
The recording adapter identifies actual reset markers and must preserve continuous history across any level changes without a reset.
It must also record a unique run identity as part of each reset episode ID.
The separate read-only recording adapter now loads explicitly selected flushed continual levels and checks every action against the reset/action log.
Its writer-to-reader tests passed, as did the audit of 1,978 primitive actions from five frozen training levels.

The statistical identity separates hashes of data, sensor semantics, program and parameter definitions, prior, and runtime configuration.
Sampler seed and numerical settings are stored separately in the result.
Program and runtime hashes are caller-supplied artifact hashes; there is no automatic discovery of imported dependencies or simulator configuration.
Before domain use, the recording adapter must freeze those artifacts and verify their completeness.
A digest alone cannot establish that a caller supplied the correct program or data to a likelihood callback.

The `from_state` helpers snapshot object feature arrays only.
They exclude simulator metadata, privileged state, and inferred memory.
The recording adapter additionally projects public joint positions and mobile base pose as exact observations.
Extra metadata requires an explicit exclusion reason; unclassified fields cause rejection.
Recorded body velocities and command-weld metadata therefore require a declared modeling decision before a domain comparison.
No inferred memory or privileged payload is accepted as a measurement.

## Sensor distribution

For each nonzero declared sensor standard deviation, the log likelihood is

$$
\log p(o_f\mid s_f) = -\tfrac12 ((o_f-s_f)/\sigma_f)^2
                       -\log\sigma_f-\tfrac12\log(2\pi).
$$

Feature classification comes from the same `ObservationNoise.feature_sigma` used by the actual injector.
Angles use the raw stored coordinate difference, including differences across a full turn.
Scalar readings remain unclipped.
No additional motion scale, robust residual, or inferred discrepancy variance is introduced.

Zero-sigma predicted features are exact constraints: equal values contribute zero log likelihood and contradictions contribute negative infinity.
There is no arbitrary epsilon or tiny Gaussian variance.
This may expose real replay mismatch; that mismatch requires investigation rather than silently increasing sensor noise.
An exact feature may instead be explicitly declared an exogenous conditioned input.
The likelihood then retains its recorded value without scoring it, and the future simulator adapter must explicitly consume that input.
This classification must be fixed before comparison; it cannot be used to dismiss inconvenient prediction errors.
No noisy feature can be declared an exact conditioned input.

A missing predicted value for an observed output is an error.
A missing measurement adds no likelihood term.
The missingness mechanism is assumed ignorable in these references; informative missingness needs its own model.

## Joint prior and numerical method

[The sampler module](../../predicators/code_sim_learning/inference_sampling.py) accepts a named joint vector containing shared dynamics parameters and uncertain initial-state coordinates for reset episodes.
The implemented reference prior is a product of normalized uniform distributions with finite bounds.
Known exact initial values must be conditioned outside this vector.
This is appropriate for the small numerical examples, and is not yet a feasible geometry, attachment, velocity, or memory prior for the physics domains.
Correlated and constrained physical priors remain unimplemented.

Each call initializes from the same original prior and evaluates the complete batch likelihood.
It never adopts previous posterior weights as the new prior and does not use the first observation as a second prior factor.
The initial prototype deliberately supports only prior initialization, so there is no uncorrected observation-guided proposal.

The temperature schedule is fixed from zero to one.
At each temperature, previous importance weights are multiplied by the likelihood increment and normalized.
Multinomial resampling occurs only when effective sample size drops below the declared fraction, initially one half of the particle count.
Symmetric Gaussian random-walk Metropolis moves then target that temperature's distribution inside the prior bounds.
Weights are retained through those target-invariant moves when no resampling occurs.
Out-of-support proposals are rejected rather than clipped.
The previous every-temperature resampling prototype failed the symmetric two-mode reference.
The conditional-resampling correction passed the original regression and independent eight-seed comparisons at 1,024 particles.
No tempering step is skipped to meet the evaluation budget.

The result contains joint samples, retained importance weights, weighted empirical marginal quantiles, the original prior, identity, seed, configuration, evaluation count, pre-resampling effective sample sizes, move counts, resampling count, and surviving initial ancestors.
Quantiles use the inverse weighted empirical CDF and exclude zero-mass samples; they do not interpolate across a gap between modes.
Completion means the configured algorithm reached temperature one, not that it discovered every mode or passed predictive adequacy checks.
Effective sample size and ancestor counts cannot certify exploration or calibration.
Independent runs and comparisons against numerical references remain necessary.

Budget exhaustion returns a failed numerical result with no posterior samples or intervals.
If initialization finds no finite likelihoods, the status is `no_particle_support`.
That is not proof that the data are impossible: finite prior sampling may have missed feasible support, particularly for exact constraints.
The implementation does not yet recover from that failure by constructing constrained proposals.
Simulator/setup exceptions and invalid likelihood numbers propagate rather than being counted as agent or model failures.

## Reference checks and remaining gates

The saved compute script is `logs/uncertainty_probability_20260911/checks.sbatch`.
It includes the following tests:

- The real noise injector against an independently evaluated Gaussian density, with raw angular differences and exact constraints.
- Cached observations, sparse measurements, trajectory alignment, immutable copies, and invalidation when any statistical identity component changes.
- A stationary unknown position against its analytic Gaussian reference, using actual injected observations and the batch ledger.
- Unknown initial position plus constant velocity against dense grid integration, including their correlation and an independent uninformed parameter.
- A two-mode likelihood, repeated identical fits, strict evaluation budgets, zero sampled support, and simulator exceptions.

Slurm submission failed because the controller could not be contacted, and a bounded queue query timed out.
No compute validation job ID was obtained.
Before the resampling correction, the stationary Gaussian and correlated position/velocity grid tests passed.
The two-mode test initially exhausted its mistakenly undersized test budget.
With that budget corrected, the same seed assigned 72.2% positive-mode mass where symmetry implies 50%, failing its existing 35%-65% acceptance range.
The acceptance range was preserved, and independent seeds 19, 0, 1 and 2 are now included in the prepared suite.
The new weighted implementation has not yet passed that suite.
A subsequent bounded Slurm submission attempt also timed out without returning a job ID; its submission status cannot be verified from this session.
The earlier scoped checks and the current unvalidated changes are distinguished in [implementation progress](implementation-progress.md).

Before fitting real recordings, complete these numerical checks, independent-seed and increasing-budget comparisons, feasible initial-state priors, exact-input adapters, and immutable artifact capture.
Exact-constraint support recovery and intentionally incomplete programs also need explicit validation.
Then connect the sampler to the existing candidate replay API and compare held-out predictions across all five domains.
Planning and execution integration remain gated on those results.

## September 12 validation

Compute job `22625595` passed the frozen numerical, recording, legacy/replay, type, lint, and formatting checks.
Independent job `22625659` evaluated three references on eight additional seeds and two particle budgets.
All 24 comparisons at 1,024 particles passed the predeclared development tolerances.
One of 24 at 256 particles failed the uninformed-parameter mean tolerance.
These repetitions assess the fixed reference distributions; they are not repeated-dataset calibration experiments.

Five-domain recording audit `22625681` passed reset/action consistency and exact-constraint checks on frozen historical training levels.
The next physical-prediction preflight explicitly distinguishes replay/setup failures, nominal model mismatch, and statistical inference.
No inference failure is automatically converted into extra sensor noise, dropped observations, or live agent actions.
See [the experiment record](experiments-20260912.md).
