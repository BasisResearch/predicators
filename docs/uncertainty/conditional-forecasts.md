# Conditional forecasts from the fitted output model

September 12, 2026.
The [simplification plan](simplification-proposal.md) requires prediction checks on future interactions, not only likelihoods of fitted recordings.
`OutputObservationModel.sample_future` now generates observation histories from the same output model used in those likelihoods.
It does not change the acting agent or provide a numerically assessed parameter posterior.

## Information boundary

The caller supplies a complete native prediction history, the observed fitting prefix, and an explicit local random-number generator.
The method has no argument for recorded future observations.
Returned observations begin immediately after the prefix and retain their primitive-step indices.

The physical prediction history must itself be generated without future readings.
For the current deterministic Domino/Fan physical models, replay the candidate's initial state and all recorded actions uninterrupted.
For an explicit stochastic transition extension such as Balloons, condition transitions only within the fitted prefix and draw unconditioned transitions afterward.
Conditioning a future transition on its recorded speed or joints would invalidate the forecast even if output-error sampling used the correct prefix.

Declared external inputs are copied from the future prediction frames and receive no stochastic density.
They must be genuinely given inputs under the fixed sensor schema.
Unknown future outputs cannot be relabeled as inputs to improve predictions.
Missing required predictions or inputs are rejected.

Offline posterior prediction must replay complete joint parameter/initial-state rows with their posterior weights.
The parameter-only projection used for later planning does not retain the initial-state dependence required by this comparison.
The new output sampler supplies conditional observation draws for a given physical history; it neither creates those joint rows nor certifies their weights.

## Same discrepancy law in fitting and forecasting

Each scalar error process is filtered using only the prefix.
The sampler draws one error value at the prefix boundary from its conditional distribution, then propagates the declared AR(1) process through the suffix.
This retains cross-time dependence instead of drawing every forecast marginal independently.
Declared sensor noise is drawn separately at each future reading.
An empty prefix uses the original initial-error distribution, with no extra transition at frame zero.

Coupled Euler outputs draw from the same antipodal four-dimensional Gaussian mixture used by the likelihood, then apply the native Euler readout to the raw quaternion components.
The components are deliberately not normalized; normalization would change the pole probabilities and native yaw branches.
The current sampling path supports the native pole threshold of 0.99999 and explicitly rejects other declared thresholds.
Checked display features are computed from their sampled source readings, preserving the same reduction used during likelihood evaluation.
Unassigned fields retain their original sensor channel, including exact discrete events.

These draws are observations under the declared discrepancy model, not corrected physical simulator states.
They must not be fed back into physical replay as if they were true states.
A zero-likelihood fitting prefix is rejected because it supplies no conditional forecast for that physical history.
Numerical failures remain separate from such an exact contradiction.

## Numerical and native checks

The component tests compare empirical forecast means and full cross-time covariance with a dense Gaussian conditional reference, including exact and noisy sensor channels.
They also check the original initial-error variance, exact events, derived displays, reproducibility, missing inputs, invalid prefixes, and the unnormalized Euler mixture's pole and yaw branches.
The existing likelihood tests remain in the check suite.
Final compute job `22651009` passed all 31 functional tests, two-file mypy and pylint, and pinned formatter checks.
The checked source hashes and outputs are in `logs/uncertainty_forecast_checks_v3_20260912`.
The preceding attempts corrected an invalid test fixture that declared a noisy conditioned input and two test-style lint findings; the forecast implementation was unchanged across those attempts.

Native compute job `22650998` completed a Domino integration check using the two fixed candidates from the original sampler preflight.
Both candidates have feasible initial geometry and each complete 161-action history repeats exactly in a fresh world, totaling 644 native steps.
The first candidate retains its zero-likelihood fitting prefix and the forecast API rejects it.
The second reproduces the original 64-action prefix log likelihood `8141.576094195281` exactly.

For that supported candidate, 32 draws each contain the full 97-action suffix with all 70 output readings per frame.
The physical trajectories were generated before consulting future observations, and output errors used only the initial frame and first 64 actions.
Repeated draws with the same seed match exactly; changing the seed changes the draws.
Four sampled complete histories were checked against the full likelihood and all had finite scores.
The native-prefix/suffix histories and all 32 observation draws are retained in the hashed forecast artifact.

The candidate's moving-object Cartesian RMSE on 1,746 future scalar readings is 0.05555 m for native predictions and 0.05426 m for the sampled observation mean.
These are descriptive diagnostics for one selected support witness, not a comparison of estimators, calibrated coverage, or evidence of improved agent performance.
Neither candidate is an assessed posterior sample.
The report and frozen runtime are in `logs/uncertainty_domino_forecast_preflight_20260912`.

This establishes a tested route from a fixed supported physical history to complete conditional observation forecasts.
Numerically adequate joint posterior rows, matched legacy comparisons and the live-agent gates remain required.
