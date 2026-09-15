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

## Causal future likelihood

`OutputObservationModel.log_future_likelihood` scores a complete future observation history conditional on the supplied fitting prefix and fixed native predictions.
The caller supplies prefix and future readings separately, preserving their original step indices.
Future readings enter only the score calculation; they must not influence the physical prediction history or the fitted particle weights.

The score is the log of the joint conditional density of the future readings, including temporal dependence in scalar output errors.
Scalar factors filter the prefix, then accumulate only future conditional observation factors.
Later future factors condition on preceding future readings through the probability chain rule; this evaluates the joint forecast and does not revise the forecast supplied for evaluation.
Independent sensor and coupled Euler factors contribute only their future terms, and checked readouts preserve their source relationship.
The implementation sums future factors directly instead of subtracting two large full-history log likelihoods.

An impossible fitting prefix raises an unsupported-conditioning error because it defines no conditional forecast for that physical candidate.
An impossible future returns negative infinity, retaining the prediction failure.
An empty future has log likelihood zero when the prefix is supported; an empty prefix scores the original unconditional history.
Missing scalar readings still advance the error process by their recorded primitive steps.

For a weighted posterior forecast, the complete-history densities must be mixed using the prefix-fitted joint particle weights.
Averaging log densities or independently mixing each time step would evaluate a different distribution.
This component scorer does not construct or assess that posterior mixture; `JointForecast`, below, supplies the deterministic physical-history mixture.
For a stochastic physical extension, it also does not replace integration over future physical transitions under the declared transition law.

These density scores are diagnostics under the declared output model, not replacements for the common feature, event and action metrics in the legacy comparison.
The incumbent robust fitting objective is not a normalized predictive likelihood and must not be compared numerically with this log density.
Density comparisons across changed observation laws additionally require a common observation representation and reference measure.

## Forecasts from the assessed joint posterior

`JointForecast.replay` connects an assessed joint posterior to the output forecast and likelihood methods.
It requires the exact fitting ledger and output-model identity from that posterior and a fitted reset episode whose history will be continued.
The requested future actions are separate from the fitting observations.
Data, model, episode and action validation occurs before invoking replay.
The parameter-summary consumer and this forecast path share `validated_posterior`, which rechecks identity, numerical protocol and sample structure without suppressing predictive failures.

Replay receives each complete positive-weight row, including all uncertain episode-state coordinates, as an owned dictionary.
It also receives the fitted reset episode and the future actions; there is no future-observation argument.
The caller remains responsible for implementing the frozen physical model, reconstructing that candidate's initial state and memory, replaying the complete prefix and disposing each world.
Interface checks cannot prove that an arbitrary callback implements the declared physics.

One resulting history corresponds to each positive-weight source particle, in its original order.
Zero-weight particles are not simulated.
An exception aborts construction, and missing histories or an unsupported fitted prefix raise errors instead of deleting particles and renormalizing the remainder.
Predictive diagnostics, including failures, remain attached through the original assessment.
The fixed output model is identical across these components; inferred output-model hyperparameters need an explicit extension to this contract.

The forecast scores a future history using a stable log-sum-exp of its complete conditional densities and the original particle weights.
An impossible future under every represented particle retains zero probability.
Observation draws select one complete source particle for an entire history, then draw the output errors under that history's supported prefix.
Returned source indices retain provenance, and explicit seeds make the draws reproducible.
Parameter and initial-state coordinates are never sampled from separate marginals, and particles are not switched between time steps.

This adapter currently covers deterministic physical continuation under joint parameter/initial-state rows.
It does not integrate unobserved future physical transitions for the Balloons stochastic extension.
Passing one stochastic rollout per particle through this adapter would omit that additional integration and must not be presented as the complete forecast law.
The separate [conditional-path integration diagnostic](stochastic-future-integration.md) implements the required density accounting but currently fails its native numerical checks.
The existing execution estimator and all acting-agent behavior remain unchanged.

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

The future-density reference tests additionally compare the scored joint suffix with a dense Gaussian conditional distribution across four persistence values and exact/noisy sensor channels.
They include missing prefix readings, deterministic contradictions, exact event/readout failures, coupled Euler poles, empty prefixes and futures, and a large-prefix cancellation case.
All 43 functional tests, including the existing output-model tests, pass on a compute node.
Job `22651367` also passed focused type checking; its sole lint failure was an overlong docstring.
After pinned formatting, job `22651945` passed two-file pylint, isort, yapf and docformatter checks.
The formatted source has an identical executable syntax tree to the functionally tested source, with documentation strings excluded from that comparison.
Source hashes and the parity record are retained in `logs/uncertainty_future_score_format_checks_20260912`.

Compute job `22651837` evaluates the archived native Domino history and all 32 archived forecast draws with both the previous and new likelihood implementations.
All 33 complete-history scores match exactly, and the original prefix log likelihood remains exactly `8141.576094195281`.
Every generated future has a finite conditional score, extending the earlier check of only four generated histories to all 32.
The recorded 97-action future also has finite conditional log likelihood `10459.790527256999` for this candidate.
These values use the model's declared mixed observation representation and are not success probabilities or a calibrated model-selection threshold.

This check reuses the hashed native predictions generated on `node1412`; it performs zero physical steps and no initial-state transforms.
Likelihood scoring ran on an Intel Xeon Gold 6230 host (`node1376`), with exact compatibility checked there against the old implementation and archived prefix score.
The originally submitted node-specific scoring job `22651798` was cancelled while pending because that node was fully allocated.
Reports and frozen sources are in `logs/uncertainty_domino_future_scores_v2_20260912`.

Compute job `22652185` passed 47 functional tests and focused four-file mypy, pylint, isort, yapf and docformatter checks for the joint forecast adapter and shared assessment validation.
The tests include an enumerated anticorrelated joint measure, unequal particle masses, impossible mixed event sequences, stable mixture densities, reproducible whole-history draws, missing observations, provenance rejection, and failed-replay handling.
The existing parameter-projection and assessment tests also pass after sharing their validation logic.
Artifacts are in `logs/uncertainty_joint_forecast_checks_20260912`.

### End-to-end numerical reference

Compute job `22652221` completed four independent fits of a noisy constant-velocity model with uncertain initial position.
Each fit uses the same original uniform box prior, two prefix readings with sensor standard deviation 0.3, and no future observations in fitting or replay construction.
The two inferred coordinates are strongly anticorrelated: the independent Gaussian posterior reference has mean `(0.2, 0.3)` and covariance `[[0.09, -0.09], [-0.09, 0.18]]`.
The probability mass excluded by the box bounds is at most `8.03e-29`, bounding the reference's truncation approximation.

The worker passes sampled joint rows through assessment, deterministic replay, weighted future scoring and 5,000 observation draws per fit.
The analytic two-step future has mean `(0.8, 1.1)` and covariance `[[0.54, 0.72], [0.72, 1.26]]`, including the original sensor noise.
The full off-diagonal covariance matters: independently drawing initial position and velocity, or switching particles between future steps, would change this reference.

| Particles | Numerical seed | Maximum parameter mean error | Future joint log-density error | All declared reference checks |
| --- | --- | ---: | ---: | --- |
| 256 | 62 | 0.01003 | 0.11927 | Fail |
| 256 | 63 | 0.01637 | 0.09974 | Pass |
| 2,048 | 62 | 0.00316 | 0.02685 | Pass |
| 2,048 | 63 | 0.00350 | 0.00402 | Pass |

The criteria were frozen before running: parameter mean error below 0.06, parameter covariance error below 0.03, future log-density error below 0.1, sampled future mean error below 0.1 and sampled future covariance error below 0.12.
The preliminary assessment used to exercise the forecast adapter checks parameter moments only, and all four fits pass that preliminary check.
The complete reference also checks future predictions; the 256-particle seed 62 fails its density criterion and is retained as a failed numerical reference.
This discrepancy is sampling approximation error in a known model, not evidence of missing physical dynamics.
It demonstrates why the real-domain numerical protocol needs prediction stability as well as parameter summaries.
The larger budget passes 2/2 numerical seeds, and the smaller passes 1/2; these are synthetic reference trials, not task or agent seeds, calibration across domains, or a production acceptance result.
The frozen plan, output, and explicit reference assessment are in `logs/uncertainty_joint_forecast_reference_20260912`.
