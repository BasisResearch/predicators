# Balloons: forecasts after a prefix-only fit

September 13, 2026.
This continues the [64-action fitting experiment](balloons-prefix-inference.md) toward the offline comparison required by the [simplification proposal](simplification-proposal.md).
The incumbent agent remains unchanged.

## Data boundary and scope

The fitted prefix contains 64 actions and 65 observations; the reserved suffix contains 171 actions.
The new fixture uses the candidate selected by the prefix-only support audit, rather than a center selected from the complete recording.
Its simulator program was historically selected using development training data, so this is a conditional forecast check for that fixed program, not an independent test of program synthesis.
Neither the fixture nor its numerical random seeds count as agent solve-rate runs.

The implementation is frozen under `logs/uncertainty_balloons_prefix_forecast_20260913/`.
Its manifest identifies the fitting worker, prefix guide and verification reports, action recording, and observation/transition modules.
The forecast snapshot adds the already implemented future-generation and output-likelihood methods without changing the live fitting snapshots.
Exact reproduction of the saved prefix candidate and its complete likelihood is an execution gate for the fixture.

## Two different computations

### Generate predictions

A fresh simulator reconstructs the candidate initial state and executes all 235 actions continuously.
The first 64 actions reproduce the fitted conditional history, including the sampled velocity directions and exact joint/speed observations.
For the remaining actions, controlled joint positions receive draws from the declared Gaussian transition discrepancy, and box velocity follows the original rest/Gaussian mixture.
The generator has no lookup entries for future observations.
Native attachment changes, balloon bursts and simulator memory continue through the whole history.
It does not restore an arbitrary observation-boundary snapshot.

The output model conditions its correlated error process on prefix observations, then draws all 58 future observation fields.
The original sensor variances remain fixed.
Initial-observation evidence and the prefix speed factors are not counted a second time as output evidence.
The artifact records native band membership, rest, burst, the `InBand` predicate, and the evaluator's no-burst goal condition separately.
These are predictions along the supplied action sequence, not an executed agent outcome.

### Evaluate future density

A separate computation receives the observed suffix explicitly.
It conditions each future transition on the observed controlled joint positions and box speed, retains their Gaussian and radial density factors, and samples the unobserved velocity directions from the conditional direction law.
The complete output likelihood retains the other exact observations, including discrete events, while integrating the declared correlated output error.
Each such history supplies an importance weight for integration over future velocity directions.

The fixture averages density contributions over eight complete conditional-history draws, including zero-support histories in the denominator.
It also reports two four-draw estimates, importance effective sample size and maximum normalized weight.
This is a finite Monte Carlo estimate, not an automatically adequate marginal likelihood.
A single path's score is never reported as the integrated future likelihood.
Zero contributions do not justify declaring the model's full conditional support empty.
Density-evaluation histories never enter generated forecast means or change the prefix fitting weights.

## Validation and acceptance

The fixture checks four generated histories and eight conditional-density histories, with selected full-history repetitions.
An independent reader checks the serialized transitions, random draws, exact readbacks, native goal semantics, output composition, and the density denominator.
It also compares radial densities against a separate noncentral-chi calculation and deliberately corrupts saved fields to check rejection.
The reader repeats one complete native history of each kind in a separate process.
A separate initialization audit holds the physical candidate fixed while changing robot numeric readouts that should be overwritten during initialization.

These checks establish forecast mechanics only.
The completed prefix populations must still be evaluated without changing their weights, assessed for numerical stability and compared with the incumbent fitter's predictions.
The physical-support and predictive-adequacy gates remain open until their own evidence is available.

## Completed native fixture

Job `22692309` completed on node1390 in 53 seconds, executing 3,824 native actions including repeated histories and the prefix reference.
All twelve histories reproduce the saved 64-action prefix exactly under the forecast snapshot.
The four generated suffixes each contain 171 observations with all 58 fields.

Independent verifier `22692420` completed in 36 seconds.
It checked 25,380 joint factors or unconditional draws, 2,136 radial-density evaluations against a separate noncentral-chi calculation, all recorded goal fields, and exact replay of one complete generated and one complete conditional-density history.
It rejected eight corrupted artifacts and verified that zero-density paths stay in the integration denominator.
The source report is `fixture-22692309.json`; the independent certificate is `verify-22692420.json` in the experiment bundle.

All eight conditional-density draws for this preliminary prefix candidate have zero likelihood on the actual reserved suffix.
Each predicts `balloon0.popped = 1` while the recorded value remains zero, with the first event disagreement between actions 97 and 114.
This is an observed failure of these eight candidate histories, not proof that no compatible history exists and not a result for either running prefix posterior population.
Small raw robot-pose differences in the diagnostic mismatch list are handled by the declared output discrepancy; they should not be confused with the uncompromised burst-event contradictions.

The first initialization audit, job `22692482`, stopped when its deliberately zeroed robot orientation violated the existing reconstruction guard.
That was a diagnostic setup failure, not an agent failure.
The revised audit `22692596` completed in 27 seconds and reports rejected initialization separately from changed physical trajectories.
All five small perturbations of robot position, orientation or finger readout leave the complete 64-action trajectory exactly unchanged, including repeated reconstruction.
The deliberately inconsistent zero orientation is rejected reproducibly.
This rules out dependence on those five tested readout perturbations for this candidate; it does not establish unrestricted initialization invariance.

Two additional finite-density checks completed as job `22692681`.
They retain the recorded future joint/speed inputs and conditional velocity draws while replacing the other future outputs with explicit synthetic draws on those same physical histories.
The physical trajectories and all transition factors remain exactly unchanged, and the independent reader verifies the finite full-history-minus-prefix output factors and their composition with transition densities.
This adds 4,230 joint-factor checks and 470 independent radial-density checks.
These synthetic cases test the finite-density calculation; they are not held-out evidence or unconditional physical forecasts.
The first attempt, `22692619`, failed during imports because the fitting directory shadowed the forecast verifier's module name; the corrected attempt loads the identified verifier by its explicit file path.

The fixture, independent reader, initialization audit and finite-density certificate are indexed by `verification-manifest.json` in the experiment bundle.
The full plan remains at Stage B, with Stage A physical-support and numerical gates still open.
