# Uncertainty simplification experiments, September 12

The production MB agent still uses the incumbent estimator.
These experiments validate the offline replacement in stages; they are not new agent solve-rate seeds.
The separately resumed MF comparison keeps its historical frozen runtime and original checkpoints.

## Frozen implementation checks

Job `22625595` completed successfully on `mit_preemptable`.
The snapshot reconstructs commit `6179fe1e7` plus a captured patch and content-hashed overlays.
It passed 39 functional tests, dependency-following mypy over 16 files, 16 configured lint checks, and pinned formatter checks.
The earlier successful 79-test foundation checks remain separate evidence and are not added to this count as unique tests.
These are scoped checks, not full-repository CI.

Source identity and reports: [validation bundle](../../logs/uncertainty_stage_a_20260912/README.md), [job status](../../logs/uncertainty_stage_a_20260912/job-22625595/status.json).

## Independent numerical references

Job `22625659` completed 48 reference runs.
The scenarios, seeds 100 through 107, budgets, and tolerances were frozen before launch in [the experiment plan](../../logs/uncertainty_reference_sweep_20260912/plan.json) and its hashed script.
Both budgets use 32 temperatures and four Metropolis moves per temperature.
The likelihood-evaluation cap is `129 * particles`.
The smaller budget is a stress test; advancement requires every larger-budget reference run to pass.

| Reference distribution | 256 particles | 1,024 particles |
| --- | ---: | ---: |
| Stationary Gaussian | 8/8 passed | 8/8 passed |
| Correlated initial position and velocity, with an uninformed parameter | 7/8 passed | 8/8 passed |
| Symmetric two-mode posterior | 8/8 passed | 8/8 passed |
| Total | 23/24 passed | 24/24 passed |

The 256-particle correlated-reference run at seed 102 returned an uninformed-parameter mean of 0.143, outside the declared absolute tolerance of 0.12.
Its informed means and covariance met their criteria.
That failure remains in the results; the tolerance was not relaxed and the seed was not replaced.
Passing the larger-budget runs supports this sampler on the tested reference distributions, but does not establish calibration over repeated datasets, adequate physical initial-state support, or robustness to incomplete simulators.

Each run preserves its weighted samples, numerical diagnostics, reference moments, observed errors, and elapsed time.
Results: [all reference summaries](../../logs/uncertainty_reference_sweep_20260912/job-22625659/summary.json).

## Real recording integrity

The corrected audit, job `22625932`, validated the first training level from historical MB seed 0 in each domain.
Initial job `22625681` checked file integrity but incorrectly treated stored states as agent observations; its statistical data projection is superseded.
These recordings are explicitly designated development data.
No test level was loaded.
The audit checks every primitive action against its reset log, retains public joint observations, and stores the original file bytes in content-addressed bundles.
Continual `episodes.pkl` stores sanitized simulator truth for replay.
The corrected reader regenerates the exact observation channel with run seed, level index, reset episode, and primitive step before building the inference ledger.
It includes those coordinates and the noise implementation source in the provenance bundle.
Already-noisy agent exports must not be passed through this reader a second time.
It verifies that an exact-output contradiction receives zero likelihood.
It does not simulate, fit a parameter, or assess the learned model's accuracy.

| Domain | Recorded actions | Observed scalar fields | Exact fields | Result |
| --- | ---: | ---: | ---: | --- |
| Bridge | 1,186 | 107 | 67 | Passed |
| Fan | 132 | 108 | 49 | Passed |
| Domino | 161 | 70 | 40 | Passed |
| Boil | 264 | 46 | 24 | Passed |
| Original balloons | 235 | 58 | 32 | Passed |

Body velocities and command-weld metadata are explicitly excluded from this feature-likelihood audit.
They are not used as privileged initial-state measurements.
Physical candidate velocities, attachment consistency, and missing model memory still need valid priors or reconstruction.
The runtime report records installed distribution versions and loaded-module hashes; it does not certify complete simulator asset or dependency capture.

Results: [recording report](../../logs/uncertainty_recording_audit_v2_20260912/job-22625932/report.json), [selection and projection policy](../../logs/uncertainty_recording_audit_20260912/plan.json).

## Fixed-program prediction preflight

The next diagnostic holds the early and late training-program snapshots fixed and replays nominal predictions from the beginning and midpoint of each selected training recording.
Each window contains up to 64 recorded actions and is replayed twice in fresh worlds.
Declared initial parameter values are used without fitting.
The initialization intentionally measures the legacy replay's observed-state, zero-velocity assumption; it is not a posterior candidate with a validated physical-state prior.
The report separates exact-output contradictions, noise-standardized feature error, repeated-replay differences, and setup failures.
A nominal candidate's zero likelihood is not proof that every possible initial state and parameter has zero support.

Optional balloons `model_params.json` overrides are explicitly absent in an isolated working directory.
This defines a reproducible development candidate; it does not reconstruct historical fitted parameter deployment.
Historical sidecar provenance must be resolved before any comparison claims that it reproduces the deployed model.

Initial job `22625747` failed before simulation because a CLI `log` option was passed to `reset_config`, whose attribute is `log_file`.
Job `22625774` corrected that alias but exposed further setup problems: prediction states needed public-field sanitization, and the current branch lacks the historical balloons scene controls.
Job `22625841` therefore restored historical runtime `b09217bb3` plus isolated offline modules and used sanitized predicted outputs.
Its raw-state comparison exposed the recording-channel bug above, so its statistical prediction results are superseded.
These setup outcomes are not agent failures.

Final corrected job `22625933` uses the historical runtime and reconstructed noisy observations for both initialization and scoring.
It completed 14 executable nominal cases, and repeated replay was identical in every case.
All 14 violated at least one exact output and therefore had zero likelihood under the strict sensor-only model.
The first violations included robot joint errors ranging from approximately 7e-14 to 2.3e-4 radians and balloons speed error of approximately 0.0079 m/s.
Later discrepancies also include substantive changes in observable events and geometry.
Two early-Fan cases did not execute because the frozen program declares a log-scaled `fan_speed` with lower bound zero.
That invalid program is preserved rather than repaired for the comparison.

| Domain | Executable nominal cases | Exact-output contradictions | Repeated replay |
| --- | ---: | ---: | --- |
| Bridge | 2 | 2 | Identical |
| Fan | 2, plus 2 invalid-program cases | 2 | Identical |
| Domino | 4 | 4 | Identical |
| Boil | 2 | 2 | Identical |
| Original balloons | 4 | 4 | Identical |

This finding does not establish that every parameter and feasible initial state has zero support.
It does establish that the current nominal initialization cannot be fed directly to a strict posterior likelihood and expected to work.
A tolerance was not added to the likelihood to hide these failures.
Before posterior fitting, separate missing initial motion and constraint state from incomplete learned dynamics and deterministic replay approximation.

Results: [corrected prediction report](../../logs/uncertainty_prediction_preflight_v4_20260912/job-22625933/report.json), [frozen plan](../../logs/uncertainty_prediction_preflight_v4_20260912/plan.json).

## Recording-channel regression and correction

Compute reproduction `22625874` exercised the actual continual session, oracle actions, recording writer, and offline reader.
It failed because the reader returned the stored true state instead of the exact noisy observation seen by the acting agent.
The reader now requires explicit seed and level coordinates, reconstructs the seeded channel once, and rejects a directory/index mismatch.
Its fixture now writes truth, matching the real recorder.
The actual session test compares reconstructed observations exactly against the agent's first two observed frames.

Validation `22625919` passed all 29 functional tests and mypy, but lint rejected a `type(...)` check.
The implementation now uses `isinstance` and explicitly rejects booleans as seed/index values.
Follow-up validation `22626126` completed successfully: 29 functional tests, mypy, all three lint checks, and pinned formatter checks passed.

## Contact-rich restoration follow-up

Job `22626183` reuses recorded training actions as mechanical stress inputs in a recreated evaluator world.
It compares explicit full-state replay and legacy zero-velocity replay from the same source trajectory, at the beginning and midpoint, with up to 256 actions per domain.
Evaluator state is used only to isolate reconstruction error and never becomes inference data or an agent observation.
This is not an agent experiment or a fitted-model comparison.
The [frozen plan](../../logs/uncertainty_contact_replay_20260912/plan.json) and [completed report](../../logs/uncertainty_contact_replay_20260912/job-22626183/report.json) preserve that distinction.
All five domains completed the mechanical audit.
Fresh replay was repeatable, but did not reproduce the source trajectory exactly.
For the midpoint continuation, the largest non-robot position-coordinate errors were:

| Domain | Explicit motion restoration | Legacy zero-velocity restoration |
| --- | ---: | ---: |
| Bridge | 5.06 mm | 5.06 mm |
| Fan | 0 mm | 0 mm |
| Domino | 2.31 mm | 2.36 mm |
| Boil | 6.41 mm | 6.75 mm |
| Original balloons | 106.17 mm | 86.78 mm |

These are mechanical discrepancies, not noise-adjusted prediction errors or agent costs.
Fan still differed in robot joints despite matching its non-robot positions.
Balloons additionally changed attachment topology, so preserving velocities alone did not repair its continuation.
The previously documented missing constraint frames, engine state, and domain-private state now require targeted diagnosis.
An equality check on captured model memory does not establish that unrepresented evaluator-private memory was restored.
These results prevent treating the current portable state as a validated complete latent-state representation for contact-rich inference.

## Remaining acceptance gates

The full Stage A gate still requires physical initial-state priors with feasible geometry and attachments, exact-output support handling, and complete runtime identity.
Stage B must then compare posterior and legacy fitting on common frozen programs and data, including held-out causal suffix predictions and incomplete programs.
Shadow planning and live agent comparisons follow predictive acceptance.
No historical MF run or current MB run is silently switched to the prototype estimator.
