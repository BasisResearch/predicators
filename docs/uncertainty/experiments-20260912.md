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

Job `22625681` validated the first training level from historical MB seed 0 in each domain.
These recordings are explicitly designated development data.
No test level was loaded.
The audit checks every primitive action against its reset log, retains public joint observations, and stores the original file bytes in content-addressed bundles.
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

Results: [recording report](../../logs/uncertainty_recording_audit_20260912/job-22625681/report.json), [selection and projection policy](../../logs/uncertainty_recording_audit_20260912/plan.json).

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
Corrected job `22625774` translates that alias and preserves all physical settings and candidates.
These setup outcomes are not model or agent failures.

Plan and eventual reports: [corrected prediction bundle](../../logs/uncertainty_prediction_preflight_v2_20260912/plan.json).

## Remaining acceptance gates

The full Stage A gate still requires physical initial-state priors with feasible geometry and attachments, exact-output support handling, and complete runtime identity.
Stage B must then compare posterior and legacy fitting on common frozen programs and data, including held-out causal suffix predictions and incomplete programs.
Shadow planning and live agent comparisons follow predictive acceptance.
No historical MF run or current MB run is silently switched to the prototype estimator.
