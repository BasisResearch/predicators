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

## In-process checkpoint diagnostic

Job `22626292` tested the balloons midpoint continuation with PyBullet `saveState` plus a deep copy of the environment's Python-side state.
It completed without a setup exception but did not reproduce the source trajectory.
The two restore attempts had maximum non-robot position differences of 703.07 mm and 406.31 mm, compared with 106.17 mm for portable replay.
The recorded attachment names matched in these in-process attempts, but that does not certify that engine constraint frames or identifiers were restored correctly.
This is a failed checkpoint method, not evidence that an arbitrarily large discrepancy variance should enter the likelihood.
The next restoration diagnosis should compare actual engine constraint frames and lifecycle state, not only the portable attachment-name list.

Report: [checkpoint diagnostic](../../logs/uncertainty_checkpoint_diagnostic_20260912/job-22626292/report.json).

## Remaining acceptance gates

The full Stage A gate still requires physical initial-state priors with feasible geometry and attachments, exact-output support handling, and complete runtime identity.
Stage B must then compare posterior and legacy fitting on common frozen programs and data, including held-out causal suffix predictions and incomplete programs.
Shadow planning and live agent comparisons follow predictive acceptance.
No historical MF run or current MB run is silently switched to the prototype estimator.

## Resolving physical continuation failures

Constraint diagnostic `22626945` reproduced the original balloons error using the same training-action continuation.
At the midpoint, two attachment commands and two lift forces were queued for the next action.
The old offline snapshot dropped that queue, causing the next step to remove existing welds and omit the lift.
Preserving the queue reduced the largest non-robot position-coordinate discrepancy from 106.17 mm to 4.00 mm.

The public balloons feature vector contains positions but not box or balloon orientations.
A physical continuation nevertheless needs those orientations and the original weld frames, rather than frames recomputed from the deflected current poses.
Follow-up `22626966` restored all three together: pending commands, complete body poses, and original command-weld frames.
Its maximum non-robot position-coordinate discrepancy was 9.73e-14 m over the 118-action continuation.
The attachment sequence matched.
Robot joint differences remained as large as 1.97e-4 radians, so this is not an exact complete-state checkpoint.

The same diagnostic identified a separate error in the previous in-process checkpoint experiment.
PyBullet restore left the third balloon's later-created constraint in the world when rewinding to a boundary that had only two constraints.
A second restore accumulated another extra constraint.
The corrected diagnostic removes constraints created after the saved boundary and verifies that every original constraint still exists before restoring.
This reduced both checkpoint attempts' non-robot position error to zero; robot joint error remained 7.52e-7 radians.
This narrow diagnostic does not implement general restoration after an original constraint has been removed or modified.
The old failed attempts remain recorded and must not be interpreted as simulator stochasticity or sensor variance.

The offline replay implementation now represents full physical body poses, next-step commands, and original command-weld frames explicitly.
These are candidate quantities or evaluator-only diagnostic quantities, never additional agent observations.
Unknown command targets, incomplete physical poses, and missing weld frames are rejected.
The production fitter, observation channel, and acting MB/MF runtime are unchanged.

A new `prefix` argument reconstructs a candidate's action history in the same fresh world before producing a requested continuation.
It retains accumulated engine state, native attachments, and model memory without a mid-trajectory restore.
Every call pays for the full prefix; a parameter change must replay that prefix under the changed parameters.
The initial candidate still needs a valid physical prior and canonical initialization.
Exact agreement with an uninterrupted candidate is a different claim from agreement with the historical evaluator or a learned program's predictive accuracy.

Reports: [constraint lifecycle](../../logs/uncertainty_constraint_diagnostic_20260912/job-22626945/report.json), [full poses and checkpoint cleanup](../../logs/uncertainty_constraint_diagnostic_v2_20260912/job-22626966/report.json).

Corrected five-domain contact audit `22627021` uses the same recorded-action stress inputs as the earlier audit.
Its `prefix` continuations exactly match uninterrupted portable-root candidate rollouts in all five domains at both tested boundaries, including every observed feature, all robot joint positions and velocities, body velocities, attachment sequences, and captured model memory.
The ten comparisons span up to 256 actions per domain, with 1,040 source actions total.
Fresh portable replays also remain exactly repeatable.
This establishes candidate continuation consistency, not equality with the source evaluator initialized through its original reset lifecycle.
Residual midpoint restoration errors relative to that source remain:

| Domain | Largest non-robot position-coordinate error | Largest robot joint-position error |
| --- | ---: | ---: |
| Bridge | 4.51 mm | 0.00975 rad |
| Fan | 0 mm | 0.000651 rad |
| Domino | 3.40 mm | 0.00298 rad |
| Boil | 6.75 mm | 0.00968 rad |
| Original balloons | 9.73e-11 mm | 0.000197 rad |

Root diagnostic `22627069` compares a repeated evaluator reset with physical-state reconstruction.
Bridge, Fan, Boil, and Balloons repeated their source reset trajectories exactly.
Domino did not; that reset-protocol inconsistency must be kept separate from the bit-identical portable-candidate prefix comparisons.
The inspected base-body poses, velocities, dynamics settings, and engine settings matched between fresh reset and physical restore, except one balloons quaternion component differing below 1e-42.
This inspection does not include contact solver caches or certify complete robot controller state.

The explicit `replay_initialized_candidate` API permits a declared initialization protocol to run before the prefix.
It does not select evaluator tasks by default.
This supports evaluator-only reset references and, separately, a future generative candidate initialization based on the declared prior.
Its initializer, inputs, and runtime must be included in artifact identity.
The initializer and every prefix action run under the requested candidate parameters, and initializer failures release the temporary world.

Reports: [corrected five-domain contact replay](../../logs/uncertainty_contact_replay_v2_20260912/job-22627021/report.json), [root protocol diagnostic](../../logs/uncertainty_root_diagnostic_20260912/job-22627069/report.json).

Explicit-initializer audit `22627125` reproduced Bridge, Fan, Boil, and Balloons exactly from their evaluator reset protocols through both tested boundaries.
Domino still differed even when the diagnostic replaced time-limited IK with a fixed attempt bound, so that hypothesis did not resolve its reset inconsistency.
Inspection found that its disk task cache retains feature values but discards the exact initial robot joint configuration, then invokes inverse kinematics again on load.
The cache boundary is being tested independently; the evaluator-only initializer must not silently mix generated and reconstructed task roots.
The diagnostic IK setting is not a change to an acting experiment.

Focused implementation validation `22627013` passed 24 functional tests and mypy but identified two test-only lint errors.
Those errors were corrected and validation `22627081` passed all 14 replay tests, type checking, lint, and pinned formatting.
After adding explicit initialization and parameter/history ownership tests, final validation `22627126` passed 26 focused functional tests, mypy for both changed files, both lint checks, and pinned formatting.
This is focused validation, not the full repository CI required before a PR.

Report: [explicit initialization audit](../../logs/uncertainty_initialized_replay_20260912/job-22627125/report.json).

## Domino task-cache reconstruction

End-to-end reproduction `22627163` generated a real Domino task, moved the physical robot to a valid alternative configuration, saved the task, and loaded it into a fresh simulator through the production cache reader.
The loader changed the first joint from -0.1533237473 to 0.4424211360 radians because its feature-only record forced another IK solution.
The missing joint configuration explains why matching object features is insufficient for a faithful task-cache round trip.
Changing the IK time limit had not addressed this information loss.

The cache writer now includes portable simulator metadata, including exact initial joints, and the reader supplies that metadata to the existing task-restoration path.
Goal semantics and task sampling rules are unchanged.
Old feature-only cache files remain readable through the legacy reconstruction path.
The existing source digest changes the cache key for future runs, so newly generated caches preserve the complete stored robot configuration.
The frozen MF sweep checkout and its running jobs are not modified by this fix.
This harness correction is separate from the offline replay implementation and does not change MB prompts or its uncertainty estimator.

Cache-fix audit `22627211` removed Domino's multi-radian initial-configuration error but retained a smaller contact discrepancy.
The remaining diagnostic mismatch came from comparing different initialization lifecycles: the source performed task-generation simulations in its execution world, while subsequent worlds loaded the cached tasks.
`ContinualRun._begin_level` already uses a fresh execution world sharing previously generated tasks when `test_fresh_env_per_episode` is enabled, as it is in this sweep.
The faithful mechanical reference must use that lifecycle for both source and replay.

Final audit `22627245` matches the continual lifecycle: generate tasks in a template, create fresh instances sharing those tasks, reset, and execute the recorded training actions.
The source and candidate use identical historical configuration values, with the task-cache correction and offline replay modules recorded as overlays.
All ten comparisons across five domains passed exact equality of the measured features, every robot joint position and velocity, body velocities, attachment sequences, and captured model memory.
No extra sensor noise or likelihood tolerance was introduced.

| Domain | Source actions | Continuation boundaries | Largest measured feature error | Largest robot joint-position error |
| --- | ---: | --- | ---: | ---: |
| Bridge | 256 | 0, 128 | 0 | 0 |
| Fan | 132 | 0, 66 | 0 | 0 |
| Domino | 161 | 0, 80 | 0 | 0 |
| Boil | 256 | 0, 128 | 0 | 0 |
| Original balloons | 235 | 0, 117 | 0 | 0 |

This establishes a validated reconstruction path from explicit initialization plus action history on the tested development prefixes.
It does not certify arbitrary portable checkpoints, every possible trajectory, historical solve-rate replication, or posterior quality.
The replay obstacle can be bypassed without changing the current MB estimator: use the declared initializer and reconstruct prefixes, paying their simulator cost.
Physical initial-state priors, exact-observation support, runtime closure, and posterior-versus-legacy predictive comparisons remain outstanding.

Reports: [cache fix with mismatched lifecycle](../../logs/uncertainty_initialized_replay_v2_20260912/job-22627211/report.json), [continual-lifecycle replay validation](../../logs/uncertainty_continual_replay_20260912/job-22627245/report.json).

Cache validation `22627201` passed both end-to-end cases but exposed a type annotation mismatch after extracting the existing float64 state dictionary into a local variable.
The annotation was corrected without changing numeric precision.
Validation `22627228` then passed both cases and type checking but flagged an overlong test import.
Final validation `22627254` passed both end-to-end cases, mypy, both lint checks, and pinned formatting after that import was shortened.
The two cases cover exact joint preservation through fresh-world actions and compatibility with older feature-only cache files.
Together with the 26 focused replay/legacy tests, these are 28 distinct passing functional tests across the two implementation chunks.

## Exact affine conditioning reference

Job `22627437` tested a conditional-coordinate construction before any physical posterior proposal was introduced.
For the declared equation `y = A(u) z + b(u)`, a square nonsingular `A(u)` determines the eliminated coordinates `z`.
The implementation retains the original joint box prior and the density factor `1 / abs(det(A(u)))` relative to the free-coordinate proposal.
It distinguishes a singular unsupported chart, a particular solution outside the original prior, and a numerical linear-algebra failure.
The returned residual and backward-error bound describe floating-point solution accuracy; they are not an observation-noise floor or an epsilon-band likelihood.
The bound is not a guarantee of small forward error in an ill-conditioned system or adequate posterior approximation.

The reference equation is `observed = theta * start`, with independent original uniforms `theta ~ U(1,2)`, `start ~ U(0,1)`, and an unused coordinate `U(-1,1)`.
The exact observation is `0.5`.
Independent continuous draws miss this equality, while simply setting `start = 0.5 / theta` leaves an incorrectly uniform parameter marginal.
The correct conditional parameter density is proportional to `1 / theta`, giving mean `1 / log(2)`, approximately 1.442695, rather than 1.5.
This explicit change of variables is a restricted reference; the [smooth constrained-inference literature](https://proceedings.mlr.press/v54/graham17a.html) does not establish a solver for this repository's contact dynamics.

An importance-sampling audit used eight independent seeds at each of two predeclared budgets.
Acceptance required absolute parameter-mean error at most 0.04, unused-coordinate mean error at most 0.08, and maximum error at 41 predeclared parameter-CDF checkpoints at most 0.04.

| Particles | Passing seeds | Largest mean error | Largest unused-coordinate mean error | Largest checked CDF error |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 5/8 | 0.031334 | 0.042561 | 0.072825 |
| 8,192 | 8/8 | 0.002930 | 0.009718 | 0.009938 |

The largest affine residual was 5.55e-17 at either budget.
The three smaller-budget failures remain in the report; no seeds were rerun or excluded to obtain a passing aggregate.
This is a numerical importance-sampling reference, not SMC deployment, a physical-domain fit, or an agent solve-rate experiment.
The existing offline SMC tests also ran unchanged.
All 15 functional tests, two-file mypy and lint checks, and pinned formatting checks passed.
The new cases cover initial-coordinate conditioning, parameter-dependent density corrections, coordinate ordering, complete elimination, output-unit changes, offsets, and support/error distinctions.

Artifacts: [plan](../../logs/uncertainty_conditioning_20260912/plan.json), [reference report](../../logs/uncertainty_conditioning_20260912/reference-22627437.json), [test report](../../logs/uncertainty_conditioning_20260912/checks-22627437.xml).

## Five-domain initial-state inventory

The [inventory](initial-state-inventory.md) identifies the public measurements, missing quantities, proposed representation requirements, and unresolved priors separately for all five frozen first training levels.
Job `22627492` reconstructs their seeded public observation ledgers and inspects each visible model without generating evaluator tasks.
All five cases use fixed-base Fetch with 24 URDF joints: nine observed movable joints, four unobserved movable joints, and eleven fixed joints.
The two wheel joints and head pan/tilt are movable but absent from the controlled-joint observation.
Before justified reductions, their positions/velocities plus nine controlled velocities give 17 possible continuous robot-state coordinates per episode root.
Known URDF fixed joints add no physical degrees of freedom.

The first public balloon-box speed is exactly zero.
An inference construction supporting only positive-speed spheres would therefore miss this development root.
A candidate rest component must have declared prior mass and conditioning semantics; a missing velocity field is not a reset guarantee.
The inventory also distinguishes the frozen Bridge and Boil no-op programs from later learned artifacts and identifies optional parameter sidecars and data-narrowed bounds that require prior/runtime provenance.
Final physical dimensions and normalized support remain unresolved rather than being reported as an independent box over noisy feature values.

The first inventory attempt, `22627481`, used a nonexistent public physics-client attribute in the audit script and produced setup errors.
The corrected audit uses the visible model's actual client handle; the failed attempt is preserved and is not an agent outcome.

Artifacts: [inventory plan](../../logs/uncertainty_state_inventory_v2_20260912/plan.json), [verified inventory report](../../logs/uncertainty_state_inventory_v2_20260912/job-22627492.json).
