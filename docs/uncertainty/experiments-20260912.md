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

## Explicit velocity prior and exact rest

The initial balloon box has exact observed speed zero, so an unconstrained continuous velocity proposal or a positive-speed sphere alone misses the required construction.
`RestOrGaussianVelocityPrior` now defines a normalized mixture: mass `rho` at velocity zero, otherwise a three-dimensional isotropic Gaussian with per-axis standard deviation `sigma`.
These are explicit modeling assumptions, not hidden evaluator values or reset guarantees.
A full initial-state prior still has to specify their values and dependencies on scene geometry and attachments.

At speed zero, the conditional velocity is exactly zero and the observation contributes mass `rho`.
At positive speed `r`, two uniform coordinates generate an isotropic direction, while the speed observation contributes `(1-rho) * sqrt(2/pi) * r^2 / sigma^3 * exp(-r^2/(2*sigma^2))`.
Both factors use the declared measure consisting of a point mass at zero plus Lebesgue measure on positive speeds.
The speed-squared factor is required; dropping it or treating the direction proposal as evidence would change the model.
When `rho=0`, conditioning on the zero-density boundary is explicitly unsupported rather than assigned an arbitrary posterior.
When `rho=1`, a positive speed is outside the prior's support.
An unrepresentable log density raises a numerical error instead of masquerading as exact zero support.

Compute job `22628133` passed 19 functional tests, two-file mypy and lint, and pinned formatting.
The four new tests check unit total mass, isotropic positive-speed directions, retained information about rest probability, and the distinction between zero speed and an arbitrarily small positive speed.
For a uniform prior on `rho`, one rest observation produces conditional density `2*rho` and mean `2/3`, while an independent uninformed moving-scale parameter retains its prior.
The tests verify that the rest factor is independent of that scale; they do not establish a fitted physical posterior for the balloon domain.

Artifacts: [plan](../../logs/uncertainty_velocity_prior_20260912/plan.json), [validation report](../../logs/uncertainty_velocity_prior_20260912/checks-22628133.xml).

## External runtime inputs affect physical predictions

The frozen balloon model reads `./model_params.json` and gives its entries precedence over `agent_param`.
Mechanical reproduction `22628126` held program bytes, declared candidate parameters, initial recorded state, and all 235 actions fixed while changing only that optional file.
Replacing the four lift coefficients through the sidecar changed predicted box height by up to 0.496114686 m.
The override values were within the declared coefficient ranges.
This is a runtime-dependency reproduction, not an agent seed, physical-prior fit, or model-quality comparison.
It demonstrates why program and parameter hashes alone do not identify a simulator.

The new offline `RuntimeInputs` snapshot records present file bytes, explicitly absent optional paths, and a complete child environment.
It materializes a fresh worker directory and verifies its declared inputs before a result is accepted.
Source edits cannot change previously captured bytes; added, removed, changed, or symlinked worker inputs invalidate verification.
Its artifact identity includes absence and environment settings, so a sidecar-free hypothesis cannot collide with the sidecar-backed model.
Callers must use separate worker processes with that complete environment, rather than temporarily changing the working directory or environment in concurrent sampler threads.
The production model loader and acting agent were not changed.

Fresh-process replay validation `22628194` ran each file condition twice under this contract.

| Frozen file condition | Declared candidate parameters | Largest repeated feature difference | Runtime-input identity prefix |
| --- | --- | ---: | --- |
| `model_params.json` absent | Same in both conditions | 0 | `f0788c84e53d` |
| Override file present | Same in both conditions | 0 | `1c592739083a` |

The two conditions still differ by 0.496114686 m in predicted box height and now have distinct runtime-input identities.
The identity prefixes describe this audit's explicit environment and inputs; they are not universal program identifiers.
Validation `22628197` passed 13 functional tests, two-file mypy and lint, and pinned formatting.
An earlier validation failed because the test fixture wrote an invalid JSON number; that fixture was corrected in the retained second snapshot.

This closes the explicit working-directory input gap, not all runtime dependencies.
Interpreter binaries, package imports, native libraries, assets, configuration, and invocation still need complete runtime capture.
Post-run file verification is not a filesystem sandbox and does not detect transient writes or arbitrary external reads.
The attempted system-call trace in job `22627944` was rejected by the compute node's `ptrace` policy; the follow-up does not claim native dependency discovery.
That setup failure is preserved separately from the successful replay reproduction.

Artifacts: [reproduction](../../logs/uncertainty_runtime_inputs_v2_20260912/job-22628126.json), [frozen replay validation](../../logs/uncertainty_runtime_replay_20260912/job-22628194.json), [contract validation](../../logs/uncertainty_runtime_contract_v2_20260912/checks-22628197.xml).

## A structural exact-output contradiction in the frozen Bridge model

The final archived Bridge program is byte-identical to the frozen cycle-000 program, SHA256 `4ef259f6c0de775971e7ecd92b1a854743471bafaa28fb5a9376623d96b1f6bf`.
Its residual hook returns `None`, it declares no model memory, and it has no declared residual parameters.
Source inspection establishes an invariant for this program under uninterrupted replay: every `glue_*` attribute retains its initial value.
The generic step advances robot control, rigid-body physics, and grasps; the glue observation reads the stored attribute.
The glue-changing process and its latch calls are in the domain-specific hook that this program replaces with a no-op.
Changing initial poses, velocities, or native physical parameters cannot add that missing hook.

Compute audit `22628262` checked the reviewed invariant against the actual public observation ledger and replayed all 1,186 training actions through the frozen model.
All 15 predicted glue attributes stayed exactly at their initial values.
Four exact observed attributes changed:

| Exact feature | Step 0 value | First differing step | Observed value at that step |
| --- | ---: | ---: | ---: |
| `span0.glue_end_a` | 0 | 58 | 0.2 |
| `span0.glue_end_b` | 0 | 156 | 0.2 |
| `span1.glue_end_b` | 0 | 122 | 0.2 |
| `span2.glue_end_a` | 0 | 200 | 0.2 |

The source invariant and either pair of differing exact observations establish inconsistent full-recording support for this frozen program under the declared deterministic sensor-only target.
The rollout confirms the inspected code path; finite candidate sampling alone would not prove inconsistency.
This is stronger than the earlier nominal-replay failures: no broader initial-state prior or larger sampler budget can make a constant output equal both observed values.
This conclusion is specific to the frozen no-op program, its runtime, and the full recorded target, not to all Bridge models or the acting agent's solve rate.
The agent still solved the recorded task; that does not imply that its simulator explains every observed mechanism.

The comparison must retain this as an explicit unavailable-posterior/model-inconsistency control.
It must not inflate sensor noise, replace later predicted glue values with observations, or select only the pre-glue prefix to claim the full target passed.
A revised simulator or an explicitly evaluated dynamics-discrepancy model is required to explain those transitions.
The legacy agent remains the deployment baseline while that separate model-adequacy problem is addressed.

The first audit, `22628239`, completed its rollout but incorrectly sent predicted states through the recording-only sanitizer.
The corrected audit reads predicted object features through `Observation.from_state`; the public observation ledger and its noise model are unchanged.
Both attempts remain in the experiment record, with the first classified as an audit setup error.

Artifact: [invariant and exact-observation report](../../logs/uncertainty_bridge_invariant_v2_20260912/job-22628262.json).

## Integrated conditional-base sampling and support assessment

The offline SMC implementation now accepts `ConditionedPrior`, which identifies the original generative prior, exact-conditioning map, and normalized uniform proposal in free coordinates.
The map returns full joint coordinates and the conditional-base/proposal log-density ratio.
Initialization retains that ratio, and every Metropolis move retains its untempered difference while tempering only the remaining likelihood.
Results include eliminated initial-state coordinates so parameter and state marginals and future predictions use the same weighted joint samples.
The caller must justify the map's coverage and density; a declaration alone does not prove them.
The implementation still requires at least one free continuous proposal coordinate and does not provide a general contact-constraint chart.

The integrated numerical model is `x(t) = start * theta**t`, with exact observation `x(1) = 0.5`.
The original prior is uniform in `theta`, `start`, and an unused coordinate; eliminating `start` contributes `1/theta` to the conditional base density.
One reference uses `theta` between 1 and 8 with no remaining noisy evidence, testing that repeated moves preserve this nonuniform base.
The other uses `theta` between 1 and 2 with noisy observations at times 0, 2, and 3, comparing the fitted joint distribution and prediction at held-out time 4 against independent midpoint integration.
The unused coordinate should retain its original marginal.

Compute job `22628442` ran the predeclared two problems, two budgets, and four seeds per combination.
All runs used 24 temperatures and three moves per temperature.

| Reference | 512 particles | 2,048 particles |
| --- | ---: | ---: |
| Conditional base with no remaining likelihood | 2/4 passed | 4/4 passed |
| Noisy dynamics and held-out prediction | 4/4 passed | 4/4 passed |
| Total | 6/8 passed | 8/8 passed |

The smaller-budget base trials at seeds 1 and 2 missed the predeclared mean tolerance of 0.08, with errors 0.1187 and 0.1440.
Their other acceptance metrics passed; both failures remain recorded, without relaxing thresholds or replacing seeds.
The largest mean error for the larger-budget base reference was 0.0612.
For the larger-budget noisy reference, maximum parameter-mean error was 0.00456 and maximum held-out predictive-mean error was 0.0156.
These are numerical reference trials, not agent seeds or a posterior-calibration study across independently generated datasets.

The first focused regression job, `22628424`, missed the broader-base mean tolerance of 0.07 at 2,000 particles and seed 4, with error 0.1109.
The revised fixed regression increases its particles to 8,192 while keeping that seed, tolerance, temperature schedule, and proposal scale unchanged.
The original finite-budget failure is preserved alongside the independent multi-seed budget comparison above.
Job `22628478` passed all 26 functional tests and four-file mypy, then found one overlong source line in lint.
Job `22628640` also passed all 26 functional tests and four-file mypy, but splitting the line still left its assignment overlong.
The final static-check snapshot shortens only that local temporary's name, without changing arithmetic or control flow.
Job `22628728` passed four-file mypy, all four configured lint checks, and pinned isort, yapf, and docformatter checks.
Together with the 26 passing functional tests, these are scoped compute-node checks, not full-repository CI.

The tests additionally cover a conditional component with log mass -1000 that an exact discrete observation selects.
Its mass must remain in log space until the first likelihood update, rather than disappearing through premature underflow.
Invalid map outputs raise errors; exhausted budgets and finite searches with no supported particles return no posterior samples.
Original Box-prior sampler parity job `22628479` compares eight old/new cases in separate processes and obtains byte-identical serialized results, including samples, weights, and diagnostics.

The separate `audit_constant_outputs` API takes a reviewed declaration tied to program bytes, runtime identity, and a review artifact.
It checks every supplied episode's exact predicted observations, respecting reset boundaries, and returns the first conflicting pair for each declared feature.
Noisy measurements and exogenous conditioned inputs cannot establish this exact contradiction.
`model_inconsistent` means witnesses contradict the supplied invariant; `not_disproved` means only that this check found no contradiction.
The API neither proves arbitrary Python semantics nor infers invariants from finite rollout samples.

The same reference job reloaded the full frozen Bridge public ledger, verified its data and sensor identities, and applied the earlier reviewed glue invariant.
It found the four recorded changes at steps 58, 122, 156, and 200, with zero sampler evaluations and zero simulation steps for this check.
That assessment remains specific to the frozen no-op program and reviewed runtime.
It is separate from sampler `no_particle_support`, unsupported conditional charts, predictive disagreement under nonzero likelihood, and agent solve outcomes.

The production agent continues to use legacy uncertainty handling.
Full physical initial-state priors, remaining exact robot/contact constraints, and complete runtime capture still gate real-domain posterior comparisons.

Artifacts: [reference plan](../../logs/uncertainty_conditional_reference_20260912/plan.json), [all reference trials and Bridge witnesses](../../logs/uncertainty_conditional_reference_20260912/job-22628442.json), [original sampler parity](../../logs/uncertainty_sampler_parity_20260912/job-22628479.json), [functional checks](../../logs/uncertainty_conditional_batch_v3_20260912/checks-22628640.xml), [final static-check plan](../../logs/uncertainty_conditional_batch_v4_20260912/plan.json).

## Rigid-assembly prior and planar-contact reference

The offline `RigidAssemblyPrior` maps normalized free coordinates to one coherent assembly, including original weld frames and correlated body motion.
Its geometry consists of fixed local body poses and enclosing collision radii inside a declared obstacle-free cell.
A full scene prior must justify those inputs and specify uncertain geometry, attachment alternatives, robot state, and component probabilities.
The implementation does not infer them from hidden recording metadata or plug noisy observed poses into prior bounds.
See [the component contract](assembly-prior.md).

| Component | Continuous coordinates | Explicit construction |
| --- | ---: | --- |
| Free/rest | 6 | Uniform root xyz and uniform SO(3) orientation; zero twist |
| Free/moving | 12 | Free/rest pose distribution plus bounded root linear and angular motion |
| Planar support/rest | 3 | Uniform xy and yaw; exact height from a declared lowest support face; zero twist |

The components are separately normalized distributions, not a mixture with unspecified implicit weights.
Root placement bounds conservatively contain every orientation permitted by the component, and nonoverlapping enclosing spheres guarantee internal separation under the declared geometry.
The tabletop case is a separate lower-dimensional contact distribution, not the result of projecting unconstrained position samples onto a table.
Child velocities include `omega cross offset`, which is needed for instantaneous compatibility with the weld.
The mechanical reference found up to 0.03851 m/s of attachment-velocity disagreement if one instead copied the parent's linear velocity to the child.

Job `22628947` used a generated 6 cm cube and an attached 2 cm-radius sphere at a fixed 12 cm root-frame offset.
It loaded no evaluator task or historical hidden state.
For each component and seeds 0 through 3, it simulated 240 gravity steps in a new PyBullet world, repeated the trajectory in another fresh world, then reconstructed the full 120-step prefix before returning the suffix in a third world.
The declared checks required no initial penetration beyond 1e-12 m engine geometry roundoff, actual plane contacts, finite states, and repeat/prefix differences no greater than 1e-12.
The geometry roundoff threshold is an audit tolerance, not a softened observation likelihood.
The twelve trials used 8,640 simulator steps.

| Component | Mechanical trials passing | Largest repeat/prefix feature difference | Largest weld position deflection during simulation |
| --- | ---: | ---: | ---: |
| Free/rest | 4/4 | 0 | 2.696 mm |
| Free/moving | 4/4 | 0 | 3.414 mm |
| Planar support/rest | 4/4 | 0 | 0.001027 mm |

The finite-force weld deflections are retained as simulation behavior, not treated as failed reconstruction or added sensor noise.
The prior establishes compatible geometry and velocity at initialization; it does not turn the engine's weld solver into a perfectly rigid constraint.
The initial supported face also does not certify static balance for arbitrary masses or geometry.
These generated trials validate a physical component and repeatability, not the full historical balloons task, a posterior fit, or agent solve performance.

Final validation `22628938` passed 21 functional tests, two-file mypy and configured lint, and pinned isort, yapf, and docformatter checks.
The tests include actual engine separation and contact distances, independently composed weld frames, a finite-difference rigid-motion check, rotational isotropy, declared support rejection, and the existing replay suite.
Initial validation `22628904` passed its 20 functional tests but found a tuple annotation too narrow for both coordinate dimensions.
Static follow-up `22628919` passed after annotating the variable-length tuple and explicitly discarding the validation property's return value.
The subsequent supported component adds one functional test and is covered by the final 21-test job.
The earlier eight free-assembly reference trials in `22628914` also passed and remain separate, overlapping evidence rather than eight additional final cases.
All validation and simulation ran on `mit_preemptable` compute nodes.

Artifacts: [final source plan](../../logs/uncertainty_assembly_v3_20260912/plan.json), [predeclared mechanical checks](../../logs/uncertainty_assembly_v3_20260912/reference-plan.json), [per-trial physical results](../../logs/uncertainty_assembly_v3_20260912/reference-22628947.json), [functional checks](../../logs/uncertainty_assembly_v3_20260912/checks-22628938.xml).

## Robot-prior conditioning and hidden-joint geometry

`JointStatePrior` requires explicit position support and velocity distributions for every movable joint, plus a mechanically fixed designation for joints with no freedom.
Exact initial-position conditioning retains the original uniform density of each eliminated coordinate and does not condition away unobserved velocities.
A fully determined conditional has no artificial free interval and still exposes its observation factor.
Out-of-support exact readings raise a distinct prior-support contradiction instead of being clipped, wrapped, or softened.
See [the robot-state contract](robot-state-prior.md).

The reference uses the five previously verified public initial joint readings and checks the matching Fetch URDF hash in each current visible simulator.
The declared trial position priors use URDF intervals for limited joints and `[-4*pi, 4*pi]` for continuous joints.
The latter is an explicit finite winding prior, not a mechanical limit.
The rest component fixes all initial velocities to zero; the moving component uses independent uniform velocities with half-width 0.25 in each joint's coordinate units per second.
These are modeling assumptions for this component audit, not calibrated task priors or measured reset velocities.
The two components have no implicit mixture probabilities.

Final reference job `22629290` reports:

| Domain | Initial-position support | Rest / moving free dimensions | Sampled joint states restored | Largest joint restoration difference |
| --- | --- | --- | ---: | ---: |
| Bridge | Compatible | 4 / 17 | 64 | 0 |
| Fan | Compatible | 4 / 17 | 64 | 0 |
| Domino | Compatible | 4 / 17 | 64 | 0 |
| Boil | Compatible | 4 / 17 | 64 | 0 |
| Original balloons | Incompatible with this bounded prior | No conditional samples | 0 | Not evaluated |

Each compatible component used 32 generated joint states, totaling 256 across the four domains.
Every exact controlled initial position and every generated position/velocity was restored exactly by the engine.
Changing the unobserved initial joint positions and velocities left the instantaneous public robot features unchanged in these checks.
That statement concerns initial kinematics only, not subsequent dynamics or likelihood invariance.

The balloons shoulder-lift reading is `-1.5119263197144368` rad while its URDF interval is `[-1.221, 1.518]` rad.
Both trial components therefore assign zero support to that initial reading, independently of sampling budget.
The initial reference `22629166` stopped on this exception after capturing the four compatible domains.
The final report catches this specific outcome and continues the geometry audit with the original bounds unchanged.
It does not convert arbitrary setup exceptions into statistical failures or manufacture a balloons conditional.
A justified initialization law still needs to address this recorded reset state.

The separate geometry check keeps observed arm/gripper positions fixed and compares head pan/tilt settings `(-1.2, -0.5)` and `(1.2, 1.0)` rad.
It searches a predeclared 9-by-9-by-9 grid over the union of their head bounding boxes using a 1 cm-radius sphere.
All five visible models produce identical public robot features but a collision witness: the probe is about 5.62 mm inside the first head collision geometry and 210.05 mm separated from the second.
The probe is intentionally a signed-distance query, not an admissible penetrating scene used for inference or evidence of contact in the historical recordings.
This rules out using arm forward kinematics alone to justify discarding unobserved head state.
A complete reduction would also have to establish that the entire declared scene/action support and the learned program cannot depend on those joints.

Validation `22629135` passed 25 functional tests and two-file mypy, then reported four missing test docstrings in lint.
Final static job `22629186` passed mypy, both configured lint checks, and pinned formatting after adding those docstrings without changing logic.
The functional suite includes four new joint-prior tests and the existing exact-conditioning and physical-replay suites.
All validation and visible-engine audits ran on `mit_preemptable` compute nodes.
These are initial-state component checks, not agent seeds, posterior fits, or full-scene feasibility certificates.

Artifacts: [source and validation plan](../../logs/uncertainty_joints_v2_20260912/plan.json), [reference assumptions](../../logs/uncertainty_joints_v2_20260912/reference-plan.json), [per-domain component and geometry report](../../logs/uncertainty_joints_v2_20260912/reference-22629290.json), [functional checks](../../logs/uncertainty_joints_20260912/checks-22629135.xml).

## Reset law and composed scene support

This increment adds `GaussianJointPosition` and generalizes the offline joint prior's field to `position_priors` with a new schema identity.
Finite uniform bounds remain supported, and their prior-specific rejection of the original balloons start remains visible.
The Gaussian alternative declares zero mean and standard deviation pi radians for revolute coordinates or 0.1 m for prismatic coordinates.
Those settings are engineering assumptions for this development reference, not estimates fitted to individual readings or independent evidence of calibration.
Exact joint measurements retain their original Gaussian density; neither angles nor observations are clipped or wrapped.
The actual robot wrapper reproduces the public initial joint vector exactly in all five domains, including the balloons shoulder outside its URDF interval.
The wrapper and vanilla IK source show why ideal joint-limit support is not a guarantee of every simulator reset.

`draw_feasible` samples a complete normalized base candidate, including its mixture case, and rejects the whole draw when its support predicate fails.
The resulting prior is proportional to the base prior times the constraint indicator.
It returns an explicit exhausted-budget outcome rather than a partial batch presented as complete or a claim of impossible support.
It does not supply an exact normalizer or model evidence; parameter-dependent normalization remains the caller's responsibility.
The numerical references check a triangular joint constraint and feasibility-induced changes in mixture case probabilities.

The physical reference combines the five recorded initial robot joint vectors with generated rigid cube/sphere assemblies in a declared cell, using equally weighted rest and moving joint/assembly cases.
It does not load the historical object layout or fit any dynamics.
The first geometry audit, `22629679`, checked sampled bodies against the robot and all existing geometry, and completed all ten requested batches.
That policy did not check robot/background intersections and is retained as a narrower reference.

The stricter audit `22629703` added robot/background checks and accepted zero candidates in all ten trials, each exhausting its 128-draw budget.
Finite rejection alone would not establish empty support.
Contact diagnostic `22629779` identified the same two wheel/plane intersections in all five visible models.
Source inspection supplies the geometric explanation: each spherical wheel collision shape has radius 0.065 m and center height 0.055325 m under the fixed base.
Its floor signed distance is therefore -0.009675 m for every wheel angle.
This makes the strict no-overlap predicate incompatible with the supplied anchored fixture geometry.
The failure was not addressed by increasing the sample count, changing geometry, or softening sensor observations.

The final declared predicate permits only those two named wheel/plane fixture contacts at their source-established signed distance, checked within the 1e-9 m geometric roundoff policy.
Any changed fixture distance raises an error; every other queried robot/background or sampled-body intersection remains a rejection.
The support identity includes that explicit contact rule and the reviewed source hashes.
The policy is conservative about queried collision geometry and does not add robot self-collision to the existing model.

Final reference `22629815` uses the standard-library Gaussian inverse CDF and runs two fixed seeds per domain, each requesting eight accepted generated scenes with a maximum of 128 complete draws.

| Domain | Exact wrapper-reset error | Complete draws | Accepted generated scenes | Collision rejections |
| --- | ---: | ---: | ---: | ---: |
| Bridge | 0 | 16 | 16 | 0 |
| Fan | 0 | 17 | 16 | 1 |
| Domino | 0 | 16 | 16 | 0 |
| Boil | 0 | 16 | 16 | 0 |
| Original balloons | 0 | 17 | 16 | 1 |
| Total | 0 | 82 | 80 | 2 |

Every accepted sample passed the same geometry predicate when checked again in reverse batch order.
This is same-world geometry rechecking, not a new claim of complete runtime closure or long-trajectory replay.
The joint components retain 4 free position coordinates at declared rest or 17 position/velocity coordinates in the moving case; the associated assembly contributes 6 or 12 coordinates respectively.
These counts describe the generated reference and its stated shared motion case, not a finalized historical task prior.
No posterior fit, agent solve-rate seed, or historical full-scene comparison was produced by this audit.

Final validation `22629786` passed 35 functional tests, four-file mypy and configured lint, and pinned isort, yapf, and docformatter checks.
The functional set covers six joint-prior tests, three global-rejection tests, five assembly tests, five affine-conditioning tests, and sixteen physical-replay tests.
Initial validation `22629663` passed the same functional set but found a missing generic list annotation.
Static follow-up `22629702` passed mypy but its lint could not resolve SciPy's dynamically exposed inverse CDF.
The final implementation uses the standard-library equivalent and reruns the functional checks with that implementation.
All tests and engine audits ran on `mit_preemptable` compute nodes.

Artifacts: [final source plan](../../logs/uncertainty_scene_prior_v3_20260912/plan.json), [declared reference and fixture policy](../../logs/uncertainty_scene_prior_v3_20260912/reference-plan.json), [final per-domain results](../../logs/uncertainty_scene_prior_v3_20260912/reference-22629815.json), [strict-policy rejection report](../../logs/uncertainty_scene_prior_v2_20260912/reference-22629703.json), [contact witnesses](../../logs/uncertainty_scene_prior_v2_20260912/contacts-22629779.json), [functional checks](../../logs/uncertainty_scene_prior_v3_20260912/checks-22629786.xml).

## Feasible scene weights in batch inference

`FeasibleConditioning` connects a declared support predicate to the existing exact-conditioning and tempered-sampling interfaces.
It distinguishes a globally constrained joint prior from a state prior normalized separately for each parameter value.
The latter retains the intended parameter marginal by including the original support probability `Z(theta)` in every base weight.
Exact-observation and proposal-density corrections remain in that weight, and the remaining noisy likelihood enters once afterward.
The distinction and analytic derivation are documented in [scene-prior composition](scene-prior-composition.md#connecting-feasible-scenes-to-parameter-inference).

Numerical reference `22630419` evaluates both laws on the same support and exact observation, using eight seeds from 100 through 107 at each of two particle counts.
Before submission, the plan fixed 12 temperatures, three moves per temperature, a maximum of 100,000 target evaluations per trial, and accuracy thresholds of 0.06 for the parameter mean, 0.09 for the uninformed coordinate mean, and 0.13 for each parameter quantile at probabilities 0.05, 0.5, and 0.95.
Every completed fit must also preserve the exact constraint to floating-point residual at most 1e-15.
These criteria check this known numerical reference; they are not general posterior-calibration or deployment thresholds.

| Prior law | Particles | Trials meeting all criteria | Largest parameter-mean error |
| --- | ---: | ---: | ---: |
| Global joint conditioning | 512 | 7/8 | 0.07232 |
| Global joint conditioning | 2,048 | 8/8 | 0.02107 |
| Conditional state normalization | 512 | 8/8 | 0.05651 |
| Conditional state normalization | 2,048 | 8/8 | 0.02851 |

All 32 numerical fits reached their final temperature, but reaching that temperature alone does not satisfy the accuracy criteria.
The smaller global-joint trial with seed 107 returned mean 2.23636 against the analytic mean 2.16404.
Its minimum effective sample size was approximately 441 out of 512, and all 512 original ancestors survived, so those diagnostics did not expose the mean and median error by themselves.
The failed trial remains part of the report, without a replacement seed or adjusted threshold.
The two larger-budget groups pass all sixteen trials.
The experiment used 1,392,329 target evaluations in total; these were algebraic reference evaluations, not simulator steps or agent actions.

The functional checks also include an exact observation that excludes part of the parameter interval, combined with a noisy reading and an independent quadrature reference.
They verify that invalid normalizers and callback failures propagate, rejected candidates cannot become posterior samples, and support callbacks cannot mutate shared candidate arrays.
Initial validation `22630373` passed 22 functional tests and then found two test callbacks without types that mypy could infer.
The corrected test snapshot adds explicit callback annotations and pinned formatting; its implementation module is identical to the completed numerical reference.
Final validation `22630449` passed all 22 functional tests, two-file dependency-following mypy, two configured lint checks, and pinned isort, yapf, and docformatter checks.
Both final validation and the numerical experiment completed with exit status zero on `mit_preemptable` compute nodes.
This is focused validation, not a full repository CI run.

These results establish the support-weight composition on the stated reference problems.
They do not provide historical scene layouts, attachment-case probabilities, unknown parameter-dependent normalizers, or a conditional representation for exact contact trajectories.
The production estimator and historical experiment runtime remain unchanged.

Artifacts: [predeclared reference plan](../../logs/uncertainty_feasible_batch_v2_20260912/plan.json), [all numerical trials](../../logs/uncertainty_feasible_batch_v2_20260912/reference-22630419.json), [final check snapshot](../../logs/uncertainty_feasible_batch_v3_20260912/plan.json), [functional checks](../../logs/uncertainty_feasible_batch_v3_20260912/checks-22630449.xml).

## Recorded scene initialization and Balloons contact constraints

The public candidate-map audit `22631317` uses only projected public observations, fresh visible-model body handles, and the full reset path.
All five domains reproduce every exact initial field.
It probes all 177 noisy initial coordinates in both directions, totaling 354 perturbations.
The results expose sixteen fixed Fan pose outputs, two reset/derived Boil scalar outputs, and coupled canonical Euler outputs near Bridge's pitch pole.
Those are properties of the visible initialization map; they do not by themselves identify a learned program's complete prior or establish geometric feasibility.
Initial setup `22630947` imported `PyBulletState` from the wrong module and performed no audit.
Follow-up `22630961` used an unnecessary float32 cast and missed Domino's component-owned handles; `22631317` corrects both without changing production or recorded data.

The new Gaussian-coordinate conditioning helper supplies exact Gaussian initial-position proposals, together with their original marginal observation density.
It supports exact coordinates by elimination, retains evidence under sequential independent readings, and rejects numerical overflow rather than manufacturing zero support or a small sensor variance.
Validation `22631716` passed seventeen functional tests, two-file dependency-following mypy, two configured lint checks, and pinned formatting.
The initial check job `22631701` named a nonexistent test file and ran no tests; that setup failure is separate from the completed validation.

The [Balloons initial-scene reference](balloons-initial-scene.md) supplies a declared original free-pose scene law with 42 through 73 continuous dimensions across sixteen motion cases after initial conditioning.
It conditions translations through the Gaussian helper, handles clip yaw with its truncated angular likelihood, preserves omitted orientations, and conditions exact initial joints and box speed.
Its support policy is explicit about permitted static-fixture overlaps and the visible wall-box-only chute rule.
All body handles and hidden candidate quantities come from the model and prior, not evaluator recording metadata.

The first attempt `22631523` called the visible rack-placement helper by the wrong name and sampled no scenes.
The next attempt `22631543` correctly refused the constructor-only wheel-contact assumption after the full robot reset changed the base frame.
The URDF inertial offset gives the corrected, source-derived reset contact distance of -11.075 mm, compared with -9.675 mm immediately after construction.
The previous generated-component reference remains valid for its own constructor-only protocol; it does not certify the full-reset geometry.
Reference `22631584` verifies the corrected geometry with inline conjugate conditioning, and final `22631706` repeats the same experiment using the checked Gaussian helper.

| Sampling seed | Complete candidate draws | Accepted roots | Exact initial observations reproduced | Repeated 16-action replay identical |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,242 | 8 | 8/8 | 8/8 |
| 1 | 1,715 | 8 | 8/8 | 8/8 |

These are conditional initial-state samples, not agent seeds or dynamics-parameter posteriors.
Each accepted root was replayed twice in fresh worlds, totaling 512 model action steps in the final reference.
Every root still contradicts at least one later exact output at action one, starting with box speed or moving robot joints.
Those failures remain visible instead of being discarded to obtain a successful full-target fit.

The separate support diagnostic `22631750` evaluates 84 one-action predictions over seven box-height offsets, three masses, and four native damping values.
The upright box at nominal public xy and table support height is admissible and reduces the first-speed discrepancy to less than `2.63e-11 m/s` for one tested setting, with exact first-step public joints.
Below-table controls remain explicitly inadmissible.
This motivates a supported prior component, not a change to the original free-pose law or an observation tolerance.

Constraint diagnostic `22631837` scans native damping over its visible `[0.01, 40]` range at three fixed masses.
The restricted supported component has a near-equality candidate around 2.2, but two other sign-changing brackets terminate at discontinuities and fail equality.
The latter's finite-difference slopes scale inversely with the step size, so their displayed inverse-slope factors are diagnostic arithmetic, not valid conditional weights.
Even the low-damping candidate retains later exact-speed discrepancies of approximately `1e-4 m/s` within the sixteen-action prefix.
A complete conditional representation must account for supported/free cases, uncertain geometry, all relevant branches, numerical conditioning, and later observations before a physical posterior comparison is available.
None of these numerical candidates was published to an acting agent.

Every audit, model replay, and functional/static validation ran on `mit_preemptable` compute nodes.
The completed reference and check jobs exited with status zero, with their statistical and model failures retained in their reports.
Full repository CI and live estimator comparisons were not run in this increment.

Artifacts: [five-domain map audit](../../logs/uncertainty_scene_map_v3_20260912/reference-22631317.json), [Gaussian source snapshot](../../logs/uncertainty_gaussian_coordinate_v2_20260912/plan.json), [Gaussian functional checks](../../logs/uncertainty_gaussian_coordinate_v2_20260912/checks-22631716.xml), [final declared root law](../../logs/uncertainty_balloons_root_v4_20260912/plan.json), [root samples and replay](../../logs/uncertainty_balloons_root_v4_20260912/reference-22631706.json), [supported-box control](../../logs/uncertainty_balloons_support_20260912/reference-22631750.json), [scalar constraint diagnostic](../../logs/uncertainty_balloons_constraint_20260912/reference-22631837.json).
