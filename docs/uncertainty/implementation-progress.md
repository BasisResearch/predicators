# Uncertainty simplification: implementation progress

Updated September 12, 2026.
This tracks implementation of the [simplification proposal](simplification-proposal.md).
The incumbent estimator remains the production default.

## Implemented boundary

The rollout fitter exposes `SysIdOutcome.inference`, a version 1 `LegacyInferenceResult`.
The summary contains the existing point estimate, selected parameters, per-parameter diagnostics, and segment coverage.
The synthesis tool and approach consume this summary without changing parameter selection or publication.
Each view owns copies of its dictionaries, so a caller cannot change the fit cache by modifying its diagnostics.
The adapter neither fits nor samples.

The metadata explicitly identifies `legacy_rollout_sysid` and `legacy_widths`.
These widths are not newly claimed credible intervals, and the adapter does not turn optimizer candidates into weighted posterior samples.
The original `FitResult`, diagnostics, caches, publication methods, and checkpoint fields remain available to their existing consumers.
The adapter is a property rather than a new stored field, so historical outcomes need no schema migration.
Canonical, diagnostic, cached, and no-survivor fits retain their existing handling.

The separate offline prototype now defines program, prior, sensor-model, runtime, and observation-ledger hashes.
Real training recordings now pass the corrected reader and content-addressed snapshot audit in all five domains.
The reader reconstructs the seeded observation channel from stored sanitized truth; it does not expose the noiseless stored poses as inference observations.
Complete simulator runtime and resource closure is still pending.
Explicit working-directory files, optional-file absence, and the complete child environment now have an immutable `RuntimeInputs` contract.
Fresh-process balloons replay validates this portion of the runtime identity; native dependencies and arbitrary external reads remain outside that guarantee.
The existing legacy cache key is not represented as an immutable statistical data identity.
There is no new public estimator flag or agent-facing tool output in this chunk.

## Candidate replay contract

The separate offline `replay_candidate` API accepts a fresh environment factory, an explicit `ReplayState`, executed actions, and fixed candidate parameters.
It returns the reconstructed initial state and every subsequent state.
This permits separate measurement of initialization and transition errors.
The incumbent `rollout_states` continues to zero velocities exactly as before.

| Quantity | Offline replay behavior |
| --- | --- |
| Object features | Restore the supplied candidate through the domain's state interface. |
| Object linear and angular velocities | Require explicit finite values for every physical object and restore them even if its pose already matches. |
| Robot joints | Require position and velocity for every URDF joint, including passive joints; check agreement with the state's controlled joint positions. |
| Robot base | Restore the supplied base velocity; mobile robots must also supply a base pose. |
| Model memory | Require explicit memory when the subclass declares it; preserve and copy it across branches. |
| Command attachments | Validate object names and restore the existing portable attachment topology. |
| Environment lifetime | Build and dispose a fresh world per candidate, including on restoration or step failure. |

`capture_replay_state` reads a simulated candidate, or evaluator state for a mechanical offline audit.
It is not installed in the agent's observation, recording, or tool interface.
It drops privileged payloads and live simulator handles while retaining candidate memory.
An inference caller must supply sampled or legitimately known unobserved quantities; it must not capture the live task to initialize its candidates.
Historical recordings do not supply all passive-joint positions or joint velocities, so those quantities still need an explicit initialization assumption or prior.

This representation is not an exact engine checkpoint.
Replay must use the same domain layout, robot/URDF, and environment configuration as the candidate state; it does not support arbitrary changes to body allocation or morphology.
The state interface omits solver caches.
The later replay correction explicitly captures full body orientations, original command-attachment frames, and pending next-step commands.
Arbitrary subclass instance variables and arbitrary native engine constraints are not implicitly captured.
Model authors must put persistent inferred quantities in declared model memory, and further audits must determine whether omitted engine state materially affects predictions.
Geometrically valid initial-state sampling and the observation likelihood remain separate work.
Passing the structural checks does not certify that an arbitrary candidate pose is feasible or explains the data.

## Validation evidence

The pre-change source is preserved in the detached worktree `/home/ycliang/predicators-uncertainty-baseline-20260911` at `6179fe1e7`.
This is the direct interface-parity baseline, distinct from the historical successful experiment tag `noisy-mb-five-domain-15of15-20260910`.
No historical experiment runtime or result was modified.

The moving-start reproduction creates a box with vertical velocity 0.5 m/s and records 15 real engine steps.
Legacy fitting replay differs by up to 0.1636 m because it discards that velocity.
The new replay tests exercise motion through table contact, a resumed prefix, independent memory branches, complete robot motion, and welded-assembly restoration.
An additional reproduction showed that `State.copy()` shares mutable object metadata across candidate worlds.
Offline replay now copies object metadata as well as feature arrays and model memory, without changing the production state-copy implementation.
They also reject missing motion, missing memory, inconsistent joint positions, and unknown attachment endpoints.

The compute-node checks additionally compare the actual `sim.fit` reports and publication calls for canonical, cached, and diagnostic fits against the pre-change source.
Scripted continual interactions compare observations, actions, tool replies, steps, and resets across the five noisy physics domains and a solved cover task.
Action traces, tool replies, and counters are compared exactly, with only the existing elapsed-clock text normalization.
Numeric observation values use an absolute comparison tolerance of 1e-12, with every nonzero difference retained in the comparison report.
The first comparison found one floating-point observation difference of 3.39e-21 across compute nodes; all actions, replies, and counters were identical.
These mechanical comparisons are not stochastic LLM solve-rate replications.
The wider regression run also exposed three pre-existing publication-test failures on the unchanged baseline: an old stub lacked the subclass-parameter synchronization method.
Those tests now exercise the real approach's publication and cache methods without launching an SDK session.
The production publication implementation was not changed to accommodate the tests.

The reusable audit is [scripts/audit_inference_replay.py](../../scripts/audit_inference_replay.py).
It runs a 30-action hold sequence and resumes from its third action in each domain, comparing reconstruction, repeated replay, and continuation against the source world.
It uses the evaluator program with registry parameters pinned identically in source and replay, solely to isolate restoration error.
It does not test learned programs or certify long task-solving trajectories.

The completed audit restored initial features exactly and repeated fresh replay exactly in all five domains.
Maximum absolute position differences from the source world, in millimeters, were:

| Domain | Replay from initial state | Replay from action 3 |
| --- | ---: | ---: |
| Bridge | 0.0154 | 0.0113 |
| Fan | 0 | 0 |
| Domino | 0 | 0.00261 |
| Boil | 0 | 0.000730 |
| Original balloons | 0 | 0.0142 |

These small errors describe the audited short hold sequences only.
They do not establish a bound for releases, sustained contacts, long trajectories, or incomplete learned models.

Commands, raw comparisons, and audit JSON are under `logs/uncertainty_migration_20260911/`.
All simulations and test suites run on `mit_preemptable` compute nodes.
The completed functional checks cover 79 tests: 12 focused adapter/replay tests and 67 existing uncertainty, fitting, publication, and synthesis tests.
Focused mypy and lint checks and the pinned formatter checks cover the changed Python files.
The shared environment had `isort` 5.13.2, so the final checks use an isolated installation of the required 5.10.1 without modifying the shared environment.
These are scoped local checks, not a full repository test run or a PR/CI result.

## Offline probability prototype

The next chunk adds immutable observations and reset-episode ledgers, the declared additive sensor likelihood, and a bounded continuous tempered sampler beside the incumbent.
The joint sample vector can represent parameters and uncertain initial states in the small reference problems.
Repeated observation reads are deduplicated, exact predictions are constraints, and failed numerical runs return no posterior samples.
The sampler preserves the original prior across repeated calls.

See [the probability model contract](offline-probability-model.md) for assumptions, result semantics, and limitations.
The independent uniform reference prior is not yet a feasible physical-state prior for the five domains.
There is no new production import, estimator flag, prompt change, or agent experiment from this chunk.

The earlier probability prototype passed eight focused functional checks, four-file mypy with `--follow-imports=skip`, configured lint, and pinned formatters.
Those results apply before the sampler correction and recording adapter described below; they do not validate the current complete working tree.
A normal dependency-following mypy attempt exceeded its bounded local allowance.

### Numerical gate findings

The original every-temperature resampling prototype passed the stationary Gaussian reference and the position/velocity grid reference, including correlation and retention of an uninformed parameter.
The two-mode test first exhausted an undersized test budget.
After assigning the budget required by its declared particle/move counts, seed 19 retained both modes but assigned 72.2% to the positive mode of a symmetric posterior.
That failed the unchanged 35%-65% tolerance.
These are numerical reference problems, not agent solve-rate results.

The candidate implementation now resamples only below the declared effective-sample-size threshold and retains importance weights otherwise.
Posterior summaries and tests now use those weights, including inverse empirical-CDF quantiles.
The correction passed its compute reference suite, including the original failing seed and three additional seeds.
A separate eight-seed, three-reference experiment passed all 24 larger-budget comparisons; the smaller budget missed one uninformed-parameter tolerance.
These results validate the tested numerical references, not physical-domain inference.

### Recording and artifact preparation

The new read-only `inference_recording` adapter reads explicitly selected flushed level recordings, verifies reset markers and every primitive action, and preserves original source bytes.
It rejects missing action boundaries, unflushed/inconsistent files, duplicate reset identities, and incompatible sensor semantics.
Public joints and mobile base pose receive explicit exact sensor entries.
Extra metadata, including body velocities or command welds, requires an explicit exclusion reason rather than silently entering or disappearing from the likelihood.
A candidate simulator's memory and privileged fields cannot enter through the observation projection.
Named source bundles preserve bytes and manifests without overwriting prior snapshots.
Dependency enumeration remains an explicit caller responsibility.
Writer-to-reader and artifact integrity tests passed on compute.
The adapter also validated frozen first-training-level recordings from all five domains, preserving 1,978 recorded primitive actions in total.

### Historical compute blocker, resolved September 12

Slurm originally returned `Unable to contact slurm controller (connect failure)`.
A fresh queue query timed out, and a bounded submission retry also timed out without a job ID.
This session cannot verify whether the retry was accepted; no compute output has been observed.
Before another submission, inspect the queue for the prepared `checks.sbatch` job once connectivity is restored.
The user reiterated that expensive tasks must run on compute nodes.
Inference sweeps, physical replay audits, full checks and live experiments therefore remain pending compute access.
No live estimator change or new agent experiment was made.

Evidence and reproducible commands are under `logs/uncertainty_probability_20260911/`.
The prepared compute entry point is `checks.sbatch`; current source hashes and validation limits are in `continuation-manifest.json`.
Only syntax and formatter checks have been applied to the new recording code and resampling correction locally.
Changes remain uncommitted.

### September 12 preparation before network access was restored

Compute access remains unavailable from this session: another bounded queue query timed out, and no output from the earlier validation submission was found.
A connection-tracing attempt was also disallowed by the execution environment, so the timeout has not been diagnosed as a cluster outage.

A frozen validation job is now prepared at `logs/uncertainty_stage_a_20260912/checks.sbatch`.
It reconstructs commit `6179fe1e7`, the captured working-tree patch, and 13 captured overlay files in compute-node scratch space.
It validates there without formatting or otherwise modifying the live checkout.
Input hashes, runtime-version verification, per-job test reports, and an exit-status record make the result attributable to that snapshot.
Preparation passed syntax, source-consistency, and standalone patch-application checks; the full job has not run.

The frozen job covers the corrected weighted sampler, four two-mode seeds, recording integrity, focused legacy/replay regressions, dependency-following mypy, configured lint, and pinned formatters.
The user has been asked to submit it from a normal cluster terminal and share its job ID, because this session still cannot reach Slurm.
Inspect the queue for the earlier `checks.sbatch` attempt before submitting another job.
The frozen job itself has not been submitted by this assistant session.
See `logs/uncertainty_stage_a_20260912/README.md` for the exact command and result paths.

### September 12 completed checks and active experiments

Network access was restored, and the assistant submitted the prepared jobs directly.
Frozen-source validation job `22625595` completed successfully on `mit_preemptable`.
It passed 24 numerical/recording tests, 15 focused legacy/replay regressions, dependency-following mypy on 16 files, 16 configured lint checks, and pinned formatting checks.
These are focused checks, not a full repository test run or CI result.

Independent numerical experiment `22625659` completed 48 runs over three reference problems, eight seeds (100 through 107), and two particle budgets.
All 24 runs at 1,024 particles passed the criteria frozen before submission.
At 256 particles, 23 of 24 passed; correlated-reference seed 102 shifted the uninformed parameter mean to 0.143, beyond its 0.12 tolerance.
The smaller budget remains a documented stress-test failure rather than a recommended default.
See [the September 12 experiments](experiments-20260912.md) for per-reference counts and source paths.

The corrected recording audit `22625932` passed for all five domains.
Initial audit `22625681` established file integrity but used the wrong observation projection; its statistical data is superseded.
It reads only explicitly selected historical training levels, reconstructs the same step-keyed noisy views as the continual agent, verifies action/reset alignment, retains public joints, rejects an exact-observation contradiction, and freezes the source bytes and channel coordinates.
This audit is not a fit or prediction-quality result.

Nominal fixed-program prediction preflight `22625747` failed during configuration setup because the launcher manifest uses the CLI alias `log`, while `reset_config` expects `log_file`.
No simulation ran in that attempt.
Corrected setup job `22625774` keeps the same programs, data, and numerical predictions; it translates that alias before configuring the environment.
This experiment measures nominal prediction disagreement and repeated-replay error from fixed training-program snapshots before posterior fitting.
It does not claim to reproduce historical fitted parameters, and explicitly excludes the optional balloons `model_params.json` override in an isolated working directory.
All such setup outcomes remain separate from model or agent outcomes.

### Observation-channel correction and physical prediction findings

The reader's first implementation mistook stored sanitized simulator truth for noisy agent observations.
A regression through the actual continual session and writer reproduced this in job `22625874`.
The corrected reader requires run seed and level index and reconstructs the same observation channel used by `ContinualRun._observed`.
Final job `22626126` passes all 29 functional tests, mypy, configured lint, and pinned formatting after correcting a lint-only type check.
This change is offline-only and does not alter existing recordings or acting MF/MB agents.

Prediction setup also had to restore the original `b09217bb3` runtime: current-branch balloons scene controls differ from the historical recording runtime.
Corrected prediction job `22625933` used reconstructed noisy starts, public predicted outputs, and frozen training programs on that historical runtime.
All 14 executable nominal cases were exactly repeatable, but all contradicted at least one exact observation under the strict likelihood.
Two early Fan cases were invalid programs with a zero lower bound for a logarithmic parameter; they were not modified to make the experiment pass.
Nominal failures are not a proof that every feasible initial state and parameter is impossible.
They require explicit initial-state support and a reconstruction/model-error diagnosis before a meaningful posterior comparison.

Mechanical follow-up `22626183` isolates full-state restoration versus the legacy zero-velocity reset under recorded-action stress inputs in recreated evaluator worlds.
It does not feed evaluator state into a fit or an agent.
The completed audit shows residual reconstruction errors in every domain when robot joints are included.
The moving-start balloons continuation differs by 106.17 mm and changes attachment topology, compared with 86.78 mm for legacy zero-velocity replay.
Subsequent diagnostics `22626945` and `22626966` isolated omitted next-step commands, unobserved body orientations, and original weld frames.
Preserving all three reduced the balloons midpoint non-robot position error to 9.73e-14 m and restored the correct attachment sequence.
Robot joint differences remain, so a portable mid-trajectory state is still not an exact engine checkpoint.

The earlier `22626292` checkpoint failure had a separate lifecycle bug: restoring the engine did not remove constraints created after the saved boundary.
Removing those later constraints and verifying the originals reduced non-robot position error to zero in both follow-up checkpoint attempts.
This is diagnostic evidence, not a general checkpoint implementation or justification for adding sensor variance.

The offline replay now supports reconstructing the full action prefix in one fresh world.
Corrected audit `22627021` produced bit-identical prefix continuations versus uninterrupted candidate trajectories across all five domains, at both tested boundaries.
The explicit initializer API makes the candidate root protocol part of the model and artifact contract.
It never selects evaluator tasks implicitly; inference initialization must use the declared prior and allowed conditioned inputs.
The task-cache investigation additionally reproduced and fixed lost initial robot joints in Domino's cache.
Matching the existing continual fresh-world lifecycle then gave zero measured replay error in all five domains at both tested boundaries in audit `22627245`.
This validates explicit initialization plus full-prefix reconstruction for the tested development trajectories; arbitrary portable checkpoints remain approximate.
Physical-prior design, exact-output feasibility, and learned-program prediction quality remain separate gates.
Detailed results and limitations are in [the experiment record](experiments-20260912.md).

### Remaining full-plan execution

The [initial-state inventory](initial-state-inventory.md) now records the observed quantities, missing state, and unresolved support choices for all five frozen development recordings.
Visible-model audit `22627492` confirms Fetch has four unobserved movable joints in addition to its nine observed arm/gripper joints.
It also verifies that the first balloon box-speed observation is exactly zero.
These findings prevent incorrectly treating robot motion as fully observed or using only a positive-speed velocity chart.
The inventory is a gap audit; normalized physical priors and final joint dimensions remain unresolved.

The new offline `AffineConditioning` primitive eliminates a square nonsingular affine observation while retaining the induced density correction.
It separates unsupported singular charts, individual points outside prior support, and numerical solve failures.
This construction handles exact initial coordinate observations and parameter-dependent affine elimination; it is not a nonlinear contact-constraint solver.
Compute job `22627437` passed 15 functional tests, mypy, configured lint, and pinned formatting.
An eight-seed importance-sampling reference passed its predeclared checks in 8/8 runs at 8,192 particles and 5/8 at 512 particles; the smaller-budget failures remain recorded.
Those importance-sampling references are not agent solve-rate seeds.
The later conditional-base extension below integrates this construction with offline SMC; production fitting remains unchanged.

The next velocity component is implemented as an explicit rest atom plus an isotropic Gaussian moving component.
Exact rest retains the prior atom's mass, while positive speed retains the Maxwell radial-density factor and two uncertain direction coordinates.
This is a normalized component prior, not a completed joint scene prior; hyperparameters and dependencies still need specification.
Validation `22628133` passed 19 functional tests and focused type/lint/format checks.

Runtime reproduction `22628126` found up to 0.4961 m of predicted balloon-height change from an optional parameter sidecar, despite unchanged program bytes and declared parameters.
The new working-directory input contract distinguishes present/absent files and fixes the child environment.
Fresh-process validation `22628194` gave identical repeated feature predictions within each condition and distinct runtime-input identities between them.
Contract validation `22628197` passed 13 functional tests and focused type/lint/format checks.
The velocity and runtime suites therefore cover 32 distinct functional tests in this chunk.

Bridge audit `22628262` establishes a separate model-adequacy obstruction: the frozen no-op model keeps glue attributes constant, but four exact recorded glue attributes change.
Under this reviewed invariant, the full sensor-only target is inconsistent regardless of initial-state prior or sampling budget.
This case must return unavailable inference with its model contradiction visible; it is not a finite-search failure or a successful posterior with ordinary predictive residuals.
See [the experiment record](experiments-20260912.md#a-structural-exact-output-contradiction-in-the-frozen-bridge-model).

The offline sampler now accepts an explicitly identified conditional base measure and returns full joint parameter/initial-state samples.
Exact-observation density and proposal corrections enter the initial weights and every Metropolis acceptance ratio; only the remaining likelihood is tempered.
An integrated affine-dynamics reference checks these weights, parameter/state dependence, an uninformed coordinate, and prediction at a held-out time against numerical integration.
All eight reference trials pass at 2,048 particles; six of eight pass at 512 particles, with both smaller-budget failures retained.
This validates the tested conditional construction, not a contact-state prior or repeated-dataset calibration.
Eight comparisons against the prior Box sampler produce identical serialized results.
Validation passed 26 functional tests, four-file mypy and configured lint, and pinned formatting on `mit_preemptable`.

`audit_constant_outputs` checks exact observation contradictions under a separately reviewed, versioned program/runtime invariant.
It returns `model_inconsistent` with observation witnesses, or `not_disproved`; neither outcome is a posterior.
It does not infer invariants from successful rollouts or declare feasibility when no contradiction is found.
The frozen Bridge ledger yields all four known witnesses without sampler or simulation work.
See [the integrated validation record](experiments-20260912.md#integrated-conditional-base-sampling-and-support-assessment).

The new offline [rigid-assembly prior component](assembly-prior.md) generates correlated body poses, velocities, and original weld frames from one root pose and twist.
It provides explicit free/rest, free/moving, and horizontal-support cases with 6, 12, and 3 continuous coordinates respectively.
The supported case fixes height from a declared physical face instead of projecting independent noisy poses onto contact.
All 12 generated mechanical trials passed initialization and replay checks across gravity and plane contact, totaling 8,640 simulator steps.
Fresh repeats and complete-prefix replay matched exactly, while finite-force welds deflected by up to 3.41 mm during impacts.
Validation passed 21 functional tests and focused type, lint, and pinned formatting checks on compute nodes.
These are known-geometry component references, not historical-domain posteriors; case probabilities, geometry uncertainty, full scene support, and robot-state priors remain unresolved.

The offline [joint-state prior component](robot-state-prior.md) now retains exact initial-position density, unobserved joint positions, and explicit initial-velocity distributions or rest atoms.
Validation passed 25 functional tests and focused type, lint, and pinned formatting checks.
A visible-model audit restored 256 sampled joint states exactly across Bridge, Fan, Domino, and Boil, with 4 free coordinates in the declared rest component and 17 in the moving component.
The same bounded prior is incompatible with the frozen balloons initial shoulder position; it correctly returns no conditional samples for that case.
Head-geometry witnesses in all five domains show why identical public robot kinematics does not establish collision irrelevance of unobserved joints.
These findings refine the remaining initialization-law and scene-support work rather than completing the full physical prior.

The next [scene-composition reference](scene-prior-composition.md) covers the balloons joint start with a declared Gaussian reset law and retains its observation density.
An actual wrapper reset reproduces every recorded initial joint exactly; the old bounded prior remains a negative control.
Whole-candidate rejection preserves the stated joint/body measure, with an explicit finite-search outcome and no claim to estimate model evidence.
The generated five-domain reference accepted 80 scenes in 82 draws after treating the two source-established wheel/plane contacts as fixed fixture geometry.
The stricter all-intersections predicate rejected every candidate because those wheel spheres overlap the floor independently of joint angle; that failed policy remains recorded.
Final validation passed 35 functional tests, four-file mypy and lint, and pinned formatting on compute nodes.
These are generated component compositions, not historical scene reconstructions or replacements for the later exact-trajectory gate.

| Stage | Required work before advancement |
| --- | --- |
| A: probability model | Complete the five domain initial-state inventories, exact-conditioning construction and reference checks, full runtime capture, and feasible physical initial-state priors; existing numerical and recording references pass at the tested budget. |
| B: recorded predictions | Freeze development programs/data and compare with legacy on all five domains, including contact transitions, incomplete programs and held-out causal suffixes. |
| C: planning | Run saved-decision shadow reports, then matched live parameter-belief comparisons; evaluate exploration and conditional-state rollout changes separately. |
| D: execution state (optional) | Validate causal conditional filtering, reconstruction and bounded degraded-mode recovery if pursuing a replacement for the observation smoother. |
| E: retirement | Predeclare margins, evaluation size and budgets; run matched per-domain comparisons, retain inconclusive results, and retire the legacy parameter fitter only after acceptance. |

Under the September 12 proposal revision, Stage C can proceed directly to Stage E with the existing execution state estimator.
Conditional state sampling for planning and the live conditional filter are optional extensions, not requirements for completing parameter-uncertainty simplification.
The revised result contract also requires returning numerically adequate posterior fits with predictive failures visible, separately from decision use; this is a requirement for the replacement, not a claim about current production behavior.

The numerical-reference gate now passes at the tested larger budget.
Stage A remains incomplete until physical initial-state support, exact-output feasibility, and simulator/runtime closure are established.
No posterior estimator has been deployed to the acting agent.
Later stages are not implicitly complete because an offline interface exists.

## Next gate

Finish Stage A by defining candidate initialization and priors with valid geometric and attachment support, completing runtime artifact capture, and resolving exact-output feasibility.
Retain the explicit Bridge inconsistency control while constructing supported positive cases.
Explaining that full recording requires model revision or a separately declared discrepancy model; additional state uncertainty alone cannot resolve the invariant contradiction.
Complete the per-domain inventories before choosing physical sampling proposals.
Exact predicted features must reject contradictions, while continuous exact observations require a valid conditional representation rather than generic sampling followed by an equality check.
Distinguish failed feasible-candidate search from demonstrated model inconsistency, and identify conditioned inputs explicitly.
The numerical references and observation-channel checks have passed; retain them while adding physical-model validation.
The sampler must not silently interpret unsupported replay state or program mismatch as sensor noise.

Then compare fixed-prior batch inference and uncertain initial states with the incumbent on fixed development programs and recorded interactions from all five domains.
Include long prefixes, releases, contact transitions, incomplete programs, and held-out future predictions.
Planning, exploration, and execution estimation remain on the incumbent until their separate validation gates pass.
