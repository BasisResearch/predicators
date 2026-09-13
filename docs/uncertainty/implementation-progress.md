# Uncertainty simplification: implementation progress

Updated September 13, 2026.
This tracks implementation of the [simplification proposal](simplification-proposal.md).
The incumbent estimator remains the production default.

## Current stage

Stage 0 interface preservation is complete, with scripted behavior parity checked.
Stage A has implemented probability and replay components, but physical support and numerical validation remain incomplete.
The active work is Stage B offline comparison; Stage C live posterior use and Stage E retirement are not yet approved by the evidence.
Stage D execution smoothing remains optional and deferred.
The full plan remains incomplete.

The [Fan prefix comparison](fan-prefix-comparison.md) now has a completed compute support audit, `22674026`.
It removes the 68-action suffix before scene construction and scoring and does not reuse the full-recording proposal guide.
Eight of 48 sampled rest scenes have valid geometry and finite prefix likelihood, and the first supported trajectory repeats exactly.
All 16 median-scene probes instead have an initial penetration, ruling out that unmodified point-state baseline.
This prepares a comparison against the completed 64-action legacy fit while the existing full-recording fits continue unchanged.
The next prefix inference preflight, `22674192`, exactly matches 12 retained native targets and the complete initial population across serial and four-process execution.
It finds 11 supported initial particles with weight effective sample size 9.10 and verifies a roughly fourfold initialization speedup.
Two full-prior prefix fits are submitted as `22674242_0` and `_1`, with numerical seeds 302 and 303 and unchanged original support for moving and resting states.
These are offline inference replicas; completed prediction comparisons and numerical assessment remain pending.
The Fan reserved-suffix forecast path now passes complete-checkpoint recovery, per-particle prefix-factor equality and checks on all 64 saved complete histories using an explicitly unassessed short fixture.
It retains every positive-weight particle even though all fixture future densities are zero.
Both full Fan prefix fits and their forecast follow-ups have completed; the two populations retain one and two initial ancestors.
The paired Fan report now passes compute validation `22676293`, reconstructing all 64 fixture histories and rejecting six corrupted-output cases, including removal of zero future density.
The completed paired report gives lower position error than legacy on this recording, but a maximum 0.33212 geometric-goal probability disagreement and only one nonzero full-future-density particle per replica.
Corrected summary `22676681` also reconstructs and verifies the two parameter-quantile reports; the first summary omitted their Fan-specific field.
A 128-particle comparison is queued as array `22676775`, gated by native preflight `22676726` and validation finalizer `22676770`; no posterior is approved by the current evidence.
The separate [carried-center comparison](carried-center-comparison.md) submits four paired legacy tasks as `22674821`, covering Fan and Domino with carrying off/on.
Its 64/96/96/full prefix schedule tests both accumulating experience and refitting identical evidence while holding the other legacy mechanisms fixed.
It explicitly records whether carrying is ever active, rather than treating unchanged results with empty carried history as evidence about removing active carrying.
The Fan pair has now completed: all four stages retain speed 0.0846, report an anchored verdict and produce identical predictions across arms, including the repeated-data stage.
No accepted center was carried, so this is an inactive-policy control.
The Domino pair has also completed with identical predictions and no active carrying; active-policy coverage remains open.
A separate repeated-data pair now uses the earlier saved Domino program, whose declared friction is 0.3 instead of the already learned 0.674.
Historical training logs identify an accepted fit and later carrying at that earlier starting point; the new controlled reproduction has three identical 161-action fits and requires exact first-fit agreement between arms.
The first off-arm fit selects friction 0.6739569, then its second-fit guard detects that applying fitted values to the reused subclass reference also changes registry defaults.
Native audit `22676108` confirms that this can shift prior centers with explicit carrying off.
The corrected isolated pair `22676127` keeps an unfitted registry reference and uses fresh selected-parameter worlds for predictions; both arms must reproduce that first fit exactly before repeated-data comparisons.
Its paired summary `22676128` is queued; the interrupted original comparison remains separate and is not an agent outcome.
The original reused-reference carrying arm has completed: selected values and predictions stay identical, while friction width grows from 0.34494 to 0.49270 on the first repeated fit, then stays stable.
The [combined Domino assessment](domino-comparison-summary.md) reproduces prior diagnostics, verifies complete weighted histories and rejects dropped particles, altered samples and altered event aggregates.
Three finite follow-up jobs will produce comparison snapshots as the individual forecast pairs terminate; the new posterior comparisons remain incomplete.
Both mixed-proposal Domino fits have now completed all 32 stages, retaining two and one initial ancestors respectively.
Seed 101 places more than 97% empirical mass at a single value for each of four physical parameters, so its coincident central quantiles cannot establish precise identification.
Matched local controls are starting; numerical adequacy remains unresolved.
Those two forecasts have now completed: both toppling-agreement checks pass, but the 3.038 mm difference between position means fails the 2.5 mm screen.
Their descriptive reserved-action errors are lower than the legacy point forecast on this recording; matched local controls, budget sensitivity and numerical adequacy remain unresolved.
The local conditional-scale audit has completed with exact saved-reference replay.
It finds rough target changes with no uniformly successful smaller step scale; no new scale heuristic is selected.
The point-start fits and forecasts have also completed, passing all three exploratory agreement screens while showing worse position error and better toppling error than legacy on this recording.
Four [population-size comparison fits](domino-budget-sensitivity.md) are running at 128 particles, two numerical seeds for each state treatment, with unchanged priors, likelihoods and proposal rules.
The population-size report passes compute validation `22675912`, reproducing all four completed forecasts and 256 histories while rejecting six incomparable-input cases.
It records all twelve within-target replica pairs, including eight cross-budget pairs, and independently checks empirical parameter quantiles and repeated-value mass.
Its first snapshot retains ten incomplete pairs; finite follow-up summaries `22675954` and `22675955` depend on the larger forecasts.

## Recent evidence

The new offline [stochastic future integration component](stochastic-future-integration.md) retains exact joint/speed density factors and reports Monte Carlo concentration explicitly.
Forty functional tests and focused static checks pass.
Its first native Balloons diagnostic reproduces the reference trajectory but fails all four numerical comparisons: eight and 64 complete paths remain dominated by a single contribution.
This is an unresolved integration problem, so these scores are not used to compare estimators or change agent behavior.
A native factor audit reproduces ten selected paths exactly and attributes their output-score variation to the box and attached balloon positions; robot factors remain invariant.
The subsequent offline sequential integrator retains complete histories and block density normalizers, with explicit ancestry diagnostics.
Forty-four functional tests and focused static checks pass.
All four native sequential integrations completed with exact replay and factor accounting, but both independent-run comparisons fail the declared density-stability diagnostic and retain only one or two original ancestors.
The next offline proposal mixes original and position-guided velocity directions while retaining the full mixture correction, so it targets the same probability model.
Fifty-one functional tests and focused static checks pass; all four guided native pilots completed, but both whole-history density comparisons still fail the declared consistency diagnostic.
The two [Domino conditioned-base pilots](domino-joint-inference.md#completed-numerical-pilots-september-13) also completed, with strongly different parameter summaries and one original ancestor each; numerical adequacy remains unestablished.
The completed Domino population replay gives similar mean Cartesian errors (1.16 and 1.21 cm), but predicted toppling differs by as much as 0.90625 at the same frame.
The disagreement therefore affects goal-relevant predictions despite similar averaged feature errors.
Fan recovery jobs are active after confirmed allocation timeouts, and the two Balloons fits have been submitted for continuation from their saved complete-stage checkpoints under the same numerical budgets.

The completed matched Domino diagnostic now compares all three forecasts on the same 97-action suffix, with stored truth used only for evaluation after predictions were frozen.
Legacy has lower Cartesian error, while the new populations have slightly lower frame-averaged toppling Brier error and substantially different final toppling probabilities.
This does not establish a replacement advantage, and the new multi-hour fits retain an unresolved cost problem.
The next [ordered batch evaluation component](batch-evaluation.md) combines each conditional map and likelihood into an indivisible worker operation, allowing isolated processes to evaluate a mutation sweep concurrently.
Its scalar path retains the original random schedule, while batch mode uses a separate checkpoint identity and reserves its numerical budget before dispatch.
Thirty-two functional tests, sixteen exact comparisons against the original scalar implementation and final focused type/lint/format checks pass.
The native Domino check reproduces all target values and complete sampler output exactly across one and four processes, with a measured 3.76-fold sampler speedup; numerical adequacy remains unestablished.

The subsequent [conditional-parameter slices and mixed proposals](proposal-refresh.md) identify a concrete exploration issue: restitution changes leave the complete fitting history unchanged at two checked scenes, yet the two populations retain narrow, different restitution ranges.
A mixed local/full-range block proposal now preserves the same fixed conditional target while allowing larger numerical moves.
Forty-one functional tests, thirty-two disabled-refresh compatibility comparisons and final focused type/lint/format checks pass.
The native mixed-proposal check also matches target values and sampler output exactly between synchronous and four-process execution.
Both mixed-proposal Domino fits and forecasts have completed; matched local-only controls are running, and numerical adequacy remains unresolved.

The [initial-state ablation](initial-state-ablation.md) now supplies a first-observation-only point-start comparison while retaining the original parameter prior and output model.
Its selected scene is feasible, repeats exactly and yields finite full-prefix likelihoods at 39 of 64 random parameter settings.
Two parameter-only fits have completed; this is an explicitly labeled approximation, not an exact state observation.
The [checkpoint-driven forecast follow-up](checkpoint-forecasts.md) now reproduces every complete history and aggregate metric from the earlier 64-row forecast exactly.
Six follow-up jobs are submitted with dependencies on successful completion of their individual source fits; the mixed-proposal and fixed-initial-state pairs are complete, while the local comparison remains pending.

The latest [likelihood cost reduction](likelihood-cost.md) preserves all 2,560 archived orientation densities and five complete Fan likelihoods exactly on the checked runtimes.
It removes array reductions from two-term quadrature sums, making the measured density evaluations about four times faster while retaining the statistical model and numerical acceptance checks.
Twenty-three functional tests and focused type/lint/format checks pass; running fits retain their existing frozen source.

The offline sampler now has optional [continuation checkpoints](sampler-checkpoints.md) for long fits on preemptable nodes.
They preserve the complete weighted population, density factors, random state and diagnostics at stage boundaries, while keeping unfinished solver state separate from an assessed posterior.
Changed inference inputs or sampler settings reject a resume, and the cumulative numerical budget remains fixed.
The checkpoint restores numerical inference state; candidate simulations still reconstruct their full validated action prefix.
Compute validation passed forty functional tests, three-file type/lint/format checks, and thirty-two exact paired comparisons with the pre-change sampler.

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

The offline `assess_inference` boundary now separates numerical availability from predictive checks.
Its identified assessment protocol requires an explicit set of numerical checks; omitted checks remain unevaluated rather than becoming implicit passes.
Completed sampler output is exposed as a posterior only after those checks pass and its sample structure and normalized weights are valid.
Predictive failures remain attached to an available posterior, while sampler failures, failed numerical checks, and missing checks expose no usable posterior through this boundary.
The boundary does not publish a canonical fit, retain an older fit, approve an action, or establish that a caller's chosen assessment protocol is scientifically sufficient.
Production integration and the remaining physical inference gates are still pending.
Compute job `22632593` passed 21 functional tests, two-file mypy and lint, and pinned formatting checks for this boundary.
The first check attempt failed because its new test fixture requested zero sampler moves, which the sampler correctly rejects; the corrected frozen fixture and final check artifacts are in `logs/uncertainty_assessment_v2_20260912`.

The offline feasible-scene adapter now connects support checks to the conditional batch sampler without discarding exact-observation or proposal-density factors.
It explicitly distinguishes globally conditioning the entire parameter/state prior from normalizing each conditional state law while preserving the parameter prior.
The latter requires the original support probability as a declared function of its retained variables, before conditioning on observations.
This closes an integration gap between the generated scene sampler and the numerical posterior reference; it does not supply the still-missing historical scene laws or their normalizers.
See [scene-prior composition](scene-prior-composition.md#connecting-feasible-scenes-to-parameter-inference) for the equations and applicability limits.
Compute validation passed 22 functional tests plus focused static and formatting checks.
Both support-normalization references passed 8/8 trials at 2,048 particles; the smaller 512-particle budget passed 15/16, with the failed trial retained.

The [Balloons initial-scene reference](balloons-initial-scene.md) now combines a declared scene law, exact initial conditioning, noisy-position conditioning, full-candidate collision checks, and fresh-world replay of an actual frozen learned program.
All sixteen accepted root samples satisfy the declared support and exact initial observations, and each has identical repeated 16-action predictions.
All sixteen still contradict a later exact output at the first action; a separately tested supported-rest configuration substantially reduces the box-speed discrepancy.
This advances the physical initial-state gate and identifies a concrete trajectory constraint; it is not a complete recording posterior or deployment acceptance.
The Gaussian-coordinate conditioning implementation used by the reference passed seventeen functional tests, focused type/lint checks, and pinned formatters on compute nodes.

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

The [explicit transition-discrepancy reference](transition-discrepancy.md) now implements analytic conditioning of a rest/Gaussian velocity transition on exact speed.
It retains the noncentral radial density and directional uncertainty instead of inflating sensor variance or projecting without a likelihood factor.
This is a separately declared stochastic model extension, motivated by the unresolved deterministic contact constraints.
The recorded Balloons diagnostic tests conditional prefixes and unconditioned future continuations; it is not a posterior comparison or agent result.
Seventeen functional tests, focused type/lint checks and pinned formatting pass for the transition component.
The accompanying native-link cache audit explains why resetting unchanged joints introduced artificial Cartesian residuals into the first diagnostic.
Preserving the native predicted link-observation phase removes those early residuals without using observed Cartesian values.
All twelve corrected diagnostic paths repeat exactly, but later robot/contact constraints still fail at actions 18 or 22, so they do not supply a complete 32-action posterior.

A separate [marginalized output-discrepancy extension](output-discrepancy.md) integrates a declared Gaussian error history without modifying physical simulator states or sensor variance.
Joint inference of velocity, uncertain initial position and error scale agrees with an independent dense grid in four of four larger-budget numerical trials; a smaller-budget prior-marginal failure remains recorded.
The five-domain diagnostic compares persistent and independent error on selected real-valued channels with fixed native predictions and a suffix held out from discrepancy fitting.
It does not yet supply a full-recording posterior, an uncertain-scene comparison or a replacement for the legacy fitter.

The [exact-readout boundary](observation-reductions.md) now verifies deterministic readouts of exactly observed source fields before reducing an observation view.
It preserves source likelihoods, refuses noisy or missing sources, and returns no reduced view for contradictions.
The runtime-derived finger map matches all 1,983 public frames across the five development domains; the integrated reducer also rejects all 1,983 perturbed-readout controls.
This removes a redundant finger constraint after verification, without relaxing the underlying joint observation or changing production observations.

The subsequent [native Euler support audit](observation-reductions.md#native-euler-readout-support) rejects a proposed canonical-angle assumption.
Of 1,983 recorded robot orientations, 1,310 have pitch exactly positive pi/2 and zero roll, including 31 whose native yaw lies outside [-pi, pi].
Direct conversion probes reproduce collapse to the pole across a nonzero range of requested pitch values.
The audit preserves those readings as valid evidence and leaves their coupled likelihood unresolved; independent continuous angle densities or silent wrapping are not a validated replacement.

The next [coupled quaternion-output model](orientation-discrepancy.md) integrates an explicit Gaussian-mixture readout discrepancy through the native Euler map.
Independent analytic and sampled references check ordinary densities, pole masses and the native yaw branches.
The corrected implementation completes all 2,560 recorded density evaluations across five domains, four scales and two numerical tolerances; the largest tolerance change is below 8e-13 per reading.
The full observation composition then accounts for every output field in the existing 64-action conditional forecasts.
Fan and Domino have finite complete-output likelihoods under the declared extension, while Bridge glue, Boil switch/faucet and Balloons speed discrepancies still force zero likelihood for their supplied forecasts.
These positive cases enable the next joint inference experiment; they do not establish an uncertain physical initial-state posterior, legacy-fit comparison or production readiness.
Final compute checks passed 27 functional tests, four-file mypy and lint, and pinned formatting; the complete-output integration accounts for 24,896 measured fields across its five conditional forecast windows.

The next [Domino joint-inference pilot](domino-joint-inference.md) specifies an uncertain physical scene under an explicit unheld-case prior and includes the initial observation in the complete likelihood.
Eight generated candidates passed geometric support and exact repeated 64-action replay; two had finite complete-output likelihood.
The sampler now supports optional disjoint proposal blocks while preserving the existing full-vector default.
These developments enable a physical integration experiment, but numerical adequacy, complete runtime capture, all-domain prior closure and the legacy-fit comparison remain open.
Both small-budget Domino pilots completed but collapsed to one initial ancestor, retaining only one and three distinct physical parameter vectors and disagreeing across independent runs.
They do not provide usable parameter uncertainty; the next numerical experiment must address first-temperature concentration and parameter movement before comparison with the incumbent.

The [sampling reproducibility follow-up](sampling-reproducibility.md) identified unordered candidate initialization and cross-node quantile differences that also confounded the initial pilots.
Canonical object ordering with identical saved physical candidates reproduced all 31 candidate trajectories and likelihoods across three CPU types and two hash seeds.
New pilots pin the candidate-generation runtime and factor the initial observation into the conditioned base before tempering the remaining trajectory, preserving the complete target.
The explicit-schedule sampler passed 19 functional tests, type/lint/format checks and eight exact default-parity comparisons.
The new physical fits remain experimental until independent agreement, budget sensitivity and prediction checks establish usable inference.

The [full legacy comparison setup](offline-fitter-comparison.md) now consumes the same reconstructed noisy public frames through a checked observation-to-state adapter.
All 295 Domino and Fan training frames round-trip exactly; seven functional tests and focused type/lint/format checks pass.
Four cold-fit comparisons are submitted for the two 64-action windows and the two complete training recordings, using the actual legacy preparation and orchestration pipeline.
The Fan audit separates its earliest broad support from later data-derived bounds and declares a fixed uniform prior for future posterior experiments without changing the latest program dynamics.
These are comparison inputs and submitted experiments, not completed Stage B evidence or a production replacement.

## Next gate

The [parameter consumer boundary](parameter-consumers.md) now derives parameter quantiles and weighted or resampled ensembles from the same assessed joint approximation.
It preserves parameter dependence and predictive diagnostics, requires explicit coordinate/name mapping, and refuses numerical results that are unavailable or unevaluated.
This prepares an offline interface for later saved-decision comparisons; it does not make the current physical posterior pilots adequate or route the acting agent through a new estimator.
Nineteen functional tests, focused type/lint checks and pinned formatters pass for this consumer boundary.
The same ensemble now supplies its weights to the incumbent per-atom information criterion through `atom_information`, with explicit observation-channel probabilities and no acting-agent routing change.
This extension passed 61 functional tests and focused static/format checks on a compute node; 768 comparisons preserve the default scoring path exactly.
The physical inference and saved-decision comparison gates remain open.
The [conditional forecast interface](conditional-forecasts.md) now draws complete future observation histories from the fitted output model, retaining scalar temporal dependence, coupled Euler outputs and checked readouts.
Its native Domino check repeats both complete 161-action histories exactly, rejects the incompatible prefix and generates all 70 fields for the supported candidate's 97-action suffix without future-observation conditioning.
This is a physical integration check on two fixed candidates; neither is an assessed posterior sample.
The same output model now provides a causal joint future-likelihood score, preserving temporal error dependence while accumulating only future observation factors.
It distinguishes an unsupported fitting prefix from a supported prefix followed by a zero-likelihood future, and avoids cancellation from subtracting large full-history scores.
Forty-three functional tests pass, including independent conditional Gaussian references, and the native scoring check preserves all 33 archived complete-history scores exactly.
The separate `JointForecast` adapter now constructs a deterministic physical-history mixture from assessed joint rows, preserving initial-state dependence, posterior mass and predictive diagnostics.
It verifies the fitting ledger and output-model identity, rejects lost or unsupported positive-weight histories, and keeps one source particle fixed across an entire sampled future.
Forty-seven functional tests and four-file static/format checks pass.
An end-to-end linear Gaussian reference passes all declared prediction checks on both 2,048-particle runs and one of two 256-particle runs; the failed smaller-budget density check is retained despite passing parameter moments.
Real-domain numerical adequacy, stochastic Balloons future integration and the matched estimator comparison remain open.
The Balloons generation audit now reproduces its original full conditional path factor and eight repeated, distinct 32-action futures from a fixed 64-action prefix, with future readings removed from the generator's lookup table.
All 58 observation fields are generated under the physical transition and output laws, and the extracted velocity sampler preserves the previous random stream exactly.
Twenty-five component/conditioning tests and focused static/format checks pass; the native audit performs 1,963 actions.
This is a selected-witness integration check, not a posterior forecast; the exact-speed density still requires conditional integration rather than finite-path equality checks.
The four full legacy comparison tasks have also completed, providing saved predictions for both 64-action fits and complete Domino/Fan recordings.
Both domains retain their anchor parameters for prediction under the incumbent policy; a matched replacement posterior is still unavailable.
The [Boil incomplete-model control](boil-incomplete-control.md) now separates missing filling/heating dynamics from sensor noise using a scalar likelihood bound and a causal suffix calculation.
Compute job `22650461` completed the source/data audit and independent numerical references.
The original Boil/Bridge control array `22650411` was cancelled before starting because its worker bypassed the parameter-free public fit dispatch.
Replacement array `22651160` completed all four tasks, verifying the actual no-fit return and declared-dynamics predictions.
The paired Boil and Bridge runs respectively replayed 264 and 1,186 actions with exactly identical predictions and zero fitting evaluations.
They retain the missing filling/heating and exact glue-transition failures; these are offline parameter-free controls, not fits or agent seeds.

The [full Fan initial-scene experiment](fan-initial-scene.md) now declares the remaining placement and geometry components, retaining all articulated and robot uncertainty under a normalized original law and explicit whole-scene support policy.
The corrected proposal preserves support for every quarter-turn case and records actual worker hardware instead of inherited CPU labels.
Eight scenes pass geometry and repeated 32-action replay; seven have finite complete-output likelihood on that prefix.
All eight earlier saved scenes still fail later exact events over the complete 132-action recording, despite exact repeatability.
Directed parameter and fixture-placement probes are investigating that conditional support; they are not posterior estimates or evidence of an improvement over legacy fitting.
Those probes have now found a complete-recording Fan support witness by adjusting a sampled switch placement within the original prior at speed 0.09.
Independent full replays verify that point and three nearby positive perturbations; a fourth perturbation fails, preserving the sensitivity evidence.
This opens a supported full-recording inference experiment while leaving numerical adequacy and comparison gates unresolved.
The subsequent [Fan joint-inference integration](fan-joint-inference.md) supplies a fixed coordinate map covering all declared cases and a broad/local mixture proposal with full density corrections.
The mapped witness reproduces its full likelihood exactly, and native proposal audits find finite targets in both local components while retaining broad-component failures.
Two full-recording Fan sampler pilots are submitted alongside the existing Domino runs; none is automatically treated as numerically adequate.

The [Balloons transition/output composition](balloons-composed-inference.md) now accounts for the exact-speed transition density and all remaining outputs in one conditional path factor.
Six paths from one sampled root have finite factors over the complete 235-action training episode and repeat exactly; six from the other root retain their exact tie/clip failures.
This supplies complete-recording support for the explicit stochastic extension, while the deterministic reference, original-prior joint inference and predictive-adequacy questions remain separate.
The subsequent native sphere audit justifies integrating out nine initial orientation coordinates for this fixed program while retaining angular velocities.
A complete joint coordinate map now covers the ten parameters, all sixteen initial motion cases and every conditional velocity direction.
All 24 mapped test cases preserve initial exact observations and repeat full trajectories; seventeen have finite full-recording factors and seven retain event contradictions.
A density-corrected proposal that accounts for table clearance passes independent normalization checks and supplies twelve finite local candidates.
Two full joint Balloons inference pilots are submitted with stage checkpoints; posterior adequacy and deployment remain unevaluated.

The [articulated replay correction](articulated-replay.md) closes an omission exposed by the Fan prior work: snapshots previously lost all four slider and twenty rotor joint states.
The corrected offline snapshot restores those states exactly and repeats a 64-action native Fan trajectory at every recorded boundary.
Twenty-two functional replay tests and focused static/format checks pass.
This does not establish arbitrary engine checkpoint portability or complete the Fan scene prior.

The [Fan articulated-state component](fan-articulated-prior.md) now conditions a normalized rest/motion law on exact switch flags while retaining the event probability in parameter inference.
Fifteen functional tests and focused static/format checks pass, and the corrected compute audit verifies all 2,048 native state readbacks plus the actual enforced slider cap.
The inventory also identifies twenty collision-bearing rotor joints and now declares a uniform position/velocity component for their initial state.
Eight sampled rotor/switch candidates restore exactly and repeat sixteen recorded actions, adding an explicit forty-dimensional rotor component while leaving scene layout and contact support open.
This closes a switch component, not Stage A or the full Fan scene model.

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
