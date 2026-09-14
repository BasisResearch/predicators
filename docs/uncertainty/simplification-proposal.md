# Proposal: simplify uncertainty handling in EMPIRIC

September 11, 2026.
Revised September 12, 2026 to clarify exact conditioning, inference availability, and the optional execution-filter extension.
Implementation decisions confirmed September 14, 2026: replay from candidate initialization is the required reference, and limited simulator discrepancy is an explicitly evaluated extension after replay and program errors are diagnosed.
This is a design proposal, not a description of an implemented or validated replacement.
The companion [implementation explanation](explained.md) documents the current behavior and source locations.
Implementation of the staged migration is tracked in [implementation progress](implementation-progress.md).

## Recommendation

Unify the uncertainty interface first, then replace its estimators one at a time.
The first complete endpoint is:

1. A posterior over the fixed parameters of the current simulator program.
2. Joint inference of uncertain episode initial states during fitting, so parameter uncertainty accounts for uncertainty about how recordings began.
3. Planning, subgoal checking, and information seeking that obtain parameter samples from this same posterior, while retaining the existing execution state estimator.

A conditional execution filter and conditional state sampling for planning are separately evaluated extensions.
Keeping the existing observation estimator is a valid final choice if those extensions do not improve results.
This endpoint unifies parameter uncertainty; it does not claim a full joint Bayesian belief during execution.

Start by preserving the current agent behind a versioned result interface, without changing its estimates, reports, or decisions.
Verify candidate initialization and recorded-action replay before implementing parameter inference with uncertain initial states.
Evaluate that inference offline using a fixed prior and all available fitting recordings, with intervals and parameter ensembles derived from the same posterior approximation.
Introduce it into planning only after it passes prediction checks, while preserving the existing execution estimator and decision rules.
Replace execution state estimation last, if it independently demonstrates a benefit.
Keep the simulator subclass interface and the agent's ability to choose its tools and actions.

The design separates the posterior implied by the model and data, the reliability of its numerical approximation, the model's predictive adequacy, and the suitability of a particular action.
Return a numerically adequate posterior even when predictive diagnostics reveal an incomplete program.
Predictive failures must remain visible to model revision and decision rules; returning or publishing the inference result does not certify an action.
One posterior does not require a single decision criterion or a single numerical algorithm.
The intended simplification is to remove duplicated statistical assumptions while retaining explicit robustness decisions.
Preserving performance is an experimental requirement, not something we can infer from a cleaner formulation.

The reference agent is the frozen noisy-sweep runtime tagged `noisy-mb-five-domain-15of15-20260910`, which achieved whole-run success on three seeds in each of five domains.
Those development results establish a working baseline, not a guarantee or an ablation of individual safeguards.
See the [recorded results](../uncertainty-results/noisy-sweep-table.md).
The current implementation remains the production default until a replacement passes the gates below.

## 1. Establish the state and replay contract

Define how an inference candidate becomes a runnable simulator state before defining its sampler.
This contract must cover object pose and velocity, robot state, attachments and constraints, and simulator-subclass memory.

| Quantity | Contract |
| --- | --- |
| Exact observations and known reset values | Condition on them using only information exposed by the task interface. |
| Noisy observations | Evaluate the declared observation likelihood; do not treat a noisy pose as an exact initial state. |
| Unobserved initial quantities | Specify a prior and a representation that supports valid sampling and restoration. |
| Inferred model memory | Keep it candidate-specific and preserve it across rest windows and rollout branches. |
| Engine implementation state | Reconstruct reproducibly; measure any remaining replay discrepancy rather than attributing it to sensor noise. |

Feasible samples must respect object geometry, attachment consistency, and the declared support of discrete and continuous state.
Document whether each exact feature is an exogenous conditioned input or an output predicted by the model.
A candidate that contradicts an exact predicted observation has zero likelihood; silently overwriting that prediction would define a different model.
This rejection rule alone is not a sampling method for continuous exact observations.

| Exact quantity | Required treatment |
| --- | --- |
| Known reset value or declared external input | Set it directly from the task interface and remove it from the sampled coordinates. |
| Exactly observed continuous initial coordinate | Fix the coordinate to its observation and derive the conditional distribution of the remaining unknowns, retaining any information it supplies about parameters or other state. |
| Exactly observed continuous trajectory output | Use an explicit constrained representation or conditional proposal that reaches the observation-consistent states and preserves the induced conditional distribution. |
| Exactly observed discrete output, such as a switch state | Use an indicator likelihood; enumeration or proposals within compatible discrete cases may be needed when rejection is inefficient. |

Sampling a continuous joint position and hoping for exact equality with its observed value almost surely fails under a continuous proposal, even when the conditional distribution is well defined.
Eliminating a coordinate or solving a constraint must preserve the appropriate density factors, including Jacobian factors where required; projecting arbitrary samples onto a constraint surface is not generally sufficient.
For smooth models, [Graham and Storkey (2017)](https://proceedings.mlr.press/v54/graham17a.html) describe conditioning on the set of inputs consistent with observed outputs; their smoothness assumptions do not establish a sampler for this repository's contact dynamics.
Report separately whether the conditional representation is unsupported, the finite search failed to find feasible candidates, or the model's constraints are demonstrably inconsistent.
None permits manufacturing a normalized posterior, and failure to find a feasible candidate is not proof that none exists.

Keep numerical replay error distinct from sensor uncertainty.
Document any numerical constraint-solver tolerance and test its effect on the conditional approximation; do not silently turn an exact observation into a tolerance-band likelihood.
An explicit observation-resolution or dynamics-discrepancy model is a separately justified model change.
The [September 12 prediction preflight](experiments-20260912.md#fixed-program-prediction-preflight) found exact-output contradictions in all 14 executable nominal cases, including very small joint discrepancies and larger errors.
These results motivate the support and replay investigation; they do not establish that every feasible initial state and parameter is inconsistent.
The later [Bridge invariant audit](experiments-20260912.md#a-structural-exact-output-contradiction-in-the-frozen-bridge-model) establishes a narrower structural impossibility: the frozen no-op program keeps glue attributes constant while four exact recorded glue attributes change.
Retain that full-recording target as an explicit model-inconsistency control, rather than attempting to repair it with a larger initial-state prior or sampling budget.
Explaining those transitions requires program revision or a separately evaluated discrepancy model.

The current [fitting replay](../../predicators/code_sim_learning/rollout_env.py), `rollout_states`, zeros velocities after restoring its initial state.
An inference path that samples initial velocities must restore and retain those velocities; it cannot inherit that rest-start assumption unchanged.
The legacy path keeps its current behavior for comparison.

Validate repeated replay, moving starts, contact transitions, attachment changes, and memory continuity on short and long recorded prefixes.
Use known dynamics and evaluator-only state to diagnose reconstruction error offline, without exposing that information to the agent.
Separate reconstruction failures from candidate-program errors before judging a statistical estimator.

The offline continuation reference uses an explicit candidate initialization protocol followed by the full action prefix in one fresh world.
This preserves engine history, native attachments, and model memory without requiring an exact portable checkpoint at every observation boundary.
Every change to the candidate parameters or initial state reconstructs the prefix under that candidate; the extra simulator steps count toward its compute budget.
Initialization is part of the probability model and runtime identity, not an implicit call to the evaluator's task generator.
Only evaluator-only mechanical audits may use evaluator reset state or private dynamics to establish a replay reference.

A portable candidate must include full body orientations, original command-weld frames, and commands queued for the next action, even when these quantities are absent from public observations.
Their values must come from the candidate prior or simulated history, not privileged recording metadata.
Arbitrary mid-trajectory restoration remains an approximation until separately validated against uninterrupted replay.
It is not required for this migration and must not block inference or planning integration that uses full-prefix replay.
Without a validated checkpoint, evaluate each alternative future by reconstructing its candidate and replaying the recorded low-level actions from initialization.
For a stochastic discrepancy model, preserve or explicitly resample the candidate's latent history under the declared conditional law; an unrelated random replay does not reconstruct the same candidate.
Checkpointing and prefix caching are optional optimizations whose identity must include the program, parameters, initial state, actions, and any latent random history.
Numerically repeatable candidate replay, faithful evaluator reconstruction, and predictive accuracy of a learned program are three distinct acceptance claims.

### Required initial-state inventory

Before defining real-domain sampling proposals, complete a separate inventory for boil, domino, fan, bridge, and the selected balloons variant, tied to the program, task-interface, and recording versions used in the comparison.
Each inventory must use the following columns and cover object pose and motion, controlled and passive robot joints, attachments, and every declared hidden-memory quantity.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| --- | --- | --- | --- | --- | --- |
| One row per state quantity or coupled group | Exact value, noisy reading, reset guarantee, or no observation, with a source reference | Unknown values and dependencies across quantities or episodes | Density or mass function, bounds, geometric and attachment constraints, and memory initialization | Fixed input, conditioned coordinate, derived quantity, analytic integration, or sampled variable | Size after these reductions, for the actual recording set |

This is a required design artifact, not a claim that physical priors have already been established.
The [September 12 inventory](initial-state-inventory.md) records the five development schemas and visible-model joint audit, with unresolved priors and dimensions explicitly marked.
A value present in engine metadata is not thereby known to the agent; in particular, do not infer passive-joint values, zero velocity, or absent attachments from recording omissions.
Do not assume URDF joint-limit intervals are hard support for every simulator initialization: the [joint audit](robot-state-prior.md) found a recorded balloons initial angle outside its URDF interval.
Retain that prior-support contradiction and specify a justified initialization law before fitting; clipping the exact reading or silently changing bounds would change the inference problem.
Distinguish actual resets from continued trajectories, and document which hidden quantities persist across task changes.
Start with the smallest valid uncertain representation and expand it only for quantities that cannot be conditioned on, derived, or integrated out under the declared model.
Report the resulting joint dimension and discrete alternatives across all episodes before selecting proposal blocks and compute budgets.

## 2. Define one inference problem

Let $P$ be the current simulator program, $\theta$ its fixed dynamics parameters, and $s_t$ the full physical and model state.
The state includes velocities and accumulated quantities such as heat or curing state when they affect future dynamics.
Let $o_t$ be the noisy object-centric observation and $a_t$ the executed action.

For the first offline reference, use deterministic candidate dynamics and the declared observation channel:

$$
s_{t+1}=F_{P,\theta}(s_t,a_t),
\qquad o_t\sim p(o_t\mid s_t).
$$

For reset episodes indexed by $e$, fit the joint posterior

$$
p(\theta,\{s_{e,0}\}_e\mid D,P)
\propto
p_0(\theta\mid P)
\prod_e \left[
p_0(s_{e,0}\mid\theta,P)
\prod_{t=0}^{T_e}
p(o_{e,t}\mid s_{e,t}(\theta,s_{e,0},a_{e,0:t-1}))
\right].
$$

This is the target distribution, independent of the numerical algorithm used to approximate it.
The likelihood notation includes deterministic constraints; continuous exact outputs require the conditional construction in section 1 rather than ordinary density multiplication and rejection in the original coordinates.
The numerical target must specify its free coordinates and the density or mass on that representation.
The parameter posterior is its marginal over initial states.
An uncertain initial pose is inferred together with the dynamics rather than fixed to one noisy measurement.
Every observation enters once; repeated calls to `observe()` at the same environment step do not constitute new evidence.
A new task without an actual state reset must preserve the state transition history rather than be treated as an independent episode.

Use the measured or declared observation variances, with the correct treatment of angles and feature types.
Exact observed components are constraints or conditioned inputs, not Gaussian observations with an arbitrary tiny variance.
Initial-state priors must describe information available to the agent, not privileged simulator state.
Set velocities or hidden memory to known reset values only when the task interface actually establishes those values.
Other initial quantities remain uncertain.
If the first observation is used to construct an informative sampling proposal for the initial state, include the proposal correction rather than counting it as both a prior and a likelihood factor.

Keep the original parameter prior fixed while fitting pooled recordings.
Previously fitted values may initialize optimization or sampling, but do not become a new prior center.
A parameter retains its prior marginal when neither the likelihood nor dependence on informed parameters supplies information about it.
There is no need to classify it as sufficiently identified before allowing it to be represented in planning.

### Missing dynamics remain a separate problem

A posterior can be narrow and wrong when the program is wrong.
Check whether posterior predictions explain the observations, and report persistent discrepancies to the agent so it can revise its simulator.
Do not inflate sensor noise or silently delete difficult trajectories until the current program looks accurate.

The initial offline comparison should use the declared sensor likelihood directly, so its limitations are visible.
This is a reference model, not an assumption that sensor noise alone is sufficient for deployment.
The existing robust loss, segmentation, and trimming also protect against replay and program error.
They remain active in the legacy agent until a replacement demonstrates comparable behavior on incomplete programs as well as accurate ones.
If replay mismatch requires a discrepancy model, make that an explicit, separately evaluated extension with its own parameters and prior.
For example, a model of occasional bad measurements addresses a different failure from temporally correlated errors caused by missing forces.
Neither should be introduced merely to reproduce the old trimming decisions.

Keep sensor variance fixed at its declared value in these comparisons.
Evaluate any transition-discrepancy model on held-out development interactions and event predictions, so it cannot earn acceptance merely by explaining every trajectory with extra flexibility.
For a stochastic discrepancy extension, infer the intermediate states under its declared transition distribution; the deterministic initial-state formula above no longer suffices.
Persistent mismatch should still inform program revision.

### Discrepancy implementation decision

Diagnose reconstruction bugs, parameter error, missing program logic, and residual approximation error separately before extending the probability model.
Fix reproducible initialization or replay defects directly.
Expose missing mechanisms and incompatible exact predictions to simulator-program revision; a discrepancy term must not silently excuse them.
Lack of informative observations, such as an all-burners-off fitting prefix, is a separate cause of broad uncertainty and does not by itself justify discrepancy.

A limited discrepancy extension may proceed when a repeatable residual pattern remains after those checks.
Declare the affected quantities, temporal law, fixed hyperparameters or original hyperprior, and physical or output-level interpretation before running the comparison.
An output discrepancy changes the distribution of observations around a simulated trajectory; a transition discrepancy changes physical histories and therefore can change contacts and events.
These are distinct model changes and must not be substituted for one another.
Keep declared sensor noise unchanged and preserve exact observations through the appropriate conditional construction.

Use the same discrepancy law in fitting, future generation, and future-density evaluation.
Select a law using fitting or designated development data, freeze it, and evaluate causal predictions on a separate suffix or recording.
If a previously held-out suffix motivates a new law, it becomes development evidence and a new untouched evaluation is required for acceptance.
Compare with the model without that extension where its conditional target is supported, retaining unsupported cases explicitly rather than manufacturing a posterior.
Report numerical repeatability, prediction error, consequential event probabilities, and compute cost; better training likelihood alone is insufficient.
Keep diagnostic interventions that suppress future noise separate from a consistently refitted model.

The production implementation should not accumulate domain-specific corrections selected to make these recordings pass.
Any domain knowledge needed for a prior or model must have an explicit source available to the agent through the task interface or learned simulator.
Retain the existing execution estimator and decision rules during this evaluation.

## 3. Standardize the inference result, evaluate the approximation

Evaluate a batch sampler over joint dynamics parameters and uncertain episode initial states as the first candidate implementation.
Use the completed domain inventories to define the target and proposals before choosing a real-domain sampler configuration.
Tempered sequential Monte Carlo with derivative-free Metropolis moves is a reasonable candidate because it does not require a reliable local contact Jacobian.
Its representation can express bounded, correlated, or multimodal parameter uncertainty; actually discovering that uncertainty requires adequate exploration.
Sequential Monte Carlo samplers support weighted approximations of distributions known up to normalization ([Del Moral, Doucet, and Jasra, 2006](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)).
That reference also emphasizes dependence on the target sequence and proposal distributions; avoiding derivatives does not solve exploration of narrow feasible regions or disconnected explanations.
The following adaptation is a proposal for this repository, not a result established by that reference.

A candidate batch fit would:

1. Snapshot the simulator program, observation model, priors, and complete fitting data.
2. Initialize joint candidates on the required exact-constraint support, from the correctly conditioned base distribution or a proposal with a known density and the appropriate importance correction.
3. Replay each recorded action sequence under each candidate, preserving model memory, and calculate the observation log likelihood.
4. Move from the base distribution toward the posterior by gradually increasing the remaining noisy-observation likelihood exponent from zero to one, maintaining exact constraints throughout.
5. Reweight, resample when weights concentrate, and apply Metropolis moves that target the current tempered distribution.
6. Return weighted samples and marginal quantiles when numerical inference succeeds, alongside predictive diagnostics and a separate record of publication and decision use.

Tempering an equality indicator cannot gradually move unconstrained continuous candidates onto a zero-volume constraint surface: its value remains zero off that surface for every positive exponent.
Exact conditioning must be handled in the base distribution and valid moves, not deferred to the tempering schedule.

Previous fits and optimizer results can help construct proposals, but are not automatically posterior samples for the new dataset.
A finite collection of perturbations around a MAP estimate is also not a posterior unless its distribution and weights justify that interpretation.
Grid integration on small synthetic problems should serve as a numerical reference for testing the sampler.
Do not commit to SMC as the sole production algorithm before these comparisons.
Whichever approximation is adopted should supply both credible intervals and planning samples; a coordinate sweep must not independently redefine its uncertainty.
Separate stress tests may still explore failure boundaries without claiming to estimate posterior mass.

Use block proposals for shared parameters and individual episode initial states, while evaluating the appropriate joint target.
Retain fresh simulator instances and correct replay initialization.
Keep all values within their declared support through a valid proposal or acceptance rule; clipping samples after generation would change the distribution.

Report effective sample size, likelihood-evaluation count, repeatability across independent sampler runs, and prediction stability as the budget increases.
Effective sample size alone cannot detect an undiscovered mode.
If numerical inference is inadequate, report that limitation rather than returning artificial zero-width intervals.
A large initial-state space or long chaotic recording may make this approach expensive or poorly mixing; the reference experiment must measure that rather than assume it away.

### What `sim.fit()` returns

A single result object should contain:

| Field | Purpose |
| --- | --- |
| Program, prior, observation-model, and data identifiers | Establish exactly what distribution was fitted. |
| Estimator identity, numerical settings, and observation-prefix identity | Make the approximation and its information boundary reproducible. |
| Joint samples and normalized weights | Supply one source for parameter and state uncertainty. |
| Point summary and marginal credible intervals | Provide readable summaries without a separate width estimator. |
| Posterior predictive diagnostics | Show discrepancies on specific recorded features and time intervals. |
| Numerical diagnostics | Distinguish an unreliable approximation from uncertainty supported by the model and data. |
| Inference availability, publication identity, and decision-use record | Distinguish a returned candidate, the current canonical fit, and its use or refusal for a particular decision. |

The initial legacy adapter preserves existing point estimates, heuristic widths, reports, and publication rules exactly.
Label those widths as legacy estimates in internal metadata; do not invent posterior samples or claim calibration that the adapter does not establish.
Keep user-visible tool output unchanged during this interface-only step because changes to the agent's input can change its behavior.

A point estimate remains useful for deterministic debugging and nominal rollouts.
A mean parameter vector can fall between incompatible modes, so nominal execution should use an explicitly selected representative candidate rather than assume every posterior mean describes plausible dynamics.

Keep canonical `sim.fit()` as the operation that publishes a new parameter posterior.
Subset fits return their results and diagnostics without replacing the canonical fit.
For the posterior path, a successful numerical fit returns its approximation even if predictive checks fail; a canonical fit publishes that result together with the failures.
Publication identifies the current inference under the declared model, not a blanket approval for execution.
Predictive diagnostics remain feature-, time-, and event-specific so an incomplete program can still supply useful diagnostic information without being declared universally reliable.
Initially preserve the existing action-acceptance and risk rules; any new diagnostic-dependent refusal or acceptance rule is a separately evaluated policy change.
The development checks for adopting the replacement estimator remain required, but must not be conflated with withholding each inadequately predictive fit from the agent.
If numerical inference fails or no conditional posterior can be constructed, return the failure diagnostics without claiming posterior samples.
Retain a previously published estimate only if its program and parameter meaning remain compatible, and explicitly report its age, failed checks, and limitations.
Retention is a continuity choice, not evidence that the older estimate predicts the new data better.
Report its predictive discrepancies on the available new observations, or explicitly mark that comparison unevaluated; never multiply the old samples by reused data merely to perform this check.
An incompatible program edit leaves no valid published posterior; it must not silently inherit old weights or memory.
Preserve the existing agent-controlled interaction flow so an unavailable posterior does not impose an automatic environment probe, reset, or indefinite refit loop.
During migration, rollback to the legacy agent is an explicit, recorded experiment choice, not a silent per-fit estimator switch.
A program edit invalidates old likelihoods and inferred memory; a refit evaluates recordings under the new program and its declared parameter space.
Do not silently carry posterior weights across changes in program meaning or parameter definitions.

## 4. Optional extension: conditional execution state estimation

This section specifies an extension beyond the first complete endpoint.
Retaining the existing observation estimator does not leave the parameter-uncertainty simplification unfinished.
Attempt this extension only as an independently evaluated change, retaining its additional inference cost and recovery behavior in the comparison.

For each retained parameter candidate, maintain a conditional state belief.
Propagate that state through the candidate simulator and update it using the next observation likelihood.
The state includes model memory, so observations of downstream effects can constrain earlier hidden quantities.
A rest detection event does not reset that memory.

Specify the conditional representation explicitly: one deterministic state trajectory per fixed parameter candidate cannot correct its state through likelihood reweighting alone.
Retain multiple plausible initial states or histories under each parameter candidate, or implement valid conditional moves that propose alternative histories.
Resampling duplicates represented possibilities; it does not create missing states or repair an incorrect program.

For execution, use filtering based only on the available observation prefix.
For recorded experience, use smoothing to revise earlier states using later observations.
These are standard distinctions in Bayesian state estimation ([Särkkä and Svensson, 2023](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf)).
With deterministic dynamics, samples of parameters and initial state already induce samples of entire trajectories; full-data weighting therefore provides a first form of smoothing without introducing a free latent variable at every time step.
A general smoother becomes necessary if we introduce stochastic transition discrepancy or need a richer state representation.

Preserve the agent's control over parameter fitting during the first rollout of this design.
Between explicit fits, update conditional state beliefs under the published parameter candidates while retaining their published outer weights.
This is an explicit approximation: new observations refine state immediately, while parameter weights are refreshed by `sim.fit()`.
It is not the full joint Bayesian update at every environment step.
Automatic online parameter reweighting can be considered later, but should not be smuggled into a replacement for the observation smoother.

Use the same raw observation ledger for the subsequent batch fit.
Do not treat filtered states as new independent measurements, and do not multiply an already-updated belief by the same data again.
When a new fit is published, reconstruct the current conditional state under its candidates from the recorded prefix.
Before live use, define a bounded recovery procedure for state-particle collapse and observations inconsistent with every represented trajectory.
First attempt valid conditional reconstruction from the available prefix within an explicit compute budget.
If reconstruction fails, expose an unavailable or unreliable model-conditioned belief and request agent-controlled revision or fitting.
Retain the established observation-only estimator as an explicitly labeled degraded mode for observable features during this migration; its output is not a joint posterior and cannot certify hidden state.
Do not continue reporting stale hidden-state certainty, silently substitute raw poses, or automatically take environment actions as recovery.
Record degraded-mode use as part of the candidate system's results rather than excluding those episodes.
Reconstruction from a longer prefix can address missing particle support but does not itself resolve program mismatch.

This replaces rest-window averaging only after it has demonstrated better state estimates and reliable online behavior.
A stationary Gaussian average remains a useful reference case for testing the filter.

## 5. Share inference while preserving decision semantics

### Planning and subgoal checks

At the first endpoint, draw parameters from the common posterior and initialize rollouts with the existing execution state estimate and model memory.
Label the resulting success estimates as conditional on that supplied state estimate: they account for parameter uncertainty but do not integrate current-state uncertainty or preserve its full dependence on parameters.
Uncertain initial-state inference during fitting improves the parameter posterior without by itself supplying a continuously updated joint execution belief.
Evaluate conditional state sampling for planning separately, using only the observation prefix available at the decision.
For that extension, draw a parameter candidate and a state conditional on that candidate, then simulate the proposed plan.
Keep the parameter vector fixed throughout a rollout because it represents an unknown constant, not process noise.
Preserve hidden memory within the rollout and copy it when branching.
The conditional extension avoids combining an independently sampled state and parameter vector that the observations rule out jointly.

Report weighted plan-success estimates and predicate probabilities under the represented belief.
Distinguish posterior uncertainty from Monte Carlo estimation error and simulator mismatch.
A successful finite sample is not a guarantee over a continuous parameter region.
The environment evaluator continues to determine whether a level is solved.

Keep posterior-weighted plan success and stress-test outcomes as separate report fields.
The first estimates success under the represented distribution; the second identifies failures at explicitly chosen plausible candidates or boundaries without assigning them posterior probability.
Stress tests should respect joint parameter/state feasibility instead of combining incompatible marginal extremes.
Keep mixed-outcome warnings and the existing information-seeking trigger semantics during the first planning comparison.
Sampling that misses a rare failure mode must not be described as the equivalent of the old interval check.

Initially keep existing subgoal decision thresholds, plan acceptance rules, and probe triggers.
This holds the decision policy fixed; changing its input distribution can still change actions, so closed-loop comparison remains necessary.
If moving to posterior probabilities requires a different acceptance rule, evaluate that as a separate policy change.
Subsequent threshold choices should reflect the cost of false completion versus unnecessary continued action, rather than being presented as an inference algorithm.

### Information seeking

Keep the name **noise-aware information-seeking score**.
At first, retain the existing score while replacing its parameter ensemble with samples from the common posterior.
That isolates the effect of consistent uncertainty estimates.
Use the posterior weights in ensemble averages, or resample to an equally weighted ensemble with the resulting sampling error recorded.
Do not treat arbitrary weighted candidates as equally probable.

A later improvement is to simulate each candidate interaction under joint parameter/state samples, generate observations through the sensor model, and estimate the expected information gained about the parameters.
This would account for candidate-specific dynamics and uncertain state, which the current predicate score does not fully do.
The agent should still decide whether an informative interaction is worth its environment steps; information gain alone is not the task objective.

## 6. What happens to the existing mechanisms?

| Existing mechanism | Proposed treatment |
| --- | --- |
| Parameter interval flag and grid/probe widths | Replace with posterior marginal quantiles and samples from the same joint approximation. |
| Noise-aware information-seeking flag | Retain the decision criterion, with a clearer name and posterior-derived candidates. |
| Noise-aware rest segmentation and start averaging | Preserve in legacy; replace only after replay and initial-state inference pass predictive checks. |
| Carry-posterior flag | Preserve in legacy; use a fixed original prior in the posterior path and compare resulting retention of previously learned dynamics. |
| Fit-evidence flag | Preserve legacy reports during parity checks; separately evaluate replacement by common-data predictive diagnostics. |
| Execution-belief flag | Retain the observation-only estimator as a valid endpoint; replace it only if the optional conditional filter and recovery behavior demonstrate a benefit. |
| Width floors and identified/weak/wide deployment verdicts | Preserve in legacy; return numerically adequate posterior inference with broad uncertainty and predictive failures visible, keeping decision use separate. |
| Anchor-pinned backward elimination | Remove from the target estimator; correlations and prior preference should arise from joint inference. |
| Segment rejection and consistency-based dropping | Preserve legacy protection until an explicit likelihood explains difficult recordings without degrading useful predictions. |
| Extra endpoint/onset losses | Preserve in legacy; in the posterior path use these as decision-relevant diagnostics, not duplicated independent evidence. |
| Huber residual transformation | Preserve in legacy; assess whether an explicit discrepancy model is needed before deploying the replacement. |
| Grid searches and zero-gradient recovery | May remain numerical proposal aids, but do not determine posterior widths or replace posterior weighting. |
| Interval stress tests and mixed-outcome warnings | Preserve their decision purpose separately from posterior probability estimation. |
| Fresh replay, snapshotting, and caches | Retain as correctness and efficiency measures, with keys that identify all relevant inputs. |

Removing a feature from the target design does not authorize deleting it before the replacement passes comparisons.
During development, prefer a single estimator selection such as `legacy` versus `posterior` to a growing collection of mutually dependent flags.
Use immutable, named development configurations to isolate migration stages rather than adding every intermediate combination to the permanent public interface.
The temporary legacy parameter fitter is a comparison and rollback mechanism; the intended endpoint has one validated parameter-inference path with explicit diagnostics and failure reporting.
The existing execution state estimator may remain part of that endpoint without retaining a second parameter fitter.
Keep sampling budgets and decision thresholds explicit because they control different tradeoffs.

## 7. Implementation order and acceptance checks

### Stage 0: preserve behavior behind the interface

Snapshot the successful tagged runtime, its configuration, and the baseline reports.
Introduce the legacy result adapter without changing fitting, observation estimates, prompts, tool replies, parameter publication, or action rules.
Check recorded-action replay and scripted end-to-end parity of observations, tool replies, action traces, steps, and resets.
These checks establish plumbing parity, not unchanged solve rate for stochastic agent conversations.
Record later runtime changes separately instead of attributing every difference from the historical tag to uncertainty inference.

### Stage A: define and verify the probability model

Implement and verify the candidate-initialization and full-prefix replay contract from section 1 before the new fitter consumes real recordings.
Arbitrary mid-trajectory restoration is not an advancement requirement.
Complete the five domain inventories and derive the reduced conditional targets before advancing to real-recording posterior comparisons.
Implement the observation likelihood, initial-state prior, immutable data identity, and posterior result format beside the existing fitter.
Verify the likelihood against the noise injector, including angle handling, missing measurements, and cached observations.
Match the injector's actual angular representation; do not assume an additive unwrapped observation is distributed identically to a wrapped observation.
Test inference on a stationary noisy object and a small parameterized dynamical system with a grid-computable reference posterior.
Check that an uninformed parameter retains its prior, correlated parameters retain their tradeoff, and repeated fitting on identical data does not accumulate confidence.
Check valid recovery from poor initialization, impossible exact observations, and insufficient numerical budget.
Add distinct reference cases for an exactly observed continuous initial coordinate, a feasible continuous trajectory constraint, a discrete exact output, and a provably inconsistent constraint set.
Verify correct conditional weights when an observed coordinate depends on parameters, and verify that likelihood tempering is not being used to repair missing exact support.
Check that a numerically adequate but poorly predictive fit is returned with its failures, while numerical failure cannot be presented as a usable posterior.
These tests check statistical behavior rather than reproduce the implementation.

### Stage B: compare parameter inference on recorded experience

Add the candidate batch sampler and uncertain initial states, leaving the existing agent in control of interactions.
Compare fixed-prior posterior inference against the full legacy fitter on the same development recordings from boil, domino, fan, bridge, and balloons.
Freeze candidate programs for each offline comparison so the estimator is the manipulated variable.
Measure predictions on held-out development interactions or causal future suffixes, not only reconstruction of the fitted recordings.
Measure feature error, contact and attachment outcomes, event timing, and goal-relevant predictions.
Assess parameter coverage only where a known parameter meaning exists, across repeated datasets; a single interval containing the truth does not establish calibration.
Also measure multimodal behavior, prediction stability across independent sampler runs, and inference cost.
Use recordings with quiet starts, moving starts, contact transitions, hidden memory, and deliberately incomplete candidate programs.
Include matched ablations that isolate fixed-prior fitting from uncertain initial-state inference, so a combined result does not conceal which change helps or hurts.
Inspect disagreement rather than selecting only examples where the posterior replacement wins.
Pass this gate only when the candidate produces trustworthy, decision-relevant predictions within the declared compute budget.
If the sensor-only model fails that gate, evaluate an explicit discrepancy model before proceeding to live posterior use.

### Stage C: use the posterior in planning

First run the new planning reports in shadow mode on saved decision points without exposing them to the acting agent.
Then route parameter sweeps through the new result object in matched live development runs, retaining plan-risk rules, warnings, and execution observations.
Change exploration ensembles in a separate comparison because they can change the data the agent chooses to collect.
Keep the existing execution observation estimate and explicitly report the approximation described in section 5.
This configuration is eligible as the completed parameter-inference replacement after Stage E; conditional state sampling is not a prerequisite.
Optionally add conditional state sampling to rollouts in a separate comparison against plugging in the existing state estimate.
This separates handling uncertain recording starts during fitting from integrating current-state uncertainty during planning.
Log changes in accepted plans, probe triggers, simulated failure cases, and actual actions in addition to final performance.

### Stage D (optional): evaluate a replacement execution smoother

Stage C may proceed directly to Stage E while retaining the existing execution state estimator.
Compare the conditional filter against the existing rest-window estimator on recorded prefixes, then in live runs.
Measure tracking error, motion lag, hidden-state accuracy where evaluable, subgoal false positives and negatives, and latency.
Exercise incomplete models, impossible observations, particle collapse, and recovery-budget exhaustion before enabling it in live runs.
Report how frequently the degraded observation-only mode is used and whether it causes extra steps, resets, or missed subgoals.
Ensure filtering has no access to future observations and that full-data smoothing results never enter an earlier online decision.
Keep parameter publication explicit as described above.

### Stage E: decide what can be retired

Run matched development experiments over multiple seeds in all five noisy domains.
Hold the LLM version, initial simulator program, prompts, tool interfaces, task distributions, and environment budgets fixed for each estimator comparison.
Learned programs and action histories may subsequently diverge; that divergence is part of the estimator's closed-loop effect and must be recorded.
Pair task and observation-noise seeds, and record conversation randomness and infrastructure differences.
Allow additional simulator work when it improves environment sample efficiency, with an explicit per-fit and per-run compute budget and observed latency reported.
Report solve rates, failures, environment steps, resets, and inference cost by domain.
Average steps only over whole-run successful seeds with qualifying counts, while reporting all-run outcomes and successful/failed run costs separately so survivor selection cannot masquerade as efficiency.
Keep infrastructure failures outside agent success/failure denominators and report incomplete comparisons as incomplete.
A few successful seeds do not establish unchanged performance.
Before launching the comparison, record numerical non-inferiority margins for solve rate and resets, an efficiency target, the seed count or sequential stopping rule, and the uncertainty calculation.
Size that comparison to the claimed margins; repeating the original three seeds per domain is a smoke test, not a tight non-regression result.
Treat inconclusive evidence as a reason to retain the incumbent default.
An aggregate efficiency gain must not hide an unacceptable regression in one domain.

Use separate development runs for selecting the implementation, then freeze it for the final continual evaluation.
This does not prohibit learning during a test level: the agent can still use its allowed step experience as required by the protocol.
It prohibits choosing the research implementation based on repeated inspection of the final evaluation outcomes.

Retire the old production parameter fitter only after the selected replacement configuration passes these checks across all five domains.
Retire the existing execution estimator only if the optional Stage D replacement also demonstrates a benefit and passes its validation checks.
Retain reproducible historical source and result artifacts for comparison.
If the posterior model fails systematically on contact-rich recordings, revisit its state and discrepancy assumptions before adding back a collection of unrelated thresholds.

## First concrete change I would make

Make the first implementation chunk the legacy result adapter and state/replay contract with parity checks.
Make the second chunk the fixed-prior batch inference prototype with uncertain initial states, restricted to offline comparisons.
Keep the current agent running on the existing implementation throughout those stages.

Before a live estimator comparison, provide the domain inventories, exact-conditioning construction, likelihood and prior definitions, numerical reference results, saved five-domain prediction comparisons, and an explanation of disagreements with the current fitter.
Improved closed-loop performance remains an experimental claim to establish after those deliverables.
