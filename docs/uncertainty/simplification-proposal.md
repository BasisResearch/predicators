# Proposal: simplify uncertainty handling in EMPIRIC

September 11, 2026.
This is a design proposal, not a description of an implemented or validated replacement.
The companion [implementation explanation](explained.md) documents the current behavior and source locations.
Implementation of the staged migration is tracked in [implementation progress](implementation-progress.md).

## Recommendation

Unify the uncertainty interface first, then replace its estimators one at a time.
The target is a shared inference model:

1. A posterior over the fixed parameters of the current simulator program.
2. A conditional belief over physical and hidden state.
3. Planning, subgoal checking, and information seeking that use samples from these distributions.

Start by preserving the current agent behind a versioned result interface, without changing its estimates, reports, or decisions.
Verify state restoration and recorded-action replay before implementing parameter inference with uncertain initial states.
Evaluate that inference offline using a fixed prior and all available fitting recordings, with intervals and parameter ensembles derived from the same posterior approximation.
Introduce it into planning only after it passes prediction checks, while preserving the existing execution estimator and decision rules.
Replace execution state estimation last, if it independently demonstrates a benefit.
Keep the simulator subclass interface and the agent's ability to choose its tools and actions.

The design separates three questions: what the model and data imply, whether that inference is trustworthy, and whether an action is worth its risk and cost.
A posterior answers the first question; predictive and numerical diagnostics address the second; planning and execution rules address the third.
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
If all candidates violate exact constraints, report an inference or model failure instead of manufacturing a normalized posterior.

The current [fitting replay](../../predicators/code_sim_learning/rollout_env.py), `rollout_states`, zeros velocities after restoring its initial state.
An inference path that samples initial velocities must restore and retain those velocities; it cannot inherit that rest-start assumption unchanged.
The legacy path keeps its current behavior for comparison.

Validate repeated replay, moving starts, contact transitions, attachment changes, and memory continuity on short and long recorded prefixes.
Use known dynamics and evaluator-only state to diagnose reconstruction error offline, without exposing that information to the agent.
Separate reconstruction failures from candidate-program errors before judging a statistical estimator.

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

## 3. Standardize the inference result, evaluate the approximation

Evaluate a batch sampler over joint dynamics parameters and uncertain episode initial states as the first candidate implementation.
Tempered sequential Monte Carlo with derivative-free Metropolis moves is a reasonable candidate because it does not require a reliable local contact Jacobian.
Its representation can express bounded, correlated, or multimodal parameter uncertainty; actually discovering that uncertainty requires adequate exploration.
Sequential Monte Carlo samplers support weighted approximations of distributions known up to normalization ([Del Moral, Doucet, and Jasra, 2006](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)).
The following adaptation is a proposal for this repository, not a result established by that reference.

A candidate batch fit would:

1. Snapshot the simulator program, observation model, priors, and complete fitting data.
2. Initialize joint candidates from the declared prior or from a proposal with a known density and the appropriate importance correction.
3. Replay each recorded action sequence under each candidate, preserving model memory, and calculate the observation log likelihood.
4. Move from the prior toward the posterior by gradually increasing the likelihood exponent from zero to one.
5. Reweight, resample when weights concentrate, and apply Metropolis moves that target the current tempered distribution.
6. Return weighted samples, marginal quantiles, predictive checks, and numerical diagnostics for validation before publication.

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
| Publication status and reason | Distinguish a diagnostic candidate from the estimate actually used by the agent. |

The initial legacy adapter preserves existing point estimates, heuristic widths, reports, and publication rules exactly.
Label those widths as legacy estimates in internal metadata; do not invent posterior samples or claim calibration that the adapter does not establish.
Keep user-visible tool output unchanged during this interface-only step because changes to the agent's input can change its behavior.

A point estimate remains useful for deterministic debugging and nominal rollouts.
A mean parameter vector can fall between incompatible modes, so nominal execution should use an explicitly selected representative candidate rather than assume every posterior mean describes plausible dynamics.

Keep canonical `sim.fit()` as the operation that publishes a new parameter posterior.
Subset fits remain diagnostic.
For the posterior path, publication requires successful numerical checks and the predictive-adequacy checks chosen before the development comparison.
If a candidate fit fails, retain a previously published estimate only if its program and parameter meaning remain compatible, and explicitly report its age, failed checks, and limitations.
An incompatible program edit leaves no valid published posterior; it must not silently inherit old weights or memory.
Preserve the existing agent-controlled interaction flow so an unavailable posterior does not impose an automatic environment probe, reset, or indefinite refit loop.
During migration, rollback to the legacy agent is an explicit, recorded experiment choice, not a silent per-fit estimator switch.
A program edit invalidates old likelihoods and inferred memory; a refit evaluates recordings under the new program and its declared parameter space.
Do not silently carry posterior weights across changes in program meaning or parameter definitions.

## 4. Replace execution state estimation only after independent validation

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

Draw a parameter candidate and a state conditional on that candidate, then simulate the proposed plan.
Keep the parameter vector fixed throughout a rollout because it represents an unknown constant, not process noise.
Preserve hidden memory within the rollout and copy it when branching.
This avoids combining an independently sampled state and parameter vector that the observations rule out jointly.

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
| Execution-belief flag | Preserve the observation-only estimator until the conditional filter and its recovery behavior pass validation. |
| Width floors and identified/weak/wide deployment verdicts | Preserve in legacy; posterior inference retains broad uncertainty, while numerical and predictive validity govern publication. |
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
The temporary legacy implementation is a comparison and rollback mechanism; the intended endpoint has one validated production inference path with explicit diagnostics and recovery behavior.
Keep sampling budgets and decision thresholds explicit because they control different tradeoffs.

## 7. Implementation order and acceptance checks

### Stage 0: preserve behavior behind the interface

Snapshot the successful tagged runtime, its configuration, and the baseline reports.
Introduce the legacy result adapter without changing fitting, observation estimates, prompts, tool replies, parameter publication, or action rules.
Check recorded-action replay and scripted end-to-end parity of observations, tool replies, action traces, steps, and resets.
These checks establish plumbing parity, not unchanged solve rate for stochastic agent conversations.
Record later runtime changes separately instead of attributing every difference from the historical tag to uncertainty inference.

### Stage A: define and verify the probability model

Implement and verify the state/restoration contract from section 1 before the new fitter consumes real recordings.
Implement the observation likelihood, initial-state prior, immutable data identity, and posterior result format beside the existing fitter.
Verify the likelihood against the noise injector, including angle handling, missing measurements, and cached observations.
Match the injector's actual angular representation; do not assume an additive unwrapped observation is distributed identically to a wrapped observation.
Test inference on a stationary noisy object and a small parameterized dynamical system with a grid-computable reference posterior.
Check that an uninformed parameter retains its prior, correlated parameters retain their tradeoff, and repeated fitting on identical data does not accumulate confidence.
Check valid recovery from poor initialization, impossible exact observations, and insufficient numerical budget.
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
Keep the existing execution observation estimate for this comparison and label it as an interim approximation.
Then add conditional state sampling to rollouts and compare it against plugging in the smoothed mean.
This separates the benefit of parameter inference from the benefit of handling initial-state uncertainty.
Log changes in accepted plans, probe triggers, simulated failure cases, and actual actions in addition to final performance.

### Stage D: replace the execution smoother

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

Retire the old production estimator only after the selected replacement configuration passes these checks across all five domains.
Retain reproducible historical source and result artifacts for comparison.
If the posterior model fails systematically on contact-rich recordings, revisit its state and discrepancy assumptions before adding back a collection of unrelated thresholds.

## First concrete change I would make

Make the first implementation chunk the legacy result adapter and state/replay contract with parity checks.
Make the second chunk the fixed-prior batch inference prototype with uncertain initial states, restricted to offline comparisons.
Keep the current agent running on the existing implementation throughout those stages.

Before a live estimator comparison, provide the likelihood and prior definitions, numerical reference results, saved five-domain prediction comparisons, and an explanation of disagreements with the current fitter.
Improved closed-loop performance remains an experimental claim to establish after those deliverables.
