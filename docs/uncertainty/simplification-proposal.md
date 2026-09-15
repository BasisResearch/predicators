# Proposal: simplify uncertainty handling in EMPIRIC

September 11, 2026.
This is a design proposal, not a description of an implemented or validated replacement.
The companion [implementation explanation](explained.md) documents the current behavior and source locations.

## Recommendation

Replace the separate uncertainty heuristics with a shared inference model:

1. A posterior over the fixed parameters of the current simulator program.
2. A conditional belief over physical and hidden state.
3. Planning, subgoal checking, and information seeking that use samples from these distributions.

Start with parameter inference and uncertain initial states in `sim.fit()`.
Use a fixed prior and all available fitting recordings, with intervals and parameter ensembles derived from the same posterior approximation.
Then replace the separate state averages with filtering and smoothing under that observation model.
Keep the simulator subclass interface and the agent's ability to choose its tools and actions.

This would simplify the statistical assumptions and remove duplicated uncertainty calculations.
It would not make contact dynamics smooth or remove the need for numerical diagnostics.
Preserving performance is an experimental requirement, not something we can infer from a cleaner formulation.

## 1. Define one inference problem

Let \(P\) be the current simulator program, \(\theta\) its fixed dynamics parameters, and \(s_t\) the full physical and model state.
The state includes velocities and accumulated quantities such as heat or curing state when they affect future dynamics.
Let \(o_t\) be the noisy object-centric observation and \(a_t\) the executed action.

For the first implementation, use deterministic candidate dynamics and the declared observation channel:

\[
s_{t+1}=F_{P,\theta}(s_t,a_t),
\qquad o_t\sim p(o_t\mid s_t).
\]

For reset episodes indexed by \(e\), fit the joint posterior

\[
p(\theta,\{s_{e,0}\}_e\mid D,P)
\propto
p_0(\theta\mid P)
\prod_e \left[
p_0(s_{e,0}\mid\theta,P)
\prod_{t=0}^{T_e}
p(o_{e,t}\mid s_{e,t}(\theta,s_{e,0},a_{e,0:t-1}))
\right].
\]

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
An uninformative parameter then retains its prior uncertainty naturally.
There is no need to classify it as sufficiently identified before allowing it to be represented in planning.

### Missing dynamics remain a separate problem

A posterior can be narrow and wrong when the program is wrong.
Check whether posterior predictions explain the observations, and report persistent discrepancies to the agent so it can revise its simulator.
Do not inflate sensor noise or silently delete difficult trajectories until the current program looks accurate.

The initial comparison should use the declared sensor likelihood directly, so its limitations are visible.
If replay mismatch requires a discrepancy model, make that an explicit, separately evaluated extension with its own parameters and prior.
For example, a model of occasional bad measurements addresses a different failure from temporally correlated errors caused by missing forces.
Neither should be introduced merely to reproduce the old trimming decisions.

## 2. Implement one posterior approximation

I propose a batch sampler over joint dynamics parameters and uncertain episode initial states as the reference implementation.
Use tempered sequential Monte Carlo with derivative-free Metropolis moves, rather than relying on the contact simulator's local Jacobian.
This provides a common representation for bounded, correlated, or multimodal parameter uncertainty.
Sequential Monte Carlo samplers support weighted approximations of distributions known up to normalization ([Del Moral, Doucet, and Jasra, 2006](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)).
The following adaptation is a proposal for this repository, not a result established by that reference.

A canonical `sim.fit()` would:

1. Snapshot the simulator program, observation model, priors, and complete fitting data.
2. Initialize joint candidates from the declared prior or from a proposal with a known density and the appropriate importance correction.
3. Replay each recorded action sequence under each candidate, preserving model memory, and calculate the observation log likelihood.
4. Move from the prior toward the posterior by gradually increasing the likelihood exponent from zero to one.
5. Reweight, resample when weights concentrate, and apply Metropolis moves that target the current tempered distribution.
6. Publish weighted samples, marginal quantiles, predictive checks, and numerical diagnostics.

Previous fits and optimizer results can help construct proposals, but are not automatically posterior samples for the new dataset.
A finite collection of perturbations around a MAP estimate is also not a posterior unless its distribution and weights justify that interpretation.
Grid integration on small synthetic problems should serve as a numerical reference for testing the sampler.
We should not maintain a second production interval estimator based on coordinate sweeps.

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
| Joint samples and normalized weights | Supply one source for parameter and state uncertainty. |
| Point summary and marginal credible intervals | Provide readable summaries without a separate width estimator. |
| Posterior predictive diagnostics | Show discrepancies on specific recorded features and time intervals. |
| Numerical diagnostics | Distinguish an unreliable approximation from uncertainty supported by the model and data. |

A point estimate remains useful for deterministic debugging and nominal rollouts.
A mean parameter vector can fall between incompatible modes, so nominal execution should use an explicitly selected representative candidate rather than assume every posterior mean describes plausible dynamics.

Keep canonical `sim.fit()` as the operation that publishes a new parameter posterior.
Subset fits remain diagnostic.
A program edit invalidates old likelihoods and inferred memory; a refit evaluates recordings under the new program and its declared parameter space.
Do not silently carry posterior weights across changes in program meaning or parameter definitions.

## 3. Use Bayesian state estimation without adding a separate uncertainty system

For each retained parameter candidate, maintain a conditional state belief.
Propagate that state through the candidate simulator and update it using the next observation likelihood.
The state includes model memory, so observations of downstream effects can constrain earlier hidden quantities.
A rest detection event does not reset that memory.

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
If particles collapse during execution, report the failure and reconstruct the state belief from a longer prefix or request a refit; silently replacing it with the latest noisy pose would reintroduce the initial-state problem.

This replaces rest-window averaging only after it has demonstrated better state estimates and reliable online behavior.
A stationary Gaussian average remains a useful reference case for testing the filter.

## 4. Make the consumers share samples

### Planning and subgoal checks

Draw a parameter candidate and a state conditional on that candidate, then simulate the proposed plan.
Keep the parameter vector fixed throughout a rollout because it represents an unknown constant, not process noise.
Preserve hidden memory within the rollout and copy it when branching.
This avoids combining an independently sampled state and parameter vector that the observations rule out jointly.

Report weighted plan-success estimates and predicate probabilities under the represented belief.
Distinguish posterior uncertainty from Monte Carlo estimation error and simulator mismatch.
A successful finite sample is not a guarantee over a continuous parameter region.
The environment evaluator continues to determine whether a level is solved.

Initially keep existing subgoal decision thresholds so the state estimator can be compared without also changing execution policy.
Subsequent threshold choices should reflect the cost of false completion versus unnecessary continued action, rather than being presented as an inference algorithm.

### Information seeking

Keep the name **noise-aware information-seeking score**.
At first, retain the existing score while replacing its parameter ensemble with samples from the common posterior.
That isolates the effect of consistent uncertainty estimates.

A later improvement is to simulate each candidate interaction under joint parameter/state samples, generate observations through the sensor model, and estimate the expected information gained about the parameters.
This would account for candidate-specific dynamics and uncertain state, which the current predicate score does not fully do.
The agent should still decide whether an informative interaction is worth its environment steps; information gain alone is not the task objective.

## 5. What happens to the existing mechanisms?

| Existing mechanism | Proposed treatment |
| --- | --- |
| Parameter interval flag and grid/probe widths | Replace with posterior marginal quantiles and samples from the same joint approximation. |
| Noise-aware information-seeking flag | Retain the decision criterion, with a clearer name and posterior-derived candidates. |
| Noise-aware rest segmentation and start averaging | Replace with uncertain initial-state inference and state estimation; keep temporarily as the comparison baseline. |
| Carry-posterior flag | Remove prior recentering; keep the original prior and reuse experience through the likelihood. |
| Fit-evidence flag | Remove from the default agent report; use predictive diagnostics on common data for program revision. |
| Execution-belief flag | Replace with the conditional state estimator after validation. |
| Width floors and identified/weak/wide deployment verdicts | Remove from inference; preserve uncertainty instead of selecting which parameters are allowed to have it. |
| Anchor-pinned backward elimination | Remove from the target estimator; correlations and prior preference should arise from joint inference. |
| Segment rejection and consistency-based dropping | Remove from the target likelihood; diagnose mismatch or model it explicitly. |
| Extra endpoint/onset losses | Keep as predictive diagnostics, not extra independent evidence alongside the original observations. |
| Huber residual transformation | Replace with an explicit observation or discrepancy model if robust treatment is needed. |
| Grid searches and zero-gradient recovery | May remain numerical proposal aids, but do not determine posterior widths or replace posterior weighting. |
| Fresh replay, snapshotting, and caches | Retain as correctness and efficiency measures, with keys that identify all relevant inputs. |

Removing a feature from the target design does not authorize deleting it before the replacement passes comparisons.
During development, prefer a single estimator selection such as `legacy` versus `posterior` to a growing collection of mutually dependent flags.
Keep sampling budgets and decision thresholds explicit because they control different tradeoffs.

## 6. Implementation order and acceptance checks

### Stage A: define and verify the probability model

Implement the observation likelihood, initial-state prior, immutable data identity, and result format beside the existing fitter.
Verify the likelihood against the noise injector, including angle handling, missing measurements, and cached observations.
Test inference on a stationary noisy object and a small parameterized dynamical system with a grid-computable reference posterior.
Check that an uninformed parameter retains its prior, correlated parameters retain their tradeoff, and repeated fitting on identical data does not accumulate confidence.
These tests check statistical behavior rather than reproduce the implementation.

### Stage B: compare parameter inference on recorded experience

Add the batch sampler and uncertain initial states, leaving the existing agent in control of interactions.
Compare fixed-prior posterior inference against the full legacy fitter on the same development recordings from boil, domino, fan, bridge, and balloons.
Measure predictive error, interval coverage where ground truth is available to the offline evaluator, multimodal behavior, and inference cost.
Use recordings with quiet starts, moving starts, contact transitions, hidden memory, and deliberately incomplete candidate programs.
Inspect disagreement rather than selecting only examples where the posterior replacement wins.

### Stage C: use the posterior in planning

Route parameter sweeps and exploration ensembles through the new result object.
Keep the existing execution observation estimate for this comparison and label it as an interim approximation.
Then add conditional state sampling to rollouts and compare it against plugging in the smoothed mean.
This separates the benefit of parameter inference from the benefit of handling initial-state uncertainty.

### Stage D: replace the execution smoother

Compare the conditional filter against the existing rest-window estimator on recorded prefixes, then in live runs.
Measure tracking error, motion lag, hidden-state accuracy where evaluable, subgoal false positives and negatives, and latency.
Ensure filtering has no access to future observations and that full-data smoothing results never enter an earlier online decision.
Keep parameter publication explicit as described above.

### Stage E: decide what can be retired

Run matched development experiments over multiple seeds in all five noisy domains.
Use the same model, prompts, tools, task distributions, and environment budgets for each estimator comparison.
Report solve rates, failures, environment steps, resets, and inference cost, including domain-level results rather than only an aggregate.
A few successful seeds do not establish unchanged performance.
Choose acceptable regression margins before the comparison and report uncertainty in the differences.

Use separate development runs for selecting the implementation, then freeze it for the final continual evaluation.
This does not prohibit learning during a test level: the agent can still use its allowed step experience as required by the protocol.
It prohibits choosing the research implementation based on repeated inspection of the final evaluation outcomes.

Retire the old safeguards only where the replacement passes these checks.
If the posterior model fails systematically on contact-rich recordings, revisit its state and discrepancy assumptions before adding back a collection of unrelated thresholds.

## First concrete change I would make

Build the fixed-prior batch inference path and a common posterior result object, initially used for offline replay comparisons.
Include uncertain initial states from the start so perceptual error is not forced into dynamics parameters.
Keep the current agent running on the existing implementation until those comparisons establish a credible replacement.

The first reviewable implementation should contain the likelihood and prior definitions, numerical reference tests, results on saved five-domain recordings, and a report explaining discrepancies with the current fitter.
It should not claim that a full Bayesian smoother or a closed-loop performance improvement has already been achieved.
