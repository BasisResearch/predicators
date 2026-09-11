# Uncertainty in EMPIRIC: what the code does and how to simplify it

Checked against `predicators` commit `d97378b5d79fc4395c21a93a6b9332727d7b0f00` on September 11, 2026.
This note explains the current simulator-subclass fitting path and proposes a simpler design.
The proposals below are not implemented or evaluated here.

The six flags are not six different kinds of uncertainty.
They combine parameter estimation, state estimation, an information-seeking score, and several fitting safeguards.
The main simplification should be to use one parameter posterior and one state belief consistently across fitting, planning, and execution.
We cannot yet promise that removing the existing safeguards would preserve performance.

## 1. Parameter intervals versus the posterior from fitting

**An interval should be a summary of a posterior, but the current implementation does not consistently work that way.**

The rollout fitter returns a MAP point estimate and, when available, a residual Jacobian that supports a local Gaussian approximation.
Its `FitResult.samples` normally contains just one row: the fitted point, not posterior samples.
The approximate covariance available to some consumers is

\[
\Sigma_{\mathrm{local}}
\approx
\left(J^\top J/\nu^2 + \Sigma_0^{-1}\right)^{-1},
\]

where the derivatives and prior are expressed in the optimizer's coordinates and \(\nu\) is the width of the scaled residual model.
The exploration ensemble and fit-evidence calculation can use this approximation.

The reported parameter intervals deliberately use a different calculation because small finite differences through contact dynamics can produce misleadingly sharp Jacobians.
For swept parameters, the code uses the range of approximately equally good grid values, widened according to the grid's resolution.
For unswept parameters, it probes the loss on either side of the fitted value at roughly the prior's scale.
It also applies a minimum width, and planning sweeps can expand to include conflicting fits from different segments.
These are useful uncertainty heuristics, but their reported `posterior_std` is not generally the standard deviation of a normalized posterior.
In particular, the fallback curvature width is computed from loss curvature without adding prior precision, and grid widths do not integrate over the other parameters.
The printed “+-1 posterior sigma” should therefore not be interpreted as a calibrated credible interval.

`code_sim_learning_interval_belief` changes how this estimate is used: a fit that moved away from its prior center but remains broad can be accepted as `WIDE`, rather than discarded for the baseline value.
It also makes the grid's equivalence tolerance account for observation noise and exposes intervals for parameter sweeps.
A finite sweep is a robustness check at selected parameter vectors, not a probability of success or a guarantee over the whole interval.

**Simplification:** compute one approximate joint posterior, report its marginal quantiles, and draw planning and exploration parameters from that same distribution.
Keep model-mismatch diagnostics separate from posterior credible intervals.

Implementation: [physical_sysid.py](../../predicators/code_sim_learning/physical_sysid.py), `fit_params_rollout`; [identifiability.py](../../predicators/code_sim_learning/identifiability.py), `identifiability_report`, `_probe_posterior_widths`, and `physics_sigma_points`; [active_experiment.py](../../predicators/code_sim_learning/active_experiment.py), `laplace_ensemble`.

## 2. Noise-aware information-seeking score

**“Noise-aware information-seeking score” is a better name than “noise-aware probe.”**
The flag is `agent_explorer_info_seeking_noise_aware`.
It changes the score used to rank candidate experiments, rather than introducing a new kind of probe.

For a candidate state, the code evaluates predicates under each parameter-ensemble member using eight noisy views of that state.
It reuses the same random draws across candidates to reduce score fluctuations.
For each predicate, let \(p_k\) be the fraction of noisy views on which it holds under member \(k\).
The score is

\[
H\left(\frac{1}{K}\sum_k p_k\right)
- \frac{1}{K}\sum_k H(p_k),
\]

averaged over predicates, where \(H\) is binary entropy.
For example, if one member predicts a predicate almost certainly true and another almost certainly false, observing it can distinguish the members.
If both predict a 50/50 noisy reading, observing it tells us little about which member is right.

This is a mutual-information score for predicate readings under an equally weighted ensemble.
The current implementation swaps parameters used by predicate classifiers at a supplied state; this scoring function does not itself simulate a separate future trajectory under every parameter member.
It is therefore a proxy for experiment informativeness, not a full expected information gain over future observation sequences.
It also does not force the agent to gather information.

Implementation: [agent_sim_learning_approach.py](../../predicators/approaches/agent_sim_learning_approach.py), `score_atom_disagreement` and `_noisy_read_views`; [active_experiment.py](../../predicators/code_sim_learning/active_experiment.py), `noisy_read_information`.

## 3. Filtering recordings for fitting

**The filter detects motion and rest for segmentation; it does not keep only windows judged to be noise-free.**
The flag is `code_sim_learning_rollout_noise_filter`.

1. For each noisy feature, compare its mean over the preceding window with its mean over the following window.
2. Mark the step active if the difference exceeds a threshold based on the declared sensor noise.
3. Use active steps to cut settled tails and separate motion segments at sufficiently long rest gaps.
4. Initialize each retained segment using the average of its preceding rest window, reducing noise in the initial pose.

The defaults use windows of eight frames and a threshold of three standard errors, floored by the existing motion tolerance.
For two full windows, the threshold is \(\max(\epsilon,3\sigma\sqrt{2/8})\).
Exact features retain their per-step motion test; angular differences and means account for angle wrapping.
The current threshold still uses the configured window size when windows are shorter at trajectory boundaries.

The moving observations inside a retained segment remain noisy and are still scored by the fitter.
This flag is separate from the later rejection of poorly explained segments described below.
The detector can miss slow motion, and averaging a supposedly stationary window can bias the initial pose.
It is a heuristic initial-state estimate, not marginalization over uncertain initial states.

**Subclass exception:** when the simulator declares persistent model state, `_rollout_fit_trajectories` keeps full trajectories instead of truncating or segmenting them.
A rest pose does not reset hidden heat, glue curing, or other accumulated state.
For other models, if segmentation finds no segments at all, the caller retains its input trajectories instead of returning an empty fitting dataset.

Implementation: [trajectory_prep.py](../../predicators/code_sim_learning/trajectory_prep.py), `_windowed_active_steps`, `_rest_mean_state`, `truncate_settled_tail`, and `split_at_rest_points`; [agent_sim_learning_approach.py](../../predicators/approaches/agent_sim_learning_approach.py), `_rollout_fit_trajectories`.

## 4. Carrying an accepted parameter estimate

**“Carry accepted parameter centers” describes this more accurately than “carry posterior.”**
The flag is `code_sim_learning_carry_posterior`.
Assuming “informative seed” means the previous estimate used in the next fit, there is no separate seed-quality test.
The estimate is carried when the parameter's fitting verdict permits deployment:

| Verdict | Main criterion before overrides |
| --- | --- |
| `IDENTIFIED` | Reported width is less than 0.3 times the prior width. |
| `WEAKLY_IDENTIFIED` | Width is less than 0.7 times the prior width; a sharp estimate supported by fewer than two segments is also downgraded to this category. |
| `WIDE` | With interval belief enabled, width is smaller than the prior width and the fitted value moved from its prior center by more than `1e-3` in fitting coordinates. |

Boundary, insensitivity, and anchor-ablation verdicts override these acceptance rules.
The carry function only copies physical parameters whose final verdict has `applies_fitted=True`.
It does not carry their covariance or interval.

This value changes the next fit's **prior center**, grid preferences, and fallback anchor.
It is more than an optimizer initialization.
Prior widths are recomputed by the existing prior-width rule, rather than copied from the previous posterior; for linear parameters that rule can also depend on the new center.

This can be useful when a new task provides little information about a parameter learned earlier: the fitter can retain the learned value instead of reverting to a registry default.
However, the current fitter already pools earlier data, so this is an additional history-dependent regularization choice.
It is not an exact Bayesian update, and not carrying widths does not remove the reuse of information through the data-dependent center.
The evidence reviewed here does not isolate this flag's contribution to solve rate or sample efficiency.

**Simplification:** with pooled data, keep the original prior fixed and use the last solution only to initialize optimization or propose samples.
Alternatively, carry the full approximate posterior and update it with new evidence only.
A continuing trajectory requires its conditional observation likelihood, not treating the new fragment as an independent episode.
If “seed” instead means the optimizer's grid seed, it is selected from low-loss candidates with an anchor preference; it has no separate information-gain test.

Implementation: [agent_sim_learning_approach.py](../../predicators/approaches/agent_sim_learning_approach.py), `fit_prior_anchors` and `note_carried_posterior`; [identifiability.py](../../predicators/code_sim_learning/identifiability.py), `Verdict.applies_fitted`.

## 5. Fit evidence

`code_sim_learning_fit_evidence` adds an approximate marginal-likelihood score to the fit report when the required Jacobian is available.
It is advisory information for comparing simulator versions, not a posterior over program structures or an automatic program-selection rule.

Its reliability depends on both the local approximation and the interpretation of the fitting residuals as a likelihood.
The current comparison checks only that the two fits have the same number of residuals.
That does not establish that they use the same observations, surviving segments, feature scope, or noise model.
Together with the robust and repeated-summary losses below, this limits its interpretation as Bayesian model evidence.

**Simplification:** keep this outside the core state/parameter inference interface until comparisons use a common observation model and explicitly matched data.
For evaluating simulator edits, predictive accuracy on a fixed set of held-out interaction recordings is easier to interpret; these should be learning data, not the evaluation protocol's test levels used for model selection.

Implementation: [evidence.py](../../predicators/code_sim_learning/evidence.py), `laplace_log_evidence`, `comparable_with`, and `format_evidence_lines`.

## 6. Execution state belief

**Yes, subgoal checking during execution is a main use, but the same estimate also supports observation summaries and rollout initialization.**
The flag is `continual_belief_frame`.

For each object, it averages a recent suffix whose noisy features appear stationary, up to eight frames by default.
It reports a mean and a spread of \(\sigma/\sqrt n\), which assumes independent sensor noise and a constant underlying value in that window.
It then draws possible states and reports the fraction on which each predicate holds.
Execution checks use a threshold of at least 0.5 to decide which atoms are likely true; the default number of predicate draws is 16.
This is a majority decision under an approximate belief, not a high-confidence subgoal certificate.
The environment evaluator remains responsible for declaring the level solved.

`sim.run(..., belief_draws=K)` can also test a plan from different initial states using the current belief when the rollout starts at the current observation.
Other initial states can use the declared observation noise as a fallback, so requesting belief draws is not exclusive to this flag.
The frame does not infer a joint posterior over hidden process state and parameters.
Inferred simulator memory is attached separately.

Implementation: [observation_belief.py](../../predicators/observation_belief.py), `smooth_frames`, `belief_draw`, `atom_fractions`, and `likely_atoms`; [continual.py](../../predicators/run/continual.py), `belief`; [belief_probe.py](../../predicators/agent_sdk/belief_probe.py), `_run_belief_draws`.

## What else happens inside `sim.fit()`?

The current subclass path uses recorded-action rollouts of the whole candidate simulator.
It fits the declared engine and agent parameters together; it does not fit a new program structure.
These mechanisms are broader than the six uncertainty flags:

| Mechanism | What it does and why |
| --- | --- |
| Fresh replay environments | Reconstructs each rollout in a fresh simulator so residual engine state cannot change the objective between evaluations. Physical parameters outside the fitted set are pinned to defaults. |
| Rest segmentation and tail truncation | Limits accumulated replay divergence and repeated scoring of settled errors. Preserves full trajectories for models with persistent memory. |
| Feature scaling and angle wrapping | Normalizes positions by observed motion ranges with a floor, wraps angle errors, and incorporates declared sensor noise into the scales. |
| Robust loss | Applies a Huber-style transformation so large residuals contribute linearly rather than quadratically beyond the threshold. |
| Extra summary losses | Adds endpoint and motion-onset errors to the per-step objective. Their default weight is 5, which means a factor of 25 in squared loss. |
| Explainability trimming | Searches candidate physical parameters separately for each segment. Rejects segments whose best tested RMS exceeds `trim_rms_factor * noise_sigma`, with default factor 2. This is a finite candidate search, not proof that no parameters explain the data. |
| Consistency refitting | If surviving segments fit much worse jointly than individually, drops the segment with the largest individual best RMS and refits. Default factor is 3. Records conflicting candidate values for later uncertainty sweeps. |
| Coordinate grid search | Searches physical parameter ranges before local optimization, using multiple passes and refinement near a flat region's edge. Prefers values near the prior anchor when losses are approximately equivalent. |
| MAP optimization | Adds Gaussian-prior residuals to the data objective. Although functions and logs call this “LM,” the actual solver is bounded `scipy.optimize.least_squares(method='trf')`. Uses larger finite-difference steps for simulation than for analytic functions. |
| Search for zero-gradient parameters | For threshold or gate parameters with a flat local data Jacobian, searches over their bounds, then restarts local optimization if it finds an improvement. |
| Anchor-pinned refits | Tries restoring moved physical parameters to their anchors while refitting the others. Accepts a restoration when the data loss stays approximately equivalent and the total prior penalty decreases. |
| Width and deployment rules | Computes the interval heuristics above, imposes a minimum width, checks boundaries and sensitivity, and chooses which physical values to apply. The surrounding approach also has cross-cycle conflict checks on its final fitting path. |
| Caching and publication | Reuses fit computations where applicable and records the simulator version. A canonical `sim.fit()` publishes fitted values; `sim.fit(traj_idxs=[...])` is diagnostic and does not deploy them. |

The six-flag list therefore understates the number of implementation choices in the fitting stack.
Several address simulator mismatch or numerical optimization, rather than perceptual uncertainty.
In particular, an endpoint observation contributes both to the per-step loss and to a weighted summary loss, so those terms are not independent measurements.
The noise-folded scale also combines sensor noise with a motion-relative tolerance; it is not just the declared sensor standard deviation.
Exponentiating this objective defines a useful loss-based distribution, but does not automatically give the posterior of the actual observation-generating process.

Implementation: [synthesis.py](../../predicators/agent_sdk/tools/synthesis.py), `run_fit` and `_evaluate_rollout_fit`; [orchestrator.py](../../predicators/code_sim_learning/orchestrator.py); [physical_sysid.py](../../predicators/code_sim_learning/physical_sysid.py); [rollout_objective.py](../../predicators/code_sim_learning/rollout_objective.py); [grid_seed.py](../../predicators/code_sim_learning/grid_seed.py); [lm.py](../../predicators/code_sim_learning/lm.py).
Defaults and flag definitions are in [settings.py](../../predicators/settings.py), with fitting configuration collected in [config.py](../../predicators/code_sim_learning/config.py).
The six features are enabled together in [protocol_continual_noisy_sweep_r1.yaml](../../scripts/configs/predicatorv3/protocol_continual_noisy_sweep_r1.yaml).

## Can we cleanly compute posterior = likelihood times prior?

**Yes. That should be the statistical definition; numerical approximation is a separate choice.**
For a fixed simulator program \(P\), let \(\theta\) include its unknown fixed parameters and let \(D\) contain observed trajectories and executed actions.
Then

\[
p(\theta\mid D,P) \propto p_0(\theta\mid P)\,p(D\mid\theta,P).
\]

The challenge is specifying \(p(D\mid\theta,P)\) when initial poses, velocities, and hidden process state are uncertain.
A noisy first observation should not be treated as the exact initial simulator state.
For one episode with full latent state \(s_t\), a useful starting model is

\[
s_{t+1}=F_{P,\theta}(s_t,a_t),\qquad
 o_t\sim p(o_t\mid s_t),
\]

\[
p(D\mid\theta,P)
=
\int p(s_0\mid\theta,P)
\prod_{t=0}^{T}p(o_t\mid s_t(\theta,s_0,a_{0:t-1}))\,ds_0.
\]

This integrates over uncertain initial state and counts each observation once.
Known components of the initial state can be fixed; unknown memory or velocity belongs in the latent state.
For independent reset episodes, multiply the episode likelihoods conditional on the shared parameters.
If we condition the initial-state distribution on the first observation instead, we must account for that observation consistently rather than multiply it in twice.

Use the declared observation channel for sensor noise.
If the candidate program misses important dynamics, represent model discrepancy explicitly or diagnose that failure; sensor noise alone should not be expected to absorb every replay error.
Introducing stochastic transition discrepancy is a modeling choice even when the true simulator is deterministic.

My proposed first implementation would keep the original prior and pooled recordings, then use numerical integration for genuinely low-dimensional cases or a derivative-free sampler for larger joint parameter sets.
Weighted samples can provide intervals and planning rollouts from the same distribution, including parameter correlations and multiple modes.
Sequential Monte Carlo samplers are one established option for approximating a sequence of such distributions, with resampling and moves that prevent a fixed candidate set from merely collapsing onto one member ([Del Moral, Doucet, and Jasra, 2006](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf)).
This is a proposed choice for our code, not a demonstrated improvement.

A coordinate grid with other parameters held fixed is not joint posterior integration.
A single local Gaussian can be adequate for smooth, well-constrained models, but should not be the only representation for contact thresholds or multiple plausible parameter combinations.
If the agent changes the simulator program, recompute likelihoods under the new program and reconstruct its hidden state; the old posterior cannot simply be relabeled as the new one.

## Would Bayesian smoothing help?

**Yes, particularly for uncertain initial conditions and hidden dynamics, but smoothing and parameter inference solve different parts of the problem.**
Filtering estimates the current state from observations available so far.
Smoothing estimates past states using later observations as well ([Särkkä and Svensson, 2023](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf)).

For execution, use a causal filter to estimate the current pose, velocity, and relevant hidden state.
For fitting recorded trajectories, a smoother can revise earlier states using the rest of the recording.
A fixed-lag smoother can revise a recent portion of the trajectory online, but does not provide future observations for the current decision.
These methods could replace the separate rest-window averages with a common state-estimation model.

For our environments, the filter should acknowledge uncertainty in the learned dynamics.
A filter driven by a wrong point estimate can confidently smooth toward the wrong trajectory.
Gaussian approximations are plausible within smooth motion regimes; contact and attachment changes motivate considering particle or hybrid representations.
This is a design hypothesis to test, not evidence that a full particle smoother will outperform the current averages.
The rest-window mean remains a useful baseline: under a stationary state and independent Gaussian observations with a flat prior, it already has a simple Bayesian interpretation.

## A simpler design and how to check it

Use three concepts in the method description:

1. **Parameter inference:** a posterior over fixed dynamics parameters for the current simulator program.
2. **State estimation:** a belief over the current physical and hidden state under noisy observations.
3. **Decision making:** use their joint predictive distribution to assess plans, check subgoals, and score informative interactions.

Intervals become posterior summaries rather than a separate mechanism.
Carrying experience becomes either refitting with a fixed prior or updating with new evidence.
The information-seeking score remains a decision criterion.
Fit evidence remains an optional program-comparison diagnostic.
For a rollout, sample one parameter vector and keep it fixed throughout; sample the initial state conditionally, preserving state-parameter correlations when represented.

This organization can simplify the explanation immediately, while the implementation needs staged comparison:

1. Preserve the current behavior as a baseline and replay the same recorded actions through both estimators.
2. First compare fixed-prior pooled inference against carried-center fitting, then compare posterior-derived intervals against the current grid/probe widths.
3. Assess predictive error, interval coverage where simulator truth is available for offline evaluation, hidden-state error, and subgoal false positives and false negatives.
4. Test replacing hard segment rejection and duplicated summary losses with an explicit observation/discrepancy model before removing those safeguards.
5. Run matched online comparisons across the five noisy domains, measuring solve rate, environment steps, resets, and inference cost over multiple seeds.

Use separate development tasks or runs for selecting these choices so the continual evaluation's test tasks remain an evaluation.
Offline replay cannot establish unchanged closed-loop performance because a different belief changes the agent's actions and the data it gathers.
No code changes, feature removals, or new experiments were performed for this note.
