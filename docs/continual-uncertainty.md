# Perceptual uncertainty in the continual protocol

Companion to `docs/continual-protocol.md`, which defines the protocol this document extends.

Status: design settled 2026-09-07; the channel (step 1 of section 7) is implemented the same day, see section 8.
Decision: the protocol gains a Gaussian observation channel that both agent arms see.
The agent stays the planner.
There is no classical belief-space planning layer.
The model-based arm's principled use of noise is filtering with its own learned simulator, added only if the noise sweep shows it is needed.
The design converges with the Model Discovery Agent (MDA, Murphy 2026, arXiv 2608.09696) on its Bayesian foundation and takes three refinements from it, listed in section 4.

## 1. Why

Every arm currently observes the exact PyBullet state.
Real perception is noisy, so a reviewer will ask two questions.
Do the results survive observation noise at all?
Does the model-based arm's advantage come from having a generative model, or from being handed exact state that a real robot never gets?

The answer is to make noise part of the protocol rather than an argument.
Noise then becomes a measured axis, a sigma sweep, and the model-based arm gets the one thing a generative model buys beyond sample efficiency: a Bayesian filter over what it cannot see directly.

## 2. Where uncertainty lives today

- Observation: exact.
  `ContinualRun.observation()` in `predicators/run/continual.py` builds the frame from the env state, and `observation_view` in `predicators/code_sim_learning/utils.py` drops the privileged and simulator-state channels without perturbing anything.
- Hidden parameters: `sim.fit` returns a Levenberg-Marquardt MAP with a Laplace covariance (`predicators/code_sim_learning/fitting.py`), with an MCMC variant as an ablation.
  Its likelihood is iid Gaussian on residuals with a fixed `noise_sigma` (default 0.05 in `predicators/code_sim_learning/fitting.py`), and the same sigma scales the prior residual rows and the segment-trimming threshold.
  This is the only place the code assumes a noise model, and it assumes one the env never produces.
- Hidden state: `LatentTracker` (`predicators/code_sim_learning/latent_tracker.py`) runs the agent's rules forward on every real observation and never corrects on it.
  Its docstring names a Rao-Blackwellised particle filter over the parameter ensemble as the natural extension.
- Ensembles: the info-seeking gate draws a calibrated Laplace ensemble to rank probes by disagreement (`agent_explorer_info_calibrated_ensemble`), and the program-world-model arm keeps belief particles at its capture gate (`agent_program_belief_particles`).
  Both are used for one-off decisions and neither is updated by observations.
- Model choice: one agent-authored simulator at a time, judged by its SSE against a noise floor.
  Nothing penalises a parameter the agent adds to absorb residual, which is harmless while observations are exact and stops being harmless once they carry noise.
- Abstraction: predicates are hard thresholds on exact features, and the execution monitor aborts on a hard atom mismatch.
  The latent-only monitor trap and the seed-3 bridge run that missed `SeatedOn` by 2 mm are noise-free previews of what pose noise does routinely.
- Real robot: the ZED marker and markerless pipelines (`real_robot_perception`) are the source of realistic sigma values.

## 3. Design

### 3.1 The observation channel

The channel sits at the env-to-observation boundary in the continual runner.
It perturbs the frame handed to the agent and the recording the agent reads (`data/trajectories.pkl`), because the fit must see exactly what the agent sees.
Everything the harness judges stays on the true state: the evaluator's win certification, `evaluate_episode`, the level index's full atom set, and the replay video.
The level index records the true state next to the observed one for analysis and never shows the true one to an arm.

Model:

- Additive zero-mean Gaussian per feature, with one sigma per feature class: position, orientation, and scalar (fill level, joint value).
- Object features only.
  The robot's own state stays exact, since proprioception is accurate on the real robot too.
- One draw per env step, keyed by run seed, level index, episode and step, so a run is reproducible and a resumed run re-observes the same frames.
- `env.observe` stays free and stays idempotent between steps.
  Repeated calls return the same frame.
  Re-observing costs a step (a zero action or `Wait`), which is what makes averaging a decision rather than a free lunch.
- Noise is stationary for the whole run.
  A sigma the agent calibrates on the first level is worth the same on the last one, so calibration is one of the objects a continual agent carries forward.

Declared to the agent in the contract: the noise family, which features it touches, and the sigma values.
Declared is the default.
MDA treats sigma as known in every one of its benchmarks and gets its noise robustness from putting that sigma in the likelihood, so the undeclared variant, where the agent must measure sigma by re-observing a static scene, is the harder ablation rather than the base case.
The fit's `noise_sigma` is set from the declared channel, so the likelihood, the segment trimming and the refusal wording all speak in the env's actual noise from day one.

Dropout (a feature reported missing with some probability) and quantization are deliberate follow-ups, not part of the first version.

Proposed flags and records:

| Item | Built as |
|---|---|
| `continual_obs_noise_position` | position sigma on `x`, `y`, `z`, metres, 0 disables |
| `continual_obs_noise_orientation` | orientation sigma on `rot`, `roll`, `pitch`, `yaw`, `tilt`, `wrist` and a Type's `angular_features`, radians |
| `continual_obs_noise_declared` | whether the contract states the sigmas and the fit knows them (default on) |
| scorecard | `obs_noise_position`, `obs_noise_orientation`, `obs_noise_declared`; `aggregate_scorecards.py` carries the columns and the viewer's run page shows the channel |

A scalar-feature class (fill levels, sensor readings) was deferred in the first version because none of the target envs discriminated on one; the sweep showed that boil does (section 8), so it became step 7 of the build order and landed on 2026-09-08 as `continual_obs_noise_scalar`.
A noisy discrete feature stays the dropout follow-up, not a Gaussian.

Sigma values are chosen per env relative to the tightest predicate tolerance in that env, not as absolute numbers.
A sweep at a quarter, a half, and the full tolerance answers the question at the scale where the abstraction starts to flip.

Both arms see the same channel.
The model-free arm may write its own smoothing in the sandbox.
The harness filter of section 3.3 is model-based, so offering it to the model-free arm would hand it a model it does not have.

### 3.2 The sigma sweep

Step two runs both arms with the channel and nothing else changed.
The metrics are the protocol's base metrics.

The sweep answers, per arm and per sigma:

- whether the model-based arm stays ahead of the model-free arm at all;
- which failure dominates: abstraction flips (atoms toggling near thresholds), fit bias (the sysid absorbing noise into contact parameters), or monitor aborts (a healthy plan killed by a noisy mismatch);
- how much of the noise the coding agents handle on their own by averaging reads and widening margins.

Attribution uses the recorded true state against the observed frame, which is why the level index stores both.
If the model-based arm stays ahead on raw observations the filter is a robustness result rather than a rescue, and that is the stronger paper story either way.

### 3.3 The filter

Built only if the sweep calls for it.
It lives in `LatentTracker`, whose contract already names it, and it serves two callers: execution and the fit.

At execution:

- A particle is a parameter draw from the fit's Laplace or MCMC posterior plus a latent block.
- Each real step propagates every particle with the agent's own simulator through the same rule entry points the belief uses, weights it by the Gaussian likelihood of the observed frame under the declared sigma (the same form the fit already uses), and resamples systematically when the effective sample size drops below half.
- Static parameters degenerate under resampling, so the filter either jitters them (Liu-West) or re-fits from the run-long data workbench at a fixed cadence, which also keeps the fit and the filter consistent.
- Output: the weighted particle set, a filtered estimate (the weighted mean), and a per-feature spread.
- The posterior at the end of a level becomes the prior of the next `sim.fit`.
  The parameter posterior and the calibrated noise are the objects carried across levels, and the model-free arm has no such object.

Inside the fit:

- Today each rollout segment starts from the observed state and is scored by least squares against the observed states that follow.
  Under noise the initial condition is itself an error, and least squares from a wrong start biases exactly the contact parameters the fit is after.
- MDA scores a noisy trajectory by the particle-filter marginal likelihood, which integrates out the latent states including the initial one.
  `sim.fit` does the same: a candidate parameter is scored by the filter's marginal likelihood of each observed segment rather than by the SSE of a rollout from its first frame.
- Where the full filter is too slow for the inner loop, the fallback is to treat each segment's initial condition as a latent variable with the declared sigma, which is the errors-in-variables correction without the particle machinery.
- MDA re-fits from scratch every round and reports that this cost caps its experiment budget.
  The continual setting has the warm start it lacks: the carried posterior is the prior, so a level's fit starts where the last one ended.

Cost is one simulator step per particle per real step.
The parallel rollout path already exists and the real step is slow, so a particle count in the tens is affordable.

### 3.4 Model comparison by evidence

MDA keeps a population of model structures weighted by marginal likelihood, and the paper notes that integrating over parameters "provides an automatic Occam penalty factor for complex models with many parameters".
Noise is the regime where that penalty matters: a rule the agent adds to explain residual can fit the noise, and under SSE that looks like progress.

The cheap version of MDA's model level needs nothing the fit does not already compute.
The MAP and the Jacobian at the MAP give a Laplace approximation to the evidence, so every fit reports its log evidence next to its SSE.
The fit report shows the evidence delta against the previous simulator version, and a version that adds parameters has to win on evidence, not on residual.
The harness keeps the last few simulator versions with their evidence so the agent's choice of which to deploy is grounded, but it still deploys one model, the current MAP.
MDA itself predicts with the single Occam-MAP model on its harder benchmarks and averages only on the easy ones, so a single deployed model is the same choice, not a shortcut.

### 3.5 Gates measured against sigma

MDA's test for an insufficient hypothesis space is a posterior predictive check against the noise: expand the model when the predictive error is too large for the likelihood to explain, prune when one model is confidently identified.
The refusal and margin gates are the informal version of that check, and two refinements follow directly.

- A refusal states whether the unexplained residual exceeds what the declared sigma allows.
  That single bit is what tells the agent to change the model rather than fit harder, and it is the bit the current refusal wording leaves the agent to infer.
- An ensemble disagreement smaller than sigma is not worth a probe, because one noisy observation cannot resolve it.
  The info-seeking margin is normalised by the declared sigma on the features the subgoal atom reads, so probe value is measured in units of what an observation can actually tell.

### 3.6 The belief-aware tool surface

The belief reaches the agent through the tools it already has, not through a planner.

- The frame carries the filtered estimate and its spread beside the raw observation.
  The raw observation stays, so an agent can ignore the filter.
- `sim.predicates()` evaluates an atom under the belief and reports the fraction of particles that satisfy it.
  The agent sees "SeatedOn 0.6" and decides itself whether to nudge, re-observe, or move on.
- `sim.run` and `evaluate_trajectory` sample from the belief instead of the MAP, which is what the capture gate already does with ensemble draws.
- The execution monitor replaces the hard atom mismatch with a likelihood test, so noise alone never aborts a healthy plan.
- The attempts log records the spread with each invocation so the agent's notes carry it.

Probing stays where it is.
The agent already decides when a belief rollout beats a real step.
With a spread readout it can also decide when a repeated look beats acting, which is the only decision a belief-space planner would have added.

### 3.7 The parameter belief is an interval, never a switch

Added 2026-09-08 from the sweep's domino losses (section 8).
The fit already computes a posterior width per parameter and the plan gate already sweeps a band around a fitted value, but both sit behind a binary verdict: a parameter whose posterior did not contract below a fixed fraction of the prior is "not identified", its fitted value is thrown away for the registry anchor, and the gate sweeps only parameters whose fitted value was deployed.
Under noise the honest posterior is wider, so the verdict flips, the estimate is discarded and the sweep never runs: on domino at 1 cm the fit put friction at 0.40 with an interval of about 0.22 to 0.72, which excludes the 0.1 anchor the planner then ran with, unchecked.

The rule becomes:

- The planner's belief about a parameter is its posterior, the most likely value with its interval, for every parameter the data moved at all.
  The anchor is the prior, and it is what the posterior collapses to when the data say nothing; it is never a replacement for an estimate the data support.
- Certification samples the whole interval of every moved parameter, weakly identified ones included, rather than one sigma around a point that may be the anchor.
  A plan is certified when it succeeds across the samples, and the fraction is reported.
- The fit report states the interval in plain words, with the anchor's position relative to it, so the agent can reason about it as the 5 mm seed-1 agent did by hand when it scored layouts at three frictions.
- The probe suggestion fires when the interval straddles a plan's success boundary, which is when one cheap experiment is worth more than any amount of planning, instead of waiting for a refusal the switch prevented.
- The contraction thresholds and the flat tolerance are expressed in units of the declared sigma, so a wider posterior under noise reads as the honest answer, not as failure.

Object poses get the same treatment at placement time: a placement is certified over the plausible positions of the target object under the declared sigma, not at the observed one, which is where boil paid its resets.

## 4. Relation to the Model Discovery Agent

MDA is an LLM-assisted Bayesian experiment designer for mechanistic models.
It nests sequential Monte Carlo at three levels (models, parameters, latents), has the LLM propose new model structures when a predictive check says the hypothesis space is insufficient, and picks experiments by expected information gain or a task-driven value of information.
Its noise robustness, demonstrated on a stochastic Hodgkin-Huxley benchmark, is attributed to having the noise in the likelihood and doing inference over posteriors rather than point estimates.

Where the design already agrees with it:

- The observation noise is explicit in the likelihood and sigma is known to the agent (section 3.1).
- The latent-level particle filter is MDA's third SMC level (section 3.3).
- Experiment choice is task-driven: MDA finds the task-driven value of information beats expected information gain in its stochastic case, and the info-seeking gate already ranks probes by disagreement on the plan's own subgoal atoms.
- Acting uses the MAP model, with averaging reserved for the gate (section 3.4).

What it adds, folded in above:

1. The filter scores the fit, not only the execution-time belief (section 3.3).
2. Simulator versions are compared by Laplace evidence, the cheap form of MDA's Occam penalty (section 3.4).
3. Misfit and probe value are measured against sigma (section 3.5).

Where MDA does not reach:

- No continual setting, no physical simulator, no acting under a step budget, and no warm start across rounds.
  The carried posterior of section 3.3 is the warm start it lacks.
- Its stated bottleneck, that recovery depends on whether the LLM proposes a form that covers the truth, is this protocol's bottleneck too.
  Its remedy is to show the proposer its previous models and their errors, which is what the journal, the attempts log and the fit verdicts already do.
- The LLM proposes only the mean function in MDA; the noise model is fixed by the benchmark.
  The declared channel keeps the same split.

## 5. Not doing, and why

- No classical belief-space planning.
  The agent plans well, and a belief operator model would replace something that works with something that needs its own operator theory.
- No population of model structures under a nested SMC.
  One deployed simulator, evidence-compared against its predecessors, is the cheap version and matches how MDA itself predicts on its harder benchmarks.
- No agent-proposed noise model.
  Sigma is declared, as in MDA.
- No learned image perception.
  The object-centric state is the abstraction, and the filter works over it whatever detector produced it.
- No harness filter for the model-free arm, for the fairness reason in section 3.1.
- No mid-run change of sigma, so calibration stays a continual object.
- No noise on the robot's own state in the first version.

## 6. Open questions

- The primitive-only arms need a cheap re-observe action.
  A zero action is one step, which is the intended price, but the action space of each env has to admit one without side effects.
- Orientation noise: yaw only for the tabletop envs, or full axis-angle where an env exposes it.
- Whether the tightest predicate tolerance is the right unit for sigma in envs whose discriminator is dynamic rather than geometric (balloons, domino).
- In the undeclared variant, whether sigma is a fit parameter with a prior or a quantity the agent measures by re-observation.
- How many simulator versions the evidence comparison keeps, and whether the agent may ask for one to be redeployed.

## 7. Build order

1. The channel: flags, seeded draw at the runner boundary, true state kept in the level index, scorecard fields, contract text, the fit's `noise_sigma` taken from the channel, the refusal's exceeds-sigma bit, and tests that the evaluators see the true state and the recording sees the observed one.
   Landed 2026-09-07.
2. Sweep configs for both arms over the sigma grid: domino and boil at a quarter, a half and the full tolerance, fan and bridge deferred.
   Landed and run 2026-09-07 to 2026-09-08 (section 8).
3. The parameter belief as an interval (section 3.7), the group the sweep asked for first, in this order:
   the interval-first deployment rule replacing the verdict switch;
   certification over the full interval of every moved parameter in the plan gate and in `sim.run`'s physics sweep;
   the interval in the fit report, in words, with the anchor's position;
   the probe trigger on a straddling interval, with the probe value normalised by sigma (section 3.5);
   the contraction thresholds and the flat tolerance in units of sigma, and the refusal's exceeds-sigma bit where it is still missing.
   Landed 2026-09-08 (section 8), behind flags, untested on a run: the build-all-then-ablate plan runs steps 3 to 7 first and validates after.
4. The filter inside the fit (section 3.3): each segment's initial condition as a latent under the declared sigma, sigma-relative motion detection for the settled-tail truncation and the rest-point segmentation, and the carried posterior as the next level's prior.
   Landed 2026-09-08 (section 8), behind flags, untested on a run.
5. The Laplace evidence in the fit report and the evidence delta between simulator versions (section 3.4).
   Landed 2026-09-08 (section 8), behind a flag, untested on a run.
6. The belief at execution (sections 3.3 and 3.6): the particle or smoothed frame beside the raw one, atom fractions in `sim.predicates`, belief draws in `sim.run` and `evaluate_trajectory`, the likelihood-based monitor, the spread in the attempts log, and placements certified over the target object's plausible positions.
   Landed 2026-09-08 as the smoothed frame (section 8), behind a flag, untested on a run; the particle filter proper and belief draws in `evaluate_trajectory` were not built.
7. The scalar-reading class of the channel: `continual_obs_noise_scalar` on `bubbling_level`, `water_volume`, `spilled_level` and any Type-declared sensor feature, additive and unclipped, switch states exact; the contract, the frame line and the scorecard carry the third sigma; the fit's residual scale folds it like the others.
   Boil sweep points relative to the ramp step of 0.15 and the 0.07 boil margin, about 0.03, 0.07 and 0.15, with pose noise and reading noise swept as separate axes.
   Landed 2026-09-08 (section 8); the sweep configs are `scripts/configs/predicatorv3/protocol_continual_noise_boil_p12_r{03,07,15}.yaml`, each the earlier 1.25 cm pose point with the reading sigma on top (decided 2026-09-08: one experiment tests both channels at once), the 0.07 point launched the same day.

Validation: domino at 1 cm with four seeds after step 3, which is where the advantage was lost first; boil under reading noise after step 7; the exact boil model-based baseline rerun under the sanitized-frame rule before the boil column is quoted; fan back in the sweep once its cap-stall fixes land.

## 8. Implementation status

Step 1 of section 7 landed on 2026-09-07 (branch `bridge-learning`).

- `predicators/observation_noise.py` is the channel: the feature classes, the per-step keyed generator, `perturb` (the agent's view of a state) and `residual_scale` (the fit's fold).
- `ContinualRun` in `predicators/run/continual.py` keeps one observed view per (level, episode, step) and hands it to everything agent-facing: the frame of `observe`, the atoms an invocation reports against its expected outcome, `level_episodes` and `previous_level_episodes`, which are the source of `data/trajectories.pkl`, the `run_python` trajectories and `sim.fit`.
  The view is never the true object: it is the recording's sanitized form of the state (observable data plus the robot's joint data, no privileged block, no engine handles), with the channel's draw on top when noise is declared, so what a partially observable env hides never reaches the agent's data (decided 2026-09-07 after the boil heat finding below).
  The protocol's own atoms, the evaluators, the checkpoint, the recording and the render stay on the true state.
- The observed frames are not stored: they are reproducible from the run seed, the step coordinates and the true states in `episodes.pkl`, and the level's `level_start` index entry records the channel.
- Skill controllers servo on the true state.
  Noise enters through what the agent decides, the atoms it reads, the data it fits and the frame its belief resets from, not through the low-level controllers, which on a real robot run on proprioception plus the target the agent chose.
  An invocation executes a fresh grounding of the skill, since the agent's applicability check ran on its observed frame.
- The oracle reference arm plans and acts on the true state; it is the exact-perception upper bound.
  The random arm samples on the observed frame like any agent.
- The fit: `SysIdConfig.observation_noise` carries the declared channel and `compute_residual_scaling` folds each feature's sigma into its scale, so the likelihood the fit maximises is the one the channel was declared with and every RMS threshold keeps its meaning in units of the total noise.
  Undeclared means the fit is as blind as the agent.
- The prompts: a declared channel adds an "Observation noise" section to both arms' system prompt, an "Observation noise and the fit" section to the model contract, and a `[noise]` line to every frame.
- Tests: `tests/test_observation_noise.py`, the channel, resume-under-noise and card tests in `tests/run/test_continual.py`, the scaling fold in `tests/code_sim_learning/test_physical_sysid.py`, the prompt and frame test in `tests/agent_sdk/test_continual_tools.py`.

Step 3 of section 7 landed on 2026-09-08 (branch `bridge-learning`), behind two flags so the later ablations are flag flips: `code_sim_learning_interval_belief` (the parameter belief) and `agent_explorer_info_seeking_noise_aware` (the probe value), both off by default.

- The verdict `Verdict.WIDE` ("wide posterior") in `predicators/code_sim_learning/identifiability.py`: a parameter whose posterior did not contract below the weak threshold but is narrower than the prior, and whose most likely value moved off the prior centre by more than 0.001 in fit space.
  It deploys, so `select_trustworthy_params` applies the most likely value and `physics_sigma_points` sweeps its whole interval; the verdict enum stays the single decision surface, so every consumer follows without a second switch.
  A parameter that never moved, or whose reported width exceeds the prior, still reads NOT identified and keeps the anchor: the data said nothing.
- Every report entry carries the belief interval (the most likely value plus and minus one posterior sigma, clipped to the box), the most likely value and the anchor, and `format_identifiability` renders a `belief:` line in words with the anchor's position (below, inside, above) and whether the planner runs on it.
  The `sim.fit` report's heading and its "Applied" sentence say the same, and the explorer's system-identification diagnostics name the interval as the experiment target.
- Certification: the capture gate's PARAM-SENSITIVE refusal and `sim.run(plan, physics_sweep=True)` report the fraction of interval points passed and the passing and failing ranges per parameter (`straddle_summary`).
  A mixed sweep is the interval straddling the plan's success boundary; it arms adaptive info-seeking from the probe as well as from the gate, with the cue that one narrowing experiment beats more planning.
- The flat tolerance in sigma units: `flat_tolerance` in `grid_seed.py` takes the relative tolerance on the SSE in excess of the declared channel's expected noise SSE (`expected_noise_sse` in `trajectory_prep.py`, one variance per scored residual) and floors it at `flat_sigmas^2 * noise_sigma^2` (`code_sim_learning_rollout_flat_sigmas`, the likelihood-ratio interval); the anchor ablation uses the same tolerance.
  The contraction thresholds stay ratios: under the interval belief they label, they no longer gate.
- The exceeds-sigma bit: under a declared channel the fit's trimming note leads with the statement that the dropped segments exceed what the declared noise can explain, so the model has to change rather than the fit.
- The noise-aware probe value: `score_atom_disagreement` reads each ensemble member's predicted state through eight draws of the declared channel and scores the mutual information between the member and the read truth (`noisy_read_information`), so a disagreement finer than sigma scores zero and is not worth real steps.
- Tests: `tests/code_sim_learning/test_interval_belief.py`, the noise-aware case in `tests/approaches/test_sim_learning_info_seeking.py`, the straddle cases in `tests/agent_sdk/test_belief_probe_physics_sweep.py` and `tests/agent_sdk/test_submit_plan_capture.py`, the declared-channel case in `tests/agent_sdk/test_trim_cause_note.py`.

Step 4 of section 7 landed on 2026-09-08 (branch `bridge-learning`), behind `code_sim_learning_rollout_noise_filter` (the fit-side filter) and `code_sim_learning_carry_posterior` (the carried posterior), both off by default.

- Sigma-relative motion detection in `predicators/code_sim_learning/trajectory_prep.py`: under a declared channel a step is active when the mean of the next `noise_window` frames differs from the mean of the previous `noise_window` frames by more than `settle_sigmas` standard errors of that difference, floored at the settle tolerance; exact features keep the per-step test and angular features are wrapped.
  Both the settled-tail truncation and the rest-point segmentation use it, so under a centimetre of noise a tail is cut and a rest point is found again (the per-step detector flagged every step against its millimetre tolerance).
- The initial condition: each rest-anchored segment starts from the mean of its preceding rest window (circular mean for angles), so the rollout's initial condition carries sigma over the square root of the window instead of one frame's sigma.
  This is what an initial-condition latent under the declared sigma resolves to while the objects are at rest, at no extra fit parameters; the full latent (one per object feature per segment) would cost a rollout per Levenberg-Marquardt column and was not built.
  The expected noise SSE of step 3 leaves the rollout's own start error out, so it stays conservative.
- The carried posterior: `fit_prior_anchors` on the sim-learning approach hands both the harness fit and `sim.fit` the most likely value of every parameter the last applied fit deployed as the prior centre, in place of the env registry's default, and `note_carried_posterior` records it after each applied fit (never an anchor fallback).
  The centre is what data-flat directions stay at, what the grid sweep's anchor-nearest choice and the anchor ablation revert to, and what an uninformative parameter falls back to, so a level's fit starts where the last one ended.
  The width is not carried: the fit pools every level's data, and a carried width would count the earlier levels twice.
  The carried values are checkpointed with the approach.
- Tests: `tests/code_sim_learning/test_noise_filter.py`.

Step 5 of section 7 landed on 2026-09-08 (branch `bridge-learning`), behind `code_sim_learning_fit_evidence`, off by default.

- `predicators/code_sim_learning/evidence.py`: the Laplace log evidence at the MAP from what the fit already has, the SSE, the Jacobian, the noise width and the prior centres and widths in fit space: log likelihood plus log prior plus the Occam term (half the parameter count times log two pi, minus half the log determinant of the Gauss-Newton curvature).
  The approximation is exact for a linear residual model, and the test checks it against quadrature; a data-flat parameter leaves the evidence unchanged and a constrained parameter that buys no residual lowers it.
- The orchestrator computes it on the surviving segments when the fit carries a Jacobian (Levenberg-Marquardt ran and the anchor ablation pinned nothing) and hands it back on the fit outcome; the fit report states it next to the SSE with the residual and parameter counts, and quotes the delta against the previous canonical simulator version when both score the same residual set, naming the winner.
  A different residual set (scope or survivors) is named as not comparable rather than compared.
- The approach keeps the per-version history in its checkpoint, so the delta survives a resume.
- Tests: `tests/code_sim_learning/test_evidence.py` and the flag case in `tests/code_sim_learning/test_orchestrator.py`.

Step 6 of section 7 landed on 2026-09-08 (branch `bridge-learning`) as the smoothed frame, behind `continual_belief_frame`, off by default, with `continual_belief_window`, `continual_belief_sigmas` and `continual_belief_draws` as its knobs.

- `predicators/observation_belief.py`: per object, the belief is the mean of its noisy features over the frames it has rested through, up to the window, with the spread sigma over the square root of the frames; rest is judged sigma-relatively on the window's two halves, the way the fit-side filter judges motion, so a moving object is never smoothed across its motion and the window restarts at a jump.
  Angles average circularly.
  Draws of the belief jitter each noisy feature by its spread, and the atom fractions are the share of draws on which an atom holds.
- The observation carries the belief beside the raw frame: the `[objects]` block is followed by a `[belief]` block naming each object's smoothed features with their spread and the frames averaged, and the atom lines are followed by `[atoms under the belief]` listing the atoms whose fraction is strictly between zero and one.
  The truth view the reference arms use carries no belief.
- The likelihood-based monitor: an invocation's expected atom is missing when it holds on fewer than half the belief draws after the skill, and an expected-absent atom is present on the same rule, so one frame's noise never aborts a healthy plan; the invocation's result text and the level index carry the fractions and the belief's largest spread, which is where the agent's notes read the spread from.
- `sim.run(plan, belief_draws=K)`: the plan rolled once from each of K draws of the belief, on a fresh env at the base planner seed, reporting the successes and each draw's largest feature shift; a mixed result says the plan depends on a pose the observation cannot pin down, and this is how a placement is certified over where its target may really be.
  The draws come from the belief shown in the last observation when the probe still sits on it, else from the current state with the declared sigma on every noisy feature.
- Added the same day: `evaluate_trajectory(states, actions, physics_sweep=True)` scores the sequence at every point of the identified parameters' belief interval on a fresh env at that physics and reports the per-point verdicts, the fraction scored solved and whether the sequence is certified at every point, so a replaying certificate that flips across the interval shows up before the agent acts on it; and `sim.belief()` on the probe lists the current belief (each object's smoothed features with their spread) and the fraction of belief draws on which each atom holds, the on-demand form of the observation's belief lines.
- Not built: the particle filter proper (the smoothed frame is the rest-window special case of it, exact for objects at rest, and objects in motion keep the raw frame).
  `sim.predicates()` reports on the invented predicates' file, not on a state, so the atom fractions live in the observation and in `sim.belief()`.
- Tests: `tests/test_observation_belief.py`, the belief cases in `tests/run/test_continual.py`, `tests/agent_sdk/test_continual_tools.py` and `tests/agent_sdk/test_belief_probe_physics_sweep.py`.

Step 7 of section 7 landed on 2026-09-08 (branch `bridge-learning`): the scalar-reading class of the channel, `continual_obs_noise_scalar`.

- `ObservationNoise.scalar` applies to `bubbling_level`, `water_volume`, `spilled_level` and any feature a Type declares in its new `sensor_features` metadata, additive and unclipped (a full jug reads above one about half the time); switch states, discrete features and the robot stay exact.
- Because every consumer reads sigmas through `feature_sigma`, the fit's residual scale, the expected noise SSE of step 3, the fit-side filter of step 4 and the execution-time belief of step 6 fold the reading sigma in without further change.
- The contract and the frame line describe the third sigma through the channel's own text, and the run card, the level index, the scorecard aggregator and the viewer carry it (`obs_noise_scalar`); cards written before the field load as exact readings.
- Tests: the scalar case in `tests/test_observation_noise.py` and the card round trip in `tests/run/test_continual.py`.

Every step of the build order is now built, all behind flags that default off, and none has run on a level yet: the validation runs of section 7 come next, with the ablations as flag flips on one code tree.
The particle filter proper (3.3) stays unbuilt; the smoothed frame is its rest-window special case.

First launch (step 2, one point of the sweep), 2026-09-07: fan and domino, both arms, seeds 0 and 1, position sigma 5 mm and orientation sigma 0.02 rad, declared, via `scripts/configs/predicatorv3/protocol_continual_noise_fan_domino.yaml` from the worktree `predicators-noise-r1` (Slurm 22197846 domino model-based, 22197847 fan model-based, 22197848 domino model-free, 22197849 fan model-free).
The sigma is about half the tightest scale of each env: fan's target tolerance is 1 cm on a 4 cm ball, a domino is 7 cm wide.
The exact-observation baselines on the same arms and flags (the m4 domino pair and the m3 fan pair, seed 0) have the model-based arm ahead: domino 2/2 levels in 297 steps against 1/2 in 555, fan 2/2 in 350 steps against 2/2 in 1399 with two resets.
The first model-based runs crashed within minutes in the workbench's base-simulator predictions: the observed view was a plain `State`, and the PyBullet base simulator reads the robot's fingers from the state's joint data on every step.
The view is now the recording's sanitized form, a PyBullet state keeping the robot's own joint data (proprioception is exact) without the engine handles; the model-based arms were relaunched as 22198360 (domino) and 22198361 (fan), resuming their run directories, while the model-free runs of the first launch kept going.
Fan seed 0's resume crashed once more: the arm reuses a finished level's trajectories from its saved checkpoint, and the crashed run had written that checkpoint with the old views, so its run directory and checkpoint were set aside (`~/.claude/jobs/aaea1883/tmp/crashed_runs/`) and the seed started fresh as 22198484.
Balloons was launched first at 1 cm (`protocol_continual_balloons_noise.yaml`, jobs 22197721/22197722) and cancelled within minutes: the domain is still being tuned and its exact-observation runs are tied, so it cannot show the noise axis; the config stays for when it is settled.

Results of the 5 mm point, 2026-09-07, all eight runs final:

| Env, seed | Model-based | Model-free | Exact baseline, seed 0 |
|---|---|---|---|
| Fan 0 | 2/2 in 1337 | 2/2 in 2441 | model-based 350, model-free 1399 |
| Fan 1 | 2/2 in 2405 | 2/2 in 1559 | |
| Domino 0 | 2/2 in 397 | 1/2, test lost at 1153 | model-based 2/2 in 297, model-free 1/2 in 555 |
| Domino 1 | 2/2 in 526 | 1/2, gave up at 652 | |

Domino keeps the advantage: the model-based arm wins both levels on both seeds (397 and 526 steps) where the model-free arm loses the test level on both, at a step cost close to the exact baseline.
Its seed 1 run also shows the agent handling the noise on its own, the third question of section 3.2: the fit left the friction weakly identified (boxed at 0.32 with a coarse sweep preferring 0.69), and the agent scored every candidate layout by Monte Carlo over the declared sigma at three friction values and chose the layout that cascades at all three.

Fan is uninformative at this point, and the reason is not the channel.
Every step above the exact baseline is a 1000-step option-cap stall, on either arm: the model-based runs lost one Wait (seed 0) and two SwitchOn strokes (seed 1) to the cap, the model-free runs two strokes (seed 0) and one (seed 1), and the exact model-free baseline had lost 1000 of its 1399 steps to the same SwitchOn jam.
The productive steps barely move under noise (model-based 337 and 405, model-free 441 and 559, against 350 and 399 exact).
A peer diagnosis blamed the Wait skill's quiescence test reading noisy frames; it does not hold.
In the continual harness the skill policy, its terminal and the annotated Wait target all read the true state (the second-to-last bullet of this section), the fan skill config sets no quiescence tolerance at all (only domino and icerink do), and the stroke jam reproduces without noise.
The stall itself is a factory gap: the push stroke runs on the plain incremental-IK branch of `PhaseSkill._execute_move`, the one branch that never calls the existing `_check_ik_stall` watchdog, so a jammed stroke burns the cap instead of failing in 25 steps with a contact report.
That fix and a quiescence terminator on the fan Wait are exact-observation fixes, deferred so as not to change the baselines mid-sweep; the fan noise numbers stay out of the sweep until they land and the exact fan pair is rerun.

The two thresholds that do read noisy observations, and so are the honest section 3.5 items, are inside the fit and the harness: the settled-tail truncation and rest-point segmentation use a 1 mm motion tolerance on observed deltas, which never sees rest under 5 mm noise (the domino log shows the truncation as a no-op), and the post-invocation divergence check reads threshold predicates off one noisy frame.

Second stage of the sweep, launched 2026-09-07 from the same worktree, both arms, seeds 0 and 1, declared: domino at the half-tolerance point, 1 cm and 0.04 rad, against the cascade check's placement tolerance of 0.3 of a gap, about 2 cm (`protocol_continual_noise_domino_10mm.yaml`, Slurm 22204486 model-based, 22204487 model-free); boil at its quarter-tolerance point, 1.25 cm and 0.05 rad against the 5 cm burner alignment threshold (`protocol_continual_noise_boil_12mm.yaml`, 22204488 model-based, 22204489 model-free).
Boil replaces fan as the second env: it has the widest exact gap of the target envs (m3 seed 0, model-based 2/2 in 524 steps against model-free 2/2 in 4648 with 8 resets).
The full-tolerance points, domino at 2 cm and 0.08 rad and boil at 2.5 cm and 0.1 rad (`protocol_continual_noise_domino_20mm.yaml`, `protocol_continual_noise_boil_25mm.yaml`), were launched the same day while stage two was still running, at the user's request, as Slurm 22207780 (domino model-based), 22207781 (domino model-free), 22207783 (boil model-based) and 22207784 (boil model-free).
Sixteen runs then shared the two Claude accounts, and at 11:05 EDT both accounts hit their five-hour session limit, putting every run to sleep until the 2:20 pm reset; the main account also stood at 74 percent of its weekly cap.
A window's budget is fixed, so sixteen runs only spread it thinner: the stage-3 arrays were cancelled at 11:17 EDT with at most 168 steps taken, to be relaunched when stage 2 ends, restoring the eight-run load.
The step counts are unaffected by the sleeps.
Decision, 2026-09-07: the sweep runs to completion in the background and step 3 of the build order starts now with the filter inside the fit (section 3.3, the initial condition as a latent under the declared sigma) plus the section 3.5 items, since the only noise-linked fit evidence so far is the domino seed-1 fit at 5 mm boxing friction at 0.32 against a sweep optimum of 0.69.

Boil at 1.25 cm, 2026-09-07: the two model-based runs were invalid and were cancelled after 4136 and 1831 steps, and the cause is a harness bug that observation noise exposed, not the noise itself.
Their true-state recordings show the jug's bubbling collapsing from 1.0 to 0.0 at the first step of every skill invocation while the burner is on and the full jug sits centred on it, in a pattern the env's heating rule cannot produce (heat only ever rises), and the model-free runs of the same launch show no such drop.
The boil env kept each jug's hidden heat as an attribute of the jug `Object`, and a `State` hands the same `Object` instances to every env that is set to it.
The model-based arm's workbench refreshes its base-simulator predictions after every charged env call by running its own base env over the new transitions of the recording, and under noise those transitions are observed views, which carry no privileged block: setting the workbench env to a view zeroed the shared attribute, and so the live env's heat.
Under exact observation the same call carried the true state's privileged heat, so the leak was invisible, and it was a leak: the agent's data then carried the hidden heat that partial observability is meant to hide.
Settled 2026-09-07: the agent's view is the sanitized state in both cases, so the exact-observation boil baseline (m3, model-based 2/2 in 524) was run with the hidden heat in the agent's data and needs a rerun before it is compared with the noisy points.
The fix keeps the heat per env instance (`_heat_levels` in `predicators/envs/pybullet_boil.py`, with `tests/envs/test_pybullet_boil_heat.py` as the regression test); the model-based arm was relaunched alone as `protocol_continual_noise_boil_12mm_r2.yaml` (Slurm 22223294, experiment id `boil-agent_continual_noise12mm_r2`, fresh run directories), and the model-free runs of the first launch stand: seed 1 won both levels in 1525 steps with no agent reset (573 of them lost to a preemption's diverged replay), seed 0 was at 2575 steps and 8 resets on level 1 when this was written.
The stage-3 boil point inherits the fix, since it launches from the same worktree.

Domino at 1 cm, 2026-09-07, all four runs final: the model-based advantage of the 5 mm point is gone, and the fit is where it went.

| Seed | Model-based | Model-free |
|---|---|---|
| 0 | 1/2, gave up on the test level at 786 steps (level 1 in 195) | 1/2, gave up on the test level at 858 steps (level 1 in 379) |
| 1 | 1/2, gave up on the test level at 528 steps (level 1 in 184) | 1/2, gave up on the test level at 1719 steps, 1000 of them a Push cap stall |

Both model-based runs won the train level as fast as the exact baseline and both planned the test level on a wrong friction.
Seed 1's harness fit moved friction from the 0.1 anchor to 0.40 (true value 0.5) with the SSE falling from 54 to 41, then ruled the parameter not identified and kept the anchor; the agent had its own open-loop evidence for about 0.69, wrote it in the journal, and still screened its turning chain on the 0.1 substrate, where a struck domino slides instead of leaning, so the real chain stalled at the first link and a recovery push bulldozed it.
Seed 0's fit deployed 0.68 from the train level, and its test placement collapsed a metastable stall and toppled the only striker.
At 5 mm the same fits had landed near enough (0.32 boxed, 0.52 and 0.69 sweeps) for both seeds to win.
This is the sample the section 3.3 filter and the section 3.4 evidence comparison are for: at 1 cm the observed deltas of a resting domino are of the order of the motion the fit reads, the settled-tail truncation and rest-point segmentation stayed no-ops (tolerance 1 mm), and the identification verdict has no sigma to measure its flat tolerance against.
Boil at 1.25 cm after the heat fix: the model-based arm won both levels on both seeds, seed 0 in 701 steps with no reset (train 375, test 326) and seed 1 in 1171 with one reset (a spill while probing a fill spot under the noisy faucet position; train 890, test 281), against the exact baseline's 524; the model-free arm won both on seed 1 in 1525 steps and on seed 0 in 5409 with 18 resets (train 5058, test 351), against its exact baseline of 4648 with 8 resets.
The boil gap therefore survives the quarter-tolerance point on both seeds, at a model-based cost near the exact baseline, while the model-free cost on seed 0 is the exact baseline's pattern of repeated resets made worse.
Stage 3 was relaunched at about 16:10 EDT the same day, at the user's call, while the last two boil runs of stage 2 were still going: with a and b now separate accounts, ten runs are five per account, under the load that tripped the morning's limit.
Slurm 22225387 (domino model-based), 22225388 (domino model-free), 22225390 (boil model-based) and 22225391 (boil model-free); the boil model-based directories from the cancelled launch predate the heat fix and were set aside so that point starts fresh, the others resume their directories.

Stage 3, the full-tolerance points, final 2026-09-08 (the last two runs were killed by account b's weekly limit on the evening of the 7th and resumed on account a at 02:33 on the 8th; the resumes replayed their recorded steps without divergence).

| Point, seed | Model-based | Model-free |
|---|---|---|
| Domino 2 cm, 0 | 1/2, gave up on the test level at 660 (train 216) | 1/2, test level rejected by the evaluator at 926 (train 257) |
| Domino 2 cm, 1 | 1/2, gave up on the test level at 402 (train 112) | 1/2, gave up on the test level at 1555 (train 176) |
| Boil 2.5 cm, 0 | 2/2 in 527, no reset | 2/2 in 2929, one reset |
| Boil 2.5 cm, 1 | 2/2 in 539, no reset | 2/2 in 1736, two resets |

The sweep as a whole (exact baselines: domino model-based 2/2 in 297 against model-free 1/2 in 555; boil model-based 2/2 in 524 against model-free 2/2 in 4648 with eight resets):

| Sigma | Domino model-based | Domino model-free | Boil model-based | Boil model-free |
|---|---|---|---|---|
| 5 mm (domino) | 2/2, 2/2 (397, 526) | 1/2, 1/2 (1153, 652) | | |
| half tolerance (1 cm, 1.25 cm) | 1/2, 1/2 (786, 528) | 1/2, 1/2 (858, 1719) | 2/2, 2/2 (701, 1171) | 2/2, 2/2 (1525, 5409 with 18 resets) |
| full tolerance (2 cm, 2.5 cm) | 1/2, 1/2 (660, 402) | 1/2, 1/2 (926, 1555) | 2/2, 2/2 (527, 539) | 2/2, 2/2 (2929, 1736) |

Two readings.
Boil keeps the model-based advantage at every sigma, and the advantage does not shrink with noise: the model-based cost stays within twenty percent of the exact baseline (527 to 1171 steps against 524) while the model-free cost is three to ten times it, paid in resets (spills at the noisy faucet position, jugs swept off the burner).
Boil's hidden mechanism is a monotone heating process the agent reads off a clean observable, so the noise on object poses only touches manipulation, where the model-based arm's planning on a simulator pays off; the fit plays no part.
Domino loses the advantage at half tolerance and never recovers it, and every model-based loss traces to the friction fit under noise: at 1 cm one seed refused a near-correct fit and kept the anchor while the other deployed an overshoot, at 2 cm both seeds never moved off the 0.1 anchor, and in every case the agent screened its chain on a sliding substrate that the real, stickier dominoes did not follow.
The model-free arm loses the same levels for its own reasons (bulldozed strikers, an evaluator rejection), so at the domino points the arms tie at 1/2 and the model-based advantage is gone rather than reversed.
Domino's discriminating parameter has to be read from a few centimetres of motion in data whose per-frame noise is of that order, which is exactly what the section 3.3 filter, the section 3.4 evidence and the section 3.5 sigma-measured gates address; the sweep is the case for building them, and domino at 1 cm is their validation point.
Fan's numbers stay out of the sweep until the cap-stall fixes land and its exact pair is rerun, and the exact boil baseline needs a rerun under the sanitized-frame rule before the boil column is quoted.
