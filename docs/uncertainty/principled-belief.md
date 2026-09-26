# Principled belief: implementation spec

This spec turns the design in the paper's Section 3 and Appendix B (B.3 belief construction, B.4 rehearsal, execution and monitoring) into code.
The main text is the reference: where the appendix or this spec disagrees with it, the appendix and this spec change.
It replaces the legacy uncertainty mechanisms (landscape widths with a max floor, the disagreement hull, the diagonal physics sweep, the Laplace rule ensemble, one shared program memory) with one belief that every decision draws from.
The parameter factor is the posterior of the fit's own model; nothing is added to widen it.
The legacy path stays behind `belief_joint_draws = 0` until the new agent has passed a matched comparison, then it is deleted.

## Target

The model state is `x = (x_base, x_res)`: the base simulator's state and the program's declared memory, mirroring `theta = (theta_base, theta_res)`.
The belief is `p(theta, x_t | D, H_t) ~= q(theta) q(x_base_t | H_t) delta(x_res_t - G_theta(H_t))`.
Planning, monitoring and experiment selection all average over the same `K` joint draws `(theta_i, x_t_i)` (paper Section 3.2).

- `q(theta)` is a product of per-parameter line factors through the MAP of a generalized posterior of the fit's replay loss, with the temperature estimated from the misfit.
- `q(x_base_t | H_t)` is the posterior of each object's noisy features under a still-since-last-motion model with a prior uniform over the feature's recorded range, averaged over the unknown run length by Bayesian online change-point detection (Adams and MacKay, 2007) with a constant hazard and at most `continual_belief_window` frames; it replaces the split-half threshold test in `observation_belief._rest_window`.
- `G_theta(H_t)` is the program's memory update run over the episode prefix under one parameter draw, through one `LatentTracker` per draw.
  It reads the received (noisy) observations where the simulator's own update reads noise-free simulated features, so it is a plug-in estimate, like the replay starts (paper B.3).

## Parameter belief (`code_sim_learning/parameter_belief.py`)

`q(theta)` approximates the generalized posterior (Bissiri, Holmes and Walker, 2016) of the rollout fit's own loss: `p0(theta) * exp(-E(theta) / (2 * lambda))`, with the Gaussian prior the fit folds in and `E` the fit's replay loss in units of the nominal width `sigma_n`.
Per-step errors are standardized by sensor noise combined with 5% of each feature's range (pi for angles) and Huber-capped at 20 nominal widths (`huber_delta = 1.0` on scaled residuals).
The settled endpoint of each segment is added with weight 5 on the residual, so 25 in `E`; the motion-onset summary is skipped under declared observation noise (see Rerun protocol).
The loss is not a frame-by-frame likelihood on purpose: mid-flight contact motion is chaotic, and the endpoint carries what replays reproduce.
With a Gaussian frame-error loss at `lambda = 1` the target is the ordinary posterior.
There is no added model-error allowance and no between-segment term; both were margins rather than parts of the target.

1. Temperature: evaluate the residual vector at the MAP; `lambda = max(1, SSE_min / (N * sigma_n^2))`, where `N` counts residual terms and `sigma_n` is the fit's Gaussian width on scaled residuals.
   It is the maximum-likelihood variance of the standardized errors if they were independent Gaussian; it does not correct for errors that persist through a segment, and the fit report should show the lag-one autocorrelation of the MAP residuals so that gap can be measured.
2. Lines: for each continuous parameter, evaluate the negative log target at that temperature along its coordinate through the MAP in fit space, expanding outward until it rises by the cutoff or reaches the declared bound, then bisecting the interval holding the most mass.
   For a discrete parameter, the line is its set of values: evaluate the target at each value with the other parameters at the MAP, and normalize.
3. Line posterior: interpolate the negative log density linearly between probes (a piecewise-exponential density), store it on a finer grid, and normalize; a flat line leaves the prior.
4. Draws: an independent sample from each line (inverse CDF), discrete parameters included.
   `K = belief_joint_draws` draws are made once per fit with a seed derived from `CFG.seed` and the fit identity, so a cached fit reuses them.
5. Before any fit of the current program (and in the no-fitting ablation), `prior_belief` draws from the same Gaussian prior, restricted to the bounds.

The fit's consistency loop drops explainable segments that disagree; under the posterior view they are pooled (`code_sim_learning_rollout_consistency_factor = 0`), and their disagreement raises the estimated noise level.
A segment is unexplainable when its own best replay over the parameter grid leaves a root-mean-square standardized error above `code_sim_learning_rollout_trim_rms_factor` (2); it stays excluded and is reported as evidence of a missing mechanism.
The belief rides on `SysIdOutcome.belief`, and the MAP becomes the applied point.

## State factor (`observation_belief.py`)

For each object, run lengths `r = 1..W` (W = `continual_belief_window`) get posterior weights from the Gaussian predictive of each new frame under a still model, with a constant per-step motion hazard `continual_belief_motion_hazard`.
Each feature's prior is uniform over its recorded range, which makes the predictive of the first frame after a motion proper (density `1 / range`); an improper flat prior would leave the change-point weight depending on an arbitrary constant.
Given `r`, each noisy feature is Gaussian with the mean of the last `r` frames and standard deviation `sigma_f / sqrt(r)`, truncated to the range (the posterior under the uniform prior; angles are wrapped instead); the belief is the mixture over `r`, and draws pick `r` first.
The run-length weights use the untruncated predictive, which differs only when a run mean sits within a few sigmas of a range end.
Exact features are read from the latest frame.

Draws perturb each object's features independently, so a draw can put touching objects inside one another, leave a resting object hovering or tilted on its support, or stretch an attachment, and the first engine step resolves that with an impulse.
The check on 2026-09-25 (the EMPIRIC arm's flags, the first training and test task of each benchmark setting, 16 draws, the robot holding still for 20 steps) found it in every domain: from draws of an object at rest over eight frames, the engine moved Balloons' box and balloons by up to 31 mm, Boil's jugs by up to 15 mm, Domino's dominoes by up to 12 mm, Fan's ball by up to 10 mm and Bridge's spans and legs by up to 6 mm, where the exact state moves nothing; from one frame after a motion, 9 to 13 of 16 draws toppled a domino and a jug tipped by 0.27 rad.
So a rollout starts from the draw after a quasi-static settle in its own engine (`PyBulletEnv.settle_state`, `pybullet_helpers/settle.py`): 240 engine substeps with every velocity zeroed after each, the robot held and no domain step, so no modeled mechanism acts.
Only bodies the static world holds up settle, those resting on a fixed or settled body through a contact within 2 cm whose normal points up, or tied to one by an engine constraint; a body a mechanism holds up (a box lifted by balloons) would sag under gravity alone and keeps its drawn pose.
After the settle the engine leaves every rest draw where it is, apart from a tilted domino rocking back upright (at most 4.3 mm and 0.06 rad).
From one frame after a motion a body can be drawn more than 2 cm above its support, so it keeps that pose and drops when the rollout starts (Balloons' box by up to 43 mm), and 1 to 2 of 16 draws still topple a domino; the posterior after one frame cannot tell a resting body from one a mechanism holds a few centimetres up.
Settling ignores the mechanisms on purpose: equilibrium under them depends on the parameter draw, and the state factor is conditioned on observations only (the cut in B.3).
Monitoring reads atoms on the unsettled draws; the settle is how a rollout starts from one.
Scripts and logs: `~/claude_sbatch/principled_belief_20260925/interpenetration_check.py`.

## Consumers

- Approach: store the published belief with the fit, save it in checkpoints, derive the rule-parameter ensemble from the draws, and stop building physics sigma points and Laplace ensembles when the belief is on.
- Session (`run/continual.py`): one `LatentTracker` per draw, rebuilt when the model revision or belief changes; `joint_draws(K, rng)` pairs draw `i`'s parameters with a state draw from `q(x_t | H_t)` carrying latent `z_i`.
- Monitor: atom fractions over the `K` joint draws (each draw's parameters and latent, state from the updated state factor); unmet below one half, as now (legacy: 16 state-only draws sharing one latent).
- Execution has no automatic gate, as in the evaluated continual arm: its tool surface is `run_python` plus the env and skill tools, and it never calls `submit_plan` (continual_tools.py:46-54, agent_continual_approach.py:148-149).
  The agent decides when to execute from the rehearsal estimates; `submit_plan` changes are out of scope for the continual arm.
- `sim.run(plan)`: rehearses the plan on the `K` joint draws (fresh env at the draw's parameters, applied through the path fresh envs read, the draw's state and latent, planner seed `i`), fixed per decision point for common random numbers.
  The draws run in parallel processes, so wall-clock time stays near one rollout.
  The report gives P-hat, its standard error, per-draw outcomes, and the parameter ranges on which draws fail, plus a step-by-step rollout from the state-belief mean at the parameter estimate with contact diagnostics.
  It replaces `belief_draws` (state only), `trials` and `solved`; `physics_sweep` stays as a labelled stress test.
- Task evaluator on a prefix: P-hat applies the task evaluator to the recorded history `H_t` (observations and actions as recorded) followed by each draw's simulated trajectory, so whole-episode constraints such as Domino's trigger rule are checked.
  Today `solved=True` requires the task's unmodified initial state (`_require_solved_evaluator` in `belief_probe.py`) and cannot be combined with belief draws (`belief_probe.py:1570`), so P-hat cannot be computed after an episode's first action until this is built.
- `sim.refine`: the backtracking search from the belief mean proposes settings whose annotated predicates hold, as now; up to `belief_refine_candidates` proposals are scored on the common draws, and the one with the highest P-hat is returned together with its P-hat on `K` fresh draws, which the selection does not bias.
- `sim.suggest_probes`: feasible candidates are scored by the noise-aware predicate information over the same `K` joint draws (each draw's state and latent), with `belief_info_noise_samples` sensor-noise samples of the final observation.
  Predicates are evaluated at fixed parameter values (the MAP) and read only the observation, not the latent, so each is a fixed function of the observation and the data-processing bound of the paper's Eq. 4 holds; the score adds information about the current state, which is small at rest.
  The legacy score re-reads atoms on one point-model rollout, and its ensemble members swap `_fitted_params` while fresh envs read `_agent_param_values`, so dynamics parameters could not change the score; per-draw rollouts fix both.
- Reweighting the draws between fits by the new data's loss would be sequential importance sampling over `theta` (Chopin, 2002); it is out of scope.

## Settings

- `belief_joint_draws` (16, matching `continual_belief_draws`; 0 keeps the legacy path)
- `belief_line_cutoff` (12.5, a rise of 5 sigma in negative log posterior)
- `belief_line_max_evals` (24 per parameter)
- `belief_refine_candidates` (8 proposals scored by P-hat per refine call)
- `belief_info_noise_samples` (64); `belief_info_param_draws` is removed, since experiments use the joint draws
- `continual_belief_motion_hazard` (constant per-step motion probability; set from the smoke runs and reported in the paper)
- `belief_prefix_frames` (`observed`, the paper's H_t: the evaluator reads the recorded prefix as observed; `smoothed` reads each prefix frame's state-belief mean instead)
- `code_sim_learning_carry_posterior` set false and `code_sim_learning_rollout_consistency_factor` set 0 in the continual configs (the prior stays at its declared centre; explainable segments are pooled).

## Validation

- Unit tests on synthetic objectives: Gaussian line recovers `1/Lambda_jj`, flat line reverts to the prior, the noise level scales the width by `sqrt(lambda)`, draws stay inside the bounds and are deterministic per seed, and discrete parameters are sampled in proportion to the target.
- Unit tests for the state factor: run-length weights do not depend on the feature's units, and a jump after a rest moves the mass to `r = 1`.
- Integration tests with a stub env for the monitor, tracker and refine-selection plumbing, and an evaluator test that scores a Domino prefix followed by a simulated suffix.
- The interpenetration check above, and unit tests of the settle on a table scene (sunk, stacked, hovering, hanging and unsupported cubes).
- A short compute-node smoke run per domain before the matched comparison.

## Rerun protocol

- The skill preflight was on in 7 of the 25 reported EMPIRIC runs and off in every ablation run; the rerun uses one setting (off) for all arms.
- The motion-onset residual (`_onset_residuals`, tolerance `settle_tol` = 1e-3 raw units) is broken under declared observation noise: in all five reported Domino fits and Bridge seed 0 the observed onset is frame 1 for every object (114/114 and 13/13 object-segments), and the simulated onset is also frame 1 because the noisy start pose makes every object settle on the first step, so the term carried no timing information.
  It also charges about 4.7 per object-segment to any model that holds a still object still, which rewards spurious motion.
  Skip the term when observation noise is declared; this leaves the reported fits unchanged.
  Scripts: the session scratchpad `onset_check/`.
- Programs with `MODEL_STATE_INIT` are fit on whole episodes (no rest-point segmentation), so their memory starts at the episode's first frame (`agent_sim_learning_approach.py` `has_model_state` branch); the paper describes this.

## Implementation status (2026-09-25)

Built on the `principled-belief` branch, behind `belief_joint_draws` (settings default 0; `continual_common.yaml` sets 16, and the No uncertainty arm keeps 0).

- Parameter factor: `ParameterBelief` with line posteriors, discrete parameters as categorical lines (`DiscretePosterior`), fresh draws (`ParameterBelief.sample`), and `prior_parameter_belief` before a fit; the fit tool reports the belief in place of the identifiability section.
- State factor: `change_point_belief` and `feature_ranges` in `observation_belief.py`; `ContinualRun.belief()` uses it when the joint belief is on, for every arm with a state estimate.
- Session: `joint_draws`, `joint_atom_fractions` and `episode_prefix` on `ContinualRun` and `ProtocolSession`, with one tracker per parameter draw keyed by level, episode, model revision and belief digest.
- Consumers: monitoring and the `[atoms under the belief]` line read the joint fractions; `sim.run(plan)` rehearses on the joint draws and scores each with the task evaluator on the recorded prefix plus the draw's rollout (`ProbeJointResult`, one line per call in `<log_dir>/rehearsals.jsonl` for calibration); `sim.refine` scores up to `belief_refine_candidates` proposals on the common draws and re-estimates the winner on fresh draws; `sim.suggest_probes` scores candidates with `noisy_read_information` over the joint draws.
- Prompts: the rehearsal, refine and robustness text switches to the joint wording when the joint belief is on (`play_system.md` sections `*_joint`); the goldens pin the experiment configuration (`continual_system_mb_joint` and the ablation arms at 16 draws).
- Tests: `tests/agent_sdk/test_belief_probe_joint.py`, `tests/run/test_continual_joint_belief.py`, the change-point and truncation tests in `tests/test_observation_belief.py`, and the discrete-draw test in `tests/code_sim_learning/test_parameter_belief.py`.
  The evaluator-on-prefix test uses a stub whole-trajectory rule; there is no Domino-specific one yet.

Open before the matched comparison: one smoke run per domain.
The observed prefix leaves Domino's solved verdict nearly intact: its staging rule allows one domino width (7 cm) of drift against 1 cm of position noise, and topples need three frames past the fallen angle.
Its lean rule (5 degrees at the frame before the push, against 0.04 rad of orientation noise) can fail on a noisy frame, which happens only when the push is already in the recorded prefix; the 2 cm slide rule counts consumed blues for the reward and does not decide the verdict.
