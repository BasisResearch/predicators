# Marginalized output discrepancy

September 12, 2026.
This evaluates an explicit statistical model extension under the [simplification proposal](simplification-proposal.md).
It does not replace the production fitter, certify a physical-state posterior, or complete the five-domain prediction gate.

## Model and implementation

For a declared real-valued output of a fixed candidate simulator, write its prediction as `h_t(theta, s0)`.
The discrepancy process is:

$$
b_0 \sim \mathcal{N}(0,\sigma_0^2),\qquad
b_t=\rho b_{t-1}+\eta_t,\quad
\eta_t\sim\mathcal{N}(0,\sigma_b^2),
$$

$$
o_t=h_t(\theta,s_0)+b_t+\epsilon_t,\qquad
\epsilon_t\sim\mathcal{N}(0,\sigma_{\mathrm{sensor}}^2).
$$

Sensor variance remains its declared value.
The separate discrepancy scale and persistence describe correlated prediction error; they have explicit values or an identified original prior.
They are not recalculated from each residual to make a candidate acceptable.
This is an output-discrepancy model rather than a force correction: the physical simulator state and its subsequent native dynamics remain unchanged.

Conditional on parameters and an initial simulator state, the entire Gaussian discrepancy history can be integrated analytically.
The implementation in `inference_output_error.py` uses the product of causal predictive densities to evaluate that marginal likelihood.
This avoids sampling a new error coordinate for every primitive step while retaining its temporal dependence.
The batch target can therefore infer simulator parameters, uncertain initial states and discrepancy hyperparameters together using the existing fixed-prior sampler.
It is the marginal of the declared augmented model, not the deterministic sensor-only target with troublesome observations removed.

Exactly observed continuous outputs condition the corresponding latent discrepancy and retain its Gaussian density.
Their filtered discrepancy variance becomes zero at the observation boundary; later innovations can make it positive again.
When both discrepancy and sensor variance are zero, an unequal prediction remains an exact contradiction for that supplied candidate history.
That does not prove that all parameters or initial states are inconsistent.
Missing readings add no evidence but still advance the process by each intervening primitive step.
Future forecasts propagate the error distribution without consuming future observations.
Repeated full-data fits start from the same original error law, rather than using the previous filtered error as a new prior.

The returned error moments are causal filtering marginals.
They are not independent joint-history samples, smoothed states, or corrected physical scene states.
Events, bounded outputs, angle branches and coupled kinematic constraints require separately justified observation models.
This scalar Gaussian construction is not automatically appropriate for those quantities.

## Independent numerical references

Functional tests compare the likelihood and each causal conditional against dense multivariate Gaussian integration, including missing readings, negative persistence, random walks and exact observations.
They also verify future-prefix separation, repeated-fit identity, deterministic contradictions and explicit numerical failure.

Compute reference `22633541` fits constant-velocity dynamics with an uncertain initial position, a log-uniform innovation scale, and an independent uninformed parameter.
The original box prior remains fixed.
Discrepancy persistence is 0.85, initial discrepancy is zero, and sensor standard deviation is 0.03.
Eight generated observations enter the fit; four additional observations are used only for predictive scoring.
The target is checked against independently constructed dense Gaussian grid integrals at 81 and 161 cells per informed coordinate.
The finer grid has 4,173,281 cells across velocity, initial position and log innovation scale.
Grid changes pass the limits declared before the sampler trials.

| Particle budget | Independent seeds | Reference checks passed |
| --- | --- | --- |
| 512 | 100, 101, 102, 103 | 3/4 |
| 2,048 | 100, 101, 102, 103 | 4/4 |

The failed 512-particle trial has an uninformed-parameter mean of -0.186 instead of zero and median of -0.251 instead of zero.
Its final weight ESS is approximately 292, demonstrating why ESS alone cannot establish approximation reliability.
The failure remains part of the result and does not justify increasing the claimed accuracy of that budget.
These are numerical sampler runs on one generated dataset, not agent seeds or repeated-dataset coverage evidence.
Artifacts and the predeclared limits are in `logs/uncertainty_output_error_reference_20260912`.

## Five-domain recorded-data diagnostic

The first real-data comparison uses the same frozen first training levels as the earlier state inventory.
It fixes each selected simulator program and its declared nominal parameters, then compares independent versus persistent output error.
This isolates the discrepancy model; it is not a comparison against the full legacy parameter fitter.
Bridge and Boil retain their frozen no-op programs as incomplete controls.

The initializer uses model-owned handles and the first noisy public frame in an explicit fixed-state rest-start ablation.
It is not an uncertain-initial-state posterior or a proof of feasible geometry.
The first reading is consumed by initialization and omitted from discrepancy fitting.
All later states come from uninterrupted native simulation, preserving its cached-link observation timing.
Fresh repeated trajectories are compared exactly.

Each selected channel is an object or robot Cartesian coordinate, or a current joint position.
The diagnostic infers an innovation scale separately for each channel under a fixed log-uniform prior: 0.0001 through 0.05 for Cartesian coordinates and 0.00001 through 0.02 for joint-coordinate units.
The independent arm fixes persistence to zero.
The persistent arm averages over persistence values 0.5, 0.9 and 0.99 with equal prior mass.
Initial discrepancy is zero in both arms.
The original sensor variance stays unchanged.

Thirty-two actions enter the discrepancy fit, followed by a 32-action suffix held out from that fit.
Both arms marginalize their hyperparameters using the same log-scale quadrature grid.
The suffix joint log score integrates future discrepancy states; its forecast means and probability-integral-transform values use only the fitted prefix.
Moving objects, robot outputs and static fixtures are reported separately.
Central 90% predictive coverage is evaluated from the mixture CDF, rather than a Gaussian interval fitted to mixture moments.
These are descriptive coverage counts on correlated observations, not calibration estimates from independent datasets.

The selected program versions are frozen historical learned artifacts; the suffix is held out from discrepancy-hyperparameter fitting, not established as unseen during those programs' synthesis.
This diagnostic therefore does not establish generalization of program learning or satisfy the final held-out evaluation requirement.
Unmodeled exact outputs remain explicitly listed, including events and Balloons speed.
Their failures prevent treating the selected-channel analysis as a posterior over the full recording.

The native trajectories and first grid report are in `logs/uncertainty_output_error_domains_20260912`.
A doubled-grid analysis reuses the exact saved prediction and observation artifacts, without rerunning physics, in `logs/uncertainty_output_error_domains_grid_20260912`.

Native replay job `22633841` completed all five domains, and every fresh repeat matched exactly over the 64-action history.
Grid-sensitivity job `22633902` doubled the innovation-scale grid from 41 to 81 cells.
The table below reports moving-object Cartesian channels only.
A positive score difference favors persistent discrepancy over independent discrepancy; it is expressed in nats per scalar future observation to make the differing channel counts visible.

| Domain | Future scalar observations | Log-score gain per scalar | Independent 90% coverage | Persistent 90% coverage |
| --- | ---: | ---: | ---: | ---: |
| Bridge | 576 | +0.0244 | 523/576 | 540/576 |
| Fan | 96 | +0.1200 | 88/96 | 91/96 |
| Domino | 576 | +0.0858 | 522/576 | 536/576 |
| Boil | 96 | +0.0423 | 85/96 | 85/96 |
| Balloons | 384 | +0.2267 | 332/384 | 354/384 |

The moving-object score gains change by less than 0.001 nats in total per domain when the grid doubles.
Robot channels are less favorable and more sensitive to quadrature: Balloons' persistent model loses approximately 37.23 nats in total, and Bridge's coverage falls from 299/384 to 271/384 despite a better joint score.
Individual robot-channel scores change by as much as 1.42 nats when the grid doubles, so their absolute numerical values are not certified by this sensitivity check.
These mixed results argue for retaining channel-specific diagnostics rather than adopting a persistence rule solely from a pooled gain.
All five cases still have unmodeled exact-output disagreements, preventing a full-recording posterior from this selected-channel model.
The [recorded assessment](../../logs/uncertainty_output_error_domains_grid_20260912/assessment.json) retains every group and the grid changes.

The final component checks, job `22633903`, passed fifteen functional tests, two-file mypy and lint, and pinned formatting.
An earlier check caught a missing local type annotation; the final result also names a supplied-history failure `exact_contradiction` to avoid implying a proof against every parameter or initial state.
The numerical and recorded diagnostics exercise successful likelihood paths, whose arithmetic is unchanged by that annotation and failure-label refinement.

## Remaining acceptance work

The scalar likelihood and small numerical posterior can be checked independently of the real-domain initial-state and support questions.
For the full agent replacement, those questions remain required.
This extension also needs a justified complete output model, stable real-data numerical inference, and predictions on interactions held out from all relevant fitting and model selection.
Improvements on selected Cartesian channels cannot establish that exact event predictions, contact behavior, parameter coverage or agent performance are adequate.
Only subsequent comparisons under the proposal's gates can select a replacement for production.
