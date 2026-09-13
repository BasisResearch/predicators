# Stochastic future density integration

This is an offline component of Stage B of the [simplification proposal](simplification-proposal.md).
It does not change the acting agent, and no physical posterior has passed the required numerical and predictive gates through this component.

## Generation and scoring

A stochastic forecast generator draws future joint and velocity transitions without consulting future observations.
A density evaluator instead integrates over latent transitions conditional on the observations being scored, retaining the associated joint and speed densities.
Using exact equality against a finite set of unconditional draws would assign zero likelihood to almost every continuous reading even when the model assigns a positive density.
These two operations therefore need separate implementations.

The new `inference_path_integral` component averages complete conditional-path densities using a stable log scale.
The caller must include every target/proposal correction and use independent paths from the same normalized sampling law.
It preserves zero contributions and propagates numerical or replay errors instead of deleting failed paths.
Its output includes the empirical relative standard error and the effective number of contributing paths.
Those diagnostics concern Monte Carlo integration, not posterior particle weights or physical model adequacy.
No sampled support yields an undefined error estimate, not a claim that the model is inconsistent.

For the current Balloons model, each scored future path conditions the joint transitions on the evaluated joint readings and the velocity law on the evaluated speed.
The joint Gaussian density and radial speed density or rest mass are retained exactly once.
Conditional directions are sampled and propagated through the native dynamics, including their later contact and event consequences.
The remaining output-observation likelihood conditions its temporal error process on the supported prefix.
The prefix transition factors are not counted again in the future score.
This construction is a conditional likelihood evaluation; its observation-guided paths must never be reused as unconditional generated forecasts or acting states.

## Checks and native diagnostic

Compute job `22653160` passed 40 functional tests, focused mypy and pylint, and pinned formatter checks.
The new tests compare a conditional speed and downstream observation integral with independent one-dimensional quadrature, including both rest and positive-speed cases.
They also check large log scales, zero contributions, an unseen rare event, deterministic random seeds, and error propagation.
Checked source hashes and outputs are in `logs/uncertainty_path_integral_checks_20260912`.

Native job `22653178` completed in 8 minutes 53 seconds on `node1381`, using the frozen historical recording runtime and the offline modules.
It reproduced the full 235-action reference factor exactly as `-33314.92152710342` before evaluating any futures.
It then evaluated a 32-action future following a fixed 64-action prefix, for both an archived generated observation history and the recorded history.
The two independent integration seeds each use 64 complete paths per history, with the 8-path estimate obtained from the same sequence's first eight paths.
The first path of every case repeats exactly, and every reconstructed prefix equals the original reference prefix.
The diagnostic performed 25,195 native environment steps in total.

| Evaluated future | Paths per integration seed | Relative standard error, seed 811 | Relative standard error, seed 812 | Between-seed log-density difference | Declared diagnostic |
| --- | ---: | ---: | ---: | ---: | --- |
| Generated history 701 | 8 | 1.0000 | 0.9989 | 10.3770 | Fail |
| Generated history 701 | 64 | 1.0000 | 0.9967 | 0.9607 | Fail |
| Recorded history | 8 | 0.9914 | 1.0000 | 13.2596 | Fail |
| Recorded history | 64 | 1.0000 | 1.0000 | 5.4757 | Fail |

The predeclared limits were relative standard error at most 0.2 and between-seed log-density difference at most 0.2.
All four comparisons fail.
At 64 paths the effective number of contributions ranges from approximately 1.0000 to 1.0065, despite retaining all paths.
These density estimates are not numerically adequate for comparing inference methods.
Increasing from eight to 64 paths did not resolve the concentration.

In generated-history seed 811, the transition log factors have standard deviation 4.99 while the remaining output log factors have standard deviation 81.25.
This motivates attributing the concentration to individual output factors before choosing an integration strategy.
It does not justify dropping those observations, inflating their declared noise, or accepting the largest sampled path as the density.

The fixed physical witness and its parameters were selected using the complete training trajectory.
This diagnostic therefore establishes neither a causal prefix-fitted forecast comparison nor an agent result.
Its purpose is to test density accounting and expose numerical problems before posterior integration.
The complete contributions, input hashes, native paths, runtime controls and failed diagnostics remain in `logs/uncertainty_balloons_future_density_20260912`.

## Factor attribution

Jobs `22653470` and `22653543` reconstructed the generated-history paths at output-score ranks 0, 16, 32, 48 and 63 for each integration seed.
Every recomputed transition and output score matched its archived value exactly, and the separately scored factors summed to the complete output score within `1e-8`.
Each audit used 1,195 native actions, including the 235-action reference check.
The first audit grouped factors; the second split the original sensor by feature while retaining the checked finger source/readout pair as one factor.
The original attempt `22653400` failed because an audit variable overwrote the native object-key set; its corrected successors retain that failure record and make no core simulator changes.

Across these ten selected paths, only the Cartesian readings of the box and balloon 0 vary in output log likelihood.
Balloon 0 is attached at the prefix boundary and through the evaluated suffix.
Robot Cartesian and Euler factors, the other objects' readings, and all exact event/readout factors remain identical.
The largest feature ranges are:

| Feature | Log-factor range, seed 811 | Log-factor range, seed 812 |
| --- | ---: | ---: |
| Box y | 180.34 | 182.76 |
| Attached balloon y | 165.83 | 171.62 |
| Attached balloon x | 33.57 | 34.58 |
| Box x | 25.09 | 26.73 |

These ranges describe the declared rank-selected paths, not all 64 draws or a population variance decomposition.
The result identifies observation mismatch along the moving attached assembly as the driver in this audit; the tightly modeled robot output channels do not explain this concentration.
It supports investigating position-guided proposals or sequential reweighting of conditional velocity directions while retaining all observations and their density factors.
It does not establish that either method will meet the numerical or latency gate.
The split audit archives the ten complete native paths for further analysis in `logs/uncertainty_balloons_future_attribution_v3_20260912`.

## Remaining work

Evaluate a position-guided or sequential integration method with explicit density corrections and independent numerical references.
Any proposed method must preserve the joint future law, exact observations, native replay contract and numerical failure reporting.
A reliable conditional-path integral still needs integration over an assessed joint parameter and prefix-state posterior.
Matched legacy predictions, initial-state and prior ablations, saved planning decisions and the live-agent validation gates remain required before replacing the incumbent.

## Sequential history integration

The offline `inference_sequential` component extends complete histories through fixed observation blocks and performs multinomial resampling between blocks.
Each extension samples a normalized conditional proposal and returns the incremental target/proposal factor for that block, including its exact-observation density factors.
The product of block-average weights estimates the complete future density.
This retains temporal dependence; it is not a product of separately fitted marginal forecasts.
No resampling is performed after the last block, so the terminal histories retain their normalized weights.

The implementation records every block normalizer, effective contribution count and surviving original ancestry count.
Resampled terminal histories are correlated, so their spread cannot be used as an iid error estimate of the normalizer.
A high terminal effective count does not undo earlier ancestry loss.
Independent complete integrations and budget comparisons remain necessary for numerical assessment.
Zero-support extensions retain zero weight, and complete sampled-support loss returns an explicit unavailable result.
Replay exceptions and nonfinite factors abort the computation instead of deleting or retrying individual extensions.

Callbacks receive private copies of their parent histories, and returned histories are copied before retention to avoid mutation across siblings.
History payloads describe reconstructible paths; they must not contain live simulator handles.
The Balloons adapter reconstructs a fresh native world from the same validated prefix for every extension, preserving hidden dynamics and the cached-link observation phase.
This has a substantial cost which must be measured rather than hidden by counting only newly extended actions.

The component reference enumerates all histories of a two-state Markov model to check both the joint observation density and terminal posterior probability across independent integration runs.
Additional tests check an observation-guided proposal with its retained density correction, exact-event zero weights, mutable-parent isolation, ancestry collapse, numerical overflow and interrupted replay.

Compute job `22654085` passed all 44 functional tests, focused mypy and pylint, and pinned formatter checks.
The initial check found missing type annotations and a reused variable name; the next found three test-only lint issues.
All attempts and final source hashes are retained in `logs/uncertainty_sequential_checks_v3_20260912` and its preceding check bundles.

Native array `22654086` evaluates both the generated701 and recorded 32-action futures with independent seeds 911 and 912, using 32 histories and eight four-action blocks.
The plan, worker and overlays are frozen in `logs/uncertainty_balloons_sequential_future_20260912`.
Each run first reproduces the full 235-action reference factor, and every path extension must reproduce the original 64-action prefix exactly.
The terminal audit repeats one retained full history and checks that its sum of block factors equals its direct whole-future factor.
Each completed run requires 21,419 native actions, including reconstruction and terminal checks, rather than just its 1,024 newly sampled future actions.
These pilots remain numerical diagnostics of a fixed full-training-selected witness, not prefix-only inference, new agent results or a comparison of model quality.

All four tasks completed and passed the full-prefix, terminal replay, block-factor decomposition and artifact checks.
The numerical consistency result is negative:

| Evaluated future | Seed 911 log density | Seed 912 log density | Between-seed difference | Final original ancestors, seeds 911 / 912 |
| --- | ---: | ---: | ---: | --- |
| Generated701 | 4950.40744 | 4960.59650 | 10.18906 | 1 / 2 |
| Recorded | -2573.78777 | -2548.09901 | 25.68876 | 1 / 1 |

Both comparisons fail the predeclared 0.2 log-density-difference diagnostic.
Individual block effective counts sometimes improve, but that does not establish an adequate complete-history integral.
The retained ancestry and independent-run disagreement show why terminal particle counts alone are insufficient.
The four jobs took approximately 6.5, 10.2, 8.6 and 6.6 allocated minutes and performed 85,676 native actions in total.
This pilot uses more replay work per integration than the earlier whole-path diagnostic; it does not demonstrate either accuracy or cost superiority.
Verified outputs and a repeatable artifact checker are in the native bundle's `verification.json` and `verify_report.py`.
Position-guided proposals or another demonstrated variance reduction remain necessary before claiming a reliable stochastic forecast score.

## Defensive position guidance

The offline `inference_guidance` component proposes a velocity direction conditional on the same exact speed using a mixture of the original direction law and a guided law.
Both components are normalized conditional Gaussian direction distributions on the speed sphere.
The guide changes the proposal mean, not the original velocity model or its discrepancy scale.

For original radial density `p(s)`, original conditional direction density `p(d | s)`, guided density `g(d | s)` and guide probability `a`, the retained factor is:

```text
p(s) * p(d | s) / ((1 - a) * p(d | s) + a * g(d | s))
```

The denominator is the complete mixture density, including both components regardless of which generated the draw.
The original component has positive probability, so the directional importance ratio cannot exceed `1 / (1 - a)`.
This bounds a single direction correction; it does not bound the variance of a complete history or establish adequate sampling.
Zero observed speed retains the original rest atom and does not introduce direction coordinates.
Disabled guidance and identical proposal means preserve the original conditional draw and radial factor exactly.

The native proposal uses the next noisy Cartesian readings of the box and currently attached, unpopped balloons.
For each axis it adds `velocity_sigma² * action_dt * sum((next_reading - current_prediction) / sensor_variance)` to the predicted box velocity.
This is the direction proposal obtained from an approximate rigid translation over one action with isotropic position sensors.
Contacts, rotation and forces can invalidate that approximation, so the native transition and complete observation model still determine the corrected target weight.
The declared action duration comes from the simulator's actual fixed time step and configured substeps.
The pilot uses guide probability 0.8 and disables guidance at the last evaluated step, which has no subsequent position reading.

Guidance is only used while evaluating the density of an observed future.
Its observation-guided paths must never enter unconditional forecast generation, prefix parameter fitting or the acting agent's state.
The scored future is already fixed when the proposal is constructed, and every proposal factor is retained exactly once.
Three independent uniform coordinates per moving extension preserve both the component selection and direction for deterministic history reconstruction.

The density calculation uses a centered Gaussian log ratio and a compensated sum of radial terms.
A direct API reproduction exposed cancellation in the first draft: with equal-strength opposite means, large common radial terms erased the direction correction.
The corrected implementation matches the independently derived direction ratio for both selected components in those cases.
The old behavior, corrected values and reference values are retained in `logs/uncertainty_guidance_checks_v2_20260913/cancellation-reproduction.json`.
Component tests also compare mixture corrections with independent Gaussian and noncentral-chi densities, and compare guided downstream observation integrals with one-dimensional integration under the original sphere law.

Compute job `22671599` passed 51 functional tests, focused type and lint checks, and pinned formatting checks.
Source hashes and verification artifacts are retained in `logs/uncertainty_guidance_checks_v2_20260913`.
Native array `22671657` uses the same two evaluated futures, independent seeds 911 and 912, 32 histories and eight four-action blocks as the preceding sequential pilot.
Its frozen plan and worker are in `logs/uncertainty_balloons_guided_future_20260913`.
Each moving extension retains its component selector, direction coordinates, proposal mean, correction and separately recomputed original speed factor.
The worker verifies their density decomposition and preserves the full-reference, prefix and terminal replay checks.
These checks establish accounting and replay consistency; numerical stability of the complete density estimate remains to be evaluated from the pilot results.

All four guided tasks completed with verified density decomposition, exact prefix/terminal replay and valid artifact hashes.
The numerical consistency result remains negative:

| Evaluated future | Seed 911 log density | Seed 912 log density | Between-seed difference | Final original ancestors, seeds 911 / 912 |
| --- | ---: | ---: | ---: | --- |
| Generated701 | 4963.60727 | 4959.95918 | 3.64809 | 1 / 1 |
| Recorded | -2548.06723 | -2527.55554 | 20.51169 | 2 / 1 |

Both differences exceed the predeclared 0.2 diagnostic limit.
Some individual block effective counts increased, but that does not establish accurate complete-history integration.
With only two replicas, smaller gaps than the preceding pilot are not evidence of a reliable variance reduction or calibrated prediction.
The jobs took approximately 10 minutes each and again performed 85,676 native actions in total.
The full results and artifact checker are retained in the guided bundle's `verification.json` and `verify_report.py`.
Guidance preserves the intended probability model, but this budget and proposal do not pass the numerical gate.
