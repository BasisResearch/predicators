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
