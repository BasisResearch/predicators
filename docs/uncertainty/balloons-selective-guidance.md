# Balloons: keeping unobserved parameters broad during inference

September 14, 2026.
This is an offline numerical investigation for Stages A/B of the [simplification proposal](simplification-proposal.md).
The production estimator is unchanged.

## Reproduced dependence and concentration

The fixed Balloons program reads a balloon's color-specific lift coefficient only when it is tied and live, and selects box mass by its material.
This suggests that a recording which activates only one color on one material cannot constrain every parameter in the program.
The native audit tests that suggestion on the actual 64-action fitting prefix at two supported retained states from the original numerical fits.

At each state, it evaluates the baseline, each of ten parameters at prior-unit coordinates 0.05 and 0.95, and two simultaneous interventions on the five suspected inactive parameters.
All 196 non-parameter coordinates remain fixed.
Every candidate repeats in a fresh native trajectory; complete observations, transition corrections, support outcomes and likelihood components are retained for comparison.

Native job `22715205` completed all 46 cases in 2:28, using 5,888 actions.
Independent reader `22715206` completed in 59 seconds with 1,792 additional native actions, checking all recorded interventions, all twenty parameter summaries, and six deliberate corruptions.
Its source checksum matches the native report.
Artifacts are in `logs/uncertainty_balloons_parameter_dependence_20260914`.

At both states, changing `lift_c0`, `lift_c1`, `lift_c2`, `mass_pine`, or `mass_teak`, separately or together, preserves the complete evaluation exactly.
The other five parameters change the evaluation, so the audit also has positive sensitivity controls.
These are finite local checks; they are not a proof of global conditional independence over every feasible latent state.

The original completed fits nevertheless retain extremely narrow distributions for these locally inactive parameters.
The following discrepancies compare the weighted empirical distribution with the declared prior after transforming that prior to a uniform unit coordinate.
A uniform coordinate has standard deviation approximately 0.289.
The CDF gap is descriptive; it is not a p-value or a complete convergence test.

| Parameter | Seed 620 unit SD | Seed 621 unit SD | Seed 620 maximum CDF gap | Seed 621 maximum CDF gap |
|---|---:|---:|---:|---:|
| Red lift | 0.0091 | 0.0102 | 0.5283 | 0.5367 |
| Blue lift | 0.0105 | 0.0165 | 0.5403 | 0.4902 |
| Green lift | 0.0066 | 0.0037 | 0.5121 | 0.6722 |
| Pine mass | 0.0105 | 0.0144 | 0.5118 | 0.5522 |
| Teak mass | 0.0067 | 0.0104 | 0.5257 | 0.4950 |

This supports a concrete numerical concern: concentration inherited from the initial proposal and resampling can persist even along directions where the checked likelihood does not change.
A higher likelihood at a retained point cannot establish that its surrounding parameter uncertainty is correct.
The default-centered fits remain a separate comparison and are not replaced by this diagnostic.

## Target-preserving numerical change

Bundle `logs/uncertainty_balloons_selective_guidance_20260914` adds a selective version of the existing normalized joint proposal.
The five diagnosed coordinates are sampled uniformly in every proposal component, while the existing local guidance remains on the other coordinates.
The broad component and full mixture density correction are preserved.
An empty selection reproduces the original proposal.

The fitting driver also mixes a 20% independent uniform proposal into each existing coordinate block's random-walk kernel.
Both moves are symmetric in the proposal chart and receive the existing Metropolis acceptance check with the complete target density.
Every evaluated candidate continues to incur its full native fitting cost.
The prior, physical simulator program, observation and dynamics-discrepancy laws, fitting prefix, and future-data boundary remain unchanged.
The paired numerical seeds, particle count, temperature schedule, number of moves and evaluation budget remain unchanged.

Neither change requires treating the selected parameters as globally independent of the data.
If a selected parameter does affect an untested history, the native likelihood and density-corrected acceptance decision still account for that effect.
No fitted marginal is manually replaced by the prior, and no state or parameter coordinate is removed from the inference problem.

Native validation `22715307` completed in 1:48 on the matching Intel compute node.
It passes 32 exact default-parity cases, 32 independent full-chart mixture-density cases, uniform-coordinate mappings, normalized importance integration and four malformed-mask rejections.
The corrected integral 0.27571630757858434 agrees with independent quadrature 0.2757163075785848.
An additional likelihood that depends on a selected uniform coordinate preserves evidence 2 and conditional mean 7/12, checking that selecting a coordinate does not remove its likelihood.
The existing exact prefix replays pass, 14 of 16 new guide candidates have finite support, serial and parallel targets agree exactly, and future-data corruption leaves the fitting target unchanged.
The maximum target-factorization discrepancy is 9.09e-13.

Array `22715405` completed paired fitting seeds 620 and 621 on `mit_preemptable`, each with four CPUs and 20 GiB.
It retains 64 particles, 32 cubic temperatures, eight moves, the original blocks and a 16,448-evaluation cap.
Final-target readers `22715406` and `22715407` completed, recovering each complete checkpoint and freshly evaluating every retained particle.
Their frozen bundle is `logs/uncertainty_balloons_selective_verification_20260914`.
The selective-proposal weighted future adapter in `logs/uncertainty_balloons_selective_forecasts_20260914` has passed native validation and independent verification.
Its particle decoder binds the new guide and all five uniform-coordinate selections; the default-guide decoder cannot be substituted for it.

## Remaining acceptance work

The completed comparison below retains the original and default-centered fits with their separate identities and proposal mappings.
Check parameter marginals, independent-run agreement, budget sensitivity and full weighted reserved-action predictions.
Broader marginals alone do not establish adequate inference in the remaining scene and dynamics coordinates.
Improving numerical exploration also does not fix an inadequate dynamics or discrepancy model.
The [Balloons transition diagnostics](balloons-transition-sensitivity.md) remain relevant to that separate predictive issue.
No Stage B acceptance or live-agent improvement follows from the completed comparison.


## Weighted prediction pipeline

The new pipeline keeps the native generation, density integration and weighted summary methods from the previously verified default-guide adapter.
It binds the new fitting identities and selective proposal, preserving the physical coordinates, complete histories, and positive weights of every retained particle.
It uses the unchanged two banks of four future generations and eight conditional-density draws per particle.
The source-publication step waits for independent completed-fit verification and then creates immutable report and checkpoint copies.

Native adapter `22715541` completed in 1:12 with 3,012 actions, and independent reader `22715542` completed in 27 seconds with another 192 prefix actions.
They pass broad and local component checks, repeatability, full checkpoint recovery, weighted summaries, density denominators and twelve malformed-result rejections.
Both completed reports match the checksums of their frozen inputs and source artifacts.
The reader successfully rejects both the old proposal center and the correct center with its uniform-coordinate selection omitted.
This distinction matters because omitting the selection changes the physical state represented by the same saved proposal coordinates.

Full forecasts `22715569` and `22715571` completed after the completed-fit readers and successful adapter verification.
Independent forecast readers `22715570` and `22715572` failed before verification because their launcher invocation omitted the required numerical index.
Comparison fixture `22715573` and final comparison `22715574` cover the original, default-centered and selective treatments, retaining the incumbent estimator as a separate reference.
The full comparison also waits for the earlier default-guide forecast readers, so unfinished treatments cannot be silently omitted.

The comparison checks identical fitting budgets and forecast draw counts across all treatments, allowing only the declared refresh-probability difference in the sampler configuration.
It reports all ten weighted parameter marginals in prior-unit coordinates, including duplicate sample masses, alongside prediction errors, event probabilities, replica agreement and available computation costs.
Comparison fixture `22715573` completed in seven seconds.
Its synthetic weighted-distribution check and six deliberate invalid-budget, invalid-refresh, invalid-weight and out-of-prior cases pass.
It independently reproduces the original two fits' parameter summaries and retains all four new forecasts as pending.
An incomplete comparison fixture is expected while new forecasts are unavailable; full comparison requires all six verified forecasts.
These reports remain offline evidence and do not establish agent solve rates.

The resulting dependent comparison `22715574` was cancelled.
The exact shell failure was reproduced, and replacement reader array `22755276` supplies both required arguments to the unchanged frozen reader.
Comparison `22755277` follows successful completion of both readers.
No fit or forecast is regenerated for this invocation-only recovery.
The recovery manifest and outputs are in `logs/uncertainty_balloons_selective_reader_recovery_20260914`.

## Completed comparison and information limitation

Recovery readers `22755276_0` and `22755276_1` completed in 27:58 and 28:15, followed by comparison `22755277` in twenty seconds.
Each reader checks 1,024 histories, 2,165,760 joint factors or draws, 153,088 radial-density references and all 64 retained particles' native prefixes, using 4,096 reference actions.
The reader source, script and plan hashes match the frozen files.
The completed comparison is `logs/uncertainty_balloons_selective_reader_recovery_20260914/comparison.json`; all eighteen referenced source hashes and its script and plan hashes were verified.
No fitting or forecasting output was regenerated for the reader recovery.

The table reports predictions on the same development continuation, not agent solve-rate seeds.
The clean trajectory finishes at the goal with no burst.
The incumbent row is its selected-point forecast and retains its different state preparation and historical defaults.

| Treatment | Numerical seed | Box-height RMSE | Box-speed RMSE | Final burst probability | Final goal probability |
|---|---:|---:|---:|---:|---:|
| Incumbent selected point | N/A | 0.0100 | 0.0480 | 0 | 0 |
| Original proposal | 620 | 0.1617 | 0.6034 | 0.4234 | 0 |
| Original proposal | 621 | 0.0973 | 0.3461 | 0.7037 | 0 |
| Default-centered proposal | 620 | 0.1890 | 0.0841 | 0 | 0.0156 |
| Default-centered proposal | 621 | 0.1755 | 0.1167 | 0.0007 | 0 |
| Selective proposal and refresh | 620 | 0.2146 | 0.4759 | 0.9922 | 0 |
| Selective proposal and refresh | 621 | 0.2098 | 0.4678 | 0.9688 | 0 |

The selective proposal broadens the five locally inactive parameter marginals, but does not recover their prior distributions or establish convergence of the joint fit.
Their prior-unit standard deviations span 0.0697 to 0.2389, compared with approximately 0.289 for a uniform prior.
Both selective fits still retain one initial lineage; gold lift and oak mass have zero empirical marginal width.
The fitting evaluations are 13,572 and 13,520, within the unchanged cap.
The between-fit height RMS gap falls to 0.0641, but both height errors worsen, and the speed gap remains 0.1997.
Agreement on a poor prediction is not predictive acceptance.

There is also an information limitation distinct from numerical exploration or simulation error.
The public noisy observations in `posterior-assessment.json` show only the gold balloon attached at step 65, red first attached at step 106, and green first attached at step 220.
The 64-action fitting prefix therefore precedes the red and green releases whose outcomes are assessed later.
The verified dependence audit already shows that changing their lift coefficients leaves the two tested fitting histories unchanged.
The model source documents historically learned gold and red defaults, so an incumbent initialized from those defaults has information that this short-prefix posterior cannot recreate from a broad prior.
These observations do not establish that every future error is caused by missing information, and a single fixed continuation cannot calibrate the broad prior.
They do preclude treating worse point prediction after broadening unseen parameters as proof that a narrower posterior is correct.

The next diagnostic should separate interpolation after observed releases from extrapolation to a new color.
Use an explicitly identified longer development prefix containing the red-release response, assess a continuation before the green release separately, and retain the green-release suffix as an unseen-color case.
Keep the original 64-action case, original priors, fixed program, sensor law and discrepancy law as controls.
First verify whether the added observations constrain the relevant dynamics conditionally before commissioning another full joint fit.
This is pending work, not a launched experiment or an acceptance result.
A new untouched evaluation remains required after selecting an implementation using this inspected recording.
