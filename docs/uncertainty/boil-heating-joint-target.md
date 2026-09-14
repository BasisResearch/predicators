# Boil: joint inference with informative heating observations

September 14, 2026.
This extends the [conditional heating diagnostic](boil-informative-heating.md) to a complete joint target over the initial scene and all eight simulator parameters.
The target and proposal checks have passed, while the matched fitting experiments are queued.
This remains Stage B development validation; the production estimator is unchanged.

## Probability model

The new fitting prefix contains 224 actions and all 225 observations, including every observed channel.
Every candidate replays from its own initialization and carries its own simulator memory.
The original physical priors, simulator program, sensor noise and previously declared discrepancy laws remain fixed.
All 84 joint coordinates are retained.
The thermal independence certificate from the earlier all-off prefix does not apply to this target.

The finite joint score includes the initial-scene correction, complete observation likelihood and robot-motion transition factors.
The transition variance is integrated under the existing inverse-gamma law.
The complete finite score is tempered during sampling, while impossible exact observations remain hard support failures.
The omitted scene-prior normalizing factor is common to all candidates; this calculation is not a model-evidence estimate.

The new prefix has a distinct data and runtime identity.
Observations and actions after 224 cannot affect its target or initialization guide.
The remaining 40 actions are reserved from this fit for subsequent forecasting.
They belong to a previously inspected development recording, so success on this suffix would still require fresh untouched evaluation before migration acceptance.
The original 132-action experiment remains a separate prior-retention and extrapolation case.

## Completed target checks

The first preflight, job `22757278`, stopped before target evaluation because the adapter supplied the raw sensor digest where the established identity requires the composite observation and transition law.
The guard caught the mismatch.
The separately frozen v2 bundle preserves that composite identity; it does not change the sensor or probability model.

Native preflight `22757366` completed with 2,016 simulator actions.
It reproduces the archived physical prefix, verifies serial/parallel equality, rejects an exact burner-state contradiction, and confirms that an informative in-prefix reading changes both the score and data identity.
Changing later observations and actions leaves the prefix target unchanged.
Independent reader `22757367` checks eight complete targets using literal feature-update rules, all observation factors and a separate closed-form calculation of the integrated joint-transition factors.
Both jobs completed successfully.
Their source and script checksums and frozen input manifests have been checked after completion.

The checked preflight checksum is `a4c537f2030d82f44f937f06278d61f3a25b04c100827d5feb881392b72dee8e`.
These checks establish target implementation consistency, not accurate posterior integration or predictions.

## Corrected sampling guide

The optional guide proposes heating onset and width near the independently verified conditional calculation.
Its centers are 30.12237 and 5.71475, with proposal standard deviations twice the conditional standard deviations.
A 25% mixture component samples the entire original uniform thermal prior.
The other component is a product of truncated Gaussian proposals.
The target adds the log ratio of the original thermal prior density to the complete mixture proposal density exactly once.
Consequently, using this guide does not narrow the prior or assert that the posterior parameters are independent.
Every candidate still receives its complete scene-dependent likelihood.

Native preflight `22757771` checks twelve combinations of source scene, mixture component and thermal proposal coordinates, with 1,344 simulator actions.
Independent reader `22757772` reconstructs the normal CDF mapping and mixture density separately, rejects a deliberately omitted correction, and repeats two native targets with 448 simulator actions.
Both pass.
The checked preflight checksum is `db0952a62ede8afa7333ca50a7ac216ba4f1447aaad0a5fe2888efda761bacc0`.

## Matched fitting experiment

All four short fits now pass their numerical and native readers after the [documented JSON transport recovery](boil-heating-reader-recovery.md).
The job table below records the original submissions; cancelled downstream jobs and their replacements are listed in that recovery note.
Both arms use numerical seeds 410 and 411, 32 particles, the same coordinate blocks and matching evaluation budgets.
The uniform arm includes an unused normalized selector coordinate so both arms have 85 proposal coordinates and 84 joint output coordinates.
The guided arm uses that selector for the thermal mixture.
The full fits use 64 temperatures, eight moves, a 20,000-evaluation cap and fresh support initialization.
No old fitting checkpoint or posterior is substituted for the original prior.

Each arm first runs a short fixture with two temperatures and one move.
Its reader reconstructs the entire numerical trace from the evaluation ledger, independently checks proposal mappings and corrections, and replays all final native targets.
Full fits depend on successful fixture readers and also check the verified reports inside the fitting program.
Each full fit has its own subsequent numerical and native reader.
Sampler completion alone leaves the assessment unavailable.

| Arm | Numerical seed | Fixture | Fixture reader | Full fit | Full reader |
|---|---:|---:|---:|---:|---:|
| Uniform | 410 | 22758853 | 22758855 | 22758883 | 22758884 |
| Uniform | 411 | 22758857 | 22758858 | 22758885 | 22758886 |
| Guided | 410 | 22758859 | 22758860 | 22758887 | 22758888 |
| Guided | 411 | 22758861 | 22758862 | 22758889 | 22758890 |

At submission, the fixtures were pending on `mit_preemptable` because their eight-hour allocations overlapped the scheduled maintenance reservation.
Those short jobs had inherited the full-fit wall-time request.
Four independently verified earlier fixtures took 76 to 180 seconds with the same particle count, two temperatures, one move and 2,048-evaluation cap.
Scaling the slowest observed seconds per evaluation to the full cap and the longer 224/132-action ratio gives an empirical estimate of 3,479 seconds, not a guaranteed runtime bound.
The eight still-pending fixture and fixture-reader allocations were therefore reduced to two hours through scheduler metadata only.
All job IDs, zero restart counts, frozen input hashes and numerical budgets were preserved; at that point the full fits retained their original eight-hour requests.
All four fixtures subsequently completed, and their recovered independent readers pass.
The before/after scheduler records and checked timing references are saved in `logs/uncertainty_boil_heating_fixture_allocation_20260914`.
The original full fits were cancelled by failed reader dependencies before starting.
Replacement full fits are underway with unchanged numerical budgets and six-hour allocations justified by verified full-fit timing references.
No full fitting result is available yet.
These are numerical inference seeds, not new agent solve-rate seeds.

## Remaining work

Verify the fixture and full-fit results and complete validation of the [corresponding 224/40-action forecast adapter](boil-heating-forecasts.md).
The native continuation fixture now passes, and the population forecast pipeline is queued behind its validation and fitting dependencies.
Compare numerical replicas, prior retention where justified, posterior predictions and computation costs.
The longer prefix supplies heating information but does not itself fix uncertainty about the initial scene or poor exploration of the joint posterior.
Between-replica agreement, budget stability and fresh predictive evidence remain required before using the replacement in live planning.

Frozen target, guide and fitting bundles are respectively `logs/uncertainty_boil_heating_joint_target_v2_20260914`, `logs/uncertainty_boil_heating_guided_target_20260914` and `logs/uncertainty_boil_heating_fits_20260914`.
