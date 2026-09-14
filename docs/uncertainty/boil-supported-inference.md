# Boil: supported initialization and finite-factor tempering

This is an offline numerical follow-up to the [shared-variance forecast comparison](shared-variance-discrepancy.md).
The current agent and its production estimator remain unchanged.
The previous replacement populations gave unstable predictions and did not meet the incumbent comparison gate.

## Reproduced failure

The original shared-variance fits use 32 particles on a fixed 132-action training prefix.
Numerical seeds 410 and 411 retain only three and one initial lineages after their first temperature update, respectively.
They eventually both retain one initial lineage.
These are sampler seeds on one development recording, not agent solve-rate seeds.

Compute jobs `22708815_0` and `22708815_1` reproduce the original initial parameter means exactly, the original finite-candidate counts, and the first effective sample sizes.
The target worker, initial-scene map, simulator program, sensor model and physical/parameter priors are the same frozen sources used by the completed shared-variance fits.

| Numerical seed | Original finite candidates / 32 | Initial base-weight ESS | First-temperature ESS | Lineages after first temperature |
|---|---:|---:|---:|---:|
| 410 | 11 / 32 | 2.2643 | 1.8032 | 3 |
| 411 | 7 / 32 | 1.0741 | 1.0572 | 1 |

The first temperature is `1 / 32768`.
The initial base-weight calculation precedes the remaining trajectory-likelihood update.
Consequently, simply inserting smaller trajectory-temperature increments cannot remove the concentration already imposed by the untempered base factors.
This establishes an early numerical problem; it does not explain every later prediction error.

## Target-preserving alternative

Use the existing [whole-joint support initializer](support-initialization.md) to draw complete 84-dimensional proposal vectors until 32 have finite complete targets.
Reject the entire vector, including parameters, when it has zero target density.
Every rejection consumes the evaluation budget.
Do not hold parameters fixed while repeatedly redrawing scenes, which would require a parameter-dependent acceptance correction.

On finite support, set the sampler's base log weight to zero and temper the sum of the original base log weight and remaining log likelihood.
Keep exact zero-support cases rejected at every temperature.
The final unnormalized target is unchanged, up to the single common normalization introduced by whole-joint proposal rejection.
No absolute-evidence estimate is claimed.
The data, program, uncertainty law and original priors are unchanged; the intermediate sampling distributions change.

The two initializations collected 32 supported candidates in 132 and 128 evaluations, respectively.
Each performed 14,652 native simulator actions.
Their original first 32 proposals and initial summaries match the earlier runs.

The following comparison uses exactly the same 32 retained candidates in each column, without temperature moves or resampling:

| Seed | Temperature | ESS with original factor placement | ESS with all finite factors tempered |
|---|---:|---:|---:|
| 410 | 0.000001 | 2.7189 | 31.7001 |
| 411 | 0.000001 | 1.9439 | 31.8150 |
| 410 | 1 / 32768 | 2.0224 | 18.6341 |
| 411 | 1 / 32768 | 1.5069 | 18.6098 |
| 410 | 1 | 1.0000 | 1.0000 |
| 411 | 1 | 1.0000 | 1.0000 |

The last two rows matter: refactoring the tempering path alone does not make the original candidates cover the final target.
Rejuvenation must move them into the important regions while the likelihood is introduced.
A nearly uniform early population is not evidence of a usable posterior.

## Verification and next experiment

The diagnostic source and original failed checker are preserved in `logs/uncertainty_boil_support_diagnostic_20260913`.
The first checker jobs, `22708819` and `22708820`, failed before native verification because the checker requested `original_factorization_max_weight` while the report field is `original_max_weight`.
This is a checker setup failure, not a simulator or agent failure.
The corrected reader lives separately in `logs/uncertainty_boil_support_verification_v2_20260913` and reads the unchanged diagnostic outputs.
Reader jobs `22709035` and `22709037` check the entire rejection/RNG ledger, checkpoint weights and joint targets, scalar ESS references, six deliberate corruptions, and fresh native evaluation of all 32 retained candidates.
Both corrected readers have completed successfully, with 4,224 fresh native actions each and exact agreement for every retained target.

The next fitting driver is frozen in `logs/uncertainty_boil_supported_fits_20260913`.
Two small fixtures, `22709043` and `22709044`, are gated on the corrected initialization readers.
They exercise cached initialization, complete fitting, checkpoint/result agreement, and subsequent numerical-ledger and native-target verification.
Their reader jobs are `22709063` and `22709064`.
Both fixtures and both readers have completed successfully.
The fixtures each perform 6,204 new native actions; each reader adds 4,224 fresh actions and exactly reproduces the complete proposal/target/checkpoint trace and all final joint targets.
The two-temperature fixtures deliberately jump to the final target and collapse to one lineage; they validate mechanics, not numerical adequacy.
The cached initialization must exactly match every saved particle, joint coordinate, factor, weight, ancestor, RNG state and evaluation count.
Cached evaluations remain charged to the fitting budget; the original initialization's native cost is reported separately.

After those checks, the paired full comparison will keep the original 32 particles, eight moves per temperature, proposal blocks, proposal scale and refresh probability.
It uses 64 geometrically spaced temperatures from `0.000001` to one and a maximum of 20,000 target evaluations per run.
The runs use compute nodes on `mit_preemptable` with the previously audited native runtime pinned to `node1412`.
Full fits `22709091` and `22709092` are now running after the fixture readers passed, with their own readers `22709093` and `22709094` dependent afterward.
Each full fit has 16 CPUs, 64 GB memory and an eight-hour allocation.
Independent replicas, numerical-budget sensitivity and reserved-future forecasts remain acceptance requirements.
No replacement posterior or agent advantage is established by this initialization diagnostic.
