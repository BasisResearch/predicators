# Domino coupled-parameter direction diagnostic

This is a Stage B investigation of numerical exploration after the [population-size comparison](domino-budget-sensitivity.md) failed to establish stable predictions.
It evaluates fixed directions under the original point-start target; it does not produce a posterior or an agent result.
The original joint-state inference problem remains unresolved.

## Question and controlled inputs

Could coordinated changes to the five dynamics parameters reach plausible candidates that separate coordinate changes cannot reach?
The diagnostic uses the two completed 128-particle point-start populations, with numerical seeds 100 and 101.
The frozen simulator, original parameter prior, first-observation state, observation model and 64 fitting actions are unchanged.
No reserved future observations select the anchors or directions.

Eight equally spaced weighted quantiles select anchor rows from each population, retaining duplicate rows if selected.
For each anchor, two ordered pairs of distinct donor rows are sampled uniformly from its own population, excluding the anchor row.
The donor difference is scaled by 0.5 or 1.0 and receives the same fixed uniform jitter in [-0.001, 0.001] in unit-prior coordinates.
For each displacement, the audit evaluates the joint five-coordinate move and the five separate single-coordinate moves from the same anchor.
The proposal random seed is 931705.
All 384 candidates are retained in the report, including the 70 outside the unit box, which receive zero acceptance without a native rollout.

The audit first requires exact reproduction of all 16 archived anchor states and likelihoods, plus serial/parallel equality for four anchors.
At the final temperature, the original parameter prior is uniform in these coordinates and the removed initial-state factor is constant.
The score proxy is therefore `min(1, exp(candidate_log_likelihood - anchor_log_likelihood))`.
Expected squared jump is that proxy times the squared displacement in unit-prior coordinates.
The report compares each joint proposal with the mean across its five scalar proposals, making the per-proposal calculation explicit.
It does not compare against five sequential accepted coordinate updates.

These fixed-direction measurements do not establish detailed balance for an interacting ensemble sampler, posterior accuracy, or improvement in planning.
Any resulting sampler change would need its own invariant-target reference checks and matched inference comparisons.

## Execution and evidence

Bundle: `logs/uncertainty_domino_coupled_directions_20260913/`.
The manifest pins both source reports and checkpoints, the archived worker and setup files, every anchor, and every proposed displacement.
The driver uses the original snapshot preparation and target worker without editing them.
Job `22681514` runs on `mit_preemptable`, node1412, with four CPUs, 20 GB and a 30-minute allocation.
The independent summary reader checks score arithmetic, paired displacements, all proposal groups, source hashes and the evaluation count.
Job `22681514` completed in 156 allocation seconds (four CPUs), with 150.426 measured driver seconds.
It performed 334 native target evaluations, including 20 anchor checks, for 21,376 simulated fitting actions.
All 16 archived anchors reproduced exactly; the four serial/parallel comparisons were exact.
The reader verified all 384 proposal records and 64 matched direction groups and rejected five corrupted report variants.

## Broad-direction result

Expected squared jumps below are per proposed evaluation, including out-of-box rejection, in squared unit-prior coordinates.
The scalar column averages five alternative one-coordinate proposals from the original anchor, not a sequential sweep.

| Numerical seed | Direction scale | Joint, all coordinates | Scalar, all coordinates | Joint excluding restitution | Scalar excluding restitution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.5 | 3.553e-8 | 0.00476847 | 3.361e-8 | 2.986e-5 |
| 100 | 1.0 | 0.0155905 | 0.0190031 | 0.00175850 | 1.987e-5 |
| 101 | 0.5 | 1.503e-8 | 0.00431245 | 1.825e-9 | 5.933e-7 |
| 101 | 1.0 | 2.287e-9 | 0.0130366 | 2.291e-10 | 0.000136446 |

Joint proposals exceed their corresponding five-scalar mean expected jump in only one of 64 directions.
However, all 50 in-box restitution-only proposals have exactly unchanged likelihood, so movement in that coordinate dominates the aggregate scalar mean.
This makes aggregate movement a poor proxy for movement in the remaining parameters.
Excluding restitution, the seed-100 scale-1 joint group has a larger mean jump than the scalar alternatives, but the effect does not recur in the other three groups.
These observations do not support deploying a broad donor-difference proposal as a general repair.
They also do not rule out useful coupled moves at smaller scales or other parameter directions.

## Smaller-scale follow-up

Job `22681592` repeats the exact anchors, donor pairs, jitter draws and archived target with scales 0.02 and 0.1.
Its separate bundle is `logs/uncertainty_domino_small_directions_20260913/`.
It retains 384 proposals, of which 376 are inside the unit box.
This isolates step scale without fitting new populations or altering the probability model.
The job completed in 178 allocation seconds (four CPUs), with 172.861 measured driver seconds.
All archive and serial/parallel checks passed again.
The report contains 396 native evaluations and 25,344 simulated fitting actions.
An independent affine-direction check confirms the same anchors, donors and jitter as the broad-scale audit.

| Numerical seed | Direction scale | Joint, all coordinates | Scalar, all coordinates | Joint excluding restitution | Scalar excluding restitution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.02 | 4.449e-9 | 1.087e-5 | 2.914e-9 | 7.055e-9 |
| 100 | 0.1 | 2.357e-8 | 0.000276109 | 1.630e-8 | 5.439e-9 |
| 101 | 0.02 | 2.886e-5 | 1.423e-5 | 2.766e-5 | 1.240e-6 |
| 101 | 0.1 | 0.000311976 | 0.000181571 | 0.000273650 | 3.930e-7 |

Smaller joint moves produce more expected movement in population 101 than the scalar alternatives, including after excluding restitution.
Population 100 still barely moves, so a common smaller scale does not establish a general remedy.
The fixed jitter is not scaled with the donor difference and may disrupt narrow correlated directions.

## Jitter isolation

Job `22681711` repeats scales 0.02 and 0.1 with exactly the same anchors and donor pairs, removing only the additive jitter.
Its bundle is `logs/uncertainty_domino_unjittered_directions_20260913/`.
All 384 proposed points have been independently reconstructed from the archived donor rows and verified exactly.
There are 22 zero-displacement proposals; these contribute zero expected jump even if accepted and must not be interpreted as exploration.
The job completed in 177 allocation seconds (four CPUs), with 171.382 measured driver seconds.
All archive and serial/parallel checks passed, with 396 native evaluations and 25,344 simulated fitting actions.

| Numerical seed | Direction scale | Joint, all coordinates | Scalar, all coordinates | Joint excluding restitution | Scalar excluding restitution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.02 | 1.821e-7 | 1.120e-5 | 8.222e-8 | 1.187e-7 |
| 100 | 0.1 | 1.831e-6 | 0.000277716 | 9.560e-7 | 1.754e-7 |
| 101 | 0.02 | 1.250e-8 | 1.299e-5 | 6.593e-11 | 7.039e-9 |
| 101 | 0.1 | 3.822e-6 | 0.000196525 | 2.632e-6 | 2.037e-5 |

Removing jitter improves expected joint movement in population 100 but reduces it in population 101 at both scales.
The earlier gain from smaller joint moves is therefore not robust to this controlled perturbation.
This is consistent with the previous irregular local-target findings, but does not identify their physical cause.
The three audits together cover 1,152 candidate proposals and 72,064 simulated fitting actions, costing 511 allocation seconds on four CPUs.
All three final report readers reproduce their complete records and reject five corrupted variants each.
No sampler or production agent was changed.

## Next decision

Do not select a new donor-direction kernel from these results alone.
Separate the score differences into observation channels and contact/event differences for matched nearby candidates to identify what prevents movement.
That attribution must retain the existing likelihood and all fitted observations; it must not remove inconvenient factors or select a discrepancy model using the reserved future suffix.
Any justified inference change still requires numerical reference validation and repeated physical prediction comparisons before Stage C.

The executable audit drivers, native reports, full per-parameter summaries and reader checks are retained in their three named bundles.
Each `reader-validation.json` records the final script, manifest, native-result and summary hashes.
