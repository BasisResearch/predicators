# Balloons: conditional red-lift profile after an observed release

September 14, 2026.
This follows the verified [action-160 preflight](balloons-selective-guidance.md#longer-prefix-native-preflight).
It remains an offline Stage B diagnostic; the production estimator is unchanged.

## Target and information boundary

The study retains the two candidates selected by maximum fitting weight from the latest selective fits, with lowest index breaking ties.
It holds their initial scene, other parameters and first 64 latent transition directions fixed.
Only red lift varies over its original uniform prior, and the later latent directions are integrated by simulation.
The fitting observations end at action 160, after the red release at 106 and before the green release at 220.
No observations after 160 enter a profile calculation.
The recording has already been inspected during development; it is not fresh acceptance data.
The original program, sensor noise, motion discrepancy and complete replay from candidate initialization remain unchanged.

At each red-lift value, the likelihood estimate averages the complete conditional-history densities over all declared latent draws.
A history that violates an exact observation contributes zero and remains in the denominator.
The midpoint grid integrates over the full prior-unit interval; it does not truncate the physical prior or turn a previous fit into a prior.
The saved first-prefix likelihood must reproduce exactly for every red-lift value.

Each resulting normalized grid is conditional on one fixed nuisance history.
It is not a full joint parameter posterior, a calibration result or a forecast.
Differences between the two fixed histories reflect different conditional targets, while differences between draw banks or grid resolutions measure numerical sensitivity within one target.

## Numerical comparisons

| Mode | Midpoint grid sizes | Independent banks | Draws per bank and node | Native profile histories |
|---|---|---:|---:|---:|
| Fixture | 4 and 8 | 2 | 1 | 48 |
| Full | 16 and 32 | 2 | 4 | 768 |

Both grid resolutions span the complete original prior support.
Within each fixed history, common random seeds are reused across red-lift values and grid resolutions; the two banks use distinct seeds.
Reports retain each bank separately and their pooled calculation.
They include the log integral, conditional mean and standard deviation, grid masses, effective number of occupied grid nodes, and latent-draw effective sample sizes at each node.
No agreement statistic automatically marks the result as an adequate posterior.
All-zero profiles remain explicitly unavailable.

The full study requires 122,880 profile actions plus up to 512 worker prefix-reference actions.
Independent verification adds its separately recorded native work.
Each job uses four CPUs and 20 GiB on a previously audited Intel node in `mit_preemptable`.
Fixture allocations request thirty minutes; full allocations request two hours.
These are fixed numerical budgets, not early-stopping rules selected after seeing a result.

## Verification and jobs

Scalar guards check a uniform reference, a linear-likelihood reference with an analytic midpoint result, stable arithmetic for large log values, all-zero support and preservation of zero-density draws in the averaging denominator.
The independent reader reconstructs the midpoint and latent averages with separate scalar arithmetic, checks every saved history's native factors and repeats selected complete histories in fresh worlds.
It verifies that only red lift changed from each selected candidate, that the old prefix remains exact, and that recorded native costs match the returned work.

Fixture `22765047` is followed by reader `22765048`.
Full native job `22765049` depends on that reader, and full reader `22765050` follows it.
The full driver also checks the completed fixture certificate inside the process.
Python and shell syntax checks pass.
The five analytical scalar guards, independent summaries of four reference tables and rejection of a corrupted mean also pass without initializing a simulator.
Native execution and verification are pending.
Frozen inputs and output locations are in `logs/uncertainty_balloons_red_profile_20260914`.

The next action depends on the observed grid and bank sensitivity.
A concentrated single-node profile or disagreement across budgets requires further numerical investigation, rather than a fitted-width claim.
A stable conditional calculation would still require a separate joint target and prediction validation, including the unseen-green continuation and fresh untouched evaluation before migration acceptance.

## Audited node eligibility

While all four jobs were still pending with zero runtime and restarts, their scheduler eligibility was expanded from `node1393` to `node1381`, `node1390` or `node1393`.
Earlier successful native jobs `22652531` and `22692309` establish the corresponding replay references on the first two nodes; the current action-160 preflight used `node1393`.
The runtime CPU-model assertion and all archived-prefix and fresh-history checks remain enabled.
Only scheduler placement changed: job IDs, dependencies, resource requests, time limits, scientific budgets and frozen source hashes were verified unchanged.
The before/after evidence is in the study bundle's `allocation.json`.
The fixture remains pending; a larger eligible node pool is not a claim that execution has started.
