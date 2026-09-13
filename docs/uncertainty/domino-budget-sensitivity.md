# Domino population-size sensitivity

September 13, 2026.
This is a Stage B numerical comparison under the [simplification proposal](simplification-proposal.md), using the same development recording and frozen simulator as the completed [64-particle forecasts](domino-comparison-summary.md).
These are offline inference replicas, not agent experiments or additional independent task datasets.

## Why this comparison

The 64-particle joint fits retain one or two initial ancestors and disagree on some decision-relevant predictions.
The point-start approximation retains five or six ancestors and passes the initial prediction-agreement screens, but that does not establish convergence or trustworthy parameter uncertainty.
The completed local-scale audit shows irregular target changes near retained scenes, with no uniformly better smaller proposal scale.
The next controlled numerical change increases population size instead of changing the probability model or selecting a new proposal heuristic.

## Controlled protocol

Run both the complete joint target and the separately labeled first-observation point-start approximation with 128 particles, using numerical seeds 100 and 101 for each.
Each run starts from its original declared prior/proposal, with no reuse of a completed 64-particle population as an initialization.
The only numerical configuration changes are 64 to 128 particles and a maximum evaluation budget of 16,448 to 32,896.
Retain 32 cubic-spaced temperatures, eight moves per stage, local scale 0.05, the 50/50 local/full-range mixture and the original proposal blocks.
The joint representation has 25 blocks; the point-start representation has five.
These are independent numerical replicas within each setting; using the same seed numbers across population sizes does not make their later random schedules identical.

The data, physical program, parameter prior, initial-state policy, output model and native runtime remain fixed within each state treatment.
Worker, setup, preparation and driver files are copied byte-for-byte from their corresponding 64-particle bundles.
The new run directories contain no previous inference checkpoints.
Each new fit repeats its archived native preflight before sampling and then saves complete-stage checkpoints under the changed numerical configuration.
All four tasks have passed those preflights and started sampling on compute nodes.

| State treatment | Fit tasks | Dependent forecast tasks |
| --- | --- | --- |
| Joint uncertain initial state | `22675729_0`, `22675729_1` | `22675741_0`, `22675742_1` |
| Fixed first-observation state | `22675730_0`, `22675730_1` | `22675743_2`, `22675744_3` |

Each fit requests four CPUs, 20 GB and six hours on node1412 in `mit_preemptable`.
Each two-task array permits at most two concurrent replicas.
Forecasts retain the existing four-CPU, 20-GB, 30-minute allocation and begin only after their corresponding fit succeeds.
Runtime accounting must include actual allocation cost and repeated work after interruption, separately from the sampler's cumulative evaluation count.

## Evaluation fixed before outcomes

Use the original 64-action fitting prefix and 97-action causal suffix.
The forecast driver remains unchanged and restores the complete checkpoint, verifies the retained physical samples and prefix factors, then propagates every positive-weight particle through the full recorded history.
It repeats the first complete history and records all weights and explicit zero-weight rows.
No forecast or future observation alters the fitted population.

For each state treatment, compare both 128-particle replicas with each other and all four 64-versus-128 replica pairs.
Retain the existing thresholds of 2.5 mm for RMS differences in conditional position means, 0.20 for the largest toppling-curve gap and 0.15 for the largest final toppling gap.
Report every pair rather than selecting the best agreement.
Also inspect parameter quantiles, mass at repeated values, ancestry, position errors, toppling errors, event timing and compute cost.
Do not pool the joint and fixed-state populations or treat them as estimates of the same target.

These thresholds remain exploratory screens, not proof of calibrated uncertainty or completed Stage B acceptance.
A stable but poorly predictive result still needs its prediction failures reported.
A failure at the larger population size is evidence against relying on apparent stability at the smaller size.
Broader recordings, active carried-center coverage, the other domains and later planning gates remain necessary.
The production agent is unchanged.

Frozen bundles and submission hashes are in `logs/uncertainty_domino_budget_joint_20260913`, `logs/uncertainty_domino_budget_point_20260913` and `logs/uncertainty_domino_budget_forecast_20260913`.
These finite inference and prediction jobs do not re-enable the MB/MF notification monitor.
