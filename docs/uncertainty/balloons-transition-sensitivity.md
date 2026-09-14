# Balloons: sensitivity to future dynamics noise

September 14, 2026.
This diagnostic follows the failed [prefix posterior forecasts](balloons-prefix-forecasts.md).
It tests conditional continuations, not a refitted probability model or agent performance.

## Controlled comparison

The current discrepancy model perturbs arm-joint positions with standard deviation 0.001 and samples a box-velocity transition after every native action.
That velocity transition has a 0.1 probability of setting all three linear velocity components to zero; otherwise it adds Gaussian noise with standard deviation 0.01 per component.
These physical interventions affect subsequent contacts and can therefore change irreversible balloon bursts.

We select the highest-weight particle from each completed prefix population, independently of future prediction error.
These are particle 48 of fit seed 620, with weight 0.05047, and particle 18 of fit seed 621, with weight 0.05974.
Each particle retains its complete 206-coordinate parameter, initial-state and conditional-direction vector.
Four archived future random seeds per particle are crossed with four interventions: both noise effects, joints only, velocity only, and neither.
All 32 histories reconstruct the same first 64 actions and then continue through the remaining 171 actions in one fresh world.
Suppressed interventions still consume their random draws, preserving the paired random schedules.
Future observations are absent from generation and enter only the subsequent assessment.

The frozen bundle is `logs/uncertainty_balloons_transition_sensitivity_20260914`.
Native job `22713764` completed in 2:32 on a compute node with the same CPU model as the archived runs.
It executed 9,400 native actions, including eight complete repeated histories.
All eight original-treatment paths reproduce their archived complete histories exactly, and all interventions preserve their original fitting-prefix trajectories and scores exactly.
Independent reader `22713765` completed in 12 seconds.
It checks all 5,472 future joint vectors and velocity draws, recomputes event labels and feature errors, and rejects four deliberately corrupted prefix, joint, velocity and event records.

## Results

Each row uses four simulated continuations from one selected fitted state.
Errors are averages of the four individual future-trajectory RMSEs against the reserved clean development trajectory.
These draws are not agent seeds, and their goal counts are not solve rates.

| Fitting seed | Active future noise | Box-height RMSE (m) | Box-speed RMSE (m/s) | Final burst count | Final predicted goal count |
|---|---|---:|---:|---:|---:|
| 620 | Joint and velocity | 0.18992 | 0.22886 | 0/4 | 0/4 |
| 620 | Joint only | 0.20728 | 0.26137 | 0/4 | 0/4 |
| 620 | Velocity only | 0.20678 | 0.25179 | 0/4 | 0/4 |
| 620 | Neither | 0.22072 | 0.23738 | 0/4 | 0/4 |
| 621 | Joint and velocity | 0.22680 | 0.65371 | 4/4 | 0/4 |
| 621 | Joint only | 0.10548 | 0.41266 | 1/4 | 0/4 |
| 621 | Velocity only | 0.36394 | 0.88007 | 4/4 | 0/4 |
| 621 | Neither | 0.11345 | 0.37631 | 0/4 | 0/4 |

At the selected seed-621 state, the velocity intervention contributes to spurious bursts and large motion errors.
Removing both interventions eliminates bursts in these four continuations, but still does not recover the goal.
At the selected seed-620 state, removing noise does not improve either reported error.
Thus future transition noise explains part of the regression, but does not explain all of the fitted-state and dynamics error.
There is no basis for promoting a simple noise-disable change to production.

The next model investigation should separate the velocity reset atom from continuous velocity noise and inspect errors already present in the conditioned prefix.
Any proposed replacement must use a coherent law during both fitting and forecasting, then be refitted and assessed across complete weighted populations.
These two selected states cannot establish posterior-wide improvement, numerical convergence, or the Stage B acceptance gate.
