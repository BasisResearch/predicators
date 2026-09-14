# Bridge causal futures from a supported prefix

September 13, 2026.
This is an offline integration check for the [uncertainty simplification plan](simplification-proposal.md).
It uses the combined initial-face correction from the [Bridge support diagnostic](bridge-glue-support.md), with parameters chosen from the 600-action development prefix.
It is not a posterior population, an agent experiment or evidence of predictive adequacy.

## Conditional prefix and unconditional continuation

The driver reproduces the entire verified 600-action conditional prefix in a fresh native world, including initial state, observations, joint factors, reach-event masses and learned memory.
It then continues the same world through all 586 reserved actions.
The conditioning context contains exactly 601 observations, including the initial frame; accessing observation 601 raises an index error.
Future observed joint positions and glue readings are never supplied to generation.
The recorded future actions are fixed inputs to this offline forecast.

For each of the nine robot joints, the driver draws one variance from its inverse-gamma posterior at the prefix boundary and retains that variance for the whole continuation.
After each future native step, it adds an independent zero-mean Gaussian position increment with that joint's sampled variance, retaining the native joint velocity.
Before each future learned glue update, it draws an independent Gaussian reach offset with standard deviation 0.005 m.
The unchanged learned update uses that offset without conditioning on any future glue reading.
The prefix's interval-conditioning algorithm is not used to select future reach offsets.

The existing output model generates complete future observations from those causal physical predictions.
It conditions correlated scalar output errors only on the fitting prefix, preserves coupled robot-orientation errors and derives the exact finger readout from its generated source joint.
Variance, joint-increment, reach and output sampling use separate random streams derived from the numerical seed.
The output-model identity matches the independently checked prefix composition exactly.

## Completed checks

Native job `22703457` completed in 2:56.
Numerical seeds 710 and 711 each generate a distinct complete continuation, and both complete histories repeat exactly from fresh worlds.
The four histories total 4,744 native actions.
These numerical seeds are future draws at one fixed candidate, not solve-rate seeds.

Independent reader `22703621` completed in 1:07.
For each continuation, it checks all 5,274 joint increments against the retained variance draws, all 586 reach draws and every literal glue update and learned-memory transition.
It also verifies that generated joint readouts match the corrected native states and that the output sampler covers all declared fields.
The previously validated output sampler round-trips both saved observation histories exactly; this part is an integration check rather than a new independent derivation of its statistical law.
Changed conditioning length, variance, reach offset, model memory and exact output values are all rejected.

## Remaining work

This supplies the causal generation component required by a future Bridge posterior forecast.
It does not supply a future-density estimator, a fitted population or a held-out comparison verdict.
Complete the [joint inference target](bridge-joint-inference.md), assess its numerical adequacy and evaluate population forecasts against the incumbent.
Keep the current agent estimator unchanged until the plan's prediction and live-validation gates pass.

Artifacts are in `logs/uncertainty_bridge_future_20260913/`.
