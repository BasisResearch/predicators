# Noisy balloons MB seed 1: failure analysis

The decisive mistake was adding blue after green on the test level.
Green alone was still recoverable: replaying that exact state and releasing red instead wins in 43 additional steps, for 77 test-level steps in total.
The failed run used a misspecified analytical model to choose the next balloon after its simulator fit had rejected the training trajectory.
This investigation also reproduced a separate task-generation error: the purported red+green jamming decoy actually wins if the rollout continues past its first low-speed turning point.

## Runs and reproduction

The failed run is `balloons-agent_continual_cross_noise_on`, seed 1, `run_20260908_201805`.
It won both training levels, then gave up on the test level after 109 test steps; its whole-run totals are 340 steps and one training reset.
The comparison is `balloons-agent_continual_cross_noise_off`, seed 1, `run_20260909_021331`, which won all three levels in 315 steps and one training reset.
Both used 10 mm position noise, 0.02 rad orientation noise, and the same frozen code based on `92d37ec9723b`.

Their recorded test initial states match exactly across every state feature.
The regenerated test task also matches that recorded initial state exactly.
Replaying all recorded primitive actions reproduced both runs' box-height trajectories with zero error: the failed run had no win, while the comparison won at test step 48.
The failure is therefore reproducible in the physical environment; it is not a scheduler crash or an inaccurate scorecard.

The test uses an oak box, red/blue/green/gold balloons, and the target band **0.505117 to 0.555117 m**.
Test resets are disabled by the protocol.

| Decision or counterfactual | Result in the physical environment |
|---|---|
| Actual first move: release green | Box stays at about 0.435 m; recovery remains possible |
| From that exact state, release red and wait | Wins after 43 additional steps, at 0.521828 m |
| Actual second move: release blue | Blue+green settles at 0.600 m, above the band |
| Keep waiting after blue+green | No win in 200 additional steps |
| Actual final move: add red to blue+green | No win; longer continuation settles at about 0.710 m |
| Release gold alone from the initial state | Wins in 48 steps, reproducing the successful comparison |
| Release red, then green, from the initial state | Wins in 76 steps |

The failed agent's invocation note explicitly predicted that blue+green would be in band for a plausible oak/pine mass ratio.
That prediction was wrong.
The real primitive actions completed successfully, and no balloon burst during the recorded failure.
The agent's later claim that red+blue would have been the correct solution was also wrong: the physical replay leaves that pair near 0.438 m, below the band.

## Model and uncertainty evidence

The saved simulator gives all balloon colors the same `lift_lambda`, applies force directly to the box, and does not implement the attachment mechanics.
It therefore lacks the color-dependent physics needed to transfer from the two training levels to the four-balloon test rack.
Its force law also depends on an assumed balloon/string height, whereas the hidden environment computes lift from the box height.

The logged first canonical `sim.fit()` rejected its only training segment: normalized RMS 0.1541 exceeded the 0.1000 acceptance threshold.
The agent then edited parameter initial values and relied on a separate one-dimensional model in `probe_ext.py` and ad hoc model comparisons.
Later harness fallback fits continued to reject most segments.
The irreversible test releases were not supported by a validated full-plan prediction from a simulator that explained the available data.

The agent's two-model Monte Carlo calculation was its own analytical calculation, not a call to `sim.belief()` or the harness posterior sampler.
The successful comparison also used analytical inference and retained a no-op simulator, so its success does not demonstrate that the feature-off simulator was better.
Its journal records a better-calibrated box-height model and a hypothesis about how training bands were positioned; it selected gold alone.

The execution-belief smoother does not replace `data/trajectories.pkl` with smoothed frames.
For both runs, every exported level-1 state feature exactly matches the independently regenerated raw noisy observation for the same seed, level, episode and step.
Thus this failure is not explained by the smoother silently corrupting the hand-fit's recorded trajectories.
The different agent decisions do not, by themselves, establish which uncertainty flag affected the outcome.

An offline replay of the first fit used the exact saved simulator and training data, changing only the six uncertainty switches.
The enabled case reproduced the original logged RMS, and both cases rejected all training segments without applying parameters.

| Uncertainty switches | Input steps | Prepared segment steps | Best normalized RMS | Accepted segments |
|---|---:|---:|---:|---:|
| All on | 82 | 41 | 0.154058 | 0 |
| All off | 82 | 82 | 0.155205 | 0 |

The acceptance threshold was 0.1000 in both cases.
The new flags therefore did not cause this particular fit rejection.
This is a fit-level counterfactual, not an end-to-end rerun of the agent or an attribution of its later decisions to individual flags.

## Separate task-generator defect

`PyBulletBalloonsEnv.subset_outcome()` stops at the first low-speed state after movement.
For an oscillating box, this can be a turning point before it has settled.
For red+green, the helper stops at step 22 with the box at 0.498572 m and returns failure without a burst.
The task generator treats that result as a jamming decoy.
Continuing the same dynamics instead reaches a valid win at step 31, at 0.521836 m.
Executing actual Release skills also wins with red+green in either tested order.

This task therefore has at least two successful balloon subsets, gold and red+green.
Its claimed unique-solution/jamming-decoy property is false.
The actual agent win/loss results remain valid, but this seed cannot support a claim that contact reasoning was necessary to distinguish a losing decoy.
The early-stop code is unchanged between the frozen experiment checkout and the current main checkout at the time of this investigation.

## Follow-up

Correct the subset-outcome check to continue through oscillation turning points and distinguish persistent contact failure from temporary low speed, then revalidate task generation separately from the current frozen sweep.
For the agent failure, require a simulator that explains the training trajectories and a validated whole-plan prediction before relying on its uncertainty estimates for irreversible test actions.
Loosening the fit rejection threshold would not supply the missing color-dependent physics.

Evidence and replay scripts are in `logs/balloons_failure_20260909/`.
Physical action replay completed in Slurm job `22364813`, and longer subset/skill counterfactuals completed in `22365081`, both on `mit_preemptable`.
The first-fit comparison completed in job `22365267` on the same partition.
The diagnostic scripts changed no runtime source, experiment configuration, or recorded agent result.

Source records: [failed scorecard](../../logs/agent_continual/balloons-agent_continual_cross_noise_on/seed1/run_20260908_201805/scorecard.json), [failed test actions and notes](../../logs/agent_continual/balloons-agent_continual_cross_noise_on/seed1/run_20260908_201805/L03/index.jsonl), [failed simulator](../../logs/agent_continual/balloons-agent_continual_cross_noise_on/seed1/run_20260908_201805/agent/sandbox/simulator.py), [successful comparison journal](../../logs/agent_continual/balloons-agent_continual_cross_noise_off/seed1/run_20260909_021331/agent/sandbox/journal.md), [action replay results](../../logs/balloons_failure_20260909/replay.json), [counterfactual results](../../logs/balloons_failure_20260909/counterfactuals.json), [fit comparison](../../logs/balloons_failure_20260909/fit_replay.json).
