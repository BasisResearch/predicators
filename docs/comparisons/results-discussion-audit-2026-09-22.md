# Evidence for the Results and Discussion draft

This note supports the Q1--Q3 draft in `/home/ycliang/sim-predicator-paper/main.tex`.
Q4 and the real-robot discussion are excluded at the author's request.
This is an author-facing evidence record, not manuscript text.

## Scope and source selection

The paper figure's [selection manifest](figures/paper-results-opus-summary.json) now contains 175 completed runs: 25 per agent.
The Fan column uses the ramp setting.
Archived maze, exposed-transfer, and inertial Fan results do not enter the paper's totals.
The wider [benchmark report](ten-agent-opus-benchmark.md) links 319 run directories, including unfinished and archived runs.
The audit read their scorecards, agent conversation logs, journals, final simulator files, level event and action logs, environment logs, and available validation logs: 3,703 files totaling 739,339,059 bytes.
It also read 374 additional top-level agent-authored Python and Markdown files for the selected runs, including standalone world models and separate level journals.
The review combines full-file programmatic inspection with detailed reading of failure reports, journals, tool results, and representative successful decision sequences.
It does not replay the binary recordings, inspect every video frame, or execute agent-authored code.

The [audit inventory](results-discussion-audit-2026-09-22.json) freezes the original 174 completed sources, outcomes, per-level accounting, and tool-request counts used for this draft.
The final Oracle completion was verified separately against its scorecard and journal and is documented below; the original inventory remains a historical snapshot.
The paper's existing configuration definitions are used to name the arms consistently.
Historical configuration differences are not described in the manuscript discussion, and the draft does not claim that the logs constitute a controlled test of the streamlined uncertainty implementation.
In particular, descriptions of posterior sampling in the method section are not established empirically merely by relabeling these historical runs.

Oracle Fan seed 0 has now finished successfully: its [scorecard](../../logs/agent_continual_oracle_dynamics/fan-oracle_dynamics_opus_fan_prompt_r2/seed0/run_20260922_055537/scorecard.json) reports `all_levels_won`, 2/2 levels, 3,280 total steps, and zero resets.
Training took 752 steps and testing took 2,528 steps.
The earlier provisional success assumption is no longer needed, and all Oracle domain means now use five completed seeds.
Its [journal](../../logs/agent_continual_oracle_dynamics/fan-oracle_dynamics_opus_fan_prompt_r2/seed0/run_20260922_055537/agent/sandbox/journal.md) describes averaging noisy geometry, testing a low-energy ramp-crossing attempt that safely returned the ball, and increasing the pulse before braking and steering toward the target.
These are the agent's recorded explanations, not an independently replayed validation of its causal claims.

## Numerical checks

- EMPIRIC: 25/25 whole-run successes.
- Direct agent: 16/25, with domain counts 2, 4, 4, 3, 3.
- Direct + scene: 14/25, with domain counts 3, 3, 5, 3, 0.
- Standalone sim.: 16/25, with domain counts 3, 4, 3, 3, 3.
- No harness fitting: 20/25, with domain counts 4, 4, 2, 5, 5.
- No explicit uncertainty: 21/25, with domain counts 4, 4, 3, 5, 5.
- Oracle dynamics: 25/25 whole-run successes.

Domain order above is Domino, Bridge, Balloons, Boil, Fan.
Among the 29 unsuccessful runs of the three comparison baselines, 27 finish their training tasks before failing on the test task.
The remaining two are Direct-agent Fan seeds 1 and 3, which stop during training after repeated switch-actuation failures.
Their failure explanations are agent diagnoses, not proof that the tasks were mechanically impossible.

At 500 total environment steps, Domino successes are EMPIRIC 5/5 and Direct 1/5.
At 4,600 steps, Bridge successes are EMPIRIC 5/5 and Direct 2/5.
These comparisons include failed runs in the denominator and avoid comparing only successful subsets.

EMPIRIC versus Oracle mean total steps are 345 versus 308 in Domino, 3,795.2 versus 3,204.8 in Bridge, 847 versus 726.4 in Balloons, and 1,331 versus 764.8 in Boil.
The corresponding relative differences are approximately 12%, 18%, 17%, and 74%.
All seeds in these four comparisons succeed, so the means use the same number of seeds.
In Fan, EMPIRIC averages 2,474.8 steps versus Oracle's 2,694.0, approximately 8.1% fewer.
Correct supplied dynamics therefore do not imply minimum interaction cost; reconstruction of noisy geometry, observation gathering, and action selection still affect the result.

Successful Direct-agent runs average fewer steps than EMPIRIC in Balloons (727 versus 847) and Boil (1,167.7 versus 1,331), with lower whole-run success in both domains.
The initial draft overcorrected by removing the broader sample-efficiency claim on the basis of these success-conditioned averages.
Across all five seeds, including failures, EMPIRIC uses fewer steps on average than Direct agent and Direct + scene in four domains, and fewer than Standalone sim. in all five.
Across all 25 runs, the means are 1,758.6 steps for EMPIRIC, 2,176.28 for Direct agent, 2,801.4 for Direct + scene, and 2,338.12 for Standalone sim.
Since terminal failures can stop early, these averages should be interpreted together with the success-versus-budget curves rather than as cost to successful completion.
EMPIRIC reaches 60% success sooner than each comparison baseline in every domain, or the baseline never reaches that success rate.
The appendix tables also still used the maze Fan cohort; they were regenerated from the paper figure's manifest with `scripts/plotting/export_paper_results_tables.py`, preserving all four other domains.
Among all-five-success comparisons, No harness fitting averages 987.6 steps in Boil, while No explicit uncertainty averages 1,851.4 steps in Fan; EMPIRIC averages 1,331 and 2,474.8 respectively.

## What the trajectories support

### Q1: reliability and failure mechanisms

Domino failures include a wrong prediction of push direction, overly sharp cascade turns, and evaluator rejection of cascades that relied on robot-body contact.
These are distinct from simply running out of steps.
All five Balloons ablation failures receive an explicit ceiling-burst rejection in their scorecards.
Bridge failures include assemblies dropped or swept off the table, a structure that fails the settling check, and one Direct + scene run reaching the pooled step cap.
Fan failures include unsuccessful actuation, insufficient braking before an open edge, and unsuccessful recovery after a ball leaves a platform.
Boil failures include spilling, loss of graspability, and objects knocked beyond reach.

The three baselines are competent on many seeds and sometimes formulate useful analytical models themselves.
Direct + scene succeeds on all five Balloons seeds, for example.
The evidence supports greater consistency of the complete EMPIRIC system, not a claim that the alternatives never learn dynamics or reason about uncertainty.

### Q2: useful models need not be globally accurate

[Domino EMPIRIC seed 2](../../logs/agent_continual/domino_high_friction_turn-mb_opus_gate_r1/seed2/run_20260917_082021/agent/sandbox/journal.md) documents rejecting a nominally successful two-relay layout after perturbed rehearsals and choosing a wider-margin three-relay layout.
It also checks the realized placements before the irreversible push.
This is evidence of decision-relevant robustness checks, not a calibrated posterior success probability.

[Bridge EMPIRIC seed 3](../../logs/agent_continual/bridge-mb_opus_benchmark_r2/seed3/run_20260919_124955/agent/sandbox/journal.md) reports unreliable reconstructed welded assemblies and uses a measured beam tilt to set the release height.
[Bridge seed 4](../../logs/agent_continual/bridge-mb_opus_benchmark_r2/seed4/run_20260919_124955/agent/sandbox/journal.md) likewise uses its model for glue events, placement, and collision screening while checking the actual lifted beam.
These observations support the value of feedback and selective use of predictions; they do not support claiming faithful full-trajectory simulation.

[Fan EMPIRIC seed 2](../../logs/agent_continual/fan_ramp-mb_opus_ramp_skill_repair_r1/seed2/run_20260921_090827/agent/sandbox/journal.md) chooses a pulse with a safe fallback if braking fails and checks predictions against the observed motion.
Other Fan journals record sensitivity to reconstructed ramp junctions and switch timing.
Successful task completion therefore does not independently validate state reconstruction or identify the correct physical parameters.

### Q3: where the ablations differ

[Balloons EMPIRIC seed 2](../../logs/agent_continual/balloons-mb_opus_benchmark_r2/seed2/run_20260920_060152/agent/sandbox/journal.md) compares candidate final balloon additions using ceiling clearance as well as settled height, and checks predictions across alternative calibrations.
[Balloons seed 3](../../logs/agent_continual/balloons-mb_opus_benchmark_r2/seed3/run_20260919_124956/agent/sandbox/journal.md) compares release orders, identifies a model-memory inconsistency between two rehearsal paths, and revises the model before execution.
These are examples of the combined modeling and decision process, not matched interventions on a single uncertainty component.

The no-fitting Balloons failures are seeds 0, 1, and 3; the no-uncertainty failures are seeds 1 and 4.
Their scorecards establish ceiling bursts.
Their postmortems describe incorrect extrapolation, inadequate transient margins, and release orders that remove safe continuations.
Those explanations are consistent with the importance of decision-relevant uncertainty, but do not prove which component would have prevented each failure.
Bridge and Domino ablation failures include execution and certification problems, further limiting a purely statistical explanation of the aggregate gap.

No harness fitting still allows the coding agent to calibrate models using its own calculations and recordings.
For example, [Boil no-fitting seed 3](../../logs/agent_continual_no_fitting/boil-no_fitting_opus_benchmark_r1/seed3/run_20260919_161617/agent/sandbox/journal.md) estimates flow from recorded frames, incorporates switch-off delay, and updates the fill schedule.
Removing a fitting service is not equivalent to removing identification.

[Fan no-uncertainty seed 4](../../logs/agent_continual_no_uncertainty/fan_ramp-no_uncertainty_opus_ramp_skill_repair_r1/seed4/run_20260921_121408/agent/sandbox/journal.md) reports a braking maneuver that tolerates a range of launch durations, then verifies the actual landing before continuing.
[EMPIRIC Fan seed 1](../../logs/agent_continual/fan_ramp-mb_opus_ramp_skill_repair_r1/seed1/run_20260921_090827/agent/sandbox/journal.md) and several other agents use the target pad's passive braking effect.
These behaviors can reduce the need for a precise point prediction or an explicit parameter distribution.
The draft does not claim that every safety margin in an agent journal came from the harness's uncertainty representation.

## Follow-up: mechanisms behind the comparisons

This follow-up supports the expanded causal interpretation in the Results and Discussion.
It re-examined the selected Direct-agent and two ablation cohorts, rather than substituting archived runs with more illustrative outcomes.
A programmatic scan read all 334 root-level agent-authored Python and Markdown files in those 75 sandboxes, excluding the supplied `CLAUDE.md`: 1,867,892 bytes.
Detailed inspection then followed selected model implementations, transcript tool requests and outputs, journals, and outcome records across those arms, EMPIRIC, Direct + scene, Standalone sim., and Oracle.
Sandbox programs were read, not executed, and no physical trajectory was rerun.
The examples establish observed behavior and plausible explanations of aggregate differences; they are not matched counterfactual interventions.
Agent postmortems are treated as interpretations, not independent confirmation of a physical law or an alternative plan's success.

### Direct agents construct models and perform inference themselves

In Direct-agent Fan seed 4, [`fit_model.py`](../../logs/agent_continual_model_free/fan_ramp-mf_opus_ramp_skill_repair_r1/seed4/run_20260921_121409/agent/sandbox/fit_model.py) integrates a discrete dynamics model with wind thrust, Coulomb resistance, viscous damping, and ramp acceleration.
It minimizes replay residuals with `scipy.optimize.least_squares` and compares resistance-model variants.
The [training transcript](../../logs/agent_continual_model_free/fan_ramp-mf_opus_ramp_skill_repair_r1/seed4/run_20260921_121409/agent/002_play_20260921_121816.md), lines 1225--1289, records the executed fit and a reported position residual of 0.0047 m.
[`lvl2.py`](../../logs/agent_continual_model_free/fan_ramp-mf_opus_ramp_skill_repair_r1/seed4/run_20260921_121409/agent/sandbox/lvl2.py) then searches launch and braking times, explicitly terminating predictions that cross an open edge.
The [journal](../../logs/agent_continual_model_free/fan_ramp-mf_opus_ramp_skill_repair_r1/seed4/run_20260921_121409/agent/sandbox/journal.md) compares predicted and executed outcomes, including a final landing close to the success boundary.
The run succeeds in 1,209 steps, but that success is not evidence of exact simulation accuracy.

Direct-agent Balloons seed 4 writes [`fit.py`](../../logs/agent_continual_model_free/balloons-mf_opus_benchmark_r2/seed4/run_20260919_161614/agent/sandbox/fit.py), sweeping exponential and linear height-dependent lift families consistent with measured equilibria.
The [test transcript](../../logs/agent_continual_model_free/balloons-mf_opus_benchmark_r2/seed4/run_20260919_161614/agent/003_play_20260919_162853.md), lines 229--322, contains the program invocation and its predictions.
The agent selects the next release using conclusions shared by those candidate fits, rather than claiming that the decay scale is identified.
The run succeeds in 608 steps.
Thus the contrast is absence of a supplied simulator, not absence of model-based reasoning or uncertainty reasoning.

Direct + scene Balloons seed 3 provides a complementary success case: its [journal](../../logs/agent_continual_model_free/balloons-mf_scene_package_opus_benchmark_r1/seed3/run_20260919_161613/agent/sandbox/journal.md) and [test transcript](../../logs/agent_continual_model_free/balloons-mf_scene_package_opus_benchmark_r1/seed3/run_20260919_161613/agent/003_play_20260919_163535.md) describe a fitted oscillator, sampled equilibrium predictions, and explicit transient-ceiling screening.
It selects a two-balloon plan over a three-balloon alternative despite a lower estimated probability of settling inside the band, because the pair has better ceiling clearance.
It measures the first release before committing to the second and succeeds in 663 steps.
This is evidence that a comparison baseline can independently implement much of the intended reasoning loop, not that supplied assets alone explain its success.

### Baseline failure: testing parameters while holding a wrong mechanism fixed

Direct + scene Domino seed 1 builds a PyBullet cascade model and performs extensive parameter and placement sampling.
Its [`mc_eval.py`](../../logs/agent_continual_model_free/domino_high_friction_turn-mf_scene_package_opus_benchmark_r1/seed1/run_20260919_040850/agent/sandbox/mc_eval.py) varies mass, friction, restitution, and domino poses while preserving the assumed initiating fall direction.
The [test transcript](../../logs/agent_continual_model_free/domino_high_friction_turn-mf_scene_package_opus_benchmark_r1/seed1/run_20260919_040850/agent/002_play_20260919_044030.md), lines 5670--5779, records 1.000 simulated success over 1,200 draws for friction in 0.9--1.3.
This is not success across every tested distribution: a wider 0.6--1.4 range returns 0.908.
The agent then requests a push expecting motion toward negative y, but the recorded result and its postmortem show motion toward positive y, away from the chain.
This example supports a structural explanation: sampling numerical uncertainty cannot expose an incorrect action model held fixed in every draw.
It does not support a universal claim that the baseline lacks uncertainty handling.

Direct-agent Domino seeds 2 and 4, Direct + scene seed 4, and Standalone sim. seed 0 describe a related extrapolation failure: models validated on head-on impacts did not predict the struck domino's yaw rotation at an oblique impact.
For example, see the [Direct seed 2 test transcript](../../logs/agent_continual_model_free/domino_high_friction_turn-mf_opus_r1/seed2/run_20260917_082029/agent/002_play_20260917_091223.md) and [Standalone seed 0 test transcript](../../logs/agent_continual_program_world_model/domino_high_friction_turn-standalone_opus_benchmark_r2/seed0/run_20260918_153933/agent/002_play_20260918_160830.md).
Their numerical turn-angle limits are agent estimates for particular configurations, not established universal limits.

In contrast, [EMPIRIC Domino seed 2](../../logs/agent_continual/domino_high_friction_turn-mb_opus_gate_r1/seed2/run_20260917_082021/agent/sandbox/journal.md) rehearses the actual push to identify its direction, tests turning contacts, and compares two- and three-relay arrangements.
The two-relay arrangement passes nominal and friction-sweep checks but succeeds in only 11/20 manually perturbed pose trials; the three-relay arrangement passes 80/80 of its sampled trials.
The agent chooses three relays, checks the realized placements, adjusts push parameters, and succeeds.
These are finite agent-selected simulation checks, not calibrated posterior probabilities.
Together the cases illustrate why reusing contact and controller behavior can matter beyond having a simulator or running many simulations.

### Baseline failure: predicting the ball but not the complete braking action

[Direct + scene Fan seed 2](../../logs/agent_continual_model_free/fan_ramp-mf_scene_package_opus_ramp_skill_repair_r1/seed2/run_20260921_121626/agent/004_play_20260921_125404.md) transfers a pulse calibration from a flat platform to a downhill crossing and issues a long wait while the ball accelerates with the fan off.
The run ends when the ball leaves the open platform.
[Standalone Fan seed 0](../../logs/agent_continual_program_world_model/fan_ramp-standalone_opus_ramp_skill_repair_r1/seed0/run_20260921_121408/agent/003_play_20260921_124331.md) recognizes the need for opposing-fan braking but fails to actuate that brake in the executed sequence.
[Standalone Fan seed 4](../../logs/agent_continual_program_world_model/fan_ramp-standalone_opus_ramp_skill_repair_r1/seed4/run_20260921_121408/agent/003_play_20260921_141236.md) records a successful-looking switch skill without the expected change in switch state.
These failures cannot all be attributed to an inaccurate wind-force estimate.
They involve the relationship between terrain, actuator timing, observed effects, and irreversible deadlines.
The postmortems' claims that particular switches were mechanically impossible to actuate were not independently verified.

[EMPIRIC Fan seed 2](../../logs/agent_continual/fan_ramp-mb_opus_ramp_skill_repair_r1/seed2/run_20260921_090827/agent/sandbox/journal.md) instead measures actual fan-on windows, tests switch parameters for timing stability, and compares landing predictions with and without target-surface braking.
Its final report corrects an earlier, overly optimistic fallback assumption: a ball already resting on the target plate could not necessarily be nudged onward.
That correction is important; the paper should not present every fallback the agent considered as physically validated.
The successful test trajectory supports the narrower claim that timing checks and feedback informed the executed plan.

### Ablations recreate removed services

No harness fitting Boil seed 3 performs explicit least-squares estimation on recorded water volumes.
The [training transcript](../../logs/agent_continual_no_fitting/boil-no_fitting_opus_benchmark_r1/seed3/run_20260919_161617/agent/001_play_20260919_161624.md), lines 1330--1359 and 1480--1543, contains regression code, estimates, and a subsequent timed switch-off decision.
Its [final simulator](../../logs/agent_continual_no_fitting/boil-no_fitting_opus_benchmark_r1/seed3/run_20260919_161617/agent/sandbox/simulator.py) stores the agent-calibrated filling and heating rates.
The [journal](../../logs/agent_continual_no_fitting/boil-no_fitting_opus_benchmark_r1/seed3/run_20260919_161617/agent/sandbox/journal.md) also documents averaging stationary poses, allowing for switch-off latency, and heating one jug while filling the other.
The 759-step success combines system identification and efficient task scheduling despite removal of the fitting service.

No harness fitting Fan seed 3 calibrates its force by replaying the recorded motion at several force values.
See its [training transcript](../../logs/agent_continual_no_fitting/fan_ramp-no_fitting_opus_ramp_skill_repair_r1/seed3/run_20260921_121424/agent/001_play_20260921_121432.md) and [simulator](../../logs/agent_continual_no_fitting/fan_ramp-no_fitting_opus_ramp_skill_repair_r1/seed3/run_20260921_121424/agent/sandbox/simulator.py).
The simulator adds a directional force to the base physics; it does not rebuild the whole world model from scratch.
No harness fitting Balloons seed 3 goes further: its [training transcript](../../logs/agent_continual_no_fitting/balloons-no_fitting_opus_benchmark_r1/seed3/run_20260919_161609/agent/001_play_20260919_161701.md) contains repeated `scipy.optimize.least_squares` fits over recorded episodes and comparisons of model families.
File creation or a fitting request alone is not the evidence here: the transcripts contain returned optimization results.
Removal of harness fitting therefore does not remove fitting behavior.

No explicit uncertainty Fan seed 4 repairs reconstructed support geometry in its [simulator](../../logs/agent_continual_no_uncertainty/fan_ramp-no_uncertainty_opus_ramp_skill_repair_r1/seed4/run_20260921_121408/agent/sandbox/simulator.py) and sweeps launch and braking durations in the [test transcript](../../logs/agent_continual_no_uncertainty/fan_ramp-no_uncertainty_opus_ramp_skill_repair_r1/seed4/run_20260921_121408/agent/002_play_20260921_125614.md).
The grid reports final x positions from 1.2043 to 1.2501 for launch waits from 10 to 20 and a fixed 50-step braking wait.
The agent chooses the 12-step launch, which predicts x=1.2063, and records an executed landing near x=1.2015 before completing the task.
This is an action-sensitivity study at the deployed model, not posterior sampling over dynamics parameters.
The distinction explains why removal of explicit uncertainty support does not eliminate all robust decision-making.

### Why the remaining gap concentrates in irreversible releases

No explicit uncertainty Balloons seed 4 is a particularly informative failure because it does rehearse the final release.
The [test transcript](../../logs/agent_continual_no_uncertainty/balloons-no_uncertainty_opus_benchmark_r2/seed4/run_20260919_161653/agent/003_play_20260919_183006.md), around lines 1453--1474, reports a predicted peak of 1.131 m against a 1.146 m contact height and two successful rehearsal trials, followed by an actual ceiling burst.
The [journal](../../logs/agent_continual_no_uncertainty/balloons-no_uncertainty_opus_benchmark_r2/seed4/run_20260919_161653/agent/sandbox/journal.md) compares that prediction with larger overshoot factors estimated from previous recordings.
The evidential claim is not that the agent skipped simulation, but that its accepted rehearsal underestimated the decision-critical transient with a small margin.
The trial count must not be described as two independent parameter-posterior samples.

No harness fitting Balloons seed 3 likewise [reports](../../logs/agent_continual_no_fitting/balloons-no_fitting_opus_benchmark_r1/seed3/run_20260919_161609/agent/sandbox/journal.md) closely fitted earlier trajectories and accurate intermediate equilibria before an unseen attachment configuration invalidates its lift extrapolation.
Its account of the exact missing law is a post-hoc hypothesis, not established ground truth.
[EMPIRIC Balloons seed 2](../../logs/agent_continual/balloons-mb_opus_benchmark_r2/seed2/run_20260920_060152/agent/sandbox/journal.md), lines 305--350, instead compares two final additions using previously observed signed peak-prediction errors and chooses the larger ceiling margin after re-estimating from intermediate measurements.
The paper uses these cases to explain why modeling decision-relevant error matters, without claiming that a particular posterior algorithm would necessarily have prevented the failed runs.

### Why Oracle need not minimize interactions

All 25 selected Oracle runs now succeed, so the remaining contrast is interaction cost, not solve rate.
The Fan means are 2,694.0 Oracle steps versus 2,474.8 EMPIRIC steps.
Oracle Fan seed 0's [journal](../../logs/agent_continual_oracle_dynamics/fan-oracle_dynamics_opus_fan_prompt_r2/seed0/run_20260922_055537/agent/sandbox/journal.md) records 560 test steps spent on observation batches, followed by a cautious failed crossing probe that returns safely and then a successful crossing.
The known dynamics are still evaluated from estimated geometry: noisy platform heights produce simulated lips, while braking depends on actuator timing and the chosen intermediate positions.
Oracle seed 4 takes only 968 steps, fewer than any selected EMPIRIC Fan run; its [journal](../../logs/agent_continual_oracle_dynamics/fan-oracle_dynamics_opus_fan_prompt_r2/seed4/run_20260922_055541/agent/sandbox/journal.md) describes coherent scene reconstruction and screening switch parameters for timing consistency.
This within-Oracle variation is consistent with a state-estimation and policy explanation, not with learned dynamics being intrinsically more accurate than the supplied true mechanism.
The logs do not measure the independent causal contribution of learning, reconstruction, and policy choices to the 8.1% mean difference.

## Additional questions

### Computational cost

Across the 25 selected EMPIRIC runs, the median scorecard totals are 424 recorded simulator rollouts, 162 model turns, and 155.09 minutes before correcting for account waits.
For Direct agent, the median is 115 turns and 68.09 minutes before this correction.
The figures include failed runs and therefore reflect each arm's stopping behavior.
`predicators/approaches/continual_play_mixin.py::_account_round` records the model's reported `num_turns` and the harness rollout counter.
`predicators/run/scorecard.py::total_wall_clock` sums recorded level time; queue downtime is recorded separately, but some in-process account-limit waits remain in the level clock.
The reproducible [runtime audit](../../scripts/plotting/audit_benchmark_runtime.py) dates each logged account-limit retry from its stated reset time and time remaining, then subtracts only the part of each retry sleep that overlaps recorded level time.
It excludes overlap with resume downtime and time after the scorecard's last clock flush, both of which the scorecard has already omitted.

| Arm | Raw median (min) | Adjusted median (min) | Runs with counted account waits |
| --- | ---: | ---: | ---: |
| EMPIRIC | 155.09 | 107.08 | 5/25 |
| Direct agent | 68.09 | 64.02 | 6/25 |
| Direct + scene | 54.20 | 54.20 | 5/25 |
| Standalone sim. | 62.23 | 62.23 | 6/25 |
| Oracle dynamics | 114.76 | 92.00 | 5/25 |
| No harness fitting | 62.44 | 59.80 | 13/25 |
| No explicit uncertainty | 109.01 | 87.70 | 8/25 |

The adjusted medians recompute the median after correcting each selected run, rather than subtracting a median wait from a median runtime.
The response time of a refused query is not separately recorded and remains in these adjusted figures.
The selected runs contain no logged server-side retry waits.
These counts are not token totals or independent simulation samples.
Arbitrary agent-written simulations can bypass the harness rollout counter, so zero recorded rollouts in another arm must not be interpreted as zero computation.
The available uniform scorecard fields do not decompose simulator execution, fitting, and model latency; the draft makes no controlled latency claim.

### Frozen-model prediction and calibrated uncertainty

The existing protocol allows learning during the test task.
It demonstrates successful adaptation to changed task configurations, but not frozen-model predictive performance on a preregistered set of placements, durations, or combinations.
The paper's uncertainty ablation also removes noise disclosure and state-processing support, so it does not isolate parameter-sample planning with fitting and state estimation held fixed.
These two supporting questions remain open.

Some historical selected logs also explicitly access noise-free initial-state information through the old simulator interface.
For example, the [Domino EMPIRIC seed 3 journal](../../logs/agent_continual/domino_high_friction_turn-mb_opus_benchmark_r2/seed3/run_20260919_124956/agent/sandbox/journal.md) describes `train_tasks[0].init` as noise-free.
The current [API documentation](../models/sim-api.md) describes the repaired public-observation interface.
The requested unified presentation does not establish equivalence between those interfaces or validate the streamlined estimator on these exact runs.
Consequently, the draft avoids attributing its outcomes to a particular posterior algorithm or claiming a matched latest-runtime sweep.

### Information-gain-based experiment selection

There are two explicit `sim.suggest_probes` requests in the selected EMPIRIC conversation logs.
Both omit the required sketch argument and fail before producing a ranking:

- [Balloons seed 1, first round](../../logs/agent_continual/balloons-mb_opus_compose_r2/seed1/run_20260917_082044/agent/001_play_20260917_082129.md), around line 321.
- [Balloons seed 3, first round](../../logs/agent_continual/balloons-mb_opus_benchmark_r2/seed3/run_20260919_124956/agent/001_play_20260919_125124.md), around line 354.

The audit also searched final top-level agent-authored helper programs for probe-ranking and information-gain implementations and found no additional candidates.
Tool-request counts count submitted code occurrences, not loop iterations or successful completions.
The traces do show experiments chosen by the coding agent, including release-and-measure decisions and glue tests.
They do not establish a benefit from the formal information-gain objective.
A controlled comparison of probe selection strategies remains necessary to answer that narrower question.
