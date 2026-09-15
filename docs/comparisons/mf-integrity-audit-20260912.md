# Bridge and Balloons MF integrity audit

Audited on 2026-09-12 against the selected noisy five-domain sweep runs and their frozen runtime.
Scope: all three selected MF seeds in Bridge and original non-hatch Balloons, including their experiment transcripts, journals, agent-visible recordings, current scorecards, and relevant harness code.

## Assessment

No evidence was found in the reviewed experiment logs of loading the hidden environment simulator or reading answers from other runs.
This is a log and data audit, not a proof that every possible sandbox escape is prevented.
Bridge has confirmed information leakage through shared controller diagnostics, and task-specific hints in the exposed controller source.
Balloons has an instantaneous success check that agents knowingly exploit, plus a strong baseline that writes and fits its own analytical models.
These results should not be described as a clean demonstration that a purely model-free agent discovers hidden physics from scratch.

## Verified results

| Domain | Arm | Whole-run successes | Mean steps, successful runs | Qualifying n | Mean resets, all three runs |
|---|---|---:|---:|---:|---:|
| Bridge | MB | 3/3 | 2214.0 | 3 | 0.00 |
| Bridge | MF | 3/3 | 2269.7 | 3 | 0.00 |
| Balloons | MB | 3/3 | 365.7 | 3 | 0.33 |
| Balloons | MF | 3/3 | 417.0 | 3 | 1.00 |

All these seeds completed every level.
Steps include all real interaction in each successful run, including earlier reset episodes; they are not limited to the final winning episode.
MF uses about 2.5% more steps in Bridge and 14.0% more in Balloons.
Three seeds are insufficient to establish a general equivalence claim.
Balloons MB seeds 0 and 1 are the explicitly reused subclass pilot; hatch and historical MF cohorts are excluded.
Task draws should be treated as distribution-matched, not assumed identical merely because the agent seed has the same number.

| Domain | Arm | Seed | Level wins | Steps | Resets | Scorecard |
|---|---|---:|---:|---:|---:|---|
| Bridge | MB | 0 | 2/2 | 2306 | 0 | [record](/home/ycliang/predicators/logs/agent_continual/bridge-agent_continual_noise_sweep_r1/seed0/run_20260910_052215/scorecard.json) |
| Bridge | MB | 1 | 2/2 | 1950 | 0 | [record](/home/ycliang/predicators/logs/agent_continual/bridge-agent_continual_noise_sweep_r1/seed1/run_20260910_052213/scorecard.json) |
| Bridge | MB | 2 | 2/2 | 2386 | 0 | [record](/home/ycliang/predicators/logs/agent_continual/bridge-agent_continual_noise_sweep_r1/seed2/run_20260910_052219/scorecard.json) |
| Bridge | MF | 0 | 2/2 | 2408 | 0 | [record](/home/ycliang/predicators/logs/agent_continual_model_free/bridge-agent_continual_model_free_noise_sweep_r1/seed0/run_20260910_144716/scorecard.json) |
| Bridge | MF | 1 | 2/2 | 2413 | 0 | [record](/home/ycliang/predicators/logs/agent_continual_model_free/bridge-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_144727/scorecard.json) |
| Bridge | MF | 2 | 2/2 | 1988 | 0 | [record](/home/ycliang/predicators/logs/agent_continual_model_free/bridge-agent_continual_model_free_noise_sweep_r1/seed2/run_20260910_144730/scorecard.json) |
| Balloons | MB | 0 | 3/3 | 466 | 0 | [record](/home/ycliang/predicators/logs/agent_continual/balloons-agent_continual_original_subclass_r1/seed0/run_20260909_171523/scorecard.json) |
| Balloons | MB | 1 | 3/3 | 338 | 1 | [record](/home/ycliang/predicators/logs/agent_continual/balloons-agent_continual_original_subclass_r1/seed1/run_20260909_171520/scorecard.json) |
| Balloons | MB | 2 | 3/3 | 293 | 0 | [record](/home/ycliang/predicators/logs/agent_continual/balloons-agent_continual_noise_sweep_r1/seed2/run_20260910_043043/scorecard.json) |
| Balloons | MF | 0 | 3/3 | 441 | 1 | [record](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed0/run_20260910_201514/scorecard.json) |
| Balloons | MF | 1 | 3/3 | 476 | 1 | [record](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_201533/scorecard.json) |
| Balloons | MF | 2 | 3/3 | 334 | 1 | [record](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed2/run_20260910_201534/scorecard.json) |

Bridge MF seed 1 retains the first completed execution, run_20260910_144727.
The snapshot documents a later duplicate execution after a scheduler restart during video generation; it is excluded as a duplicate, not treated as a fourth seed.

## Bridge: a hidden attachment is exposed through an error

In [seed 1, play 003](/home/ycliang/predicators/logs/agent_continual_model_free/bridge-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_144727/agent/003_play_20260910_151152.md:1006), a failed Place returns `welded span1` and an exact table penetration of 0.0061 m.
The agent immediately writes: `The failure message is gold: "welded span1" - the bond did form on the previous place`.
It then uses the reported geometry to adjust its release height.
The [shared planner diagnostic](/home/ycliang/predicators-noisy-sweep-20260910/predicators/ground_truth_models/skill_factories/base.py:2134) constructs these labels from held attachments.
This is observed use of information about the hidden attachment, not a speculative exploit.
Internal attachment-aware collision planning may be necessary for the controller, but exposing that attachment identity and precise internal geometry is a separate observation-policy decision.

Before interaction, the agent also [reads the supplied controller references](/home/ycliang/predicators/logs/agent_continual_model_free/bridge-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_144727/agent/001_play_20260910_144733.md:1687) and extracts placement heights, weld behavior, and the recipe of lifting a welded three-span row by its middle block.
The [reference exporter](/home/ycliang/predicators-noisy-sweep-20260910/predicators/approaches/agent_model_free_approach.py:345) supplies controller implementation files, including comments.
The public task already asks for a rigid three-block span, so the references do not reveal the entire goal for the first time; they provide additional implementation-specific solution guidance.
These references and controllers are shared with MB, so this is a shared benchmark issue rather than evidence of preferential MF access.

## Balloons: legitimate self-modeling and an instantaneous win

The [seed 1 journal](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_201533/agent/sandbox/journal.md:230) compares competing dynamics hypotheses.
Its [uncertainty calculations](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed1/run_20260910_201533/agent/sandbox/journal.md:273) use damped-sinusoid fits and residual bootstrap estimates.
Other seeds similarly fit oscillation or lift models, enumerate balloon subsets, and use real probes before committing to releases.
These calculations operate on the agent’s own recorded observations and are allowed by the baseline prompt.
The absence of supplied simulator calls does not mean an absence of learned dynamics or uncertainty reasoning.
A more accurate baseline name is direct coding agent without supplied simulator or model-learning tools.

The [InBand classifier](/home/ycliang/predicators-noisy-sweep-20260910/predicators/envs/pybullet_balloons.py:283) requires the box centre to be in the band and its speed to be below 0.01 m/s in the current state.
It imposes no sustained dwell condition.
The [seed 2 journal](/home/ycliang/predicators/logs/agent_continual_model_free/balloons-agent_continual_model_free_noise_sweep_r1/seed2/run_20260910_201534/agent/sandbox/journal.md:494) explicitly recognizes that an oscillation turning point can certify success even when the equilibrium height is outside the band.
All nine recorded MF level wins have a final speed below 0.01 m/s and a preceding frame above that threshold.
For example, seed 0 level 1 wins at z=0.78166 m in the [0.75952, 0.80952] m band, with speed falling from 0.11596 to 0.00991 m/s.
Those are valid wins under the implemented predicate, but do not demonstrate sustained hovering.
The recordings terminate at success, so this audit does not establish which runs would eventually settle successfully or fail a longer dwell test.
The predicate also applies to MB; this audit does not establish that the issue favors MF more than MB.

## Observation and tool checks

The [recording audit](/home/ycliang/predicators/logs/mf_integrity_audit_20260912/data-audit.json) examined 8,075 MF frames across the six runs.
No frame carried a populated privileged or latent state channel, and no simulator-state dictionary keys were exposed.
However, serialized Object instances retain sim_data metadata.
Bridge objects expose cure and attachment field names, although their values remain the default zero or -1 throughout these recordings; Balloons metadata includes body and clip joint IDs.
This is unnecessary implementation metadata and should be removed from the public serialization contract, but it is not evidence that these agents read live hidden cure/attachment values.
The [extracted tool calls](/home/ycliang/predicators/logs/mf_integrity_audit_20260912/tool-calls.json) show analysis of recorded trajectories and supplied reference files, with no use of registered simulator/model tools.
Some attempts to repair sandbox working-directory errors involved broader tools or path changes; these do not justify claiming that the sandbox is adversarially secure.

## Recommended next steps

1. Keep and report the existing results as the current protocol cohort, with the above limitations.
2. Replace controller implementation references with a public skill API specification, removing task-specific recipes and hidden-mechanism comments for both arms.
3. Sanitize controller diagnostics so they do not name hidden attachments or expose privileged geometry, while preserving useful feedback based on public observations.
4. Remove simulator metadata from agent-visible serialized objects and validate the complete observation/tool boundary.
5. If the intended Balloons task is sustained hovering, specify and mechanically validate a dwell criterion, then rerun both arms under the changed task as a separate cohort.
6. Preserve ordinary Python analysis and learned journals in the direct coding baseline; banning its successful reasoning would weaken the comparison artificially.

No experiments, controllers, or task definitions were changed by this audit.
Cleaning these issues may change either arm’s performance; it does not guarantee a larger MB advantage.
