# Oracle Fan prompt-alignment rerun

Launched September 22, 2026 as Slurm array `23469223`, seeds 0 through 4, on `mit_preemptable` compute nodes.
Seeds 0 and 3 exited before doing work because account a's organization had disabled Claude subscription access.
They were relaunched on the verified dat account as jobs `23469550` and `23469551`; seeds 1, 2, and 4 continued unchanged.

The cohort identifier is `fan-oracle_dynamics_opus_fan_prompt_r2`.
Run records are under `logs/agent_continual_oracle_dynamics/fan-oracle_dynamics_opus_fan_prompt_r2/`.
The launch configuration is `scripts/configs/predicatorv3/continual_fan_oracle_prompt_r2.yaml`.
The isolated runtime is `logs/fan-oracle-prompt-r2-runtime-20260922`, based on commit `08db6c00a` with the Oracle prompt clarification and its regression assertions applied.

The agent shares EMPIRIC's state-reconstruction, timing-check, and execution-discrepancy guidance, while supplied dynamics and physical parameters remain fixed.
Oracle-specific API guidance explicitly excludes parameter sweeps and retains state-belief draws and controller trials.
Automatic skill preflight and the validation audit remain disabled.
The environment is the reviewed Fan ramp with 0.003 m rise and 0.10 m landing extension, using the repaired shared skills.
These results must remain separate from the prior Oracle cohort when assessing the prompt change.

The 33 prompt regression tests passed on a compute node before submission.
The account usage endpoint returned HTTP 403, so remaining quota was unavailable at launch.
Future launches must verify an actual lightweight Claude response when this lookup fails rather than treating the round-robin fallback as proof of availability.
