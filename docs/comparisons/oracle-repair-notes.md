# Oracle validation repairs, September 19

The follow-up pilot runs Oracle dynamics on Domino and Bridge, seeds 0-2, with automatic skill preflight disabled.
It uses new `oracle_dynamics_opus_benchmark_r2` run keys so the original recordings are retained and cannot be resumed under different code.
The other domains are not part of this launch.
The parsed launch flags match the recorded original Oracle commands except for the new round key and explicitly specifying the already-disabled preflight flag.

## Launch receipt

Frozen runtime: `d7e8ce408`, at `/home/ycliang/predicators-oracle-validation-frozen-20260919`.
Bridge array: `23104058`, seeds 0-2.
Domino array: `23104059`, seeds 0-2.
Account `c` hit its weekly limit before Bridge seed 2 took any actions; that seed resumed as `23104110_2` on configured account `d`.
Domino seed 1 had also selected `c`; its task-generation job was stopped before agent play and resubmitted as `23104122_1` on configured account `a`.
The remaining original array tasks are unchanged, and both replacements use the same frozen runtime and round keys.
Both arrays run on `mit_preemptable`, with automatic checkpoint resume and requeue.
The figure/report monitor is compute-node job `23104071`, checking every 60 seconds with a 48-hour allocation.
It does not launch further experiments, cancel jobs, or automatically commit generated updates.
Its status file is `logs/benchmark_monitor/status.json`; its output is `logs/benchmark_monitor/slurm-23104071.out`.
The final targeted static, lint, and formatter checks completed successfully in compute-node job `23103985`.

## Changes

- Domino certification now resolves deployment skills for supplied and learned subclasses and can run against scene-built candidates.
  Candidate errors and missing per-step trajectories are reported as unavailable/error, never accepted.
- Candidate trials and explicitly fresh rehearsals use isolated physics instances with the deployed model and parameter values.
  This changes isolation, not the requested number of rollouts or the real-environment step budget.
- Bridge Oracle restoration no longer applies new-joint pose snapping to an already assembled, tilted beam.
  Prediction snapshots preserve model-owned weld frames; observation-only starts reconstruct relative frames from observed poses.
- Bridge Oracle observation memory uses visible consumption of fully wet glue to recover a latch when the wet interval identified exactly one contact partner.
  Noisy contact gaps no longer necessarily erase that evidence.
  Ambiguous partner histories are not resolved by guessing the nearest block.

No automatic per-action gate, mandatory uncertainty sweep, or shared real-controller change was introduced.
The validation API changes apply to engine-backed learned candidates as well as Oracle.
The Bridge observation-memory repair changes the supplied Oracle model, not EMPIRIC's agent-written mechanism code.

## Direct evidence

The recorded Oracle Domino seed-1 training cascade is accepted and its failed test cascade is rejected by the corrected supplied-model certificate.
Both recorded trajectories satisfy the terminal goal predicate.
The test rejection explicitly comes from the fingertips-only counterfactual push, not an exception.
See [certificate replay output](../../logs/oracle-historical-cert-23103408.out) and [replay script](../../scripts/replay_oracle_certificate.py).
This re-evaluates recorded trajectories and their counterfactual pushes; it does not claim a full new agent plan replay.

A public `sim.run(..., trials=2)` regression reproduced shared-world candidate trials before the isolation repair.
A tilted four-block reset regression reproduced pose changes during weld reconstruction.
A noisy observation-history regression reproduced missing attachment memory despite an actual cured joint.
The repaired tests cover held/unheld assemblies, fresh-world frame restoration, reset isolation, and observed-only memory.
The expanded compute-node regression batch passed 196 tests (`23103772`), and the candidate-parameter and launch-scope follow-up passed four tests (`23103843`).
These are targeted checks, not a claim that the full repository CI suite was run.

Reconstructing Oracle Bridge seed 2 at test step 1267 initially produced zero beam joints.
The repaired observer reconstructs all three, while retaining the observed grasp of `span3`.
See [saved-decision replay output](../../logs/bridge-memory-check-23103561.out) and [replay script](../../scripts/replay_bridge_rehearsal.py).

## Remaining limits

At that saved Bridge decision, three fresh rehearsals predict the requested lift can complete, although the recorded real controller refused it.
These are controller diagnostics, not task-success certificates.
The reconstructed noisy state is not the true execution state, and fresh physics does not repair observation error or recover unobserved contact-solver history.
Standing-block Euler coordinates also undergo canonicalization during reconstruction.
The pilot therefore tests whether the repairs improve end-to-end outcomes; it does not assume that Oracle is now a perfect execution predictor.

Historical EMPIRIC runs mixed preflight settings.
Any final claim about relative performance needs matched runtime and preflight settings, not a comparison that silently replaces only Oracle's old failures.
The [pilot report](oracle-dynamics-validation-r2.md) keeps new results separate from the main benchmark figure.
