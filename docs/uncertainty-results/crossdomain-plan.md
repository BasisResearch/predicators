# Bridge, fan and balloons noise validation

The next comparison uses seeds 0 and 1 in bridge, fan and balloons.
All arms use the same frozen code, task generation, skill controllers and observation channel.
The experiment checkout is `/home/ycliang/predicators-noise-crossdomain-r1`, based on `92d37ec97` with the bridge manipulation fixes and fan stall fixes applied before validation.
Its uncommitted patch and file hashes accompany the run manifest.
The validated runtime changes are also committed on `bridge-learning`: `c57b9972b` (manipulation), `f9b85d79c` (bridge goal), `c13a7d524` (rehearsal prompt) and `ab23a35ef` (fan Wait).
The experiment checkout remains frozen independently of later main-checkout edits.

| Domain | Position sigma | Orientation sigma | Train + test levels per seed |
|---|---:|---:|---:|
| Bridge | 0.005 m | 0.02 rad | 1 + 1 |
| Fan | 0.005 m | 0.02 rad | 1 + 1 |
| Balloons | 0.010 m | 0.02 rad | 2 + 1 |

Scalar-reading noise is zero in this first comparison.
The noise parameters are declared to both arms.
Robot proprioception, skill control and evaluator decisions retain their existing true-state semantics.
Balloons uses the current contact-task generator, with `balloons_require_jam_decoy=True`.

## Comparison

Each noisy setting has three arms: plain MF, MB without the six added uncertainty flags, and MB with all six flags enabled.
The two MB arms share the existing adaptive information-seeking and margin settings; their only flag differences are the six uncertainty switches.
MF has no harness belief frame or simulator.
The noisy matrix contains 18 runs.

The existing noiseless results are the references; no new noiseless runs are scheduled.
The 12 newly queued exact controls were cancelled at the user's request on 2026-09-08 before any started.
Report the source code versions alongside comparisons that reuse those earlier results.

| Added uncertainty flag | MB off | MB on | MF |
|---|---|---|---|
| `code_sim_learning_interval_belief` | False | True | False |
| `agent_explorer_info_seeking_noise_aware` | False | True | False |
| `code_sim_learning_rollout_noise_filter` | False | True | False |
| `code_sim_learning_carry_posterior` | False | True | False |
| `code_sim_learning_fit_evidence` | False | True | False |
| `continual_belief_frame` | False | True | False |

## Metrics

Solve rate is the fraction of train and test levels won, averaged over all seeds.
Resets are the whole-run reset count, averaged over all seeds.
Steps are whole-run steps averaged only over completed seeds that won every level, including their earlier attempts and resets.
Report the number of qualifying seeds beside each step average; report N/A when there are none.
These two-seed comparisons are exploratory evidence, not a precise estimate of reliability.

## Launch status

The 18 noisy agent runs are submitted as nine arrays on `mit_preemptable`, with automatic requeue and continual-protocol resume.
Every array requires preflight gate `22320276` to pass.
Each array after the first also requires the preceding array to finish, limiting this sweep to two concurrent agent processes.
The gate checks regression job `22318792`, fan replay job `22318760`, all six oracle tasks in array `22320170`, and the frozen source hashes.
The regression suite passed 160 tests plus four fan integration tests.
The recorded fan replays no longer hit the 1000-step cap: stationary Wait terminates in 16 steps, and the two infeasible switch commands return bounded motion failures.
Short validation jobs may use `mit_quicktest` or `mit_normal`; all agent arrays use `mit_preemptable`.

The first oracle attempt selected `oracle` instead of `oracle_process_planning`, so all six tasks failed before executing a skill.
That configuration failure caused the original gate `22319420` to fail and first agent array `22319426` to be cancelled; no agent task started.
The corrected oracle uses the canonical process-planning flags and fresh experiment IDs ending in `crossdomain_oracle_r2`.
The dependency chain was also corrected so every array requires successful validation even if an earlier array is cancelled.
At this status update, both bridge seeds, both fan seeds and balloons seed 0 passed; balloons seed 1 is waiting to start, and all agent arrays remain pending.
Account selection uses the existing limit-aware launcher with accounts `a,c`.

| Domain | Noisy MB on | Noisy MB off | Noisy MF |
|---|---|---|---|
| Bridge | 22320299 | 22319429 | 22319432 |
| Fan | 22319427 | 22319430 | 22319433 |
| Balloons | 22319428 | 22319431 | 22319435 |

Each array contains seed tasks 0 and 1.
The MB-on arrays for bridge, fan and balloons are first in the queue, followed by the noisy comparators.
The additional boil jobs remain held and are outside this launch.

Config: [noisy comparison](../../scripts/configs/predicatorv3/protocol_continual_uncertainty_crossdomain.yaml).
Validation logs and the launch manifest live under `logs/uncertainty_crossdomain_20260908/` in the shared checkout.
