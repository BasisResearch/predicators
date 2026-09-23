# Fan inertial-transfer candidate

This candidate isolates dynamical complexity from path complexity.
It keeps the protected calibration tray and exposed L-shaped test layout from [the original transfer pilot](fan-exposed-transfer.md).
It does not replace that pilot, the Fan maze cohort, or any paper result.

## Physical change

With `fan_exposed_transfer=true` and `fan_inertial_transfer=true`, ball linear damping is 0.08, angular damping is 0.008, and platform rolling friction is 0.0001.
Wind force is 0.0004 N per active fan, reduced from 0.002 N to leave time for robot-operated switching.
Lateral contact friction, geometry, observation noise, goal tolerance, failure conditions, and control access remain unchanged.
Training and test use identical physical parameters.
The learned simulator base reconstructs the same contact physics; the environment's wind mechanism remains something the agent must model.

Momentum now persists across switch operations, so turning a fan off does not promptly stop motion along that axis.
An opposing fan can actively brake the ball, and turning onto the next segment while still moving can cause cross-axis drift.
Stopping before the turn remains legal: this is not a restriction on baseline strategies.
The candidate does not guarantee that a direct agent's fitted pulse model will fail.

## Validation

Compute job `23239571` passed 27 checks covering both old and inertial physics, both launch configurations, support reconstruction, failure and settling evaluation, real/model trajectory agreement, and moving-state restoration.
After a 35-step wind pulse and 40 unpowered steps, measured travel after switch-off was 0.1136 m for the candidate versus 0.0195 m for the original transfer pilot.
The coasting measurement stays clear of the protective wall.
After 12 braking steps, the candidate's horizontal velocity was 0.0137 m/s, compared with 0.0413 m/s without braking from the same saved state.
The original reference controller, which switches off only 10 cm before the turn, fell off in the new setting.
These are physical diagnostics, not agent results.

Actual-controller feasibility searches for seeds 0 and 1 are recorded in compute jobs `23239667` and `23239678`.
They search pulse timings using privileged simulation, then replay the selected sequence with the real switch controllers.
Their success must not be reported as EMPIRIC success or as evidence of an agent performance gap.
Both replays succeeded: seed 0 in 435 steps and seed 1 in 425 steps, including conservative settling waits but excluding diagnostic search cost.
The selected plans used timed forward pulses and passive coasting, not opposing-fan braking.
Thus active braking is available and useful, but is not required to solve these layouts.
The remaining experimental question is whether an agent can learn sufficiently accurate timing without the diagnostic's privileged simulation access.

## Reproduction

Candidate agent configuration: `scripts/configs/predicatorv3/continual_fan_inertial_pilot_r1.yaml`.
This uses separate `fan_inertial` run keys for EMPIRIC and the direct agent, with two matched seeds each.
EMPIRIC array `23239940` and direct-agent array `23239941` were submitted on 2026-09-20, with seeds 0 and 1 on `mit_preemptable`.
They use frozen runtime `logs/fan-inertial-runtime-20260920` at commit `8d12ae07d89a994889b03f5cfe2488dacbdf8140`.
Accounts a through d are selectable with automatic limit handling; dat remains backup.
Direct seed 0 completed both levels in 1,577 real steps (924 train, 653 test), with zero resets and `all_levels_won` certification.
EMPIRIC seed 0 completed both levels in 1,405 real steps (406 train, 999 test), also with zero resets and `all_levels_won` certification.
Direct seed 1 ended after an explicit give-up with training solved and test unsolved, in 3,367 total steps (1,453 train, 1,914 test), zero resets.
This is a forfeit after an overshoot, not an evaluator-triggered terminal fall.
The postmortem's claimed hidden switch on/off access was checked against the frozen runtime: the same flag is already observable as `fan.is_on`, as documented in the development record.
EMPIRIC seed 1 completed both levels in 2,272 steps (1,003 train, 1,269 test), zero resets, with accepted evaluator wins and `all_levels_won`.
The pilot is therefore EMPIRIC 2/2 versus direct control 1/2, sufficient for screening but not yet a confirmed robustness difference.
Fresh confirmation arrays `23252966` (EMPIRIC) and `23252967` (direct) were launched from the same frozen runtime with identical resolved environment and agent settings and separate confirmation run keys.
The user reduced the batch to three additional seeds per agent: seeds 2 through 4 continue, while seeds 5 and 6 were cancelled with partial logs preserved and must not be resumed or counted as task failures.
On seed 0 both agents solved, and EMPIRIC's lower total cost came from training rather than test efficiency.
See the [development record](fan-development.md) for interpretation and subsequent updates.
Run physical tests with `scripts/validate_fan_transfer.sh` on a compute node.
Run the feasibility diagnostic with `scripts/probe_fan_transfer.py --inertial --skills --search-bursts --seed 0` or seed 1, also on a compute node.
