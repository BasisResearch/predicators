# Articulated state in offline replay

September 12, 2026.
This fixes a missing state component in the [simplification replay contract](simplification-proposal.md#1-establish-the-state-and-replay-contract).
The production fitter and acting agent are unchanged.

## Reproduction and correction

The original `ReplayState` preserved every robot joint but omitted nonrobot joint states.
Replaying a candidate could therefore replace an uncertain slider position with its controller endpoint and reset a rotor's position and velocity to fresh-world defaults.
Repeated runs could agree perfectly while both discarded the same candidate uncertainty.

The [pre-fix native reproduction](../../logs/uncertainty_articulated_replay_repro_20260912/pilot-22639384_1.json) exercises `capture_replay_state` followed by `replay_candidate` with the frozen learned Fan simulator and reconstructed public initial frame.
It supplies nonendpoint positions and nonzero velocities to the four sliders and twenty rotors.
All 24 differ after restoration, consistently across two fresh worlds.
For example, one slider changes from position 7.4 mm and velocity 20 mm/s to zero position and velocity.
This is a diagnostic candidate, not a complete physical initial-state prior or an agent result.

[ReplayState](../../predicators/code_sim_learning/inference_replay.py) now carries an `ArticulatedBody` record for every nonrobot native body with joints, including bodies omitted from public object keys.
Each record includes native body ID, body names, joint names/types/link names, and every joint's position and velocity in native order.
Restoration verifies the complete record sequence against the fresh world's layout after domain initialization, then restores the supplied joint states after the controller reconciliation performed by `_set_state`.
Missing or duplicate bodies, differing layouts, incomplete joint arrays and nonfinite joint values raise an explicit error.
These values come from the simulated candidate or an evaluator-only mechanical audit, never from an added agent observation channel.

The native IDs require the same body-allocation protocol as the source candidate.
Names and joint topology detect mismatches; they are not a semantic remapping algorithm or a proof that interchangeable identical assets were allocated in the same order.
The existing runtime-identity and same-layout requirements therefore remain necessary.
This change does not capture solver warm starts, arbitrary motor-controller changes or every other form of engine history.
Use full-prefix reconstruction for continued contact histories and continue to distinguish repeatable candidate replay from agreement with evaluator physics.

## Validation

The [fixed native audit](../../logs/uncertainty_articulated_replay_fixed_20260912/pilot-22639468_1.json) restores all 24 supplied joint states exactly.
It also repeats the first 64 recorded actions from that candidate in two fresh worlds.
All 65 boundaries agree exactly in robot joint states, nonrobot joint states and projected public observations.
This validates the tested candidate and runtime, not an arbitrary portable mid-contact checkpoint or a calibrated Fan posterior.

Compute job `22639455` passes 22 functional replay tests plus focused mypy, pylint and pinned formatting checks.
The regression includes a supplementary physical fan without a public object, a moving clip slider, repeated continuations and five malformed-state controls.
Existing tests continue to cover robot motion, model memory, original weld frames, queued commands and full-prefix replay.

The remaining Fan work is to compose supported scene geometry and articulated priors with the complete recording likelihood, then establish numerical and predictive adequacy against the incumbent fitter.

The [whole-scene follow-up](fan-initial-scene.md) has since supplied a declared geometric root law and exposed later exact-event failures over the full recording.
Runtime addenda for [the reproduction](../../logs/uncertainty_articulated_replay_repro_20260912/runtime-audit.json) and [the fixed audit](../../logs/uncertainty_articulated_replay_fixed_20260912/runtime-audit.json) correct inherited node1412 labels: those jobs actually ran on node1408 and node1926 respectively.
The measured within-job repeatability claims remain unchanged; the raw artifacts are preserved.
